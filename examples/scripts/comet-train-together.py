# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os, sys
import random
import datetime

import numpy as np
import torch
sys.path.insert(0, "..")
sys.path.insert(1, ".")
from pytorch_transformers import (CONFIG_NAME, WEIGHTS_NAME, AdamW,
                                  GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
                                  XLNetLMHeadModel, XLNetTokenizer, XLNetConfig, 
                                  OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig,
                                  WarmupLinearSchedule, cached_path)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
import torch.nn.functional as F
from scripts.utils import (save_model, tokenize_and_encode, split_into_words, set_seed, PADDING_TEXT)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def pre_process_datasets(encoded_datasets, input_len, max_e1, max_r, max_e2, paddings=[], predict_rel=False):
    tensor_datasets = []
    padding_length = len(paddings)
    input_len += padding_length
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.full((n_batch, input_len), fill_value=0, dtype=np.int64)

        for i, (e1, r, e2, label), in enumerate(dataset):
            # truncate if input is too long
            if len(e1) > max_e1:
                e1 = e1[:max_e1]
            if len(r) > max_r:
                r = r[:max_r]
            if len(e2) > max_e2:
                e2 = e2[:max_e2]

            input_ids[i, :padding_length] = paddings
            input_ids[i, padding_length:padding_length+len(e1)] = e1
            start_r = max_e1 + padding_length
            end_r = max_e1 + len(r) + padding_length
            input_ids[i, start_r:end_r] = r
            start_e2 = max_e1 + max_r + padding_length
            end_e2 = max_e1 + max_r + len(e2) + padding_length
            input_ids[i, start_e2:end_e2] = e2
            if i == 0:
                print("one encoded sample: e1", e1, "r", r, "e2", e2, "input_ids:", input_ids[i])


        lm_labels = np.copy(input_ids)
        # do not calculate loss on paddings
        lm_labels[lm_labels == 0] = -1  
        if not predict_rel:
            # do not calculate loss on e1/r
            lm_labels[:, :max_e1 + max_r + padding_length] = -1    
        else:
            lm_labels[:, :padding_length + max_e1] = -1
            lm_labels[:, max_e1 + max_r + padding_length:] = -1
        input_mask = (input_ids == 0)   # attention mask
        all_inputs = (input_ids, lm_labels, input_mask)
        tensor_datasets.append((torch.tensor(input_ids), torch.tensor(lm_labels), 
                                torch.tensor(input_mask).to(torch.float32)))
    return tensor_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, eval_dataloader, tokenizer, max_e1, max_r, max_e2, args, encoded_padding=[], predict_rel=False):
    model.eval()
    eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    num_displays = 5
    eos_token = tokenizer.encode(tokenizer.eos_token)[0]
    print("\n\nsome examples\n")
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        batch_size = len(batch)
        input_ids, lm_labels, input_mask = batch
        with torch.no_grad():
            if args.model_type == "gpt2":
                results = model(input_ids, labels=lm_labels, input_mask=input_mask)
                loss, logits, past = results
            elif args.model_type == "openai-gpt":
                results = model(input_ids, labels=lm_labels, input_mask=input_mask)
                loss, logits = results
            elif args.model_type == "xlnet" and not predict_rel:
                padding_length = len(encoded_padding)
                seq_length = input_ids.size(1)
                batch_size = input_ids.size(0)
                lm_labels = lm_labels[:, -max_e2:]
                start_pos = padding_length + max_e1 + max_r
                perm_mask = torch.zeros((batch_size, seq_length, seq_length), dtype=torch.float, device=device)
                perm_mask[:, :, start_pos:] = 1.0
                perm_mask = torch.triu(perm_mask)
                target_mapping = torch.zeros((batch_size, max_e2, seq_length), dtype=torch.float, device=device)
                for i in range(max_e2):
                    target_mapping[:, i, start_pos + i] = 1.0
                inputs = {'input_ids': input_ids, 'input_mask': input_mask, 'perm_mask': perm_mask, 'target_mapping': target_mapping}
                outputs = model(**inputs)
                logits = outputs[0]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lm_labels.contiguous().view(-1), ignore_index=-1)
            elif args.model_type == "xlnet" and predict_rel:
                padding_length = len(encoded_padding)
                seq_length = input_ids.size(1)
                batch_size = input_ids.size(0)
                lm_labels = lm_labels[:, max_e1 + padding_length : max_e1 + padding_length + max_r]
                perm_mask = torch.zeros((batch_size, seq_length, seq_length), dtype=torch.float, device=device)
                perm_mask[:, :, max_e1 + padding_length : max_e1 + padding_length + max_r] = 1.0
                perm_mask = torch.triu(perm_mask)
                perm_mask[:, max_e1 + max_r + padding_length:, max_e1 + padding_length : max_e1 + padding_length + max_r] = 1.0
                target_mapping = torch.zeros((batch_size, max_r, seq_length), dtype=torch.float, device=device)
                for i in range(max_r):
                    target_mapping[:, i, max_e1 + padding_length + i] = 1.0
                inputs = {'input_ids': input_ids, 'input_mask': input_mask, 'perm_mask': perm_mask, 'target_mapping': target_mapping}
                outputs = model(**inputs)
                logits = outputs[0]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lm_labels.contiguous().view(-1), ignore_index=-1)
            eval_loss += loss * batch_size
            nb_eval_steps += batch_size
            # display some examples
            if num_displays:
                num_displays -= 1
                value, indices = logits.max(dim=-1)
                sample_index = random.randint(0, batch_size - 1)
                print("input:", tokenizer.decode(input_ids[sample_index].tolist()))
                if not args.model_type == "xlnet":
                    output = indices[sample_index].tolist()[max_e1 + max_r - 1:]
                else:
                    output = indices[sample_index].tolist()
                    
                try:
                    eos_pos = output.index(eos_token)
                    output = tokenizer.decode(output[:eos_pos])
                except:
                    output = tokenizer.decode(output)
                print("output:", output)

    eval_loss = eval_loss / nb_eval_steps
    return eval_loss

def load_comet_datasets(dataset_paths, end_token, rel_lang, toy=False):
    if not end_token:
        end_token = ""
    output = []
    for dataset_path, dataset_label in dataset_paths.items():
        dataset_is_rel_lang = rel_lang[dataset_label]
        with open(dataset_path, encoding='utf_8') as f:
            f = f.read().splitlines()
            if toy:
                f = f[:1000]
            for line in tqdm(f):
                rel, e1, e2, label = line.split("\t")
                if label == "0":
                    continue
                e1 = dataset_label + " " + e1
                e2 += (" " + end_token)
                if dataset_is_rel_lang:
                    rel = split_into_words[rel]
                    if not rel:
                        continue
                    output.append((e1, rel, e2, label))
                else:
                    output.append((e1, rel.lower(), e2, label))
            # print some samples for debugging
            print(output[-3:])
    return output

# Save a trained model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='openai-gpt',
                        help="model type: openai-gpt/gpt2/xlnet/...")
    parser.add_argument('--model_name_or_path', type=str, default='openai-gpt', help="pretrained model path")              
    parser.add_argument("--do_train", action='store_true', help="do training")
    parser.add_argument("--do_eval", action='store_true', help="do evaluation")
    parser.add_argument("--predict_rel", action='store_true')
    parser.add_argument("--toy", action='store_true', help="test code")
    parser.add_argument("--padding_text", action='store_true', help="xlnet needs a padding text")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument("--no_pretrain", action='store_true', help="w/o pretrained parameters initialized")
    parser.add_argument('--eval_per_steps', type=int, default=500)
    parser.add_argument('--num_train_epochs', type=int, default=64)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # select model type
    if args.model_type == "openai-gpt":
        Tokenizer = OpenAIGPTTokenizer
        Model = OpenAIGPTLMHeadModel
        Config = OpenAIGPTConfig
    elif args.model_type == "gpt2":
        Tokenizer = GPT2Tokenizer
        Model = GPT2LMHeadModel
        Config = GPT2Config
    elif args.model_type == "xlnet":
        Tokenizer = XLNetTokenizer
        Model = XLNetLMHeadModel
        Config = XLNetConfig
    else:
        exit()

    # load pretrained model
    tokenizer = Tokenizer.from_pretrained(args.model_name_or_path)
    if args.no_pretrain:
        # from scratch
        config = Config.from_pretrained(args.model_type)
        model = Model(config)
    else:
        model = Model.from_pretrained(args.model_name_or_path)

    # if not reloading from existing checkpoints, add special tokens
    if args.model_name_or_path == "openai-gpt" or args.model_name_or_path == "gpt2":
        tokenizer.add_special_tokens({"bos_token": "<bos>", 
                                    "eos_token": "<eos>",
                                    "unk_token": "<unk>"})
    tokenizer.add_tokens(["<from_CN>", "<from_VG>", "<from_FB>"])
    print("vocab size:", len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    end_token = tokenizer.eos_token
    print("\nspecial tokens:", tokenizer.special_tokens_map)

    # Load and encode the datasets
    logger.info("Encoding dataset...")
    train_datasets = {"data/conceptnet/train100k_CN.txt": "<from_CN>",
                      "data/vg_train.txt": "<from_VG>",
                      "data/FB15K237_train.txt": "<from_FB>"}
    eval_datasets = {"data/conceptnet/dev1_CN.txt": "<from_CN>",
                      "data/conceptnet/dev2_CN.txt": "<from_CN>",
                      "data/vg_dev.txt": "<from_VG>",
                      "data/FB15K237_valid.txt": "<from_FB>"}
    test_datasets = {"data/conceptnet/test_CN.txt": "<from_CN>",
                      "data/vg_test.txt": "<from_VG>",
                      "data/FB15K237_test.txt": "<from_FB>"} 
    rel_lang = {"<from_CN>": True, "<from_VG>": False, "<from_FB>": True}           
    train_dataset = load_comet_datasets(train_datasets, end_token, rel_lang=rel_lang, toy=args.toy)
    eval_dataset = load_comet_datasets(eval_datasets, end_token, rel_lang=rel_lang, toy=args.toy)
    test_dataset = load_comet_datasets(test_datasets, end_token, rel_lang=rel_lang, toy=args.toy)
    datasets = (train_dataset, eval_dataset, test_dataset)
    encoded_datasets = tokenize_and_encode(datasets, tokenizer)
    max_e1 = 15
    max_r = 10
    max_e2 = 15 + 1
    input_length = max_e1 + max_r + max_e2
    best_loss = 1e10
   
    encoded_padding = tokenize_and_encode(PADDING_TEXT, tokenizer) if args.padding_text else []
    # Prepare inputs tensors and dataloaders
    tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_e1, max_r, max_e2, paddings=encoded_padding)
    train_tensor_dataset, eval_tensor_dataset, test_tensor_dataset = tensor_datasets[0], tensor_datasets[1], tensor_datasets[2]

    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    test_data = TensorDataset(*test_tensor_dataset)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs
        print("total steps:", num_train_optimization_steps)
        num_warmup_steps = args.warmup_proportion * num_train_optimization_steps
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_optimization_steps)

    if args.do_train:
        nb_tr_steps = 0
        model.train()
        for cur_epoch_num in range(int(args.num_train_epochs)):
            print("Epoch:", cur_epoch_num)
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, lm_labels, input_mask = batch
                if args.model_type == "xlnet" and not args.predict_rel:
                    padding_length = len(encoded_padding)
                    batch_size = input_ids.size(0)
                    seq_length = input_ids.size(1)
                    lm_labels = lm_labels[:, -max_e2:]
                    start_pos = padding_length + max_e1 + max_r
                    perm_mask = torch.zeros((batch_size, seq_length, seq_length), dtype=torch.float, device=device)
                    perm_mask[:, :, start_pos:] = 1.0
                    perm_mask = torch.triu(perm_mask)
                    target_mapping = torch.zeros((batch_size, max_e2, seq_length), dtype=torch.float, device=device)
                    for i in range(max_e2):
                        target_mapping[:, i, start_pos + i] = 1.0
                    inputs = {'input_ids': input_ids, 'input_mask': input_mask, 'perm_mask': perm_mask, 'target_mapping': target_mapping}
                    outputs = model(**inputs)
                    logits = outputs[0]
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lm_labels.contiguous().view(-1), ignore_index=-1)
                elif args.model_type == "xlnet" and args.predict_rel:
                    padding_length = len(encoded_padding)
                    batch_size = input_ids.size(0)
                    seq_length = input_ids.size(1)
                    lm_labels = lm_labels[:, max_e1 + padding_length : max_e1 + padding_length + max_r]
                    perm_mask = torch.zeros((batch_size, seq_length, seq_length), dtype=torch.float, device=device)
                    perm_mask[:, :, max_e1 + padding_length : max_e1 + padding_length + max_r] = 1.0
                    perm_mask = torch.triu(perm_mask)
                    perm_mask[:, max_e1 + max_r + padding_length:, max_e1 + padding_length : max_e1 + padding_length + max_r] = 1.0
                    target_mapping = torch.zeros((batch_size, max_r, seq_length), dtype=torch.float, device=device)
                    for i in range(max_r):
                        target_mapping[:, i, max_e1 + padding_length + i] = 1.0
                    inputs = {'input_ids': input_ids, 'input_mask': input_mask, 'perm_mask': perm_mask, 'target_mapping': target_mapping}
                    outputs = model(**inputs)
                    logits = outputs[0]
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lm_labels.contiguous().view(-1), ignore_index=-1)
                elif args.model_type == "gpt2":
                    results = model(input_ids, labels=lm_labels, input_mask=input_mask)
                    loss, logits, past = results
                elif args.model_type == "openai-gpt":
                    results = model(input_ids, labels=lm_labels, input_mask=input_mask)
                    loss, logits = results
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                loss = loss.item()
                nb_tr_steps += 1
                if nb_tr_steps % (args.eval_per_steps/10) == 0:
                    PPL = np.exp(loss) if loss < 300 else np.inf
                    print("Training loss:", loss, "ppl:", PPL)
                if nb_tr_steps % args.eval_per_steps == 0:
                    model.eval()
                    # evaluate
                    eval_loss = evaluate(model, eval_dataloader, tokenizer, max_e1, max_r, max_e2, args, encoded_padding).item()
                    print("\n\nevaluating\neval loss:", eval_loss, "ppl", np.exp(eval_loss) if eval_loss < 300 else np.inf)
                    # decide to save
                    if eval_loss < best_loss:
                        # save
                        save_model(model, tokenizer, args.output_dir)
                        print("model saved at step", nb_tr_steps)
                        print(str(datetime.datetime.now()))
                        print("prev loss:", best_loss, "cur loss:", eval_loss)
                        best_loss = eval_loss
                    # test
                    test_loss = evaluate(model, test_dataloader, tokenizer, max_e1, max_r, max_e2, args, encoded_padding).item()
                    print("\n\ntesting\ntest loss:", test_loss, "ppl:", np.exp(test_loss) if test_loss < 300 else np.inf)
                    model.train()
    if args.do_eval:
        model.eval()
        eval_loss = evaluate(model, eval_dataloader, tokenizer, max_e1, max_r, max_e2, args, encoded_padding, predict_rel=args.predict_rel).item()
        print("\n\nevaluating\neval loss:", eval_loss, "ppl", np.exp(eval_loss) if eval_loss < 300 else np.inf)
        test_loss = evaluate(model, test_dataloader, tokenizer, max_e1, max_r, max_e2, args, encoded_padding, predict_rel=args.predict_rel).item()
        print("\n\ntesting\ntest loss:", test_loss, "ppl:", np.exp(test_loss) if test_loss < 300 else np.inf)



if __name__ == '__main__':
    main()
