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
                                  BertForMaskedLM, BertConfig, BertTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                                  WarmupLinearSchedule, cached_path)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
import torch.nn.functional as F
from scripts.utils import (load_comet_dataset, save_model, tokenize_and_encode, set_seed)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def pre_process_datasets(encoded_datasets, input_len, max_e1, max_r, max_e2, mask_parts, mask_token):
    tensor_datasets = []
    assert(mask_parts in ["e1", "r", "e2"])
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.full((n_batch, input_len), fill_value=0, dtype=np.int64)
        lm_labels = np.full((n_batch, input_len), fill_value=-1, dtype=np.int64)
        for i, (e1, r, e2, label), in enumerate(dataset):
            # truncate if input is too long
            if len(e1) > max_e1:
                e1 = e1[:max_e1]
            if len(r) > max_r:
                r = r[:max_r]
            if len(e2) > max_e2:
                e2 = e2[:max_e2]

            if mask_parts == "e1":
                input_ids[i, :len(e1)] = mask_token
                lm_labels[i, :len(e1)] = e1
            else:
                input_ids[i, :len(e1)] = e1
            start_r = max_e1
            end_r = max_e1 + len(r)
            if mask_parts == "r":
                input_ids[i, start_r:end_r] = mask_token
                lm_labels[i, start_r:end_r] = r
            else:
                input_ids[i, start_r:end_r] = r
            start_e2 = max_e1 + max_r
            end_e2 = max_e1 + max_r + len(e2)
            if mask_parts == "e2":
                input_ids[i, start_e2:end_e2] = mask_token
                lm_labels[i, start_e2:end_e2] = e2
            else:
                input_ids[i, start_e2:end_e2] = e2

            if i == 0:
                print("one encoded sample: e1", e1, "r", r, "e2", e2)
                print("input_ids:", input_ids[i])
                print("lm_labels", lm_labels[i])

        input_mask = (input_ids != 0)   # attention mask
        all_inputs = (input_ids, lm_labels, input_mask)
        tensor_datasets.append((torch.tensor(input_ids), torch.tensor(lm_labels), 
                                torch.tensor(input_mask).to(torch.float32)))
    return tensor_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, eval_dataloader, tokenizer, max_e1, max_r, max_e2, args):
    model.eval()
    eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    num_displays = 5
    eos_token = tokenizer.encode(tokenizer.sep_token)[0]
    print("\n\nsome examples\n")
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        batch_size = len(batch)
        input_ids, lm_labels, input_mask = batch
        with torch.no_grad():
            results = model(input_ids, masked_lm_labels=lm_labels, attention_mask=input_mask)
            loss, logits = results
            eval_loss += loss * batch_size
            nb_eval_steps += batch_size
            # display some examples
            if num_displays:
                num_displays -= 1
                value, indices = logits.max(dim=-1)
                sample_index = random.randint(0, batch_size - 1)
                print("input:", tokenizer.decode(input_ids[sample_index].tolist()))
                output = indices[sample_index][lm_labels[sample_index]!=-1].tolist()
                print("output:", tokenizer.decode(output))

    eval_loss = eval_loss / nb_eval_steps
    return eval_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='openai-gpt',
                        help="model type: openai-gpt/gpt2/xlnet/...")
    parser.add_argument('--model_name_or_path', type=str, default='openai-gpt', help="pretrained model path")              
    parser.add_argument("--rel_lang", action='store_true', help="Use natural language to represent relations.")
    parser.add_argument("--do_train", action='store_true', help="do training")
    parser.add_argument("--do_eval", action='store_true', help="do evaluation")
    parser.add_argument("--predict_rel", action='store_true', help="predict relation rather than objects")
    parser.add_argument("--toy", action='store_true', help="test code")
    parser.add_argument("--padding_text", action='store_true', help="xlnet needs a padding text")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='data/conceptnet/train100k.txt')
    parser.add_argument('--eval_dataset1', type=str, default='data/conceptnet/dev1.txt')
    parser.add_argument('--eval_dataset2', type=str, default='data/conceptnet/dev2.txt')
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--test_dataset', type=str, default='data/conceptnet/test.txt')
    parser.add_argument('--mask_parts', type=str, default='e1, r, or e2, which part to mask and predict')
    parser.add_argument('--max_e1', type=int, default=10)
    parser.add_argument('--max_r', type=int, default=5)
    parser.add_argument('--max_e2', type=int, default=15)
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
    if args.model_type == "bert":
        Tokenizer = BertTokenizer
        Model = BertForMaskedLM
        Config = BertConfig
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

    print("vocab size:", len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    end_token = tokenizer.sep_token
    mask_token_id = tokenizer.encode(tokenizer.mask_token)[0]
    print("\nspecial tokens:", tokenizer.special_tokens_map)

    # Load and encode the datasets
    logger.info("Encoding dataset...")
    train_dataset = load_comet_dataset(args.train_dataset, end_token, rel_lang=args.rel_lang, toy=args.toy, sep=True)
    if args.eval_dataset:
        eval_dataset = load_comet_dataset(args.eval_dataset, end_token, rel_lang=args.rel_lang, toy=args.toy, sep=True)
    else:
        eval_dataset1 = load_comet_dataset(args.eval_dataset1, end_token, rel_lang=args.rel_lang, toy=args.toy, sep=True)
        eval_dataset2 = load_comet_dataset(args.eval_dataset2, end_token, rel_lang=args.rel_lang, toy=args.toy, sep=True)
        eval_dataset = eval_dataset1 + eval_dataset2
    test_dataset = load_comet_dataset(args.test_dataset, end_token, rel_lang=args.rel_lang, toy=args.toy, sep=True)
    datasets = (train_dataset, eval_dataset, test_dataset)
    encoded_datasets = tokenize_and_encode(datasets, tokenizer)
    max_e1 = args.max_e1 + 1
    max_r = args.max_r + 1
    max_e2 = args.max_e2 + 1
    input_length = max_e1 + max_r + max_e2
    best_loss = 1e10
   
    # Prepare inputs tensors and dataloaders
    tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_e1, max_r, max_e2, mask_parts=args.mask_parts, mask_token=mask_token_id)
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
                results = model(input_ids, masked_lm_labels=lm_labels, attention_mask=input_mask)
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
                    eval_loss = evaluate(model, eval_dataloader, tokenizer, max_e1, max_r, max_e2, args).item()
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
                    test_loss = evaluate(model, test_dataloader, tokenizer, max_e1, max_r, max_e2, args).item()
                    print("\n\ntesting\ntest loss:", test_loss, "ppl:", np.exp(test_loss) if test_loss < 300 else np.inf)
                    model.train()
    if args.do_eval:
        model.eval()
        eval_loss = evaluate(model, eval_dataloader, tokenizer, max_e1, max_r, max_e2, args).item()
        print("\n\nevaluating\neval loss:", eval_loss, "ppl", np.exp(eval_loss) if eval_loss < 300 else np.inf)
        test_loss = evaluate(model, test_dataloader, tokenizer, max_e1, max_r, max_e2, args).item()
        print("\n\ntesting\ntest loss:", test_loss, "ppl:", np.exp(test_loss) if test_loss < 300 else np.inf)



if __name__ == '__main__':
    main()
