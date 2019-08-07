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
""" transformer model fine-tuning on comet.
    Adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    It self adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py
"""
import argparse
import logging
import os
import random
import datetime

import numpy as np
import torch
from pytorch_transformers import (CONFIG_NAME, WEIGHTS_NAME, AdamW,
                                  GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
                                  XLNetLMHeadModel, XLNetTokenizer, XLNetConfig, 
                                  OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig,
                                  WarmupLinearSchedule, cached_path)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from utils import (load_comet_dataset, save_model, tokenize_and_encode, set_seed)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def pre_process_datasets(encoded_datasets, input_len, max_e1, max_r, max_e2):
    tensor_datasets = []
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

            input_ids[i, :len(e1)] = e1
            start_r = max_e1
            end_r = max_e1 + len(r)
            input_ids[i, start_r:end_r] = r
            start_e2 = max_e1 + max_r
            end_e2 = max_e1 + max_r + len(e2)
            input_ids[i, start_e2:end_e2] = e2
            if i == 0:
                print("one encoded sample: e1", e1, "r", r, "e2", e2, "input_ids:", input_ids[i])


        lm_labels = np.copy(input_ids)
        lm_labels[lm_labels == 0] = -1  # do not calculate loss on paddings
        lm_labels[:, :max_e1+max_r] = -1    # do not calculate loss on the first part
        input_mask = (input_ids != 0)   # attention mask
        all_inputs = (input_ids, lm_labels, input_mask)
        tensor_datasets.append((torch.tensor(input_ids), torch.tensor(lm_labels), 
                                torch.tensor(input_mask).to(torch.float32)))
    return tensor_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, eval_dataloader, tokenizer, max_e1, max_r, args):
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
            results = model(input_ids, labels=lm_labels, input_mask=input_mask)
            if args.model_name == "gpt2":
                loss, logits, past = results
            elif args.model_name == "openai-gpt":
                loss, logits = results
            eval_loss += loss * batch_size
            nb_eval_steps += batch_size
            # display some examples
            if num_displays:
                num_displays -= 1
                value, indices = logits.max(dim=-1)
                sample_index = random.randint(0, batch_size - 1)
                print("input:", tokenizer.decode(input_ids[sample_index].tolist()))
                output = indices[sample_index].tolist()[max_e1 + max_r - 1:]
                try:
                    eos_pos = output.index(eos_token)
                except:
                    eos_pos = -1
                output = tokenizer.decode(output[:eos_pos])
                print("output:", output)

    eval_loss = eval_loss / nb_eval_steps
    return eval_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2',
                        help='pretrained model name')
    parser.add_argument("--rel_lang", action='store_true', help="Use natural language to represent relations.")
    parser.add_argument("--do_train", action='store_true', help="do training")
    parser.add_argument("--toy", action='store_true', help="test code")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='data/conceptnet/train100k.txt')
    parser.add_argument('--eval_dataset1', type=str, default='data/conceptnet/dev1.txt')
    parser.add_argument('--eval_dataset2', type=str, default='data/conceptnet/dev2.txt')
    parser.add_argument('--test_dataset', type=str, default='data/conceptnet/test.txt')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument("--no_pretrain", action='store_true', help="w/o pretrained parameters initialized")
    parser.add_argument('--eval_per_steps', type=int, default=500)
    parser.add_argument('--num_train_epochs', type=int, default=64)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5) # 6.25e-5 default
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

    # Load tokenizer and model
    if args.model_name == "openai-gpt":
        Tokenizer = OpenAIGPTTokenizer
        Model = OpenAIGPTLMHeadModel
        Config = OpenAIGPTConfig
    elif args.model_name == "gpt2":
        Tokenizer = GPT2Tokenizer
        Model = GPT2LMHeadModel
        Config = GPT2Config
    else:
        exit()

    tokenizer = Tokenizer.from_pretrained(args.model_name)
    if args.no_pretrain:
        config = Config.from_pretrained(args.model_name)
        model = Model(config)
    else:
        model = Model.from_pretrained(args.model_name)
    if args.model_name == "openai-gpt" or args.model_name == "gpt2":
        tokenizer.add_special_tokens({"bos_token": "<bos>", 
                                    "eos_token": "<eos>",
                                    "unk_token": "<unk>"})
    print("vocab size:", len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    end_token = tokenizer.eos_token
    print("special tokens:", tokenizer.special_tokens_map)

    # Load and encode the datasets
    logger.info("Encoding dataset...")
    train_dataset = load_comet_dataset(args.train_dataset, end_token, toy=args.toy)
    eval_dataset1 = load_comet_dataset(args.eval_dataset1, end_token, toy=args.toy)
    eval_dataset2 = load_comet_dataset(args.eval_dataset2, end_token, toy=args.toy)
    eval_dataset = eval_dataset1 + eval_dataset2
    test_dataset = load_comet_dataset(args.test_dataset, end_token, toy=args.toy)
    datasets = (train_dataset, eval_dataset, test_dataset)
    encoded_datasets = tokenize_and_encode(datasets, tokenizer)
    max_e1 = 10
    max_r = 5
    max_e2 = 15 + 1
    input_length = max_e1 + max_r + max_e2
    best_loss = 1e10
   
    # Prepare inputs tensors and dataloaders
    tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_e1, max_r, max_e2)
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
                results = model(input_ids, labels=lm_labels, input_mask=input_mask)
                if args.model_name == "gpt2":
                    loss, logits, past = results
                elif args.model_name == "openai-gpt":
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
                    eval_loss = evaluate(model, eval_dataloader, tokenizer, max_e1, max_r, args).item()
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
                    test_loss = evaluate(model, test_dataloader, tokenizer, max_e1, max_r, args).item()
                    print("\n\ntesting\ntest loss:", test_loss, "ppl:", np.exp(test_loss) if test_loss < 300 else np.inf)
                    model.train()


if __name__ == '__main__':
    main()
