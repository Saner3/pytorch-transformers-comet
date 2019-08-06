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
""" OpenAI GPT model fine-tuning script.
    Adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    It self adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py

    This script with default values fine-tunes and evaluate a pretrained OpenAI GPT on the RocStories dataset:
        python run_openai_gpt.py \
          --model_name openai-gpt \
          --do_train \
          --do_eval \
          --train_dataset ... 
          --eval_dataset ...
          --output_dir ../log \
          --train_batch_size 16 \
"""
import argparse
import os
import random
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_transformers import (GPT2LMHeadModel, GPT2Tokenizer,
                                     AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME, WarmupLinearSchedule)


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

relations = [
    'AtLocation', 'CapableOf', 'Causes', 'CausesDesire',
    'CreatedBy', 'DefinedAs', 'DesireOf', 'Desires', 'HasA',
    'HasFirstSubevent', 'HasLastSubevent', 'HasPainCharacter',
    'HasPainIntensity', 'HasPrerequisite', 'HasProperty',
    'HasSubevent', 'InheritsFrom', 'InstanceOf', 'IsA',
    'LocatedNear', 'LocationOfAction', 'MadeOf', 'MotivatedByGoal',
    'NotCapableOf', 'NotDesires', 'NotHasA', 'NotHasProperty',
    'NotIsA', 'NotMadeOf', 'PartOf', 'ReceivesAction', 'RelatedTo',
    'SymbolOf', 'UsedFor'
]


split_into_words = {
    'AtLocation': "at location",
    'CapableOf': "capable of",
    'Causes': "causes",
    'CausesDesire': "causes desire",
    'CreatedBy': "created by",
    'DefinedAs': "defined as",
    'DesireOf': "desire of",
    'Desires': "desires",
    'HasA': "has a",
    'HasFirstSubevent': "has first subevent",
    'HasLastSubevent': "has last subevent",
    'HasPainCharacter': "has pain character",
    'HasPainIntensity': "has pain intensity",
    'HasPrerequisite': "has prequisite",
    'HasProperty': "has property",
    'HasSubevent': "has subevent",
    'InheritsFrom': "inherits from",
    'InstanceOf': 'instance of',
    'IsA': "is a",
    'LocatedNear': "located near",
    'LocationOfAction': "location of action",
    'MadeOf': "made of",
    'MotivatedByGoal': "motivated by goal",
    'NotCapableOf': "not capable of",
    'NotDesires': "not desires",
    'NotHasA': "not has a",
    'NotHasProperty': "not has property",
    'NotIsA': "not is a",
    'NotMadeOf': "not made of",
    'PartOf': "part of",
    'ReceivesAction': "receives action",
    'RelatedTo': "related to",
    'SymbolOf': "symbol of",
    'UsedFor': "used for"
}
def load_comet_dataset(dataset_path, end_token, rel_lang=True):
    """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
    with open(dataset_path, encoding='utf_8') as f:
        f = f.read().splitlines()
        output = []
        for line in tqdm(f):
            line = line.split("\t")
            line[2] += end_token
            if rel_lang:
                output.append((line[1], split_into_words[line[0]], line[2])) # e1, r, e2
            else:
                output.append((line[1], line[0], line[2]))
    return output

def pre_process_datasets(encoded_datasets, input_len, max_e1, max_r, max_e2):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.full((n_batch, input_len), fill_value=0, dtype=np.int64)
        lm_labels = np.full((n_batch, input_len), fill_value=0, dtype=np.int64)

        for i, (e1, r, e2), in enumerate(dataset):
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

        #def make_loss_mask(sequences, max_event):
        #    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #    # print(sequences.size())
        #    mask = (sequences != -1).float()
        #    mask[:, :max_event] = 0
        #    return mask.to(device)

        lm_labels = np.copy(input_ids)
        lm_labels[lm_labels == 0] = -1
        lm_labels[:, :max_e1+max_r] = -1
        input_mask = (input_ids != 0)
        all_inputs = (input_ids, lm_labels, input_mask)
        #tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
        tensor_datasets.append((torch.tensor(input_ids), torch.tensor(lm_labels), torch.tensor(input_mask).to(torch.float32)))
    return tensor_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Save a trained model
#if args.do_train:
def save_model(model, tokenizer, output_dir):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

    # Load a trained model and vocabulary that you have fine-tuned
    #model = GPT2LMHeadModel.from_pretrained(output_dir)
    #tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
    #model.to(device)

def evaluate(model, eval_dataloader, tokenizer, max_e1, max_r):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in eval_dataloader:#tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, lm_labels, input_mask = batch
        with torch.no_grad():
            results = model(input_ids, labels=lm_labels, input_mask=input_mask)
            loss, logits, past = results
            eval_loss += loss.item()
            nb_eval_steps += 1
            if nb_eval_steps == 1:
                value, indices = logits.max(dim=-1)
                sample_index = random.randint(0, logits.size()[0]-1)
                print("input:", tokenizer.decode(input_ids[sample_index].tolist()))
                print("output:", tokenizer.decode(indices[sample_index].tolist()[max_e1 + max_r - 1:]))

    eval_loss = eval_loss / nb_eval_steps
    return eval_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2',
                        help='pretrained model name')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--rel_lang", action='store_true', help="Use natural language to represent relations.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='/nas/home/jsun/comet-commonsense/data/conceptnet/train100k.txt')
    parser.add_argument('--eval_dataset1', type=str, default='/nas/home/jsun/comet-commonsense/data/conceptnet/dev1.txt')
    parser.add_argument('--eval_dataset2', type=str, default='/nas/home/jsun/comet-commonsense/data/conceptnet/dev2.txt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_per_steps', type=int, default=500)
    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)#6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load tokenizer and model
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    #model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    end_token = tokenizer.eos_token

    # Load and encode the datasets
    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.encode(obj)
        elif isinstance(obj, int):
            return obj
        return list(tokenize_and_encode(o) for o in obj)
    logger.info("Encoding dataset...")
    train_dataset = load_comet_dataset(args.train_dataset, end_token)
    eval_dataset1 = load_comet_dataset(args.eval_dataset1, end_token)
    eval_dataset2 = load_comet_dataset(args.eval_dataset2, end_token)
    eval_dataset = eval_dataset1 + eval_dataset2
    datasets = (train_dataset, eval_dataset)
    encoded_datasets = tokenize_and_encode(datasets)
    max_e1 = 10
    max_r = 5
    max_e2 = 15 + 1
    input_length = max_e1 + max_r + max_e2
    best_loss = 1e10
   
    # Prepare inputs tensors and dataloaders
    tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_e1, max_r, max_e2)
    train_tensor_dataset, eval_tensor_dataset = tensor_datasets[0], tensor_datasets[1]

    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

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
        for cur_epoch_num in range(int(args.num_train_epochs)):#, desc="Epoch"):
            print("Epoch:", cur_epoch_num)
            #tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(train_dataloader):#tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, lm_labels, input_mask = batch
                results = model(input_ids, labels=lm_labels, input_mask=input_mask)
                loss, logits, past = results
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                loss = loss.item()
                #exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                nb_tr_steps += 1
                #tqdm_bar.desc = "Training loss: {:.2e} PPL: {:.3e}".format(exp_average_loss, PPL)
                if nb_tr_steps % (args.eval_per_steps/10) == 0:
                    PPL = np.exp(loss) if loss < 300 else np.inf
                    print("Training loss:", loss, "PPL: {.3e}", PPL)
                if nb_tr_steps % args.eval_per_steps == 0:
                    # evaluate
                    eval_loss = evaluate(model, eval_dataloader, tokenizer, max_e1, max_r)
                    # decide to save
                    if eval_loss < best_loss:
                        # save
                        save_model(model, tokenizer, args.output_dir)
                        print("model saved at step", nb_tr_steps)
                        print("prev loss:", best_loss, "cur loss:", eval_loss)
                        best_loss = eval_loss
                    model.train()


if __name__ == '__main__':
    main()
