#!/usr/bin/env python3
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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
sys.path.insert(0, "..")
from utils import (set_seed, split_into_words,
                   load_comet_dataset, tokenize_and_encode)
from pytorch_transformers import BertForMaskedLM, BertConfig, BertTokenizer, RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import argparse
import logging
from tqdm import trange

import pickle
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

#ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig)), ())
ALL_MODELS = []

MODEL_CLASSES = {
    'bert': (BertForMaskedLM, BertTokenizer)
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="Output file to store results")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--is_greedy", action='store_true',
                        help="Use greedy decoding or topk/topp.")
    parser.add_argument('--mask_parts', type=str, default="",
                        help="e1, r, or e2, which part to mask and predict")
    parser.add_argument("--rel_lang", action='store_true',
                        help="Use natural language for relations.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--test_dataset', type=str,
                        default='data/conceptnet/test.txt')
    parser.add_argument('--eval_batch_size', type=int, default=1)
    args = parser.parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        # No generation bigger than model size
        args.length = model.config.max_position_embeddings
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    print(args)
    logger.info("Encoding dataset...")
    eos_token = tokenizer.sep_token
    mask_token_id = tokenizer.encode(tokenizer.mask_token)[0]
    eos_token_id = tokenizer.encode(eos_token)[0]
    print("\nspecial tokens:", tokenizer.special_tokens_map)
    # Load and encode the datasets
    test_dataset = load_comet_dataset(
        args.test_dataset, eos_token, rel_lang=args.rel_lang, sep=True)
    encoded_dataset = tokenize_and_encode(test_dataset, tokenizer)
    max_e1 = 10
    max_r = 5
    max_e2 = 15 + 1
    input_length = max_e1 + max_r + max_e2
    test_tensor_dataset = pre_process_datasets(
        [encoded_dataset], input_length, max_e1, max_r, max_e2, mask_parts=args.mask_parts, mask_token=mask_token_id)[0]
    test_data = TensorDataset(*test_tensor_dataset)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
    model.eval()
    results = []
    logger.setLevel(level=logging.CRITICAL)
    for step, batch in tqdm(enumerate(test_dataloader)):
        batch = tuple([t.to(device) for t in batch])
        batch_size = len(batch)
        input_ids, lm_labels, input_mask = batch
        with torch.no_grad():
            loss, logits = model(input_ids, masked_lm_labels=lm_labels, attention_mask=input_mask)
            predict_tokens = torch.argmax(logits, dim=-1)
            for i, single_out in enumerate(predict_tokens):
                e1 = input_ids[i, :max_e1]
                r = input_ids[i, max_e1:max_e1+max_r]
                e2 = input_ids[i, max_e1+max_r:max_e1+max_r+max_e2]
                if args.mask_parts == "e1":
                    truth = lm_labels[i, :max_e1]
                    e1 = single_out[:max_e1]
                elif args.mask_parts == "r":
                    truth = lm_labels[i, max_e1:max_e1+max_r]
                    r = single_out[max_e1:max_e1+max_r]
                elif args.mask_parts == "e2":
                    truth = lm_labels[i, max_e1+max_r:max_e1+max_r+max_e2]
                    e2 = single_out[max_e1+max_r:max_e1+max_r+max_e2]
                e1 = e1.tolist()
                e2 = e2.tolist()
                r = r.tolist()
                truth = truth.tolist()
                e1 = tokenizer.decode(e1, clean_up_tokenization_spaces=True)[0]
                r = tokenizer.decode(r, clean_up_tokenization_spaces=True)[0]
                e2 = tokenizer.decode(e2, clean_up_tokenization_spaces=True)[0]
                truth = tokenizer.decode(truth, clean_up_tokenization_spaces=True)[0]
                results.append({'e1': e1, 'r': r, 'sequence': e2, 'reference': truth})
    output_file = open(args.output_file, "wb")
    pickle.dump(results, output_file)


if __name__ == '__main__':
    main()
