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
from utils import set_seed, split_into_words, PADDING_TEXT
from pytorch_transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig
import numpy as np
import torch.nn.functional as F
import torch
from tqdm import trange
import logging
import argparse



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (
    GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
}


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model=None, padding_length=0, length=5, batch=None, temperature=1, top_k=0, top_p=0.0, is_xlnet=False, device='cpu', is_greedy=True, predict_rel=False, max_e1=10, max_r=5, max_e2=16):
    context, _, input_mask = batch
    context = context.to(device)
    input_mask = input_mask.to(device)
    if not predict_rel:
        context = context[:, :max_e1 + max_r + padding_length]
        input_mask = input_mask[:, :max_e1 + max_r + padding_length]
    else:
        assert(is_xlnet)

    generated = context
    with torch.no_grad():
        for i in trange(length if not predict_rel else max_r):
            inputs = {"input_ids": generated, "input_mask": input_mask}
            if is_xlnet and not predict_rel:
                # add dummy token for prediction
                input_ids = torch.cat((generated, torch.zeros(
                    (1, 1), dtype=torch.long, device=device)), dim=1)
                input_mask_tmp = torch.cat((input_mask, torch.zeros(
                    (1, 1), dtype=torch.float32, device=device)), dim=1)
                perm_mask = torch.zeros(
                    (1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                # Previous tokens don't see last token
                perm_mask[:, :, -1] = 1.0
                target_mapping = torch.zeros(
                    (1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask,
                          'target_mapping': target_mapping, "input_mask": input_mask_tmp}
            elif is_xlnet and predict_rel:
                input_ids = generated
                # dummy tokens already exists in the middle
                perm_mask = torch.zeros(
                    (1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                # e1 and e2 do not see relations
                perm_mask[:, :, max_e1 + padding_length + i : max_e1 + padding_length + max_r] = 1.0
                perm_mask[:, max_e1 + padding_length + i + 1: max_e1 + padding_length + max_r, :] = 1.0
                input_mask[:, max_e1 + padding_length + i] = 0.0
                target_mapping = torch.zeros(
                    (1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, max_e1 + padding_length + i] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask,
                          'target_mapping': target_mapping, "input_mask": input_mask}
                
            outputs = model(**inputs)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p)
            if not is_greedy:
                next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1), num_samples=1).unsqueeze(0)
            else:
                next_token = torch.argmax(
                    F.softmax(filtered_logits, dim=-1), dim=-1).unsqueeze(0).unsqueeze(0)
            if not predict_rel:
                generated = torch.cat((generated, next_token), dim=1)
                input_mask = torch.cat((input_mask, torch.zeros(1, 1).to(torch.float32).to(device)), dim=1)
            else:
                generated[:, max_e1 + padding_length + i] = next_token

            
    return generated


def create_input(tokenizer, e1, r, e2, encoded_paddings, max_e1, max_r, max_e2):
    padding_lengths = len(encoded_paddings)
    input_len = max_e1 + max_r + max_e2 + padding_lengths

    input_ids = np.full((1, input_len), fill_value=0, dtype=np.int64)

    if len(e1) > max_e1:
        e1 = e1[:max_e1]
    if len(r) > max_r:
        r = r[:max_r]
    if len(e2) > max_e2:
        e2 = e2[:max_e2]
    input_ids[0, :padding_lengths] = encoded_paddings
    input_ids[0, padding_lengths:padding_lengths+len(e1)] = e1
    start_r = max_e1 + padding_lengths
    end_r = max_e1 + len(r) + padding_lengths
    input_ids[0, start_r:end_r] = r
    start_e2 = max_e1 + max_r + padding_lengths
    end_e2 = max_e1 + max_r + len(e2) + padding_lengths
    input_ids[0, start_e2:end_e2] = e2
    lm_labels = None
    input_mask = torch.FloatTensor(input_ids == 0)
    input_ids = torch.tensor(input_ids)
    all_inputs = (input_ids, lm_labels, input_mask)
    return all_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--predict_rel", action='store_true')
    parser.add_argument("--is_greedy", action='store_true')
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--padding_text", action='store_true',
                        help="add padding text for short inputs (xlnet)")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
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
    encoded_paddings = tokenizer.encode(
        PADDING_TEXT) if args.padding_text else []
    padding_length = len(encoded_paddings)
    eos_token = tokenizer.eos_token
    max_e1 = 10
    max_r = 5
    max_e2 = 15 + 1
    while True:
        e1 = input("Input an entity:")
        e1 = tokenizer.encode(e1)
        if args.predict_rel:
            r = []
            e2 = input("Input an second entity:") + " " + eos_token
            e2 = tokenizer.encode(e2)
        else:
            r = input("Input a relation:")
            r = tokenizer.encode(r)
            e2 = []
        batch = create_input(
            tokenizer, e1, r, e2, encoded_paddings, max_e1, max_r, max_e2)
        out = sample_sequence(
            model=model,
            length=args.length,
            batch=batch,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
            is_xlnet=bool(args.model_type == "xlnet"),
            is_greedy=args.is_greedy,
            max_e1=max_e1,
            max_r=max_r,
            max_e2=max_e2,
            predict_rel=args.predict_rel
        )
        out = out[0].tolist()
        try:
            eos_pos = out.index(tokenizer.encode(tokenizer.eos_token)[0])
        except:
            eos_pos = -1
        #print(eos_pos)
        #text = tokenizer.decode(
        #    out[:eos_pos], clean_up_tokenization_spaces=True)
        text = tokenizer.decode(
            out, clean_up_tokenization_spaces=True)
        print(text + "\n")
    return text


if __name__ == '__main__':
    main()
