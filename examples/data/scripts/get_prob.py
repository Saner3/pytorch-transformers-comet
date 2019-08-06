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

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
<<<<<<< HEAD
from torch.nn import CrossEntropyLoss
=======
>>>>>>> b7b14d583f606144c32547d8aa6b408eb9a733ae
import numpy as np

from pytorch_transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
from pytorch_transformers import TransfoXLLMHeadModel, TransfoXLTokenizer


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

<<<<<<< HEAD
def get_prob(model, sentence, tokenizer, args, skip=False):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_loss = 0
    input_ids = tokenizer.encode(sentence + " " + tokenizer.eos_token)
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    labels = input_ids.clone().to(device)
    with torch.no_grad():
        results = model(input_ids) # no input_mask because we use no paddings here
        if args.model_type == "gpt2":
            logits, past = results
        elif args.model_type == "openai-gpt":
            logits = results[0]
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-1, reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        if not skip:
            length = loss.size(0)
            for i in range(length):
                print(tokenizer.decode(input_ids[0, :i+1].tolist()), \
                    "->", tokenizer.decode(input_ids[0, i+1].tolist()), loss[i].item())
            print(loss.mean().item())
    
=======

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
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, tokenizer, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, is_xlnet=False, device='cpu', padding_length=0):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    length = context.size(1) - padding_length
    logprob = 0.
    with torch.no_grad():
        for step in range(length):
            if step+1 >= length:
                break
            generated = context[:, :padding_length+step+1]
            inputs = {'input_ids': generated}
            if is_xlnet: 
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :]
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            #next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            next_token_probs = F.log_softmax(filtered_logits, dim=-1)
            wish = context[:, padding_length+step+1]
            prob = torch.gather(next_token_probs, dim=-1, index=wish)
            print("length:", length, "prob ", tokenizer.decode(generated[0].tolist()[padding_length:]), "->", tokenizer.decode(context[0, padding_length+step+1].tolist()), prob.item())
            logprob += prob.item()
        logprob /= (length-1)
    return logprob


>>>>>>> b7b14d583f606144c32547d8aa6b408eb9a733ae
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
<<<<<<< HEAD
=======
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
>>>>>>> b7b14d583f606144c32547d8aa6b408eb9a733ae
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
<<<<<<< HEAD
    if args.model_type == "openai-gpt" or args.model_type == "gpt2":
        tokenizer.add_special_tokens({"bos_token": "<bos>", 
                                    "eos_token": "<eos>",
                                    "unk_token": "<unk>"})
=======
>>>>>>> b7b14d583f606144c32547d8aa6b408eb9a733ae
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    print(args)
<<<<<<< HEAD
    # the first output is always different with the following ones
    # even with the same inputs. I dont know why yet, so I just skip the first time
    get_prob(model, "...", tokenizer, args, skip=True)
    while True:
        raw_text = args.prompt if args.prompt else input("Input Sentence >>> ")
        get_prob(model, raw_text, tokenizer, args)
        if args.prompt:
            break
=======
    while True:
        raw_text = args.prompt if args.prompt else input("Model prompt >>> ")
        context_tokens = tokenizer.encode(raw_text)
        padding_length = 0
        if args.model_type in ["transfo-xl", "xlnet"]:
            # Models with memory likes to have a long prompt for short inputs.
            padding = args.padding_text if args.padding_text else PADDING_TEXT
            encoded_padding = tokenizer.encode(padding)
            padding_length = len(encoded_padding)
            context_tokens = encoded_padding + context_tokens
        out = sample_sequence(
            model=model,
            tokenizer=tokenizer,
            context=context_tokens,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
            is_xlnet=bool(args.model_type == "xlnet"),
            padding_length=padding_length
        )
        #out = out[0, len(context_tokens):].tolist()
        #text = tokenizer.decode(out, clean_up_tokenization_spaces=True)
        #print(text)
        print(out)
        if args.prompt:
            break
    return text

>>>>>>> b7b14d583f606144c32547d8aa6b408eb9a733ae

if __name__ == '__main__':
    main()
