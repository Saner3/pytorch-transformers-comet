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

import pickle
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
from pytorch_transformers import TransfoXLLMHeadModel, TransfoXLTokenizer

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    #assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
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
<<<<<<< HEAD

=======
>>>>>>> b7b14d583f606144c32547d8aa6b408eb9a733ae
def load_comet_dataset(dataset_path, end_token, rel_lang=True):
    """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
    with open(dataset_path, encoding='utf_8') as f:
        f = f.read().splitlines()
        output = []
        for line in f:
            line = line.split("\t")
            if line[3] == '0':
                continue
            line[2] += end_token
            if rel_lang:
                output.append((line[1], split_into_words[line[0]], line[2])) # e1, r, e2
            else:
                output.append((line[1], line[0], line[2]))
    return output
<<<<<<< HEAD

=======
>>>>>>> b7b14d583f606144c32547d8aa6b408eb9a733ae
def pre_process_datasets(encoded_datasets, encoded_paddings, input_len, max_e1, max_r, max_e2, is_xlnet=False):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    padding_lengths = len(encoded_paddings)
    input_len += padding_lengths
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

            input_ids[i, :padding_lengths] = encoded_paddings
            input_ids[i, padding_lengths:padding_lengths+len(e1)] = e1
            start_r = max_e1+padding_lengths
            end_r = max_e1 + len(r)+padding_lengths
            input_ids[i, start_r:end_r] = r
            start_e2 = max_e1 + max_r+padding_lengths
            end_e2 = max_e1 + max_r + len(e2)+padding_lengths
            input_ids[i, start_e2:end_e2] = e2

        #def make_loss_mask(sequences, max_event):
        #    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #    # print(sequences.size())

        #    mask = (sequences != -1).float()
        #    mask[:, :max_event] = 0
        #    return mask.to(device)

        lm_labels = np.copy(input_ids) #(input_ids[:, 1:])
        lm_labels[lm_labels == 0] = -1
        lm_labels[:, :padding_lengths+max_e1+max_r] = -1
        if not is_xlnet:
            input_mask = torch.FloatTensor(input_ids != 0)
        else:
            input_mask = torch.FloatTensor(input_ids == 0)

        all_inputs = (input_ids, lm_labels, input_mask)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets


def sample_sequence(model, max_length, padding_length, tokenizer, batch, max_e1=10, max_r=5, max_e2=16, temperature=1, top_k=0, top_p=0.0, is_xlnet=False, device='cpu', is_greedy=True):
    context, _, input_mask = batch
    num_samples = context.size(0)
    assert(num_samples==1)
    context = context[:1, :max_e1 + max_r + padding_length]
    input_mask = input_mask[:1, :max_e1 + max_r + padding_length]
    context = torch.tensor(context, dtype=torch.long, device=device)
    generated = context
    with torch.no_grad():
        for _ in range(max_length):
            inputs = {'input_ids': generated, 'input_mask': input_mask}
            if is_xlnet:
                # add dummy token for prediction
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                input_mask = torch.cat((input_mask, torch.zeros((1, 1), dtype=torch.float, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'input_mask': input_mask, 'perm_mask': perm_mask, 'target_mapping': target_mapping}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if not is_greedy:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
<<<<<<< HEAD
                NotImplementedError
=======
>>>>>>> b7b14d583f606144c32547d8aa6b408eb9a733ae
            else:
                next_token = torch.argmax(F.softmax(filtered_logits, dim=-1), dim=-1).unsqueeze(0).unsqueeze(0)
            generated = torch.cat((generated, next_token), dim=1)
            if next_token.item() == tokenizer.encode(tokenizer.eos_token)[0]:
                break
            if not is_xlnet:
                input_mask = torch.cat((input_mask, torch.ones(1,1).float().to(device)), dim=-1)
    #print(generated)
    return generated[:, padding_length:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
<<<<<<< HEAD
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="Output file to store results")
=======
>>>>>>> b7b14d583f606144c32547d8aa6b408eb9a733ae
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--is_greedy", action='store_true', help="Use greedy decoding or topk/topp.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
<<<<<<< HEAD
    parser.add_argument("--top_p", type=float, default=0.0)
=======
    parser.add_argument("--top_p", type=float, default=0.9)
>>>>>>> b7b14d583f606144c32547d8aa6b408eb9a733ae
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
<<<<<<< HEAD
    parser.add_argument('--test_dataset', type=str, default='data/conceptnet/test.txt')
=======
    parser.add_argument('--test_dataset', type=str, default='/nas/home/jsun/comet-commonsense/data/conceptnet/test.txt')
>>>>>>> b7b14d583f606144c32547d8aa6b408eb9a733ae
    parser.add_argument('--eval_batch_size', type=int, default=1)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
<<<<<<< HEAD
    if args.model_type == "openai-gpt" or args.model_type == "gpt2":
        tokenizer.add_special_tokens({"bos_token": "<bos>", 
                                    "eos_token": "<eos>",
                                    "unk_token": "<unk>"})
    print("vocab size:", len(tokenizer))
    #model.resize_token_embeddings(len(tokenizer))
=======
>>>>>>> b7b14d583f606144c32547d8aa6b408eb9a733ae
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    print(args)
    logger.info("Encoding dataset...")
    end_token = tokenizer.eos_token
    # Load and encode the datasets
    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.encode(obj)
        elif isinstance(obj, int):
            return obj
        return list(tokenize_and_encode(o) for o in obj)

    test_dataset = load_comet_dataset(args.test_dataset, end_token)
    encoded_datasets = tokenize_and_encode([test_dataset])
    encoded_paddings = tokenize_and_encode(PADDING_TEXT) if bool(args.model_type == "xlnet") else []
    padding_length = len(encoded_paddings)
    print("padding_length", padding_length)
    max_e1 = 10
    max_r = 5
    max_e2 = 15 + 1
    input_length = max_e1 + max_r + max_e2
    tensor_datasets = pre_process_datasets(encoded_datasets, encoded_paddings, input_length, max_e1, max_r, max_e2, bool(args.model_type == "xlnet"))
    test_tensor_dataset = tensor_datasets[0]
    test_data = TensorDataset(*test_tensor_dataset)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
    model.eval()
    results = [] 
<<<<<<< HEAD
    for step, batch in enumerate(test_dataloader):
=======
    for step, batch in enumerate(test_dataloader):#tqdm_bar):
>>>>>>> b7b14d583f606144c32547d8aa6b408eb9a733ae
        batch = tuple(t.to(device) for t in batch)
        out = sample_sequence(
            model=model,
            padding_length=padding_length,
            tokenizer=tokenizer,
            max_length=args.length,
            batch=batch,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
            is_xlnet=bool(args.model_type == "xlnet"),
            is_greedy=args.is_greedy,
            max_e1=max_e1,
            max_r=max_r,
            max_e2=max_e2
        )
        out = out.tolist()
        for single_out in out:
            e1 = single_out[:max_e1]
            e1 = [word for word in e1 if word > 0]
            r = single_out[max_e1:max_e1+max_r]
            r =  [word for word in r if word > 0]
            e2 = single_out[max_e1+max_r:-1]
            e1 = tokenizer.decode(e1, clean_up_tokenization_spaces=True)
            r = tokenizer.decode(r, clean_up_tokenization_spaces=True)
            e2 = tokenizer.decode(e2, clean_up_tokenization_spaces=True)
            results.append({'e1': e1, 'r': r, 'sequence': e2})
<<<<<<< HEAD
    output_file = open(args.output_file, "wb")
    pickle.dump(results, output_file)
=======
    outputfile = open("gen-result-gpt2-sameargs.pickle", "wb")
    pickle.dump(results, outputfile)
>>>>>>> b7b14d583f606144c32547d8aa6b408eb9a733ae

if __name__ == '__main__':
    main()
