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

import argparse, logging, os, sys, random, datetime, math
import numpy as np
import torch
sys.path.insert(0, "..")
from pytorch_transformers import (CONFIG_NAME, WEIGHTS_NAME, AdamW,
                                  GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
                                  XLNetLMHeadModel, XLNetTokenizer, XLNetConfig, 
                                  OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig,
                                  WarmupLinearSchedule, cached_path)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
import torch.nn.functional as F
from utils import (load_comet_dataset, save_model, tokenize_and_encode, 
                            set_seed)

logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt = "%m/%d/%Y %H:%M:%S",
                    level = logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO: move print to logging

def pre_process_datasets(encoded_datasets, max_e1, max_r, max_e2, predict_part="obj"):
    tensor_datasets = []
    input_len = max_e1 + max_e2 + max_r
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
            input_ids[i, max_e1:max_e1 + len(r)] = r
            input_ids[i, max_e1+max_r:max_e1+max_r+len(e2)] = e2
            if i == 0:
                print("one encoded sample: e1", e1, "r", r, "e2", e2, "input_ids:", input_ids[i])

        lm_labels = np.copy(input_ids)
        # do not calculate loss on paddings
        lm_labels[lm_labels == 0] = -1  
        if predict_part == "obj":
            # do not calculate loss on sub/rel
            lm_labels[:, :max_e1 + max_r] = -1   
        # Actually below three condition is not needed 
        # because in xlnet steps, lm_labels are cropped 
        elif predict_part == "rel":
            # do not calculate loss on sub/obj
            lm_labels[:, :max_e1] = -1
            lm_labels[:, max_e1 + max_r:] = -1
        elif predict_part == "sub":
            # do not calculate loss on rel/obj
            lm_labels[:, max_e1:] = -1
        elif predict_part == "all":
            NotImplementedError
        else:
            print("unknown predict_part, must be obj, rel or sub")
            exit()
        input_mask = (input_ids == 0)   # attention mask
        all_inputs = (input_ids, lm_labels, input_mask)
        tensor_datasets.append((torch.tensor(input_ids), torch.tensor(lm_labels), 
                                torch.tensor(input_mask).to(torch.float32)))
    return tensor_datasets


def batch_step(model, model_type, batch, predict_part, max_e1, max_r, max_e2, prefix):
    input_ids, lm_labels, input_mask = batch
    batch_size = input_ids.size(0)
    if model_type == "xlnet" and predict_part == "obj":
        seq_length = input_ids.size(1)
        lm_labels = lm_labels[:, -max_e2:]
        perm_mask = torch.zeros((batch_size, seq_length, seq_length), dtype=torch.float, device=device)
        perm_mask[:, :, max_e1 + max_r:] = 1.0
        perm_mask = torch.triu(perm_mask)
        target_mapping = torch.zeros((batch_size, max_e2, seq_length), dtype=torch.float, device=device)
        for i in range(max_e2):
            target_mapping[:, i, max_e1 + max_r + i] = 1.0
        inputs = {"input_ids": input_ids, "input_mask": input_mask, "perm_mask": perm_mask, "target_mapping": target_mapping}
        outputs = model(**inputs)
        logits = outputs[0]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lm_labels.contiguous().view(-1), ignore_index=-1)
    elif model_type == "xlnet" and predict_part == "rel":
        seq_length = input_ids.size(1)
        lm_labels = lm_labels[:, max_e1 : max_e1 + max_r]
        perm_mask = torch.zeros((batch_size, seq_length, seq_length), dtype=torch.float, device=device)
        perm_mask[:, :, max_e1 : max_e1 + max_r] = 1.0
        perm_mask = torch.triu(perm_mask)
        perm_mask[:, max_e1 + max_r:, max_e1 : max_e1 + max_r] = 1.0
        target_mapping = torch.zeros((batch_size, max_r, seq_length), dtype=torch.float, device=device)
        for i in range(max_r):
            target_mapping[:, i, max_e1 + i] = 1.0
        inputs = {"input_ids": input_ids, "input_mask": input_mask, "perm_mask": perm_mask, "target_mapping": target_mapping}
        outputs = model(**inputs)
        logits = outputs[0]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lm_labels.contiguous().view(-1), ignore_index=-1)
    elif model_type == "xlnet" and predict_part == "sub":
        # ! if there is a prefix, we should not perm-mask it, 
        # ! and should not compute loss on that token.
        seq_length = input_ids.size(1)
        lm_labels = lm_labels[:, :max_e1] if not prefix else lm_labels[:, 1:max_e1]
        perm_mask = torch.zeros((batch_size, seq_length, seq_length), dtype=torch.float, device=device)
        perm_mask[:, :, :max_e1] = 1.0
        perm_mask = torch.triu(perm_mask)
        perm_mask[:, max_e1:, :max_e1] = 1.0
        num_e1_tokens = max_e1
        if prefix:
            perm_mask[:, :, 0] = 0.0
            num_e1_tokens = max_e1 - 1
        target_mapping = torch.zeros((batch_size, num_e1_tokens, seq_length), dtype=torch.float, device=device)
        for i in range(num_e1_tokens):
            if not prefix:
                target_mapping[:, i, i] = 1.0
            else:
                target_mapping[:, i, 1+i] = 1.0
        inputs = {"input_ids": input_ids, "input_mask": input_mask, "perm_mask": perm_mask, "target_mapping": target_mapping}
        outputs = model(**inputs)
        logits = outputs[0]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lm_labels.contiguous().view(-1), ignore_index=-1)
    elif model_type == "gpt2":
        results = model(input_ids, labels=lm_labels, input_mask=input_mask)
        loss, logits, _ = results
    elif model_type == "openai-gpt":
        results = model(input_ids, labels=lm_labels, input_mask=input_mask)
        loss, logits = results
    return loss, logits


def evaluate(model, model_type, predict_part, eval_dataloader, tokenizer, max_e1, max_r, max_e2, add_prefix):
    model.eval()
    eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    num_displays = 5
    display_batch_indices = list(range(len(eval_dataloader)))
    random.shuffle(display_batch_indices)
    display_batch_indices = display_batch_indices[:num_displays]
    eos_token = tokenizer.encode(tokenizer.eos_token)[0]
    print("\n\nsome examples")
    for batch_idx, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(device) for t in batch)
        batch_size = len(batch)
        input_ids, lm_labels, input_mask = batch
        with torch.no_grad():
            loss, logits = batch_step(model, model_type, batch, predict_part, max_e1, max_r, max_e2, add_prefix)
            eval_loss += loss * batch_size
            nb_eval_steps += batch_size
            # print some examples
            if batch_idx in display_batch_indices:
                value, indices = logits.max(dim=-1)
                sample_index = random.randint(0, batch_size - 1)
                print("input:", tokenizer.decode(input_ids[sample_index].tolist()))
                if not model_type == "xlnet":
                    output = indices[sample_index].tolist()[max_e1 + max_r - 1:]
                else:
                    output = indices[sample_index].tolist()
                print("output ids:", output)
                try:
                    eos_pos = output.index(eos_token)
                    output = tokenizer.decode(output[:eos_pos])
                except:
                    output = tokenizer.decode(output)
                print("output:", output)

    eval_loss = eval_loss / nb_eval_steps
    return eval_loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="openai-gpt",
                        help="model type: openai-gpt/gpt2/xlnet/...")
    parser.add_argument("--model_name_or_path", type=str, default="openai-gpt", help="pretrained model path")              
    parser.add_argument("--toy", action="store_true", help="test code")

    parser.add_argument("--do_train", action="store_true", help="do training")
    parser.add_argument("--do_eval", action="store_true", help="do evaluation in the end")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_dataset", type=str, nargs="+", default=["data/conceptnet/train100k_CN.txt"])
    parser.add_argument("--eval_dataset", type=str, nargs="+", default=["data/conceptnet/dev1_CN.txt", "data/conceptnet/dev2_CN.txt"])
    parser.add_argument("--test_dataset", type=str, nargs="+", default=["data/conceptnet/test_CN.txt"])
    
    parser.add_argument("--add_prefix", action="store_true", 
                        help="add a prefix at the beginning of each input when train with multiple dataset")
    parser.add_argument("--add_separator", action="store_true", help="add <sep> between sub/rel/obj")
    parser.add_argument("--predict_part", type=str, default="obj", choices=["sub", "rel", "obj", "all"],
                        help="predict which part of the triples")
    parser.add_argument("--max_e1", type=int, default=10)
    parser.add_argument("--max_r", type=int, default=5)
    parser.add_argument("--max_e2", type=int, default=15)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no_pretrain", action="store_true", help="w/o pretrained parameters initialized")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--logging_steps', type=int, default=250)
    parser.add_argument("--eval_per_steps", type=int, default=500)
    parser.add_argument("--num_train_epochs", type=int, default=-1)
    parser.add_argument("--max_steps", default=100000, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_proportion", type=float, default=0.002)
    parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    args = parser.parse_args()
    print(args)

    assert(args.predict_part == "obj" or args.model_type == "xlnet")

    set_seed(args.seed)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    MODEL_CLASSES = {
        "gpt2": (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config),
        "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig),
        "xlnet": (XLNetLMHeadModel, XLNetTokenizer, XLNetConfig),
    }
    Model, Tokenizer, Config = MODEL_CLASSES[args.model_type]

    # load pretrained model
    tokenizer = Tokenizer.from_pretrained(args.model_name_or_path)
    # add special tokens
    # TODO: something feels not so right
    print("\nspecial tokens:", tokenizer.special_tokens_map)
    if not tokenizer.eos_token:
        tokenizer.add_special_tokens({"eos_token": "<eos>"})
    if not tokenizer.sep_token:
        tokenizer.add_special_tokens({"sep_token": "<sep>"})
    
    tokenizer.add_tokens(["<from_CN>", "<from_VG>", "<from_FB>"])

    if args.no_pretrain:
        # from scratch
        config = Config.from_pretrained(args.model_type)
        model = Model(config)
    else:
        model = Model.from_pretrained(args.model_name_or_path)

    print("vocab size:", len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    # Here is a bug:
    # the original HuggingFace code only resize LMHead weight but not LMHead bias, it will cause runtime error
    # here we manually change the size of LMHead bias in a silly way
    if args.model_type == "xlnet":
        from torch.nn.parameter import Parameter
        model.lm_loss.bias = Parameter(torch.Tensor(len(tokenizer)))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(model.lm_loss.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(model.lm_loss.bias, -bound, bound)
        print("weight size:", model.lm_loss.weight.size())
        print("bias size:", model.lm_loss.bias.size())
    model.to(device)
    
    print("\nspecial tokens:", tokenizer.special_tokens_map)

    # Load and encode the datasets
    logger.info("Loading datasets ...")

    def prefix_mapping(filename):
        if "vg" in filename.lower():
            return "<from_VG>"
        elif "cn" in filename.lower():
            return "<from_CN>"
        elif "fb" in filename.lower():
            return "<from_FB>"

    def rel_lang(filename):
        if "vg" in filename.lower():
            return False
        elif "cn" in filename.lower():
            return True
        elif "easyfb" in filename.lower():
            return False
        elif "fb" in filename.lower():
            return True

    train_datasets = [load_comet_dataset(dataset_path=train_dataset, 
                                         eos_token=tokenizer.eos_token, 
                                         sep_token=tokenizer.sep_token,
                                         rel_lang=rel_lang(train_dataset), 
                                         toy=args.toy, 
                                         discard_negative=True,
                                         add_sep=args.add_separator,
                                         prefix=prefix_mapping(train_dataset) if args.add_prefix else None
                                        ) for train_dataset in args.train_dataset]
    eval_datasets = [load_comet_dataset(dataset_path=eval_dataset, 
                                         eos_token=tokenizer.eos_token, 
                                         sep_token=tokenizer.sep_token,
                                         rel_lang=rel_lang(eval_dataset), 
                                         toy=args.toy, 
                                         discard_negative=True,
                                         add_sep=args.add_separator,
                                         prefix=prefix_mapping(eval_dataset) if args.add_prefix else None
                                        ) for eval_dataset in args.eval_dataset]
    test_datasets = [load_comet_dataset(dataset_path=test_dataset, 
                                         eos_token=tokenizer.eos_token, 
                                         sep_token=tokenizer.sep_token,
                                         rel_lang=rel_lang(test_dataset), 
                                         toy=args.toy, 
                                         discard_negative=True,
                                         add_sep=args.add_separator,
                                         prefix=prefix_mapping(test_dataset) if args.add_prefix else None
                                        ) for test_dataset in args.test_dataset]
    train_datasets = [data for train_dataset in train_datasets for data in train_dataset]
    eval_datasets = [data for eval_dataset in eval_datasets for data in eval_dataset]
    test_datasets = [data for test_dataset in test_datasets for data in test_dataset]
    datasets = (train_datasets, eval_datasets, test_datasets)
    logger.info("Encoding datasets ...")
    encoded_datasets = tokenize_and_encode(datasets, tokenizer)
    max_e1 = args.max_e1 if not args.add_separator else (args.max_e1 + 1)
    max_r = args.max_r if not args.add_separator else (args.max_r + 1)
    max_e2 = args.max_e2 + 1    # always add <eos> 
    best_loss = 1e10
   
    # Prepare inputs tensors and dataloaders
    tensor_datasets = pre_process_datasets(encoded_datasets, max_e1, max_r, max_e2, predict_part=args.predict_part)
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

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_datasets))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Each Epoch has %d steps, and %d actual steps w/ accumulation", 
                    len(train_dataloader), len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Total train batch size (w. accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
            ]
        print("total steps:", t_total)
        num_warmup_steps = args.warmup_proportion * t_total
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=t_total)

        global_steps = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.train()
        for cur_epoch_num in range(int(args.num_train_epochs)):
            print("Epoch:", cur_epoch_num)
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                batch_size = len(batch)
                loss, logits = batch_step(model, args.model_type, batch, args.predict_part, max_e1, max_r, max_e2, args.add_prefix)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_steps += 1
                    if global_steps % args.logging_steps == 0:
                        loss = (tr_loss - logging_loss)/args.logging_steps
                        PPL = np.exp(loss) if loss < 300 else np.inf
                        print("Step", global_steps, "Training Loss:", loss, "ppl:", PPL)
                        logging_loss = tr_loss

                    if global_steps % args.eval_per_steps == 0:
                        model.eval()
                        # evaluate
                        eval_loss = evaluate(model, args.model_type, args.predict_part, eval_dataloader, tokenizer, max_e1, max_r, max_e2, args.add_prefix)
                        print("\n\nevaluating\neval loss:", eval_loss, "ppl", np.exp(eval_loss) if eval_loss < 300 else np.inf)
                        # decide to save
                        if eval_loss < best_loss:
                            # save
                            save_model(model, tokenizer, args.output_dir)
                            print("model saved at step", global_steps)
                            print(str(datetime.datetime.now()))
                            print("prev loss:", best_loss, "cur loss:", eval_loss)
                            best_loss = eval_loss
                        # test
                        test_loss = evaluate(model, args.model_type, args.predict_part, test_dataloader, tokenizer, max_e1, max_r, max_e2, args.add_prefix)
                        print("\n\ntesting\ntest loss:", test_loss, "ppl:", np.exp(test_loss) if test_loss < 300 else np.inf)
                        model.train()
                
    if args.do_eval:
        model.eval()
        eval_loss = evaluate(model, args.model_type, args.predict_part, eval_dataloader, tokenizer, max_e1, max_r, max_e2, args.add_prefix)
        print("\n\nevaluating\neval loss:", eval_loss, "ppl", np.exp(eval_loss) if eval_loss < 300 else np.inf)
        test_loss = evaluate(model, args.model_type, args.predict_part, test_dataloader, tokenizer, max_e1, max_r, max_e2, args.add_prefix)
        print("\n\ntesting\ntest loss:", test_loss, "ppl:", np.exp(test_loss) if test_loss < 300 else np.inf)



if __name__ == "__main__":
    main()
