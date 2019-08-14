import argparse
import os, sys
import random
import logging
from tqdm import tqdm, trange

#put pytorch_transformers on the path
sys.path.insert(0, "..")
sys.path.insert(1, ".")

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_transformers import (OpenAIGPTModel, OpenAIGPTTokenizer, SequenceSummary, OpenAIGPTPreTrainedModel,
                                    GPT2Tokenizer, GPT2Model, OpenAIGPTConfig, 
                                     AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME, WarmupLinearSchedule)

from scripts.train_classifier import (OpenAIGPTCLFModel, pre_process_datasets, evaluate)
from utils import (load_comet_dataset, save_model, split_into_words, tokenize_and_encode, set_seed, relations)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='gpt2',
                        help='pretrained model name')
    parser.add_argument("--rel_lang", action='store_true', help="Use natural language to represent relations.")
    parser.add_argument("--interactive", action='store_true')
    parser.add_argument("--eval_testset", action='store_true')
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--two_test_dataset", action='store_true', help="do we have two test dataset")
    parser.add_argument('--eval_dataset2', type=str, default='data/conceptnet/dev2.txt')
    parser.add_argument('--test_dataset', type=str, default='data/conceptnet/test.txt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_per_steps', type=int, default=500)
    parser.add_argument('--eval_batch_size', type=int, default=16)

    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    # Load tokenizer and model
    tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_path)
    if args.model_type == "openai-gpt" or args.model_type == "gpt2":
        tokenizer.add_special_tokens({"bos_token": "<bos>", 
                                    "eos_token": "<eos>",
                                    "unk_token": "<unk>"})
    print("vocab size:", len(tokenizer))
    config = OpenAIGPTConfig.from_pretrained(args.model_path)
    model = OpenAIGPTModel(config)
    model = OpenAIGPTCLFModel(model, config)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(os.path.join(args.model_path, WEIGHTS_NAME)))
    print(model.config)
    model.to(device)
    end_token = tokenizer.eos_token
    model.eval()

    if args.eval_testset:
        logger.info("Encoding dataset...")
        if args.two_test_dataset:
            test_dataset1 = load_comet_dataset(args.eval_dataset2, end_token, rel_lang=args.rel_lang, discard_negative=False)
        else:
            test_dataset1 = []
        
        test_dataset2 = load_comet_dataset(args.test_dataset, end_token, rel_lang=args.rel_lang, discard_negative=False)
        datasets = (test_dataset1, test_dataset2)
        encoded_datasets = tokenize_and_encode(datasets, tokenizer)
        max_e1 = 10
        max_r = 5
        max_e2 = 15 + 1
        input_length = max_e1 + max_r + max_e2
        best_loss = 1e10
        best_step = -1
        best_accu = -1

        # Prepare inputs tensors and dataloaders
        tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_e1, max_r, max_e2)
        test_tensor_dataset1, test_tensor_dataset2 = tensor_datasets[0], tensor_datasets[1]

        test_data1 = TensorDataset(*test_tensor_dataset1)
        test_sampler1 = SequentialSampler(test_data1)
        test_dataloader1 = DataLoader(test_data1, sampler=test_sampler1, batch_size=args.eval_batch_size)

        test_data2 = TensorDataset(*test_tensor_dataset2)
        test_sampler2 = SequentialSampler(test_data2)
        test_dataloader2 = DataLoader(test_data2, sampler=test_sampler2, batch_size=args.eval_batch_size)


        # Prepare optimizer
        print("Testing ...")
        eval_loss, eval_accu = evaluate(model, test_dataloader1, tokenizer, max_e1, max_r, print_wrong=True)
        print("dev2: Loss", eval_loss, "accuracy", eval_accu)
        eval_loss, eval_accu = evaluate(model, test_dataloader2, tokenizer, max_e1, max_r, print_wrong=True)
        print("test: Loss", eval_loss, "accuracy", eval_accu)
    
    max_e1 = 10
    max_r = 5
    max_e2 = 15 + 1
    input_length = max_e1 + max_r + max_e2

    if args.interactive:
        while(True):
            print("input a tuple")
            e1 = input("input a subject: ")
            r = input("input a relation: ")
            e2 = input("input a object: ") + " " + tokenizer.eos_token
            if r in relations:
                r = split_into_words[r]
            dataset = [(e1, r, e2, -1)]
            datasets = (dataset,)
            encoded_datasets = tokenize_and_encode(datasets, tokenizer)
            tensor_dataset = pre_process_datasets(encoded_datasets, input_length, max_e1, max_r, max_e2)[0]
            tensor_dataset = [t.to(device) for t in tensor_dataset]
            input_ids, label, input_mask, mc_token_ids = tensor_dataset
            logits = model(input_ids, mc_token_ids=mc_token_ids, input_mask=input_mask)
            predict = logits.argmax(-1)
            print(e1, r, e2)
            print("predict:", predict.item())

if __name__ == '__main__':
    main()
