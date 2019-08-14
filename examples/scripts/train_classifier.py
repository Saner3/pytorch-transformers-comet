import argparse
import logging
import os, sys
import random

#put pytorch_transformers on the path
sys.path.insert(0, "..")

import numpy as np
from tqdm import tqdm, trange

import torch
from torch import nn
from pytorch_transformers import (CONFIG_NAME, WEIGHTS_NAME, AdamW, GPT2Config,
                                  GPT2Model, GPT2PreTrainedModel,
                                  GPT2Tokenizer, OpenAIGPTConfig,
                                  OpenAIGPTModel, OpenAIGPTPreTrainedModel,
                                  OpenAIGPTTokenizer, SequenceSummary,
                                  WarmupLinearSchedule, cached_path)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from utils import (load_comet_dataset, save_model, tokenize_and_encode, set_seed)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIGPTCLFModel(OpenAIGPTPreTrainedModel):
    def __init__(self, model, config):#, num_features, dropout, num_labels):
        super(OpenAIGPTCLFModel, self).__init__(config)
        self.transformer = model
        self.clf_head = SequenceSummary(config)
        # n_hid = 768
        # self.clf_head = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(num_features, n_hid),
        #     nn.ReLU(),
        #     #nn.BatchNorm1d(n_hid),
        #     nn.Dropout(dropout),            
        #     nn.Linear(n_hid, n_hid // 4),
        #     nn.ReLU(),
        #     #nn.BatchNorm1d(n_hid // 4),
        #     nn.Dropout(dropout),
        #     nn.Linear(n_hid // 4, num_labels),
        # )
    def forward(self, x, input_mask=None, mc_token_ids=None, mc_labels=None):
        h = self.transformer(x, input_mask=input_mask)[0]
        # cls_index = torch.full_like(h[..., :1, :], 
        #                         h.shape[-2]-1, dtype=torch.long)
        # # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
        # h = h.gather(-2, cls_index).squeeze(-2) # shape (bsz, XX, hidden_size)
        logits = self.clf_head(h, mc_token_ids)
        #logits = self.clf_head(h)
        if mc_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)),
                            mc_labels.view(-1))
            predicts = logits.argmax(-1)
            return loss, logits, predicts
        else:
            return logits

def pre_process_datasets(encoded_datasets, input_len, max_e1, max_r, max_e2):
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.full((n_batch, input_len), fill_value=0, dtype=np.int64)
        labels = np.full((n_batch), fill_value=0, dtype=np.int64)
        mc_token_ids = np.full((n_batch), fill_value=0, dtype=np.int64)

        for i, (e1, r, e2, label), in enumerate(dataset):
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

            labels[i] = label
            mc_token_ids[i] = end_e2 - 1
            if i == 0:
                print("one encoded sample: e1", e1, "r", r, "e2", e2, "input_ids:", input_ids[i])
        
        input_mask = (input_ids == 0)
        all_inputs = (input_ids, labels, input_mask, mc_token_ids)
        tensor_datasets.append((torch.tensor(input_ids), torch.tensor(labels), 
                                torch.tensor(input_mask).to(torch.float32), torch.tensor(mc_token_ids)))
    return tensor_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, eval_dataloader, tokenizer, max_e1, max_r, print_wrong=False):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    true_num = 0
    total_num = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    t_t, f_f, t_f, f_t = 0, 0, 0, 0
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, labels, input_mask, mc_token_ids = batch
        with torch.no_grad():
            results = model(input_ids, mc_labels=labels, mc_token_ids=mc_token_ids, input_mask=input_mask)
            loss, logits, predicts = results
            eval_loss += loss.item()
            bs = logits.size(0)
            total_num += bs
            for i in range(bs):
                if predicts[i] == labels[i]:
                    true_num += 1
                    if predicts[i] == 1:
                        t_t += 1
                    else:
                        f_f += 1
                else:
                    if print_wrong:
                        print(tokenizer.decode(input_ids[i].tolist()).replace("<unk>", ""), "label:", labels[i].item(), "predict:", predicts[i].item())
                    if predicts[i] == 0:
                        t_f += 1
                    else:
                        f_t += 1
            nb_eval_steps += 1

    try:
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = true_num / total_num
    except:
        eval_loss = 0
        eval_accuracy = 0
    print(" --------------------------------------------")
    print("| \\ label |  0  |  1  | Precision | Recall")
    print("|pred\\     ")
    print("|-----\\-------------------------------------")
    try:
        print("| 0 |        ", f_f, t_f, "   %.3f"%(f_f/(f_f+t_f)), "  %.3f"%(f_f/(f_f+f_t)))
    except:
        print("| 0 |        ", f_f, t_f)
    try:
        print("| 1 |        ", f_t, t_t, "   %.3f"%(t_t/(f_t+t_t)), "  %.3f"%(t_t/(t_t+t_f)))
    except:
        print("| 1 |        ", f_t, t_t)
    print("|___|________________________________________")
    return eval_loss, eval_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='openai-gpt',
                        help='pretrained model name')
    parser.add_argument('--model_name_or_path', type=str, default='', required=True,
                        help='pretrained model name')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--rel_lang", action='store_true', help="Use natural language to represent relations.")
    parser.add_argument("--toy", action='store_true', help="Test mode.")
    parser.add_argument("--fix_weights", action='store_true', help="fix weight except for the last MLP layer")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='data/conceptnet/train400k.txt')
    parser.add_argument('--eval_dataset1', type=str, default='data/conceptnet/dev1.txt')
    parser.add_argument("--two_test_dataset", action='store_true', help="do we have two test dataset")
    parser.add_argument('--eval_dataset2', type=str, default='data/conceptnet/dev2.txt')
    parser.add_argument('--test_dataset', type=str, default='data/conceptnet/test.txt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_per_steps', type=int, default=500)
    parser.add_argument('--num_train_epochs', type=int, default=20)
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

    # Load tokenizer and model
    tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_name_or_path)
    model = OpenAIGPTModel.from_pretrained(args.model_name_or_path)
    if args.model_type == "openai-gpt" or args.model_type == "gpt2":
        tokenizer.add_special_tokens({"bos_token": "<bos>", 
                                    "eos_token": "<eos>",
                                    "unk_token": "<unk>"})
    print("vocab size:", len(tokenizer))

    config = OpenAIGPTConfig.from_pretrained(args.model_name_or_path)
    # change config
    config.num_labels = 2
    config.summary_type = "cls_index"
    config.summary_proj_to_labels=True
    config.summary_first_dropout=0.1
    model = OpenAIGPTCLFModel(model, config=config)
    model.resize_token_embeddings(len(tokenizer))
    print(model.config)
    model.to(device)
    end_token = tokenizer.eos_token

    logger.info("Encoding dataset...")
    train_dataset = load_comet_dataset(args.train_dataset, end_token, toy=args.toy, rel_lang=args.rel_lang, discard_negative=False)
    eval_dataset = load_comet_dataset(args.eval_dataset1, end_token, toy=args.toy, rel_lang=args.rel_lang, discard_negative=False)
    if args.two_test_dataset:
        test_dataset1 = load_comet_dataset(args.eval_dataset2, end_token, toy=args.toy, rel_lang=args.rel_lang, discard_negative=False)
    else:
        test_dataset1 = []
    test_dataset2 = load_comet_dataset(args.test_dataset, end_token, toy=args.toy, rel_lang=args.rel_lang, discard_negative=False)

    datasets = (train_dataset, eval_dataset, test_dataset1, test_dataset2)
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
    train_tensor_dataset, eval_tensor_dataset, test_tensor_dataset1, test_tensor_dataset2 = tensor_datasets[0], tensor_datasets[1], tensor_datasets[2], tensor_datasets[3]

    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    test_data1 = TensorDataset(*test_tensor_dataset1)
    test_sampler1 = SequentialSampler(test_data1)
    test_dataloader1 = DataLoader(test_data1, sampler=test_sampler1, batch_size=args.eval_batch_size)

    test_data2 = TensorDataset(*test_tensor_dataset2)
    test_sampler2 = SequentialSampler(test_data2)
    test_dataloader2 = DataLoader(test_data2, sampler=test_sampler2, batch_size=args.eval_batch_size)

    # fix parameters, and print trainable parameters
    if args.fix_weights:
        for param in model.transformer.parameters():
            param.requires_grad = False
        for param in model.clf_head.parameters():
            param.requires_grad = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

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
                input_ids, labels, input_mask, mc_token_ids = batch
                results = model(input_ids, mc_token_ids=mc_token_ids, mc_labels=labels, input_mask=input_mask)
                loss, logits, predicts = results
                bs = labels.size(0)
                accuracy = sum([1 if predicts[i] == labels[i] else 0 for i in range(bs)])/bs
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                loss = loss.item()
                nb_tr_steps += 1
                #tqdm_bar.desc = "Training loss: {:.2e} PPL: {:.3e}".format(exp_average_loss, PPL)
                if nb_tr_steps % (args.eval_per_steps/10) == 0:
                    print("Training Loss:", loss, "Accuracy: ", accuracy)
                if nb_tr_steps % args.eval_per_steps == 0:
                    # evaluate
                    print("Evaluating ...")
                    model.eval()
                    eval_loss, eval_accu = evaluate(model, eval_dataloader, tokenizer, max_e1, max_r)
                    print("Dev Loss", eval_loss, "dev accuracy", eval_accu)
                    print("Best step:", best_step, "best accuracy", best_accu)
                    # decide to save
                    if eval_accu > best_accu:
                        save_model(model, tokenizer, args.output_dir)
                        print("model saved at step", nb_tr_steps)
                        print("prev loss:", best_loss, "cur loss:", eval_loss)
                        print("prev accu:", best_accu, "cur accu:", eval_accu)
                        best_loss = eval_loss
                        best_accu = eval_accu
                        best_step = nb_tr_steps
                
                    print("Testing ...")
                    eval_loss, eval_accu = evaluate(model, test_dataloader1, tokenizer, max_e1, max_r)
                    print("dev2: Loss", eval_loss, "accuracy", eval_accu)
                    eval_loss, eval_accu = evaluate(model, test_dataloader2, tokenizer, max_e1, max_r)
                    print("test: Loss", eval_loss, "accuracy", eval_accu)
                    model.train()


if __name__ == '__main__':
    main()
