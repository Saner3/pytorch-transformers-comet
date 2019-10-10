import argparse, logging, os, sys, random
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
                                  XLNetModel, XLNetPreTrainedModel, XLNetConfig,
                                  XLNetTokenizer, WarmupLinearSchedule, cached_path)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from utils import (load_comet_dataset, save_model, tokenize_and_encode, set_seed)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIGPTCLFModel(OpenAIGPTPreTrainedModel):
    def __init__(self, model, config):
        super(OpenAIGPTCLFModel, self).__init__(config)
        self.transformer = model
        self.clf_head = SequenceSummary(config)
    def forward(self, x, input_mask=None, mc_token_ids=None, mc_labels=None):
        h = self.transformer(x, input_mask=input_mask)[0]
        logits = self.clf_head(h, mc_token_ids)
        if mc_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)),
                            mc_labels.view(-1))
            predicts = logits.argmax(-1)
            return loss, logits, predicts
        else:
            return logits

class XLNetCLFModel(XLNetPreTrainedModel):
    def __init__(self, model, config):
        super(XLNetCLFModel, self).__init__(config)
        self.transformer = model
        self.clf_head = SequenceSummary(config)
    def forward(self, x, input_mask=None, mc_token_ids=None, mc_labels=None):
        h = self.transformer(x, input_mask=input_mask)[0]
        logits = self.clf_head(h, mc_token_ids)
        if mc_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)),
                            mc_labels.view(-1))
            predicts = logits.argmax(-1)
            return loss, logits, predicts
        else:
            return logits

def pre_process_datasets(encoded_datasets, max_e1, max_r, max_e2):
    tensor_datasets = []
    input_len = max_e1 + max_r + max_e2
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
                print("one encoded sample: e1", e1, "r", r, "e2", e2, "label", label, "input_ids:", input_ids[i])
        
        input_mask = (input_ids == 0)
        all_inputs = (input_ids, labels, input_mask, mc_token_ids)
        tensor_datasets.append((torch.tensor(input_ids), torch.tensor(labels), 
                                torch.tensor(input_mask).to(torch.float32), torch.tensor(mc_token_ids)))
    return tensor_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, eval_dataloader, max_e1, max_r, max_e2):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
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
            for i in range(bs):
                if predicts[i] == labels[i]:
                    if predicts[i] == 1:
                        t_t += 1
                    else:
                        f_f += 1
                else:
                    if predicts[i] == 0:
                        t_f += 1
                    else:
                        f_t += 1
            nb_eval_steps += 1

    try:
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = (t_t + f_f) / (t_f + t_t + f_f + f_t)
    except:
        eval_loss = -1
        eval_accuracy = -1
    print(" -----------------------")
    print("| \\ label |  0  |  1  | ")
    print("|pred\\     ")
    print("|-----\\----------------")
    print("| 0 |        ", f_f, t_f)
    print("| 1 |        ", f_t, t_t)
    print("|___|___________________")
    try:
        precision = t_t / (f_t + t_t)
        print("precision: ", precision)
    except:
        pass
    try:
        recall = t_t / (t_t + t_f)
        print("recall: ", recall)
    except:
        pass
    try:
        print("F1: ", 2 * precision * recall / (precision + recall))
    except:
        pass
    return eval_loss, eval_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='openai-gpt',
                        help='pretrained model name')
    parser.add_argument('--model_name_or_path', type=str, default='', required=True,
                        help='pretrained model name')
    parser.add_argument("--toy", action='store_true', help="Test mode.")
    parser.add_argument("--fix_weights", action='store_true', help="fix weight except for the last MLP layer")
    
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="do evaluation in the end")
    parser.add_argument("--interactive", action="store_true", help="test in interactive mode")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_dataset", type=str, nargs="+", default=["data/conceptnet/train400k_CN.txt"])
    parser.add_argument("--eval_dataset", type=str, nargs="+", default=["data/conceptnet/dev1_CN.txt", "data/conceptnet/dev2_CN.txt"])
    parser.add_argument("--test_dataset", type=str, nargs="+", default=["data/conceptnet/test_CN.txt"])
    
    parser.add_argument("--add_prefix", action="store_true", 
                        help="add a prefix at the beginning of each input when train with multiple dataset")
    parser.add_argument("--add_separator", action="store_true", help="add <sep> between sub/rel/obj")
    parser.add_argument("--rel_lang", action='store_true', help="Use natural language to represent relations.")
    parser.add_argument("--max_e1", type=int, default=10)
    parser.add_argument("--max_r", type=int, default=5)
    parser.add_argument("--max_e2", type=int, default=15)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--logging_steps', type=int, default=250)
    parser.add_argument('--eval_per_steps', type=int, default=500)
    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument("--max_steps", default=100000, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    
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

    MODEL_CLASSES = {
        #"gpt2": (GPT2Model, GPT2Tokenizer, GPT2Config, GPT2CLFModel),
        "openai-gpt": (OpenAIGPTModel, OpenAIGPTTokenizer, OpenAIGPTConfig, OpenAIGPTCLFModel),
        "xlnet": (XLNetModel, XLNetTokenizer, XLNetConfig, XLNetCLFModel),
    }
    Model, Tokenizer, Config, CLFModel = MODEL_CLASSES[args.model_type]

    # Load tokenizer and model
    tokenizer = Tokenizer.from_pretrained(args.model_name_or_path)
    model = Model.from_pretrained(args.model_name_or_path)
    print("\nspecial tokens:", tokenizer.special_tokens_map)
    if args.do_train:
        if not tokenizer.eos_token:
            tokenizer.add_special_tokens({"eos_token": "<eos>"})
        # if not tokenizer.sep_token:
        #     tokenizer.add_special_tokens({"sep_token": "<sep>"})

    print("vocab size:", len(tokenizer))

    config = Config.from_pretrained(args.model_name_or_path)
    # change config
    config.num_labels = 2
    config.summary_type = "cls_index"
    config.summary_proj_to_labels=True
    config.summary_first_dropout=0.1
    model = CLFModel(model, config=config)
    model.resize_token_embeddings(len(tokenizer))
    if args.do_eval:
        # also load clf layer
        model.load_state_dict(torch.load(os.path.join(args.model_name_or_path, WEIGHTS_NAME)))
    model.to(device)
    eos_token = tokenizer.eos_token
    def prefix_mapping(filename):
        if "vg" in filename.lower():
            return "<from_VG>"
        elif "cn" in filename.lower():
            return "<from_CN>"
        elif "fb" in filename.lower():
            return "<from_FB>"
    logging.info("Loading datasets ... ")
    train_datasets = [load_comet_dataset(dataset_path=train_dataset, 
                                         eos_token=tokenizer.eos_token, 
                                         sep_token=tokenizer.sep_token,
                                         rel_lang=args.rel_lang, 
                                         toy=args.toy, 
                                         discard_negative=False,
                                         add_sep=args.add_separator,
                                         prefix=prefix_mapping(train_dataset) if args.add_prefix else None
                                        ) for train_dataset in args.train_dataset]
    eval_datasets = [load_comet_dataset(dataset_path=eval_dataset, 
                                         eos_token=tokenizer.eos_token, 
                                         sep_token=tokenizer.sep_token,
                                         rel_lang=args.rel_lang, 
                                         toy=args.toy, 
                                         discard_negative=False,
                                         add_sep=args.add_separator,
                                         prefix=prefix_mapping(eval_dataset) if args.add_prefix else None
                                        ) for eval_dataset in args.eval_dataset]
    test_datasets = [load_comet_dataset(dataset_path=test_dataset, 
                                         eos_token=tokenizer.eos_token, 
                                         sep_token=tokenizer.sep_token,
                                         rel_lang=args.rel_lang, 
                                         toy=args.toy, 
                                         discard_negative=False,
                                         add_sep=args.add_separator,
                                         prefix=prefix_mapping(test_dataset) if args.add_prefix else None
                                        ) for test_dataset in args.test_dataset]
    train_datasets = [data for train_dataset in train_datasets for data in train_dataset]
    eval_datasets = [data for eval_dataset in eval_datasets for data in eval_dataset]
    test_datasets = [data for test_dataset in test_datasets for data in test_dataset]
    if args.do_train:
        datasets = (train_datasets, eval_datasets, test_datasets)
    else:
        datasets = (eval_datasets, test_datasets)
    logging.info("Encoding datasets ...")
    encoded_datasets = tokenize_and_encode(datasets, tokenizer)
    max_e1 = args.max_e1 if not args.add_separator else (args.max_e1 + 1)
    max_r = args.max_r if not args.add_separator else (args.max_r + 1)
    max_e2 = args.max_e2 + 1    # always add <eos> 
    best_loss = 1e10
    best_step, best_accu = -1, -1

    # Prepare inputs tensors and dataloaders
    tensor_datasets = pre_process_datasets(encoded_datasets, max_e1, max_r, max_e2)
    if args.do_train:
        train_tensor_dataset, eval_tensor_dataset, test_tensor_dataset = tensor_datasets[0], tensor_datasets[1], tensor_datasets[2]
        train_data = TensorDataset(*train_tensor_dataset)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    else:
        eval_tensor_dataset, test_tensor_dataset = tensor_datasets[0], tensor_datasets[1]

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    test_data = TensorDataset(*test_tensor_dataset)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    # fix parameters, and print trainable parameters
    if args.fix_weights:
        for param in model.transformer.parameters():
            param.requires_grad = False
        for param in model.clf_head.parameters():
            param.requires_grad = True
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        t_total = len(train_dataloader) * args.num_train_epochs
        print("total steps:", t_total)
        num_warmup_steps = args.warmup_proportion * t_total
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=t_total)

        global_steps = 0
        tr_loss, logging_loss = 0.0, 0.0
        true_count, total_count, logging_true_count, logging_total_count = 0, 0, 0, 0
        model.train()
        for cur_epoch_num in range(int(args.num_train_epochs)):
            print("Epoch:", cur_epoch_num)
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                batch_size = len(batch)
                input_ids, labels, input_mask, mc_token_ids = batch
                results = model(input_ids, mc_token_ids=mc_token_ids, mc_labels=labels, input_mask=input_mask)
                loss, logits, predicts = results
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                tr_loss += loss.item()
                total_count += batch_size
                true_count += sum([1 if predicts[i] == labels[i] else 0 for i in range(batch_size)])
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_steps += 1
                    if global_steps % args.logging_steps == 0:
                        loss = (tr_loss - logging_loss)/args.logging_steps
                        accuracy = (true_count - logging_true_count)/(total_count - logging_total_count)
                        print("Step", global_steps, "Training Loss:", loss, "Accuracy", accuracy)
                        logging_loss = tr_loss
                        logging_true_count = true_count
                        logging_total_count = total_count

                    if global_steps % args.eval_per_steps == 0:
                        # evaluate
                        print("Evaluating ...")
                        model.eval()
                        eval_loss, eval_accu = evaluate(model, eval_dataloader, max_e1, max_r, max_e2)
                        print("Dev Loss", eval_loss, "dev accuracy", eval_accu)
                        print("Best step:", best_step, "best accuracy", best_accu)
                        # decide to save
                        if eval_accu > best_accu:
                            save_model(model, tokenizer, args.output_dir)
                            print("model saved at step", global_steps)
                            print("prev loss:", best_loss, "cur loss:", eval_loss)
                            print("prev accu:", best_accu, "cur accu:", eval_accu)
                            best_loss = eval_loss
                            best_accu = eval_accu
                            best_step = global_steps
                    
                        print("Testing ...")
                        eval_loss, eval_accu = evaluate(model, test_dataloader, max_e1, max_r, max_e2)
                        print("test: Loss", eval_loss, "accuracy", eval_accu)
                        model.train()
    if args.do_eval:
        model.eval()
        eval_loss, eval_accu = evaluate(model, eval_dataloader, max_e1, max_r, max_e2)
        print("eval: Loss", eval_loss, "accuracy", eval_accu)
        eval_loss, eval_accu = evaluate(model, test_dataloader, max_e1, max_r, max_e2)
        print("test: Loss", eval_loss, "accuracy", eval_accu)
                        
    # if args.interactive:
    #     while(True):
    #         print("input a tuple")
    #         e1 = input("input a subject: ")
    #         r = input("input a relation: ")
    #         e2 = input("input a object: ") + " " + tokenizer.eos_token
    #         if r in relations:
    #             r = split_into_words[r]
    #             if not r:
    #                 print("invalid relation name")
    #                 continue
    #         dataset = [(e1, r, e2, -1)]
    #         datasets = (dataset,)
    #         encoded_datasets = tokenize_and_encode(datasets, tokenizer)
    #         tensor_dataset = pre_process_datasets(encoded_datasets, input_length, max_e1, max_r, max_e2)[0]
    #         tensor_dataset = [t.to(device) for t in tensor_dataset]
    #         input_ids, label, input_mask, mc_token_ids = tensor_dataset
    #         logits = model(input_ids, mc_token_ids=mc_token_ids, input_mask=input_mask)
    #         predict = logits.argmax(-1)
    #         print(e1, r, e2)
    #         print("predict:", predict.item())
if __name__ == '__main__':
    main()