import torch
import numpy as np

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()  # disable dropout (or leave in train mode to finetune)

tokens = roberta.encode('<mask>')
print(tokens)

print(roberta.fill_mask('The first Star wars movie came out in <mask> <mask> <mask>', topk=3))