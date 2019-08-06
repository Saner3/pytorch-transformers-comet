# Commonsense Project

In order to do the transfer from GPT to other transformer models more easily, I re-implemented COMET based on the huggingface-pytorch-transformer API: https://github.com/huggingface/pytorch-transformer. So most of the experiments are not done with COMET's original codes.

First, change the working directory to `pytorch-transformers/examples`

```
cd pytorch-transformers/examples
```

## Setup

TO be added.

##Training

**Train COMET model (on gpt), as in the original paper** https://arxiv.org/pdf/1906.05317.pdf)

```
python scripts/comet-train.py --output_dir SOME_DIR --model_name openai-gpt --train_batch_size 32
```

**Train COMET model on gpt-2**

```
python scripts/comet-train.py --output_dir SOME_DIR --model_name gpt2 --train_batch_size 32
```

**Train COMET model on XLNet**

I have not integrated the XLNet-comet-training code into the same file yet, it should appear soon

**Train COMET model with gpt framework but from scratch**

```
python scripts/comet-train.py --output_dir SOME_DIR --model_name openai-gpt --train_batch_size 32 --no_pretrain
```

## Evaluating

**Get interactive generation results**

```
python scripts/interactive.py --model_type openai-gpt/gpt2/xlnet --model_name_or_path NAME_OR_PATH [--top_k 5]
```

if the argument `NAME_OR_PATH` is the same as `model_type`, i.e. `gpt2` or `openai-gpt` or `xlnet`, then it will be evaluating on the original pretrained model (without fine tuning on ConceptNet). else if we are to test the COMET model we have trained, we can provide a specific path of the saved model directory.

TODO: real top_k 

**Measuring the likelihood of a given sentence**

```
python scripts/get_prob.py 
```

the output number is the negative log likelihood of a given sentence.

the number from commonsense sentence should be lower than from the anti-commonsense one

**Evaluate on metrics**

First generate the results of test set into a file

```
python scripts/generate.py --model_type openai-gpt/gpt2/xlnet --model_name_or_path NAME_OR_PATH --output_file OUTFILE
```

Then run the next script

```
python scripts/evaluate.py OUTFILE
```

### Applying COMET model to distinguish pos/neg samples

**simply use a threshold.**

```

```

**train a classifier on top of COMET models**

```

```








