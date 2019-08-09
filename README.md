# Commonsense Project

In order to do the transfer from GPT to other transformer models more easily, I re-implemented COMET based on the huggingface-pytorch-transformer API: https://github.com/huggingface/pytorch-transformer. So most of the experiments are not done with COMET's original codes.

First, change the working directory to `examples`

```
cd examples
```

## Setup

## Training

**Train COMET model (on gpt), as in the original paper** https://arxiv.org/pdf/1906.05317.pdf)

```
python scripts/comet-train.py --output_dir SOME_DIR --do_train --model_name openai-gpt --train_batch_size 32
```

**Train COMET model on gpt-2**

```
python scripts/comet-train.py --output_dir SOME_DIR --do_train --model_name gpt2 --train_batch_size 32
```

**Train COMET model on XLNet**

python scripts/comet-train.py --output_dir SOME_DIR --do_train --model_name xlnet-large-cased --train_batch_size 4

**Train COMET model with gpt framework but from scratch (no pre-train)**

```
python scripts/comet-train.py --output_dir SOME_DIR --do_train --model_name openai-gpt --train_batch_size 32 --no_pretrain
```

## Evaluating

**Get interactive generation results**

```
python scripts/interactive.py --model_type openai-gpt/gpt2/xlnet --model_name_or_path NAME_OR_PATH 
```

if the argument `NAME_OR_PATH` is the same as `model_type`, i.e. `gpt2` or `openai-gpt` or `xlnet`, then it will be evaluating on the original pretrained model (without fine tuning on ConceptNet). else if we are to test the COMET model that we have trained, we should provide a specific path of the saved model directory.

**Measuring the likelihood of a given sentence**

```
python scripts/get_prob.py --model_type openai-gpt/gpt2/xlnet --model_name_or_path NAME_OR_PATH 
```

the output score is the negative log likelihood of a given sentence.

the score from commonsense sentence should be lower than from the anti-commonsense one

**Evaluate on metrics**

To use the Scorer in COMET paper, first download a pretrained scorer from http://.../ and unzip to examples/ckbc_demo

To evaluate a model, first generate the results of test set into a file

```
python scripts/generate.py --model_type openai-gpt/gpt2/xlnet --model_name_or_path NAME_OR_PATH --output_file OUTFILE
```

Then run the next script to get the scores

```
python scripts/evaluate.py --input_file OUTFILE
```

### Applying COMET model to distinguish pos/neg samples

**simply use a threshold.**

```

```

**train a classifier on top of COMET models**

Training:
```
python scripts/train_classifier.py --model_type openai-gpt/gpt2/xlnet --do_train --train_batch_size 32 --model_name_or_path NAME_OR_PATH
```
The classifier is trained on top of the NAME_OR_PATH model, for example, if we want to train the classifier on top of COMET, specify the path to a pretrained COMET model. If we want to fix the parameters of the transformer during training, add `--fix_weights` argument.

Test a classifier:
```
python scripts/use_classifier.py --model_type openai-gpt/gpt2/xlnet --interactive --eval_testset --model_path PATH
```








