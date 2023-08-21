
# JP Language Model Evaluation Harness

## Leaderboard
| model                                                                                                                                                                                                                                                 |   average |   jcommonsenseqa |   jnli |   marc_ja |   jsquad |   jaqket_v2 |   xlsum_ja |   xwinograd_ja |   mgsm | eval script                                                                                                                                                                                                                                                                                                                                                     |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|-----------------:|-------:|----------:|---------:|------------:|-----------:|---------------:|-------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <a target="_blank" href="https://huggingface.co/stabilityai/japanese-stablelm-instruct-alpha-7b" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">stabilityai-japanese-stablelm-instruct-alpha-7b</a> |     54.71 |            82.22 |  52.05 |     82.88 |    63.26 |       74.83 |       7.79 |          72.68 |    2   | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/stabilityai/stabilityai-japanese-stablelm-instruct-alpha-7b/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/stabilityai/stabilityai-japanese-stablelm-instruct-alpha-7b/harness.sh</a> |
| <a target="_blank" href="https://huggingface.co/stabilityai/japanese-stablelm-base-alpha-7b" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">stabilityai-japanese-stablelm-base-alpha-7b</a>         |     51.06 |            33.42 |  43.34 |     96.73 |    70.62 |       78.09 |      10.65 |          72.78 |    2.8 | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/stabilityai/stabilityai-japanese-stablelm-base-alpha-7b/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/stabilityai/stabilityai-japanese-stablelm-base-alpha-7b/harness.sh</a>         |
| <a target="_blank" href="https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-sft" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">rinna-bilingual-gpt-neox-4b-instruction-sft</a>         |     47.75 |            49.51 |  47.08 |     95.28 |    55.99 |       61.17 |       5.51 |          64.65 |    2.8 | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/rinna/rinna-bilingual-gpt-neox-4b-instruction-sft/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/rinna/rinna-bilingual-gpt-neox-4b-instruction-sft/harness.sh</a>                     |
| <a target="_blank" href="https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-ppo" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">rinna-bilingual-gpt-neox-4b-instruction-ppo</a>         |     47.18 |            48.79 |  48.23 |     96.09 |    54.16 |       57.65 |       5.03 |          65.07 |    2.4 | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/rinna/rinna-bilingual-gpt-neox-4b-instruction-ppo/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/rinna/rinna-bilingual-gpt-neox-4b-instruction-ppo/harness.sh</a>                     |
| <a target="_blank" href="https://huggingface.co/llama2/13b-chat" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">llama2-13b-chat</a>                                                                 |     47.02 |            72.56 |  35.62 |     59.92 |    67.69 |       48.2  |      15.14 |          63.82 |   13.2 | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/llama2/llama2-13b-chat/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/llama2/llama2-13b-chat/harness.sh</a>                                                                           |
| <a target="_blank" href="https://huggingface.co/llama2/13b" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">llama2-13b</a>                                                                           |     46.32 |            74.89 |  21.98 |     38.89 |    76.14 |       67.7  |      18.11 |          62.88 |   10   | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/llama2/llama2-13b/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/llama2/llama2-13b/harness.sh</a>                                                                                     |
| <a target="_blank" href="https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">rinna-japanese-gpt-neox-3.6b-instruction-ppo</a>       |     46.32 |            44.06 |  54.19 |     89.61 |    51.62 |       50.95 |       6.63 |          69.13 |    4.4 | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-ppo/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-ppo/harness.sh</a>                   |
| <a target="_blank" href="https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">rinna-japanese-gpt-neox-3.6b-instruction-sft-v2</a> |     45.23 |            40.57 |  53.45 |     89.88 |    44.91 |       52.84 |       6.14 |          71.22 |    2.8 | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-sft-v2/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-sft-v2/harness.sh</a>             |
| <a target="_blank" href="https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">rinna-japanese-gpt-neox-3.6b-instruction-sft</a>       |     43.82 |            38.07 |  44.58 |     90.62 |    47.41 |       53.69 |       4.74 |          69.45 |    2   | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-sft/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-sft/harness.sh</a>                   |
| <a target="_blank" href="https://huggingface.co/llama2/7b" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">llama2-7b</a>                                                                             |     42.96 |            52.64 |  28.23 |     86.05 |    58.4  |       38.83 |       9.32 |          64.65 |    5.6 | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/llama2/llama2-7b/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/llama2/llama2-7b/harness.sh</a>                                                                                       |
| <a target="_blank" href="https://huggingface.co/rinna/japanese-gpt-neox-3.6b" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">rinna-japanese-gpt-neox-3.6b</a>                                       |     41.79 |            31.64 |  34.43 |     74.82 |    47.91 |       68.38 |       5.16 |          70.8  |    1.2 | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/rinna/rinna-japanese-gpt-neox-3.6b/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/rinna/rinna-japanese-gpt-neox-3.6b/harness.sh</a>                                                   |
| <a target="_blank" href="https://huggingface.co/llama2/7b-chat" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">llama2-7b-chat</a>                                                                   |     41.31 |            55.59 |  29.54 |     90.41 |    59.34 |       17.96 |       2.34 |          66.11 |    9.2 | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/llama2/llama2-7b-chat/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/llama2/llama2-7b-chat/harness.sh</a>                                                                             |
| <a target="_blank" href="https://huggingface.co/rinna/bilingual-gpt-neox-4b" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">rinna-bilingual-gpt-neox-4b</a>                                         |     40.03 |            20.82 |  55.22 |     59.55 |    50.79 |       59.45 |       5.55 |          66.42 |    2.4 | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/rinna/rinna-bilingual-gpt-neox-4b/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/rinna/rinna-bilingual-gpt-neox-4b/harness.sh</a>                                                     |
| <a target="_blank" href="https://huggingface.co/cyberagent/open-calm-7b" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">cyberagent-open-calm-7b</a>                                                 |     38.8  |            24.22 |  37.63 |     74.12 |    45.79 |       60.74 |       2.04 |          65.07 |    0.8 | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/cyberagent/cyberagent-open-calm-7b/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/cyberagent/cyberagent-open-calm-7b/harness.sh</a>                                                   |
| <a target="_blank" href="https://huggingface.co/cyberagent/open-calm-3b" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">cyberagent-open-calm-3b</a>                                                 |     38.61 |            27.79 |  40.35 |     86.21 |    40.45 |       46.91 |       1.95 |          63.61 |    1.6 | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/cyberagent/cyberagent-open-calm-3b/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/cyberagent/cyberagent-open-calm-3b/harness.sh</a>                                                   |
| <a target="_blank" href="https://huggingface.co/rinna/japanese-gpt-1b" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">rinna-japanese-gpt-1b</a>                                                     |     36.92 |            34.76 |  37.67 |     87.86 |    26.18 |       37.03 |       5.34 |          64.55 |    2   | <a target="_blank" href="https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/models/rinna/rinna-japanese-gpt-1b/harness.sh" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">models/rinna/rinna-japanese-gpt-1b/harness.sh</a>                                                                 |

## How to evaluate your model

1. git clone https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable
    ```bash
    git clone -b jp-stable https://github.com/Stability-AI/lm-evaluation-harness.git
    cd lm-evaluation-harness
    pip install -e ".[ja]"
    ```
2. Choose your prompt template based on [docs/prompt_templates.md]((https://github.com/Stability-AI/lm-evaluation-harness/blob/jp-stable/docs/prompt_templates.md))
3. Replace `TEMPLATE` to the version and change `MODEL_PATH` . And, save the script as `harness.sh`

    ```bash
    MODEL_ARGS="pretrained=MODEL_PATH"
    TASK="jsquad-1.1-TEMPLATE,jcommonsenseqa-1.1-TEMPLATE,jnli-1.1-TEMPLATE,marc_ja-1.1-TEMPLATE"
    python main.py \
        --model hf-causal \
        --model_args $MODEL_ARGS \
        --tasks $TASK \
        --num_fewshot "2,3,3,3" \
        --device "cuda" \
        --output_path "result.json"
    ```

4. Run! 
   ```bash
   sh harness.sh
   ```

We evaluated some open-sourced Japanese LMs. Pleasae refer to `harness.sh` inside `models` folder. 


## JP Tasks
For more details, please see [docs/jptasks.md](https://github.com/Stability-AI/lm-evaluation-harness/blob/jp-stable/docs/jptasks.md).

| Tasks | [Supported Prompt Templates](https://github.com/Stability-AI/lm-evaluation-harness/blob/jp-stable/docs/prompt_templates.md) |
| :- | -: | 
| JSQuAD | 0.1 / 0.2 / 0.3 / 0.4 |
| JCommonsenseQA |  0.1 / 0.2 / 0.3 / 0.4 |
| JNLI | 0.2 / 0.3 / 0.4 |
| MARC-ja | 0.2 / 0.3 / 0.4 |
| JaQuAD | 0.1 / 0.2 / 0.3 / 0.4 |
| JBLiMP | - |
| XLSum-ja | 0.0 / 0.3 / 0.4 |
| JAQKET | 0.1 / 0.2 / 0.3 / 0.4 |

-----------------
# Language Model Evaluation Harness

![](https://github.com/EleutherAI/lm-evaluation-harness/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/EleutherAI/lm-evaluation-harness/branch/master/graph/badge.svg?token=JSG3O2427J)](https://codecov.io/gh/EleutherAI/lm-evaluation-harness)

## Overview

This project provides a unified framework to test generative language models on a large number of different evaluation tasks.

Features:

- 200+ tasks implemented. See the [task-table](./docs/task_table.md) for a complete list.
- Support for models loaded via [transformers](https://github.com/huggingface/transformers/) (including quantization via [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), and [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/), with a flexible tokenization-agnostic interface.

- Support for evaluation on adapters (e.g. LoRa) supported in [Hugging Face's PEFT library](https://github.com/huggingface/peft).
- Task versioning to ensure reproducibility.

## Install

To install `lm-eval` from the github repository main branch, run:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

To install additional multilingual tokenization and text segmentation packages, you must install the package with the `multilingual` extra:

```bash
pip install -e ".[multilingual]"
```

To support loading GPTQ quantized models, install the package with the `auto-gptq` extra:

```bash
pip install -e ".[auto-gptq]"
```

## Basic Usage

> **Note**: When reporting results from eval harness, please include the task versions (shown in `results["versions"]`) for reproducibility. This allows bug fixes to tasks while also ensuring that previously reported scores are reproducible. See the [Task Versioning](#task-versioning) section for more info.

To evaluate a model hosted on the [Hugging Face Hub](https://huggingface.co/models) (e.g. GPT-J-6B) on tasks with names matching the pattern `lambada_*` and `hellaswag` you can use the following command:


```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks lambada_*,hellaswag \
    --device cuda:0
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most notably, this supports the common practice of using the `revisions` feature on the Hub to store partially trained checkpoints:

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000 \
    --tasks lambada_openai,hellaswag \
    --device cuda:0
```

To evaluate models that are loaded via `AutoSeq2SeqLM` in Hugging Face, you instead use `hf-seq2seq`. *To evaluate (causal) models across multiple GPUs, use `--model hf-causal-experimental`*

> **Warning**: Choosing the wrong model may result in erroneous outputs despite not erroring.

To use with [PEFT](https://github.com/huggingface/peft), take the call you would run to evaluate the base model and add `,peft=PATH` to the `model_args` argument as shown below:
```bash
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=EleutherAI/gpt-j-6b,peft=nomic-ai/gpt4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0
```

Our library also supports the OpenAI API:

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag
```

While this functionality is only officially maintained for the official OpenAI API, it tends to also work for other hosting services that use the same API such as [goose.ai](goose.ai) with minor modification. We also have an implementation for the [TextSynth](https://textsynth.com/index.html) API, using `--model textsynth`.

To verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag \
    --check_integrity
```

To evaluate mesh-transformer-jax models that are not available on HF, please invoke eval harness through [this script](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py).

💡 **Tip**: You can inspect what the LM inputs look like by running the following command:

```bash
python write_out.py \
    --tasks all_tasks \
    --num_fewshot 5 \
    --num_examples 10 \
    --output_base_path /path/to/output/folder
```

This will write out one text file for each task.

## Advanced Usage

For models loaded with the HuggingFace  `transformers` library, any arguments provided via `--model_args` get passed to the relevant constructor directly. This means that anything you can do with `AutoModel` can be done with our library. For example, you can pass a local path via `pretrained=` or use models finetuned with [PEFT](https://github.com/huggingface/peft) by taking the call you would run to evaluate the base model and add `,peft=PATH` to the `model_args` argument:
```bash
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=EleutherAI/gpt-j-6b,peft=nomic-ai/gpt4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0
```

GPTQ quantized models can be loaded by specifying their file names in `,quantized=NAME` (or `,quantized=True` for default names) in the `model_args` argument:

```bash
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=model-name-or-path,quantized=model.safetensors,gptq_use_triton=True \
    --tasks hellaswag
```

We support wildcards in task names, for example you can run all of the machine-translated lambada tasks via `--task lambada_openai_mt_*`.

We currently only support one prompt per task, which we strive to make the "standard" as defined by the benchmark's authors. If you would like to study how varying prompts causes changes in the evaluation score, check out the [BigScience fork](https://github.com/bigscience-workshop/lm-evaluation-harness) of this repo. We are currently working on upstreaming this capability to `main`.

## Implementing new tasks

To implement a new task in the eval harness, see [this guide](./docs/task_guide.md).

## Task Versioning

To help improve reproducibility, all tasks have a `VERSION` field. When run from the command line, this is reported in a column in the table, or in the "version" field in the evaluator return dict. The purpose of the version is so that if the task definition changes (i.e to fix a bug), then we can know exactly which metrics were computed using the old buggy implementation to avoid unfair comparisons. To enforce this, there are unit tests that make sure the behavior of all tests remains the same as when they were first implemented. Task versions start at 0, and each time a breaking change is made, the version is incremented by one.

When reporting eval harness results, please also report the version of each task. This can be done either with a separate column in the table, or by reporting the task name with the version appended as such: taskname-v0.

## Test Set Decontamination

To address concerns about train / test contamination, we provide utilities for comparing results on a benchmark using only the data points nto found in the model training set. Unfortunately, outside of models trained on the Pile and C4, its very rare that people who train models disclose the contents of the training data. However this utility can be useful to evaluate models you have trained on private data, provided you are willing to pre-compute the necessary indices. We provide computed indices for 13-gram exact match deduplication against the Pile, and plan to add additional precomputed dataset indices in the future (including C4 and min-hash LSH deduplication).

For details on text decontamination, see the [decontamination guide](./docs/decontamination.md).

Note that the directory provided to the `--decontamination_ngrams_path` argument should contain the ngram files and info.json. See the above guide for ngram generation for the pile, this could be adapted for other training sets.

```bash
python main.py \
    --model gpt2 \
    --tasks sciq \
    --decontamination_ngrams_path path/containing/training/set/ngrams \
    --device cuda:0
```

## Cite as

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
