
# JP Tasks 

## [JGLUE](https://github.com/yahoojapan/JGLUE)
### JSQuAD
> JSQuAD is a Japanese version of [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Rajpurkar+, 2016), one of the datasets of reading comprehension.
Each instance in the dataset consists of a question regarding a given context (Wikipedia article) and its answer. JSQuAD is based on SQuAD 1.1 (there are no unanswerable questions). We used [the Japanese Wikipedia dump](https://dumps.wikimedia.org/jawiki/) as of 20211101.

**sample script**
```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "jsquad-1.1-0.2" \
    --num_fewshot "2" \
    --output_path "result.json"
```

### JCommonsenseQA
> JCommonsenseQA is a Japanese version of [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa) (Talmor+, 2019), which is a multiple-choice question answering dataset that requires commonsense reasoning ability. It is built using crowdsourcing with seeds extracted from the knowledge base [ConceptNet](https://conceptnet.io/).

**sample script**
```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "jcommonsenseqa-1.1-0.2" \
    --num_fewshot "3" \
    --output_path "result.json"
```

### JNLI
> JNLI is a Japanese version of the NLI (Natural Language Inference) dataset. NLI is a task to recognize the inference relation that a premise sentence has to a hypothesis sentence. The inference relations are `entailment`, `contradiction`, and `neutral`.

**sample script**
```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "jnli-1.1-0.2" \
    --num_fewshot "3" \
    --output_path "result.json"
```

### MARC-ja
> MARC-ja is a dataset of the text classification task. This dataset is based on the Japanese portion of [Multilingual Amazon Reviews Corpus (MARC)](https://docs.opendata.aws/amazon-reviews-ml/readme.html) (Keung+, 2020).

**sample script**
```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "marc_ja-1.1-0.2" \
    --num_fewshot "3" \
    --output_path "result.json"
```

## [JaQuAD](https://huggingface.co/datasets/SkelterLabsInc/JaQuAD)

> Japanese Question Answering Dataset (JaQuAD), released in 2022, is a human-annotated dataset created for Japanese Machine Reading Comprehension. JaQuAD is developed to provide a SQuAD-like QA dataset in Japanese. 

**sample script**
```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "jaquad-1.1-0.2" \
    --num_fewshot "2" \
    --output_path "result.json"
```

## [JBLiMP](https://github.com/osekilab/JBLiMP)

> JBLiMP is a novel dataset for targeted syntactic evaluations of language models in Japanese. JBLiMP consists of 331 minimal pairs, which are created based on acceptability judgments extracted from journal articles in theoretical linguistics. These minimal pairs are grouped into 11 categories, each covering a different linguistic phenomenon.

**NOTE:** JBLiMP is not used in official evaluations because it is too small compared to other datasets.

**sample script**
```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "jblimp" \
    --num_fewshot "0" \
    --output_path "result.json"
```

## [Wikilingua](https://github.com/esdurmus/Wikilingua)

Wikilingua is a summarization task using documents constructed from Wikihow articles. While the original dataset is multilingual, for now it just the Japanese version has been added.

**sample script**

```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "wikilingua_ja" \
    --num_fewshot "1" \
    --output_path "result.json"
```

## [XLSum-ja](https://huggingface.co/datasets/csebuetnlp/xlsum)
This is a filtered Japanese subset of [XLSum](https://huggingface.co/datasets/csebuetnlp/xlsum) based on ROUGE-2, where [PaLM 2](https://arxiv.org/abs/2305.10403) uses. 

**main features**
- Filtered data based on 15-gram overlap as PaLM 2 did.
  - link to dataset: https://huggingface.co/datasets/mkshing/xlsum_ja
  - link to script: https://gist.github.com/mkshing/d6371cbfdd50d4f352cee247fd4dd86a
- Compute ROUGE-2 based on Mecab Tokenizer

**sample scripts**

```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "xlsum_ja" \
    --num_fewshot "1" \
    --output_path "result.json"
```

* \* 1-shot setting In [PaLM 2](https://arxiv.org/abs/2305.10403)

## [XWinograd](https://huggingface.co/datasets/Muennighoff/xwinograd)

XWinograd is a set of Winograd Schema sentence pairs. For example:

- ボブはトムに尋ねた。トムはお金をいくらか貸してくれるかと。
- ボブはトムに尋ねた。ボブはお金をいくらか貸してくれるかと。

In this case the first sentence is correct, because it doesn't make sense for Bob to ask Tom how much money Bob himself will loan.

The task is for the model to assign the higher log likelihood to the reasonable sentence. Because of the way the task is defined, it's always zero-shot with no prompt.

While XWinograd is a multilingual task, this only uses the Japanese subset, which has 959 pairs.

**sample scripts**

```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks xwinograd_ja \
    --num_fewshot 0 \
    --output_path result.json
```

## [JAQKET v2](https://sites.google.com/view/project-aio/dataset#h.gv38mvfvqk77)

Japanese Questions on Knowledge of Entity (JAQKET) is a Japanese open-domain QA dataset where the answers are Wikipedia article titles. 
In version 2 (v2), the dataset is designed to be more similar to a typical quiz competition by excluding answer choices and not providing the answer list used in v1. 

**sample script**

```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "jaqket_v2" \
    --num_fewshot "1" \
    --output_path "result.json"
```

## [JAQKET v1](https://sites.google.com/view/project-aio/dataset#h.9lkgxzw759e3)

Japanese Questions on Knowledge of Entity (JAQKET) is a Japanese open-domain QA dataset where the answers are Wikipedia article titles. 
In version 1 (v1), Each question is accompanied by 20 candidate answers, meaning that the setup involves solving a multiple-choice question with 20 options to choose from.

**sample script**

```
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks "jaqket_v1" \
    --num_fewshot "1" \
    --output_path "result.json"
```

## [MGSM](https://huggingface.co/datasets/juletxara/mgsm)

[Multilingual Grade School Math](https://arxiv.org/pdf/2210.03057.pdf) is a set of 250 math word problems in Japanese, and the task is to get the right integer solution to the problem.
The dataset includes step-by-step solutions of example problems which are included in the prompt to induce Chain-of-Thought reasoning.

The task is rather hard, with gpt3/text-davinci-002 achieving 26% accuracy at 4-shot. 
Though this dataset was translated to Japanese by a human, some errors nonetheless occur including in the few-shot examples that are included in the prompt at every generation.

The implementation here features an auto-reduction of the number of examples included in the prompt if the prompt becomes larger than the model's context length.

Use mgsm-1.0-0.0 by default and mgsm-1.0-0.4 for the rinna instruction prompt.

**sample scripts**

Hugging Face models

```
python main.py \
    --model hf-causal
    --model_args $MODEL_ARGS 
    --tasks mgsm-1.0-0.4
    --num_fewshot "4" 
    --device "cuda" 
    --output_path "result.mgsm.json"
```
