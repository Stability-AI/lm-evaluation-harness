#!/bin/bash
set -eu

MODEL_ARGS="pretrained=/PATH/TO/rinna-instruct-1b_0.1.0/,tokenizer=rinna/japanese-gpt-1b,use_fast=False"
TASK="jsquad-1.1-0.3,jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot "2,3,3,3" \
    --device "cuda" \
    --output_path "models/rinna-instruct-1b_0.1.0/result.json"