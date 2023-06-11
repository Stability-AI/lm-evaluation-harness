#!/bin/bash
set -eu

MODEL_ARGS="pretrained=/PATH/TO/stablelm-jp-instruct-3b_1.3.0/,tokenizer=/PATH/TO/nai-hf-tokenizer/,use_fast=False"
TASK="jsquad-1.1-0.3,jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3"
NUM_FEW_SHOTS="2,3,3,3"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEW_SHOTS \
    --device "cuda" \
    --output_path "models/stablelm-jp-instruct-3b_1.3.0/result.json"