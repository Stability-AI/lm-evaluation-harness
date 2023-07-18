#!/bin/bash

PROJECT_DIR=""
PRETRAINED="${PROJECT_DIR}/sft/checkpoints/stablelm-jp-instruct-1b_1.5.2"
TOKENIZER="${PROJECT_DIR}/tokenizers/nai-hf-tokenizer/,use_fast=False"
MODEL_ARGS="pretrained=${PRETRAINED},tokenizer=${TOKENIZER}"
TASK="jsquad-1.1-0.3,jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3,xlsum_ja-1.0-0.3,jaqket_v2-0.1-0.3,xwinograd_ja,mgsm-1.0-0.3"
NUM_FEWSHOTS="2,3,3,3,1,1,0,4"
OUTPUT_PATH="models/stablelm/stablelm-jp-instruct-1b_1.5.2/result.json"
time python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEWSHOTS \
    --device "cuda" \
    --output_path $OUTPUT_PATH