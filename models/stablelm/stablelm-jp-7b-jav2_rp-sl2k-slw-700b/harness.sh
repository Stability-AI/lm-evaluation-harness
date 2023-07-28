#!/bin/bash

PROJECT_DIR=""
PRETRAINED="${PROJECT_DIR}/hf_model/7b-jav2_rp-sl2k-slw_fixed-conversion_w-codes,trust_remote_code=True/"
TOKENIZER="${PROJECT_DIR}/tokenizers/nai-hf-tokenizer/,use_fast=False"
MODEL_ARGS="pretrained=${PRETRAINED},tokenizer=${TOKENIZER}"
TASK="jcommonsenseqa-1.1-0.2,jnli-1.1-0.2,marc_ja-1.1-0.2,jsquad-1.1-0.2,jaqket_v2-0.1-0.2,xlsum_ja,xwinograd_ja,mgsm"
NUM_FEWSHOT="3,3,3,2,1,1,0,5"
OUTPUT_PATH="models/stablelm/stablelm-jp-7b-jav2_rp-sl2k-slw-700b/result.json"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEWSHOT \
    --device "cuda" \
    --no_cache \
    --output_path $OUTPUT_PATH