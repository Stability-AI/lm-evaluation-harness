#!/bin/bash

PRETRAINED="DAMO-NLP-MT/polylm-13b"
TOKENIZER="DAMO-NLP-MT/polylm-13b,use_fast=False"
MODEL_ARGS="pretrained=${PRETRAINED},tokenizer=${TOKENIZER}"
TASK="jcommonsenseqa-1.1-0.2,jnli-1.1-0.2,marc_ja-1.1-0.2,jsquad-1.1-0.2,jaqket_v2-0.1-0.2,xlsum_ja,xwinograd_ja,mgsm"
NUM_FEWSHOT="3,3,3,2,1,1,0,5"
OUTPUT_PATH="models/DAMO-NLP-MT/polylm-13b/result.json"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEWSHOT \
    --device "cuda" \
    --no_cache \
    --output_path $OUTPUT_PATH