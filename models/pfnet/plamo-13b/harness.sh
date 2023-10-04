#!/bin/bash
set -eu
MODEL_ARGS="pretrained=pfnet/plamo-13b,use_fast=False,trust_remote_code=True,device_map=auto"
TASK="jcommonsenseqa-1.1-0.2,jnli-1.1-0.2,marc_ja-1.1-0.2,jsquad-1.1-0.2,jaqket_v2-0.2-0.2,xlsum_ja,xwinograd_ja,mgsm"
NUM_FEW_SHOTS="3,3,3,2,1,1,0,5"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEW_SHOTS \
    --device "cuda" \
    --output_path "models/pfnet/plamo-13b/result.json"
