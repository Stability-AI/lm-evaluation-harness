#!/bin/bash

MODEL_NAME="weblab-10b"
MODEL_ARGS="pretrained=matsuo-lab/${MODEL_NAME},torch_dtype=auto"
TASK="jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3,jsquad-1.1-0.3,jaqket_v2-0.2-0.3,xlsum_ja-1.0-0.3,xwinograd_ja,mgsm-1.0-0.3"
NUM_FEWSHOT="3,3,3,2,1,1,0,5"
OUTPUT_PATH="models/matsuo-lab/${MODEL_NAME}/result.json"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEWSHOT \
    --device "cuda" \
    --output_path $OUTPUT_PATH
