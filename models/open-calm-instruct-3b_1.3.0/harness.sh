#!/bin/bash
set -eu

MODEL_ARGS="pretrained=/PATH/TO/open-calm-instruct-3b_1.3.0,tokenizer=cyberagent/open-calm-3b"
TASK="jsquad-1.1-0.3,jcommonsenseqa-1.1-0.3,jnli-1.1-0.3,marc_ja-1.1-0.3"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot "2,3,3,3" \
    --device "cuda" \
    --output_path "models/open-calm-instruct-3b_1.3.0/result.json"