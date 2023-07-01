#!/bin/bash
# Run a single eval task for a given model. Should be called from srun.
# NOTE: requires a venv to be prepared.
set -eou pipefail

TASK=$1         # ex: xwinograd_ja
FEWSHOT=$2      # ex: 0
MODEL_ARGS="$3" # ex: pretrained=abeja/gpt-neox-japanese-2.7b
# Note that model path is often slightly different from hf name
MODEL_PATH=$4   # ex: abeja/gpt-neox-japanese-2.7b

# assuems this is in the scripts/ dir of the harness repo
cd $(dirname "$0")/..
source env/bin/activate
python main.py \
	--model hf-causal \
	--model_args $MODEL_ARGS \
	--tasks $TASK \
	--num_fewshot "$FEWSHOT" \
	--device "cuda" \
	--output_path "models/$MODEL_PATH/result.$TASK.json"
