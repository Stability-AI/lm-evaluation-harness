MODEL_ARGS="pretrained=rinna/japanese-gpt-1b,use_fast=False"
TASK="jsquad-1.2-0.2"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "2" --device "cuda" --output_path "models/rinna/rinna-japanese-gpt-1b/result.jsquad-1.2.json"
