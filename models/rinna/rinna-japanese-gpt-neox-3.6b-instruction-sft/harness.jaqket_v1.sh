MODEL_ARGS="pretrained=rinna/japanese-gpt-neox-3.6b-instruction-sft,use_fast=False,torch_dtype=auto,device_map=autp"
TASK="jaqket_v1-0.1-0.4"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "1" --device "cuda" --output_path "models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-sft/result.json"
