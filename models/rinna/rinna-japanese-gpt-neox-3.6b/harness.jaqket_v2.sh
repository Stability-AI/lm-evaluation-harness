MODEL_ARGS="pretrained=rinna/japanese-gpt-neox-3.6b,use_fast=False,torch_dtype=auto,device_map=auto"
TASK="jaqket_v2-0.2-0.2"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "1" --device "cuda" --output_path "models/rinna/rinna-japanese-gpt-neox-3.6b/result.jaqket_v2.json"
