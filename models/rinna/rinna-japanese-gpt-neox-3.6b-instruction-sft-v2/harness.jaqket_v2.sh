MODEL_ARGS="pretrained=rinna/japanese-gpt-neox-3.6b-instruction-sft-v2,use_fast=False,torch_dtype=auto,device_map=auto"
TASK="jaqket_v2-0.2-0.4"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "1" --device "cuda" --output_path "models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-sft-v2/result.jaqket_v2.json"
