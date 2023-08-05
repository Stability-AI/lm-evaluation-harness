MODEL_ARGS="pretrained=abeja/gpt-neox-japanese-2.7b,torch_dtype=auto,device_map=auto"
TASK="jaqket_v1-0.1-0.2"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "1" --device "cuda" --output_path "models/abeja-gpt-neox-japanese-2.7b/result.jaqket_v1.json"