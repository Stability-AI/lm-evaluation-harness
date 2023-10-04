MODEL_ARGS="pretrained=abeja/gpt-neox-japanese-2.7b,device_map=auto,torch_dtype=auto"
TASK="jsquad-1.2-0.2"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "3" --device "cuda" --output_path "models/abeja-gpt-neox-japanese-2.7b/result.jsquad-1.2.json"
