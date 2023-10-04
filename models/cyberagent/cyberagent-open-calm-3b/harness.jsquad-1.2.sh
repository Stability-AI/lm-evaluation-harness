MODEL_ARGS="pretrained=cyberagent/open-calm-3b,device_map=auto,torch_dtype=auto"
TASK="jsquad-1.2-0.2"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "2" --device "cuda" --output_path "models/cyberagent/cyberagent-open-calm-3b/result.jsquad-1.2.json"
