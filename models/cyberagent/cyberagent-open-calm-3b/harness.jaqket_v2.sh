MODEL_ARGS="pretrained=cyberagent/open-calm-3b,torch_dtype=auto,device_map=auto"
TASK="jaqket_v2-0.2-0.2"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "1" --device "cuda" --output_path "models/cyberagent/cyberagent-open-calm-3b/result.jaqket_v2.json"
