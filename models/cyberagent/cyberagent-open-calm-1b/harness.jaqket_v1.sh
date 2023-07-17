MODEL_ARGS="pretrained=cyberagent/open-calm-1b"
TASK="jaqket_v1-0.1-0.2"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "1" --device "cuda" --output_path "models/cyberagent/cyberagent-open-calm-1b/result.jaqket_v1.json"