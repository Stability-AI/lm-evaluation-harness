MODEL_ARGS="pretrained=cyberagent/open-calm-medium,use_fast=True"
TASK="jaqket_v1-0.1-0.2"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "1" --device "cuda" --output_path "models/cyberagent/cyberagent-open-calm-medium/result.jaqket_v1.json"