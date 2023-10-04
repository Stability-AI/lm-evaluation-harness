MODEL_ARGS="pretrained=cyberagent/open-calm-large,use_fast=True,device_map=auto,torch_dtype=auto"
TASK="jsquad-1.2-0.2"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "3" --device "cuda" --output_path "models/cyberagent-open-calm-large/result.jsquad-1.2.json"
