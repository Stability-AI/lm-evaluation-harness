MODEL_ARGS="pretrained=rinna/bilingual-gpt-neox-4b,use_fast=False,device_map=auto,torch_dtype=auto"
TASK="jsquad-1.2-0.2"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "2" --device "cuda" --output_path "models/rinna/rinna-bilingual-gpt-neox-4b/result.jsquad-1.2.json"
