MODEL_ARGS="pretrained=line-corporation/japanese-large-lm-3.6b-instruction-sft,use_fast=False,device_map=auto,torch_dtype=auto"
TASK="jcommonsenseqa-1.1-0.5,jnli-1.1-0.5,marc_ja-1.1-0.5,jsquad-1.1-0.5,jaqket_v2-0.2-0.5,xlsum_ja-1.0-0.5,xwinograd_ja,mgsm-1.0-0.5"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "3,3,3,2,1,1,0,5" --device "cuda" --output_path "models/line-corporation/line-corporation-japanese-large-lm-3.6b-instruction-sft/result.json"
