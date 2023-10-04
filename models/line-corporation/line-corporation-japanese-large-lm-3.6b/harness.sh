MODEL_ARGS="pretrained=line-corporation/japanese-large-lm-3.6b,use_fast=False,device_map=auto,torch_dtype=auto"
TASK="jcommonsenseqa-1.1-0.2,jnli-1.1-0.2,marc_ja-1.1-0.2,jsquad-1.1-0.2,jaqket_v2-0.2-0.2,xlsum_ja,xwinograd_ja,mgsm"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "3,3,3,2,1,1,0,5" --device "cuda" --output_path "models/line-corporation/line-corporation-japanese-large-lm-3.6b/result.json"
