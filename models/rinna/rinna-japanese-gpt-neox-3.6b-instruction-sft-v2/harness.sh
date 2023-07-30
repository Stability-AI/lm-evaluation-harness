MODEL_ARGS="pretrained=rinna/japanese-gpt-neox-3.6b-instruction-sft-v2,use_fast=False"
TASK="jcommonsenseqa-1.1-0.4,jnli-1.1-0.4,marc_ja-1.1-0.4,jsquad-1.1-0.4,jaqket_v2-0.2-0.4,xlsum_ja-1.0-0.4,xwinograd_ja,mgsm-1.0-0.4"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "3,3,3,2,1,1,0,5" --device "cuda" --output_path "models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-sft-v2/result.json"
