MODEL_ARGS="pretrained=cyberagent/open-calm-large,use_fast=True"
TASK="jaqket_v2-0.2-0.2,jcommonsenseqa-1.1-0.2,jnli-1.1-0.2,marc_ja-1.1-0.2,jsquad-1.1-0.2,xlsum_ja"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "1,2,3,3,3,1" --device "cuda" --output_path "models/cyberagent-open-calm-large/result.json"