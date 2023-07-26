MODEL_ARGS="pretrained=cyberagent/open-calm-3b"
TASK="jcommonsenseqa-1.1-0.2,jnli-1.1-0.2,marc_ja-1.1-0.2,jsquad-1.1-0.2,xlsum_ja,jaqket_v2-0.1-0.2,xwinograd_ja,mgsm"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "2,3,3,3,1,1,0,4" --device "cuda" --output_path "models/cyberagent-open-calm-3b/result.json"
