MODEL_ARGS="pretrained=meta-llama/Llama-2-7b-hf,use_accelerate=True,dtype=auto"
TASK="jsquad-1.2-0.3"
python main.py --model hf-causal-experimental --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "2" --device "cuda" --output_path "models/llama2/llama2-7b/result.jsquad-1.2.json" --batch_size 2
