MODEL_ARGS="pretrained=huggyllama/llama-7b,use_accelerate=True,load_in_8bit=True"
TASK="jsquad-1.2-0.3"
python main.py --model hf-causal-experimental --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "2" --device "cuda" --output_path "models/llama/llama-7b/result.jsquad-1.2.json" --batch_size 2
