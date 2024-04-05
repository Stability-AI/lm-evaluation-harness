"""
Usage: Run from project root directory with the following command:

```sh
python x_benchmark.py --model ""
```
"""
import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dtype", type=str, default="auto")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--use-fast", action="store_true")
parser.add_argument("--add-special-tokens", action="store_true")
args = parser.parse_args()


def main():
    results_file = os.path.join("results_x", f"{args.model.replace('/', '_')}.json")
    results_file = os.path.abspath(results_file)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    assert not os.path.exists(results_file), f"{results_file} results already exists! Check the `results` dir!"
    subprocess.run(
        [
            "python",
            "main.py",
            "--tasks",
            "xcopa_*,xwinograd_*,xstory_cloze_*,lambada_openai_mt_deepl_*",
            "--model",
            "hf-causal-experimental",
            "--model_args",
            f"pretrained={args.model},trust_remote_code=True,add_special_tokens={args.add_special_tokens},dtype={args.dtype}",
            "--batch_size",
            f"{args.batch_size}",
            "--output_path",
            results_file,
            "--device",
            args.device,
            "--num_fewshot",
            "0",
        ]
    )


if __name__ == "__main__":
    main()
