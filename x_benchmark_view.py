"""
Usage:
python multilingual_benchmark_view.py --dir "results/"
"""

import argparse
import json
import os

import numpy as np
from pytablewriter import MarkdownTableWriter


# Creates markdown table for the given directory of lm-eval results


parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="directory to list", default="./results_x")
args = parser.parse_args()


def read_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def main(args):
    file_paths = [fp for fp in os.listdir(args.dir) if fp.endswith(".json")]
    print(file_paths)

    task2name = {
        "xcopa_": "xCOPA",
        "xstory_cloze_": "xStoryCloze",
        "xwinograd_": "xWinograd",
        "lambada_openai_mt_deepl_": "LAMBADA OpenAI (DeepL)",
    }

    lambada = "lambada_openai_mt_deepl_"
    xstory = "xstory_cloze_"
    xwinograd = "xwinograd_"
    xcopa = "xcopa_"

    task_headers = []
    for task in sorted(task2name.keys()):
        task_headers.append(f"{task2name[task]}")
    headers = ["Model", "Average", *task_headers]
    rows = []
    for result in file_paths:
        data = read_json(os.path.join(args.dir, result))
        model_name = [
            k for k in data["config"]["model_args"].split(",") if "pretrained" in k
        ][0].split("=")[1]
        row = [model_name]

        # Compute mean accuracy
        # Copied from `multilingual_benchmark_view.py` - this is hacky and should be refactored TBD
        sum = 0
        for task in sorted(task2name.keys()):
            if task == lambada:
                i = []
                for key, item in data["results"].items():
                    if key.startswith(lambada):
                        i.append(item["acc"])
                sum += np.mean(i)
            elif task == xstory:
                i = []
                for key, item in data["results"].items():
                    if key.startswith(xstory):
                        i.append(item["acc"])
                sum += np.mean(i)
            elif task == xwinograd:
                i = []
                for key, item in data["results"].items():
                    if key.startswith(xwinograd):
                        i.append(item["acc"])
                sum += np.mean(i)
            elif task == xcopa:
                i = []
                for key, item in data["results"].items():
                    if key.startswith(xcopa):
                        i.append(item["acc"])
                sum += np.mean(i)
            else:
                raise ValueError(f"Unknown task: {task}")

        row.append(f"{(sum / len(task2name)) * 100.0:.2f}")

        for task in sorted(task2name.keys()):
            if task == lambada:
                i = []
                for key, item in data["results"].items():
                    if key.startswith(lambada):
                        i.append(item["acc"])
                score = f"{np.mean(i) * 100:.2f}"
            elif task == xstory:
                i = []
                for key, item in data["results"].items():
                    if key.startswith(xstory):
                        i.append(item["acc"])
                score = f"{np.mean(i) * 100:.2f}"
            elif task == xwinograd:
                i = []
                for key, item in data["results"].items():
                    if key.startswith(xwinograd):
                        i.append(item["acc"])
                score = f"{np.mean(i) * 100:.2f}"
            elif task == xcopa:
                i = []
                for key, item in data["results"].items():
                    if key.startswith(xcopa):
                        i.append(item["acc"])
                score = f"{np.mean(i) * 100:.2f}"

            row.append(score)

        rows.append(row)

    # Sort by average accuracy
    rows = sorted(rows, key=lambda x: float(x[1]), reverse=True)
    print(rows)

    # Print table in markdown
    writer = MarkdownTableWriter(
        table_name="0-shot Multilingual Results",
        headers=headers,
        value_matrix=rows,
        margin=1,
        flavor="github",
    )
    writer.dump("x_benchmark_results.md")
    with open("x_benchmark_results.md", "a") as f:
        f.write("\n\* : Byte-length Normalized Accuracy\n")


if __name__ == "__main__":
    main(args)
