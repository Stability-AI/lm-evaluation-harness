import os
import sys
import configparser
from pathlib import Path
import argparse

"""
A script to generate a harness shell script for running eval in a cluster.

Given a model directory, this script looks for a `harness.conf` file. The file describes the path to the pretrained model (or the HuggingFace model name), the model arguments, the tasks to run, and other details.

Example usage:

    python scripts/generate_harness.py models/stablelm/stablelm-jp-3b-ja50_rp50-700b/

Given these details the script outputs a command line which will run eval.

In detail, a config has two kinds of sections: the `[model]` section contains general model options, while task-specific sections have names like `[tasks.MY_TASK-0.1]`.

The model section contains the following values:

- model: almost always hf-causal
- path: the first arg to AutoModel.load_model
- tokenizer: path to the tokenizer
- args: any other model arguments

You can use interpolation in the config file, so it's commmon to include a `project_dir` value and use it like this:

    tokenizer = ${project_dir}/tokenizers/hogehoge

Task sections just contain the `fewshot` parameter for now.

Configs support inheritance. The file at `models/harness.conf` specifies global defaults, while files for each model group can define options not specific for a model. These files are usually the most detailed, since task and prompt selection are usually consistent for a model group.
"""

def generate_harness(path):

    # grandparent is global config, parent is org config.
    # it's ok if they don't exist.
    hc = "harness.conf"
    config_paths = [path.parent.parent / hc, path.parent / hc, path / hc]
    # used to make PROJECT_DIR work just like shell
    interp = configparser.ExtendedInterpolation()
    conf = configparser.ConfigParser(interpolation=interp)
    conf.read(config_paths)

    # Build the model args. We don't have to interpolate the path here, it can
    # be handled in the file directly thanks to interpolation. Also, fall back
    # to the last two parts of the directory path as the HF name.
    fallback_path = os.path.join(*path.parts[-2:])

    model_path = conf["model"].get("path", fallback_path)
    # tokenizer is technically not required, but almost always present
    tokenizer = conf["model"].get("tokenizer")
    # args are technically not required
    args = conf["model"].get("args")
    model_args = f"pretrained={model_path}"

    if tokenizer in ("''", '""'):
        tokenizer = None
    if tokenizer:
        model_args += "," + f"tokenizer={tokenizer}"

    if args in ("''", '""'):
        args = None
    if args:
        model_args += "," + args

    # Make the task list. Basically just attaching prompt versions, but some tasks
    # have no prompt versions because they don't use prompts.
    tasks = []
    fewshot = []
    for key, val in conf.items():
        if not key.startswith("tasks."):
            continue

        name = key.split(".", 1)[1]
        prompt = val["prompt"]
        # By default configparser doesn't handle empty strings
        # properly, so check for obvious attempts and fix them.
        if prompt in ("''", '""'):
            prompt = ""
        if prompt:
            name = f"{name}-{prompt}"
        tasks.append(name)
        fewshot.append(val["fewshot"])

    tasks = ",".join(tasks)
    fewshot = ",".join(fewshot)

    output_path = path / "result.json"
    model_type = conf["model"]["model"]
    script = (
      "python main.py "
      f"--device cuda "
      f"--model {model_type} "
      f"--model_args {model_args} "
      f"--tasks {tasks} "
      f"--num_fewshot {fewshot} "
      f"--output_path {output_path} "
      )
    return script.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog="generate_harness.py",
            description="Generate an eval command based on configs.",
            )

    parser.add_argument("model_path")
    parser.add_argument("-w", "--write", action="store_true",
            help="write harness script to default location")

    args = parser.parse_args()
    path = Path(args.model_path).absolute()
    if path.is_file():
        # if we specified a file for some reason, just take the dir
        path = path.parent

    cmd = generate_harness(path)
    print(cmd)
    if args.write:
        opath = path / "harness.sh"
        with open(opath, "w") as ofile:
            ofile.write(cmd)
        print(f"wrote script to: {opath}")


