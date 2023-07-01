#!/bin/bash
#SBATCH --account="stablegpt"
#SBATCH --job-name="jp-eval"
#SBATCH --partition=g40
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=11G
#SBATCH --output=./slurm_outs/%x_%j.out
#SBATCH --error=./slurm_outs/%x_%j.err

# This command just holds the sbatch config and should be called from
# run_task_for_models.sh.
set -eou pipefail

# would be better if this wasn't relative to a home dir, but slurm runs a copy
# of this script instead of the original, and finding the original path can get
# a little complicated, so keeping it simple for now. This could be improved.
cd ~/lm-evaluation-harness/scripts
srun run_task.sh "$@"
