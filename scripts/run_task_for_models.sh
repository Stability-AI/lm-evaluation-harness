#!/bin/bash
# Given a task, run it on all relevant models and update their results.
# See run_task_batch.sh for project, job name, and other batch settings.
set -eou pipefail

task=$1
fewshot=$2
# cd to script dir and then go up one so all paths work
cd $(dirname -- "$0")/..

# models.txt is a space-separated file. To generate it, use the code in this
# function from the project root, then edit the file to remove models you don't
# want to use (like community). 
function generate_models_txt() {
   find models/ -name harness.sh \
	   | xargs grep MODEL_ARGS= \
	   | sed -e 's:^models/::' -e 's:/harness.sh.MODEL_ARGS=: :' -e 's:"::g' \
	   > scripts/models.txt
}

cat scripts/models.txt | while read model_path args; do
  # The echo is just for debugging
  echo sbatch scripts/run_task_batch.sh $task $fewshot $args $model_path
  sbatch scripts/run_task_batch.sh $task $fewshot $args $model_path
done

# after the batches have finished, use the following command to update results.json
#   python scripts/merge_json.py

