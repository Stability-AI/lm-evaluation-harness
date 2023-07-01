from glob import glob
from pathlib import Path
import json
import sys

# Find task-specific json files and add them to result json files. 

# task name, ex: xwinograd_ja
try:
    task = sys.argv[1]
except IndexError:
    print("Give task name as first argument, like: xwinograd_ja")
    sys.exit(1)

# task-specific result files have names like result.TASK.json
task_results = glob(f"models/**/result.{task}.json", recursive=True)

# given a task-specific file, the result.json file always exists
for tres in task_results:
    tres = Path(tres)
    res = tres.parent / "result.json"

    with open(res) as resfile:
        res_data = json.loads(resfile.read())

    with open(tres) as resfile:
        tres_data = json.loads(resfile.read())

    if task in res_data["results"]:
        # Ideally we would overwrite these, but it can be tricky to get the few
        # shot order correct, so adding that later.
        # TODO overwrite
        print(f"Not updating {tres.parent.name} because results already present")
        continue

    # update the relevant keys
    for key in ("results", "versions"):
        res_data[key][task] = tres_data[key][task]

    # because the result is new, fewshot goes at the end
    # for a single task, fewshow is a scalar and not an array
    # XXX is the type change a bug?
    tres_fewshot = tres_data["config"]["num_fewshot"]
    res_data["config"]["num_fewshot"].append(tres_fewshot)

    with open(res, "w") as resfile:
        out = json.dumps(res_data, indent=2)
        resfile.write(out)
