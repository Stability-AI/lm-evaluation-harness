"""
Tabulates MMMLU (Hendrycks Test) average accuracy from a given results dir.
"""

import argparse
import os
import json
import numpy as np
from pytablewriter import MarkdownTableWriter


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir", help="Results file with MMMLU (Hendrycks Test) results", default="results"
)
args = parser.parse_args()


subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}
# Categories -> Subcategories
categories2subcategories = {}
for key, value in subcategories.items():
    for v in value:
        if v not in categories2subcategories:
            categories2subcategories[v] = []
        categories2subcategories[v].append(key)
print(categories2subcategories)

categories = {
    "STEM": [
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "math",
        "engineering",
    ],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}


def read_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def main(args):
    file_paths = [fp for fp in os.listdir(args.dir) if fp.endswith(".json")]
    print(file_paths)

    categories = sorted([c for c in list(categories2subcategories.keys())])
    headers = ["Model", "Average", *[c.title() for c in categories]]

    rows = []
    for results_file in file_paths:
        row = []
        data = json.load(open(os.path.join(args.dir, results_file)))

        # Add model name to first column of row
        model_name = [
            k for k in data["config"]["model_args"].split(",") if "pretrained" in k
        ][0].split("=")[1]
        row.append(model_name)

        # Handle results data
        results = data["results"]
        mmlu_results = {
            r: v for r, v in results.items() if r.startswith("hendrycksTest-")
        }

        # Bin the category accuracies for this model
        category_acc = {c: [] for c in categories}
        print(category_acc)
        for subcategory, category_metrics in mmlu_results.items():
            # Grab the category by its subcategory name
            subcategory_name = subcategory.split("-")[1]
            category_name = subcategories[subcategory_name][0]
            # Collect accuracy
            category_acc[category_name].append(category_metrics["acc"])

        # Compute total accuracy across all categories and add to row
        total_acc = []
        for c, acc in category_acc.items():
            total_acc.extend(acc)
        avg_acc = np.mean(total_acc)
        row.append(f"{avg_acc * 100:.2f}")

        # Compute average accuracy per category and add to row
        for c, acc in category_acc.items():
            print(c, acc)
            avg_category_acc = np.mean(acc)
            row.append(f"{avg_category_acc * 100:.2f}")

        rows.append(row)

    # Sort rows by the average accuracy
    rows = sorted(rows, key=lambda x: float(x[1]), reverse=True)

    # Print table in markdown
    writer = MarkdownTableWriter(
        table_name="MMLU (Hendrycks Test)",
        headers=headers,
        value_matrix=rows,
        margin=1,
        flavor="github",
    )
    writer.dump("mmlu_results.md")


if __name__ == "__main__":
    main(args)
