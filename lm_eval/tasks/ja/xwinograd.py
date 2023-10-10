"""
It’s All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning
https://aclanthology.org/2021.findings-acl.310/

xwinograd is a collection of Winograd schema coreference and commonsense reasoning problems in multiple languages.
"""

# XXX: This dataset is multilingual, but was added specifically for Japanese eval.
# If there's interest it could easily be used in other scenarios.

from lm_eval.base import rf, Task
from lm_eval.metrics import mean
import numpy as np

_CITATION = """
@misc{tikhonov2021heads,
    title={It's All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning},
    author={Alexey Tikhonov and Max Ryabinin},
    year={2021},
    eprint={2106.12066},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""  # noqa: W605


class XWinograd(Task):
    VERSION = 1.0
    DATASET_PATH = "polm-stability/xwinograd-ja"

    # data samples have sentence1, sentence2, and answer keys.
    # answer is 1 or 2 (as strings).

    # docs are not split, everything is in "test", so treat it as val.

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["test"]

    def construct_requests(self, doc, ctx):
        assert not ctx

        return [
            rf.loglikelihood("", doc["sentence1"]),
            rf.loglikelihood("", doc["sentence2"]),
        ]

    def doc_to_text(self, doc):
        return ""

    def doc_to_target(self, doc):
        ans = doc["answer"]
        return doc[f"sentence{ans}"]

    def process_results(self, doc, results):
        li1, li2 = results

        goal = int(doc["answer"])
        if goal == 1 and li1 > li2:
            acc = 1.0
        elif goal == 2 and li2 > li1:
            acc = 1.0
        else:
            acc = 0.0

        return {
            "acc": acc,
        }

    def higher_is_better(self):
        return {
            "acc": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
        }


class XWinogradJA(XWinograd):
    DATASET_NAME = "jp"
