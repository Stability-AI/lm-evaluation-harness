""" Official evaluation script for v1.1 of the SQuAD dataset. """

import argparse
import json
import re
import string
import sys
from collections import Counter


def remove_punc(tokens):
    exclude = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    )
    exclude += string.punctuation
    exclude = [*exclude]
    return [tok for tok in tokens if tok not in exclude]


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import emoji
    import neologdn

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_emoji(text):
        text = "".join(["" if emoji.is_emoji(c) else c for c in text])
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r"", text)

    return white_space_fix((neologdn.normalize(remove_emoji(s))))


def f1_score(prediction, ground_truth):
    from fugashi import Tagger

    tagger = Tagger("-Owakati")
    prediction_tokens = remove_punc(tagger.parse(normalize_answer(prediction)).split())
    ground_truth_tokens = remove_punc(
        tagger.parse(normalize_answer(ground_truth)).split()
    )
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def contains_score(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = contains = total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    message = (
                        "Unanswered question " + qa["id"] + " will receive score 0."
                    )
                    print(message, file=sys.stderr)
                    continue
                ground_truths = [x["text"] for x in qa["answers"]]
                prediction = predictions[qa["id"]]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths
                )
                contains += metric_max_over_ground_truths(
                    contains_score, prediction, ground_truths
                )
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    contains = 100.0 * contains / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "contains": contains, "f1": f1}


if __name__ == "__main__":
    expected_version = "1.1"
    parser = argparse.ArgumentParser(
        description="Evaluation for Japanese SQuAD " + expected_version
    )
    parser.add_argument("dataset_file", help="Dataset file")
    parser.add_argument("prediction_file", help="Prediction File")
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json["version"] != expected_version:
            print(
                "Evaluation expects v-"
                + expected_version
                + ", but got dataset with v-"
                + dataset_json["version"],
                file=sys.stderr,
            )
        dataset = dataset_json["data"]
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))
