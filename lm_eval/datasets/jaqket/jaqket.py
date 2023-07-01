import json
from typing import Dict, List, Optional, Union

import datasets as ds
import pandas as pd

_CITATION = """
@InProceedings{Kurihara_nlp2020,
  author =  "鈴木正敏 and 鈴木潤 and 松田耕史 and ⻄田京介 and 井之上直也",
  title =   "JAQKET: クイズを題材にした日本語 QA データセットの構築",
  booktitle =   "言語処理学会第26回年次大会",
  year =    "2020",
  url = "https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf"
  note= "in Japanese"
"""

_DESCRIPTION = """\
JAQKET: JApanese Questions on Knowledge of EnTitie
"""

_HOMEPAGE = "https://sites.google.com/view/project-aio/dataset"

_LICENSE = """\
This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
"""

_DESCRIPTION_CONFIGS = {
    "v1.0": "v1.0",
    "v2.0": "v2.0",
}

_URLS = {
    "v1.0": {
        "train": "https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_01/train_questions.json",
        "valid": "https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_01/dev1_questions.json",
        "candidate_entities": "https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_01/candidate_entities.json.gz",
    },
    "v2.0": {
        "train": "https://huggingface.co/datasets/kumapo/JAQKET/resolve/main/train_jaqket_59.350.json",
        "valid": "https://huggingface.co/datasets/kumapo/JAQKET/resolve/main/dev_jaqket_59.350.json",
    },
}


def dataset_info_v1() -> ds.Features:
    features = ds.Features(
        {
            "qid": ds.Value("string"),
            "question": ds.Value("string"),
            "answer_entity": ds.Value("string"),
            "label": ds.Value("int32"),
            "answer_candidates": ds.Sequence(
                ds.Value("string"),
            ),
            "contexts": ds.Sequence(
                ds.Value("string")
            )
        }
    )
    return ds.DatasetInfo(
        description=_DESCRIPTION,
        citation=_CITATION,
        homepage=_HOMEPAGE,
        license=_LICENSE,
        features=features,
    )


def dataset_info_v2() -> ds.Features:
    features = ds.Features(
        {
            "qid": ds.Value("string"),
            "question": ds.Value("string"),
            "answers": ds.Sequence({
                "text": ds.Value("string"),
                "answer_start": ds.Value("int32"),
            }),
            "ctxs": ds.Sequence({
                "id": ds.Value("string"),
                "title": ds.Value("string"),
                "text": ds.Value("string"),
                "score": ds.Value("float32"),
                "has_answer": ds.Value("bool"),
            })
        }
    )
    return ds.DatasetInfo(
        description=_DESCRIPTION,
        citation=_CITATION,
        homepage=_HOMEPAGE,
        license=_LICENSE,
        features=features,
    )


class JAQKET(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("0.1.0")
    BUILDER_CONFIGS = [
        ds.BuilderConfig(
            name="v1.0",
            version=VERSION,
            description=_DESCRIPTION_CONFIGS["v1.0"],
        ),
        ds.BuilderConfig(
            name="v2.0",
            version=VERSION,
            description=_DESCRIPTION_CONFIGS["v2.0"],
        ),
    ]

    def _info(self) -> ds.DatasetInfo:
        if self.config.name == "v1.0":
            return dataset_info_v1()
        elif self.config.name == "v2.0":
            return dataset_info_v2()
        else:
            raise ValueError(f"Invalid config name: {self.config.name}")

    def _split_generators(self, dl_manager: ds.DownloadManager):
        file_paths = dl_manager.download_and_extract(_URLS[self.config.name])
        if self.config.name == "v1.0":
            return [
                ds.SplitGenerator(
                    name=ds.Split.TRAIN,
                    gen_kwargs={"file_path": file_paths["train"], "entities_file_path": file_paths["candidate_entities"]},
                ),
                ds.SplitGenerator(
                    name=ds.Split.VALIDATION,
                    gen_kwargs={"file_path": file_paths["valid"], "entities_file_path": file_paths["candidate_entities"]},
                ),
            ]
        elif self.config.name == "v2.0":
            return [
                ds.SplitGenerator(
                    name=ds.Split.TRAIN,
                    gen_kwargs={"file_path": file_paths["train"]},
                ),
                ds.SplitGenerator(
                    name=ds.Split.VALIDATION,
                    gen_kwargs={"file_path": file_paths["valid"]},
                ),
            ]
        else:
            raise ValueError(f"Invalid config name: {self.config.name}")            

    def _generate_examples_v1(
        self,
        file_path: Optional[str] = None,
        entities_file_path: Optional[str] = None,
        num_candidates: Optional[int] = None,
    ):
        if file_path is None or entities_file_path is None:
            raise ValueError(f"Invalid argument for {self.config.name}")

        entities = dict()
        with open(entities_file_path, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            for line in lines:
                entity = json.loads(line.strip())
                entities[entity["title"]] = entity["text"]

        with open(file_path, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            for line in lines:
                data_raw = json.loads(line.strip("\n"))
                q_id = data_raw["qid"]
                question = data_raw["question"].replace("_", "")
                answer_entity = data_raw["answer_entity"]
                answer_candidates = data_raw["answer_candidates"][:num_candidates]

                if answer_entity not in answer_candidates:
                    continue
                if len(answer_candidates) != num_candidates:
                    continue
            
                contexts = [entities[answer_candidates[i]] for i in range(num_candidates)]
                label = str(answer_candidates.index(answer_entity))
                example_dict = {
                    "qid": q_id,
                    "question": question,
                    "answer_entity": answer_entity,
                    "label": label,
                    "answer_candidates": answer_candidates,
                    "contexts": contexts,
                }
                yield q_id, example_dict

    def _generate_examples_v2(
        self,
        file_path: Optional[str] = None
    ):
        if file_path is None:
            raise ValueError(f"Invalid argument for {self.config.name}")

        with open(file_path, "r") as rf:
            json_data = json.load(rf)

        for json_dict in json_data:
            q_id = json_dict["qid"]
            question = json_dict["question"]
            answers = [
                {"text": answer, "answer_start": -1 } # -1: dummy
                for answer in json_dict["answers"]
            ]
            ctxs = [
                {
                    "id": ctx["id"],
                    "title": ctx["title"],
                    "text": ctx["text"],
                    "score": float(ctx["score"]),
                    "has_answer": ctx["has_answer"]

                }
                for ctx in json_dict["ctxs"]
            ]
            example_dict = {
                "qid": q_id,
                "question": question,
                "answers": answers,
                "ctxs": ctxs
            }
            yield q_id, example_dict

    def _generate_examples(
        self,
        file_path: Optional[str] = None,
        entities_file_path: Optional[str] = None,
        num_candidates: int = 5,
    ):
        if self.config.name == "v1.0":
            yield from self._generate_examples_v1(file_path, entities_file_path, num_candidates)
        elif self.config.name == "v2.0":
            yield from self._generate_examples_v2(file_path)
        else:
            raise ValueError(f"Invalid config name: {self.config.name}")
