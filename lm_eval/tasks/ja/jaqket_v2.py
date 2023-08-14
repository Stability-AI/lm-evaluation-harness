"""
JAQKET: JApanese Questions on Knowledge of EnTitie
https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf


Homepage: https://www.nlp.ecei.tohoku.ac.jp/projects/jaqket/
"""
import os
import inspect
import datasets
from math import exp
from lm_eval.base import rf, Task
from functools import partial
from lm_eval.jasquad import jasquad

_CITATION = """
@InProceedings{Kurihara_nlp2020,
  author =  "鈴木正敏 and 鈴木潤 and 松田耕史 and ⻄田京介 and 井之上直也",
  title =   "JAQKET: クイズを題材にした日本語 QA データセットの構築",
  booktitle =   "言語処理学会第26回年次大会",
  year =    "2020",
  url = "https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf"
  note= "in Japanese"
"""

_TOP_K_LIMIT = 10
DYNAMIC_MAX_LENGTH = os.getenv("DYNAMIC_MAX_LENGTH", "true").lower()

class JAQKETV2(Task):
    """
    prompt template is taken from [日本語に特化した60億パラメータ規模のGPTモデルの構築と評価](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H9-4.pdf)
    """
    VERSION = 0.1
    PROMPT_VERSION = 0.1
    DATASET_PATH = "kumapo/JAQKET"
    DATASET_NAME = "v2.0"
    LOAD_TOKENIZER = True
    DESCRIPTION = "[題名]と[問題]から[質問]に対する[答え]を抜き出しなさい\n\n"
    SEP = "\n"
    REMOVE_IDS = []
    TOP_K_LIMIT = _TOP_K_LIMIT
   

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.jasqaud_metric = datasets.load_metric(jasquad.__file__)

    def has_training_docs(self):
        return True
    
    def has_validation_docs(self):
        return True
    
    def has_test_docs(self):
        return False
    
    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        dataset = self.dataset["validation"]
        if len(self.REMOVE_IDS) > 0:
            dataset = [item for item in dataset if item["id"] not in self.REMOVE_IDS]
        return dataset
    
    def doc_to_text(self, doc):
        topk_titles = doc["ctxs"]["title"][:self.TOP_K_LIMIT]
        topk_contexts = doc["ctxs"]["text"][:self.TOP_K_LIMIT]
        context = f"{self.SEP}".join([
            "[題名]:"
            + title
            + f"{self.SEP}"
            + "[問題]:"
            + context
            for title, context in zip(topk_titles, topk_contexts)
        ])
        return (
            context
            + f"{self.SEP}"
            + "[質問]:"
            + doc["question"]
            + f"{self.SEP}"
            + "[答え]:"
        )

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        answer_list = doc["answers"]["text"]
        answer = answer_list[0]
        return answer

    def construct_requests(self, doc, ctx):
        if DYNAMIC_MAX_LENGTH == "false" or not hasattr(self.tokenizer, "encode"):
            continuation = rf.greedy_until(ctx, [self.SEP])
        else:
            encode_fn = self.tokenizer.encode
            if "add_special_tokens" in inspect.getfullargspec(encode_fn).args:
                encode_params = dict(add_special_tokens=False)
            else:
                encode_params = {}
            max_num_tokens = max([len(encode_fn(answer, **encode_params)) for answer in doc["answers"]["text"]])
            continuation = rf.greedy_until(ctx, [self.SEP], max_num_tokens)
        return continuation
    
    def process_results(self, doc, results):
        assert len(results) == 1, f"results should be a list with 1 str element, but is {results}"
        continuation = results[0]
        predictions = {
            "id": doc["qid"],
            "prediction_text": continuation,
        }

        references = {
            "id": doc["qid"],
            "answers": doc["answers"],
        }
        return {
            "exact_match": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
        }


    def aggregation(self):
        return {
            "exact_match": partial(
                self._squad_agg, "exact_match"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": partial(
                self._squad_agg, "f1"
            ),  # The F-score of predicted tokens versus the gold answer
        }
    
    def higher_is_better(self):
        return {
            "exact_match": True,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": True,  # The F-score of predicted tokens versus the gold answer
        }

    def _squad_metric(self, predictions, references):
        return self.jasqaud_metric.compute(predictions=predictions, references=references)


    def _squad_agg(self, key, item):
        predictions, references = zip(*item)
        return self._squad_metric(predictions=predictions, references=references)[key]

class JAQKETV2WithFintanPrompt(JAQKETV2):
    """
    prompt template is taken from [ChatGPT vs BERT: どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
    """
    PROMPT_VERSION = 0.2
    DESCRIPTION = "質問に対する回答を文章から一言で抽出してください。回答は名詞で答えてください。\n\n"
    SEP = "\n"
    TOP_K_LIMIT = _TOP_K_LIMIT
    def doc_to_text(self, doc):
        context = f"{self.SEP}".join([ctx for ctx in doc["ctxs"]["text"][:self.TOP_K_LIMIT]])
        return (
            "文章:"
            + context
            + f"{self.SEP}"
            + "質問:"
            + doc["question"]
            + f"{self.SEP}"
            + "回答:"
        )
    

class JAQKETV2WithJAAlpacaPrompt(JAQKETV2):
    """
    This prompt format was inspired by the below data in fujiki/japanese_alpaca_data. 
    ```
    {
        'instruction': '与えられた文脈に最も適した文を選択してください。', 
        'input': '文脈：あなたは親友と現在の仕事の状況について話しています。\nA）私にはあまり選択肢がありません。\nB）他に選択肢がありません。\nC）私には本当に決断する必要がありません。', 
        'output': 'A) 私には多くの選択肢がありません。'
    }
    ```
    Reference:
    - data: https://huggingface.co/datasets/fujiki/japanese_alpaca_data
    - code: https://github.com/Stability-AI/gpt-neox/blob/c130a4edc1120dccec8f02a34eb60d3e8f484cd3/finetune/finetune_base_ja.py#LL118C23-L127C11
    """
    PROMPT_VERSION = 0.3
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    INSTRUCTION = "与えられた文脈から、質問に対する答えを抜き出してください。"
    TOP_K_LIMIT = _TOP_K_LIMIT
    def doc_to_text(self, doc):
        """
        以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

        ### 指示: 
        {instruction}

        ### 入力: 
        {input}

        ### 応答: 
        {response}
        """
        context = f"{self.SEP}".join([ctx for ctx in doc["ctxs"]["text"][:self.TOP_K_LIMIT]])
        input_text = f"文脈：{context}\n質問：{doc['question']}"
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{input_text}\n\n### 応答:\n"


class JAQKETV2WithRinnaInstructionSFT(JAQKETV2):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """
    PROMPT_VERSION = 0.4
    DESCRIPTION = "ユーザー: 与えられた文脈から、質問に対する答えを抜き出してください。<NL>システム: 分かりました。<NL>"
    TOP_K_LIMIT = _TOP_K_LIMIT
    SEP = "<NL>"
    FEWSHOT_SEP = "<NL>"

    def doc_to_text(self, doc):
        context = self.SEP.join([ctx for ctx in doc["ctxs"]["text"][:self.TOP_K_LIMIT]])
        input_text = f"文脈：{context}{self.SEP}質問：{doc['question']}"
        return f"ユーザー: {input_text}{self.SEP}システム: "


class JAQKETV2WithRinnaBilingualInstructionSFT(JAQKETV2WithRinnaInstructionSFT):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-sft
    """
    PROMPT_VERSION = 0.5
    DESCRIPTION = "ユーザー: 与えられた文脈から、質問に対する答えを抜き出してください。\nシステム: 分かりました。\n"
    SEP = "\n"
    FEWSHOT_SEP = "\n"


VERSIONS = [
    JAQKETV2,
    JAQKETV2WithFintanPrompt,
    JAQKETV2WithJAAlpacaPrompt,
    JAQKETV2WithRinnaInstructionSFT,
    JAQKETV2WithRinnaBilingualInstructionSFT
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[f"jaqket_v2-{version_class.VERSION}-{version_class.PROMPT_VERSION}"] = version_class
    return tasks
