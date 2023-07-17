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

_TOP_K_LIMIT = 5
_FALLBACK_DOC = {
    "question": "人気漫画『ドラえもん』の登場人物で、ジャイアンの苗字は剛田ですが、スネ夫の苗字は何でしょう?",
    "answers": [
        "骨川"
    ],
    "ctxs": [
        {
            "id": "1075197",
            "title": "大長編ドラえもん",
            "text": "通常の『ドラえもん』が掲載1回毎の完結を基本としているのに対し、『大長編』は映画1作の原作となる1つの長編が数回に分けて連載され、ドラえもん・野比のび太・源静香・剛田武(ジャイアン)・骨川スネ夫の5人が編毎に異なる様々な冒険に立ち向かう様が描かれる。単行本も『ドラえもん』から独立した『大長編ドラえもん』として発行されている。",
            "score": "34.9877",
            "has_answer": True
        }
    ]
}
DYNAMIC_MAX_LENGTH = os.getenv("DYNAMIC_MAX_LENGTH", "true").lower()

class JAQKETV2(Task):
    """
    prompt template is taken from [日本語に特化した60億パラメータ規模のGPTモデルの構築と評価](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H9-4.pdf)
    """
    VERSION = 0.2
    PROMPT_VERSION = 0.1
    DATASET_PATH = "kumapo/JAQKET"
    DATASET_NAME = "v2.0"
    LOAD_TOKENIZER = True
    DESCRIPTION = "[題名]と[問題]から[質問]に対する[答え]を抜き出しなさい\n\n"
    SEP = "\n"
    FEWSHOT_SEP = "\n\n"
    REMOVE_IDS = []
    TOP_K_LIMIT = _TOP_K_LIMIT
    FALLBACK_DOC = _FALLBACK_DOC

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
    
    def doc_to_qa_prompt(self, doc):
        return (
            "[質問]:"
            + doc["question"]
            + self.SEP
            + "[答え]:"
        )

    def doc_to_text(self, doc):
        topk_titles = doc["ctxs"]["title"][:self.TOP_K_LIMIT]
        topk_contexts = doc["ctxs"]["text"][:self.TOP_K_LIMIT]
        answer_candidate = self.SEP.join([
            (
                "[題名]:"
                + title
                + self.SEP
                + "[問題]:"
                + context
            )
            for title, context in zip(topk_titles, topk_contexts)
        ])
        qa_prompt = self.doc_to_qa_prompt(doc)
        return (
            answer_candidate
            + self.SEP
            + qa_prompt   
        )

    def doc_to_answering_text(self, doc, fallback_doc):
        answering_contexts = [
            {k: v[i] for k, v in doc["ctxs"].items()}
            for i, a in enumerate(doc["ctxs"]["has_answer"]) if a == True
        ]
        if len(answering_contexts) < 1:
            doc = fallback_doc
            answering_contexts = fallback_doc["ctxs"]
        answer_candidate = (
            "[題名]:"
            + answering_contexts[0]["title"]
            + self.SEP
            + "[問題]:"
            + answering_contexts[0]["text"]
        )
        qa_prompt = self.doc_to_qa_prompt(doc)
        return (
            answer_candidate
            + self.SEP
            + qa_prompt   
        )

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        answer_list = doc["answers"]["text"]
        answer = answer_list[0]
        return answer

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        if hasattr(self, "FEWSHOT_SEP"):
            FEWSHOT_SEP = self.FEWSHOT_SEP
        elif hasattr(self, "SEP"):
            FEWSHOT_SEP = f"{self.SEP}{self.SEP}"
        else:        
            FEWSHOT_SEP = "\n\n"
            
        if description:
            description += FEWSHOT_SEP
        elif hasattr(self, "DESCRIPTION"):
            description = self.DESCRIPTION
        else:
            description = ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs()
                        if self.has_validation_docs()
                        else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = (
                FEWSHOT_SEP.join(
                    [
                        self.doc_to_answering_text(doc, self.FALLBACK_DOC) + self.doc_to_target(doc)
                        for doc in fewshotex
                    ]
                )
                + FEWSHOT_SEP
            )

        example = self.doc_to_text(doc)
        return description + labeled_examples + example

    def preprocess_ctx(self, ctx, max_length):
        # if ctx fits in max length, return
        if len(self.tokenizer.encode(ctx)) <= max_length:
            return ctx
        
        # if ctx is too long, split on a tag that separates each example
        description, remainder = ctx.split(self.FEWSHOT_SEP, 1)
        ctxs = remainder.split(self.FEWSHOT_SEP)

        # if there is no example and still the prompt is too long, fail
        if len(ctxs) < 2:
            raise ValueError(f"0-shot description+example doesn't fit in max length. ctx: {ctx}")

        # delete the first example, last is questioning example
        del ctxs[0]

        # recurse
        return self.preprocess_ctx(self.FEWSHOT_SEP.join([description, *ctxs]), max_length)

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
            ctx = self.preprocess_ctx(ctx, max_length=self.max_length-max_num_tokens)
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
    def doc_to_qa_prompt(self, doc):
        return (
            "質問:"
            + doc["question"]
            + self.SEP
            + "回答:"
        )

    def doc_to_text(self, doc):
        topk_contexts = doc["ctxs"]["text"][:self.TOP_K_LIMIT]
        context = self.SEP.join([text for text in topk_contexts])
        answer_candidate =  "文章:" + context
        qa_prompt = self.doc_to_qa_prompt(doc)
        return (
            answer_candidate
            + self.SEP
            + qa_prompt   
        )

    def doc_to_answering_text(self, doc, fallback_doc):
        answering_contexts = [
            {k: v[i] for k, v in doc["ctxs"].items()}
            for i, a in enumerate(doc["ctxs"]["has_answer"]) if a == True
        ]
        if len(answering_contexts) < 1:
            doc = fallback_doc
            answering_contexts = fallback_doc["ctxs"]
        answer_candidate =  "文章:" + answering_contexts[0]["text"]
        qa_prompt = self.doc_to_qa_prompt(doc)
        return (
            answer_candidate
            + self.SEP
            + qa_prompt
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
    def doc_to_qa_prompt(self, doc):
        return (
            "質問：" 
            + doc["question"]
        )

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
        topk_contexts = doc["ctxs"]["text"][:self.TOP_K_LIMIT]
        context = self.SEP.join([text for text in topk_contexts])
        answer_candidate = "文脈：" + context
        qa_prompt = self.doc_to_qa_prompt(doc)
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{answer_candidate}\n{qa_prompt}\n\n### 応答:\n"

    def doc_to_answering_text(self, doc, fallback_doc):
        answering_contexts = [
            {k: v[i] for k, v in doc["ctxs"].items()}
            for i, a in enumerate(doc["ctxs"]["has_answer"]) if a == True
        ]
        if len(answering_contexts) < 1:
            doc = fallback_doc
            answering_contexts = fallback_doc["ctxs"]
        answer_candidate = "文脈：" + answering_contexts[0]["text"]
        qa_prompt = self.doc_to_qa_prompt(doc)
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{answer_candidate}\n{qa_prompt}\n\n### 応答:\n"

class JAQKETV2WithRinnaInstructionSFT(JAQKETV2):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """
    PROMPT_VERSION = 0.4
    DESCRIPTION = "ユーザー: 与えられた文脈から、質問に対する答えを抜き出してください。<NL>システム: 分かりました。<NL>"
    SEP = "<NL>"
    FEWSHOT_SEP = "<NL>"
    END_OF_DESCRIPTION = "システム: 分かりました。<NL>"
    START_OF_FEWSHOT = "ユーザー: 文脈："

    def doc_to_text(self, doc):
        context = self.SEP.join([ctx for ctx in doc["ctxs"]["text"][:self.TOP_K_LIMIT]])
        input_text = f"文脈：{context}{self.SEP}質問：{doc['question']}"
        return f"ユーザー: 文脈：{context}{self.SEP}質問：{doc['question']}{self.SEP}システム: "

    def doc_to_qa_prompt(self, doc):
        return (
            "質問：" 
            + doc["question"]
        )

    def doc_to_text(self, doc):
        topk_contexts = doc["ctxs"]["text"][:self.TOP_K_LIMIT]
        context = self.SEP.join([text for text in topk_contexts])
        answer_candidate = "文脈：" + context
        qa_prompt = self.doc_to_qa_prompt(doc)
        return f"ユーザー: {answer_candidate}{self.SEP}質問：{doc['question']}{self.SEP}システム: "

    def doc_to_answering_text(self, doc, fallback_doc):
        answering_contexts = [
            {k: v[i] for k, v in doc["ctxs"].items()}
            for i, a in enumerate(doc["ctxs"]["has_answer"]) if a == True
        ]
        if len(answering_contexts) < 1:
            doc = fallback_doc
            answering_contexts = fallback_doc["ctxs"]
        answer_candidate = "文脈：" + answering_contexts[0]["text"]
        qa_prompt = self.doc_to_qa_prompt(doc)
        return f"ユーザー: {answer_candidate}{self.SEP}質問：{doc['question']}{self.SEP}システム: "

    def preprocess_ctx(self, ctx, max_length):
        # if ctx fits in max length, return
        if len(self.tokenizer.encode(ctx)) <= max_length:
            return ctx

        # if ctx is too long, split on a tag that separates each example
        description, remainder = ctx.split(self.END_OF_DESCRIPTION, 1)
        ctxs = remainder.split(self.START_OF_FEWSHOT)

        # if there is no example and still the prompt is too long, fail
        if len(ctxs) < 2:
            raise ValueError(f"0-shot description+example doesn't fit in max length. ctx: {ctx}")

        # delete the first example, last is questioning example
        del ctxs[1]

        new_ctx = self.END_OF_DESCRIPTION.join([description, self.START_OF_FEWSHOT.join(ctxs)])
        # recurse
        return self.preprocess_ctx(new_ctx, max_length)

VERSIONS = [
    JAQKETV2,
    JAQKETV2WithFintanPrompt,
    JAQKETV2WithJAAlpacaPrompt,
    JAQKETV2WithRinnaInstructionSFT,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[f"jaqket_v2-{version_class.VERSION}-{version_class.PROMPT_VERSION}"] = version_class
    return tasks
