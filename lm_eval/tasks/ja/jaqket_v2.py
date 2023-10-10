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

TOP_K_LIMIT = 5
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.jasqaud_metric = datasets.load_metric(jasquad.__file__)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        """Downloads and returns the task dataset.
        Override this method to download the dataset from a custom API.

        :param data_dir: str
            Stores the path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when
            the dataset is not publicly accessible).
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`
            NOTE: You can change the cache location globally for a given process
            by setting the shell environment variable, `HF_DATASETS_CACHE`,
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        :param download_mode: datasets.DownloadMode
            How to treat pre-existing `Task` downloads and data.
            - `datasets.DownloadMode.REUSE_DATASET_IF_EXISTS`
                Reuse download and reuse dataset.
            - `datasets.DownloadMode.REUSE_CACHE_IF_EXISTS`
                Reuse download with fresh dataset.
            - `datasets.DownloadMode.FORCE_REDOWNLOAD`
                Fresh download and fresh dataset.
        """
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
            num_contexts=TOP_K_LIMIT,
        )

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
        return "[質問]:" + doc["question"] + self.SEP + "[答え]:"

    def doc_to_text(self, doc):
        answer_candidate = self.SEP.join(
            [
                ("[題名]:" + title + self.SEP + "[問題]:" + context)
                for title, context in zip(doc["ctxs"]["title"], doc["ctxs"]["text"])
            ]
        )
        qa_prompt = self.doc_to_qa_prompt(doc)
        return answer_candidate + self.SEP + qa_prompt

    def doc_to_answering_text(self, doc):
        has_answer = doc["ctxs"]["has_answer"]
        answering_index = has_answer.index(True)
        answering_contexts = {
            k: v[answering_index : answering_index + 1] for k, v in doc["ctxs"].items()
        }
        answer_candidate = (
            "[題名]:"
            + answering_contexts["title"][0]
            + self.SEP
            + "[問題]:"
            + answering_contexts["text"][0]
        )
        qa_prompt = self.doc_to_qa_prompt(doc)
        return answer_candidate + self.SEP + qa_prompt

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
                        self.doc_to_answering_text(doc) + self.doc_to_target(doc)
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

        # if there is no example and still the description + QA prompt is too long, fail
        if len(ctxs) < 2:
            raise ValueError(
                f"description + QA prompt with no example (0-shot) doesn't fit in max_length. ctx: {ctx}"
            )

        # delete the first example, the last includes QA prompt to be answered by lm
        del ctxs[0]

        # recur
        return self.preprocess_ctx(
            self.FEWSHOT_SEP.join([description, *ctxs]), max_length
        )

    def construct_requests(self, doc, ctx):
        if DYNAMIC_MAX_LENGTH == "false" or not hasattr(self.tokenizer, "encode"):
            continuation = rf.greedy_until(ctx, [self.SEP])
        else:
            encode_fn = self.tokenizer.encode
            if "add_special_tokens" in inspect.getfullargspec(encode_fn).args:
                encode_params = dict(add_special_tokens=False)
            else:
                encode_params = {}
            max_num_tokens = max(
                [
                    len(encode_fn(answer, **encode_params))
                    for answer in doc["answers"]["text"]
                ]
            )
            ctx = self.preprocess_ctx(ctx, max_length=self.max_length - max_num_tokens)
            continuation = rf.greedy_until(ctx, [self.SEP], max_num_tokens)
        return continuation

    def process_results(self, doc, results):
        assert (
            len(results) == 1
        ), f"results should be a list with 1 str element, but is {results}"
        continuation = results[0]
        predictions = {
            "id": doc["qid"],
            "prediction_text": continuation,
        }

        references = {
            "id": doc["qid"],
            "answers": doc["answers"],
        }
        out = {
            "exact_match": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
        }

        # add details. Because the metric computation isn't simple (probably?)
        # always include it.
        out["details"] = {
            "question": doc["question"],
            "response": continuation,
            "gold": doc["answers"],
        }

        return out

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
        return self.jasqaud_metric.compute(
            predictions=predictions, references=references
        )

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
        return "質問:" + doc["question"] + self.SEP + "回答:"

    def doc_to_text(self, doc):
        context = self.SEP.join([text for text in doc["ctxs"]["text"]])
        answer_candidate = "文章:" + context
        qa_prompt = self.doc_to_qa_prompt(doc)
        return answer_candidate + self.SEP + qa_prompt

    def doc_to_answering_text(self, doc):
        has_answer = doc["ctxs"]["has_answer"]
        answering_index = has_answer.index(True)
        answering_contexts = {
            k: v[answering_index : answering_index + 1] for k, v in doc["ctxs"].items()
        }
        answer_candidate = "文章:" + answering_contexts["text"][0]
        qa_prompt = self.doc_to_qa_prompt(doc)
        return answer_candidate + self.SEP + qa_prompt


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
        return "質問：" + doc["question"]

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
        context = self.SEP.join([text for text in doc["ctxs"]["text"]])
        answer_candidate = "文脈：" + context
        qa_prompt = self.doc_to_qa_prompt(doc)
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{answer_candidate}\n{qa_prompt}\n\n### 応答:\n"

    def doc_to_answering_text(self, doc):
        has_answer = doc["ctxs"]["has_answer"]
        answering_index = has_answer.index(True)
        answering_contexts = {
            k: v[answering_index : answering_index + 1] for k, v in doc["ctxs"].items()
        }
        answer_candidate = "文脈：" + answering_contexts["text"][0]
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

    def doc_to_qa_prompt(self, doc):
        return "質問：" + doc["question"]

    def doc_to_text(self, doc):
        context = self.SEP.join([text for text in doc["ctxs"]["text"]])
        answer_candidate = "文脈：" + context
        qa_prompt = self.doc_to_qa_prompt(doc)
        return f"ユーザー: {answer_candidate}{self.SEP}{qa_prompt}{self.SEP}システム: "

    def doc_to_answering_text(self, doc):
        has_answer = doc["ctxs"]["has_answer"]
        answering_index = has_answer.index(True)
        answering_contexts = {
            k: v[answering_index : answering_index + 1] for k, v in doc["ctxs"].items()
        }
        answer_candidate = "文脈：" + answering_contexts["text"][0]
        qa_prompt = self.doc_to_qa_prompt(doc)
        return f"ユーザー: {answer_candidate}{self.SEP}{qa_prompt}{self.SEP}システム: "

    def preprocess_ctx(self, ctx, max_length):
        # if ctx fits in max length, return
        if len(self.tokenizer.encode(ctx)) <= max_length:
            return ctx

        # if ctx is too long, split on a tag that separates each example
        description, remainder = ctx.split(self.END_OF_DESCRIPTION, 1)
        ctxs = remainder.split(self.START_OF_FEWSHOT)

        # if there is no example and still the description + QA prompt is too long, fail
        if len(ctxs) < 2:
            raise ValueError(
                f"description + QA prompt with no example (0-shot) doesn't fit in max_length. ctx: {ctx}"
            )

        # delete the first example, the last includes QA prompt to be answered by lm
        del ctxs[1]

        new_ctx = self.END_OF_DESCRIPTION.join(
            [description, self.START_OF_FEWSHOT.join(ctxs)]
        )
        # recur
        return self.preprocess_ctx(new_ctx, max_length)


class JAQKETV2WithRinnaBilingualInstructionSFT(JAQKETV2WithRinnaInstructionSFT):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-sft
    """

    PROMPT_VERSION = 0.5
    DESCRIPTION = "ユーザー: 与えられた文脈から、質問に対する答えを抜き出してください。\nシステム: 分かりました。\n"
    SEP = "\n"
    FEWSHOT_SEP = "\n"


class JAQKETV2WithLlama2(JAQKETV2WithJAAlpacaPrompt):
    """
    This prompt version follows the Llama2-chat's prompt format:
    ```
    <s>[INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
    ```
    reference: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    """

    PROMPT_VERSION = 0.6
    DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    DESCRIPTION = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
    FEWSHOT_SEP = " </s><s>[INST] "

    def doc_to_text(self, doc):
        """
        Insert the following prompt into `{{ user_msg }}`, which is based on prompt version 0.3
        ```
        与えられた文脈から、質問に対する答えを抜き出してください。

        文脈：{context}
        質問：{question} [/INST]
        ```
        """
        context = self.SEP.join([text for text in doc["ctxs"]["text"]])
        answer_candidate = "文脈：" + context
        qa_prompt = self.doc_to_qa_prompt(doc)
        return f"{self.INSTRUCTION}\n\n{answer_candidate}\n{qa_prompt} [/INST] "

    def doc_to_answering_text(self, doc):
        has_answer = doc["ctxs"]["has_answer"]
        answering_index = has_answer.index(True)
        answering_contexts = {
            k: v[answering_index : answering_index + 1] for k, v in doc["ctxs"].items()
        }
        answer_candidate = "文脈：" + answering_contexts["text"][0]
        qa_prompt = self.doc_to_qa_prompt(doc)
        return f"{self.INSTRUCTION}\n\n{answer_candidate}\n{qa_prompt} [/INST] "


VERSIONS = [
    JAQKETV2,
    JAQKETV2WithFintanPrompt,
    JAQKETV2WithJAAlpacaPrompt,
    JAQKETV2WithRinnaInstructionSFT,
    JAQKETV2WithRinnaBilingualInstructionSFT,
    JAQKETV2WithLlama2,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[
            f"jaqket_v2-{version_class.VERSION}-{version_class.PROMPT_VERSION}"
        ] = version_class
    return tasks
