"""
JGLUE: Japanese General Language Understanding Evaluation
https://aclanthology.org/2022.lrec-1.317/

JGLUE, Japanese General Language Understanding Evaluation, is built to measure the general NLU ability in Japanese.
JGLUE has been constructed from scratch without translation.

Homepage: https://github.com/yahoojapan/JGLUE
"""
import os
import inspect
import datasets
from math import exp
from lm_eval.base import rf, Task
from functools import partial
from lm_eval.jasquad import jasquad

_CITATION = """
@inproceedings{kurihara-etal-2022-jglue,
    title = "{JGLUE}: {J}apanese General Language Understanding Evaluation",
    author = "Kurihara, Kentaro  and
      Kawahara, Daisuke  and
      Shibata, Tomohide",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.317",
    pages = "2957--2966",
    abstract = "To develop high-performance natural language understanding (NLU) models, it is necessary to have a benchmark to evaluate and analyze NLU ability from various perspectives. While the English NLU benchmark, GLUE, has been the forerunner, benchmarks are now being released for languages other than English, such as CLUE for Chinese and FLUE for French; but there is no such benchmark for Japanese. We build a Japanese NLU benchmark, JGLUE, from scratch without translation to measure the general NLU ability in Japanese. We hope that JGLUE will facilitate NLU research in Japanese.",
}
"""


DYNAMIC_MAX_LENGTH = os.getenv("DYNAMIC_MAX_LENGTH", "true").lower()


class JSQuAD(Task):
    """
    prompt template is taken from [日本語に特化した60億パラメータ規模のGPTモデルの構築と評価](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H9-4.pdf)
    """

    VERSION = 1.1
    PROMPT_VERSION = 0.1
    DATASET_PATH = "shunk031/JGLUE"
    DATASET_NAME = "JSQuAD"
    LOAD_TOKENIZER = True
    DESCRIPTION = "[題名]と[問題]から[質問]に対する[答え]を抜き出しなさい\n\n"
    SEP = "\n"
    REMOVE_IDS = []
    # REMOVE_IDS = ['a10743p19q0', 'a10743p19q1', 'a10743p19q2', 'a10743p19q3', 'a13221p1q0', 'a13221p1q1', 'a13221p1q2', 'a13221p1q3', 'a14985p1q0', 'a14985p1q1', 'a14985p1q2', 'a14985p1q3', 'a14985p1q4', 'a14985p93q0', 'a14985p93q1', 'a14985p93q2', 'a14985p93q3', 'a14985p93q4', 'a1540503p36q0', 'a1540503p36q1', 'a1540503p36q2', 'a1540503p36q3', 'a1540503p36q4', 'a18783p1q0', 'a18783p3q0', 'a18783p3q1', 'a18783p3q2', 'a18783p8q0', 'a18873p25q0', 'a18873p25q1', 'a18873p25q2', 'a18873p25q3', 'a18873p26q0', 'a18873p26q1', 'a18873p26q2', 'a20898p10q0', 'a20898p15q0', 'a20898p15q1', 'a20898p15q2', 'a20898p15q3', 'a2164640p22q0', 'a2164640p22q1', 'a2164640p22q2', 'a2164640p22q3', 'a2164640p22q4', 'a22392p20q0', 'a22392p20q1', 'a22392p20q2', 'a22392p20q3', 'a3011628p3q0', 'a3011628p3q1', 'a3011628p3q2', 'a3011628p3q3', 'a3189p4q0', 'a3189p4q1', 'a3189p4q2', 'a369953p0q0', 'a369953p0q1', 'a369953p0q2', 'a369953p0q3', 'a3949p1q0', 'a3949p1q1', 'a4596p0q0', 'a4596p0q1', 'a4596p0q2', 'a4596p0q3', 'a4596p1q0', 'a4596p1q1', 'a4596p1q2', 'a4596p1q3', 'a4596p1q4', 'a4596p38q0', 'a4596p38q1', 'a4596p38q2', 'a4596p38q3', 'a4596p38q4', 'a4768p13q0', 'a4768p13q1', 'a4768p13q2', 'a4768p3q0', 'a4768p3q1', 'a4768p3q2', 'a4768p3q3', 'a4768p8q0', 'a4768p8q1', 'a4768p8q2', 'a51481p0q0', 'a51481p0q1', 'a51481p0q2', 'a51481p10q0', 'a51481p10q1', 'a51481p10q2', 'a51481p10q3', 'a51481p6q0', 'a51481p6q1', 'a51481p6q2', 'a51481p6q3', 'a51481p7q0', 'a51481p7q1', 'a67892p11q0', 'a67892p11q1', 'a67892p11q2', 'a67892p11q3', 'a67892p2q0', 'a8874p6q0', 'a8874p6q1', 'a916079p3q0', 'a916079p3q1', 'a95156p4q0', 'a95156p4q1', 'a95156p4q2', 'a95156p4q3', 'a95156p6q0', 'a95156p6q1', 'a95156p6q2', 'a95156p6q3']
    """
    @mkshing's comment
    I found that JSQuAD contains errors inside contexts such as below.
    ```
    {'id': 'a4596p0q0', 'title': 'ポルトガル', 'context': 'ポルトガル [SEP] 正式名称はポルトガル語で、。通称、 。', 'question': 'ポルトガルね正式名称は何語であるか', 'answers': {'text': ['正式名称はポルトガル語', 'ポルトガル語', 'ポルトガル語'], 'answer_start': [12, 17, 17]}, 'is_impossible': False}
    ```
    So, I tried to identify all of them and found that the following processing can be okay to detect the ids
    ```python
    from datasets import load_dataset
    from transformers import T5Tokenizer
    dataset = load_dataset("shunk031/JGLUE", name="JSQuAD", split="validation")
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
    remove_ids = []
    for item in dataset:
        ctx = item["context"].split("[SEP]")[-1].strip()
        input_ids = tokenizer.encode(ctx, add_special_tokens=False)
        if len(input_ids) < 25:
            print(item)
            remove_ids.append(item["id"])
    ```
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.jasquad_metric = datasets.load_metric(jasquad.__file__)

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
        return (
            "[題名]:"
            + doc["title"]
            + f"{self.SEP}"
            + "[問題]:"
            + doc["context"].split("[SEP]")[-1].strip()
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
            max_num_tokens = max(
                [
                    len(encode_fn(answer, **encode_params))
                    for answer in doc["answers"]["text"]
                ]
            )
            continuation = rf.greedy_until(ctx, [self.SEP], max_num_tokens)
        return continuation

    def process_results(self, doc, results):
        assert (
            len(results) == 1
        ), f"results should be a list with 1 str element, but is {results}"
        continuation = results[0]
        predictions = {
            "id": doc["id"],
            "prediction_text": continuation,
        }

        references = {
            "id": doc["id"],
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

        # add verbose output
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
        return self.jasquad_metric.compute(
            predictions=predictions, references=references
        )

    def _squad_agg(self, key, item):
        predictions, references = zip(*item)
        return self._squad_metric(predictions=predictions, references=references)[key]


class JSQuADWithFintanPrompt(JSQuAD):
    """
    prompt template is taken from [ChatGPT vs BERT: どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
    """

    PROMPT_VERSION = 0.2
    DESCRIPTION = "質問に対する回答を文章から一言で抽出してください。回答は名詞で答えてください。\n\n"
    SEP = "\n"

    def doc_to_text(self, doc):
        return (
            "文章:"
            + doc["context"].split("[SEP]")[-1].strip()
            + f"{self.SEP}"
            + "質問:"
            + doc["question"]
            + f"{self.SEP}"
            + "回答:"
        )


class JSQuADWithFintanPromptV12(JSQuADWithFintanPrompt):
    """
    prompt template is taken from [ChatGPT vs BERT: どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
    """

    VERSION = 1.2
    DESCRIPTION = "質問に対する回答を題名と文章から一言で抽出してください。回答は名詞で答えてください。\n\n"

    def doc_to_text(self, doc):
        return (
            "題名:"
            + doc["title"]
            + f"{self.SEP}"
            + "文章:"
            + doc["context"].split("[SEP]")[-1].strip()
            + f"{self.SEP}"
            + "質問:"
            + doc["question"]
            + f"{self.SEP}"
            + "回答:"
        )


class JSQuADWithJAAlpacaPrompt(JSQuAD):
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
        input_text = (
            f"文脈：{doc['context'].split('[SEP]')[-1].strip()}\n質問：{doc['question']}"
        )
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{input_text}\n\n### 応答:\n"


class JSQuADWithJAAlpacaPromptV12(JSQuADWithJAAlpacaPrompt):
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

    VERSION = 1.2

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
        input_text = f"文脈：{doc['title']}\n{doc['context'].split('[SEP]')[-1].strip()}\n質問：{doc['question']}"
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{input_text}\n\n### 応答:\n"


class JSQuADWithRinnaInstructionSFT(JSQuAD):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """

    PROMPT_VERSION = 0.4
    DESCRIPTION = "ユーザー: 与えられた文脈から、質問に対する答えを抜き出してください。<NL>システム: 分かりました。<NL>"
    SEP = "<NL>"
    FEWSHOT_SEP = "<NL>"

    def doc_to_text(self, doc):
        input_text = f"文脈：{doc['context'].split('[SEP]')[-1].strip()}{self.SEP}質問：{doc['question']}"
        return f"ユーザー: {input_text}{self.SEP}システム: "


class JSQuADWithRinnaInstructionSFTV12(JSQuADWithRinnaInstructionSFT):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """

    VERSION = 1.2

    def doc_to_text(self, doc):
        input_text = f"文脈：{doc['title']}{self.SEP}{doc['context'].split('[SEP]')[-1].strip()}{self.SEP}質問：{doc['question']}"
        return f"ユーザー: {input_text}{self.SEP}システム: "


class JSQuADWithRinnaBilingualInstructionSFT(JSQuADWithRinnaInstructionSFT):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-sft
    """

    PROMPT_VERSION = 0.5
    DESCRIPTION = "ユーザー: 与えられた文脈から、質問に対する答えを抜き出してください。\nシステム: 分かりました。\n"
    SEP = "\n"
    FEWSHOT_SEP = "\n"


class JSQuADWithRinnaBilingualInstructionSFTV12(JSQuADWithRinnaBilingualInstructionSFT):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-sft
    """

    VERSION = 1.2

    def doc_to_text(self, doc):
        input_text = f"文脈：{doc['title']}{self.SEP}{doc['context'].split('[SEP]')[-1].strip()}{self.SEP}質問：{doc['question']}"
        return f"ユーザー: {input_text}{self.SEP}システム: "


class JSQuADWithLlama2(JSQuAD):
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
    INSTRUCTION = "与えられた文脈から、質問に対する答えを抜き出してください。"
    FEWSHOT_SEP = " </s><s>[INST] "

    def doc_to_text(self, doc):
        """
        Insert the following prompt into `{{ user_msg }}`
        ```
        与えられた文脈から、質問に対する答えを抜き出してください。

        文脈：...
        質問：... [/INST]
        ```
        """
        input_text = (
            f"文脈：{doc['context'].split('[SEP]')[-1].strip()}\n質問：{doc['question']}"
        )
        return f"{self.INSTRUCTION}\n\n{input_text} [/INST] "


class JSQuADWithLlama2V12(JSQuADWithLlama2):
    VERSION = 1.2

    def doc_to_text(self, doc):
        """
        Insert the following prompt into `{{ user_msg }}`
        ```
        与えられた文脈から、質問に対する答えを抜き出してください。

        文脈：...
        質問：... [/INST]
        ```
        """
        input_text = f"文脈：{doc['title']}\n{doc['context'].split('[SEP]')[-1].strip()}\n質問：{doc['question']}"
        return f"{self.INSTRUCTION}\n\n{input_text} [/INST] "


VERSIONS = [
    JSQuAD,
    JSQuADWithFintanPrompt,
    JSQuADWithFintanPromptV12,
    JSQuADWithJAAlpacaPrompt,
    JSQuADWithJAAlpacaPromptV12,
    JSQuADWithRinnaInstructionSFT,
    JSQuADWithRinnaInstructionSFTV12,
    JSQuADWithRinnaBilingualInstructionSFT,
    JSQuADWithRinnaBilingualInstructionSFTV12,
    JSQuADWithLlama2,
    JSQuADWithLlama2V12,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[
            f"jsquad-{version_class.VERSION}-{version_class.PROMPT_VERSION}"
        ] = version_class
    return tasks
