"""
Language Models are Multilingual Chain-of-Thought Reasoners
https://arxiv.org/pdf/2210.03057.pdf

Multilingual Grade School Math problems with a numerical answer and a chain-of-thought prompt.
"""
import os
from lm_eval.base import rf
from lm_eval.tasks.gsm8k import GradeSchoolMath8K, INVALID_ANS
import re
import inspect

_CITATION = """
@misc{shi2022language,
      title={Language Models are Multilingual Chain-of-Thought Reasoners},
      author={Freda Shi and Mirac Suzgun and Markus Freitag and Xuezhi Wang and Suraj Srivats and Soroush Vosoughi and Hyung Won Chung and Yi Tay and Sebastian Ruder and Denny Zhou and Dipanjan Das and Jason Wei},
      year={2022},
      eprint={2210.03057},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")


class MGSM(GradeSchoolMath8K):
    DATASET_PATH = "juletxara/mgsm"
    DATASET_NAME = "ja"

    VERSION = 1.0
    PROMPT_VERSION = 0.0
    SEP = "\n"
    LOAD_TOKENIZER = True

    def doc_to_text(self, doc):
        # 問題：has to be removed and re-added because
        # the training set has it but the test set doesn't
        return f"問題：{doc['question'].replace('問題：','')}{self.SEP}ステップごとの答え："

    def doc_to_target(self, doc):
        # ステップごとの答え： is in text instead of target
        # so that the model doesn't have to generate it
        return "" + doc["answer"].replace("ステップごとの答え：", "")

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        max_length = self.max_length - self.max_gen_toks

        # If the prompt is too long with fewshot examples, reduce the number of
        # examples until it fits.
        while num_fewshot >= 0:
            ctx = super().fewshot_context(doc, num_fewshot, **kwargs)
            if len(self._tokenize(ctx)) <= max_length:
                doc["context"] = ctx
                return ctx
            num_fewshot -= 1

        # if we got here then even 0 fewshot is too long
        return ValueError(
            f"0-shot prompt is too long for max length {max_length}:\n{ctx}"
        )

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(
            ctx, [self.tokenizer.eos_token, self.SEP], self.max_gen_toks
        )

    def _tokenize(self, text, **kwargs):
        encode_fn = self.tokenizer.encode
        if "add_special_tokens" in inspect.getfullargspec(encode_fn).args:
            encode_params = dict(add_special_tokens=False)
        else:
            encode_params = {}
        return encode_fn(text, **encode_params, **kwargs)

    def _extract_answer(self, completion):
        matches = ANS_RE.findall(completion)
        if matches:
            match_str = matches[-1].strip(".")
            match_str = match_str.replace(",", "")
            try:
                match_float = float(match_str)
            except ValueError:
                return INVALID_ANS
            if match_float.is_integer():
                return int(match_float)

        return INVALID_ANS

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        assert (
            len(results) == 1
        ), f"results should be a list with 1 str element, but is {results}"
        completion = results[0]
        extracted_answer = self._extract_answer(completion)
        answer = doc["answer_number"]
        acc = extracted_answer == answer
        out = {"acc": acc}
        out["details"] = {
            "question": doc["question"],
            "context": doc["context"],
            "completion": completion,
            "extracted_answer": extracted_answer,
            "answer": answer,
            "acc": acc,
        }
        return out


class MGSMWithJAAlpacaPrompt(MGSM):
    PROMPT_VERSION = 0.3
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    INSTRUCTION = "与えられた問題に対して、ステップごとに答えを導き出してください。"

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
        input_text = f"{doc['question'].replace('問題：','')}"
        return f"### 指示:\n{self.INSTRUCTION}### 入力:\n{input_text}\n\n### 応答:\n"


class MGSMWithRinnaInstructionSFT(MGSM):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """

    PROMPT_VERSION = 0.4
    FEWSHOT_SEP = "<NL>"
    DESCRIPTION = f"ユーザー: 与えられた問題をステップごとに解説してください。<NL>システム: 分かりました。<NL>"

    def doc_to_text(self, doc):
        input_text = f"問題：{doc['question'].replace('問題：','')}"
        return f"ユーザー: {input_text}<NL>システム: ステップごとの答え："


class MGSMWithRinnaBilingualInstructionSFT(MGSMWithRinnaInstructionSFT):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-sft
    """

    PROMPT_VERSION = 0.5
    DESCRIPTION = f"ユーザー: 与えられた問題をステップごとに解説してください。\nシステム: 分かりました。\n"
    FEWSHOT_SEP = "\n"


class MGSMWithLlama2(MGSMWithJAAlpacaPrompt):
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
        与えられた問題に対して、ステップごとに答えを導き出してください。

        {question} [/INST]
        ```
        """
        input_text = f"{doc['question'].replace('問題：','')}"
        return f"{self.INSTRUCTION}\n\n{input_text} [/INST] "


VERSIONS = [
    MGSM,
    MGSMWithJAAlpacaPrompt,
    MGSMWithRinnaInstructionSFT,
    MGSMWithRinnaBilingualInstructionSFT,
    MGSMWithLlama2,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[
            f"mgsm-{version_class.VERSION}-{version_class.PROMPT_VERSION}"
        ] = version_class
    return tasks
