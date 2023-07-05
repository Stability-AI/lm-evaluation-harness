"""
Language Models are Multilingual Chain-of-Thought Reasoners
https://arxiv.org/pdf/2210.03057.pdf

Multilingual Grade School Math problems with a numerical answer and a chain-of-thought prompt.
"""
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
        return  f"問題：{doc['question'].replace('問題：','')}{self.SEP}ステップごとの答え："

    def doc_to_target(self, doc):
        # ステップごとの答え： is in text instead of target
        # so that the model doesn't have to generate it
        return "" + doc["answer"].replace('ステップごとの答え：','')
  
    def construct_requests(self, doc, ctx):
        # trim n-shot examples from the context until it fits in max length
        ctx = self.preprocess_ctx(ctx, max_length=self.max_length-self.max_gen_toks)

        # add this trimmed context to the doc for logging
        doc['context'] = ctx

        return rf.greedy_until(ctx, [self.tokenizer.eos_token, self.SEP], self.max_gen_toks)

    def preprocess_ctx(self, ctx, max_length, question_tag="問題："):
        """Remove n-shot examples from the context until it fits in max length

        Args:
            ctx (str): the prompt and n-shot examples 
            max_length (int): the max number of tokens allocated for the prompt and examples
            question_tag (str, optional): A string that occurs before every n-shot example 
                and before the final question. Used to split the context. It is ok to if a common string like
                User: occurs before this tag.
                Defaults to "問題：".

        Raises:
            ValueError: If the prompt doesn't fit in max length at 0-shot

        Returns:
            str: context which fits in max length
        """

        # if ctx fits in max length, return
        if len(self._tokenize(ctx)) <= max_length:
            return ctx

        # if ctx is too long, split on a tag that separates each example
        ctxs = ctx.split(question_tag)

        # if there is no example and still the prompt is too long, fail
        if len(ctxs) < 2:
            raise ValueError(f"0-shot description+question doesn't fit in max length. ctx: {ctx}")
        
        # delete the first example
        del ctxs[1]

        # recurse
        return self.preprocess_ctx(question_tag.join(ctxs), max_length, question_tag)

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
            match_str = matches[-1].strip('.')
            match_str = match_str.replace(",", "")
            match_float = float(match_str)
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
        assert len(results) == 1, f"results should be a list with 1 str element, but is {results}"
        completion = results[0]
        extracted_answer = self._extract_answer(completion)
        answer = doc["answer_number"]
        acc = extracted_answer == answer
        out = {"acc": acc}
        out["details"] = {"question": doc["question"], "context": doc["context"], 
                          "completion": completion, "extracted_answer": extracted_answer,
                          "answer": answer, "acc": acc}
        return out

class MGSMWithJAAlpacaPrompt(MGSM):
    PROMPT_VERSION = 0.3
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    INSTRUCTION = "与えられた問題に対して、ステップごとに答えを導き出してください。\n\n"
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

VERSIONS = [
    MGSM,
    MGSMWithJAAlpacaPrompt,
    MGSMWithRinnaInstructionSFT
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[f"mgsm-{version_class.VERSION}-{version_class.PROMPT_VERSION}"] = version_class
    return tasks