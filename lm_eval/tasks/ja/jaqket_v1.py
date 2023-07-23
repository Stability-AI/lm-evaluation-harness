"""
JAQKET: JApanese Questions on Knowledge of EnTitie
https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf


Homepage: https://www.nlp.ecei.tohoku.ac.jp/projects/jaqket/
"""
import os
import inspect
import datasets
from lm_eval.base import MultipleChoiceTask, rf
import numpy as np


_CITATION = """
@InProceedings{Kurihara_nlp2020,
  author =  "鈴木正敏 and 鈴木潤 and 松田耕史 and ⻄田京介 and 井之上直也",
  title =   "JAQKET: クイズを題材にした日本語 QA データセットの構築",
  booktitle =   "言語処理学会第26回年次大会",
  year =    "2020",
  url = "https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf"
  note= "in Japanese"}
"""

DYNAMIC_MAX_LENGTH = os.getenv("DYNAMIC_MAX_LENGTH", "true").lower()
TOP_K_LIMIT = 5

class JAQKETV1(MultipleChoiceTask):
    """
    prompt format was inspired by [日本語に特化した60億パラメータ規模のGPTモデルの構築と評価](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H9-4.pdf)
    """
    VERSION = 0.1
    PROMPT_VERSION = 0.1
    DATASET_PATH = "kumapo/JAQKET"
    DATASET_NAME = "v1.0"
    LOAD_TOKENIZER = True
    DESCRIPTION = "[題名]と[問題]から[質問]に対する[答え]を[選択肢]の中から選んでください。\n\n"
    CONTEXT_LIMIT = 128
    ANSWERING_CONTEXT_LIMIT = CONTEXT_LIMIT // 2
    SEP = "\n"
    FEWSHOT_SEP = "\n\n"

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
             num_contexts=TOP_K_LIMIT
         )

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        return {
            "goal": doc["question"],
            "choices": doc["answer_candidates"],
            "gold": doc["label"],
            "contexts": doc["contexts"],
        }

    def batch_truncate_text(self, batch_text, token_limit):
        encode_fn = self.tokenizer.batch_encode_plus
        encode_params = {}
        if "add_special_tokens" in inspect.getfullargspec(encode_fn).args:
            encode_params.update(dict(add_special_tokens=False))
        if "padding" in inspect.getfullargspec(encode_fn).args:
            encode_params.update(dict(padding=False))
        if "truncation" in inspect.getfullargspec(encode_fn).args:
            encode_params.update(dict(truncation=True))
        if "max_length" in inspect.getfullargspec(encode_fn).args:
            encode_params.update(dict(max_length=token_limit))

        batch_encoded = encode_fn(batch_text, **encode_params)
        batch_input_ids = [
            input_ids[:token_limit]
            for input_ids in batch_encoded["input_ids"]
        ]
        decode_fn = self.tokenizer.batch_decode
        if "skip_special_tokens" in inspect.getfullargspec(decode_fn).args:
            decode_params = dict(skip_special_tokens=True)
        else:
            decode_params = {}
        truncated = decode_fn(batch_input_ids, **decode_params)
        return truncated

    def doc_to_qa_prompt(self, doc):
        """
        [問題]:question
        [選択肢]:[choice0, choice1, ..., choice4]
        [答え]:
        """
        return (
            f"[質問]:{doc['goal']}\n" + 
            f"[選択肢]:[{', '.join(doc['choices'])}]\n"
            "[答え]:"
        )

    def doc_to_text(self, doc):
        truncated_contexts = [
            context
            for context in self.batch_truncate_text(doc["contexts"], self.CONTEXT_LIMIT)
        ]
        answer_context = "\n".join([
            (
                f"[題名]:{choice}\n" +
                f"[問題]:{context}"
            )
            for choice, context in zip(doc["choices"], truncated_contexts)
        ])
        qa_prompt = self.doc_to_qa_prompt(doc)
        return (
            answer_context + 
            "\n" + 
            qa_prompt
        )

    def doc_to_answering_text(self, doc):
        choices_and_contexts = []
        for choice, context in zip(doc["choices"], doc["contexts"]):
            if doc["gold"] == choice:
                # need gold choice
                choices_and_contexts.append((choice,context))
            elif len(choices_and_contexts) < 2:
                # and wrong choice
                choices_and_contexts.append((choice,context))
            if 1 < len(choices_and_contexts):
                # 1 gold and 1 wrong are enough
                break
        doc["choices"] = [tup[0] for tup in choices_and_contexts]
        doc["contexts"] = self.batch_truncate_text([
            tup[1] for tup in choices_and_contexts
        ], self.ANSWERING_CONTEXT_LIMIT)
        answer_context = "\n".join([
            (
                f"[題名]:{choice}\n" +
                f"[問題]:{context}"
            )
            for choice, context in zip(doc["choices"], doc["contexts"])])
        qa_prompt = self.doc_to_qa_prompt(doc)
        return (
            answer_context + 
            "\n" + 
            qa_prompt
        )

    def doc_to_target(self, doc):
        return doc["choices"][doc["gold"]]

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

        # if there is no example and still the prompt is too long, fail
        if len(ctxs) < 2:
            raise ValueError(f"0-shot description+example doesn't fit in max length. ctx: {ctx}")

        # delete the first example, last is questioning example
        del ctxs[0]

        # recurse
        return self.preprocess_ctx(self.FEWSHOT_SEP.join([description, *ctxs]), max_length)

    def construct_requests(self, doc, ctx):
        if DYNAMIC_MAX_LENGTH == "false" or not hasattr(self.tokenizer, "encode"):
            lls = [
                rf.loglikelihood(ctx, " {}".format(choice))[0]
                for choice in doc["choices"]
            ]
        else:
            encode_fn = self.tokenizer.encode
            if "add_special_tokens" in inspect.getfullargspec(encode_fn).args:
                encode_params = dict(add_special_tokens=False)
            else:
                encode_params = {}
            max_num_tokens = max([
                len(encode_fn(choice, **encode_params))
                for choice in doc["choices"]
            ])
            ctx = self.preprocess_ctx(ctx, max_length=self.max_length-max_num_tokens)
            lls = [
                rf.loglikelihood(ctx, " {}".format(choice))[0]
                for choice in doc["choices"]
            ]
        return lls

    def process_results(self, doc, results):
        gold = doc["gold"]

        response = np.argmax(results)
        acc = 1.0 if response == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        out = {
            "acc": acc,
            "acc_norm": acc_norm,
        }
        # only include details if we were wrong
        if acc == 0.0:
            # without the cast it won't serialize
            response = int(response)
            out["details"] = {
                "question": doc["goal"],
                "choices": doc["choices"],
                "gold": doc["gold"],
                "response": response,
            }
        return out

class JAQKETV1WithFintanPrompt(JAQKETV1):
    """
    prompt template was inspired by [ChatGPT vs BERT: どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
    """
    VERSION = 0.1
    PROMPT_VERSION = 0.2
    DESCRIPTION = "文章と質問と回答の選択肢を入力として受け取り、選択肢から質問に対する回答を選択してください。なお、回答は選択肢の番号(例:0)でするものとします。 \n\n"

    def doc_to_qa_prompt(self, doc):
        """
        質問:question
        選択肢:0.choice0,1.choice1, ...,4.choice4
        回答:
        """
        choices = ",".join([f"{idx}.{choice}" for idx, choice in enumerate(doc['choices'])])
        return (
            f"質問:{doc['goal']}\n"
            f"選択肢:{choices}\n"
            "回答:"
        )

    def doc_to_text(self, doc):
        combined_context = "\n".join([
            context
            for context in self.batch_truncate_text(doc["contexts"], self.CONTEXT_LIMIT)
        ])
        answer_context = f"文章:{combined_context}"
        qa_prompt = self.doc_to_qa_prompt(doc)
        text = (
            answer_context + 
            "\n" + 
            qa_prompt
        )
        return text

    def doc_to_answering_text(self, doc):
        choices_and_contexts = []
        for choice, context in zip(doc["choices"], doc["contexts"]):
            if doc["gold"] == choice:
                # need gold choice
                choices_and_contexts.append((choice,context))
            elif len(choices_and_contexts) < 2:
                # and wrong choice
                choices_and_contexts.append((choice,context))
            if 1 < len(choices_and_contexts):
                # 1 gold and 1 wrong are enough
                break
        doc["choices"] = [tup[0] for tup in choices_and_contexts]
        doc["contexts"] = [tup[1] for tup in choices_and_contexts]
        combined_context = "\n".join([
            context
            for context in self.batch_truncate_text(doc["contexts"], self.ANSWERING_CONTEXT_LIMIT)
        ])
        answer_context = f"文章:{combined_context}"
        qa_prompt = self.doc_to_qa_prompt(doc)
        text = (
            answer_context + 
            "\n" + 
            qa_prompt
        )
        return text
    
    def doc_to_target(self, doc):
        return f"{doc['gold']}"
    
class JAQKETV1WithJAAlpacaPrompt(JAQKETV1):
    """
    This prompt format was inspired by the below data in fujiki/japanese_alpaca_data. 
    ```
    {
        'instruction': 'この課題では、以下の選択肢から文の出典を特定する必要があります。\n\n出力は以下から選択してください：\n- 新聞\n- 教科書\n- オンライン記事\n- 百科事典', 
        'input': '彼はローマの政治家であり哲学者であり、史上最も偉大な軍事指導者の一人と考えられています。', 
        'output': '百科事典'
    }
    ```
    Reference:
    - data: https://huggingface.co/datasets/fujiki/japanese_alpaca_data
    - code: https://github.com/Stability-AI/gpt-neox/blob/c130a4edc1120dccec8f02a34eb60d3e8f484cd3/finetune/finetune_base_ja.py#LL118C23-L127C11
    """
    VERSION = 0.1
    PROMPT_VERSION = 0.3
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    INSTRUCTION = "与えられた文脈と選択肢の中から、質問に対する答えを選んでください。"

    def doc_to_qa_prompt(self, doc):
        raise NotImplementedError()

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
        choices = "\n".join([f"- {choice}" for choice in doc['choices']])
        instruction_text = self.INSTRUCTION + f"出力は以下から選択してください：\n{choices}"
        combined_context = "\n".join([
            context
            for context in self.batch_truncate_text(doc["contexts"], self.CONTEXT_LIMIT)
        ])
        input_text = f"文脈：{combined_context}\n質問：{doc['goal']}"
        return (
            f"### 指示:\n{instruction_text}\n\n"
            f"### 入力:\n{input_text}\n\n"
            f"### 応答:\n"
        )

    def doc_to_answering_text(self, doc):
        choices_and_contexts = []
        for choice, context in zip(doc["choices"], doc["contexts"]):
            if doc["gold"] == choice:
                # need gold choice
                choices_and_contexts.append((choice,context))
            elif len(choices_and_contexts) < 2:
                # and wrong choice
                choices_and_contexts.append((choice,context))
            if 1 < len(choices_and_contexts):
                # 1 gold and 1 wrong are enough
                break
        doc["choices"] = [tup[0] for tup in choices_and_contexts]
        doc["contexts"] = [tup[1] for tup in choices_and_contexts]
        choices = "\n".join([f"- {choice}" for choice in doc['choices']])
        instruction_text = self.INSTRUCTION + f"出力は以下から選択してください：\n{choices}"
        combined_context = "\n".join([
            context
            for context in self.batch_truncate_text(doc["contexts"], self.ANSWERING_CONTEXT_LIMIT)
        ])
        input_text = f"文脈：{combined_context}\n質問：{doc['goal']}"
        return (
            f"### 指示:\n{instruction_text}\n\n"
            f"### 入力:\n{input_text}\n\n"
            f"### 応答:\n"
        )

class JAQKETV1WithRinnaInstructionSFT(JAQKETV1):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """
    VERSION = 0.1
    PROMPT_VERSION = 0.4
    DESCRIPTION = "ユーザー: 与えられた文脈と選択肢から、質問に対する答えを選択肢の中から選んでください。<NL>システム: 分かりました。<NL>"
    SEP = "<NL>"
    FEWSHOT_SEP = "<NL>"
    END_OF_DESCRIPTION = "システム: 分かりました。<NL>"
    START_OF_FEWSHOT = "ユーザー: 文脈："

    def doc_to_qa_prompt(self, doc):
        raise NotImplementedError()

    def doc_to_text(self, doc):
        choices = self.SEP.join([f"- {choice}" for choice in doc['choices']])
        combined_context = self.SEP.join([
            context
            for context in self.batch_truncate_text(doc["contexts"], self.CONTEXT_LIMIT)
        ])
        input_text = f"文脈：{combined_context}{self.SEP}質問：{doc['goal']}{self.SEP}" + f"選択肢：{self.SEP}{choices}"
        return f"ユーザー: {input_text}{self.SEP}システム: "

    def doc_to_answering_text(self, doc):
        choices_and_contexts = []
        for choice, context in zip(doc["choices"], doc["contexts"]):
            if doc["gold"] == choice:
                # need gold choice
                choices_and_contexts.append((choice,context))
            elif len(choices_and_contexts) < 2:
                # and wrong choice
                choices_and_contexts.append((choice,context))
            if 1 < len(choices_and_contexts):
                # 1 gold and 1 wrong are enough
                break
        doc["choices"] = [tup[0] for tup in choices_and_contexts]
        doc["contexts"] = [tup[1] for tup in choices_and_contexts]
        choices = self.SEP.join([f"- {choice}" for choice in doc['choices']])
        combined_context = self.SEP.join([
            context
            for context in self.batch_truncate_text(doc["contexts"], self.ANSWERING_CONTEXT_LIMIT)
        ])
        input_text = f"文脈：{combined_context}{self.SEP}質問：{doc['goal']}{self.SEP}" + f"選択肢：{self.SEP}{choices}"
        return f"ユーザー: {input_text}{self.SEP}システム: "

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
    JAQKETV1,
    JAQKETV1WithFintanPrompt,
    JAQKETV1WithJAAlpacaPrompt,
    JAQKETV1WithRinnaInstructionSFT,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[f"jaqket_v1-{version_class.VERSION}-{version_class.PROMPT_VERSION}"] = version_class
    return tasks
