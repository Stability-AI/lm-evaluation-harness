import os
import numpy as np
import transformers
from lm_eval.base import BaseLM
from lm_eval import utils
from tqdm import tqdm
import time


def get_result(choice, ctxlen):
    is_greedy = True
    logprobs = choice.logprobs.token_logprobs
    continuation_logprobs = sum(logprobs[ctxlen:])

    for i in range(ctxlen, len(choice.logprobs.tokens)):
        token = choice.logprobs.tokens[i]
        top_tokens = choice.logprobs.top_logprobs[i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


def fw_completion(**kwargs):
    """Query Fireworks API for completion.
    Retry with back-off until they respond
    """
    import fireworks.client

    backoff_time = 3
    while True:
        try:
            return fireworks.client.Completion.create(**kwargs)
        except fireworks.client.error.FireworksError:
            import traceback

            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5


class FireworksLM(BaseLM):
    def __init__(self, model, tokenizer, base_url = None, num_reqs = 16, max_length=4096):
        """

        :param engine: str
            Fireworks API engine (e.g. davinci)
        """
        super().__init__()

        import fireworks.client

        self.model = model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)

        fireworks.client.api_key = os.environ.get("FIREWORKS_API_KEY", "NONE")
        if base_url is not None:
            fireworks.client.base_url = base_url
        self.num_reqs = num_reqs
        self._max_length = max_length

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []

        def _collate(x):
            # this doesn't efficiently handle last-token differences yet, but those are kinda annoying because
            # it's not guaranteed that the 100 or so logprobs we get to see actually contain all the continuations
            # we care about and so we need some kind of backup for when it isn't
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        for chunk in tqdm(
            list(utils.chunks(re_ord.get_reordered(), self.num_reqs)),
            disable=disable_tqdm,
        ):
            inps = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                # max_length+1 because the API takes up to 2049 tokens, including the first context token
                inp = (context_enc + continuation_enc)[-(self.max_length + 1) :]
                # TODO: the logic is much simpler if we just look at the length of continuation tokens
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length + 1)
                )

                inps.append(inp)
                ctxlens.append(ctxlen)

            response = fw_completion(
                model=self.model,
                prompt=inps,
                echo=True,
                max_tokens=0,
                temperature=0.0,
                logprobs=5,
            )

            for resp, ctxlen, (cache_key, context_enc, continuation_enc) in zip(
                response.choices, ctxlens, chunk
            ):
                answer = get_result(resp, ctxlen)

                res.append(answer)

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        return re_ord.get_original(res)

    def greedy_until(self, requests):
        if not requests:
            return []
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = xs[0][1]
            for x in xs:
                if len(ret) >= size or x[1] != lastuntil:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = x[1]
                ret.append(x)

            if ret:
                yield ret, lastuntil

        # todo: more intelligent batching for heterogeneous `until`
        for chunk, until in tqdm(
            list(sameuntil_chunks(re_ord.get_reordered(), self.num_reqs))
        ):
            inps = []
            for request in chunk:
                context = request[0]
                context_enc = self.tok_encode(context)
                inp = context_enc[-(self.max_length - self.max_gen_toks) :]
                inps.append(inp)

            response = fw_completion(
                model=self.model,
                prompt=inps,
                max_tokens=self.max_gen_toks,
                temperature=0.0,
                logprobs=5,
                stop=until,
            )

            for choice, request in zip(response.choices, chunk):
                context = request[0]
                until_ = request[1]
                s = choice.text

                for term in until_:
                    s = s.split(term)[0]

                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until_), s)

                res.append(s)

        return re_ord.get_original(res)

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
