"""Prompt and answer tokenisation at the answer boundary (Appendix A.2, "Scoring conventions").

A prompt encoder turns the four prompt fields into the token sequence that ends exactly at
the answer boundary, and turns a candidate output string into its answer tokens terminated by
the model's fixed end-of-answer token.  Only those answer tokens contribute to a score.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from sigma.guidance.render import PromptFields

_WORD = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")


class PromptEncoder(Protocol):
    """Interface between rendered prompts / candidate strings and token identifiers."""

    pad_token_id: int

    def encode_prompt(self, fields: PromptFields) -> list[int]:
        """Token ids of the prompt up to (and including) the answer boundary."""

    def encode_answer(self, answer: str) -> list[int]:
        """Token ids of a candidate answer followed by the end-of-answer token."""


class WordTokenizer:
    """A deterministic word-level tokenizer built in code (no download).

    Words are maximal alphanumeric runs; every other non-space character is its own token.
    It is used with the tiny randomly initialised language model of the quick start and of
    the test-suite.
    """

    PAD, BOS, ANSWER, EOA, UNK = "<pad>", "<bos>", "<answer>", "<eoa>", "<unk>"
    SPECIALS = (PAD, BOS, ANSWER, EOA, UNK)

    def __init__(self, vocabulary: Iterable[str]) -> None:
        words = sorted(set(vocabulary) - set(self.SPECIALS))
        self.id_to_token: list[str] = [*self.SPECIALS, *words]
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.id_to_token)}

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> WordTokenizer:
        """Build the vocabulary from every word occurring in ``texts``."""
        vocabulary: set[str] = set()
        for text in texts:
            vocabulary.update(cls.split(text))
        return cls(vocabulary)

    @staticmethod
    def split(text: str) -> list[str]:
        return _WORD.findall(text)

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        unknown = self.token_to_id[self.UNK]
        return [self.token_to_id.get(word, unknown) for word in self.split(text)]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.id_to_token[i] for i in ids)


class WordPromptEncoder:
    """Prompt encoder for :class:`WordTokenizer`: ``<bos> prompt <answer>`` / ``answer <eoa>``."""

    def __init__(self, tokenizer: WordTokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.token_to_id[WordTokenizer.PAD]
        self._bos = tokenizer.token_to_id[WordTokenizer.BOS]
        self._answer = tokenizer.token_to_id[WordTokenizer.ANSWER]
        self._eoa = tokenizer.token_to_id[WordTokenizer.EOA]

    def encode_prompt(self, fields: PromptFields) -> list[int]:
        return [self._bos, *self.tokenizer.encode(fields.as_text()), self._answer]

    def encode_answer(self, answer: str) -> list[int]:
        return [*self.tokenizer.encode(answer), self._eoa]


class ChatPromptEncoder:
    """Prompt encoder for Hugging Face chat models (Qwen2.5 / Llama-3 / Qwen3 Instruct).

    The four prompt fields form one user message; the answer boundary is the position right
    after the chat template's generation prompt.  A candidate is tokenised on its own (as the
    assistant turn starts there) and terminated by the end-of-turn token of the template,
    which plays the role of the "fixed end-of-answer token" of Appendix A.2.

    Parameters
    ----------
    tokenizer:
        A ``transformers`` tokenizer.
    end_of_answer:
        ``"auto"`` infers the end-of-turn token from the chat template (``<|im_end|>`` for
        Qwen, ``<|eot_id|>`` for Llama-3); otherwise the literal token string to use.
    template_kwargs:
        Extra arguments for ``apply_chat_template`` (e.g. ``{"enable_thinking": False}`` for
        the non-thinking mode of Qwen3, Appendix C.5).
    """

    _SENTINEL = "SIGMA-ANSWER-SENTINEL"

    def __init__(
        self,
        tokenizer: Any,
        end_of_answer: str = "auto",
        template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.template_kwargs = dict(template_kwargs or {})
        self._has_template = getattr(tokenizer, "chat_template", None) is not None
        pad = tokenizer.pad_token_id
        self.pad_token_id = int(pad if pad is not None else tokenizer.eos_token_id)
        self.end_of_answer_id = self._resolve_end_of_answer(end_of_answer)

    def _ids(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _resolve_end_of_answer(self, end_of_answer: str) -> int:
        if end_of_answer != "auto":
            token_id = self.tokenizer.convert_tokens_to_ids(end_of_answer)
            if token_id is None or token_id == self.tokenizer.unk_token_id:
                raise ValueError(f"unknown end-of-answer token {end_of_answer!r}")
            return int(token_id)
        if self._has_template:
            rendered = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": self._SENTINEL},
                ],
                tokenize=False,
                add_generation_prompt=False,
                **self.template_kwargs,
            )
            suffix = rendered.split(self._SENTINEL, 1)[1]
            suffix_ids = self._ids(suffix)
            if suffix_ids:
                return int(suffix_ids[0])
        if self.tokenizer.eos_token_id is None:
            raise ValueError("the tokenizer defines neither a chat template nor an EOS token")
        return int(self.tokenizer.eos_token_id)

    def encode_prompt(self, fields: PromptFields) -> list[int]:
        if self._has_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": fields.as_text()}],
                tokenize=False,
                add_generation_prompt=True,
                **self.template_kwargs,
            )
            return self._ids(text)
        ids = self._ids(fields.as_text() + "\n\nAnswer:\n")
        bos = self.tokenizer.bos_token_id
        return ids if bos is None else [int(bos), *ids]

    def encode_answer(self, answer: str) -> list[int]:
        return [*self._ids(answer), self.end_of_answer_id]
