"""Candidate scoring with a causal LM (Eq. (3), Appendix A.2)."""

from __future__ import annotations

import pytest
import torch

from sigma.guidance import BindingSwap, Reversal, apply_edit
from sigma.guidance.render import PromptFields
from sigma.scoring import (
    CandidateScorer,
    CandidateTokens,
    ChatPromptEncoder,
    ContextLimitError,
    WordTokenizer,
)


@pytest.fixture
def views(toy_task, tiny_scorer):
    """Three prompts of different lengths: base, a binding swap and the reversal."""
    base = toy_task.base_guidance()
    swapped = apply_edit(base, BindingSwap("billing", "refund", "invoice")).guidance
    reversed_ = apply_edit(base, Reversal()).guidance
    return [
        tiny_scorer.encode_view(base, "please give my money back now"),
        tiny_scorer.encode_view(swapped, "i cannot log in"),
        tiny_scorer.encode_view(reversed_, "money"),
    ]


@pytest.fixture
def candidates(toy_task, tiny_scorer):
    return tiny_scorer.encode_candidates(toy_task.candidates)


def _manual_score(scorer: CandidateScorer, prompt: list[int], answer: tuple[int, ...]) -> float:
    """Mean log-probability of the answer tokens (incl. end-of-answer) via a plain forward."""
    ids = torch.tensor([prompt + list(answer)])
    logits = scorer.model(input_ids=ids).logits[0].double()
    log_probs = torch.log_softmax(logits, dim=-1)
    start = len(prompt)
    token_scores = [log_probs[start - 1 + t, token] for t, token in enumerate(answer)]
    return float(torch.stack(token_scores).mean())


@torch.no_grad()
def test_score_is_the_mean_answer_token_log_probability(tiny_scorer, views, candidates):
    scores = tiny_scorer.score(views, candidates)
    assert scores.shape == (3, len(candidates))
    for b, prompt in enumerate(views):
        for j, answer in enumerate(candidates.token_ids):
            assert scores[b, j].item() == pytest.approx(
                _manual_score(tiny_scorer, prompt, answer), abs=2e-5
            )


def test_answers_end_with_the_end_of_answer_token(tiny_scorer, candidates, toy_task):
    eoa = tiny_scorer.encoder.tokenizer.token_to_id[WordTokenizer.EOA]
    assert all(ids[-1] == eoa for ids in candidates.token_ids)
    # "desk red" -> two word tokens plus end-of-answer; "fallback" -> one plus end-of-answer.
    lengths = dict(zip(toy_task.candidates, map(len, candidates.token_ids), strict=True))
    assert lengths["desk red"] == 3 and lengths["fallback"] == 2


@torch.no_grad()
def test_packed_and_row_strategies_agree(tiny_scorer, views, candidates):
    rows = CandidateScorer(tiny_scorer.model, tiny_scorer.encoder, strategy="rows")
    assert torch.allclose(
        tiny_scorer.score(views, candidates), rows.score(views, candidates), atol=2e-5
    )


@torch.no_grad()
def test_batching_and_padding_do_not_change_scores(tiny_scorer, views, candidates):
    together = tiny_scorer.score(views, candidates)  # one padded batch of three prompts
    one_by_one = CandidateScorer(tiny_scorer.model, tiny_scorer.encoder, view_batch_size=1)
    separate = torch.cat([one_by_one.score([v], candidates) for v in views])
    assert torch.allclose(together, separate, atol=2e-5)
    reordered = tiny_scorer.score(views[::-1], candidates)
    assert torch.allclose(reordered, together.flip(0), atol=2e-5)


@torch.no_grad()
def test_candidate_microbatching_and_subsets_do_not_change_scores(tiny_scorer, views, candidates):
    reference = tiny_scorer.score(views, candidates)
    for chunk in (1, 2, 4):
        chunked = CandidateScorer(
            tiny_scorer.model, tiny_scorer.encoder, view_batch_size=2, candidate_chunk_size=chunk
        )
        assert torch.allclose(chunked.score(views, candidates), reference, atol=2e-5)
    subset = [4, 0, 2]
    assert torch.allclose(
        tiny_scorer.score(views, candidates, subset), reference[:, subset], atol=2e-5
    )


def test_gradients_agree_between_strategies(tiny_scorer, views, candidates):
    model = tiny_scorer.model
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    weights = torch.linspace(-1.0, 1.0, 3 * len(candidates)).view(3, -1)

    def gradient(scorer: CandidateScorer) -> torch.Tensor:
        model.zero_grad()
        (scorer.score(views, candidates) * weights).sum().backward()
        return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    rows = CandidateScorer(model, tiny_scorer.encoder, strategy="rows")
    assert torch.allclose(gradient(tiny_scorer), gradient(rows), atol=1e-4)


def test_candidates_need_distinct_nonempty_token_sequences(tiny_scorer):
    with pytest.raises(ValueError, match="distinct"):
        tiny_scorer.encode_candidates(["desk red", "desk  red"])  # same tokens
    with pytest.raises(ValueError, match="at least one token"):
        CandidateTokens(((7,),))


@torch.no_grad()
def test_context_limit_is_enforced(tiny_scorer, views, candidates):
    limited = CandidateScorer(
        tiny_scorer.model, tiny_scorer.encoder, max_context_tokens=len(views[0]) + 1
    )
    limited.score([views[2]], candidates)  # the shortest prompt still fits
    with pytest.raises(ContextLimitError):
        limited.score([views[0]], candidates)


@torch.no_grad()
def test_processed_token_accounting(tiny_scorer, views, candidates):
    tiny_scorer.stats.reset()
    tiny_scorer.score(views, candidates)
    block = sum(len(ids) - 1 for ids in candidates.token_ids)
    assert tiny_scorer.stats.processed_tokens == sum(len(v) for v in views) + 3 * block
    assert tiny_scorer.stats.scored_views == 3 and tiny_scorer.stats.forward_calls == 1


def test_word_tokenizer_round_trip():
    tokenizer = WordTokenizer.from_texts(['- class "refund": "money back" => "desk red"'])
    ids = tokenizer.encode('class "refund" => unknown-word')
    assert tokenizer.decode(ids) == 'class " refund " = > <unk> - <unk>'
    assert len(tokenizer) == len(set(tokenizer.id_to_token))


class _FakeChatTokenizer:
    """Minimal stand-in for a Hugging Face chat tokenizer (character-level)."""

    chat_template = "fake"
    pad_token_id = None
    eos_token_id = 1
    bos_token_id = None
    unk_token_id = 0
    specials = {"<|start|>": 2, "<|end|>": 3}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        text = "".join(f"<|start|>{m['role']}\n{m['content']}<|end|>\n" for m in messages)
        return text + ("<|start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        ids, position = [], 0
        while position < len(text):
            for token, token_id in self.specials.items():
                if text.startswith(token, position):
                    ids.append(token_id)
                    position += len(token)
                    break
            else:
                ids.append(10 + ord(text[position]))
                position += 1
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token):
        return self.specials.get(token, self.unk_token_id)


def test_chat_encoder_finds_the_answer_boundary_and_end_of_turn_token():
    tokenizer = _FakeChatTokenizer()
    encoder = ChatPromptEncoder(tokenizer)
    assert encoder.end_of_answer_id == 3  # inferred from the template: <|end|>
    assert encoder.pad_token_id == tokenizer.eos_token_id
    fields = PromptFields("Do it.", '- group "root": "r"', "hello", ("a", "b"))
    prompt = encoder.encode_prompt(fields)
    generation_prompt = tokenizer("<|start|>assistant\n")["input_ids"]
    assert prompt[-len(generation_prompt) :] == generation_prompt
    assert encoder.encode_answer("ab") == [10 + ord("a"), 10 + ord("b"), 3]
    explicit = ChatPromptEncoder(tokenizer, end_of_answer="<|start|>")
    assert explicit.end_of_answer_id == 2
    with pytest.raises(ValueError):
        ChatPromptEncoder(tokenizer, end_of_answer="<|missing|>")
