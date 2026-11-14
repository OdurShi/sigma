"""Shared fixtures: small task records and a tiny randomly initialised language model."""

from __future__ import annotations

import pytest
import torch

from sigma.guidance import TaskRecord, grouped_task
from sigma.models import build_tiny_backbone, tiny_encoder
from sigma.scoring import CandidateScorer

torch.set_num_threads(2)


@pytest.fixture
def toy_task() -> TaskRecord:
    """Root -> {billing (3 leaves), account (2 leaves), misc (1 leaf)}; outputs differ from ids."""
    return grouped_task(
        "toy",
        {
            "billing": (
                "Billing problems.",
                {
                    "refund": "wants money back",
                    "invoice": "asks for an invoice",
                    "charge": "disputes a charge",
                },
            ),
            "account": (
                "Account problems.",
                {"login": "cannot log in", "closure": "wants to close the account"},
            ),
            "misc": ("Everything else.", {"other": "anything else"}),
        },
        binding={
            "refund": "desk red",
            "invoice": "desk blue",
            "charge": "desk green",
            "login": "team alpha",
            "closure": "team beta",
            "other": "fallback",
        },
    )


@pytest.fixture
def tiny_scorer(toy_task: TaskRecord) -> CandidateScorer:
    """Packed scorer over a 2-layer random Qwen2-style model with a word-level tokenizer."""
    from sigma.guidance import render_prompt

    texts = [
        render_prompt(toy_task.base_guidance(), "please give my money back now").as_text(),
        "i cannot log in today",
        *toy_task.candidates,
        *toy_task.labels,
    ]
    encoder = tiny_encoder(texts)
    model = build_tiny_backbone(len(encoder.tokenizer), seed=0)
    return CandidateScorer(model, encoder, strategy="packed", view_batch_size=4)
