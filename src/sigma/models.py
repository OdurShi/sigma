"""Backbones and low-rank adaptation (Section 4.3, Appendix C.5, Appendix D.9).

Only LoRA parameters are optimised; the pretrained weights stay frozen.  The paper's adapter
uses rank 16, scaling 32 and zero dropout on the attention query/key/value/output projections
and the MLP gate/up/down projections.
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from sigma.scoring.encoders import (
    ChatPromptEncoder,
    PromptEncoder,
    WordPromptEncoder,
    WordTokenizer,
)

logger = logging.getLogger(__name__)

#: Projections adapted in the paper, in the order used by the knockout study (Appendix D.9).
PAPER_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: Hugging Face identifiers of the three backbones (Appendix C.5).
PAPER_BACKBONES: dict[str, str] = {
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
}

TINY_BACKBONE = "tiny-random"


@dataclass(frozen=True)
class LoraSettings:
    """LoRA configuration; the defaults are the paper's values (Appendix C.5)."""

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = PAPER_TARGET_MODULES


@dataclass(frozen=True)
class TinyBackboneSettings:
    """Size of the randomly initialised Qwen2-style model used for the demo and the tests."""

    hidden_size: int = 96
    intermediate_size: int = 192
    num_layers: int = 2
    num_heads: int = 4
    max_positions: int = 2048


def build_tiny_backbone(
    vocab_size: int, seed: int, settings: TinyBackboneSettings | None = None
) -> torch.nn.Module:
    """Build a small Qwen2-architecture causal LM with random weights (no download).

    It has the same module names as the paper's backbone (``q_proj`` ... ``down_proj``), so
    the LoRA configuration, the candidate scorer and the knockout utility run unchanged.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    settings = settings or TinyBackboneSettings()
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=settings.hidden_size,
        intermediate_size=settings.intermediate_size,
        num_hidden_layers=settings.num_layers,
        num_attention_heads=settings.num_heads,
        num_key_value_heads=settings.num_heads,
        max_position_embeddings=settings.max_positions,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(config)
    torch.random.set_rng_state(generator_state)
    return model.eval()


def load_backbone(
    name: str,
    dtype: str = "bfloat16",
    device: str | None = None,
    gradient_checkpointing: bool = False,
    end_of_answer: str = "auto",
    template_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.nn.Module, PromptEncoder]:
    """Load a pretrained causal LM and its chat prompt encoder from the Hugging Face hub."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype))
    except TypeError:  # older transformers releases only know the former keyword
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype))
    if device is not None:
        model = model.to(device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.eval()
    return model, ChatPromptEncoder(tokenizer, end_of_answer, template_kwargs)


def tiny_encoder(texts: list[str]) -> WordPromptEncoder:
    """Word-level prompt encoder whose vocabulary covers ``texts``."""
    return WordPromptEncoder(WordTokenizer.from_texts(texts))


# --------------------------------------------------------------------------- adaptation
def apply_lora(model: torch.nn.Module, settings: LoraSettings, seed: int) -> torch.nn.Module:
    """Freeze ``model`` and attach LoRA adapters to the configured projections."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    adapted = get_peft_model(model, config)
    torch.random.set_rng_state(generator_state)
    return adapted.eval()


def make_all_trainable(model: torch.nn.Module) -> torch.nn.Module:
    """Train every weight (used only for the from-scratch tiny model of the quick start)."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model.eval()


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in trainable_parameters(model))


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """CPU copy of all trainable parameters (a checkpoint of the adapter)."""
    return {
        name: parameter.detach().to("cpu", copy=True)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Restore a checkpoint produced by :func:`trainable_state`."""
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))


_LORA_B = re.compile(r"layers\.(\d+)\..*?\.(\w+)\.lora_B\.[^.]+\.weight$")


def lora_updates(model: torch.nn.Module) -> dict[tuple[int, str], torch.nn.Parameter]:
    """Map ``(layer index, projection name)`` to the ``B`` factor of its LoRA update."""
    updates: dict[tuple[int, str], torch.nn.Parameter] = {}
    for name, parameter in model.named_parameters():
        match = _LORA_B.search(name)
        if match is not None:
            updates[(int(match.group(1)), match.group(2))] = parameter
    return updates


@contextlib.contextmanager
def lora_knockout(model: torch.nn.Module, layer: int, projection: str) -> Iterator[None]:
    """Temporarily remove one low-rank update, ``W_eff = W0`` for that projection (Eq. (D.6)).

    The pretrained projection and all other LoRA updates are unchanged, and nothing is
    retrained.  The update is restored when the context exits.
    """
    factor = lora_updates(model)[(layer, projection)]
    saved = factor.detach().clone()
    with torch.no_grad():
        factor.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            factor.copy_(saved)
