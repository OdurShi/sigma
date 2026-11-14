"""Inference: scoring the candidate inventory under the current guidance (Section 4.3).

Two predictors expose the same interface:

* :class:`DirectPredictor` renders the current guidance into the prompt and scores every
  candidate output with the (adapted) language model.  It is used by frozen prompting,
  LoRA-SFT, paired SFT, the transported-KL control and SIGMA.
* :class:`FactorizedPredictor` implements SIGMA-F (Appendix A.3): a semantic classifier
  ``r(z | x)`` evaluated under a canonical guidance representation, followed by exact binding
  lookup ``q(c | x, g) = r(a_g^{-1}(c) | x)`` (Eq. (A.3)).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch

from sigma.guidance.schema import Guidance, TaskRecord, with_binding
from sigma.scoring.scorer import CandidateScorer


class Predictor(Protocol):
    """Maps ``(input, guidance)`` pairs to candidate-score vectors in candidate order."""

    tie_by_label: bool

    def score_views(self, views: Sequence[tuple[str, Guidance]]) -> np.ndarray:
        """Return an array of shape ``(len(views), m)``."""


class DirectPredictor:
    """Scores output strings directly from the guidance text (the SIGMA interface)."""

    tie_by_label = False

    def __init__(self, scorer: CandidateScorer, task: TaskRecord) -> None:
        self.scorer = scorer
        self.task = task
        self.candidates = scorer.encode_candidates(task.candidates)

    @torch.no_grad()
    def score_views(self, views: Sequence[tuple[str, Guidance]]) -> np.ndarray:
        if not views:
            return np.zeros((0, self.task.num_candidates))
        prompts = [self.scorer.encode_view(guidance, text) for text, guidance in views]
        return self.scorer.score(prompts, self.candidates).cpu().double().numpy()


def semantic_task(task: TaskRecord) -> TaskRecord:
    """The canonical classification task behind SIGMA-F: every leaf is "bound" to its own
    immutable identifier, so that candidate order equals the fixed semantic-label order."""
    return with_binding(task, {label: label for label in task.labels}, name=f"{task.name}-semantic")


class FactorizedPredictor:
    """SIGMA-F: semantic classification plus exact binding lookup (Appendix A.3).

    The semantic distribution is computed once per input under the canonical representation
    (base child order, identifiers and descriptions, no output bindings) and is held fixed
    when bindings change.  Consequently ``q_{g'} = P_e q_g`` holds exactly, and ties are
    resolved in the fixed semantic-label order.
    """

    tie_by_label = True

    def __init__(self, scorer: CandidateScorer, task: TaskRecord) -> None:
        self.scorer = scorer
        self.task = task
        self.semantic = semantic_task(task)
        self._canonical = self.semantic.base_guidance()
        self._labels = scorer.encode_candidates(self.semantic.candidates)
        self._cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def semantic_scores(self, texts: Sequence[str]) -> np.ndarray:
        """``(len(texts), |Z|)`` scores of the semantic labels in semantic-label order."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            prompts = [
                self.scorer.encode_view(self._canonical, text, include_outputs=False)
                for text in missing
            ]
            scores = self.scorer.score(prompts, self._labels).cpu().double().numpy()
            self._cache.update(zip(missing, scores, strict=True))
        return np.stack([self._cache[t] for t in texts])

    def score_views(self, views: Sequence[tuple[str, Guidance]]) -> np.ndarray:
        if not views:
            return np.zeros((0, self.task.num_candidates))
        semantic = self.semantic_scores([text for text, _ in views])
        lookup = np.array([guidance.label_of_candidate() for _, guidance in views])
        return np.take_along_axis(semantic, lookup, axis=-1)
