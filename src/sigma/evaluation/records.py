"""Raw evaluation records: candidate scores of every (test instance, guidance view) pair.

All reported metrics are pure functions of these arrays (:mod:`sigma.evaluation.aggregate`),
so that tables can be recomputed, and external systems compared, from saved scores alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EvalRecords:
    """Scores and bookkeeping for ``N`` instances, ``V`` views and ``m`` candidates.

    Attributes
    ----------
    conditions:
        Condition name of every view column (``base``, ``reversal``, ``binding``, ``k2``,
        ``k2+R``, ...), length ``V``.
    scores:
        ``(N, V, m)`` candidate scores ``s_g`` in the fixed candidate order.
    permutations:
        ``(N, V, m)``; ``permutations[i, v, c]`` is the image of base candidate ``c`` under
        the ordered product of the edits that produced view ``v``.
    accepted:
        ``(N, V)``; False for composed documents inside ``G_train`` (excluded from Comp.).
    gold:
        ``(N,)`` candidate index of the base answer ``y = a_{g0}(z)``.
    gold_label:
        ``(N,)`` semantic-label index of ``z``.
    base_label_of_candidate:
        ``(m,)`` semantic-label index of ``a_{g0}^{-1}(c)``.
    tie_by_label:
        Resolve score ties in semantic-label order (SIGMA-F) instead of candidate order.
    """

    conditions: list[str]
    scores: np.ndarray
    permutations: np.ndarray
    accepted: np.ndarray
    gold: np.ndarray
    gold_label: np.ndarray
    base_label_of_candidate: np.ndarray
    tie_by_label: bool = False

    def __post_init__(self) -> None:
        n, v, m = self.scores.shape
        if self.permutations.shape != (n, v, m) or self.accepted.shape != (n, v):
            raise ValueError("inconsistent record shapes")
        if len(self.conditions) != v or self.gold.shape != (n,):
            raise ValueError("inconsistent record shapes")

    @property
    def num_instances(self) -> int:
        return self.scores.shape[0]

    def columns(self, condition: str) -> np.ndarray:
        """Indices of the view columns belonging to ``condition``."""
        return np.array([i for i, c in enumerate(self.conditions) if c == condition], dtype=int)

    def save(self, path: str | Path) -> None:
        """Write the records to a compressed ``.npz`` file."""
        np.savez_compressed(
            Path(path),
            conditions=np.array(json.dumps(self.conditions)),
            scores=self.scores.astype(np.float32),
            permutations=self.permutations.astype(np.int16),
            accepted=self.accepted,
            gold=self.gold,
            gold_label=self.gold_label,
            base_label_of_candidate=self.base_label_of_candidate,
            tie_by_label=np.array(self.tie_by_label),
        )

    @classmethod
    def load(cls, path: str | Path) -> EvalRecords:
        with np.load(Path(path)) as data:
            return cls(
                conditions=json.loads(str(data["conditions"])),
                scores=data["scores"].astype(np.float64),
                permutations=data["permutations"].astype(np.int64),
                accepted=data["accepted"],
                gold=data["gold"],
                gold_label=data["gold_label"],
                base_label_of_candidate=data["base_label_of_candidate"],
                tie_by_label=bool(data["tie_by_label"]),
            )
