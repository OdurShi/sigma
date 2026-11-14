"""Explicit, named random streams.

The paper states that each run seed "jointly controls subset sampling, initialization, and
edits" (Appendix C.3).  Every stochastic component of this repository draws from its own named
stream derived from the run seed, so that, for example, the few-shot subsets and the test-time
guidance draws are identical across the methods compared under one seed.
"""

from __future__ import annotations

import random
import zlib

import numpy as np

#: Seeds used for the main experiments (Appendix C.3).
PAPER_SEEDS: tuple[int, ...] = (13, 21, 42, 87, 100)

#: Seeds used for the preliminary study (Appendix C.7.1).
PRELIMINARY_SEEDS: tuple[int, ...] = (13, 37, 73, 101, 137)


def stream(seed: int, *names: str | int) -> np.random.Generator:
    """Return a generator that depends only on ``seed`` and the given stream names.

    Names may be strings or integers (e.g. an instance index); the same arguments always
    produce the same stream, independently of any other random state.
    """
    keys = [int(seed)]
    for name in names:
        if isinstance(name, str):
            keys.append(zlib.crc32(name.encode("utf-8")))
        else:
            keys.append(int(name))
    return np.random.default_rng(np.random.SeedSequence(keys))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch global generators (model initialisation, LoRA init)."""
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))  # noqa: NPY002 - legacy global seeding is intentional here
    torch.manual_seed(seed)
