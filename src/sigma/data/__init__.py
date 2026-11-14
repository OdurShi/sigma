"""Datasets, few-shot sampling, task hierarchies and the synthetic quick-start task."""

from sigma.data.fewshot import SWEEP_SHOTS, Example, class_orderings, nested_fewshot_subset
from sigma.data.synthetic import SyntheticData, make_synthetic_data, make_synthetic_task

__all__ = [
    "SWEEP_SHOTS",
    "Example",
    "SyntheticData",
    "class_orderings",
    "make_synthetic_data",
    "make_synthetic_task",
    "nested_fewshot_subset",
]
