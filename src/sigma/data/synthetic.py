"""A small synthetic intent-routing task for the quick start and the test-suite.

The task mimics a customer-support router: utterances mention a few keywords of one intent,
the guidance lists every intent with its keywords and the output (a routing destination)
currently bound to it.  Destinations are arbitrary words, so the only way to answer correctly
after a binding swap is to read the binding from the guidance.  Everything is generated in
code from a fixed word list; no data is downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass

from sigma.data.fewshot import Example
from sigma.guidance.schema import TaskRecord, grouped_task
from sigma.seeding import stream

_WORLD: dict[str, dict[str, tuple[str, ...]]] = {
    "billing": {
        "refund": ("refund", "money", "back", "reimburse", "repay"),
        "invoice": ("invoice", "receipt", "bill", "statement", "copy"),
        "charge": ("charge", "twice", "double", "debit", "unknown"),
        "discount": ("discount", "coupon", "promo", "voucher", "sale"),
    },
    "account": {
        "login": ("login", "sign", "access", "locked", "enter"),
        "password": ("password", "reset", "forgot", "code", "secret"),
        "profile": ("profile", "name", "photo", "birthday", "edit"),
        "closure": ("close", "delete", "cancel", "deactivate", "leave"),
    },
    "shipping": {
        "tracking": ("track", "where", "parcel", "status", "locate"),
        "delay": ("late", "delay", "waiting", "slow", "overdue"),
        "address": ("address", "street", "moved", "city", "zip"),
        "returns": ("return", "send", "label", "exchange", "wrong"),
    },
    "technical": {
        "crash": ("crash", "freeze", "stuck", "broken", "error"),
        "install": ("install", "setup", "download", "update", "version"),
        "sync": ("sync", "backup", "cloud", "restore", "missing"),
        "speed": ("lag", "loading", "performance", "battery", "hot"),
    },
}
_DESTINATIONS = (
    "amber", "birch", "cedar", "delta", "ember", "fjord", "grove", "harbor",
    "iris", "jade", "kite", "lotus", "maple", "nova", "onyx", "pearl",
)  # fmt: skip
_FILLERS = ("please", "hello", "i", "my", "the", "can", "you", "help", "with", "need", "a", "it")

SYNTHETIC_INSTRUCTION = (
    "Answer with the output that the guidance binds to the class of the input. "
    "Sibling order does not denote priority."
)


@dataclass(frozen=True)
class SyntheticData:
    """Synthetic task record with a training pool, a development and a test split."""

    task: TaskRecord
    train_pool: list[Example]
    dev: list[Example]
    test: list[Example]


def make_synthetic_task(num_groups: int = 3, classes_per_group: int = 4) -> TaskRecord:
    """Root -> group -> intent tree whose outputs are arbitrary routing destinations."""
    if not 1 <= num_groups <= len(_WORLD) or not 1 <= classes_per_group <= 4:
        raise ValueError("the synthetic world has at most 4 groups of 4 classes")
    groups: dict[str, tuple[str, dict[str, str]]] = {}
    binding: dict[str, str] = {}
    for group in list(_WORLD)[:num_groups]:
        members = dict(list(_WORLD[group].items())[:classes_per_group])
        groups[group] = (
            f"{group} requests",
            {label: " ".join(keywords) for label, keywords in members.items()},
        )
        for label in members:
            binding[label] = _DESTINATIONS[len(binding)]
    return grouped_task(
        "synthetic-routing",
        groups,
        binding=binding,
        root_description="support requests",
        instruction=SYNTHETIC_INSTRUCTION,
    )


def _keywords(task: TaskRecord, label: str) -> list[str]:
    return task.nodes[label].description.split()


def sample_utterances(task: TaskRecord, per_class: int, seed: int, split: str) -> list[Example]:
    """Generate ``per_class`` utterances per intent: two keywords of the intent plus fillers."""
    examples: list[Example] = []
    for label in task.labels:
        rng = stream(seed, "synthetic", split, label)
        keywords = _keywords(task, label)
        for _ in range(per_class):
            words = [keywords[i] for i in rng.choice(len(keywords), size=2, replace=False)]
            words += [_FILLERS[i] for i in rng.choice(len(_FILLERS), size=3, replace=False)]
            order = rng.permutation(len(words))
            examples.append(Example(" ".join(words[i] for i in order), label))
    return examples


def make_synthetic_data(
    seed: int = 0,
    num_groups: int = 3,
    classes_per_group: int = 4,
    train_per_class: int = 16,
    dev_per_class: int = 4,
    test_per_class: int = 8,
) -> SyntheticData:
    """Build the synthetic task together with its three splits."""
    task = make_synthetic_task(num_groups, classes_per_group)
    return SyntheticData(
        task=task,
        train_pool=sample_utterances(task, train_per_class, seed, "train"),
        dev=sample_utterances(task, dev_per_class, seed, "dev"),
        test=sample_utterances(task, test_per_class, seed, "test"),
    )
