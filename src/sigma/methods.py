"""Methods of the main comparison that are variants of one training pipeline (Appendix C.4).

Every trainable variant shares the guidance, the candidate scorer, the labeled subsets, the
LoRA configuration and the optimiser; they differ only in the fields of :class:`MethodSpec`.
Baselines that are other papers' complete systems are *not* re-implemented here; they are
listed in :data:`EXTERNAL_BASELINES` and requesting one raises :class:`ExternalBaselineError`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MethodSpec:
    """Training recipe of one method.

    Attributes
    ----------
    trainable:
        ``False`` for frozen prompting (the unadapted backbone with the same scorer).
    views:
        ``"single"`` trains on the base guidance ``g0`` only; ``"paired"`` trains on
        ``(g0, T_e(g0))`` pairs.
    consistency:
        Term coupling the two views: ``"none"`` (paired SFT, ``lambda = 0``),
        ``"centered_l2"`` (SIGMA, Eq. (5)), ``"transported_kl"`` (R-Drop-style control) or
        ``"pbm"`` (PBM-LoRA-style control, supplementary experiment E07).
    coefficient:
        ``lambda`` (or ``beta``); nominal default 1, selected on the development pool.
    edit_schedule:
        ``"alternate"`` (binding edits on odd steps, reversal on even steps), ``"binding"``
        or ``"reversal"`` (the single-family ablations of Table 2), or ``"global"`` (random
        rebinding of all outputs, supplementary experiment E08).
    factorized:
        SIGMA-F: train a semantic classifier and apply exact binding lookup (Appendix A.3).
    token_matched:
        Single-view SFT repeated until it reaches a given processed-token budget (Table 11).
    """

    name: str
    paper_name: str
    trainable: bool = True
    views: str = "paired"
    consistency: str = "none"
    coefficient: float = 0.0
    edit_schedule: str = "alternate"
    factorized: bool = False
    token_matched: bool = False

    @property
    def tunes_coefficient(self) -> bool:
        """Whether the coefficient grid of Appendix C.5 applies to this method."""
        return self.trainable and self.consistency != "none"

    def with_coefficient(self, coefficient: float) -> MethodSpec:
        return replace(self, coefficient=float(coefficient))


METHODS: dict[str, MethodSpec] = {
    spec.name: spec
    for spec in (
        MethodSpec("frozen", "Frozen prompting", trainable=False, views="single"),
        MethodSpec("sft", "LoRA-SFT", views="single"),
        MethodSpec("sft_token_matched", "SFT (token-matched)", views="single", token_matched=True),
        MethodSpec("paired_sft", "Paired SFT (lambda = 0)"),
        MethodSpec(
            "transported_kl", "Transported KL", consistency="transported_kl", coefficient=1.0
        ),
        MethodSpec("sigma", "SIGMA", consistency="centered_l2", coefficient=1.0),
        MethodSpec(
            "sigma_binding_only",
            "SIGMA: binding only",
            consistency="centered_l2",
            coefficient=1.0,
            edit_schedule="binding",
        ),
        MethodSpec(
            "sigma_reversal_only",
            "SIGMA: reversal only",
            consistency="centered_l2",
            coefficient=1.0,
            edit_schedule="reversal",
        ),
        MethodSpec("sigma_f", "SIGMA-F", views="single", factorized=True),
        # Controls of the supplementary experiments E07 and E08.
        MethodSpec("pbm_adapted", "PBM-adapted + paired CE", consistency="pbm", coefficient=1.0),
        MethodSpec("global_aug_paired", "Global binding aug. + paired CE", edit_schedule="global"),
        MethodSpec(
            "global_aug_sigma",
            "Global binding aug. + SIGMA alignment",
            consistency="centered_l2",
            coefficient=1.0,
            edit_schedule="global",
        ),
    )
}

#: Baselines of Table 1 / Table 6 that are other papers' systems.  They are hooks only.
EXTERNAL_BASELINES: dict[str, str] = {
    "icl_full": "Retrieval ICL with 20 demonstrations and the full inventory (Milios et al., 2023)",
    "dlr_full": "Dynamic label refinement with the full inventory (Park et al., 2025)",
    "miprov2": "MIPROv2 instruction/demonstration optimisation (Opsahl-Ong et al., 2024)",
    "gepa": "GEPA reflective prompt evolution (Agrawal et al., 2026)",
    "metatuner_j": "MetaTuner-J prompt/parameter co-generation (Bo et al., 2026)",
    "cpft": "CPFT contrastive pre-training and fine-tuning (Zhang et al., 2021)",
    "qaid": "QAID question-answering inspired intent detection (Yehudai et al., 2023)",
    "icda": "ICDA in-context data augmentation (Lin et al., 2023)",
    "fcslm": "FCSLM small/large model collaboration (Chen & Wang, 2025)",
    "qwen3_reranker": "Frozen Qwen3-Reranker-8B with exact binding lookup (Zhang et al., 2025)",
}


class ExternalBaselineError(NotImplementedError):
    """Raised when a baseline that is another paper's system is requested."""


def resolve_method(name: str) -> MethodSpec:
    """Look up a method preset by name."""
    if name in METHODS:
        return METHODS[name]
    if name in EXTERNAL_BASELINES:
        raise ExternalBaselineError(
            f"'{name}' ({EXTERNAL_BASELINES[name]}) is an external system that this repository "
            "does not re-implement.  To compare against it, run the original implementation "
            "with the task record, labeled subsets and guidance views produced here and feed "
            "its candidate scores to sigma.evaluation.records.EvalRecords."
        )
    raise KeyError(f"unknown method {name!r}; available: {sorted(METHODS)}")
