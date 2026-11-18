# Implementation notes

This document maps the paper to the code, lists every detail the paper leaves open together
with the choice made here, records the places where the paper appears internally
inconsistent, and keeps a record of the runs that were executed.

Section, equation, table and figure numbers refer to the submitted manuscript
"SIGMA: Label-Efficient Training of Language Models to Follow Changing Structured Guidance"
(main text plus Appendices A–D).

## 1. Paper → code map

### Structured guidance and interventions (Section 4.1, Appendix A.1)

| Paper | Code |
|---|---|
| Task record: rooted tree, identifiers, descriptions, bijection `a : Z → C`; validation of malformed records | `sigma/guidance/schema.py`: `TaskRecord`, `_validate`, `Node` |
| Fixed candidate order and fixed semantic-label order (lexicographic) | `TaskRecord.candidates`, `TaskRecord.labels` |
| Guidance `g` (child order + binding `a_g`), base guidance `g0` (ascending child identifiers) | `Guidance`, `TaskRecord.base_guidance` |
| Nested-list serialisation with unambiguous records; parse-back | `sigma/guidance/render.py`: `render_guidance`, `parse_guidance`, `same_structure` |
| Prompt `Render(x, C, g)`: instruction, guidance, input, candidate list | `render_prompt`, `PromptFields.as_text` |
| Eligible parents `V`, leaf children `L(v)` | `TaskRecord.eligible_parents`, `leaf_children`, `sibling_groups` |
| Eq. (1): binding swap, `a_g' = π_e ∘ a_g`, `y' = π_e(y)` | `sigma/guidance/edits.py`: `BindingSwap.apply` |
| Representation edit (reverse every child list, `π_e = id`) | `Reversal.apply` |
| Eq. (A.1): binding-edit law `ρ_bind` | `binding_edit_support`, `sample_binding_edit` |
| `ρ_T = ½ ρ_bind + ½ ρ_repr`; odd/even step schedule | `sample_mixture_edit`, `scheduled_edit` |
| Validation of constructed documents and identity replacement | `validate_edit`, `apply_edit` |
| Eq. (A.2): target-change probability | `target_change_probability`; empirical counters in `InterventionCounts` (`sigma/training/trainer.py`) |
| `G_train` and held-out compositions | `train_support_keys`, `apply_sequence`, `sample_composition` |

### Scoring and objective (Section 4.2, Appendix A.2, B.1, C.4)

| Paper | Code |
|---|---|
| Eq. (2): permutation matrix `[P_e v]_c = v_{π_e⁻¹(c)}` | `sigma/losses.py`: `transport` (O(m) scatter); NumPy twin in `sigma/evaluation/metrics.py` |
| Eq. (3): mean answer-token log-probability including end-of-answer; `q_g = softmax(s_g)` | `sigma/scoring/scorer.py`: `CandidateScorer.score`, `_score_packed`, `_score_rows`; `losses.candidate_log_probs` |
| Answer boundary / end-of-answer token | `sigma/scoring/encoders.py`: `ChatPromptEncoder`, `WordPromptEncoder` |
| Eq. (4): centring | `losses.center_scores` |
| Eq. (5): paired loss with coefficient `λ/(2m)` | `losses.paired_loss`, `equivariance_penalty`, `equivariance_residual` |
| Eq. (6): expectation over `ρ_T` via the alternating schedule | `Trainer.plan_step`, `Trainer.fit` |
| Identity pair reuses one score vector | `StepPlan.base_view == edit_view`, `Trainer.loss_from_scores` |
| Paired SFT (`λ = 0`), transported-KL control `β/2 {KL(P_e q_g‖q_g') + KL(q_g'‖P_e q_g)}`, single-view SFT, binding-only / reversal-only ablations, token-matched SFT, frozen prompting | `sigma/methods.py` presets; `losses.transported_kl`, `losses.single_view_loss` |
| Eq. (B.2), (B.3): pairwise log-odds identity, distance to the aligned subspace | verified in `tests/test_losses.py` |

### Adaptation, optimisation and selection (Section 4.3, Appendix A.2, C.5)

| Paper | Code |
|---|---|
| LoRA rank 16, scaling 32, dropout 0 on Q/K/V/O and gate/up/down | `sigma/models.py`: `LoraSettings`, `apply_lora`; `configs/base.yaml` |
| AdamW (0.9, 0.95), weight decay 0.01, clipping 1.0, cosine decay, 3% warm-up | `Trainer.__init__`, `learning_rate_factor` |
| `B = 16`, uniform sampling with replacement, `T = 2⌈10n/(2B)⌉` | `Trainer.plan_step`, `num_training_steps` |
| Five even-numbered checkpoints including the last | `checkpoint_steps` |
| Development pool: one base view and one `ρ_T` edit per instance | `sigma/evaluation/evaluate.py`: `build_dev_pool`, `DevPool.accuracy` |
| Selection rule and tie-breaks; grids `lr ∈ {1e-5, 5e-5, 1e-4}`, `λ ∈ {0, 0.01, 0.1, 1, 10}` | `sigma/pipeline.py`: `train_with_selection`, `select_trial`, `Trial.sort_key` |
| Memory-bounded gradient (scores without graphs → loss gradient w.r.t. scores → recomputation of candidate microbatches) | `Trainer.backward` (`gradient_mode="two_pass"`) |
| Inference under the current guidance, ties by fixed candidate order | `sigma/predictors.py`: `DirectPredictor`; `metrics.top1` |
| SIGMA-F, Eq. (A.3) | `FactorizedPredictor`, `semantic_task` |
| Eq. (7), Proposition B.3, Corollary B.4 | `sigma/evaluation/diagnostics.py`: `trajectory_diagnostic`, `TrajectoryDiagnostic.bound_holds` |

### Data and evaluation (Appendix C)

| Paper | Code |
|---|---|
| Table 3 datasets and splits | `sigma/data/datasets.py`: `DATASETS`, `prepare_dataset`, `verify_split_sizes`; shipped in `data/<name>/{train,valid,test}.jsonl` |
| Loading (shipped files first, download only if absent) | `sigma/data/datasets.py`: `resolve_data_dir`, `load_dataset`, `load_fewshot_subset`; `sigma/pipeline.py`: `load_task_and_splits`, `training_subset` |
| Nested few-shot subsets per seed | `sigma/data/fewshot.py`: `class_orderings`, `nested_fewshot_subset`; shipped in `data/<name>/fewshot/seed<seed>.json` |
| Hierarchies: domain/scenario trees, label-only clusters cut into `⌈√m⌉` groups | `sigma/data/hierarchy.py`: `prefix_groups`, `average_linkage_groups`, `label_cluster_groups`, `task_from_groups` |
| Opaque-output control (Table 10) | `with_opaque_outputs` |
| Test guidance: `g0`, reversal, ten binding draws, ten sequences of `k ∈ {2,4,8}` swaps with and without reversal, acceptance outside `G_train` | `sigma/evaluation/protocol.py`: `EvalPlan`, `ViewPlanner` |
| Accuracy, macro-F1, Hit@K, MRR | `sigma/evaluation/metrics.py` |
| Eq. (C.1)–(C.3): PairAcc, EquivAcc, `TV_eq` | `metrics.pair_correct`, `equivariant_prediction`, `tv_equivariance` |
| Base / Comp. aggregation, matched Base, changed/unchanged strata, acceptance table (C.6, Tables 1, 2, 7, 11, 12, 14) | `sigma/evaluation/aggregate.py`: `summarize`, `composition_average`, `matched_base`, `acceptance_table` |
| AULC, paired bootstrap (Tables 5, 13) | `metrics.aulc`, `metrics.paired_bootstrap_interval`; `sigma/evaluation/report.py` |
| Eq. (C.4)–(C.10): preliminary study | `diagnostics.aligned_rank_transitions`, `rank_retention`, `normalized_entropy`, `upper_percentiles`, `uncertainty_bins`, `response_error`, `binned_error_rates`; `sigma/analysis.py`: `revision_scores`; `scripts/preliminary_study.py` |
| Eq. (D.1), (D.2): margin–residual diagnostic | `diagnostics.trajectory_diagnostic`, `summarize_trajectories`; `analysis.margin_residual_study` |
| Eq. (D.3)–(D.5): intervention spectrum, Helmert contrasts, directional energy | `diagnostics.contrast_operator`, `edit_averaged_contrast`, `helmert_contrasts`, `spectral_basis`, `directional_energy`, `log_energy_ratio`; `analysis.spectral_contraction` |
| Eq. (D.6)–(D.8): LoRA knockout | `models.lora_knockout`, `diagnostics.knockout_effects`, `analysis.knockout_study` |

### Theory (Section 4.4, Appendix B.3–B.6)

`sigma/theory.py` implements the fixed-feature proxy: Eq. (B.6) (`proxy_objective`),
Lemma B.5 (`proxy_estimator`, `ProxyDesign.gram_matrices`), Theorem B.6 (`exact_risk`), the
hierarchy model of Appendix B.6 (`hierarchy_design`) and Eq. (B.38)/(B.39)
(`hierarchy_risk`).  `tests/test_theory.py` checks numerically that the closed form minimises
the objective, that `0 ⪯ K ⪯ I`, that the exact-risk identity agrees with a direct
bias/variance computation, that Theorem 4.1 holds sample-wise, that `S = I` and
`K = 0 ⊕ C_out` in the hierarchy model, and that Eq. (B.38) and (B.39) agree with the general
formula.  Appendices B.5 and B.7–B.9 (sharp mismatch bound, effective dimension,
concentration, minimax) are not implemented.

## 2. Unspecified details and choices

Each item states what the paper leaves open and what this repository does.  Where a second
reading is plausible it is named.

**Prompt and tokenisation**

1. *Prompt wording.*  The paper describes the content of the instruction (select the output
   bound to the input's semantic class; sibling order does not denote priority) but not its
   text.  The text is `DEFAULT_INSTRUCTION` in `sigma/guidance/schema.py`; the four fields are
   joined as `instruction / "Guidance:" / "Input:" / "Candidate outputs:"` with the candidates
   as a bulleted list in the fixed candidate order.
2. *Serialisation syntax.*  "Nested list, unambiguously delimited" is implemented as one
   record per line, two spaces per level, `- group "<id>": "<description>"` and
   `- class "<id>": "<description>" => "<output>"`, all fields being JSON string literals.
   The root record is part of the document.
3. *Chat formatting.*  The whole prompt is one user message passed through the model's chat
   template with the generation prompt appended; no system message is supplied (Qwen's
   template then inserts its default system prompt).  Alternative: instruction as the system
   message.
4. *Answer boundary and end-of-answer token.*  The candidate is tokenised on its own (no
   leading space) directly after the generation prompt and terminated by the chat template's
   end-of-turn token (`<|im_end|>` for Qwen, `<|eot_id|>` for Llama-3), which is inferred
   from the template (`ChatPromptEncoder`).  It can be overridden with `model.end_of_answer`.
5. *Context limit.*  The paper shortens label descriptions when a prompt would exceed 8,192
   tokens.  The scorer raises `ContextLimitError` instead of shortening automatically.

**Guidance content**

6. *Leaf descriptions.*  The paper does not give the semantic descriptions.  Leaves use the
   humanised intent name (`card_arrival` → "card arrival"); `task_from_groups` accepts
   user-supplied descriptions.
7. *Group descriptions.*  Label-derived groups list their member intent names (as in the
   paper).  Domain and scenario groups use "`<domain>` intents".
8. *Label embeddings for BANKING77.*  The humanised label names are embedded with
   `all-mpnet-base-v2`; ties in average linkage are broken by the smallest member labels in
   sorted order (`average_linkage_groups`).  The clustering was cross-checked against
   SciPy (`linkage(method="average", metric="cosine")`, `fcluster(..., 9, "maxclust")`):
   identical partition.  The resulting groups are strongly unbalanced, see Section 4.
9. *HWU64 hierarchy.*  The tree is built from the scenario prefix of the intent names.  On
   the public release this gives 18 scenarios, 15 of them eligible, and three intents
   (`cooking_recipe`, `news_query`, `weather_query`) outside any eligible group.  The paper's
   Appendix D.8 reports `J = 17` and one uncovered candidate; see Section 4.
10. *Opaque identifiers.*  `out_000 …`, assigned to the sorted labels by a permutation drawn
    from a fixed seed (0) that does not depend on the run seed.

**Training**

11. *Learning-rate schedule.*  Linear warm-up over `⌈0.03 T⌉` steps (first step at `lr/W`),
    then cosine decay to zero at step `T`.
12. *Weight decay* applies to all trainable (LoRA) parameters.
13. *Checkpoint steps.*  `2·round(k T / 10)` for `k = 1..5`, e.g. 96, 192, 290, 386, 482 for
    `T = 482`.
14. *Development accuracy.*  Mean top-1 accuracy over all `2·n_dev` views; an identity
    replacement in the edited position counts as a second copy of the base view.
15. *Development search.*  The learning-rate × coefficient grid (3 × 5 = 15 trials) is used
    for methods with a consistency term; methods without one (LoRA-SFT, paired SFT, SIGMA-F)
    search the three learning rates.  All trials start from the same adapter
    initialisation and use the same minibatches, edits and development pool.
16. *Transported KL.*  Both half-weighted cross-entropies are kept and
    `β/2 · {KL(P_e q_g‖q_g') + KL(q_g'‖P_e q_g)}` is added without a `1/m` factor.
17. *Binding-only and reversal-only ablations* use the named edit family on every step, which
    keeps the two-view budget.
18. *Token-matched SFT.*  Processed tokens are the token positions fed through the model in
    training passes (including the recomputation pass of the two-pass gradient), excluding
    development evaluation.  The number of single-view steps is fixed before training as
    `2⌈budget / (2 · B · passes · mean tokens per view)⌉`, so the schedule is stretched over
    the longer run.  The budget is read from the SIGMA run with the same dataset and seed.
19. *Minibatches and edits* come from separate named random streams, so that single-view
    and paired methods see the same original instances at every step for a given seed.
20. *SIGMA-F classifier.*  The paper specifies the lookup (Eq. (A.3)) but not how the
    semantic classifier is trained beyond "same original labeled instances, same semantic
    descriptions, canonical guidance representation".  Here it is the same backbone and
    LoRA configuration trained with single-view cross-entropy, where the document is the
    base-order tree *without output bindings* and the candidates are the immutable leaf
    identifiers; ties are resolved in semantic-label order.  With this reading SIGMA-F is
    independent of the output strings, which matches the identical SIGMA-F numbers of
    Tables 1 and 10.
21. *Dropout.*  The model is kept in evaluation mode for all scoring, including training.

**Evaluation**

22. *Composition draws.*  Each of the `k` swaps is drawn independently from `ρ_bind` and
    applied to the current document; the same sequence is evaluated without and with a final
    reversal.  A composed document is rejected iff it equals a member of `G_train`.
23. *"Comp." aggregation.*  "Aggregate conditions and instances using the metric's
    evaluation weights" is read as: mean over accepted views within an instance, mean over
    instances with at least one accepted view, mean over the six conditions.  Alternative:
    pooling all instance–condition cells.  The two coincide when every instance has an
    accepted view in every condition (as in Table 14).
24. *Pair-level diagnostics* (PairAcc, EquivAcc, `TV_eq`, changed/unchanged strata) pool
    accepted pairs with equal weight; binding diagnostics pool the binding draws of all test
    instances.
25. *Macro-F1 over several views.*  Views are pooled with weights that give every instance
    total weight one per condition; predictions are mapped back with `a_g⁻¹`.
26. *Ranking metrics.*  Hit@K and MRR average repeated views within an instance first, then
    instances, then conditions.
27. *Paired bootstrap.*  Seeds and test instances are resampled jointly with replacement;
    the per-instance statistic is the mean Comp. accuracy over conditions, so all views of
    an instance move together.
28. *AULC* is computed per seed and then averaged.
29. *Reduced evaluations.*  `data.max_test_instances` / `max_dev_instances` select a fixed
    random subset that depends only on the split, not on the seed or the method.

**Diagnostics**

30. *12-candidate subsets (Appendix D.7).*  The paper does not say how the restricted
    inventories were chosen.  `analysis.diagnostic_subtask` visits eligible sibling groups in
    a seeded random order and takes leaves until twelve are collected; one trajectory is
    drawn per in-subset test instance with depths cycling through 2, 4, 8.
31. *Null-space basis (Appendix D.8).*  The paper fixes Helmert contrasts for the exposed
    directions only.  The remaining centred directions are Helmert-type contrasts between
    the blocks on which null vectors are constant (`diagnostics.spectral_basis`).
32. *Preliminary study.*  The dataset is not named; the experiment file uses CLINC150 (the
    benchmark with 150 candidates) and a fixed random subset of 1,500 test inputs.  The
    binding revision exchanges the gold output with a uniformly drawn sibling partner, the
    order revision is the reversal, and the reference distribution is the unadapted backbone
    under `g0`.

**Quick start**

33. The tiny model of the quick start is randomly initialised, so the demo trains all of
    its weights (`model.adaptation: full`).  LoRA on the tiny model is exercised by the
    test-suite and by `configs/experiments/smoke_tiny.yaml`.

**Data**

34. *Few-shot index lists.*  The paper defines the subsets as prefixes of a seed-defined
    random ordering within each class but does not publish the index lists.  The orderings
    shipped in `data/<name>/fewshot/` are those of this repository's sampler
    (`class_orderings`: a NumPy generator seeded with the run seed, the stream name
    `"fewshot"` and the label) for the seeds 13, 21, 42, 87, 100, truncated to 16 entries per
    class; they cover the budgets 1, 2, 4, 5, 8, 10 and 16.
35. *Shipped benchmark files.*  The splits are those of the few-shot intent-detection
    release at revision `fa9553312b1dba54b1be0aae2d30b13171f898e5`, converted to JSONL.
    That release repository has no licence file of its own; the files are redistributed
    under the licences of the original datasets (CC BY 4.0, CC BY 3.0, CC BY 4.0) with
    attribution in `data/README.md`.

## 3. Candidate scoring implementation

The default `packed` strategy processes the prompt once per view: the answer tokens of all
candidates are appended to one sequence, a 4-D attention mask lets each candidate attend to
the prompt and to its own earlier tokens only, and position ids restart at the answer
boundary for every candidate.  This is exactly equivalent to scoring each candidate in a
separate forward pass; `tests/test_scoring.py` checks it against the row-wise reference
(`strategy: rows`), against a manual forward pass, across batch sizes, padding and candidate
microbatches, and for gradients.  Token log-probabilities are computed in float32 for the
answer positions only.

## 4. Possible inconsistencies in the paper

1. **HWU64 sibling groups.**  Appendix D.8 states `J = 17` eligible groups, 46 exposed and
   17 null directions and "one candidate outside the eligible sibling groups"; Table 14 gives
   98.95% acceptance at depth two.  The scenario prefixes of the public 64-intent release
   give 15 eligible groups with sizes (2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 9, 10) and
   three uncovered intents: again 46 exposed and 17 null directions, but a different `J`, a
   largest eigenvalue of `1/30` instead of the ≈ `1/68` visible in Figure 7, and an expected
   depth-two rejection rate of about 2.5% instead of 1.05%.  The paper's HWU64 tree therefore
   differs from the plain scenario grouping in a way the text does not describe.
   (For CLINC150 the code reproduces `J = 10`, 140/9 directions and `κ = 1/280`; the
   expected depth-two acceptance `1 − 1/1050` matches Table 14.)
2. **BANKING77 sibling groups.**  Appendix C.2 specifies frozen all-mpnet-base-v2 label
   embeddings, cosine distance and average-linkage clustering cut into `⌈√77⌉ = 9` groups.
   Run on the 77 label names this gives cluster sizes (42, 18, 5, 4, 3, 2, 1, 1, 1): three
   singleton clusters, hence only six eligible sibling groups and three intents that can
   never be swapped (embedding the raw identifiers instead of the humanised names gives
   (38, 19, 8, 4, 3, 2, 1, 1, 1)).  Appendix D.8 states `J = 9` for BANKING77; its
   upper-quartile eigenvalue 0.007937 = 1/(2·9·7) and the 99.66% depth-two acceptance of
   Table 14 correspond to nine roughly balanced groups (the tree above would give about
   95.5%).  The number of exposed directions, 68 = 77 − 9, is the same for any partition
   into nine clusters and therefore does not discriminate.  Complete linkage or Ward
   linkage on the same embeddings give balanced groups (e.g. Ward: 3, 6, 6, 7, 9, 9, 10,
   11, 16), so the linkage, the embedded text or the cut rule used for the paper probably
   differs from the description.  The code follows the description (average linkage).
3. **Trainable parameters.**  Rank-16 LoRA on the seven projections of Qwen2.5-7B-Instruct
   (hidden size 3584, MLP size 18944, key/value projections of width 512, 28 layers) has
   `28 · 16 · (2·7168 + 2·4096 + 3·22528) = 40,370,176` parameters, i.e. 40.37 M; Table 8
   reports 40.42 M.  (The same formula gives 8,798,208 for Qwen2.5-0.5B, which is the count
   PEFT reported in the reduced-scale run.)
4. **Knockout reference.**  Appendix D.9 lists intact Base/Comp. means (e.g. 90.698/88.408
   for BANKING77) that differ slightly from Table 1 (90.71/88.56); the paper notes this
   itself.  The quoted layer-22 example (0.378 and 1.340 pp) differs by 0.962, reported as
   0.963, presumably from unrounded values.
5. **Preliminary study.**  Its seeds (13, 37, 73, 101, 137) differ from the main seeds and
   its dataset is not named.
6. **SIGMA-F in Table 6.**  Its Base and Comp. entries differ slightly (and Comp. exceeds
   Base for POWERPLAY11) although the variant is exactly equivariant; this is possible only
   because that table does not use matched evaluation support, which the caption does not
   say.

## 5. Scope of the implementation

* The external systems of Tables 1, 6 and 8 (ICL-Full, DLR-Full, MIPROv2, GEPA, MetaTuner-J,
  CPFT, QAID, ICDA, FCSLM, Qwen3-Reranker) are represented by hooks (`sigma/methods.py`);
  the trainable variants of the pipeline are implemented as methods.
* MASSIVE-EN and HINT3 (Table 6) have no data-preparation script.
* Of the cost columns of Table 8, runs record trainable parameters and processed tokens;
  GPU-hours, peak memory and latency are not measured.
* `scripts/make_figures.py` plots the reported quantities without the decorative elements
  of the paper's figures (smoothed colour fields, density contours, display jitter).
* Appendices B.5 and B.7–B.9 of the theory are not part of `sigma/theory.py`.

## 6. Record of executed runs

All runs below were made on one Apple-silicon laptop (no CUDA) on 2026-11-13.  Experiments
with the paper's backbones, budgets, seeds and full candidate inventories were not part of
this record; the numbers of the paper quoted in the README come from the manuscript.

**Test-suite and lint.**  `pytest` passes on Python 3.10 (torch 2.14.1, transformers 5.20.2),
3.11 and 3.12 (torch 2.15.0, transformers 5.23.0), peft 0.21.3; `ruff check .` and
`ruff format --check .` are clean.  The tests cover, among others: hand-computed values of
Eq. (5); the reduction to a single cross-entropy for identity pairs; invariance to score
shifts; Eq. (B.2) and (B.3); equality of packed, row-wise and manual candidate scores under
batching, padding and candidate microbatching; equality of the two-pass and direct
gradients; the sampling law of Eq. (A.1) and the rate of Eq. (A.2); Eq. (7) and
Corollary B.4 on random trajectories; Eq. (D.3) against explicit enumeration; Theorem B.6
and Eq. (B.38); the consistency of the shipped data files.

**Quick start** (`sigma demo`, seed 13; synthetic 6-class routing task, tiny random
Qwen2-style model, all weights trained).  The output is reproduced in the README.  An
additional paired-SFT run with the same configuration (1,600 steps) gave Base
97.9, Rev. 97.9, Bind.Chg 91.8, Comp. 48.6, Comp.Chg 27.9, EquivAcc 47.8, TV_eq 0.536.  On
this task single-view fine-tuning keeps the outdated answer after a binding swap, the two
paired methods learn the edit families seen in training (single swaps, reversal), and none
of the three generalises to unseen compositions of several swaps (Comp.Chg between 3 and
29%); paired SFT and SIGMA reach the same composition accuracy.

**Experiment machinery.**  `configs/experiments/smoke_tiny.yaml` (16 runs, tiny model, 8
optimiser steps), followed by tables 1, 2, 4, 5, 7, 11, 12, 13, 14 of `make_tables.py`,
`margin_residual.py`, `spectral_contraction.py`, `lora_knockout.py`,
`preliminary_study.py`, and figures 2, 4, 5, 6, 7, 8 of `make_figures.py`.  All stages ran;
with this configuration the numbers carry no information.

**Data preparation.**  `scripts/prepare_data.py --dataset all` on the upstream release; the
split sizes equal Table 3.  The label clustering of BANKING77 ran with the real sentence
encoder and gives the same partition as SciPy's average linkage with `maxclust=9`.

| Dataset | Eligible sibling groups (sizes) | Intents outside | Exposed / null directions | Paper (App. D.8) |
|---|---|---|---|---|
| CLINC150 | 10 (ten groups of 15) | 0 | 140 / 9 | 10 groups, 140 / 9 |
| HWU64 | 15 (2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 9, 10) | 3 | 46 / 17 | 17 groups, 1 intent outside, 46 / 17 |
| BANKING77 | 6 (2, 3, 4, 5, 18, 42) | 3 | 68 / 8 | 9 groups, 68 / 8 |

The differences for HWU64 and BANKING77 are discussed in Section 4.

**Scorer on a pretrained backbone.**  Packed and row-wise candidate scores of
`Qwen/Qwen2.5-0.5B-Instruct` on two prompts with six candidates agree to 2.4e-7 (3.0e-6
across batching and candidate microbatches).

**Reduced-scale run** (`configs/reduced_scale/hwu64_subset_qwen2.5-0.5b.yaml`).
`Qwen/Qwen2.5-0.5B-Instruct` in float32 on `mps`, LoRA rank 16 (8,798,208 trainable
parameters), 7-intent subset of HWU64 (scenario groups `alarm` and `transport`), 4 shots
(28 training labels), 12 optimiser steps of 4 instances (about 1.7 passes instead of 10),
learning rate 1e-4, coefficient 1, no development search, 2 checkpoints on 12 development
instances, 30 test instances, 2 binding draws and 1 composition draw per depth, seed 13,
one run per method:

```bash
for m in sft paired_sft sigma; do
  sigma train --config configs/reduced_scale/hwu64_subset_qwen2.5-0.5b.yaml \
      --set method=$m output_dir=runs/reduced_scale/$m
done
```

| Method | Base | Rev. | Rev. Inv. | Bind. Chg. Acc. | Comp. | Matched Base | Comp. Chg. Acc. | Comp. PairAcc | EquivAcc | TV_eq |
|---|---|---|---|---|---|---|---|---|---|---|
| LoRA-SFT | 70.0 | 73.3 | 86.7 | 16.7 | 41.5 | 68.9 | 12.5 | 33.3 | 46.6 | 0.352 |
| Paired SFT (λ = 0) | 76.7 | 73.3 | 83.3 | 0.0 | 41.5 | 75.8 | 14.6 | 34.5 | 42.5 | 0.393 |
| SIGMA (λ = 1) | 66.7 | 80.0 | 83.3 | 16.7 | 35.1 | 65.5 | 11.5 | 25.3 | 34.5 | 0.393 |

Accuracies in percent; one test instance is 3.3 points; 12 changed-target binding pairs and
96 changed-target composition views.  In this run SIGMA has the lowest composition accuracy
and the highest reversal accuracy of the three methods.  The changed-target accuracy of all
three adapters lies between 0 and 17% (chance is 14.3% with seven candidates), i.e. after
12 optimiser steps none of them follows binding edits, and a gap of six points corresponds
to about two test instances of a single seed.  The run exercises the pretrained-backbone
path (chat template, end-of-answer token, LoRA, two-pass gradient, evaluation protocol); its
size does not allow conclusions about the method.  The raw files and a generated report are
in `results/reduced_scale/`.  A first, larger configuration (14 intents, 16 steps of 8
instances) was stopped because one optimiser step took 94–140 s on the shared machine.

## 7. Supplementary experiments (E01–E09)

The reported tables are stored in `results/supplementary/` and rendered in
[supplementary_results.md](supplementary_results.md); `results/paper/` holds Tables 1, 2, 5,
8, 11, 12 and 13 of the paper as CSV.  `sigma.supplementary.check_consistency` verifies every
value that a supplementary table repeats from another table or from the paper (for example
the SIGMA rows of E05 against Table 5, the gains of E09 against Tables 12 and 13, and the
13.73 GPU-hour column of E04 against Tables 1, 8 and 11).

The code that runs the experiments was added to the pipeline of the paper.  The table
descriptions leave several details open; the choices below are this repository's.

| ID | Code | Choices made here |
|---|---|---|
| E01 | `guidance/edits.py`: `CrossGroupSwap`, `RandomOrder`, `sample_cross_group_swap`; `evaluation/protocol.py`: `EvalPlan.transfer`; `evaluation/aggregate.py`: `summary["transfer"]`; `guidance/schema.py`: `with_descriptions`; `data.description_file` | Each broader edit is appended to the standard composition with the same draw (same swaps), for every depth in `composition_depths`. A cross-group swap exchanges two leaves with different parents, drawn uniformly over such pairs. A random order shuffles the children of every internal node. Comp. applies the rule of Appendix C.6 to the conditions of one family; changed-target accuracy pools its accepted changed views. Paraphrases are read from a JSON file (node identifier to description) and are not generated by this repository. SIGMA-F sees the same descriptions because the task record is replaced before any model is built. |
| E02 | `scripts/margin_residual.py --depths 8 16 --num-candidates all --all-paths`; `analysis.margin_residual_study`; `evaluation/diagnostics.path_residual_summary` | The path-average local residual of one trajectory is `E_local / k`; the table entry is its median over trajectories of one depth. Accuracy counts every sampled path. |
| E03, E06 | `training/trainer.py`: `TrainResult.training_seconds`, `gpu_hours`; `optim.num_gpus` | Wall-clock time of the optimisation steps, development evaluation excluded, times `optim.num_gpus`. Label counts are `shots × intents` plus the development split. |
| E04 | `optim.max_gpu_hours`; `configs/experiments/supp_e04_time_caps.yaml` | The cap is checked after every optimiser step; a capped run evaluates the development pool once more and keeps its best checkpoint. The learning-rate schedule is that of the uncapped run. |
| E05 | `configs/experiments/table5_label_efficiency.yaml` (method `sigma_f`) | No new code. |
| E07 | `losses.pbm_consistency`; method `pbm_adapted` | Dispersion is the variance, across the two views, of the aligned full-inventory log-probabilities, averaged over candidates; the entropy term is the entropy of the view-averaged distribution; both share one coefficient, searched on the grid of Appendix C.5 (at most 15 trials). |
| E08 | `guidance/edits.py`: `GlobalRebinding`, schedule `global`; methods `global_aug_paired`, `global_aug_sigma`; `data.outputs: opaque` | Every training step draws a uniform permutation of all outputs, independently of the labels. |
| E09 | `scripts/make_tables.py --table 12` and `--table 13` | No new code. |

The definition of the certificate in this repository is the paper's (`R_mr = sqrt(2) E_local /
gamma_0`, Eq. (D.2), Corollary B.4).

The new code paths are covered by `tests/test_supplementary_code.py` on the toy task and the
tiny model; the reported tables by `tests/test_supplementary_results.py`.
