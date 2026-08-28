# GEPA prompt optimization — Concept Mapper

Follows on from [`fewshot_ablation_results.md`](fewshot_ablation_results.md), which
established that the zero-shot instruction (no worked examples) is at least as good as
either few-shot format on every metric, and strictly better than message-history
few-shot on `indicator_model` accuracy. This document optimizes that zero-shot
instruction with [DSPy](https://dspy.ai)'s **GEPA** optimizer (Agrawal et al. 2025,
*"GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"*, ICLR 2026
oral) and evaluates the result against the real production pipeline.

## Why GEPA

GEPA was chosen over DSPy's other optimizers (MIPROv2, BootstrapFewShot) because it is
designed to work well with small datasets (as few as 10 examples, 20-100 evaluations),
which fits our 51-row human-relabeled gold set, and because it optimizes the
*instruction text* via natural-language reflection on execution traces/feedback rather
than collapsing everything to a scalar reward — a good match for a metric that already
produces structured, explanatory feedback (see below).

## Setup

- **Target model** (the one being optimized, i.e. the agent's runtime model):
  `gpt-4o-mini`, temperature 0.0 — matches `configs/openai.yaml`.
- **Reflection model** (GEPA's own model, used to propose new instruction candidates):
  `gpt-5` — deliberately a stronger model than the target, since it only runs a handful
  of times per optimization (21-23 calls), so its cost impact is small even though it's
  a more expensive model.
- **Judge model** (see `concept_mapper/llm_judge.py`): `gpt-4.1-mini` — a different
  model family from the target, to reduce self-preference bias when the judge scores
  the target model's own indicator lists.
- **Data**: `data/gesis_concept_mapper_assertion_evaluation_adjusted.xlsx`,
  `Concept Mapper Gold` sheet, human-relabeled `*_leo` columns (51 rows, 20 CI / 31 CP),
  split 36 train / 15 val (seed 0).
- **Budget**: `auto="light"` (~440 metric calls total for this dataset size).
- **Seed instruction**: the zero-shot instruction from the ablation study (definitions
  of CI/CP and formative/reflective/mixed, no worked examples) — GEPA mutates this text
  directly rather than starting from scratch.

### Metric (fed to GEPA as feedback, not just a score)

Weighted sum, reweighted from the ablation study's rule-based checks now that indicator
quality can be scored:

- **0.4** — `ci_or_cp` exact match against gold.
- **0.3** — `indicator_model` exact match against gold (or `indicator_model == "NA"`
  for CI topics).
- **0.3** — indicator quality, via `llm_judge.judge_indicator_quality()`: for CP rows,
  the judge scores coverage and distinctiveness (1-5 each) against
  `gold_indicators_conceptual`; for CI rows, full credit iff `indicators == []`
  (deterministic, no judge call needed).

Each component that fails also appends a natural-language reason (e.g. *"Wrong
indicator_model: predicted 'formative', gold is 'reflective' (formative = components
build the construct; reflective = manifestations of one latent disposition)"*) — this
is what GEPA's reflection model actually reads when proposing the next instruction
candidate, not just the numeric score.

## Two bugs found while porting the result back to production (worth documenting)

Both are about the gap between DSPy's `Signature` abstraction and the production
`ConceptMap` Pydantic schema — not flaws in GEPA's search itself. Listing them because
they're a real methodological lesson: **a DSPy Signature must mirror the deployed
output schema field-for-field, or the discovered prompt silently fails when ported
back**, and the failure mode (Pydantic validation error, not a low accuracy score) can
look like a prompt-quality problem when it's actually a schema mismatch.

1. **Missing `warnings` field.** The first `ConceptMapperSignature` only declared
   `ci_or_cp`, `indicator_model`, `construct_definition`, `indicators`, `rationale` as
   output fields — `warnings` was omitted by oversight. GEPA optimized perfectly
   consistently within that (incomplete) schema, reaching 100%/86.7% on its own 15-row
   val set. But the discovered instruction never asked the model to emit `warnings`,
   so every one of the 51 production-pipeline calls failed Pydantic validation
   (`Field required: warnings`) the moment the instruction was lifted out of DSPy.
   Fixed by adding `warnings: list[str] = dspy.OutputField()` to the signature and
   re-running GEPA from scratch (full cost re-incurred, see below).
2. **`input_topic` is correctly input-only in DSPy, but production expects it echoed
   in the output.** DSPy models `input_topic` as an `InputField` — there's no reason
   to ask the model to redundantly repeat a string the caller already has, so the
   discovered instruction never asks for it in the output JSON. The production
   `ConceptMap` schema, however, requires `input_topic` present in the output. This is
   a harness gap, not a signature bug: fixed in `experiments/run_gepa_eval.py` by
   injecting the already-known topic into the parsed JSON before schema validation,
   rather than asking GEPA to make the model waste tokens repeating it.

## Results

Two different numbers exist for the "optimized" condition and they disagree
noticeably — reporting both, because the discrepancy itself is informative.

**(1) DSPy-internal, 15-row val split, DSPy's own lightweight prediction comparison**
(`plain_accuracy()` in `dspy_gepa_optimize.py` — no Pydantic validation, no judge in
this specific comparison):

| | CI/CP accuracy | indicator_model accuracy |
|---|---|---|
| Baseline (hand-written seed) | 93.3% | 53.3% |
| GEPA-optimized | **100%** | **86.7%** |

**(2) Full 51-row gold set, real production pipeline** (`ConceptMapperAgent`-equivalent
generation → `parse_concept_map` → Pydantic `ConceptMap` → `evaluator.py` +
`llm_judge.py`, gpt-4.1-mini judge) — this is the number that matters for the paper:

| Metric | a_zero_shot (hand-written) | GEPA-optimized |
|---|---|---|
| CI/CP accuracy | 86.27% | **88.24%** |
| indicator_model accuracy (CP only) | 51.61% | **67.74%** |
| Mean \|indicator count diff\| (CP only) | **1.39** | 1.68 |
| Mean indicator coverage 1-5 (CP only, judge) | **2.97** | 2.87 |
| Mean indicator distinctiveness 1-5 (CP only, judge) | 4.03 | **4.06** |
| Parse/validation errors | 0 | 0 |

**Takeaway: the full-set gains are real but smaller than the val-set numbers suggest**
— indicator_model is the clear, solid win (+16.1pp), CI/CP is a small win (+2.0pp), but
indicator coverage and count-accuracy are essentially flat-to-slightly-worse. The
optimizer's own reported val-set score (100%/86.7%) should not be quoted as "the"
result — it's measured on 15 rows with a simpler comparison than the full pipeline
uses, and doesn't include the indicator-quality axis at all. This gap between
optimizer-reported and independently-measured performance is worth stating explicitly
in the paper as a caution about trusting an optimizer's self-reported score at face
value.

## Cost

| | Calls | Cost |
|---|---|---|
| target (gpt-4o-mini) | 486 | $0.1737 |
| reflection (gpt-5) | 23 | $0.8185 |
| judge (gpt-4.1-mini) | 315 | $0.0936 |
| **Total (final successful run)** | | **$1.0858** |

(A first run without the `warnings` field cost an additional ~$0.94 before the bug was
found — not counted above since that run's output was discarded, but worth mentioning
as the real cost of the signature/schema mismatch, including debugging time.)
Plus the two 51-row production-pipeline evaluation passes (gpt-4o-mini generation +
gpt-4.1-mini judge on ~31 CP rows each): a few tens of cents each. All well inside the
$50 budget.

## The optimized instruction (full text)

See [`gepa_optimized_instruction.txt`](gepa_optimized_instruction.txt) for the complete
text used to produce the results above. Notable differences from the hand-written seed:
explicit output field ordering, a "Quality checks before finalizing" section phrased as
yes/no questions, and domain-specific heuristics for constructs that appeared
repeatedly in the training data (e.g., "fear/worry/anxiety about X" → reflective;
"trust in [institution]" → reflective with explicit integrity/competence/benevolence/
reliability facets).

**Caveat for the paper**: several of these domain-specific heuristics reference topics
that are themselves rows in the 51-item gold set (e.g. "trust in parliament", "fear of
crime" appear as canonical examples inside the instruction). This is expected — GEPA
trained on exactly this data — but it means the instruction is partly tailored to this
specific topic list, and some of the measured gain may not generalize to survey topics
outside the gold set. Worth flagging as a limitation rather than treating the 88%/68%
numbers as topic-agnostic capability.

## Decision

Adopt the GEPA-optimized instruction as the new candidate production `SYSTEM_PROMPT`
for the Concept Mapper, on the strength of the indicator_model gain (+16.1pp, the
metric the hand-written prompt was weakest on) and the CI/CP gain (+2.0pp), while
noting the indicator-count and coverage metrics did not improve — future prompt
iteration should specifically target indicator *count calibration* (the optimized
prompt's mean count deviates from gold slightly more than the hand-written one),
which the current metric weights only indirectly via the judge's coverage score.

## Next step

Run this same optimized instruction (and the original hand-written one, for
comparison) against the LRZ open-source models (Qwen3-8B, Llama-3.1-8B-Instruct,
DeepSeekR1-0528-Qwen3-8B — upgraded from the originally-planned Qwen2.5-7B-Instruct
and DeepSeek-R1-Distill-Qwen-7B after benchmark research showed both are meaningfully
stronger, more current models in the same size class) via `TransformersClient`, scored
with the same fixed `gpt-4.1-mini` judge for cross-model comparability, to test whether
a prompt optimized against gpt-4o-mini transfers to smaller open models or needs
per-model re-optimization.
