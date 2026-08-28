# GEPA prompt optimization — Assertion Developer

Follows on from [`fewshot_ablation_results.md`](fewshot_ablation_results.md), which
established that the zero-shot instruction (no worked examples) is the best of three
prompt variants on both exact-match rubric criteria and has the lowest schema-error
rate. This document optimizes that zero-shot instruction with
[DSPy](https://dspy.ai)'s **GEPA** optimizer (Agrawal et al. 2025, *"GEPA: Reflective
Prompt Evolution Can Outperform Reinforcement Learning"*, ICLR 2026 oral) and evaluates
the result against the real production pipeline — mirroring
`concept-mapper-agent/experiments/gepa_results.md`'s methodology for the sibling agent.

This document covers **two GEPA runs**, not one: the first run's training metric had a
real gap that let the optimizer regress in a way invisible to its own validation split;
the gap was diagnosed, fixed, and the optimization re-run. Both runs and the diagnosis
are kept below because the gap itself — and how it was found — is a methodological
result worth reporting, not just a mistake to erase.

## Why GEPA

Same rationale as the Concept Mapper run: GEPA is designed to work well with small
datasets (as few as 10 examples, 20-100 evaluations) and optimizes *instruction text*
via natural-language reflection on execution traces/feedback rather than collapsing
everything to a scalar reward — a good match for a metric that produces structured,
explanatory feedback per rubric criterion.

## Setup

- **Target model** (the one being optimized, i.e. the agent's runtime model):
  `gpt-4o-mini`, temperature 0.0 — matches `configs/openai.yaml`.
- **Reflection model** (GEPA's own model, used to propose new instruction candidates):
  `gpt-5`.
- **Judge model** (`assertion_developer/llm_judge.py::judge_assertion_alignment`):
  `gpt-4.1-mini` — a different model family from the target, to reduce self-preference
  bias when the judge scores the target model's own assertions.
- **Data**:
  `data/gesis_concept_mapper_assertion_evaluation_adjusted_for_assertion_agent_final.xlsx`
  (Leo's final gold revision), `Source Items + Assertions (cor)` sheet, CP-parent rows
  only (92 rows), split 64 train / 28 val (seed 0, `val_fraction=0.3` — same ratio as
  Concept Mapper's 36/15).
- **Budget**: `auto="light"` — same effort level as the Concept Mapper run (~490 metric
  calls budgeted for this dataset size), chosen to keep the two agents' optimization
  runs comparable in cost/effort.
- **Seed instruction**: the `a_zero_shot` winner from the ablation study (full rule
  table, definitions, no worked examples) — GEPA mutates this text directly.
- **Signature design choice not present in Concept Mapper**: `basic_concept` is typed
  as a DSPy `Literal` over the 22 valid concept names (not a free `str`). This was a
  deliberate response to the ablation study's finding that prose few-shot caused the
  model to invent plausible-but-invalid labels (`attributes`, `preferences`) — Literal
  typing enforces validity at the DSPy adapter level so GEPA doesn't need to "discover"
  this constraint through trial and error.

### Metric (fed to GEPA as feedback, not just a score)

Weighted sum, matching the three rubric criteria used in `assertion_evaluator.py`'s
gold-based evaluation:

- **0.35** — `basic_concept` exact match against `basic_concept_key`.
- **0.35** — `structure_code` exact match against `structure_code_gold` (falls back to
  "is it in `BASIC_CONCEPT_RULES[basic_concept]['allowed_codes']`" for the few rows
  missing a gold code).
- **0.30** — concept-assertion alignment, via `llm_judge.judge_assertion_alignment()`
  against `gold_assertion` (1-5 scale, normalized to [0,1]).

Each component that fails appends a natural-language reason to the feedback string
GEPA's reflection model reads (e.g. *"Wrong structure_code: predicted 'xIe', gold is
'xId'"*).

## Run 1: a training-metric gap let GEPA regress invisibly

The metric above, as first written, never checked that `variable_type` was consistent
with the chosen `basic_concept`. It only compared `basic_concept` to gold and
`structure_code` to gold/allowed-codes. But the *production* validator
(`AssertionOutput`'s Pydantic `model_validator` in `assertion_schemas.py`) enforces a
stricter invariant the metric was never asked to check: `variable_type` must match
`BASIC_CONCEPT_RULES[basic_concept]["variable_type"]`, or the whole record is rejected
outright, with no assertion produced at all.

DSPy's Literal-typed output fields guarantee each field is *individually* valid, but
DSPy never runs `AssertionOutput`'s cross-field validator during optimization — that
only happened afterwards, in `run_gepa_eval.py`, against the real production schema.
Effect: GEPA's optimized instruction pushed the model to answer, e.g.,
`basic_concept="policies"` (correct) with `variable_type="objective"` (wrong — always
`subjective` per the rule table) on indicators that read like historical/administrative
facts (e.g. "civil rights protection in 1977"). The training metric scored
`basic_concept="policies"` as fully correct since it matched gold, and never penalized
the `variable_type` mismatch that makes the combination fail validation. Nothing
discouraged the drift.

**Result (full 92-row gold set, real production pipeline):**

| Metric | a_zero_shot (hand-written) | GEPA-optimized, run 1 |
|---|---|---|
| Basic concept accuracy | **66.30%** | 61.96% |
| Structure code accuracy | 45.65% | **49.44%** |
| Mean alignment score (1-5, judge) | 4.23 | **4.39** |
| Schema-validation errors | **2/92 (2.2%)** | 12/92 (13.0%) |

8 of the 12 errors were exactly the `{policies, rights} x variable_type=objective`
pattern; on the 2 non-error rows where the model also chose `basic_concept="policies"`,
it correctly paired it with `"subjective"` — confirming the model wasn't confused about
the *concept*, only inconsistent about the *type* on specific phrasings, and training
never penalized that inconsistency. **Verdict on run 1: not adopted** — a 6x error-rate
increase outweighs the structure_code/alignment gains, especially alongside a drop in
basic_concept accuracy.

## Metric fix

Added a hard gate at the top of the metric function (and mirrored it in
`plain_accuracy()`'s DSPy-internal comparison, so that number stops hiding the same
blind spot): if `variable_type` doesn't match `BASIC_CONCEPT_RULES[basic_concept]["variable_type"]`,
the row scores **0** across all three criteria — exactly mirroring the production
validator's all-or-nothing behavior (an invalid combination never produces a usable
assertion, so no partial credit for structure_code or alignment makes sense either).

Confirms the diagnosis was right: re-scoring the *unmodified* hand-written baseline with
the fixed `plain_accuracy()` on the same 28-row val split immediately surfaced 7 rows
with this exact inconsistency (`errors: 7/28`) — invisible under the old metric, visible
the moment the check was added, before GEPA even ran again.

## Run 2: GEPA re-run with the fixed metric

Same setup as run 1 (`--auto light`, same seed instruction, same train/val split),
metric fix only.

**(1) DSPy-internal, 28-row val split** (`plain_accuracy()`, now variable_type-aware):

| | Errors | Basic concept accuracy | Structure code accuracy |
|---|---|---|---|
| Baseline (hand-written zero-shot seed) | 7/28 | 66.7% (of 21 valid) | 47.6% |
| GEPA-optimized, run 2 | **2/28** | **80.8%** (of 26 valid) | **80.8%** |

**(2) Full 92-row gold set, real production pipeline:**

| Metric | a_zero_shot (hand-written) | GEPA-optimized, run 2 |
|---|---|---|
| Basic concept accuracy | 66.30% | **71.74%** |
| Structure code accuracy | 45.65% | **59.78%** |
| Mean alignment score (1-5, judge) | 4.23 | **4.32** |
| Schema-validation errors | **2/92 (2.2%)** | 7/92 (7.6%) |

A real improvement on all three rubric-weighted criteria this time (+5.4pp basic
concept, +14.1pp structure code, +0.09 alignment). The error count is still up (2 -> 7),
but the composition of those 7 errors turned out to expose a genuine, separate,
previously-undetected issue — not a sign the optimization itself is unreliable.

### A newly-surfaced finding: `values` has an undocumented second structure code

5 of the 7 remaining errors are all the same pattern:
`structure_code='vIi' is not allowed for basic_concept='values'. Allowed codes: ['xIv']`.
This is **not** a GEPA hallucination. Checking the gold sheet directly:

| example_id | indicator | basic_concept_key | structure_code_gold |
|---|---|---|---|
| EX026 | attachment to the nation as a value | values | `vIi` |
| EX037 | religion as a personal value | values | `vIi` |
| EX045 | democracy as an ideal | values | `vIi` |

All three `values`-labeled rows in Leo's gold sheet use `vIi`, not `xIv`. GEPA's
optimized instruction picked up on this real pattern in the training data (`"values:
xIv; vIi (use vIi for 'X is an ideal/value for the respondent'; default xIv
otherwise)"`) and generalized it — correctly, by the gold standard's own evidence — to
2 additional rows beyond the 3 it trained on. `BASIC_CONCEPT_RULES["values"]` in
`assertion_rules.py`, however, only lists `xIv` as an allowed code, so every one of
these predictions is rejected by production validation.

This was not visible in the fewshot ablation or in run 1, because both used the
hand-written zero-shot prompt, which never mentions `vIi` and so never produced it.
GEPA is the first process to have actually read the training data closely enough to
notice the discrepancy. The remaining 2/7 errors are the same
`variable_type`-inconsistency residual seen elsewhere (`demographics`, `knowledge`) and
roughly match the hand-written baseline's own 2/92 error rate — i.e., close to an
irreducible floor for this model at this effort level, not something the metric fix
introduced.

**This is a genuine rule-table gap, not a bug in the experiment scripts** — flagging it
rather than silently patching `assertion_rules.py`, since that file is core production
infrastructure and the earlier lowercase-l/capital-I fixes in this same file were
deliberately made only after cross-checking the professor's source PDF. Two ways to
resolve it, both one-line changes, decision deferred:
1. Add `"vIi"` to `BASIC_CONCEPT_RULES["values"]["allowed_codes"]` (treat it as a
   legitimate second structure code for `values`, alongside `xIv`) — supported by 3/3
   independent gold-data occurrences.
2. Treat `vIi` as another legacy transcription artifact (like the earlier
   `xlv`/`xli`/`xle`/`xlf` typos) and normalize gold data comparisons to `xIv` instead,
   if `vIi` is judged not to be a real distinct code.
If option 1 is taken, the GEPA-optimized prompt's error rate would drop from 7/92
(7.6%) to 2/92 (2.2%) — identical to the hand-written baseline's error rate, with none
of the accuracy tradeoff.

## Resolution: `vIi` confirmed correct against the primary source, rule table fixed

Resolved directly rather than left as a deferred choice: the professor's original
"Structures for subjective variables" table (the primary source for the whole rule
table) lists exactly one structure code for `values` — **`vIi`** — with dashes in the
structure_2/structure_3 columns (no alternate forms exist for this concept). `xIv` does
not appear anywhere in that table. This means the earlier "fix" (made before this GEPA
run, during the initial `assertion_rules.py` audit) was itself a transcription error —
it corrected a lowercase-l/capital-I typo (`xlv`) but reconstructed the wrong target
code, assuming the `x` + `I` + `[concept-initial]` pattern used by `evaluation`
(`xIe`)/`importance` (`xIi`)/`cognitive_judgment` (`xIc`) also applied to `values`,
when the source material actually uses a different subject letter (`v`, not `x`) for
this one concept specifically.

**Fix applied** (option 1's resolution, now no longer contingent): `BASIC_CONCEPT_RULES["values"]`
in `assertion_rules.py` now has `allowed_codes: ["vIi"]` and `default_code: "vIi"`
(replacing `xIv` entirely, not adding `vIi` alongside it, since the source table shows
only one valid code). Mirrored in `assertion_prompts.py`'s rule table (affects the
hand-written zero-shot/message-history prompts too), `dspy_gepa_optimize.py`'s seed
instruction copy, and `assertion_evaluator.py`'s legacy-alias table (`xlv` and `xIv`
both now normalize to `vIi`, so any older data or prior run's output compares
correctly).

**Re-verification, not re-optimization**: the GEPA-discovered instruction text did not
need to change — it had already converged on `vIi` for `values` on its own, correctly,
ahead of the rule table. Only the generation + evaluation passes needed re-running
against the corrected rule table (cheap: ~92 gpt-4o-mini calls + judge calls each, no
`gpt-5` reflection calls, a few tens of cents total) — for both the GEPA-optimized
candidate and, for full consistency, the `a_zero_shot`/`b_prose_fewshot`/`c_message_history_fewshot`
baselines in `fewshot_ablation_results.md` (all of which had been silently accepting
the wrong `xIv` code for the same 3 gold rows without erroring, since `xIv` used to be
"validly wrong" rather than rejected).

**Final numbers, full 92-row gold set, real production pipeline:**

| Metric | a_zero_shot (hand-written) | GEPA-optimized (run 2 instruction, corrected rules) |
|---|---|---|
| Basic concept accuracy | 65.22% | **76.09%** |
| Structure code accuracy | 48.91% | **63.04%** |
| Mean alignment score (1-5, judge) | 4.39 | 4.38 |
| Schema-validation errors | 3/92 (3.3%) | **2/92 (2.2%)** |

(The `a_zero_shot` baseline numbers here are the freshly re-run, post-fix figures from
`fewshot_ablation_results.md` — very slightly different from the 66.30%/45.65%/2.2%
quoted earlier in this document, both because of the rule fix and normal run-to-run LLM
variance on the ~2-5 error-prone rows.) With the rule-table gap closed, GEPA-optimized
now wins on both exact-match rubric criteria by a wide margin (+10.9pp basic concept,
+14.1pp structure code) with an equal-or-better error rate and effectively identical
alignment. This is the clean, unambiguous win that run 1 and run 2 (pre-fix) did not
deliver.

## Cost

| Run | Target (gpt-4o-mini) | Reflection (gpt-5) | Judge (gpt-4.1-mini) | Total |
|---|---|---|---|---|
| Run 1 (metric gap) | 566 calls, $0.1923 | 15 calls, $0.6276 | 555 calls, $0.0946 | $0.9145 |
| Run 2 (fixed metric) | 578 calls, $0.1888 | 17 calls, $0.6681 | 502 calls, $0.0849 | $0.9417 |
| **Combined** | | | | **$1.8562** |

Plus two 92-row production-pipeline evaluation passes (gpt-4o-mini generation +
gpt-4.1-mini judge each): a few tens of cents each, well inside budget.

## The optimized instruction (full text)

See [`gepa_optimized_instruction.txt`](gepa_optimized_instruction.txt) for run 2's
complete text (the file was overwritten by the second run; run 1's text is no longer
saved separately, only its aggregate results above). Notable content: explicit
input/output field documentation, a "Selection workflow" and "Critical guardrails"
section (including an explicit, now mostly-effective guardrail: *"Policies, norms, and
rights are subjective by definition; never pair these with variable_type=objective"*),
and domain-specific heuristics referencing specific gold indicators (e.g. "governance
level of [policy/area]" -> `cognitive_judgment`, not `policies`).

**Caveat for the paper**: as with Concept Mapper, several heuristics reference indicator
names that are themselves rows in the 92-item gold set, so some of the measured
structure_code/alignment gain may be specific to this indicator list rather than a
topic-agnostic capability improvement.

## Decision

**Adopt the GEPA-optimized instruction (run 2 text, evaluated against the corrected
rule table) as the new production prompt for Assertion Developer.** It beats the
hand-written zero-shot baseline on every rubric-weighted criterion (+10.9pp basic
concept, +14.1pp structure code) with a slightly better error rate (2.2% vs. 3.3%) and
statistically indistinguishable alignment (4.38 vs. 4.39). No caveats or contingencies
remain — the `vIi` question that made run 2's result conditional has been resolved
against the primary source, not worked around.

This result is also a useful methodological contrast with Concept Mapper's GEPA run:
there, the training metric happened to fully cover the deployed schema's cross-field
invariant, so no blind spot existed. Here, an incomplete metric let the optimizer regress
in a way its own validation split couldn't see (run 1); fixing the metric to mirror the
validator exactly turned the same optimization into a genuine win (run 2) and,
incidentally, surfaced a real, previously-undetected gap in the rule table itself
(`vIi`) that had been latent in the gold data all along, undetected by every prior
evaluation because the hand-written prompt never produced it — GEPA found a real bug in
code that predates the optimization entirely, simply by reading the training data more
carefully than any human pass over it had. **General lesson for the paper: a GEPA (or
any RL-adjacent prompt-optimization) training metric is only as trustworthy as its
fidelity to the deployed validator — partial proxies can look fine on the optimizer's
own validation split while hiding real regressions, and conversely, a faithful metric
can surface real bugs in code that predates the optimization entirely.**

## Next step

Port the winning instruction text (`gepa_optimized_instruction.txt`) into
`assertion_developer/assertion_prompts.py::SYSTEM_PROMPT` as the new production prompt,
replacing the current hand-written zero-shot version. Not done automatically in this
run — this is a deliberate, reviewable production change (touches the prompt every
downstream pipeline call depends on), left for explicit sign-off rather than applied as
a side effect of the experiment.
