> ## ⚠️ MOCK / PLACEHOLDER DATA — NOT REAL EXPERIMENTAL RESULTS ⚠️
>
> **Every number in this file for Qwen3-8B, Llama-3.1-8B-Instruct, and
> DeepSeekR1-0528-Qwen3-8B is a fabricated, educated estimate — these models have
> NOT actually been run through this pipeline yet.** They exist only to let the
> paper's table structure, narrative, and analysis be drafted while real LRZ/Ollama
> compute is still pending. **The gpt-4o-mini column is the only real data here** —
> those numbers are copied verbatim from `fewshot_ablation_results.md` and
> `gepa_results.md` (actual pipeline runs, 92-row CP-parent gold set, post `vIi`
> rule-table fix).
>
> **DO NOT cite the open-model numbers in the submitted paper.** Before submission,
> replace this entire file's open-model columns with real measurements from
> `run_fewshot_ablation.py` / `run_gepa_eval.py` run against `--provider transformers`
> (LRZ) or `--provider ollama` (local). If real numbers can't be obtained in time,
> the open-model comparison must be dropped from the paper or explicitly marked as
> unavailable — do not submit estimated numbers as if they were measured.

# Mock benchmark — Assertion Developer across models (draft table structure only)

Simulates the final "4 techniques × 4 models" table: zero-shot / prose few-shot /
message-history few-shot / GEPA-optimized, each against gpt-4o-mini (real) and three
open-weight models (estimated), on the same 92-row CP-parent gold set
(`data/gesis_concept_mapper_assertion_evaluation_adjusted_for_assertion_agent_final.xlsx`).

## How the open-model numbers were estimated

Same method as `concept-mapper-agent/experiments/mock_results.md` — see that file for
the full reasoning (benchmark-tier gaps, 4-bit quantization discount, preserving the
real qualitative patterns already measured for gpt-4o-mini). Task-specific choices
made here:

- **Structure-code selection is modeled as reasoning-sensitive** (matching one of 22
  concepts to one of up to 8 allowed grammatical codes is closer to a deduction task
  than free-text generation), so **DeepSeekR1-0528-Qwen3-8B is estimated relatively
  stronger on `structure_code` accuracy** than its `basic_concept` accuracy would
  suggest on its own — but still worst on **schema-validation error rate**, since its
  long `<think>` traces compound with 4-bit quantization to increase malformed/invalid
  `variable_type`-`basic_concept` pairings (exactly the failure mode already found
  *for real* with gpt-4o-mini's own GEPA run 1, before the metric fix — see
  `gepa_results.md`).
- **Prose few-shot is assumed to have the worst error rate for every model**,
  mirroring the one completely consistent real finding across every gpt-4o-mini run
  of this ablation (hallucinated `basic_concept` labels like `attributes`/`preferences`
  not in the 22-item table).
- **GEPA is assumed to help every model on `basic_concept`/`structure_code`**, since
  the optimized instruction's explicit guardrails (e.g. "policies/norms/rights are
  subjective by definition") target *concept-table* mistakes that are model-agnostic,
  not gpt-4o-mini-specific quirks.

## (a) Zero-shot

| Model | Basic concept acc. | Structure code acc. | Alignment 1-5 (judge) | Errors (of 92) |
|---|---|---|---|---|
| **gpt-4o-mini** (real) | **65.22%** | **48.91%** | **4.39** | **3** |
| Qwen3-8B *(mock)* | 60.87% | 44.57% | 4.15 | 5 |
| DeepSeekR1-0528-Qwen3-8B *(mock)* | 59.78% | 47.83% | 4.20 | 9 |
| Llama-3.1-8B-Instruct *(mock)* | 54.35% | 39.13% | 3.95 | 7 |

## (b) Prose few-shot

| Model | Basic concept acc. | Structure code acc. | Alignment 1-5 (judge) | Errors (of 92) |
|---|---|---|---|---|
| **gpt-4o-mini** (real) | **60.87%** | **45.65%** | 4.51 | 8 |
| Qwen3-8B *(mock)* | 57.61% | 41.30% | 4.25 | 11 |
| DeepSeekR1-0528-Qwen3-8B *(mock)* | 54.35% | 44.57% | **4.28** | 16 |
| Llama-3.1-8B-Instruct *(mock)* | 50.00% | 36.96% | 4.05 | **14** |

## (c) Message-history few-shot (production format)

| Model | Basic concept acc. | Structure code acc. | Alignment 1-5 (judge) | Errors (of 92) |
|---|---|---|---|---|
| **gpt-4o-mini** (real) | **60.87%** | 46.74% | **4.40** | **5** |
| Qwen3-8B *(mock)* | 56.52% | 42.39% | 4.12 | 8 |
| DeepSeekR1-0528-Qwen3-8B *(mock)* | 53.26% | **45.65%** | 4.18 | 13 |
| Llama-3.1-8B-Instruct *(mock)* | 51.09% | 38.04% | 3.98 | 10 |

## (d) GEPA-optimized

| Model | Basic concept acc. | Structure code acc. | Alignment 1-5 (judge) | Errors (of 92) |
|---|---|---|---|---|
| **gpt-4o-mini** (real) | **76.09%** | **63.04%** | 4.38 | **2** |
| Qwen3-8B *(mock)* | 70.65% | 57.61% | 4.20 | 4 |
| DeepSeekR1-0528-Qwen3-8B *(mock)* | 68.48% | 59.78% | **4.25** | 7 |
| Llama-3.1-8B-Instruct *(mock)* | 63.04% | 51.09% | 4.02 | 6 |

## Speculative narrative (for drafting only — verify against real numbers before writing this into the paper)

- **gpt-4o-mini wins on almost every metric** — same caveat as the Concept Mapper mock
  file: the interesting question isn't whether an 8B model beats it, but how much of
  the gap each technique/model combination closes.
- **DeepSeekR1-0528-Qwen3-8B is modeled with the classic reasoning-model trade-off**:
  competitive or best-among-open-models on `structure_code` accuracy (the more
  deduction-like metric) but consistently worst on error rate across all four
  techniques. If real numbers don't reproduce this, it's a genuine finding worth
  discussing (e.g. our `<think>`-stripping parser fix and higher `max_new_tokens`
  budget may fully neutralize the error-rate penalty in practice).
- **Prose few-shot is assumed worst-error technique for every model**, replicating
  the one fully consistent real finding from all gpt-4o-mini runs.
- **The `vIi` rule-table bug (see `gepa_results.md`) is assumed already fixed** in
  whatever prompt/rules are used for these runs — if real open-model runs are done
  against a stale rule table, the error-rate numbers for `values`-labeled indicators
  will look artificially worse and should not be blamed on the model.

## Reminder

Replace this file's open-model numbers with real ones as soon as
`--provider transformers` (LRZ) or `--provider ollama` (local Mac) runs complete.
See `fewshot_ablation_results.md` and `gepa_results.md` for the exact real
methodology to replicate per model.
