> ## ⚠️ MOCK / PLACEHOLDER DATA — NOT REAL EXPERIMENTAL RESULTS ⚠️
>
> **Every number in this file for Qwen3-8B, Llama-3.1-8B-Instruct, and
> DeepSeekR1-0528-Qwen3-8B is a fabricated, educated estimate — these models have
> NOT actually been run through this pipeline yet.** They exist only to let the
> paper's table structure, narrative, and analysis be drafted while real LRZ/Ollama
> compute is still pending. **The gpt-4o-mini column is the only real data here** —
> those numbers are copied verbatim from `fewshot_ablation_results.md` and
> `gepa_results.md` (actual pipeline runs, 51-row human-relabeled gold set).
>
> **DO NOT cite the open-model numbers in the submitted paper.** Before submission,
> replace this entire file's open-model columns with real measurements from
> `run_fewshot_ablation.py` / `run_gepa_eval.py` run against `--provider transformers`
> (LRZ) or `--provider ollama` (local). If real numbers can't be obtained in time,
> the open-model comparison must be dropped from the paper or explicitly marked as
> unavailable — do not submit estimated numbers as if they were measured.

# Mock benchmark — Concept Mapper across models (draft table structure only)

Simulates what the final "4 techniques × 4 models" results table will look like:
zero-shot / prose few-shot / message-history few-shot / GEPA-optimized, each run
against gpt-4o-mini (real) and three open-weight models (estimated) on the same
51-row human-relabeled gold set (`data/gesis_concept_mapper_assertion_evaluation_adjusted.xlsx`).

## How the open-model numbers were estimated (so the guess is at least principled)

Not random — derived from three inputs, all cited in the conversation that produced
this file:

1. **Public benchmark tier gaps.** Qwen3-8B scores meaningfully above Qwen2.5-7B-Instruct
   on general/reasoning benchmarks (MMLU-Redux 79.5 vs 75.4, GPQA-Diamond 39.3 vs 36.4,
   LiveBench 53.5 vs 34.9 — Qwen3 Technical Report) and sits closer to gpt-4o-mini's
   tier (MMLU 82.0) than any 2024-generation 7B open model. Llama-3.1-8B-Instruct is
   treated as the weakest of the three (older generation, well-documented open
   baseline). DeepSeekR1-0528-Qwen3-8B is a reasoning-distilled model that beats plain
   Qwen3-8B by +10pp on AIME 2024 — modeled here as *relatively stronger on judgment
   calls that benefit from step-by-step reasoning* (e.g. formative vs. reflective
   classification) but *relatively weaker on raw output validity* (reasoning models'
   long `<think>` traces interact badly with aggressive quantization and increase
   malformed-output risk).
2. **4-bit quantization discount.** Published GPTQ/AWQ evaluations report ~2-8% quality
   degradation from 4-bit quantization, worse for smaller models and less-calibrated
   quantization methods (our bitsandbytes NF4 / Ollama Q4_K_M path is closer to the
   worse end than calibrated AWQ). Applied as an across-the-board discount on top of
   the base capability gap.
3. **Preserve the real qualitative patterns already measured for gpt-4o-mini** on this
   exact task: CI/CP accuracy is flat across zero-shot/prose/message-history and only
   GEPA moves it; indicator_model accuracy is where technique choice actually matters,
   with message-history the worst format and GEPA the clear winner (+16.1pp for
   gpt-4o-mini). The mock numbers below reproduce this same shape per model, scaled
   down, rather than inventing an unrelated pattern.

## (a) Zero-shot

| Model | CI/CP acc. | Indicator model acc. (CP only) | Mean \|count diff\| (CP only) | Coverage 1-5 (judge) | Distinctiveness 1-5 (judge) | Errors |
|---|---|---|---|---|---|---|
| **gpt-4o-mini** (real) | **86.27%** | 51.61% | 1.39 | **2.97** | 4.03 | **0** |
| Qwen3-8B *(mock)* | 84.31% | 48.39% | 1.55 | 2.75 | 3.85 | 1 |
| DeepSeekR1-0528-Qwen3-8B *(mock)* | 82.35% | **51.61%** | 1.65 | 2.80 | 3.88 | 3 |
| Llama-3.1-8B-Instruct *(mock)* | 78.43% | 38.71% | 1.90 | 2.45 | 3.60 | 2 |

## (b) Prose few-shot

| Model | CI/CP acc. | Indicator model acc. (CP only) | Mean \|count diff\| (CP only) | Coverage 1-5 (judge) | Distinctiveness 1-5 (judge) | Errors |
|---|---|---|---|---|---|---|
| **gpt-4o-mini** (real) | **86.27%** | 51.61% | 1.32 | **3.06** | 4.06 | **0** |
| Qwen3-8B *(mock)* | 84.31% | 45.16% | 1.48 | 2.81 | 3.90 | 2 |
| DeepSeekR1-0528-Qwen3-8B *(mock)* | 82.35% | **48.39%** | 1.58 | 2.85 | **3.92** | 5 |
| Llama-3.1-8B-Instruct *(mock)* | 76.47% | 35.48% | 1.84 | 2.52 | 3.65 | 4 |

## (c) Message-history few-shot (production format)

| Model | CI/CP acc. | Indicator model acc. (CP only) | Mean \|count diff\| (CP only) | Coverage 1-5 (judge) | Distinctiveness 1-5 (judge) | Errors |
|---|---|---|---|---|---|---|
| **gpt-4o-mini** (real) | **86.27%** | 38.71% | **1.26** | **3.06** | 4.00 | **0** |
| Qwen3-8B *(mock)* | 80.39% | 32.26% | 1.42 | 2.70 | 3.78 | 2 |
| DeepSeekR1-0528-Qwen3-8B *(mock)* | 82.35% | 35.48% | 1.50 | 2.72 | **3.80** | 4 |
| Llama-3.1-8B-Instruct *(mock)* | 76.47% | 25.81% | 1.71 | 2.38 | 3.50 | 3 |

## (d) GEPA-optimized

| Model | CI/CP acc. | Indicator model acc. (CP only) | Mean \|count diff\| (CP only) | Coverage 1-5 (judge) | Distinctiveness 1-5 (judge) | Errors |
|---|---|---|---|---|---|---|
| **gpt-4o-mini** (real) | **88.24%** | **67.74%** | 1.68 | 2.87 | **4.06** | **0** |
| Qwen3-8B *(mock)* | 84.31% | 58.06% | 1.81 | 2.90 | 3.95 | 1 |
| DeepSeekR1-0528-Qwen3-8B *(mock)* | 84.31% | 64.52% | 1.87 | **3.05** | 3.98 | 2 |
| Llama-3.1-8B-Instruct *(mock)* | 80.39% | 48.39% | 2.13 | 2.65 | 3.75 | 2 |

## Speculative narrative (for drafting only — verify against real numbers before writing this into the paper)

- **gpt-4o-mini wins on almost every metric, as expected** — the point of this
  comparison isn't "does an 8B model beat GPT-4o-mini" (it shouldn't, and if a mock
  number ever shows otherwise treat it as noise, not signal) but *how much of the
  gap GEPA optimization closes* for each open model.
- **Qwen3-8B is estimated as the strongest open model**, consistent with its
  documented benchmark lead over the Qwen2.5 generation.
- **DeepSeekR1-0528-Qwen3-8B is modeled as competitive-to-best on indicator_model
  accuracy** (a judgment call that benefits from reasoning) **but worst on error
  rate** (reasoning trace + quantization interaction) — if the real numbers don't
  show this trade-off, that's a genuine finding, not a failure to match the mock.
- **Message-history few-shot is assumed worst for indicator_model across every
  model**, mirroring the one clear real finding from the gpt-4o-mini ablation.
- **GEPA is assumed to help every model**, but the prompt was optimized *for*
  gpt-4o-mini — a real, open question this file cannot answer is whether the gains
  transfer as cleanly to open models with different instruction-following behavior,
  or whether per-model re-optimization would be needed. That transfer question is
  the actual research contribution of running the real experiment; this mock file
  only exists to unblock writing everything else.

## Reminder

Replace this file's open-model numbers with real ones as soon as
`--provider transformers` (LRZ) or `--provider ollama` (local Mac) runs complete.
See `fewshot_ablation_results.md` and `gepa_results.md` for the exact real
methodology to replicate per model.
