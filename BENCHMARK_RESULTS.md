# Open-Model Prompt-Technique Benchmark — Consolidated Results

**Status: real, measured data.** Covers both pipeline stages (Concept Mapper, Assertion
Developer), all four prompt techniques (zero-shot / prose few-shot / message-history
few-shot / GEPA-optimized), and four generation models (gpt-4o-mini + three open
models run locally via Ollama). This supersedes `experiments/mock_results.md` in both
agent folders — those files are fabricated placeholders and should be deleted or
replaced before the paper is written from this data (see §6).

Every number below was read directly from `*/experiments/outputs/{model_tag}_summary.json`
(open models) and from `fewshot_ablation_results.md` / `gepa_results.md` (gpt-4o-mini,
pre-existing, not re-run) in each agent's `experiments/` folder. Best-value bolding in
every table was computed programmatically from those same files, not eyeballed.

## 1. Setup

| | |
|---|---|
| Runtime (open models) | Ollama, Apple M4 Pro (48GB), GGUF **Q4_K_M** for all three |
| Models | gpt-4o-mini (OpenAI API), Llama-3.1-8B-Instruct, Qwen3-8B, Granite-4.2-8B |
| Thinking mode | disabled for Qwen3 and Granite (both hybrid-reasoning; enabling it was measured to cost 191 generated tokens vs 3 on a trivial prompt, so it was turned off via the Ollama API's `think` flag — never via an in-prompt token, to avoid contaminating the prompt variable under test) |
| Judge | gpt-4.1-mini, temperature 0, identical for every model/technique |
| Generation | temperature 0; max_tokens 1200 (Concept Mapper) / 800 (Assertion Developer) |
| Data | Concept Mapper: 51-row human-relabeled gold set (20 CI / 31 CP). Assertion Developer: 92-row CP-parent gold set |
| Quantization consistency | **All four open-model conditions (Concept Mapper × Assertion Developer, all three models) use the same backend (Ollama/GGUF Q4_K_M)** — an earlier partial run of Llama-3.1-8B-Instruct on LRZ (bitsandbytes NF4 on a V100) was discarded in favor of re-running it on Ollama, specifically so no cross-model comparison mixes two different quantization algorithms. The discarded LRZ files are kept locally (`lrz_llama3_outputs/`) for reference only, not cited here. |
| Model selection note | DeepSeekR1-0528-Qwen3-8B was dropped in favor of Granite-4.2-8B: it shares Qwen3-8B's pretraining (same base model, reasoning-distilled), so it added the least family diversity for the most compute (~4x, due to long `<think>` traces). Granite (IBM) is a genuinely independent model family at the same parameter count, and is marketed specifically for structured/JSON output — a directly relevant property to test. |
| Production prompts | **Not modified.** `a_zero_shot` *is* the production `SYSTEM_PROMPT` in both agents; `b`/`c` add examples to that same text. Editing it would invalidate the whole comparison, not just one condition. |

## 2. Concept Mapper (n=51: 20 CI, 31 CP)

Bold = best value in that column (metric) across the four models, for that technique
(ties both bolded). CI/CP accuracy is over all 51 rows; every other metric is CP-only
(n=31). `errors` = generation/parse failures (schema-validation errors), lower better.

### (a) Zero-shot

| Model | CI/CP acc. | Indicator model acc. | \|count diff\| | Coverage 1-5 | Distinctiveness 1-5 | Errors |
|---|---|---|---|---|---|---|
| gpt-4o-mini | 86.27% | **51.61%** | **1.39** | 2.97 | **4.03** | **0** |
| Llama-3.1-8B-Instruct | 92.16% | 45.16% | 1.61 | 2.77 | 3.84 | **0** |
| Qwen3-8B | 90.20% | 22.58% | 1.48 | 2.74 | 4.00 | **0** |
| Granite-4.2-8B | **96.08%** | 19.35% | 2.00 | **3.23** | 3.94 | **0** |

### (b) Prose few-shot

| Model | CI/CP acc. | Indicator model acc. | \|count diff\| | Coverage 1-5 | Distinctiveness 1-5 | Errors |
|---|---|---|---|---|---|---|
| gpt-4o-mini | 86.27% | **51.61%** | **1.32** | 3.06 | **4.06** | **0** |
| Llama-3.1-8B-Instruct | 88.24% | **51.61%** | 1.42 | 3.03 | 3.97 | **0** |
| Qwen3-8B | **90.20%** | 41.94% | 1.39 | 2.90 | 4.03 | **0** |
| Granite-4.2-8B | **90.20%** | 22.58% | 1.74 | **3.26** | **4.06** | **0** |

### (c) Message-history few-shot (production format)

| Model | CI/CP acc. | Indicator model acc. | \|count diff\| | Coverage 1-5 | Distinctiveness 1-5 | Errors |
|---|---|---|---|---|---|---|
| gpt-4o-mini | 86.27% | **38.71%** | **1.26** | 3.06 | 4.00 | **0** |
| Llama-3.1-8B-Instruct | **94.12%** | 25.81% | 1.61 | 3.00 | 3.87 | **0** |
| Qwen3-8B | 90.20% | 35.48% | 1.39 | 2.77 | 4.13 | **0** |
| Granite-4.2-8B | 88.24% | **38.71%** | 1.52 | **3.26** | **4.19** | **0** |

### (d) GEPA-optimized

| Model | CI/CP acc. | Indicator model acc. | \|count diff\| | Coverage 1-5 | Distinctiveness 1-5 | Errors |
|---|---|---|---|---|---|---|
| gpt-4o-mini | 88.24% | **67.74%** | 1.68 | 2.87 | **4.06** | **0** |
| Llama-3.1-8B-Instruct | 90.20% | 54.84% | 1.94 | 2.87 | 3.90 | **0** |
| Qwen3-8B | 92.16% | 51.61% | 1.68 | 3.13 | 3.97 | **0** |
| Granite-4.2-8B | **96.08%** | **67.74%** | **1.58** | **3.26** | 4.03 | **0** |

## 3. Assertion Developer (n=92, CP-parent rows only)

### (a) Zero-shot

| Model | Basic concept acc. | Structure code acc. | Alignment 1-5 | Errors |
|---|---|---|---|---|
| gpt-4o-mini | **65.22%** | **48.91%** | 4.39 | **3** |
| Llama-3.1-8B-Instruct | 46.74% | 29.55% | 4.23 | 14 |
| Qwen3-8B | 53.26% | 46.74% | 4.16 | 15 |
| Granite-4.2-8B | 44.57% | 42.86% | **4.52** | 26 |

### (b) Prose few-shot

| Model | Basic concept acc. | Structure code acc. | Alignment 1-5 | Errors |
|---|---|---|---|---|
| gpt-4o-mini | **60.87%** | 45.65% | **4.51** | **8** |
| Llama-3.1-8B-Instruct | 54.35% | 36.26% | 4.36 | **8** |
| Qwen3-8B | 56.52% | 42.39% | 4.18 | 10 |
| Granite-4.2-8B | 52.17% | **48.91%** | 4.29 | 16 |

### (c) Message-history few-shot (production format)

| Model | Basic concept acc. | Structure code acc. | Alignment 1-5 | Errors |
|---|---|---|---|---|
| gpt-4o-mini | **60.87%** | 46.74% | **4.40** | 5 |
| Llama-3.1-8B-Instruct | 57.61% | 40.22% | 4.27 | **2** |
| Qwen3-8B | 57.61% | 44.57% | 4.39 | 10 |
| Granite-4.2-8B | 56.52% | **48.35%** | 4.25 | 3 |

### (d) GEPA-optimized

| Model | Basic concept acc. | Structure code acc. | Alignment 1-5 | Errors |
|---|---|---|---|---|
| gpt-4o-mini | **71.74%** | **63.04%** | **4.38** | 2 |
| Llama-3.1-8B-Instruct | 53.26% | 38.04% | 4.30 | **1** |
| Qwen3-8B | 47.83% | 42.39% | 3.99 | 11 |
| Granite-4.2-8B | 59.78% | 58.70% | 4.12 | 9 |

## 4. Findings (verified against the raw summary.json files, not just asserted)

**1. GEPA reliably wins the harder Concept Mapper metric (`indicator_model`), for all
four models — including three it was never optimized on.** GEPA was tuned against
gpt-4o-mini's zero-shot seed only. Checked against the best of (a/b/c) per model:
gpt-4o-mini +15.1pp, Llama +3.2pp, Qwen3 +9.7pp, Granite **+29.0pp** (worst cell in the
whole table, 19.35%, to tied-best, 67.74%). This does *not* extend cleanly to CI/CP
accuracy — that metric is already near-ceiling for every technique/model (86-96%), and
GEPA is not the best technique there for Llama (message-history reaches 94.12% vs
GEPA's 90.20%) or Qwen3/Granite (both tie their own best non-GEPA condition rather than
clearly beating it). **Scope this claim to `indicator_model`, not "the Concept Mapper
stage" generally.**

**2. "Zero-shot is enough" does not generalize — but *which* technique zero-shot loses
to is model-specific, not uniform.** On `indicator_model`, zero-shot is the *worst*
technique for Qwen3 (22.58%) and Granite (19.35%), but for Llama and gpt-4o-mini the
worst technique is message-history (25.81% / 38.71%) — zero-shot is actually
mid-to-good for both of those. So the real pattern is "the production message-history
format is the single most model-dependent-fragile condition," not simply "zero-shot is
bad for open models." Prose few-shot is the one technique that never hurts and mostly
helps `indicator_model` across every open model (Llama +6.5pp, Qwen3 +19.4pp, Granite
+3.2pp over zero-shot), while leaving gpt-4o-mini's CI/CP exactly unchanged (86.27% =
86.27%).

**3. GEPA transfer is a (task, model) property, not a model property.** The same
GEPA-optimized instruction that helps `indicator_model` for all four models on Concept
Mapper *hurts* two of them on Assertion Developer's `basic_concept` accuracy: it's
Qwen3's single worst condition on that stage (47.83%, below all three non-GEPA
techniques) and a net negative for Llama too (53.26%, below message-history's 57.61%).
It only clearly wins Assertion Developer for gpt-4o-mini (+6.5pp over its own best
non-GEPA technique) and Granite (+3.3pp over its best non-GEPA technique). A
prompt-optimization result from one stage/model pair should not be assumed to transfer
to a different stage, even holding the model fixed.

**4. Few-shot teaches schema validity for some models, not others — independent of the
model's accuracy on content.** Assertion Developer schema-error counts (a→b→c): Llama
14→8→2 (monotonic), Granite 26→16→3 (monotonic), Qwen3 15→10→10 (drops then plateaus),
gpt-4o-mini 3→8→5 (no monotonic pattern — actually worst *with* prose few-shot). Every
one of these errors is the same failure class: syntactically valid JSON with a
`basic_concept`/`structure_code` pairing `BASIC_CONCEPT_RULES` disallows — a
schema-validity problem, not a JSON-formatting problem.

**5. Negative result worth stating explicitly: Granite's marketed structured-output
strength did not show up zero-shot.** Granite-4.2-8B produced the single worst
zero-shot error count in the entire 16-cell Assertion Developer table (26/92, vs.
gpt-4o-mini's 3/92) despite being positioned for JSON/structured generation. Whatever
that training targeted did not transfer to this task's specific closed vocabulary (22
`basic_concept` values, model-specific `structure_code` sets) without examples — though
Granite recovers fastest of any model once given examples (26→16→3), and reaches the
best zero-shot alignment score in the whole table (4.52) despite the high error rate on
the rows that *do* pass validation.

## 5. Statistical-power caveat (apply before writing any of the above as a strong claim)

`indicator_model` accuracy is computed over **31 CP rows** — one row = 3.23 percentage
points, and a rough 95% binomial interval at these sample sizes is roughly ±17pp. Do
not treat single-cell differences (e.g. Llama's GEPA vs. its own zero-shot, +3.2pp) as
individually significant. The findings in §4 are stated as **patterns that repeat
across multiple models/techniques**, which is the right level of evidence for n=31 —
individual cells should not be cited in isolation in the paper. Assertion Developer's
n=92 is a meaningfully stronger evidence base and should carry more argumentative
weight where the two stages' findings could conflict.

## 6. Housekeeping needed before the paper is written from this data

1. **Delete or replace `experiments/mock_results.md` in both agents.** That file is
   100% fabricated placeholder data (clearly labeled as such), fully superseded by this
   document. Leaving it in the repo risks it being mistaken for real data later.
2. **The gpt-4o-mini Assertion Developer zero-shot baseline is already reconciled —
   no action needed, but worth knowing this was checked.** `fewshot_ablation_results.md`
   and `gepa_results.md`'s own final "Resolution" section both currently state the same
   numbers (65.22% / 48.91% / 4.39 / 3 errors) — the numbers used in §3 above. An
   earlier intermediate table inside `gepa_results.md` (its "Run 1"/"Run 2" sections,
   pre-dating the `vIi` rule-table fix) shows a different, superseded number (66.30% /
   45.65% / 4.23 / 2 errors) — that's a documented before/after within the same file,
   not an unresolved conflict between files.
3. **`_example_leak_markers` (in the ablation scripts) does not measure copying** —
   zero-shot runs (which see no examples at all) still trip the same marker, because it
   detects natural phrasing convergence, not reproduction. Do not cite leak-marker
   counts as evidence for or against verbatim copying in the paper.
4. **Nothing in this benchmark run is committed to git yet** — the new `outputs/*.jsonl`
   / `*_summary.json` files exist locally only. Decide on a commit message and push
   before switching machines again.
5. **How much runtime detail to put in the paper**: one sentence is enough for
   reproducibility — e.g. "open-source models were run locally via Ollama with Q4_K_M
   quantization; Qwen3-8B and Granite-4.2-8B in non-thinking mode; gpt-4.1-mini judged
   all conditions identically." The quantization-consistency decision (§1, discarding
   the LRZ/bitsandbytes Llama run) is worth one clause since it's a real methodological
   choice, not just an implementation detail.
