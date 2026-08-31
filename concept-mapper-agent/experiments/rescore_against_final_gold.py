"""Re-score the existing Concept Mapper predictions against the FINAL gold annotation.

Why this exists
---------------
Every Concept Mapper run committed so far (`run_fewshot_ablation.py`, `run_gepa_eval.py`,
`run_model_sweep.py`) scored predictions against
`data/gesis_concept_mapper_assertion_evaluation_adjusted.xlsx`, `Concept Mapper Gold`
sheet, `*_leo` columns — 51 topics, 20 CI / 31 CP. That annotation has since been
superseded: the `*_leo` relabel was folded into the `*_gold` columns and revised again,
and the final version lives in the assertion agent's data folder (46 topics, 21 CI /
25 CP). Five topics were dropped and ten of the 46 survivors carry a different CI/CP
label than the `*_leo` columns gave them.

Generation and scoring are independent steps here: no prompt ever contained a gold
label, so the stored predictions in `experiments/outputs/*.jsonl` remain valid and only
the scoring pass has to be repeated. This script does exactly that, reusing
`evaluator.evaluate_batch` / `evaluator.compute_summary` unchanged so the metric
definitions stay identical to the original runs.

No API key required
-------------------
`evaluate_batch(..., judge_client=None)` skips the LLM-as-judge step, which is the only
part of the evaluation that costs a model call. The four deterministic metrics
(CI/CP accuracy, indicator_model accuracy, mean |indicator count diff|, error counts)
are recomputed. The two judge metrics (coverage, distinctiveness) are NOT recomputed and
are reported as None — the originals used gpt-4.1-mini and would need a key to redo.

Usage (run from concept-mapper-agent/):
    python experiments/rescore_against_final_gold.py

Output:
    experiments/outputs/{model_tag}_summary_final_gold.json   (one per model)
    a comparison table printed to stdout (old vs. new, per model x technique)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from concept_mapper.evaluator import compute_summary, evaluate_batch  # noqa: E402
from concept_mapper.io import read_concept_mapper_gold_xlsx, read_jsonl  # noqa: E402

# The final annotation lives in the sibling package's data folder. It has no `*_leo`
# columns, so `read_concept_mapper_gold_xlsx`'s leo-preference branch is inert here and
# the `*_gold` columns are read directly — which is what we want.
FINAL_GOLD_PATH = (
    REPO_ROOT.parent
    / "assertion-developer-agent"
    / "data"
    / "gesis_concept_mapper_assertion_evaluation_adjusted_for_assertion_agent_final.xlsx"
)
OLD_GOLD_PATH = REPO_ROOT / "data" / "gesis_concept_mapper_assertion_evaluation_adjusted.xlsx"
OUT_DIR = REPO_ROOT / "experiments" / "outputs"

# (file prefix, display name). gpt-4o-mini's outputs are the unprefixed ones written by
# run_fewshot_ablation.py / run_gepa_eval.py before the model sweep existed.
MODELS: list[tuple[str, str]] = [
    ("", "gpt-4o-mini"),
    ("llama-3.1-8b-instruct_", "Llama-3.1-8B-Instruct"),
    ("qwen3-8b_", "Qwen3-8B"),
    ("granite-4.2-8b_", "Granite-4.2-8B"),
]

VARIANTS: list[tuple[str, str]] = [
    ("a_zero_shot", "zero-shot"),
    ("b_prose_fewshot", "prose few-shot"),
    ("c_message_history_fewshot", "msg-history"),
    ("d_gepa_optimized", "GEPA"),
]

# gpt-4o-mini has no *_summary.json (it predates run_model_sweep.py), so its old numbers
# are transcribed from the two results documents instead. Sources:
#   a/b/c -> experiments/fewshot_ablation_results.md, "Results" table
#   d     -> experiments/gepa_results.md, "(2) Full 51-row gold set" table
OLD_GPT4O_MINI: dict[str, dict[str, float]] = {
    "a_zero_shot": {"ci_cp": 0.8627, "ind_model": 0.5161, "count_diff": 1.3871},
    "b_prose_fewshot": {"ci_cp": 0.8627, "ind_model": 0.5161, "count_diff": 1.3226},
    "c_message_history_fewshot": {"ci_cp": 0.8627, "ind_model": 0.3871, "count_diff": 1.2581},
    "d_gepa_optimized": {"ci_cp": 0.8824, "ind_model": 0.6774, "count_diff": 1.68},
}


def load_old_summary(prefix: str, variant: str) -> dict[str, float | None]:
    """Return the previously reported metrics for one model/technique cell.

    Open models read from their committed `{prefix}summary.json`; gpt-4o-mini falls back
    to the transcribed constants above.
    """
    if prefix == "":
        old = OLD_GPT4O_MINI[variant]
        return {"ci_cp": old["ci_cp"], "ind_model": old["ind_model"], "count_diff": old["count_diff"]}

    summary_path = OUT_DIR / f"{prefix}summary.json"
    if not summary_path.exists():
        return {"ci_cp": None, "ind_model": None, "count_diff": None}

    with open(summary_path, encoding="utf-8") as f:
        block = json.load(f).get(variant, {})
    return {
        "ci_cp": block.get("ci_cp_accuracy"),
        "ind_model": block.get("indicator_model_accuracy_cp_only"),
        "count_diff": block.get("mean_indicator_count_abs_diff_cp_only"),
    }


def fmt_pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.2f}"


def fmt_delta(new: float | None, old: float | None) -> str:
    if new is None or old is None:
        return "   n/a"
    return f"{(new - old) * 100:+6.2f}"


def main() -> None:
    if not FINAL_GOLD_PATH.exists():
        raise SystemExit(f"Final gold workbook not found: {FINAL_GOLD_PATH}")

    final_gold = read_concept_mapper_gold_xlsx(FINAL_GOLD_PATH)
    old_gold = read_concept_mapper_gold_xlsx(OLD_GOLD_PATH)

    final_ids = {row["concept_id"] for row in final_gold}
    dropped = sorted({row["concept_id"] for row in old_gold} - final_ids)
    relabeled = sorted(
        row["concept_id"]
        for row in final_gold
        for prev in old_gold
        if prev["concept_id"] == row["concept_id"]
        and prev["concept_level_ci_cp_gold"] != row["concept_level_ci_cp_gold"]
    )

    n_ci = sum(1 for r in final_gold if r["concept_level_ci_cp_gold"] == "CI")
    n_cp = sum(1 for r in final_gold if r["concept_level_ci_cp_gold"] == "CP")

    print(f"Final gold : {len(final_gold)} topics ({n_ci} CI / {n_cp} CP)  <- {FINAL_GOLD_PATH.name}")
    print(f"Old gold   : {len(old_gold)} topics  <- {OLD_GOLD_PATH.name}")
    print(f"Dropped    : {len(dropped)} topics no longer scored ({', '.join(dropped)})")
    print(f"Relabeled  : {len(relabeled)} topics changed CI/CP ({', '.join(relabeled)})")
    print()

    header = (
        f"| {'Model':<22} | {'Technique':<14} | {'CI/CP old':>9} | {'CI/CP new':>9} | {'d':>6} "
        f"| {'IndM old':>8} | {'IndM new':>8} | {'d':>6} | {'|cnt| new':>9} | {'err':>3} |"
    )
    print(header)
    print("|" + "|".join("-" * (len(part) + 2) for part in header.split("|")[1:-1]) + "|")

    for prefix, display in MODELS:
        summaries: dict[str, dict] = {}
        for variant, variant_label in VARIANTS:
            pred_path = OUT_DIR / f"{prefix}{variant}.jsonl"
            if not pred_path.exists():
                print(f"| {display:<22} | {variant_label:<14} | missing predictions: {pred_path.name}")
                continue

            predictions = read_jsonl(pred_path)
            records = evaluate_batch(final_gold, predictions, judge_client=None)
            summary = compute_summary(records)
            # Predictions for topics the final annotation dropped are simply not scored
            # (evaluate_batch iterates gold rows, so unmatched predictions fall away).
            summary["n_predictions_ignored"] = sum(
                1 for pred in predictions if pred.get("concept_id") not in final_ids
            )
            summaries[variant] = summary

            old = load_old_summary(prefix, variant)
            print(
                f"| {display:<22} | {variant_label:<14} "
                f"| {fmt_pct(old['ci_cp']):>9} | {fmt_pct(summary['ci_cp_accuracy']):>9} "
                f"| {fmt_delta(summary['ci_cp_accuracy'], old['ci_cp']):>6} "
                f"| {fmt_pct(old['ind_model']):>8} "
                f"| {fmt_pct(summary['indicator_model_accuracy_cp_only']):>8} "
                f"| {fmt_delta(summary['indicator_model_accuracy_cp_only'], old['ind_model']):>6} "
                f"| {summary['mean_indicator_count_abs_diff_cp_only']:>9.2f} "
                f"| {summary['n_errors']:>3} |"
            )

        if summaries:
            out_path = OUT_DIR / f"{prefix or 'gpt-4o-mini_'}summary_final_gold.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summaries, f, indent=2)

    print()
    print("Coverage / distinctiveness are NOT in the new summaries: they need the")
    print("gpt-4.1-mini judge and therefore an API key. Carry the old values forward")
    print("with a footnote, or re-run with a judge_client to refresh them.")


if __name__ == "__main__":
    main()
