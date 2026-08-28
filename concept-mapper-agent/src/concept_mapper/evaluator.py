"""Evaluation metrics for Concept Mapper predictions.

Covers:
- Exact match: CI/CP classification, indicator model (CP rows only)
- Indicator count: predicted vs gold, absolute difference (CP rows only)
- Indicator coverage/distinctiveness: LLM-as-judge score against the gold indicator
  list (CP rows only, opt-in via judge_client) — see llm_judge.py. There is no single
  correct indicator list, so this is graded rather than exact-matched, mirroring the
  rubric's own "Indicator distinctiveness" / "Coverage of the construct domain" criteria.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from survey_agent_lib.llm_clients.base import BaseLLMClient


# ---------------------------------------------------------------------------
# Per-row evaluation
# ---------------------------------------------------------------------------


def evaluate_single(
    gold: dict,
    prediction: dict,
    judge_client: "BaseLLMClient | None" = None,
) -> dict[str, Any]:
    """Compute evaluation metrics for one gold row against one prediction.

    Args:
        gold: A row dict from read_concept_mapper_gold_xlsx.
        prediction: A prediction record dict saved by run-batch
                    (must contain concept_id + all ConceptMap fields).
        judge_client: Optional LLM client. If given and the gold row is CP, scores the
                      predicted indicators against gold_indicators_conceptual via
                      llm_judge.judge_indicator_quality. Left None, indicator quality
                      fields are None (no LLM calls, matches prior behaviour).

    Returns:
        A flat dict of evaluation metrics for this row.
    """
    concept_id = gold["concept_id"]
    input_topic = gold["input_topic_parent_concept"]

    gold_ci_cp: str = gold["concept_level_ci_cp_gold"]
    gold_indicator_model: str = gold["concept_level_indicator_model_gold"]  # "NA" for CI
    gold_indicator_count: int = int(gold["indicator_count_gold"])

    pred_ci_cp: str = prediction.get("ci_or_cp", "")
    pred_indicator_model: str = prediction.get("indicator_model", "")
    pred_indicators: list = prediction.get("indicators", [])
    pred_indicator_count: int = len(pred_indicators)

    # CI/CP exact match
    ci_cp_correct: bool = pred_ci_cp == gold_ci_cp

    # Indicator model exact match — only meaningful for CP rows
    if gold_ci_cp == "CP":
        indicator_model_correct: bool | None = pred_indicator_model == gold_indicator_model
    else:
        indicator_model_correct = None  # not applicable for CI

    # Indicator count difference — only meaningful for CP rows
    if gold_ci_cp == "CP":
        indicator_count_abs_diff: int | None = abs(pred_indicator_count - gold_indicator_count)
    else:
        indicator_count_abs_diff = None

    result = {
        "concept_id": concept_id,
        "input_topic": input_topic,
        "gold_ci_cp": gold_ci_cp,
        "pred_ci_cp": pred_ci_cp,
        "ci_cp_correct": ci_cp_correct,
        "gold_indicator_model": gold_indicator_model,
        "pred_indicator_model": pred_indicator_model,
        "indicator_model_correct": indicator_model_correct,
        "gold_indicator_count": gold_indicator_count,
        "pred_indicator_count": pred_indicator_count,
        "indicator_count_abs_diff": indicator_count_abs_diff,
        "indicator_coverage_score": None,
        "indicator_distinctiveness_score": None,
        "indicator_judge_score": None,
        "indicator_judge_feedback": None,
    }

    if judge_client is not None and gold_ci_cp == "CP" and "error" not in prediction:
        from .llm_judge import judge_indicator_quality

        judge_result = judge_indicator_quality(
            judge_client,
            topic=input_topic,
            gold_indicators=gold.get("gold_indicators_conceptual", ""),
            predicted_indicators=pred_indicators,
        )
        result["indicator_coverage_score"] = judge_result["coverage_score"]
        result["indicator_distinctiveness_score"] = judge_result["distinctiveness_score"]
        result["indicator_judge_score"] = judge_result["score"]
        result["indicator_judge_feedback"] = judge_result["feedback"]

    return result


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def evaluate_batch(
    gold_rows: list[dict],
    predictions: list[dict],
    judge_client: "BaseLLMClient | None" = None,
) -> list[dict[str, Any]]:
    """Evaluate all predictions against gold rows.

    Joins on concept_id. Predictions that cannot be matched to a gold row are
    skipped with a warning. Gold rows with no matching prediction are recorded
    with null metric values.

    judge_client: passed through to evaluate_single (see its docstring). Costs one LLM
                  call per CP row when set — omit to keep evaluation free/deterministic.
    """
    pred_by_id: dict[str, dict] = {p["concept_id"]: p for p in predictions if "concept_id" in p}

    results: list[dict] = []
    for gold in gold_rows:
        cid = gold["concept_id"]
        pred = pred_by_id.get(cid)
        if pred is None:
            # No prediction for this row — record as missing
            results.append({
                "concept_id": cid,
                "input_topic": gold["input_topic_parent_concept"],
                "gold_ci_cp": gold["concept_level_ci_cp_gold"],
                "pred_ci_cp": None,
                "ci_cp_correct": None,
                "gold_indicator_model": gold["concept_level_indicator_model_gold"],
                "pred_indicator_model": None,
                "indicator_model_correct": None,
                "gold_indicator_count": gold["indicator_count_gold"],
                "pred_indicator_count": None,
                "indicator_count_abs_diff": None,
                "indicator_coverage_score": None,
                "indicator_distinctiveness_score": None,
                "indicator_judge_score": None,
                "indicator_judge_feedback": None,
                "error": "no prediction",
            })
        else:
            row = evaluate_single(gold, pred, judge_client=judge_client)
            if "error" in pred:
                row["error"] = pred["error"]
            else:
                row["error"] = None
            results.append(row)

    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def compute_summary(eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics over all evaluation records."""
    total = len(eval_records)
    if total == 0:
        return {"total": 0}

    # CI/CP accuracy (all rows)
    ci_cp_results = [r["ci_cp_correct"] for r in eval_records if r["ci_cp_correct"] is not None]
    ci_cp_accuracy = sum(ci_cp_results) / len(ci_cp_results) if ci_cp_results else None

    # Indicator model accuracy (CP rows only)
    im_results = [r["indicator_model_correct"] for r in eval_records if r["indicator_model_correct"] is not None]
    indicator_model_accuracy = sum(im_results) / len(im_results) if im_results else None

    # Indicator count abs diff (CP rows only)
    count_diffs = [r["indicator_count_abs_diff"] for r in eval_records if r["indicator_count_abs_diff"] is not None]
    mean_count_abs_diff = sum(count_diffs) / len(count_diffs) if count_diffs else None

    # Indicator quality — LLM-judge (CP rows only, only present if evaluate_batch
    # was called with a judge_client)
    coverage_scores = [r["indicator_coverage_score"] for r in eval_records if r["indicator_coverage_score"] is not None]
    mean_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else None
    distinctiveness_scores = [r["indicator_distinctiveness_score"] for r in eval_records if r["indicator_distinctiveness_score"] is not None]
    mean_distinctiveness = sum(distinctiveness_scores) / len(distinctiveness_scores) if distinctiveness_scores else None

    # Breakdown by gold class
    n_ci_gold = sum(1 for r in eval_records if r["gold_ci_cp"] == "CI")
    n_cp_gold = sum(1 for r in eval_records if r["gold_ci_cp"] == "CP")

    return {
        "total_rows": total,
        "n_gold_ci": n_ci_gold,
        "n_gold_cp": n_cp_gold,
        "ci_cp_accuracy": round(ci_cp_accuracy, 4) if ci_cp_accuracy is not None else None,
        "indicator_model_accuracy_cp_only": round(indicator_model_accuracy, 4) if indicator_model_accuracy is not None else None,
        "mean_indicator_count_abs_diff_cp_only": round(mean_count_abs_diff, 4) if mean_count_abs_diff is not None else None,
        "mean_indicator_coverage_1to5_cp_only": round(mean_coverage, 4) if mean_coverage is not None else None,
        "mean_indicator_distinctiveness_1to5_cp_only": round(mean_distinctiveness, 4) if mean_distinctiveness is not None else None,
        "n_missing_predictions": sum(1 for r in eval_records if r["pred_ci_cp"] is None),
        "n_errors": sum(1 for r in eval_records if r.get("error")),
    }
