"""Evaluation for AssertionOutput records.

Two independent evaluation modes, matching the professor's rubric
(Protocol Meeting Berke & Leonardo.pdf, "2. Development of assertions"):

- Rule-based (evaluate_single/evaluate_batch, no gold needed): internal consistency
  and structural validity against the Saris & Gallhofer rule table. Covers
  "Correct semantic structure" only loosely (is the code *allowed* for the concept,
  not whether it's the *specific* code a human would have picked).
- Gold-based (evaluate_single_against_gold/evaluate_batch_against_gold): compares
  against data/*_for_assertion_agent.xlsx, "Source Items + Assertions (cor)" sheet
  (CP-parent rows only, via io.read_assertion_gold_xlsx). Covers all three rubric
  criteria directly:
    - "Identification of basic concept" (5/5 objectivity) -> exact match against
      basic_concept_key.
    - "Correct semantic structure" (5/5 objectivity) -> exact match against
      structure_code_gold (falls back to the rule-based allowed-code check for the
      few rows missing a gold code).
    - "Concept-assertion alignment" (4/5 objectivity, i.e. not fully objective) ->
      LLM-judge score against gold_assertion (llm_judge.py), since there is no
      single correct assertion wording.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .assertion_rules import BASIC_CONCEPT_RULES
from .assertion_schemas import _SCALE_MARKERS

if TYPE_CHECKING:
    from survey_agent_lib.llm_clients.base import BaseLLMClient


# ---------------------------------------------------------------------------
# Per-record evaluation
# ---------------------------------------------------------------------------


def evaluate_single(record: dict[str, Any]) -> dict[str, Any]:
    """Run rule-based quality checks on one assertion record.

    Returns a flat dict of check results (bool | None) plus a summary flag.
    """
    variable_type = record.get("variable_type", "")
    basic_concept = record.get("basic_concept", "")
    structure_code = record.get("structure_code", "")
    assertion = record.get("assertion", "")

    result: dict[str, Any] = {
        "input_indicator": record.get("input_indicator", "?"),
        "parent_concept": record.get("parent_concept", "?"),
    }

    # Check 1: variable_type is known
    result["valid_variable_type"] = variable_type in ("subjective", "objective")

    # Check 2: basic_concept is known
    result["valid_basic_concept"] = basic_concept in BASIC_CONCEPT_RULES

    # Check 3: basic_concept matches variable_type
    if result["valid_basic_concept"] and result["valid_variable_type"]:
        expected_vt = BASIC_CONCEPT_RULES[basic_concept]["variable_type"]
        result["variable_type_consistent"] = variable_type == expected_vt
    else:
        result["variable_type_consistent"] = None

    # Check 4: structure_code is allowed for basic_concept
    if result["valid_basic_concept"]:
        allowed = BASIC_CONCEPT_RULES[basic_concept]["allowed_codes"]
        result["valid_structure_code"] = structure_code in allowed
    else:
        result["valid_structure_code"] = None

    # Check 5: assertion is a declarative statement (not a question)
    result["not_a_question"] = not assertion.strip().endswith("?")

    # Check 6: no response scale or option markers in assertion
    lower = assertion.lower()
    result["no_scale_markers"] = not any(m in lower for m in _SCALE_MARKERS)

    # Overall: all boolean checks must pass
    bool_checks = [v for v in result.values() if isinstance(v, bool)]
    result["all_pass"] = all(bool_checks)

    return result


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def evaluate_batch(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate all records and return one result dict per record."""
    return [evaluate_single(r) for r in records if "error" not in r or not r["error"]]


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def compute_summary(eval_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate pass-rate statistics over all evaluation results."""
    total = len(eval_results)
    if total == 0:
        return {"total": 0}

    def _rate(key: str) -> float | None:
        vals = [r[key] for r in eval_results if isinstance(r.get(key), bool)]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "total_records": total,
        "valid_variable_type_rate": _rate("valid_variable_type"),
        "valid_basic_concept_rate": _rate("valid_basic_concept"),
        "variable_type_consistent_rate": _rate("variable_type_consistent"),
        "valid_structure_code_rate": _rate("valid_structure_code"),
        "not_a_question_rate": _rate("not_a_question"),
        "no_scale_markers_rate": _rate("no_scale_markers"),
        "all_pass_rate": _rate("all_pass"),
        "n_all_pass": sum(1 for r in eval_results if r.get("all_pass") is True),
        "n_any_fail": sum(1 for r in eval_results if r.get("all_pass") is False),
    }


# ---------------------------------------------------------------------------
# Gold-based evaluation (Protocol Meeting Berke & Leonardo.pdf rubric)
# ---------------------------------------------------------------------------

# The gold sheet (transcribed independently from the same Saris & Gallhofer table)
# carries the same lowercase-l/capital-I confusion that assertion_rules.py had before
# it was fixed against the source PDF. Normalise both sides before comparing so a
# stale transcription artifact in the gold data doesn't count a now-correct
# structure_code as wrong.
_LEGACY_STRUCTURE_CODE_ALIASES = {
    "xle": "xIe",
    "xli": "xIi",
    "xlv": "vIi",
    "xlf": "xIf",
    # "xIv" was our own earlier (incorrect) transcription of the values structure code —
    # the source PDF's "Structures for subjective variables" table shows "vIi", not "xIv"
    # (subject=value, not subject=topic). Alias it so any leftover data using our old fix
    # still normalises to the now-correct code.
    "xIv": "vIi",
}


def _normalize_structure_code(code: str | None) -> str | None:
    if code is None:
        return None
    return _LEGACY_STRUCTURE_CODE_ALIASES.get(code, code)


def evaluate_single_against_gold(
    gold: dict,
    prediction: dict,
    judge_client: "BaseLLMClient | None" = None,
) -> dict[str, Any]:
    """Compare one prediction against its gold row on all three rubric criteria.

    Args:
        gold: A row dict from io.read_assertion_gold_xlsx.
        prediction: A prediction record (must contain example_id + AssertionOutput fields).
        judge_client: Optional LLM client. If given and prediction has no error, scores
                      concept-assertion alignment against gold_assertion via
                      llm_judge.judge_assertion_alignment. Left None, alignment fields
                      are None (no LLM calls).
    """
    example_id = gold["example_id"]
    parent_concept = gold["input_topic_parent_concept"]
    indicator_name = gold["indicator_concept_gold"]
    gold_basic_concept = gold["basic_concept_key"]
    gold_structure_code = _normalize_structure_code(gold.get("structure_code_gold"))
    gold_assertion = gold["gold_assertion"]

    pred_basic_concept = prediction.get("basic_concept", "")
    pred_structure_code = prediction.get("structure_code", "")
    pred_assertion = prediction.get("assertion", "")

    # Criterion 1: "Identification of basic concept" (5/5 objectivity) — exact match.
    basic_concept_correct = pred_basic_concept == gold_basic_concept

    # Criterion 2: "Correct semantic structure" (5/5 objectivity) — exact match against
    # the specific gold code. Falls back to "is it an allowed code at all" for the rows
    # missing a gold structure_code, rather than silently skipping the check.
    if gold_structure_code:
        structure_code_correct: bool | None = pred_structure_code == gold_structure_code
    else:
        rule = BASIC_CONCEPT_RULES.get(pred_basic_concept)
        structure_code_correct = (
            pred_structure_code in rule["allowed_codes"] if rule else None
        )

    result: dict[str, Any] = {
        "example_id": example_id,
        "parent_concept": parent_concept,
        "indicator_name": indicator_name,
        "gold_basic_concept": gold_basic_concept,
        "pred_basic_concept": pred_basic_concept,
        "basic_concept_correct": basic_concept_correct,
        "gold_structure_code": gold_structure_code,
        "pred_structure_code": pred_structure_code,
        "structure_code_correct": structure_code_correct,
        "gold_assertion": gold_assertion,
        "pred_assertion": pred_assertion,
        "alignment_score": None,
        "alignment_feedback": None,
    }

    # Criterion 3: "Concept-assertion alignment" (4/5 objectivity — not fully objective,
    # so graded via LLM-judge rather than exact-matched).
    if judge_client is not None and "error" not in prediction:
        from .llm_judge import judge_assertion_alignment

        judge_result = judge_assertion_alignment(
            judge_client, parent_concept, indicator_name, gold_assertion, pred_assertion
        )
        result["alignment_score"] = judge_result["alignment_score"]
        result["alignment_feedback"] = judge_result["feedback"]

    return result


def evaluate_batch_against_gold(
    gold_rows: list[dict],
    predictions: list[dict],
    judge_client: "BaseLLMClient | None" = None,
) -> list[dict[str, Any]]:
    """Evaluate all predictions against gold rows, joined on example_id.

    Gold rows with no matching prediction are recorded with null metric values.
    """
    pred_by_id: dict[str, dict] = {p["example_id"]: p for p in predictions if "example_id" in p}

    results: list[dict] = []
    for gold in gold_rows:
        eid = gold["example_id"]
        pred = pred_by_id.get(eid)
        if pred is None:
            results.append({
                "example_id": eid,
                "parent_concept": gold["input_topic_parent_concept"],
                "indicator_name": gold["indicator_concept_gold"],
                "gold_basic_concept": gold["basic_concept_key"],
                "pred_basic_concept": None,
                "basic_concept_correct": None,
                "gold_structure_code": gold.get("structure_code_gold"),
                "pred_structure_code": None,
                "structure_code_correct": None,
                "gold_assertion": gold["gold_assertion"],
                "pred_assertion": None,
                "alignment_score": None,
                "alignment_feedback": None,
                "error": "no prediction",
            })
        else:
            row = evaluate_single_against_gold(gold, pred, judge_client=judge_client)
            row["error"] = pred.get("error")
            results.append(row)

    return results


def compute_gold_summary(eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics over gold-based evaluation records."""
    total = len(eval_records)
    if total == 0:
        return {"total": 0}

    bc_results = [r["basic_concept_correct"] for r in eval_records if r["basic_concept_correct"] is not None]
    basic_concept_accuracy = sum(bc_results) / len(bc_results) if bc_results else None

    sc_results = [r["structure_code_correct"] for r in eval_records if r["structure_code_correct"] is not None]
    structure_code_accuracy = sum(sc_results) / len(sc_results) if sc_results else None

    alignment_scores = [r["alignment_score"] for r in eval_records if r["alignment_score"] is not None]
    mean_alignment_score = sum(alignment_scores) / len(alignment_scores) if alignment_scores else None

    return {
        "total_rows": total,
        "basic_concept_accuracy": round(basic_concept_accuracy, 4) if basic_concept_accuracy is not None else None,
        "structure_code_accuracy": round(structure_code_accuracy, 4) if structure_code_accuracy is not None else None,
        "mean_alignment_score_1to5": round(mean_alignment_score, 4) if mean_alignment_score is not None else None,
        "n_missing_predictions": sum(1 for r in eval_records if r["pred_basic_concept"] is None),
        "n_errors": sum(1 for r in eval_records if r.get("error")),
    }
