"""Rule-based quality evaluation for AssertionOutput records.

No gold labels required — checks internal consistency and structural validity
of each assertion record against the Saris & Gallhofer rule table.
"""

from __future__ import annotations

from typing import Any

from .assertion_rules import BASIC_CONCEPT_RULES
from .assertion_schemas import _SCALE_MARKERS


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
