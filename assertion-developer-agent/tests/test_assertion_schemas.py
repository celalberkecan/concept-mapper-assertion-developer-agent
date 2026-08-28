"""Tests for assertion_schemas.py — no API calls needed."""

import pytest
from pydantic import ValidationError

from assertion_developer.assertion_schemas import AssertionOutput

# ---------------------------------------------------------------------------
# Valid payloads
# ---------------------------------------------------------------------------

_VALID_SUBJECTIVE = {
    "parent_concept": "fear of crime",
    "input_indicator": "fear of burglary",
    "variable_type": "subjective",
    "basic_concept": "feelings",
    "domain": "burglary / home victimization",
    "structure_code": "rFy",
    "assertion": "The respondent fears that their home may be burglarized.",
    "rationale": "The indicator expresses an affective state of fear.",
    "warnings": [],
}

_VALID_OBJECTIVE = {
    "parent_concept": "age",
    "input_indicator": "age",
    "variable_type": "objective",
    "basic_concept": "demographics",
    "domain": "age",
    "structure_code": "xId",
    "assertion": "The respondent is of a specific chronological age.",
    "rationale": "Age is a directly observable demographic attribute.",
    "warnings": [],
}


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------


def test_valid_subjective_feelings():
    out = AssertionOutput(**_VALID_SUBJECTIVE)
    assert out.variable_type == "subjective"
    assert out.basic_concept == "feelings"
    assert out.structure_code == "rFy"


def test_valid_objective_demographics():
    out = AssertionOutput(**_VALID_OBJECTIVE)
    assert out.variable_type == "objective"
    assert out.basic_concept == "demographics"
    assert out.structure_code == "xId"


def test_structure_id_derived_not_model_set():
    """structure_id must be derived from rule table, even if model passes a wrong value."""
    payload = {**_VALID_SUBJECTIVE, "structure_id": "structure_99_WRONG"}
    out = AssertionOutput(**payload)
    assert out.structure_id == "structure_2"  # feelings → structure_2 per rule table


def test_structure_id_attached_when_absent():
    """structure_id is set even when not provided."""
    payload = {k: v for k, v in _VALID_OBJECTIVE.items() if k != "structure_id"}
    out = AssertionOutput(**payload)
    assert out.structure_id == "structure_1"  # demographics → structure_1


# ---------------------------------------------------------------------------
# Rejection tests
# ---------------------------------------------------------------------------


def test_invalid_basic_concept_rejected():
    payload = {**_VALID_SUBJECTIVE, "basic_concept": "nonexistent_concept"}
    with pytest.raises((ValidationError, ValueError)):
        AssertionOutput(**payload)


def test_wrong_variable_type_for_basic_concept_rejected():
    """demographics is objective; passing variable_type=subjective must fail."""
    payload = {
        **_VALID_OBJECTIVE,
        "variable_type": "subjective",   # wrong — demographics is objective
    }
    with pytest.raises((ValidationError, ValueError)):
        AssertionOutput(**payload)


def test_invalid_structure_code_rejected():
    """rFy is only valid for feelings, not for demographics."""
    payload = {
        **_VALID_OBJECTIVE,
        "structure_code": "rFy",  # not allowed for demographics
    }
    with pytest.raises((ValidationError, ValueError)):
        AssertionOutput(**payload)


def test_assertion_with_question_mark_rejected():
    payload = {**_VALID_SUBJECTIVE, "assertion": "Does the respondent fear burglary?"}
    with pytest.raises((ValidationError, ValueError)):
        AssertionOutput(**payload)


def test_assertion_with_scale_marker_rejected():
    payload = {
        **_VALID_SUBJECTIVE,
        "assertion": "On a scale of 1 to 5, the respondent fears burglary.",
    }
    with pytest.raises((ValidationError, ValueError)):
        AssertionOutput(**payload)


def test_missing_required_field_rejected():
    payload = {k: v for k, v in _VALID_SUBJECTIVE.items() if k != "assertion"}
    with pytest.raises((ValidationError, ValueError)):
        AssertionOutput(**payload)
