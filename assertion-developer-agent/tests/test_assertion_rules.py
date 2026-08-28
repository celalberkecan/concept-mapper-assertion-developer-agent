"""Tests for assertion_rules.py — no API calls needed."""

import pytest

from assertion_developer.assertion_rules import (
    BASIC_CONCEPT_RULES,
    OBJECTIVE_CONCEPTS,
    SUBJECTIVE_CONCEPTS,
    get_allowed_codes,
    get_default_code,
    get_structure_id,
    get_valid_basic_concepts,
    validate_combination,
)


def test_all_basic_concepts_have_required_keys():
    required = {"variable_type", "allowed_codes", "default_code", "structure_id"}
    for name, rule in BASIC_CONCEPT_RULES.items():
        missing = required - rule.keys()
        assert not missing, f"Rule for {name!r} is missing keys: {missing}"


def test_all_default_codes_are_in_allowed_codes():
    for name, rule in BASIC_CONCEPT_RULES.items():
        assert rule["default_code"] in rule["allowed_codes"], (
            f"default_code {rule['default_code']!r} not in allowed_codes for {name!r}"
        )


def test_get_allowed_codes_feelings():
    codes = get_allowed_codes("feelings")
    assert set(codes) == {"xIf", "xFy", "xPf", "rFy"}


def test_get_default_code_feelings():
    assert get_default_code("feelings") == "rFy"


def test_get_default_code_behavior():
    assert get_default_code("behavior") == "rD"


def test_get_structure_id_feelings():
    assert get_structure_id("feelings") == "structure_2"


def test_get_structure_id_demographics():
    assert get_structure_id("demographics") == "structure_1"


def test_get_structure_id_behavior():
    assert get_structure_id("behavior") == "structure_3"


def test_get_valid_basic_concepts_subjective():
    subjective = get_valid_basic_concepts("subjective")
    assert set(subjective) == set(SUBJECTIVE_CONCEPTS)
    assert len(subjective) == 14


def test_get_valid_basic_concepts_objective():
    objective = get_valid_basic_concepts("objective")
    assert set(objective) == set(OBJECTIVE_CONCEPTS)
    assert len(objective) == 8


def test_validate_combination_valid():
    validate_combination("subjective", "feelings", "rFy")   # should not raise


def test_validate_combination_invalid_concept():
    with pytest.raises(ValueError, match="Unknown basic_concept"):
        validate_combination("subjective", "made_up", "rFy")


def test_validate_combination_wrong_variable_type():
    with pytest.raises(ValueError, match="variable_type"):
        validate_combination("subjective", "demographics", "xId")


def test_validate_combination_wrong_code():
    with pytest.raises(ValueError, match="not allowed"):
        validate_combination("objective", "demographics", "rFy")


def test_get_unknown_concept_raises():
    with pytest.raises(ValueError, match="Unknown basic_concept"):
        get_allowed_codes("totally_made_up")
