import json

import pytest

from concept_mapper.parser import extract_json_object, parse_concept_map

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VALID_CI = {
    "input_topic": "age",
    "ci_or_cp": "CI",
    "indicator_model": "NA",
    "construct_definition": "The respondent's age.",
    "indicators": [],
    "rationale": "Age is direct.",
    "warnings": [],
}

_VALID_CP = {
    "input_topic": "fear of crime",
    "ci_or_cp": "CP",
    "indicator_model": "formative",
    "construct_definition": "A multidimensional fear construct.",
    "indicators": [
        {"name": "fear of burglary", "definition": "Worry about burglary.", "role": "component"},
        {"name": "fear of assault", "definition": "Worry about assault.", "role": "component"},
    ],
    "rationale": "Covers multiple crime-specific fears.",
    "warnings": [],
}

VALID_CI_JSON = json.dumps(_VALID_CI, indent=2)
VALID_CP_JSON = json.dumps(_VALID_CP, indent=2)


# ---------------------------------------------------------------------------
# extract_json_object
# ---------------------------------------------------------------------------


def test_extract_bare_json():
    result = extract_json_object(VALID_CI_JSON)
    assert json.loads(result) == _VALID_CI


def test_extract_json_with_prose_before():
    text = "Here is the concept map:\n\n" + VALID_CI_JSON
    result = extract_json_object(text)
    assert json.loads(result)["ci_or_cp"] == "CI"


def test_extract_json_with_code_fence():
    text = f"```json\n{VALID_CI_JSON}\n```"
    result = extract_json_object(text)
    assert json.loads(result)["input_topic"] == "age"


def test_extract_json_with_trailing_text():
    text = VALID_CI_JSON + "\n\nNote: this is a CI concept."
    result = extract_json_object(text)
    assert json.loads(result)["ci_or_cp"] == "CI"


def test_extract_no_json_raises():
    with pytest.raises(ValueError, match="No JSON object"):
        extract_json_object("This is plain text with no braces.")


def test_extract_unterminated_raises():
    with pytest.raises(ValueError, match="[Uu]nterminated|balanced"):
        extract_json_object('{"input_topic": "age"')


# ---------------------------------------------------------------------------
# parse_concept_map
# ---------------------------------------------------------------------------


def test_parse_valid_ci():
    cm = parse_concept_map(VALID_CI_JSON)
    assert cm.input_topic == "age"
    assert cm.ci_or_cp == "CI"
    assert cm.indicators == []


def test_parse_valid_cp():
    cm = parse_concept_map(VALID_CP_JSON)
    assert cm.ci_or_cp == "CP"
    assert len(cm.indicators) == 2
    assert cm.indicators[0].role == "component"


def test_parse_with_preamble():
    cm = parse_concept_map("Sure! Here you go:\n" + VALID_CI_JSON)
    assert cm.ci_or_cp == "CI"


def test_parse_no_json_raises_value_error():
    with pytest.raises(ValueError, match="No JSON object"):
        parse_concept_map("Sorry, I cannot help with that.")


def test_parse_schema_violation_raises():
    # CP with indicator_model="NA" should fail validation
    bad = {**_VALID_CP, "indicator_model": "NA"}
    with pytest.raises(ValueError, match="[Ss]chema|validation"):
        parse_concept_map(json.dumps(bad))


def test_parse_ci_with_indicators_raises():
    # CI with non-empty indicators should fail validation
    bad = {**_VALID_CI, "indicators": [
        {"name": "x", "definition": "y", "role": "component"}
    ]}
    with pytest.raises(ValueError, match="[Ss]chema|validation|empty"):
        parse_concept_map(json.dumps(bad))
