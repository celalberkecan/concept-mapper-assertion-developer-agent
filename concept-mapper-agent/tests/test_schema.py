import pytest
from pydantic import ValidationError

from concept_mapper.schemas import ConceptMap, Indicator


def _indicator(name: str = "fear of burglary", role: str = "component") -> dict:
    return {"name": name, "definition": "Some definition.", "role": role}


# ---------------------------------------------------------------------------
# CI tests
# ---------------------------------------------------------------------------


def test_valid_ci():
    cm = ConceptMap(
        input_topic="age",
        ci_or_cp="CI",
        indicator_model="NA",
        construct_definition="The respondent's age.",
        indicators=[],
        rationale="Age is direct.",
        warnings=[],
    )
    assert cm.ci_or_cp == "CI"
    assert cm.indicators == []
    assert cm.indicator_model == "NA"


def test_ci_rejects_non_na_indicator_model():
    with pytest.raises(ValidationError, match="NA"):
        ConceptMap(
            input_topic="age",
            ci_or_cp="CI",
            indicator_model="formative",
            construct_definition="The respondent's age.",
            indicators=[],
            rationale="Age is direct.",
            warnings=[],
        )


def test_ci_rejects_non_empty_indicators():
    with pytest.raises(ValidationError, match="empty"):
        ConceptMap(
            input_topic="age",
            ci_or_cp="CI",
            indicator_model="NA",
            construct_definition="The respondent's age.",
            indicators=[Indicator(**_indicator())],
            rationale="Age is direct.",
            warnings=[],
        )


# ---------------------------------------------------------------------------
# CP tests
# ---------------------------------------------------------------------------


def test_valid_cp_formative():
    cm = ConceptMap(
        input_topic="fear of crime",
        ci_or_cp="CP",
        indicator_model="formative",
        construct_definition="A multidimensional fear construct.",
        indicators=[
            Indicator(**_indicator("fear of burglary", "component")),
            Indicator(**_indicator("fear of assault", "component")),
        ],
        rationale="Covers multiple crime-specific fears.",
        warnings=[],
    )
    assert cm.ci_or_cp == "CP"
    assert len(cm.indicators) == 2


def test_valid_cp_reflective():
    cm = ConceptMap(
        input_topic="political trust",
        ci_or_cp="CP",
        indicator_model="reflective",
        construct_definition="Latent trust disposition.",
        indicators=[
            Indicator(**_indicator("trust in parliament", "manifestation")),
            Indicator(**_indicator("trust in parties", "manifestation")),
        ],
        rationale="Reflective indicators of the same latent.",
        warnings=[],
    )
    assert cm.indicator_model == "reflective"


def test_cp_requires_at_least_2_indicators():
    with pytest.raises(ValidationError, match="at least 2"):
        ConceptMap(
            input_topic="fear of crime",
            ci_or_cp="CP",
            indicator_model="formative",
            construct_definition="A multidimensional fear construct.",
            indicators=[Indicator(**_indicator())],
            rationale="...",
            warnings=[],
        )


def test_cp_rejects_na_indicator_model():
    with pytest.raises(ValidationError):
        ConceptMap(
            input_topic="fear of crime",
            ci_or_cp="CP",
            indicator_model="NA",
            construct_definition="A multidimensional fear construct.",
            indicators=[
                Indicator(**_indicator("fear of burglary", "component")),
                Indicator(**_indicator("fear of assault", "component")),
            ],
            rationale="...",
            warnings=[],
        )


def test_warnings_can_be_non_empty():
    cm = ConceptMap(
        input_topic="job satisfaction",
        ci_or_cp="CI",
        indicator_model="NA",
        construct_definition="Overall job satisfaction.",
        indicators=[],
        rationale="Global single-item construct.",
        warnings=["Consider CP if facet-level measurement is needed."],
    )
    assert len(cm.warnings) == 1
