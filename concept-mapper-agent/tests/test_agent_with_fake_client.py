"""Agent integration tests using FakeLLMClient — no external API calls."""

import pytest

from concept_mapper.agent import ConceptMapperAgent
from concept_mapper.llm_clients.fake_client import FakeLLMClient
from concept_mapper.schemas import ConceptMap


@pytest.fixture
def agent() -> ConceptMapperAgent:
    return ConceptMapperAgent(FakeLLMClient())


def test_map_fear_of_crime_returns_cp(agent: ConceptMapperAgent):
    cm = agent.map_concept("fear of crime")
    assert isinstance(cm, ConceptMap)
    assert cm.ci_or_cp == "CP"
    assert cm.indicator_model == "formative"
    assert len(cm.indicators) >= 2
    assert cm.input_topic == "fear of crime"


def test_map_age_returns_ci(agent: ConceptMapperAgent):
    cm = agent.map_concept("age")
    assert isinstance(cm, ConceptMap)
    assert cm.ci_or_cp == "CI"
    assert cm.indicator_model == "NA"
    assert cm.indicators == []


def test_map_political_trust_returns_reflective_cp(agent: ConceptMapperAgent):
    cm = agent.map_concept("political trust")
    assert isinstance(cm, ConceptMap)
    assert cm.ci_or_cp == "CP"
    assert cm.indicator_model == "reflective"
    assert len(cm.indicators) >= 2


def test_map_concept_with_raw_returns_tuple(agent: ConceptMapperAgent):
    cm, raw = agent.map_concept_with_raw("fear of crime")
    assert isinstance(cm, ConceptMap)
    assert isinstance(raw, str)
    assert "{" in raw


def test_unknown_topic_falls_back_gracefully(agent: ConceptMapperAgent):
    # Unknown topics fall back to the 'fear of crime' canned response
    cm = agent.map_concept("some completely unknown topic xyz")
    assert isinstance(cm, ConceptMap)
    # The fallback is a CP, so basic schema rules must hold
    assert cm.ci_or_cp in ("CI", "CP")


def test_retry_on_invalid_response():
    """Agent retries once when the first response is invalid JSON."""

    call_count = 0
    valid_json = """{
        "input_topic": "age",
        "ci_or_cp": "CI",
        "indicator_model": "NA",
        "construct_definition": "The respondent's age.",
        "indicators": [],
        "rationale": "Age is direct.",
        "warnings": []
    }"""

    class BrokenThenFixedClient(FakeLLMClient):
        def generate(self, messages, temperature=0.0, max_tokens=1200) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Sorry, I cannot produce the JSON right now."
            return valid_json

    agent = ConceptMapperAgent(BrokenThenFixedClient())
    cm = agent.map_concept("age")
    assert cm.ci_or_cp == "CI"
    assert call_count == 2  # first failed, second succeeded
