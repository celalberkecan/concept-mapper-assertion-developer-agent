"""Tests for AssertionDeveloperAgent using FakeLLMClient — no API calls."""

import json

import pytest
from survey_agent_lib.llm_clients.base import BaseLLMClient
from survey_agent_lib.llm_clients.fake_client import FakeLLMClient

from assertion_developer.assertion_agent import AssertionDeveloperAgent
from assertion_developer.assertion_schemas import AssertionOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent() -> AssertionDeveloperAgent:
    return AssertionDeveloperAgent(FakeLLMClient())


# ---------------------------------------------------------------------------
# Basic agent behaviour
# ---------------------------------------------------------------------------


def test_known_indicator_returns_valid_assertion():
    agent = _make_agent()
    result = agent.develop_assertion(
        parent_concept="fear of crime",
        indicator_name="fear of burglary",
        indicator_role="component",
    )
    assert isinstance(result, AssertionOutput)
    assert result.input_indicator == "fear of burglary"
    assert result.variable_type == "subjective"
    assert result.basic_concept == "feelings"
    assert result.structure_code == "rFy"


def test_structure_id_is_derived():
    """structure_id must be set by validation, not by the fake client output."""
    agent = _make_agent()
    result = agent.develop_assertion(
        "fear of crime", "fear of burglary", "component",
    )
    assert result.structure_id == "structure_2"


def test_unknown_indicator_falls_back_to_default():
    """Indicators not in canned responses fall back to the fear-of-burglary response."""
    agent = _make_agent()
    result = agent.develop_assertion(
        "some concept", "some unknown indicator", "component",
    )
    assert isinstance(result, AssertionOutput)
    # Falls back to fear of burglary canned response — input_indicator is overwritten
    assert result.input_indicator == "some unknown indicator"


def test_develop_assertion_with_raw_returns_raw_string():
    agent = _make_agent()
    result, raw = agent.develop_assertion_with_raw(
        "fear of crime", "fear of burglary", "component",
    )
    assert isinstance(result, AssertionOutput)
    assert isinstance(raw, str)
    assert "{" in raw  # raw is JSON text


def test_objective_demographics_assertion():
    agent = _make_agent()
    result = agent.develop_assertion(
        "age", "age", "direct",
    )
    assert result.variable_type == "objective"
    assert result.basic_concept == "demographics"
    assert result.structure_id == "structure_1"


# ---------------------------------------------------------------------------
# Repair retry behaviour
# ---------------------------------------------------------------------------


class _FailThenSucceedClient(BaseLLMClient):
    """Returns invalid JSON on first call, valid assertion JSON on second."""

    def __init__(self) -> None:
        self._calls = 0
        self._valid = json.dumps({
            "parent_concept": "fear of crime",
            "input_indicator": "fear of burglary",
            "variable_type": "subjective",
            "basic_concept": "feelings",
            "domain": "burglary",
            "structure_code": "rFy",
            "assertion": "The respondent fears that their home may be burglarized.",
            "rationale": "A feelings indicator.",
            "warnings": [],
        })

    def generate(self, messages, temperature=0.0, max_tokens=800) -> str:
        self._calls += 1
        if self._calls == 1:
            return "NOT VALID JSON }{{"
        return self._valid


def test_repair_retry_triggered_on_parse_failure():
    client = _FailThenSucceedClient()
    agent = AssertionDeveloperAgent(client)
    result = agent.develop_assertion(
        "fear of crime", "fear of burglary", "component",
    )
    assert isinstance(result, AssertionOutput)
    assert client._calls == 2   # exactly one retry


class _AlwaysFailClient(BaseLLMClient):
    def generate(self, messages, temperature=0.0, max_tokens=800) -> str:
        return "NOT JSON AT ALL"


def test_raises_after_two_failures():
    agent = AssertionDeveloperAgent(_AlwaysFailClient())
    with pytest.raises(ValueError):
        agent.develop_assertion(
            "fear of crime", "fear of burglary", "component",
        )
