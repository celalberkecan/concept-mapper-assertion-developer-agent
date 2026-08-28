"""FakeLLMClient: returns canned JSON responses without any API calls.

Handles both ConceptMapper requests ("Map this concept: ...") and
AssertionDeveloper requests ("Develop assertion for indicator: ...").
Used by smoke-test CLI commands and the test suite.
"""

from __future__ import annotations

import json

from .base import BaseLLMClient

# ---------------------------------------------------------------------------
# Canned concept map responses
# ---------------------------------------------------------------------------

_CANNED_CONCEPT_MAPS: dict[str, dict] = {
    "age": {
        "input_topic": "age",
        "ci_or_cp": "CI",
        "indicator_model": "NA",
        "construct_definition": "The respondent's chronological age in years.",
        "indicators": [],
        "rationale": "Age is a directly observable demographic attribute.",
        "warnings": [],
    },
    "fear of crime": {
        "input_topic": "fear of crime",
        "ci_or_cp": "CP",
        "indicator_model": "formative",
        "construct_definition": (
            "A multidimensional affective construct capturing worry or fear "
            "about specific types of criminal victimization."
        ),
        "indicators": [
            {
                "name": "fear of burglary",
                "definition": "Worry that one's home may be broken into.",
                "role": "component",
            },
            {
                "name": "fear of physical assault",
                "definition": "Worry about being physically attacked.",
                "role": "component",
            },
            {
                "name": "fear of theft",
                "definition": "Worry about having personal property stolen.",
                "role": "component",
            },
        ],
        "rationale": (
            "Fear of crime covers distinct crime-specific fears that together constitute "
            "the construct. Each component represents a different domain; removing one changes "
            "the breadth of coverage, making the indicators formative."
        ),
        "warnings": [],
    },
    "political trust": {
        "input_topic": "political trust",
        "ci_or_cp": "CP",
        "indicator_model": "reflective",
        "construct_definition": (
            "A latent disposition to have confidence in political institutions and actors."
        ),
        "indicators": [
            {
                "name": "trust in parliament",
                "definition": "Confidence in the national parliament.",
                "role": "manifestation",
            },
            {
                "name": "trust in political parties",
                "definition": "Confidence in political parties.",
                "role": "manifestation",
            },
            {
                "name": "trust in politicians",
                "definition": "Confidence in politicians in general.",
                "role": "manifestation",
            },
        ],
        "rationale": (
            "Political trust is a latent disposition that manifests across institutional targets. "
            "The indicators are expected to correlate because they all reflect the same disposition."
        ),
        "warnings": [],
    },
}

_CONCEPT_MAP_FALLBACK = "fear of crime"

# ---------------------------------------------------------------------------
# Canned assertion responses
# ---------------------------------------------------------------------------

_CANNED_ASSERTIONS: dict[str, dict] = {
    "fear of burglary": {
        "parent_concept": "fear of crime",
        "input_indicator": "fear of burglary",
        "variable_type": "subjective",
        "basic_concept": "feelings",
        "domain": "burglary / home victimization",
        "structure_code": "rFy",
        "assertion": "The respondent fears that their home may be burglarized.",
        "rationale": (
            "The indicator expresses an affective state of fear toward a possible crime event, "
            "making it a feelings concept with structure rFy (respondent Fears y)."
        ),
        "warnings": [],
    },
    "age": {
        "parent_concept": "age",
        "input_indicator": "age",
        "variable_type": "objective",
        "basic_concept": "demographics",
        "domain": "age",
        "structure_code": "xId",
        "assertion": "The respondent is of a specific chronological age.",
        "rationale": "Age is a directly observable demographic attribute.",
        "warnings": [],
    },
    "voted in last election": {
        "parent_concept": "electoral participation",
        "input_indicator": "voted in last election",
        "variable_type": "objective",
        "basic_concept": "behavior",
        "domain": "electoral participation",
        "structure_code": "rD",
        "assertion": "The respondent voted in the last election.",
        "rationale": "Voting is an observable past action, making this an objective behavior indicator.",
        "warnings": [],
    },
}

_ASSERTION_FALLBACK = "fear of burglary"


class FakeLLMClient(BaseLLMClient):
    """Returns canned JSON without any network or API calls.

    Dispatches on the last user message prefix:
    - 'Develop assertion for indicator:' → assertion JSON
    - 'Map this concept:'                → concept map JSON
    """

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> str:
        last_user = _get_last_user_message(messages)

        if "Develop assertion for indicator:" in last_user:
            indicator = _extract_quoted(last_user)
            key = (indicator or "").lower()
            payload = _CANNED_ASSERTIONS.get(key, _CANNED_ASSERTIONS[_ASSERTION_FALLBACK])
            if indicator:
                payload = {**payload, "input_indicator": indicator}
            return json.dumps(payload, indent=2)

        # Default: concept map request
        topic = _extract_quoted(last_user)
        key = (topic or "").lower()
        payload = _CANNED_CONCEPT_MAPS.get(key, _CANNED_CONCEPT_MAPS[_CONCEPT_MAP_FALLBACK])
        if topic:
            payload = {**payload, "input_topic": topic}
        return json.dumps(payload, indent=2)


def _get_last_user_message(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _extract_quoted(text: str) -> str | None:
    """Extract the first double-quoted string from *text*."""
    start = text.find('"')
    if start == -1:
        return None
    end = text.find('"', start + 1)
    if end == -1:
        return None
    return text[start + 1 : end]
