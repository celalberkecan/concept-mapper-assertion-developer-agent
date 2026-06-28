"""System prompt and message builder for the Assertion Developer agent."""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Few-shot examples (stored as dicts, serialised to JSON in the messages)
# ---------------------------------------------------------------------------

_EXAMPLE_FEAR_OF_BURGLARY: str = json.dumps(
    {
        "parent_concept": "fear of crime",
        "input_indicator": "fear of burglary",
        "indicator_definition": "Worry or fear that one's home may be broken into.",
        "variable_type": "subjective",
        "basic_concept": "feelings",
        "domain": "burglary / home victimization",
        "structure_code": "rFy",
        "assertion": "The respondent fears that their home may be burglarized.",
        "rationale": (
            "The indicator expresses an affective state of fear toward a specific type of "
            "victimization. Feelings about a concrete object use structure rFy "
            "(respondent Fears y)."
        ),
        "warnings": [],
    },
    indent=2,
)

_EXAMPLE_VOTED: str = json.dumps(
    {
        "parent_concept": "electoral participation",
        "input_indicator": "voted in last election",
        "indicator_definition": "Whether the respondent cast a vote in the most recent election.",
        "variable_type": "objective",
        "basic_concept": "behavior",
        "domain": "electoral participation",
        "structure_code": "rD",
        "assertion": "The respondent voted in the last election.",
        "rationale": (
            "Voting is an observable past action with no explicit object in the assertion. "
            "Completed behaviors without a direct object use structure rD."
        ),
        "warnings": [],
    },
    indent=2,
)

_EXAMPLE_IMPORTANCE_OF_RELIGION: str = json.dumps(
    {
        "parent_concept": "religiosity",
        "input_indicator": "importance of religion",
        "indicator_definition": "The degree to which religion matters in the respondent's life.",
        "variable_type": "subjective",
        "basic_concept": "importance",
        "domain": "religious salience",
        "structure_code": "xli",
        "assertion": "Religion is important to the respondent.",
        "rationale": (
            "The indicator measures the subjective salience of religion. "
            "Importance judgments use structure xli (x is important)."
        ),
        "warnings": [],
    },
    indent=2,
)

_EXAMPLE_AGE: str = json.dumps(
    {
        "parent_concept": "age",
        "input_indicator": "age",
        "indicator_definition": "The respondent's chronological age in years.",
        "variable_type": "objective",
        "basic_concept": "demographics",
        "domain": "age",
        "structure_code": "xId",
        "assertion": "The respondent is of a specific chronological age.",
        "rationale": (
            "Age is a directly observable demographic attribute. "
            "Demographic characteristics use structure xId."
        ),
        "warnings": [],
    },
    indent=2,
)

_EXAMPLE_GENDER_EQUALITY_POLICY: str = json.dumps(
    {
        "parent_concept": "gender equality attitudes",
        "input_indicator": "responsibility to promote gender equality",
        "indicator_definition": (
            "Belief that it is the government's responsibility to promote gender equality."
        ),
        "variable_type": "subjective",
        "basic_concept": "policies",
        "domain": "gender equality / government responsibility",
        "structure_code": "g(H+I)y",
        "assertion": "The respondent believes the government should promote gender equality.",
        "rationale": (
            "The indicator captures a policy position: what the government ought to do about "
            "a social issue. Policy beliefs use structure g(H+I)y."
        ),
        "warnings": [],
    },
    indent=2,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are a survey methodology expert specialising in assertion development for \
questionnaire design, following the Saris & Gallhofer (2007) framework.

Your task: given one CI-level survey indicator (a name, definition, and role), \
produce one formal declarative assertion that captures exactly what the indicator measures.

## What is an assertion?

An assertion is a single declarative statement about what a respondent does, feels, \
believes, or is. It is NOT a survey question. It does NOT contain response options, \
Likert scales, or measurement formats. It describes the content of measurement, \
not the method.

## Variable types

**Subjective** — captures internal states: what the respondent thinks, feels, \
values, believes, or intends.
**Objective** — captures external, observable facts: what the respondent did, \
experienced, owns, or is.

## Basic concepts

**Subjective basic concepts** (14):
evaluation, importance, values, feelings, cognitive_judgment, causal_relationship, \
similarity_relationship, preference, norms, policies, rights, action_tendencies, \
expectations_future_events, evaluative_belief

**Objective basic concepts** (8):
behavior, events, demographics, knowledge, time, place, quantities, procedures

## Assertion structures

Three grammatical structures are used in this framework:

**structure_1** — Subject + link verb + subject complement
Form: X is/was Y.
Example: "The government was a good president." / "Religion is important to the respondent."
Use for: evaluation, importance, values, cognitive_judgment, demographics

**structure_2** — Subject + predicator + direct object
Form: X does/feels/believes/prefers Y.
Example: "The respondent fears burglary." / "The respondent prefers city life."
Use for: feelings, causal_relationship, similarity_relationship, preference, norms, \
policies, rights, action_tendencies, evaluative_belief, knowledge, quantities

**structure_3** — Subject + predicator (no object)
Form: X happened / X acted / X changed.
Example: "The respondent voted." / "The respondent's position changed."
Use for: behavior, events, time, place, procedures, expectations_future_events

## Rule table (basic_concept → allowed structure codes)

| basic_concept               | allowed codes              | default  |
|-----------------------------|----------------------------|----------|
| evaluation                  | xle                        | xle      |
| importance                  | xli                        | xli      |
| values                      | xlv                        | xlv      |
| feelings                    | xlf, xFy, rFy              | rFy      |
| cognitive_judgment          | xIc                        | xIc      |
| causal_relationship         | xIca, xCy                  | xCy      |
| similarity_relationship     | xIs, xSy                   | xSy      |
| preference                  | xIpr, xPRy                 | xPRy     |
| norms                       | o(H+I)y, o(H+I)            | o(H+I)y  |
| policies                    | g(H+I)y                    | g(H+I)y  |
| rights                      | xIri, xHRy                 | xHRy     |
| action_tendencies           | rFDy                       | rFDy     |
| expectations_future_events  | xFDy, xFD                  | xFD      |
| evaluative_belief           | xPey, xPye, xPe            | xPey     |
| behavior                    | rDy, rD                    | rD       |
| events                      | xDy, xD                    | xD       |
| demographics                | xId                        | xId      |
| knowledge                   | xIsc, xPy, xP              | xPy      |
| time                        | xDti                       | xDti     |
| place                       | xDpl                       | xDpl     |
| quantities                  | xDqu                       | xDqu     |
| procedures                  | xDpl_pro                   | xDpl_pro |

## Strict Output Rules

1. Output ONLY a valid JSON object — no prose, no markdown, no code fences.
2. Do NOT write survey questions (no question marks at the end of the assertion).
3. Do NOT write answer options, Likert scales, or response formats.
4. Do NOT include `structure_id` in your output — it is derived automatically.
5. Choose `structure_code` from the allowed codes for your selected `basic_concept`.
6. `basic_concept` must belong to the selected `variable_type`.
7. `assertion` must be one declarative sentence about the respondent.

## JSON Schema

{
  "parent_concept": string,
  "input_indicator": string,
  "indicator_definition": string,
  "variable_type": "subjective" | "objective",
  "basic_concept": string,
  "domain": string,
  "structure_code": string,
  "assertion": string,
  "rationale": string,
  "warnings": [string]
}
"""


def _format_user_message(
    parent_concept: str,
    indicator_name: str,
    indicator_definition: str,
    indicator_role: str,
) -> str:
    return (
        f'Develop assertion for indicator: "{indicator_name}"\n'
        f'Parent concept: "{parent_concept}"\n'
        f"Indicator definition: {indicator_definition}\n"
        f"Indicator role: {indicator_role}"
    )


def build_assertion_messages(
    parent_concept: str,
    indicator_name: str,
    indicator_definition: str,
    indicator_role: str,
) -> list[dict]:
    """Build the full message list including few-shot examples for the assertion developer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        # Few-shot example 1: subjective / feelings / rFy
        {
            "role": "user",
            "content": _format_user_message(
                "fear of crime",
                "fear of burglary",
                "Worry or fear that one's home may be broken into.",
                "component",
            ),
        },
        {"role": "assistant", "content": _EXAMPLE_FEAR_OF_BURGLARY},
        # Few-shot example 2: objective / behavior / rD
        {
            "role": "user",
            "content": _format_user_message(
                "electoral participation",
                "voted in last election",
                "Whether the respondent cast a vote in the most recent election.",
                "component",
            ),
        },
        {"role": "assistant", "content": _EXAMPLE_VOTED},
        # Few-shot example 3: subjective / importance / xli
        {
            "role": "user",
            "content": _format_user_message(
                "religiosity",
                "importance of religion",
                "The degree to which religion matters in the respondent's life.",
                "component",
            ),
        },
        {"role": "assistant", "content": _EXAMPLE_IMPORTANCE_OF_RELIGION},
        # Few-shot example 4: objective / demographics / xId
        {
            "role": "user",
            "content": _format_user_message(
                "age",
                "age",
                "The respondent's chronological age in years.",
                "direct",
            ),
        },
        {"role": "assistant", "content": _EXAMPLE_AGE},
        # Few-shot example 5: subjective / policies / g(H+I)y
        {
            "role": "user",
            "content": _format_user_message(
                "gender equality attitudes",
                "responsibility to promote gender equality",
                "Belief that it is the government's responsibility to promote gender equality.",
                "component",
            ),
        },
        {"role": "assistant", "content": _EXAMPLE_GENDER_EQUALITY_POLICY},
        # Actual request
        {
            "role": "user",
            "content": _format_user_message(
                parent_concept,
                indicator_name,
                indicator_definition,
                indicator_role,
            ),
        },
    ]
