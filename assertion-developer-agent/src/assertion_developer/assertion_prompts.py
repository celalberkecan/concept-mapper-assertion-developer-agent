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
        "variable_type": "subjective",
        "basic_concept": "feelings",
        "domain": "burglary / home victimization",
        "structure_code": "rFy",
        "assertion": "The respondent fears that their home may be burglarized.",
        "rationale": (
            "The indicator expresses an affective state of fear toward a specific type "
            "of victimization. The respondent is the experiencer of the feeling, so rFy is appropriate."
        ),
        "warnings": [],
    },
    indent=2,
)

_EXAMPLE_VOTED: str = json.dumps(
    {
        "parent_concept": "political participation",
        "input_indicator": "voted in last election",
        "variable_type": "objective",
        "basic_concept": "behavior",
        "domain": "electoral participation",
        "structure_code": "rD",
        "assertion": "The respondent voted in the last election.",
        "rationale": (
            "The indicator refers to a completed action performed by the respondent. "
            "A behavior without an explicit direct object can use rD."
        ),
        "warnings": [],
    },
    indent=2,
)

_EXAMPLE_IMPORTANCE_OF_RELIGION: str = json.dumps(
    {
        "parent_concept": "religiosity",
        "input_indicator": "importance of religion",
        "variable_type": "subjective",
        "basic_concept": "importance",
        "domain": "religious salience",
        "structure_code": "xIi",
        "assertion": "Religion is important in the respondent's life.",
        "rationale": (
            "The indicator concerns the subjective importance of a domain. "
            "Importance assertions use xIi: X is important."
        ),
        "warnings": [],
    },
    indent=2,
)

_EXAMPLE_AGE: str = json.dumps(
    {
        "parent_concept": "age",
        "input_indicator": "age",
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
        "variable_type": "subjective",
        "basic_concept": "policies",
        "domain": "gender equality / government responsibility",
        "structure_code": "g(H+I)y",
        "assertion": "The government should promote gender equality.",
        "rationale": (
            "The indicator concerns what the government ought to do about a social issue. "
            "The assertion states the policy proposition that the respondent may later evaluate."
        ),
        "warnings": [],
    },
    indent=2,
)

_EXAMPLE_DEMOCRACY_EVALUATION = json.dumps(
    {
        "parent_concept": "satisfaction with democracy",
        "input_indicator": "democracy works well",
        "variable_type": "subjective",
        "basic_concept": "evaluation",
        "domain": "functioning of democracy",
        "structure_code": "xIe",
        "assertion": "Democracy in the country works well.",
        "rationale": (
            "The indicator evaluates an object or state of affairs. "
            "Evaluation assertions use xIe: X is evaluated as good, bad, effective, or ineffective."
        ),
        "warnings": [],
    },
    indent=2,
)

_EXAMPLE_IMMIGRANT_NORM = json.dumps(
    {
        "parent_concept": "integration norms",
        "input_indicator": "immigrants should adapt to local culture",
        "variable_type": "subjective",
        "basic_concept": "norms",
        "domain": "immigrant integration / cultural adaptation",
        "structure_code": "o(H+I)y",
        "assertion": "Immigrants should adapt to the culture of the receiving country.",
        "rationale": (
            "The indicator expresses a social norm about what a group ought to do. "
            "Norm assertions use o(H+I)y: actors should do something."
        ),
        "warnings": [],
    },
    indent=2,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """
You are a survey methodology expert specialising in assertion development for 
questionnaire design, following the Saris & Gallhofer framework.

Your task: given one CI-level survey indicator (a name and role),
produce one formal declarative assertion that captures exactly what the indicator measures.

## What is an assertion?

An assertion is a single declarative statement that expresses the content to be measured. 
It is NOT a survey question. It does NOT contain response options, Likert scales, 
or measurement formats. It describes the measurement target, not the measurement method.

The respondent may be the grammatical subject when appropriate, for example in behavior, 
feelings, demographics, knowledge, or action tendencies. However, for evaluations, norms, 
policies, rights, causal beliefs, and similar concepts, the assertion may be a proposition 
about an object, actor, policy, group, or state of affairs.

Correct:

* "The respondent fears burglary."
* "The respondent voted in the last election."
* "Religion is important in the respondent's life."
* "The government should promote gender equality."
* "Immigrants should adapt to the culture of the receiving country."
* "Democracy in the country works well."

Avoid unnecessary belief wrappers:

* Do NOT write "The respondent believes that the government should promote gender equality" 
  when the selected structure is g(H+I)y.
* Instead write: "The government should promote gender equality."
* Do NOT write "The respondent believes democracy works well" when the selected structure is xIe.
* Instead write: "Democracy in the country works well."

## Variable types

**Subjective** — captures evaluations, importance, values, feelings, judgments, beliefs, 
preferences, norms, policy attitudes, rights, intentions, or expectations.

**Objective** — captures external or observable facts, attributes, actions, events, 
knowledge, time, place, quantities, or procedures.

## Basic concepts

**Subjective basic concepts** (14):
evaluation, importance, values, feelings, cognitive_judgment, causal_relationship, 
similarity_relationship, preference, norms, policies, rights, action_tendencies, 
expectations_future_events, evaluative_belief

**Objective basic concepts** (8):
behavior, events, demographics, knowledge, time, place, quantities, procedures

## Assertion structures

Three grammatical structures are used in this framework:

**structure_1** — Subject + link verb + subject complement
Form: X is/was Y.
Example: "The government is effective." / "Religion is important in the respondent's life."
Use for: evaluation, importance, values, cognitive_judgment, demographics

**structure_2** — Subject + predicator + direct object or proposition
Form: X does/feels/believes/prefers/should-do Y.
Example: "The respondent fears burglary." / "The government should promote gender equality."
Use for: feelings, causal_relationship, similarity_relationship, preference, norms, 
policies, rights, action_tendencies, evaluative_belief, knowledge, quantities

**structure_3** — Subject + predicator, usually without an explicit direct object
Form: X happened / X acted / X changed.
Example: "The respondent voted." / "Prices increased."
Use for: behavior, events, time, place, procedures, expectations_future_events

## Rule table (basic_concept → allowed structure codes)

| basic_concept              | allowed codes   | default  |
| -------------------------- | --------------- | -------- |
| evaluation                 | xIe             | xIe      |
| importance                 | xIi             | xIi      |
| values                     | vIi             | vIi      |
| feelings                   | xIf, xFy, xPf, rFy | rFy   |
| cognitive_judgment         | xIc             | xIc      |
| causal_relationship        | xIca, xCy       | xCy      |
| similarity_relationship    | xIs, xSy        | xSy      |
| preference                 | xIpr, xPRy      | xPRy     |
| norms                      | o(H+I)y, o(H+I) | o(H+I)y  |
| policies                   | g(H+I)y         | g(H+I)y  |
| rights                     | xIri, xHRy      | xHRy     |
| action_tendencies          | rFDy            | rFDy     |
| expectations_future_events | xFDy, xFD       | xFD      |
| evaluative_belief          | xPey, xPye, xPe | xPey     |
| behavior                   | rDy, rD         | rD       |
| events                     | xDy, xD         | xD       |
| demographics               | xId             | xId      |
| knowledge                  | xIsc, xPy, xP   | xPy      |
| time                       | xDti            | xDti     |
| place                      | xDpl            | xDpl     |
| quantities                 | xDqu            | xDqu     |
| procedures                 | xDpl_pro        | xDpl_pro |

## Strict Output Rules

1. Output ONLY a valid JSON object — no prose, no markdown, no code fences.
2. Do NOT write survey questions.
3. Do NOT put a question mark at the end of the assertion.
4. Do NOT write answer options, Likert scales, or response formats.
5. Do NOT include `structure_id` in your output — it is derived automatically.
6. Choose `structure_code` only from the allowed codes for your selected `basic_concept`.
7. `basic_concept` must belong to the selected `variable_type`.
8. The assertion must be one declarative sentence that captures the measurement target.
9. The assertion must be compatible with the selected `structure_code`.
10. Do not add "The respondent believes..." unless the selected basic concept and structure code require the respondent as the grammatical subject.

## JSON Schema

{
"parent_concept": string,
"input_indicator": string,
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
    indicator_role: str,
) -> str:
    return (
        f'Develop assertion for indicator: "{indicator_name}"\n'
        f'Parent concept: "{parent_concept}"\n'
        f"Indicator role: {indicator_role}"
    )


def build_assertion_messages(
    parent_concept: str,
    indicator_name: str,
    indicator_role: str,
) -> list[dict]:
    """Build the full message list including few-shot examples for the assertion developer.

    Few-shot strategy:
    - Include cases where the respondent is the grammatical subject when appropriate:
      feelings, behavior.
    - Include proposition-style assertions where the respondent is NOT necessarily the
      grammatical subject:
      importance, policies, evaluation, norms.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},

        # Few-shot example 1: subjective / feelings / rFy
        # Here the respondent is the experiencer of the feeling, so "The respondent..." is appropriate.
        {
            "role": "user",
            "content": _format_user_message(
                "fear of crime",
                "fear of burglary",
                "component",
            ),
        },
        {"role": "assistant", "content": _EXAMPLE_FEAR_OF_BURGLARY},

        # Few-shot example 2: objective / behavior / rD
        # Here the respondent performed the action, so "The respondent..." is appropriate.
        {
            "role": "user",
            "content": _format_user_message(
                "political participation",
                "voted in last election",
                "component",
            ),
        },
        {"role": "assistant", "content": _EXAMPLE_VOTED},

        # Few-shot example 3: subjective / importance / xIi
        # Proposition-style assertion: the domain is important, not "the respondent believes..."
        {
            "role": "user",
            "content": _format_user_message(
                "religiosity",
                "importance of religion",
                "component",
            ),
        },
        {"role": "assistant", "content": _EXAMPLE_IMPORTANCE_OF_RELIGION},

        # Few-shot example 4: subjective / policies / g(H+I)y
        # Proposition-style assertion: government should do something.
        {
            "role": "user",
            "content": _format_user_message(
                "gender equality attitudes",
                "responsibility to promote gender equality",
                "component",
            ),
        },
        {"role": "assistant", "content": _EXAMPLE_GENDER_EQUALITY_POLICY},

        # Few-shot example 5: subjective / evaluation / xIe
        # Proposition-style assertion: the object is evaluated directly.
        {
            "role": "user",
            "content": _format_user_message(
                "satisfaction with democracy",
                "democracy works well",
                "component",
            ),
        },
        {"role": "assistant", "content": _EXAMPLE_DEMOCRACY_EVALUATION},

        # Few-shot example 6: subjective / norms / o(H+I)y
        # Proposition-style assertion: a group should do something.
        {
            "role": "user",
            "content": _format_user_message(
                "integration norms",
                "immigrants should adapt to local culture",
                "component",
            ),
        },
        {"role": "assistant", "content": _EXAMPLE_IMMIGRANT_NORM},

        # Actual request
        {
            "role": "user",
            "content": _format_user_message(
                parent_concept,
                indicator_name,
                indicator_role,
            ),
        },
    ]
