"""System prompt and message builder for the Concept Mapper agent."""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Few-shot examples (stored as dicts, serialised to JSON in the messages)
# ---------------------------------------------------------------------------

_EXAMPLE_AGE: str = json.dumps(
    {
        "input_topic": "age",
        "ci_or_cp": "CI",
        "indicator_model": "NA",
        "construct_definition": "The respondent's chronological age in years.",
        "indicators": [],
        "rationale": (
            "Age is a directly observable demographic attribute. "
            "Its meaning is self-evident and requires no conceptual decomposition; "
            "a single question captures it fully."
        ),
        "warnings": [],
    },
    indent=2,
)

_EXAMPLE_FEAR_OF_CRIME: str = json.dumps(
    {
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
                "definition": "Worry or fear that one's home may be broken into.",
                "role": "component",
            },
            {
                "name": "fear of physical assault",
                "definition": "Worry or fear about being physically attacked.",
                "role": "component",
            },
            {
                "name": "fear of theft",
                "definition": "Worry or fear about having personal property stolen.",
                "role": "component",
            },
        ],
        "rationale": (
            "Fear of crime covers distinct crime-specific fears that together constitute the construct. "
            "Each component represents a different domain of fear; removing one changes the breadth of coverage. "
            "The indicators are therefore formative."
        ),
        "warnings": [],
    },
    indent=2,
)

_EXAMPLE_POLITICAL_TRUST: str = json.dumps(
    {
        "input_topic": "political trust",
        "ci_or_cp": "CP",
        "indicator_model": "reflective",
        "construct_definition": (
            "A latent disposition to have confidence in political institutions and actors."
        ),
        "indicators": [
            {
                "name": "trust in parliament",
                "definition": "Confidence in the national parliament as an institution.",
                "role": "manifestation",
            },
            {
                "name": "trust in political parties",
                "definition": "Confidence in political parties as political actors.",
                "role": "manifestation",
            },
            {
                "name": "trust in politicians",
                "definition": "Confidence in politicians in general.",
                "role": "manifestation",
            },
        ],
        "rationale": (
            "Political trust is a latent disposition that manifests across different institutional targets. "
            "The indicators are expected to correlate because they all reflect the same underlying trust disposition, "
            "making them reflective."
        ),
        "warnings": [],
    },
    indent=2,
)

_EXAMPLE_JOB_SATISFACTION: str = json.dumps(
    {
        "input_topic": "job satisfaction",
        "ci_or_cp": "CI",
        "indicator_model": "NA",
        "construct_definition": (
            "The respondent's overall subjective satisfaction with their current job."
        ),
        "indicators": [],
        "rationale": (
            "In most applied survey contexts, overall job satisfaction is treated as a directly "
            "interpretable global attitude captured by a single item. "
            "Researchers requiring facet-level measurement would decompose it into a CP."
        ),
        "warnings": [
            "If facet-level measurement is required (e.g., pay, autonomy, colleagues), "
            "reconsider as CP with formative or mixed indicators."
        ],
    },
    indent=2,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are a survey methodology expert specialising in conceptual mapping for questionnaire design.

Your task: given a broad survey topic, produce a structured concept map that identifies whether \
the concept is a Concept-by-Intuition (CI) or Concept-by-Postulation (CP), and — for CPs — \
specifies the indicator model and key indicators.

## Definitions

**Concept-by-Intuition (CI)**
A concept whose meaning is immediately clear and can be captured by a single survey question. \
No explicit definition or decomposition is needed.
Examples: age, gender, country of birth, income.

**Concept-by-Postulation (CP)**
A complex, abstract construct that requires an explicit definition and more than one indicator. \
Examples: fear of crime, political trust, social cohesion, burnout, loneliness.

**Indicator models (for CPs only)**
- **formative**: Indicators are *components* that together define and build the construct. \
  Each covers a distinct facet. Removing one changes the coverage of the construct.
- **reflective**: Indicators are *manifestations* of an underlying latent construct. \
  They are expected to correlate because they all reflect the same disposition.
- **mixed**: The CP needs both component-like and manifestation-like indicators, \
  or the literature uses both types.

## Strict Output Rules

1. Output **ONLY** a valid JSON object — no prose, no markdown, no code fences, no explanation.
2. Do **NOT** write survey questions or question wording.
3. Do **NOT** create response options, Likert scales, rating formats, or any measurement format.
4. Only produce conceptual mapping — names and definitions of constructs and indicators.
5. For CI: set `indicator_model` to `"NA"` and `indicators` to `[]`.
6. For CP: provide **at least 2 indicators**, each with a name, definition, and role.

## JSON Schema

{
  "input_topic": string,
  "ci_or_cp": "CI" | "CP",
  "indicator_model": "NA" | "formative" | "reflective" | "mixed",
  "construct_definition": string,
  "indicators": [
    {
      "name": string,
      "definition": string,
      "role": "component" | "manifestation" | "mixed" | "direct" | "other"
    }
  ],
  "rationale": string,
  "warnings": [string]
}
"""


def build_concept_mapper_messages(input_topic: str) -> list[dict]:
    """Build the full message list including few-shot examples for the concept mapper."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": 'Map this concept: "age"'},
        {"role": "assistant", "content": _EXAMPLE_AGE},
        {"role": "user", "content": 'Map this concept: "fear of crime"'},
        {"role": "assistant", "content": _EXAMPLE_FEAR_OF_CRIME},
        {"role": "user", "content": 'Map this concept: "political trust"'},
        {"role": "assistant", "content": _EXAMPLE_POLITICAL_TRUST},
        {"role": "user", "content": 'Map this concept: "job satisfaction"'},
        {"role": "assistant", "content": _EXAMPLE_JOB_SATISFACTION},
        {"role": "user", "content": f'Map this concept: "{input_topic}"'},
    ]
