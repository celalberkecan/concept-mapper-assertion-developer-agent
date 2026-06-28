"""Rule table mapping basic concepts to allowed assertion structure codes.

Based on Saris & Gallhofer (2007) assertion structure taxonomy.
Each entry contains:
  variable_type  : 'subjective' or 'objective'
  allowed_codes  : list of valid structure codes for this concept
  default_code   : the code to use when no specific preference applies
  structure_id   : 'structure_1', 'structure_2', or 'structure_3'
  example        : canonical example assertion
"""

from __future__ import annotations

BASIC_CONCEPT_RULES: dict[str, dict] = {
    # ------------------------------------------------------------------
    # Subjective basic concepts
    # ------------------------------------------------------------------
    "evaluation": {
        "variable_type": "subjective",
        "allowed_codes": ["xle"],
        "default_code": "xle",
        "structure_id": "structure_1",
        "example": "The government is good.",
    },
    "importance": {
        "variable_type": "subjective",
        "allowed_codes": ["xli"],
        "default_code": "xli",
        "structure_id": "structure_1",
        "example": "Religion is important to the respondent.",
    },
    "values": {
        "variable_type": "subjective",
        "allowed_codes": ["xlv"],
        "default_code": "xlv",
        "structure_id": "structure_1",
        "example": "Freedom is an important value to the respondent.",
    },
    "feelings": {
        "variable_type": "subjective",
        "allowed_codes": ["xlf", "xFy", "rFy"],
        "default_code": "rFy",
        "structure_id": "structure_2",
        "example": "The respondent fears burglary.",
    },
    "cognitive_judgment": {
        "variable_type": "subjective",
        "allowed_codes": ["xIc"],
        "default_code": "xIc",
        "structure_id": "structure_1",
        "example": "The respondent considers the economy to be weak.",
    },
    "causal_relationship": {
        "variable_type": "subjective",
        "allowed_codes": ["xIca", "xCy"],
        "default_code": "xCy",
        "structure_id": "structure_2",
        "example": "The respondent believes unemployment causes crime.",
    },
    "similarity_relationship": {
        "variable_type": "subjective",
        "allowed_codes": ["xIs", "xSy"],
        "default_code": "xSy",
        "structure_id": "structure_2",
        "example": "The respondent considers party A similar to party B.",
    },
    "preference": {
        "variable_type": "subjective",
        "allowed_codes": ["xIpr", "xPRy"],
        "default_code": "xPRy",
        "structure_id": "structure_2",
        "example": "The respondent prefers living in the city over the countryside.",
    },
    "norms": {
        "variable_type": "subjective",
        "allowed_codes": ["o(H+I)y", "o(H+I)"],
        "default_code": "o(H+I)y",
        "structure_id": "structure_2",
        "example": "The respondent believes one ought to pay taxes.",
    },
    "policies": {
        "variable_type": "subjective",
        "allowed_codes": ["g(H+I)y"],
        "default_code": "g(H+I)y",
        "structure_id": "structure_2",
        "example": "The respondent believes the government should reduce inequality.",
    },
    "rights": {
        "variable_type": "subjective",
        "allowed_codes": ["xIri", "xHRy"],
        "default_code": "xHRy",
        "structure_id": "structure_2",
        "example": "The respondent believes everyone has the right to free speech.",
    },
    "action_tendencies": {
        "variable_type": "subjective",
        "allowed_codes": ["rFDy"],
        "default_code": "rFDy",
        "structure_id": "structure_2",
        "example": "The respondent intends to vote in the next election.",
    },
    "expectations_future_events": {
        "variable_type": "subjective",
        "allowed_codes": ["xFDy", "xFD"],
        "default_code": "xFD",
        "structure_id": "structure_3",
        "example": "The respondent expects the economic situation to worsen.",
    },
    "evaluative_belief": {
        "variable_type": "subjective",
        "allowed_codes": ["xPey", "xPye", "xPe"],
        "default_code": "xPey",
        "structure_id": "structure_2",
        "example": "The respondent perceives the current government as effective.",
    },
    # ------------------------------------------------------------------
    # Objective basic concepts
    # ------------------------------------------------------------------
    "behavior": {
        "variable_type": "objective",
        "allowed_codes": ["rDy", "rD"],
        "default_code": "rD",
        "structure_id": "structure_3",
        "example": "The respondent voted in the last election.",
    },
    "events": {
        "variable_type": "objective",
        "allowed_codes": ["xDy", "xD"],
        "default_code": "xD",
        "structure_id": "structure_3",
        "example": "The respondent experienced a burglary.",
    },
    "demographics": {
        "variable_type": "objective",
        "allowed_codes": ["xId"],
        "default_code": "xId",
        "structure_id": "structure_1",
        "example": "The respondent is of a specific chronological age.",
    },
    "knowledge": {
        "variable_type": "objective",
        "allowed_codes": ["xIsc", "xPy", "xP"],
        "default_code": "xPy",
        "structure_id": "structure_2",
        "example": "The respondent knows who the current prime minister is.",
    },
    "time": {
        "variable_type": "objective",
        "allowed_codes": ["xDti"],
        "default_code": "xDti",
        "structure_id": "structure_3",
        "example": "The respondent worked a specific number of hours last week.",
    },
    "place": {
        "variable_type": "objective",
        "allowed_codes": ["xDpl"],
        "default_code": "xDpl",
        "structure_id": "structure_3",
        "example": "The respondent lives in a specific region.",
    },
    "quantities": {
        "variable_type": "objective",
        "allowed_codes": ["xDqu"],
        "default_code": "xDqu",
        "structure_id": "structure_2",
        "example": "The respondent owns a specific number of cars.",
    },
    "procedures": {
        "variable_type": "objective",
        "allowed_codes": ["xDpl_pro"],
        "default_code": "xDpl_pro",
        "structure_id": "structure_3",
        "example": "The respondent followed a specific procedure to apply for benefits.",
    },
}

# Ordered list for prompt display
SUBJECTIVE_CONCEPTS: list[str] = [
    k for k, v in BASIC_CONCEPT_RULES.items() if v["variable_type"] == "subjective"
]
OBJECTIVE_CONCEPTS: list[str] = [
    k for k, v in BASIC_CONCEPT_RULES.items() if v["variable_type"] == "objective"
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_allowed_codes(basic_concept: str) -> list[str]:
    """Return the list of allowed structure codes for *basic_concept*."""
    rule = BASIC_CONCEPT_RULES.get(basic_concept)
    if rule is None:
        raise ValueError(f"Unknown basic_concept: {basic_concept!r}")
    return rule["allowed_codes"]


def get_default_code(basic_concept: str) -> str:
    """Return the default structure code for *basic_concept*."""
    rule = BASIC_CONCEPT_RULES.get(basic_concept)
    if rule is None:
        raise ValueError(f"Unknown basic_concept: {basic_concept!r}")
    return rule["default_code"]


def get_structure_id(basic_concept: str) -> str:
    """Return the structure_id for *basic_concept*."""
    rule = BASIC_CONCEPT_RULES.get(basic_concept)
    if rule is None:
        raise ValueError(f"Unknown basic_concept: {basic_concept!r}")
    return rule["structure_id"]


def get_valid_basic_concepts(variable_type: str) -> list[str]:
    """Return all basic concept names for the given *variable_type*."""
    return [k for k, v in BASIC_CONCEPT_RULES.items() if v["variable_type"] == variable_type]


def validate_combination(variable_type: str, basic_concept: str, structure_code: str) -> None:
    """Raise ValueError if the combination is invalid."""
    rule = BASIC_CONCEPT_RULES.get(basic_concept)
    if rule is None:
        raise ValueError(f"Unknown basic_concept: {basic_concept!r}")
    if rule["variable_type"] != variable_type:
        raise ValueError(
            f"basic_concept={basic_concept!r} requires variable_type={rule['variable_type']!r}, "
            f"got {variable_type!r}"
        )
    if structure_code not in rule["allowed_codes"]:
        raise ValueError(
            f"structure_code={structure_code!r} is not allowed for basic_concept={basic_concept!r}. "
            f"Allowed: {rule['allowed_codes']}"
        )
