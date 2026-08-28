"""Rule table mapping basic concepts to allowed assertion structure codes.

Based on Saris & Gallhofer (2007), Table 2.1 "The basic structures of simple
assertions" and the accompanying notation.

Notation (S&G): x denotes the grammatical subject, I the link verb, P the
predicator. Standing subject substitutions are g (the government), o (anyone),
r (the respondent) and v (a value, i.e. a basic goal or state individuals
strive for). Predicators include C (subject causes object), D (deeds),
F (feelings), FD (future deeds), H(+I) ("has to"/"should" + infinitive),
HR ("has the right to"), PR (preferences), S (similarity/difference).

Corrections applied against Table 2.1:
  * evaluation  xle -> xIe, importance xli -> xIi, feelings xlf -> xIf
    (capital I, the link verb, had been transcribed as a lowercase l)
  * values      xlv -> vIi  (the subject is the value itself)
  * examples rewritten so the assertion's grammatical subject matches its code.
    Only codes whose subject is r (the respondent) are phrased "The respondent
    ..."; xIc, xCy, xSy, o(H+I)y, g(H+I)y, xHRy, xFD and xPey take the object
    as subject, as the notation requires.

rFy, rDy and rD are retained as instantiations of xFy / xDy / xD with the
respondent as grammatical subject, which the notation explicitly allows.

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
        "allowed_codes": ["xIe"],
        "default_code": "xIe",
        "structure_id": "structure_1",
        "example": "The government is good.",
    },
    "importance": {
        "variable_type": "subjective",
        "allowed_codes": ["xIi"],
        "default_code": "xIi",
        "structure_id": "structure_1",
        "example": "My work is important.",
    },
    "values": {
        "variable_type": "subjective",
        "allowed_codes": ["vIi"],
        "default_code": "vIi",
        "structure_id": "structure_1",
        "example": "Honesty is important.",
    },
    "feelings": {
        "variable_type": "subjective",
        # xIf is the structure-1 form ("The situation is frightening.");
        # xFy / xPf are structure 2, with rFy the respondent-subject case.
        "allowed_codes": ["xIf", "xFy", "xPf", "rFy"],
        "default_code": "rFy",
        "structure_id": "structure_2",
        "example": "The respondent fears burglary.",
    },
    "cognitive_judgment": {
        "variable_type": "subjective",
        "allowed_codes": ["xIc"],
        "default_code": "xIc",
        "structure_id": "structure_1",
        "example": "The economy is weak.",
    },
    "causal_relationship": {
        "variable_type": "subjective",
        "allowed_codes": ["xIca", "xCy"],
        "default_code": "xCy",
        "structure_id": "structure_2",
        "example": "Unemployment causes crime.",
    },
    "similarity_relationship": {
        "variable_type": "subjective",
        "allowed_codes": ["xIs", "xSy"],
        "default_code": "xSy",
        "structure_id": "structure_2",
        "example": "Party A is similar to party B.",
    },
    "preference": {
        "variable_type": "subjective",
        "allowed_codes": ["xIpr", "xPRy"],
        "default_code": "xPRy",
        "structure_id": "structure_2",
        "example": "The respondent prefers the city to the countryside.",
    },
    "norms": {
        "variable_type": "subjective",
        "allowed_codes": ["o(H+I)y", "o(H+I)"],
        "default_code": "o(H+I)y",
        "structure_id": "structure_2",
        "example": "One ought to pay taxes.",
    },
    "policies": {
        "variable_type": "subjective",
        "allowed_codes": ["g(H+I)y"],
        "default_code": "g(H+I)y",
        "structure_id": "structure_2",
        "example": "The government should reduce inequality.",
    },
    "rights": {
        "variable_type": "subjective",
        "allowed_codes": ["xIri", "xHRy"],
        "default_code": "xHRy",
        "structure_id": "structure_2",
        "example": "Everyone has the right to free speech.",
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
        "example": "The economic situation will worsen.",
    },
    "evaluative_belief": {
        "variable_type": "subjective",
        "allowed_codes": ["xPey", "xPye", "xPe"],
        "default_code": "xPey",
        "structure_id": "structure_2",
        "example": "Immigration is good for the economy.",
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
        "example": "A burglary occurred.",
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
        "example": "The respondent goes to work by public transport.",
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
