"""Pydantic schema for AssertionOutput with cross-field validation.

The model outputs every field except `structure_id`, which is derived
programmatically from the rule table so the rule table stays the single
source of truth.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

from .assertion_rules import BASIC_CONCEPT_RULES

# Assertion text must not contain these scale/option markers
_SCALE_MARKERS: list[str] = [
    "on a scale",
    "likert",
    "agree",
    "disagree",
    "yes/no",
    "true/false",
    "1 =",
    "2 =",
    "3 =",
    "4 =",
    "5 =",
    "answer options",
    "response options",
]


class AssertionOutput(BaseModel):
    """One formal declarative assertion produced by the Assertion Developer Agent."""

    parent_concept: str
    input_indicator: str
    variable_type: Literal["subjective", "objective"]
    basic_concept: str
    domain: str
    structure_code: str
    structure_id: str = ""   # derived by validator — model must NOT output this
    assertion: str
    rationale: str
    warnings: list[str]

    @model_validator(mode="after")
    def _validate_and_attach_structure_id(self) -> "AssertionOutput":
        # 1. basic_concept must be known
        rule = BASIC_CONCEPT_RULES.get(self.basic_concept)
        if rule is None:
            raise ValueError(
                f"Unknown basic_concept: {self.basic_concept!r}. "
                f"Must be one of: {sorted(BASIC_CONCEPT_RULES)}"
            )

        # 2. variable_type must match the rule
        if self.variable_type != rule["variable_type"]:
            raise ValueError(
                f"basic_concept={self.basic_concept!r} requires "
                f"variable_type={rule['variable_type']!r}, got {self.variable_type!r}"
            )

        # 3. structure_code must be allowed for this basic_concept
        if self.structure_code not in rule["allowed_codes"]:
            raise ValueError(
                f"structure_code={self.structure_code!r} is not allowed for "
                f"basic_concept={self.basic_concept!r}. "
                f"Allowed codes: {rule['allowed_codes']}"
            )

        # 4. Derive structure_id from the rule table (overwrites any model output)
        self.structure_id = rule["structure_id"]

        # 5. Assertion must be a declarative statement, not a question
        if self.assertion.strip().endswith("?"):
            raise ValueError(
                "assertion must be a declarative statement, not a question "
                f"(ends with '?'): {self.assertion!r}"
            )

        # 6. Assertion must not contain response scale or option markers
        lower = self.assertion.lower()
        for marker in _SCALE_MARKERS:
            if marker in lower:
                raise ValueError(
                    f"assertion must not contain response scale or option markers "
                    f"(found {marker!r}): {self.assertion!r}"
                )

        return self
