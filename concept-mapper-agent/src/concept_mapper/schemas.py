from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


class Indicator(BaseModel):
    name: str
    definition: str
    role: Literal["component", "manifestation", "mixed", "direct", "other"]


class ConceptMap(BaseModel):
    input_topic: str
    ci_or_cp: Literal["CI", "CP"]
    indicator_model: Literal["NA", "formative", "reflective", "mixed"]
    construct_definition: str
    indicators: list[Indicator]
    rationale: str
    warnings: list[str]

    @model_validator(mode="after")
    def _validate_ci_cp_consistency(self) -> ConceptMap:
        if self.ci_or_cp == "CI":
            if self.indicator_model != "NA":
                raise ValueError(
                    f"CI concepts must have indicator_model='NA', got {self.indicator_model!r}"
                )
            if self.indicators:
                raise ValueError(
                    "CI concepts must have an empty indicators list"
                )
        elif self.ci_or_cp == "CP":
            if self.indicator_model not in ("formative", "reflective", "mixed"):
                raise ValueError(
                    f"CP concepts must have indicator_model in "
                    f"['formative', 'reflective', 'mixed'], got {self.indicator_model!r}"
                )
            if len(self.indicators) < 2:
                raise ValueError(
                    f"CP concepts must have at least 2 indicators, got {len(self.indicators)}"
                )
        return self
