"""I/O helpers: read the gold-standard Excel sheet for Assertion Developer evaluation.

Mirrors concept_mapper/io.py's pattern for the sibling agent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Only CP-parent rows carry a real indicator_concept_gold to develop an assertion
# for (CI-parent rows in this sheet describe the topic itself, not a sub-indicator —
# out of scope for now, see conversation decision to isolate Assertion Developer
# testing from Concept Mapper's own indicator generation).
GOLD_COLUMNS = [
    "example_id",
    "concept_id",
    "input_topic_parent_concept",
    "concept_level_ci_cp_gold",
    "indicator_concept_gold",
    "basic_concept_gold",
    "basic_concept_key",
    "structure_code_gold",
    "gold_assertion",
]


def read_assertion_gold_xlsx(
    path: str | Path,
    sheet_name: str = "Source Items + Assertions (cor)",
) -> list[dict]:
    """Read the gold-standard Excel sheet and return CP-parent rows only.

    basic_concept_key is the snake_case form (e.g. "cognitive_judgment") matching
    assertion_rules.py's BASIC_CONCEPT_RULES keys directly — prefer it over
    basic_concept_gold (Title Case display form, e.g. "Cognitive judgment") for any
    programmatic comparison.

    structure_code_gold is missing for a few rows (left as None) — evaluator should
    skip the structure-code check for those rather than counting it as wrong.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)

    missing = [c for c in GOLD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Excel sheet {sheet_name!r} is missing expected columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df[df["concept_level_ci_cp_gold"] == "CP"].copy()
    df = df[GOLD_COLUMNS]

    records = df.to_dict(orient="records")
    # pandas float columns coerce None back to NaN on assignment, so normalise
    # missing structure_code_gold to real None only after converting to plain dicts
    # (bool(float("nan")) is True in Python, which would silently break any
    # `if gold_structure_code:` check downstream).
    for r in records:
        if isinstance(r["structure_code_gold"], float) and r["structure_code_gold"] != r["structure_code_gold"]:
            r["structure_code_gold"] = None

    return records
