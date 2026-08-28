"""I/O helpers: read gold Excel, write/read JSONL prediction files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Gold data
# ---------------------------------------------------------------------------

GOLD_COLUMNS = [
    "concept_id",
    "input_topic_parent_concept",
    "concept_level_ci_cp_gold",
    "concept_level_indicator_model_gold",
    "gold_indicators_conceptual",
    "indicator_count_gold",
]

# Placeholder value used in the sheet for CI rows that have no indicators
_CI_PLACEHOLDER = "NA — direct concept-by-intuition."


def read_concept_mapper_gold_xlsx(
    path: str | Path,
    sheet_name: str = "Concept Mapper Gold",
) -> list[dict]:
    """Read the gold-standard Excel sheet and return a list of row dicts.

    Only the columns in GOLD_COLUMNS are returned. The indicator_model field
    is normalised: NaN (CI rows) becomes the string "NA". The
    gold_indicators_conceptual placeholder for CI rows is normalised to "".

    If the sheet has concept_level_ci_cp_leo / concept_level_indicatory_leo columns
    (human-relabeled gold, more reliable than the original *_gold columns), those
    values are used instead wherever present, falling back to the *_gold columns for
    any row left unlabeled.
    """
    raw_df = pd.read_excel(path, sheet_name=sheet_name)

    missing = [c for c in GOLD_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(
            f"Excel sheet {sheet_name!r} is missing expected columns: {missing}\n"
            f"Found columns: {list(raw_df.columns)}"
        )

    df = raw_df[GOLD_COLUMNS].copy()

    if "concept_level_ci_cp_leo" in raw_df.columns:
        leo_ci_cp = raw_df["concept_level_ci_cp_leo"]
        df["concept_level_ci_cp_gold"] = leo_ci_cp.where(
            leo_ci_cp.notna(), df["concept_level_ci_cp_gold"]
        )
    if "concept_level_indicatory_leo" in raw_df.columns:
        leo_indicator_model = raw_df["concept_level_indicatory_leo"]
        df["concept_level_indicator_model_gold"] = leo_indicator_model.where(
            leo_indicator_model.notna(), df["concept_level_indicator_model_gold"]
        )

    # Normalise NaN indicator_model (CI rows) → "NA"
    df["concept_level_indicator_model_gold"] = (
        df["concept_level_indicator_model_gold"].fillna("NA")
    )

    # Normalise CI placeholder in indicators column → empty string
    df["gold_indicators_conceptual"] = df["gold_indicators_conceptual"].apply(
        lambda v: "" if str(v).strip() == _CI_PLACEHOLDER else str(v)
    )

    # CI rows always have 0 indicators; fill in any un-annotated CI rows before
    # casting to int. CP rows left blank stay NaN -> Int64 pd.NA (can't be guessed).
    is_ci = df["concept_level_ci_cp_gold"] == "CI"
    df.loc[is_ci, "indicator_count_gold"] = df.loc[is_ci, "indicator_count_gold"].fillna(0)
    df["indicator_count_gold"] = df["indicator_count_gold"].astype("Int64")

    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write a list of dicts to a JSONL file (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dicts."""
    path = Path(path)
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return records
