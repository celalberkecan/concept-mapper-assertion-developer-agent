"""Shared I/O utilities: JSONL read/write."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
