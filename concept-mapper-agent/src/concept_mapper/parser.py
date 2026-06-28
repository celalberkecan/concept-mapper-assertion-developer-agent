"""Robust JSON extraction and ConceptMap parsing from raw LLM output."""

from __future__ import annotations

import json

from pydantic import ValidationError

from .schemas import ConceptMap


def extract_json_object(text: str) -> str:
    """Return the first complete JSON object found in *text*.

    Handles leading/trailing prose, code fences, and extra whitespace.
    Raises ValueError if no balanced JSON object is found.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in the response (no '{' character).")

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError(
        "Unterminated JSON object in the response — braces are not balanced."
    )


def parse_concept_map(text: str) -> ConceptMap:
    """Extract, parse, and validate a ConceptMap from raw LLM output.

    Raises ValueError with a helpful preview on any failure.
    """
    preview = text[:300].replace("\n", " ")

    try:
        json_str = extract_json_object(text)
    except ValueError as exc:
        raise ValueError(f"{exc}\n\nResponse preview: {preview!r}") from exc

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON decode error: {exc}\n\nResponse preview: {preview!r}"
        ) from exc

    try:
        return ConceptMap(**data)
    except (ValidationError, TypeError) as exc:
        raise ValueError(
            f"Schema validation failed: {exc}\n\nResponse preview: {preview!r}"
        ) from exc
