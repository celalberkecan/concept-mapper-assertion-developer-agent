"""JSON extraction and AssertionOutput parsing from raw LLM output."""

from __future__ import annotations

import json

from pydantic import ValidationError
from survey_agent_lib.parser import extract_json_object

from .assertion_schemas import AssertionOutput


def parse_assertion(text: str) -> AssertionOutput:
    """Extract, parse, and validate an AssertionOutput from raw LLM output.

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
        return AssertionOutput(**data)
    except (ValidationError, TypeError) as exc:
        raise ValueError(
            f"Schema validation failed: {exc}\n\nResponse preview: {preview!r}"
        ) from exc
