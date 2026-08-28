"""Generic JSON extraction utility for parsing raw LLM output."""

from __future__ import annotations


def extract_json_object(text: str) -> str:
    """Return the first complete JSON object found in *text*.

    Handles leading/trailing prose, code fences, and extra whitespace.
    Raises ValueError if no balanced JSON object is found.
    """
    # Reasoning models (e.g. DeepSeek-R1-distill) emit a <think>...</think> block
    # before the actual answer. Without stripping it, the first-'{' scan below can
    # grab a stray brace from inside the reasoning trace instead of the real JSON.
    think_end = text.find("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :]

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
