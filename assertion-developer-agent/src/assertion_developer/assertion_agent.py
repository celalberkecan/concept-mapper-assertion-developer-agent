"""AssertionDeveloperAgent: stateless wrapper around an LLM call with retry logic."""

from __future__ import annotations

from survey_agent_lib.llm_clients.base import BaseLLMClient

from .assertion_parser import parse_assertion
from .assertion_prompts import build_assertion_messages
from .assertion_schemas import AssertionOutput

_REPAIR_MESSAGE = (
    "Your previous response could not be parsed as a valid JSON object matching the required schema. "
    "Output ONLY the JSON object — no prose, no markdown fences, no explanation before or after. "
    "Remember: do not include 'structure_id' in your output; do not end the assertion with '?'."
)


class AssertionDeveloperAgent:
    """Develops a formal declarative assertion from a CI-level indicator.

    Args:
        client: Any BaseLLMClient (OpenAI, Ollama, Transformers, or Fake).
    """

    def __init__(self, client: BaseLLMClient) -> None:
        self.client = client

    def develop_assertion(
        self,
        parent_concept: str,
        indicator_name: str,
        indicator_role: str,
    ) -> AssertionOutput:
        """Develop an assertion, retrying once with a repair prompt on failure."""
        messages = build_assertion_messages(
            parent_concept, indicator_name, indicator_role
        )
        raw = self.client.generate(messages)

        try:
            return parse_assertion(raw)
        except ValueError:
            pass  # first attempt failed — try once more with repair context

        repair_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _REPAIR_MESSAGE},
        ]
        raw2 = self.client.generate(repair_messages)
        # Let this exception propagate — we already retried once
        return parse_assertion(raw2)

    def develop_assertion_with_raw(
        self,
        parent_concept: str,
        indicator_name: str,
        indicator_role: str,
    ) -> tuple[AssertionOutput, str]:
        """Same as develop_assertion but also returns the raw LLM response used for parsing."""
        messages = build_assertion_messages(
            parent_concept, indicator_name, indicator_role
        )
        raw = self.client.generate(messages)

        try:
            assertion = parse_assertion(raw)
            return assertion, raw
        except ValueError:
            pass

        repair_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _REPAIR_MESSAGE},
        ]
        raw2 = self.client.generate(repair_messages)
        assertion = parse_assertion(raw2)
        return assertion, raw2
