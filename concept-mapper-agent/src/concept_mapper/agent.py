"""ConceptMapperAgent: stateless wrapper around an LLM call with retry logic."""

from __future__ import annotations

from .llm_clients.base import BaseLLMClient
from .parser import parse_concept_map
from .prompts import build_concept_mapper_messages
from .schemas import ConceptMap

_REPAIR_MESSAGE = (
    "Your previous response could not be parsed as a valid JSON object matching the required schema. "
    "Output ONLY the JSON object — no prose, no markdown fences, no explanation before or after."
)


class ConceptMapperAgent:
    """Maps a broad survey topic to a structured ConceptMap using an LLM.

    Args:
        client: Any BaseLLMClient (OpenAI, Ollama, Transformers, or Fake).
    """

    def __init__(self, client: BaseLLMClient) -> None:
        self.client = client

    def map_concept(self, input_topic: str) -> ConceptMap:
        """Map *input_topic* to a ConceptMap, retrying once with a repair prompt on failure."""
        messages = build_concept_mapper_messages(input_topic)
        raw = self.client.generate(messages)

        try:
            return parse_concept_map(raw)
        except ValueError:
            pass  # first attempt failed — try once more with repair context

        repair_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _REPAIR_MESSAGE},
        ]
        raw2 = self.client.generate(repair_messages)
        # Let this exception propagate — we already retried once
        return parse_concept_map(raw2)

    def map_concept_with_raw(self, input_topic: str) -> tuple[ConceptMap, str]:
        """Same as map_concept but also returns the raw LLM response used for parsing."""
        messages = build_concept_mapper_messages(input_topic)
        raw = self.client.generate(messages)

        try:
            concept_map = parse_concept_map(raw)
            return concept_map, raw
        except ValueError:
            pass

        repair_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _REPAIR_MESSAGE},
        ]
        raw2 = self.client.generate(repair_messages)
        concept_map = parse_concept_map(raw2)
        return concept_map, raw2
