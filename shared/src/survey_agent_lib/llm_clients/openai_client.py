"""OpenAI client using the Responses API (client.responses.create)."""

from __future__ import annotations

import os

from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """LLM client backed by the OpenAI Responses API.

    Uses provider-neutral plain-text output — structured output mode is
    intentionally disabled so the same JSON parser works across all providers.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and fill in your key, or run:\n"
                "  export OPENAI_API_KEY=sk-..."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Run: pip install openai") from exc

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens

    def generate(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        # The Responses API separates the system prompt into `instructions`
        # and conversation turns into `input`.
        instructions: str | None = None
        input_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                instructions = msg["content"]
            else:
                input_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs: dict = {
            "model": self.model,
            "input": input_messages,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if instructions:
            kwargs["instructions"] = instructions

        response = self._client.responses.create(**kwargs)

        if response.output_text is None:
            raise RuntimeError(
                "OpenAI returned an empty response. "
                "This may be caused by a content filter or an empty model output."
            )
        return response.output_text
