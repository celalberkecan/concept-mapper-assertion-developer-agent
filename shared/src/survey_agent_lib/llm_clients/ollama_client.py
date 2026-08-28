"""Ollama client using the local HTTP API (/api/chat)."""

from __future__ import annotations

from .base import BaseLLMClient


class OllamaClient(BaseLLMClient):
    """LLM client backed by a locally running Ollama instance.

    Start Ollama with:  ollama serve
    Pull a model with:  ollama pull qwen2.5:7b-instruct
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 1200,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.timeout = timeout

    def generate(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        try:
            import requests
        except ImportError as exc:
            raise ImportError("Run: pip install requests") from exc

        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running:  ollama serve"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise TimeoutError(
                f"Ollama did not respond within {self.timeout}s. Reasoning models "
                "(e.g. deepseek-r1) can take a while for long <think> traces — "
                "increase the 'timeout' config value if this keeps happening."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"Ollama HTTP error: {exc}") from exc

        return resp.json()["message"]["content"]
