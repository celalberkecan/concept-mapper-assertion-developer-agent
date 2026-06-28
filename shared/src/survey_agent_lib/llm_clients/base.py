from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> str:
        """Send *messages* to the LLM and return the assistant reply as plain text."""
