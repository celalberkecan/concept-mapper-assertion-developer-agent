"""Hugging Face Transformers client for use on LRZ LMU servers.

torch and transformers are imported lazily inside __init__ so that the rest of
the package works on machines without GPU drivers or the transformers library.
"""

from __future__ import annotations

from typing import Any

from .base import BaseLLMClient


class TransformersClient(BaseLLMClient):
    """LLM client backed by a locally loaded Hugging Face causal LM.

    Intended for use on LRZ LMU servers with downloaded model checkpoints.
    Install extras with:  pip install torch transformers accelerate

    Args:
        model_path: Local path or HuggingFace Hub identifier of the model.
        torch_dtype: Weight dtype — 'bfloat16' (recommended), 'float16', 'float32', or 'auto'.
        device_map: Passed to from_pretrained; 'auto' distributes across available GPUs.
        max_new_tokens: Hard token budget for generation.
        temperature: Sampling temperature; 0.0 uses greedy decoding.
    """

    def __init__(
        self,
        model_path: str,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 1200,
        temperature: float = 0.0,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for TransformersClient.\n"
                "Install with:  pip install torch transformers accelerate"
            ) from exc

        _dtype_map: dict[str, Any] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        resolved_dtype = _dtype_map.get(torch_dtype, "auto")

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=resolved_dtype,
            device_map=device_map,
        )
        self.model.eval()

        self.max_new_tokens = max_new_tokens
        self.default_temperature = temperature

    def generate(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_new_tokens

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text: str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = _build_simple_prompt(messages)

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        input_len = input_ids.shape[-1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with self._torch.no_grad():
            output_ids = self.model.generate(input_ids, **gen_kwargs)

        new_token_ids = output_ids[0][input_len:]
        return self.tokenizer.decode(new_token_ids, skip_special_tokens=True)


def _build_simple_prompt(messages: list[dict]) -> str:
    """Fallback prompt builder when the tokenizer has no chat template."""
    parts: list[str] = []
    for msg in messages:
        role = msg["role"].upper()
        parts.append(f"[{role}]\n{msg['content']}")
    parts.append("[ASSISTANT]")
    return "\n\n".join(parts)
