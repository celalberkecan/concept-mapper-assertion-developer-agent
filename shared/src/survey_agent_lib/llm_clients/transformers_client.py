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
        max_new_tokens: Hard token budget for generation. Reasoning models (e.g.
            DeepSeek-R1-distill) need a much larger budget than instruct models — their
            <think>...</think> trace alone can run several hundred to a few thousand
            tokens before the actual answer starts.
        temperature: Sampling temperature; 0.0 uses greedy decoding.
        trust_remote_code: Passed to from_pretrained. Some model families (e.g. GLM)
            ship custom modeling code and fail to load without this.
        load_in_4bit: Load weights quantized to 4-bit (bitsandbytes nf4) instead of
            torch_dtype. A 7-8B model drops from ~15-16GB to ~4-5GB — needed on
            16GB-per-GPU hardware (e.g. LRZ's V100 nodes), where an 8B model in
            fp16/bf16 does not fit at all and a 7B model leaves no room for KV cache.
            Requires the `bitsandbytes` package. Takes precedence over load_in_8bit.
        load_in_8bit: Load weights quantized to 8-bit instead of torch_dtype. Ignored
            if load_in_4bit is also set. Requires `bitsandbytes`.
    """

    def __init__(
        self,
        model_path: str,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 1200,
        temperature: float = 0.0,
        trust_remote_code: bool = False,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
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

        quantization_config = None
        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    "bitsandbytes is required for load_in_4bit/load_in_8bit.\n"
                    "Install with:  pip install bitsandbytes"
                ) from exc
            compute_dtype = resolved_dtype if isinstance(resolved_dtype, torch.dtype) else torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit and not load_in_4bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
            )

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=resolved_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config,
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

        prompt_text = self._render_prompt(messages)

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
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
            output_ids = self.model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

        new_token_ids = output_ids[0][input_len:]
        return self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

    def _render_prompt(self, messages: list[dict]) -> str:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return _build_simple_prompt(messages)

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Some chat templates (e.g. Gemma-2, older Mistral-Instruct) reject a
            # leading "system" role outright. Fall back to folding it into the first
            # user turn rather than failing the whole generation call.
            merged = _merge_system_into_first_user(messages)
            return self.tokenizer.apply_chat_template(
                merged,
                tokenize=False,
                add_generation_prompt=True,
            )


def _merge_system_into_first_user(messages: list[dict]) -> list[dict]:
    """Fold a leading system message into the first user turn.

    Used as a fallback when a tokenizer's chat template doesn't support a
    separate "system" role at all (raises instead of rendering it).
    """
    if not messages or messages[0]["role"] != "system":
        return messages

    system_content = messages[0]["content"]
    rest = messages[1:]
    if rest and rest[0]["role"] == "user":
        merged_first = {"role": "user", "content": f"{system_content}\n\n{rest[0]['content']}"}
        return [merged_first, *rest[1:]]
    # No user turn to merge into (shouldn't normally happen) — demote system to user.
    return [{"role": "user", "content": system_content}, *rest]


def _build_simple_prompt(messages: list[dict]) -> str:
    """Fallback prompt builder when the tokenizer has no chat template."""
    parts: list[str] = []
    for msg in messages:
        role = msg["role"].upper()
        parts.append(f"[{role}]\n{msg['content']}")
    parts.append("[ASSISTANT]")
    return "\n\n".join(parts)
