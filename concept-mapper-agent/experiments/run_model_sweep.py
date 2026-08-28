"""Run all 4 prompt techniques (zero-shot, prose few-shot, message-history few-shot,
GEPA-optimized) against ONE generation model/provider, over the full 51-row gold set.

This is the open-source-model counterpart to run_fewshot_ablation.py + run_gepa_eval.py
(which both hardcode OpenAIClient/gpt-4o-mini) — same evaluation logic, reused directly,
just parameterized over --provider/--config so it also runs against
`--provider transformers` (LRZ, local HF checkpoints) or `--provider ollama` (local Mac).

The judge is always OpenAI gpt-4.1-mini regardless of which model is being evaluated,
so scores stay comparable across models — this requires outbound internet access to
the OpenAI API from wherever this script runs (confirmed working on LRZ login/compute
nodes via `curl https://api.openai.com`).

Usage (run from concept-mapper-agent/, with its venv/conda env active):
    python experiments/run_model_sweep.py --provider transformers --config configs/transformers.yaml --model-tag qwen3-8b
    python experiments/run_model_sweep.py --provider transformers --config configs/transformers_llama.yaml --model-tag llama-3.1-8b-instruct
    python experiments/run_model_sweep.py --provider transformers --config configs/transformers_deepseek_reasoning.yaml --model-tag deepseekr1-0528-qwen3-8b
    python experiments/run_model_sweep.py --provider ollama --config configs/ollama.yaml --model-tag qwen3-8b

Output: experiments/outputs/{model_tag}_{variant}.jsonl (one per technique) and
experiments/outputs/{model_tag}_summary.json (all 4 techniques' summary metrics).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from pydantic import ValidationError  # noqa: E402

from concept_mapper.evaluator import compute_summary, evaluate_batch  # noqa: E402
from concept_mapper.io import read_concept_mapper_gold_xlsx, write_jsonl  # noqa: E402
from concept_mapper.parser import extract_json_object, parse_concept_map  # noqa: E402
from concept_mapper.schemas import ConceptMap  # noqa: E402
from survey_agent_lib.llm_clients.base import BaseLLMClient  # noqa: E402
from survey_agent_lib.llm_clients.openai_client import OpenAIClient  # noqa: E402

# Reuse the exact message-builders and repair-prompt text from the original ablation
# script rather than duplicating them (sibling file in the same directory, importable
# because `python experiments/run_model_sweep.py` puts this dir on sys.path).
from run_fewshot_ablation import (  # noqa: E402
    _REPAIR_MESSAGE,
    build_message_history_fewshot,
    build_prose_fewshot,
    build_zero_shot,
)

DATA_PATH = REPO_ROOT / "data" / "gesis_concept_mapper_assertion_evaluation_adjusted.xlsx"
OUT_DIR = REPO_ROOT / "experiments" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GEPA_INSTRUCTION_PATH = REPO_ROOT / "experiments" / "gepa_optimized_instruction.txt"


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_client(provider: str, cfg: dict) -> BaseLLMClient:
    if provider == "openai":
        return OpenAIClient(
            model=cfg.get("model", "gpt-4o-mini"),
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 1200),
        )
    elif provider == "ollama":
        from survey_agent_lib.llm_clients.ollama_client import OllamaClient

        return OllamaClient(
            model=cfg.get("model", "qwen2.5:7b-instruct"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 1200),
            timeout=cfg.get("timeout", 120),
        )
    elif provider == "transformers":
        from survey_agent_lib.llm_clients.transformers_client import TransformersClient

        model_path = cfg.get("model_path")
        if not model_path:
            raise ValueError("'model_path' is required in the transformers config.")
        return TransformersClient(
            model_path=model_path,
            torch_dtype=cfg.get("torch_dtype", "bfloat16"),
            device_map=cfg.get("device_map", "auto"),
            max_new_tokens=cfg.get("max_new_tokens", 1200),
            temperature=cfg.get("temperature", 0.0),
            trust_remote_code=cfg.get("trust_remote_code", False),
            load_in_4bit=cfg.get("load_in_4bit", False),
            load_in_8bit=cfg.get("load_in_8bit", False),
        )
    raise ValueError(f"Unknown provider {provider!r}. Choose one of: openai, ollama, transformers.")


def _parse_with_topic_injected(raw: str, topic: str) -> ConceptMap:
    """Used for the GEPA-optimized instruction only — see run_gepa_eval.py's docstring
    for why input_topic/warnings need injecting before validation for that variant."""
    json_str = extract_json_object(raw)
    data = json.loads(json_str)
    data.setdefault("input_topic", topic)
    data.setdefault("warnings", [])
    return ConceptMap(**data)


def run_variant(
    name: str, build_fn, gold_rows: list[dict], client: BaseLLMClient, use_topic_injection: bool
) -> list[dict]:
    records = []
    for i, row in enumerate(gold_rows, 1):
        topic = row["input_topic_parent_concept"]
        cid = row["concept_id"]
        messages = build_fn(topic)
        raw = client.generate(messages)
        try:
            cm = _parse_with_topic_injected(raw, topic) if use_topic_injection else parse_concept_map(raw)
            record = {"concept_id": cid, "input_topic": topic, **cm.model_dump()}
        except (ValueError, ValidationError):
            repair_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _REPAIR_MESSAGE},
            ]
            raw2 = client.generate(repair_messages)
            try:
                cm = _parse_with_topic_injected(raw2, topic) if use_topic_injection else parse_concept_map(raw2)
                record = {"concept_id": cid, "input_topic": topic, **cm.model_dump()}
            except (ValueError, ValidationError) as exc:
                record = {"concept_id": cid, "input_topic": topic, "error": str(exc)}

        status = "OK" if "error" not in record else f"ERROR: {record['error'][:150]}"
        print(f"[{name}] {i}/{len(gold_rows)} {topic!r} -> {status}")
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["transformers", "ollama", "openai"])
    parser.add_argument("--config", required=True, help="Path to the generation model's YAML config.")
    parser.add_argument("--model-tag", required=True, help="Short label for output filenames, e.g. 'qwen3-8b'.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    client = _build_client(args.provider, cfg)
    judge_client = OpenAIClient(model="gpt-4.1-mini", temperature=0.0, max_tokens=300)

    gold_rows = read_concept_mapper_gold_xlsx(DATA_PATH)
    print(f"Loaded {len(gold_rows)} gold rows. Model tag: {args.model_tag!r}, provider: {args.provider!r}")

    variants = {
        "a_zero_shot": (build_zero_shot, False),
        "b_prose_fewshot": (build_prose_fewshot, False),
        "c_message_history_fewshot": (build_message_history_fewshot, False),
    }

    all_summaries: dict[str, dict] = {}

    for name, (build_fn, use_injection) in variants.items():
        print(f"\n=== {args.model_tag} / {name} ===")
        records = run_variant(name, build_fn, gold_rows, client, use_injection)
        write_jsonl(OUT_DIR / f"{args.model_tag}_{name}.jsonl", records)
        eval_records = evaluate_batch(gold_rows, records, judge_client=judge_client)
        summary = compute_summary(eval_records)
        all_summaries[name] = summary
        print(f"Summary: {summary}")

    if GEPA_INSTRUCTION_PATH.exists():
        instruction = GEPA_INSTRUCTION_PATH.read_text().strip()

        def build_gepa(topic: str) -> list[dict]:
            return [
                {"role": "system", "content": instruction},
                {"role": "user", "content": f'Map this concept: "{topic}"'},
            ]

        print(f"\n=== {args.model_tag} / d_gepa_optimized ===")
        records = run_variant("d_gepa_optimized", build_gepa, gold_rows, client, use_topic_injection=True)
        write_jsonl(OUT_DIR / f"{args.model_tag}_d_gepa_optimized.jsonl", records)
        eval_records = evaluate_batch(gold_rows, records, judge_client=judge_client)
        summary = compute_summary(eval_records)
        all_summaries["d_gepa_optimized"] = summary
        print(f"Summary: {summary}")
    else:
        print(f"\nNo GEPA instruction found at {GEPA_INSTRUCTION_PATH}, skipping technique (d).")

    print(f"\n=== Final summary for {args.model_tag} ===")
    for name, summary in all_summaries.items():
        print(f"{name}: {summary}")

    summary_path = OUT_DIR / f"{args.model_tag}_summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"\nWrote combined summary to {summary_path}")


if __name__ == "__main__":
    main()
