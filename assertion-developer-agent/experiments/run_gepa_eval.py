"""Evaluate the GEPA-optimized instruction over the full 92-row gold set, using the
same pipeline (OpenAIClient gpt-4o-mini generation, llm_judge gpt-4.1-mini) as
experiments/run_fewshot_ablation.py's a_zero_shot variant, so the two are directly
comparable apples-to-apples (both zero-shot, same judge, same full gold set — not just
the 28-row DSPy val split used during optimization itself).

Does NOT reuse run_fewshot_ablation.run_variant()'s parsing as-is: the DSPy Signature
correctly models parent_concept/indicator_name/indicator_role as InputFields only (no
reason to ask the model to echo back strings the caller already has), so the
GEPA-discovered instruction never asks for parent_concept or input_indicator in its
output JSON. The production AssertionOutput schema requires both present in the
output regardless (same gap as Concept Mapper's input_topic issue) — so we inject the
known values before validation instead of asking GEPA to make the model redundantly
repeat them. structure_id is never requested from the model in any variant — it's
always derived programmatically by AssertionOutput's validator from BASIC_CONCEPT_RULES.

Usage:
    python experiments/run_gepa_eval.py path/to/optimized_instruction.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from pydantic import ValidationError  # noqa: E402
from survey_agent_lib.llm_clients.openai_client import OpenAIClient  # noqa: E402
from survey_agent_lib.parser import extract_json_object  # noqa: E402

from assertion_developer.assertion_evaluator import (  # noqa: E402
    compute_gold_summary,
    evaluate_batch_against_gold,
)
from assertion_developer.assertion_schemas import AssertionOutput  # noqa: E402
from assertion_developer.io import read_assertion_gold_xlsx  # noqa: E402

DATA_PATH = (
    REPO_ROOT / "data" / "gesis_concept_mapper_assertion_evaluation_adjusted_for_assertion_agent_final.xlsx"
)
OUT_DIR = REPO_ROOT / "experiments" / "outputs"

_REPAIR_MESSAGE = (
    "Your previous response could not be parsed as a valid JSON object matching the required schema. "
    "Output ONLY the JSON object — no prose, no markdown fences, no explanation before or after. "
    "Remember: do not include 'structure_id' in your output; do not end the assertion with '?'."
)


def _format_user_message(parent_concept: str, indicator_name: str, indicator_role: str) -> str:
    return (
        f'Develop assertion for indicator: "{indicator_name}"\n'
        f'Parent concept: "{parent_concept}"\n'
        f"Indicator role: {indicator_role}"
    )


def _parse_with_context_injected(raw: str, parent_concept: str, indicator_name: str) -> AssertionOutput:
    json_str = extract_json_object(raw)
    data = json.loads(json_str)
    data.setdefault("parent_concept", parent_concept)
    data.setdefault("input_indicator", indicator_name)
    data.setdefault("warnings", [])
    return AssertionOutput(**data)


def run_variant(name: str, instruction: str, gold_rows: list[dict], client: OpenAIClient) -> list[dict]:
    records = []
    for i, row in enumerate(gold_rows, 1):
        example_id = row["example_id"]
        parent_concept = row["input_topic_parent_concept"]
        indicator_name = row["indicator_concept_gold"]
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": _format_user_message(parent_concept, indicator_name, "component")},
        ]
        raw = client.generate(messages)
        try:
            out = _parse_with_context_injected(raw, parent_concept, indicator_name)
            record = {"example_id": example_id, **out.model_dump()}
        except (ValueError, ValidationError):
            repair_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _REPAIR_MESSAGE},
            ]
            raw2 = client.generate(repair_messages)
            try:
                out = _parse_with_context_injected(raw2, parent_concept, indicator_name)
                record = {"example_id": example_id, **out.model_dump()}
            except (ValueError, ValidationError) as exc:
                record = {"example_id": example_id, "error": str(exc)}
        status = "OK" if "error" not in record else f"ERROR: {record['error'][:150]}"
        print(f"[{name}] {i}/{len(gold_rows)} {indicator_name!r} -> {status}")
        records.append(record)
    return records


def main() -> None:
    instruction_path = Path(sys.argv[1])
    instruction = instruction_path.read_text().strip()

    gold_rows = read_assertion_gold_xlsx(DATA_PATH)
    generation_client = OpenAIClient(model="gpt-4o-mini", temperature=0.0, max_tokens=800)
    judge_client = OpenAIClient(model="gpt-4.1-mini", temperature=0.0, max_tokens=300)

    records = run_variant("d_gepa_optimized", instruction, gold_rows, generation_client)

    from survey_agent_lib.io import write_jsonl

    write_jsonl(OUT_DIR / "d_gepa_optimized.jsonl", records)

    eval_records = evaluate_batch_against_gold(gold_rows, records, judge_client=judge_client)
    summary = compute_gold_summary(eval_records)
    print("\nSummary for d_gepa_optimized (full 92-row gold set):")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
