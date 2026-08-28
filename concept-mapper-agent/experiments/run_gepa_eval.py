"""Evaluate the GEPA-optimized instruction over the full 51-row gold set, using the
same pipeline (OpenAIClient gpt-4o-mini generation, llm_judge gpt-4.1-mini) as
experiments/run_fewshot_ablation.py's a_zero_shot variant, so the two are directly
comparable apples-to-apples (both zero-shot, same judge, same full gold set — not just
the 15-row DSPy val split used during optimization itself).

Does NOT reuse run_fewshot_ablation.run_variant()'s parsing as-is: the DSPy Signature
correctly models input_topic as an InputField only (no reason to ask the model to
echo back a string the caller already has), so the GEPA-discovered instruction never
asks for input_topic in its output JSON. The production ConceptMap schema requires it
in the output regardless. That's a schema-integration gap in this eval harness, not a
flaw in the optimized instruction — so we inject the known topic before validation
instead of asking GEPA to make the model redundantly repeat it.

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

from concept_mapper.evaluator import compute_summary, evaluate_batch  # noqa: E402
from concept_mapper.io import read_concept_mapper_gold_xlsx, write_jsonl  # noqa: E402
from concept_mapper.llm_clients.openai_client import OpenAIClient  # noqa: E402
from concept_mapper.parser import extract_json_object  # noqa: E402
from concept_mapper.schemas import ConceptMap  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "gesis_concept_mapper_assertion_evaluation_adjusted.xlsx"
OUT_DIR = REPO_ROOT / "experiments" / "outputs"

_REPAIR_MESSAGE = (
    "Your previous response could not be parsed as a valid JSON object matching the required schema. "
    "Output ONLY the JSON object — no prose, no markdown fences, no explanation before or after."
)


def _parse_with_topic_injected(raw: str, topic: str) -> ConceptMap:
    json_str = extract_json_object(raw)
    data = json.loads(json_str)
    data.setdefault("input_topic", topic)
    data.setdefault("warnings", [])
    return ConceptMap(**data)


def run_variant(name: str, instruction: str, gold_rows: list[dict], client: OpenAIClient) -> list[dict]:
    records = []
    for i, row in enumerate(gold_rows, 1):
        topic = row["input_topic_parent_concept"]
        cid = row["concept_id"]
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f'Map this concept: "{topic}"'},
        ]
        raw = client.generate(messages)
        try:
            cm = _parse_with_topic_injected(raw, topic)
            record = {"concept_id": cid, "input_topic": topic, **cm.model_dump()}
        except (ValueError, ValidationError):
            repair_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _REPAIR_MESSAGE},
            ]
            raw2 = client.generate(repair_messages)
            try:
                cm = _parse_with_topic_injected(raw2, topic)
                record = {"concept_id": cid, "input_topic": topic, **cm.model_dump()}
            except (ValueError, ValidationError) as exc:
                record = {"concept_id": cid, "input_topic": topic, "error": str(exc)}
        print(f"[{name}] {i}/{len(gold_rows)} {topic!r} -> {'OK' if 'error' not in record else 'ERROR: ' + record['error'][:150]}")
        records.append(record)
    return records


def main() -> None:
    instruction_path = Path(sys.argv[1])
    instruction = instruction_path.read_text().strip()

    gold_rows = read_concept_mapper_gold_xlsx(DATA_PATH)
    generation_client = OpenAIClient(model="gpt-4o-mini", temperature=0.0, max_tokens=1200)
    judge_client = OpenAIClient(model="gpt-4.1-mini", temperature=0.0, max_tokens=300)

    records = run_variant("d_gepa_optimized", instruction, gold_rows, generation_client)
    write_jsonl(OUT_DIR / "d_gepa_optimized.jsonl", records)

    eval_records = evaluate_batch(gold_rows, records, judge_client=judge_client)
    summary = compute_summary(eval_records)
    print("\nSummary for d_gepa_optimized (full 51-row gold set):")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
