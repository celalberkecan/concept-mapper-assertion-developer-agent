"""Few-shot format ablation for the Concept Mapper agent.

Compares three prompt variants that hold the underlying instruction (the real
production SYSTEM_PROMPT from concept_mapper/prompts.py) constant and vary only how
— or whether — worked examples are shown:

  (a) zero_shot                 — no examples at all
  (b) prose_fewshot              — 2 worked examples embedded as text inside the system
                                    message (classic "old-school" in-prompt few-shot)
  (c) message_history_fewshot    — the same 2 examples as separate user/assistant
                                    conversation turns (the format concept_mapper/
                                    prompts.py actually uses in production)

Motivation: an earlier run showed the production agent copying a message-history
few-shot example verbatim when the query topic matched the example topic exactly
(fear of crime). This ablation asks two separable questions: (1) does few-shot help
at all against a strong zero-shot baseline, and (2) does the *format* (message-history
vs prose) change how strongly the model anchors to the examples? To keep the
comparison uncontaminated, the two worked examples used here ("number of children",
"sense of belonging to a local community") are deliberately NOT among the 51 gold
topics, so no variant can score well on any gold row by simply copying an example.

No DSPy involved — this is a static comparison against the real agent/evaluator/judge
pipeline, run once per variant over the full 51-row gold set (no train/val split
needed since nothing is being fit).

Usage:
    python experiments/run_fewshot_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from concept_mapper.evaluator import compute_summary, evaluate_batch  # noqa: E402
from concept_mapper.io import read_concept_mapper_gold_xlsx, write_jsonl  # noqa: E402
from concept_mapper.llm_clients.openai_client import OpenAIClient  # noqa: E402
from concept_mapper.parser import parse_concept_map  # noqa: E402
from concept_mapper.prompts import SYSTEM_PROMPT  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "gesis_concept_mapper_assertion_evaluation_adjusted.xlsx"
OUT_DIR = REPO_ROOT / "experiments" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_REPAIR_MESSAGE = (
    "Your previous response could not be parsed as a valid JSON object matching the required schema. "
    "Output ONLY the JSON object — no prose, no markdown fences, no explanation before or after."
)

# Two fresh worked examples, deliberately absent from the 51-row gold set.
EX_CI = {
    "input_topic": "number of children",
    "ci_or_cp": "CI",
    "indicator_model": "NA",
    "construct_definition": "The total number of children the respondent has.",
    "indicators": [],
    "rationale": "A directly countable demographic fact, fully captured by one direct question.",
    "warnings": [],
}

EX_CP = {
    "input_topic": "sense of belonging to a local community",
    "ci_or_cp": "CP",
    "indicator_model": "reflective",
    "construct_definition": "A latent feeling of connectedness and identification with one's local community.",
    "indicators": [
        {
            "name": "feeling accepted by neighbors",
            "definition": "Sense of being welcomed and accepted by people in the local area.",
            "role": "manifestation",
        },
        {
            "name": "attachment to the neighborhood",
            "definition": "Emotional attachment to the local area as a place.",
            "role": "manifestation",
        },
        {
            "name": "willingness to stay in the community",
            "definition": "Desire to keep living in the current local community.",
            "role": "manifestation",
        },
    ],
    "rationale": (
        "Sense of belonging is a latent affective disposition; these facets are correlated "
        "manifestations of one underlying feeling rather than independent components, so the "
        "model is reflective."
    ),
    "warnings": [],
}

_EXAMPLE_LEAK_MARKERS = [
    "number of children",
    "sense of belonging",
    "feeling accepted by neighbors",
    "attachment to the neighborhood",
    "willingness to stay in the community",
]


def _user_msg(topic: str) -> str:
    return f'Map this concept: "{topic}"'


def build_zero_shot(topic: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_msg(topic)},
    ]


def build_prose_fewshot(topic: str) -> list[dict]:
    examples_block = (
        "\n\n## Worked examples\n\n"
        f"Topic: \"{EX_CI['input_topic']}\"\n"
        f"{json.dumps(EX_CI, indent=2)}\n\n"
        f"Topic: \"{EX_CP['input_topic']}\"\n"
        f"{json.dumps(EX_CP, indent=2)}\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT + examples_block},
        {"role": "user", "content": _user_msg(topic)},
    ]


def build_message_history_fewshot(topic: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_msg(EX_CI["input_topic"])},
        {"role": "assistant", "content": json.dumps(EX_CI, indent=2)},
        {"role": "user", "content": _user_msg(EX_CP["input_topic"])},
        {"role": "assistant", "content": json.dumps(EX_CP, indent=2)},
        {"role": "user", "content": _user_msg(topic)},
    ]


VARIANTS = {
    "a_zero_shot": build_zero_shot,
    "b_prose_fewshot": build_prose_fewshot,
    "c_message_history_fewshot": build_message_history_fewshot,
}


def _leak_markers_in(record: dict) -> list[str]:
    blob = json.dumps(record).lower()
    return [m for m in _EXAMPLE_LEAK_MARKERS if m in blob]


def run_variant(name: str, build_fn, gold_rows: list[dict], client: OpenAIClient) -> list[dict]:
    records = []
    for i, row in enumerate(gold_rows, 1):
        topic = row["input_topic_parent_concept"]
        cid = row["concept_id"]
        messages = build_fn(topic)
        raw = client.generate(messages)
        try:
            cm = parse_concept_map(raw)
            record = {"concept_id": cid, "input_topic": topic, **cm.model_dump()}
        except ValueError:
            repair_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _REPAIR_MESSAGE},
            ]
            raw2 = client.generate(repair_messages)
            try:
                cm = parse_concept_map(raw2)
                record = {"concept_id": cid, "input_topic": topic, **cm.model_dump()}
            except ValueError as exc:
                record = {"concept_id": cid, "input_topic": topic, "error": str(exc)}

        leaks = _leak_markers_in(record) if "error" not in record else []
        status = "OK" if "error" not in record else "ERROR"
        if leaks:
            status += f" LEAK{leaks}"
        print(f"[{name}] {i}/{len(gold_rows)} {topic!r} -> {status}")
        record["_example_leak_markers"] = leaks
        records.append(record)
    return records


def main() -> None:
    gold_rows = read_concept_mapper_gold_xlsx(DATA_PATH)
    print(f"Loaded {len(gold_rows)} gold rows")

    generation_client = OpenAIClient(model="gpt-4o-mini", temperature=0.0, max_tokens=1200)
    judge_client = OpenAIClient(model="gpt-4.1-mini", temperature=0.0, max_tokens=300)

    all_summaries: dict[str, dict] = {}
    all_eval_records: dict[str, list[dict]] = {}
    all_leaks: dict[str, list[dict]] = {}

    for name, build_fn in VARIANTS.items():
        print(f"\n=== Running variant: {name} ===")
        records = run_variant(name, build_fn, gold_rows, generation_client)
        write_jsonl(OUT_DIR / f"{name}.jsonl", records)

        eval_records = evaluate_batch(gold_rows, records, judge_client=judge_client)
        summary = compute_summary(eval_records)
        all_summaries[name] = summary
        all_eval_records[name] = eval_records
        all_leaks[name] = [
            {"topic": r["input_topic"], "markers": r["_example_leak_markers"]}
            for r in records
            if r.get("_example_leak_markers")
        ]

        print(f"\nSummary for {name}:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        if all_leaks[name]:
            print(f"  [!] Example-leak markers found in {len(all_leaks[name])} rows: {all_leaks[name]}")

    gen_cost_note = "See OpenAI usage dashboard for exact $ (OpenAIClient does not expose per-call cost)."
    write_markdown_report(all_summaries, all_eval_records, all_leaks, gen_cost_note)


def write_markdown_report(
    all_summaries: dict[str, dict],
    all_eval_records: dict[str, list[dict]],
    all_leaks: dict[str, list[dict]],
    cost_note: str,
) -> None:
    lines: list[str] = []
    lines.append("# Few-shot format ablation — Concept Mapper\n")
    lines.append(
        "Compares three prompt variants that hold the underlying instruction "
        "(the production `SYSTEM_PROMPT` from `concept_mapper/prompts.py`) constant "
        "and vary only how, or whether, worked examples are shown. Run once per "
        "variant over the full 51-row human-relabeled gold set "
        "(`data/gesis_concept_mapper_assertion_evaluation_adjusted.xlsx`, "
        "`Concept Mapper Gold` sheet, `*_leo` columns) — no train/val split, since "
        "nothing is being fit here (that's what the DSPy/GEPA run afterwards is for).\n"
    )

    lines.append("## Motivation\n")
    lines.append(
        "An earlier manual test showed the production agent (which always includes 4 "
        "message-history few-shot examples) reproducing one of those examples "
        "**verbatim** when the query topic exactly matched an example topic "
        "(`\"fear of crime\"`) — the model copied the demonstration instead of "
        "generalizing. That raised two separable questions this ablation is designed "
        "to answer:\n\n"
        "1. Does few-shot conditioning help at all, against a strong zero-shot "
        "baseline (the same agent already reached CI/CP=93.3%, indicator_model=53.3% "
        "with zero examples in an earlier DSPy run)?\n"
        "2. Does the *format* of the examples change how strongly the model anchors "
        "to them — i.e. is message-history few-shot (separate conversation turns, the "
        "format actually used in production) more prone to copying than the same "
        "content shown as prose inside a single system message?\n\n"
        "To keep the comparison uncontaminated, the two worked examples used below "
        "(`\"number of children\"` — CI, `\"sense of belonging to a local "
        "community\"` — CP) are **deliberately absent from the 51-row gold set**, so "
        "no variant can score well on any gold row by simply copying an example. Every "
        "generated record is also scanned for literal substrings from the two example "
        "topics/indicators (`_example_leak_markers`) as a direct copying check, "
        "independent of the accuracy metrics.\n"
    )

    lines.append("## Variants\n")
    lines.append(
        "| Variant | Description |\n"
        "|---|---|\n"
        "| `a_zero_shot` | System prompt only, no worked examples. |\n"
        "| `b_prose_fewshot` | Same system prompt + 2 worked examples appended as JSON "
        "text inside the single system message (\"classic\" in-prompt few-shot). |\n"
        "| `c_message_history_fewshot` | Same 2 examples as separate user/assistant "
        "conversation turns before the real query — the format `prompts.py` actually "
        "uses in production. |\n"
    )

    lines.append("## Results\n")
    lines.append(
        "| Metric | a_zero_shot | b_prose_fewshot | c_message_history_fewshot |\n"
        "|---|---|---|---|\n"
    )
    metric_rows = [
        ("CI/CP accuracy", "ci_cp_accuracy"),
        ("Indicator model accuracy (CP only)", "indicator_model_accuracy_cp_only"),
        ("Mean |indicator count diff| (CP only)", "mean_indicator_count_abs_diff_cp_only"),
        ("Mean indicator coverage 1-5 (CP only, judge)", "mean_indicator_coverage_1to5_cp_only"),
        ("Mean indicator distinctiveness 1-5 (CP only, judge)", "mean_indicator_distinctiveness_1to5_cp_only"),
        ("Errors / parse failures", "n_errors"),
    ]
    for label, key in metric_rows:
        row = [label]
        for variant in VARIANTS:
            row.append(str(all_summaries[variant].get(key)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Example-leak check (copying a worked example verbatim)\n")
    any_leak = any(all_leaks[v] for v in VARIANTS)
    if not any_leak:
        lines.append(
            "No variant produced output containing literal substrings from the two "
            "worked examples (`number of children`, `sense of belonging to a local "
            "community`, or their indicator names) on any gold-set topic. This is "
            "expected since the example topics don't overlap with the gold set, but it "
            "confirms the model did not bleed example content into unrelated topics "
            "regardless of few-shot format.\n"
        )
    else:
        for variant in VARIANTS:
            if all_leaks[variant]:
                lines.append(f"**{variant}**:")
                for entry in all_leaks[variant]:
                    lines.append(f"- {entry['topic']!r}: markers {entry['markers']}")
        lines.append("")

    lines.append("## Per-topic breakdown (CI/CP + indicator_model correctness)\n")
    lines.append("| Topic | Gold | a zero-shot | b prose | c msg-history |")
    lines.append("|---|---|---|---|---|")
    zero_records = {r["input_topic"]: r for r in all_eval_records["a_zero_shot"]}
    prose_records = {r["input_topic"]: r for r in all_eval_records["b_prose_fewshot"]}
    msg_records = {r["input_topic"]: r for r in all_eval_records["c_message_history_fewshot"]}
    for topic in sorted(zero_records):
        gold_label = zero_records[topic]["gold_ci_cp"]

        def _mark(rec: dict) -> str:
            if rec.get("error"):
                return "ERR"
            ci_ok = rec["ci_cp_correct"]
            im_ok = rec["indicator_model_correct"]
            if ci_ok is False:
                return "✗ci/cp"
            if im_ok is False:
                return "~model"
            return "✓"

        lines.append(
            f"| {topic} | {gold_label} | {_mark(zero_records[topic])} | "
            f"{_mark(prose_records[topic])} | {_mark(msg_records[topic])} |"
        )
    lines.append("")

    lines.append("## Cost\n")
    lines.append(f"{cost_note} Generation model: gpt-4o-mini. Judge model: gpt-4.1-mini. "
                  f"51 topics x 3 variants = 153 generation calls + judge calls on CP rows only.\n")

    lines.append("## Decision\n")
    lines.append(
        "_Fill in after reading the table above: which variant (if any) was carried "
        "forward as the DSPy/GEPA seed, and why._\n"
    )

    out_path = REPO_ROOT / "experiments" / "fewshot_ablation_results.md"
    out_path.write_text("\n".join(lines))
    print(f"\nWrote report to {out_path}")


if __name__ == "__main__":
    main()
