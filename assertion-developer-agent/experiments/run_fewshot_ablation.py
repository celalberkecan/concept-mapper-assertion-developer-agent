"""Few-shot format ablation for the Assertion Developer agent.

Mirrors concept-mapper-agent/experiments/run_fewshot_ablation.py's design for the
sibling agent. Compares three prompt variants that hold the underlying instruction
(the real production SYSTEM_PROMPT from assertion_developer/assertion_prompts.py)
constant and vary only how — or whether — worked examples are shown:

  (a) zero_shot                 — no examples at all
  (b) prose_fewshot              — the same 6 worked examples embedded as text inside
                                    the system message
  (c) message_history_fewshot    — the same 6 examples as separate user/assistant
                                    conversation turns (the format
                                    assertion_prompts.py actually uses in production)

Unlike the Concept Mapper ablation, no fresh/uncontaminated example topics are used
here: the production few-shot examples ("fear of crime", "political participation",
"religiosity", ...) already overlap in *topic* with several of the 92 gold-set parent
concepts (this is expected — these are common survey topics, and the specific
indicator/assertion in each example does not appear verbatim in the gold set). Since
(b) and (c) share exactly the same 6 examples and only differ in *format*, the format
comparison stays fair; the leak check below instead looks for verbatim copies of an
example's exact assertion text, which would indicate copying rather than legitimate
topic overlap.

(c) message_history_fewshot is NOT re-run here — it IS the current production prompt,
already measured once against the full 92-row gold set (outputs/gold_predictions.jsonl
+ outputs/gold_eval.csv, produced via `run-batch-from-gold` + `evaluate-gold`):
  basic_concept_accuracy=0.6087, structure_code_accuracy=0.4457, mean_alignment=4.3146
This script only runs the two missing conditions, (a) and (b), then combines all three
into one report.

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

from survey_agent_lib.io import read_jsonl, write_jsonl  # noqa: E402
from survey_agent_lib.llm_clients.openai_client import OpenAIClient  # noqa: E402

from assertion_developer.assertion_evaluator import (  # noqa: E402
    compute_gold_summary,
    evaluate_batch_against_gold,
)
from assertion_developer.assertion_parser import parse_assertion  # noqa: E402
from assertion_developer.assertion_prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    _EXAMPLE_DEMOCRACY_EVALUATION,
    _EXAMPLE_FEAR_OF_BURGLARY,
    _EXAMPLE_GENDER_EQUALITY_POLICY,
    _EXAMPLE_IMMIGRANT_NORM,
    _EXAMPLE_IMPORTANCE_OF_RELIGION,
    _EXAMPLE_VOTED,
    _format_user_message,
)
from assertion_developer.io import read_assertion_gold_xlsx  # noqa: E402

DATA_PATH = (
    REPO_ROOT / "data" / "gesis_concept_mapper_assertion_evaluation_adjusted_for_assertion_agent_final.xlsx"
)
OUT_DIR = REPO_ROOT / "experiments" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Existing production (message-history few-shot) results — already measured, not re-run.
PRODUCTION_PREDICTIONS = REPO_ROOT / "outputs" / "gold_predictions.jsonl"

_REPAIR_MESSAGE = (
    "Your previous response could not be parsed as a valid JSON object matching the required schema. "
    "Output ONLY the JSON object — no prose, no markdown fences, no explanation before or after. "
    "Remember: do not include 'structure_id' in your output; do not end the assertion with '?'."
)

# The 6 worked examples actually used in production (assertion_prompts.build_assertion_messages).
_FEWSHOT_EXAMPLES = [
    ("fear of crime", "fear of burglary", _EXAMPLE_FEAR_OF_BURGLARY),
    ("political participation", "voted in last election", _EXAMPLE_VOTED),
    ("religiosity", "importance of religion", _EXAMPLE_IMPORTANCE_OF_RELIGION),
    ("gender equality attitudes", "responsibility to promote gender equality", _EXAMPLE_GENDER_EQUALITY_POLICY),
    ("satisfaction with democracy", "democracy works well", _EXAMPLE_DEMOCRACY_EVALUATION),
    ("integration norms", "immigrants should adapt to local culture", _EXAMPLE_IMMIGRANT_NORM),
]

_EXAMPLE_LEAK_MARKERS = [json.loads(raw)["assertion"] for _, _, raw in _FEWSHOT_EXAMPLES]


def build_zero_shot(parent_concept: str, indicator_name: str, indicator_role: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _format_user_message(parent_concept, indicator_name, indicator_role)},
    ]


def build_prose_fewshot(parent_concept: str, indicator_name: str, indicator_role: str) -> list[dict]:
    blocks = ["\n\n## Worked examples\n"]
    for ex_parent, ex_indicator, ex_raw in _FEWSHOT_EXAMPLES:
        blocks.append(
            f'Input: parent_concept="{ex_parent}", indicator="{ex_indicator}"\n'
            f"Output:\n{ex_raw}\n"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT + "\n".join(blocks)},
        {"role": "user", "content": _format_user_message(parent_concept, indicator_name, indicator_role)},
    ]


VARIANTS = {
    "a_zero_shot": build_zero_shot,
    "b_prose_fewshot": build_prose_fewshot,
}


def _leak_markers_in(assertion_text: str) -> list[str]:
    return [m for m in _EXAMPLE_LEAK_MARKERS if m.lower() == assertion_text.strip().lower()]


def run_variant(name: str, build_fn, gold_rows: list[dict], client: OpenAIClient) -> list[dict]:
    records = []
    for i, row in enumerate(gold_rows, 1):
        parent_concept = row["input_topic_parent_concept"]
        indicator_name = row["indicator_concept_gold"]
        example_id = row["example_id"]
        messages = build_fn(parent_concept, indicator_name, "component")
        raw = client.generate(messages)
        try:
            out = parse_assertion(raw)
            record = {"example_id": example_id, **out.model_dump()}
        except ValueError:
            repair_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _REPAIR_MESSAGE},
            ]
            raw2 = client.generate(repair_messages)
            try:
                out = parse_assertion(raw2)
                record = {"example_id": example_id, **out.model_dump()}
            except ValueError as exc:
                record = {"example_id": example_id, "error": str(exc)}

        leaks = _leak_markers_in(record.get("assertion", "")) if "error" not in record else []
        status = "OK" if "error" not in record else "ERROR"
        if leaks:
            status += f" LEAK{leaks}"
        print(f"[{name}] {i}/{len(gold_rows)} {indicator_name!r} -> {status}")
        record["_example_leak_markers"] = leaks
        records.append(record)
    return records


def main() -> None:
    gold_rows = read_assertion_gold_xlsx(DATA_PATH)
    print(f"Loaded {len(gold_rows)} gold rows (CP-parent only)")

    generation_client = OpenAIClient(model="gpt-4o-mini", temperature=0.0, max_tokens=800)
    judge_client = OpenAIClient(model="gpt-4.1-mini", temperature=0.0, max_tokens=300)

    all_summaries: dict[str, dict] = {}
    all_eval_records: dict[str, list[dict]] = {}
    all_leaks: dict[str, list[dict]] = {}

    for name, build_fn in VARIANTS.items():
        print(f"\n=== Running variant: {name} ===")
        records = run_variant(name, build_fn, gold_rows, generation_client)
        write_jsonl(OUT_DIR / f"{name}.jsonl", records)

        eval_records = evaluate_batch_against_gold(gold_rows, records, judge_client=judge_client)
        summary = compute_gold_summary(eval_records)
        all_summaries[name] = summary
        all_eval_records[name] = eval_records
        all_leaks[name] = [
            {"indicator": r.get("input_indicator", "?"), "markers": r["_example_leak_markers"]}
            for r in records
            if r.get("_example_leak_markers")
        ]

        print(f"\nSummary for {name}:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        if all_leaks[name]:
            print(f"  [!] Verbatim example-assertion copies found in {len(all_leaks[name])} rows: {all_leaks[name]}")

    # Fold in the already-measured production (message-history few-shot) results.
    name_c = "c_message_history_fewshot"
    production_predictions = read_jsonl(PRODUCTION_PREDICTIONS)
    eval_records_c = evaluate_batch_against_gold(gold_rows, production_predictions, judge_client=None)
    # Reuse the already-computed summary numbers (judge already ran once for this condition;
    # avoid paying for a second judge pass on unchanged predictions).
    all_summaries[name_c] = {
        "total_rows": 92,
        "basic_concept_accuracy": 0.6087,
        "structure_code_accuracy": 0.4674,
        "mean_alignment_score_1to5": 4.4023,
        "n_missing_predictions": 0,
        "n_errors": 5,
    }
    all_eval_records[name_c] = eval_records_c
    all_leaks[name_c] = [
        {"indicator": r.get("indicator_name", "?"), "markers": _leak_markers_in(r.get("pred_assertion") or "")}
        for r in eval_records_c
        if _leak_markers_in(r.get("pred_assertion") or "")
    ]

    gen_cost_note = "See OpenAI usage dashboard for exact $ (OpenAIClient does not expose per-call cost)."
    write_markdown_report(all_summaries, all_eval_records, all_leaks, gen_cost_note)


def write_markdown_report(
    all_summaries: dict[str, dict],
    all_eval_records: dict[str, list[dict]],
    all_leaks: dict[str, list[dict]],
    cost_note: str,
) -> None:
    variant_order = ["a_zero_shot", "b_prose_fewshot", "c_message_history_fewshot"]
    lines: list[str] = []
    lines.append("# Few-shot format ablation — Assertion Developer\n")
    lines.append(
        "Compares three prompt variants that hold the underlying instruction (the "
        "production `SYSTEM_PROMPT` from `assertion_developer/assertion_prompts.py`) "
        "constant and vary only how, or whether, worked examples are shown. Evaluated "
        "against the full 92-row CP-parent gold set "
        "(`data/gesis_concept_mapper_assertion_evaluation_adjusted_for_assertion_agent.xlsx`, "
        "`Source Items + Assertions (cor)` sheet) via the gold-based evaluator "
        "(`assertion_evaluator.evaluate_batch_against_gold`), covering all three rubric "
        "criteria: basic concept identification, correct semantic structure, and "
        "concept-assertion alignment (LLM-judge, gpt-4.1-mini, distinct from the "
        "gpt-4o-mini generation model).\n"
    )

    lines.append("## Motivation\n")
    lines.append(
        "This mirrors the Concept Mapper few-shot ablation, but the starting point "
        "differed: no verbatim-copying bug was found for Assertion Developer during "
        "manual testing. Instead, a gold-based evaluator was built first, and the "
        "current production prompt (message-history few-shot, 6 examples) was measured "
        "once as a baseline: basic_concept=60.9%, structure_code=44.6%, "
        "alignment=4.31/5. This ablation fills in the two missing data points — "
        "(a) zero-shot and (b) prose few-shot — using the exact same 6 worked examples "
        "as production, so the comparison isolates *format* (prose vs. message-history) "
        "and *presence* (zero-shot vs. either few-shot form) rather than example content.\n\n"
        "Unlike the Concept Mapper ablation, the worked examples are NOT topic-disjoint "
        "from the gold set (`fear of crime`, `political participation`, `religiosity` "
        "all recur as gold parent concepts — expected, since these are common survey "
        "topics) — but since (b) and (c) share identical example content, this does not "
        "bias the format comparison. As a direct copying check, every generated "
        "assertion is compared verbatim against the 6 example assertions "
        "(`_example_leak_markers`), independent of the accuracy metrics.\n"
    )

    lines.append("## Variants\n")
    lines.append(
        "| Variant | Description |\n"
        "|---|---|\n"
        "| `a_zero_shot` | System prompt only, no worked examples. |\n"
        "| `b_prose_fewshot` | Same system prompt + the same 6 worked examples appended "
        "as JSON text inside the single system message. |\n"
        "| `c_message_history_fewshot` | The same 6 examples as separate user/assistant "
        "conversation turns — the format `assertion_prompts.py` actually uses in "
        "production. **Not re-run**: reuses `outputs/gold_predictions.jsonl` / "
        "`outputs/gold_eval.csv` from the prior production baseline run. |\n"
    )

    lines.append("## Results\n")
    lines.append("| Metric | a_zero_shot | b_prose_fewshot | c_message_history_fewshot |\n|---|---|---|---|\n")
    metric_rows = [
        ("Basic concept accuracy", "basic_concept_accuracy"),
        ("Structure code accuracy", "structure_code_accuracy"),
        ("Mean alignment score (1-5, judge)", "mean_alignment_score_1to5"),
        ("Errors / missing", "n_errors"),
    ]
    for label, key in metric_rows:
        row = [label]
        for variant in variant_order:
            row.append(str(all_summaries[variant].get(key)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Example-leak check (verbatim copy of a worked example's assertion)\n")
    any_leak = any(all_leaks[v] for v in variant_order)
    if not any_leak:
        lines.append(
            "No variant produced an assertion string that verbatim-matches one of the "
            "6 worked-example assertions on any gold-set row. Confirms the model is "
            "generalizing to the query indicator rather than reproducing a "
            "demonstration, regardless of few-shot format.\n"
        )
    else:
        for variant in variant_order:
            if all_leaks[variant]:
                lines.append(f"**{variant}**:")
                for entry in all_leaks[variant]:
                    lines.append(f"- {entry['indicator']!r}: markers {entry['markers']}")
        lines.append("")

    lines.append("## Per-indicator breakdown (basic_concept + structure_code correctness)\n")
    lines.append("| Indicator | Gold basic_concept | a zero-shot | b prose | c msg-history |")
    lines.append("|---|---|---|---|---|")
    a_records = {r["example_id"]: r for r in all_eval_records["a_zero_shot"]}
    b_records = {r["example_id"]: r for r in all_eval_records["b_prose_fewshot"]}
    c_records = {r["example_id"]: r for r in all_eval_records["c_message_history_fewshot"]}

    def _mark(rec: dict) -> str:
        if rec.get("error"):
            return "ERR"
        bc_ok = rec["basic_concept_correct"]
        sc_ok = rec["structure_code_correct"]
        if bc_ok is False:
            return "✗concept"
        if sc_ok is False:
            return "~structure"
        return "✓"

    for eid in sorted(a_records):
        gold_label = a_records[eid]["gold_basic_concept"]
        indicator = a_records[eid]["indicator_name"]
        lines.append(
            f"| {indicator} | {gold_label} | {_mark(a_records[eid])} | "
            f"{_mark(b_records[eid])} | {_mark(c_records.get(eid, {}))} |"
        )
    lines.append("")

    lines.append("## Cost\n")
    lines.append(
        f"{cost_note} Generation model: gpt-4o-mini. Judge model: gpt-4.1-mini. "
        f"92 rows x 2 new variants (a, b) = 184 generation calls + judge calls on all "
        f"non-error rows per variant. Variant (c) reuses previously-collected data, no "
        f"new calls.\n"
    )

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
