"""GEPA-based prompt optimization for the Assertion Developer agent's system instruction.

Mirrors concept-mapper-agent/experiments/dspy_gepa_optimize.py's design for the sibling
agent. Optimizes against the human-authored gold data (basic_concept_key /
structure_code_gold / gold_assertion columns) in
data/gesis_concept_mapper_assertion_evaluation_adjusted_for_assertion_agent_final.xlsx,
"Source Items + Assertions (cor)" sheet, CP-parent rows only.

The seed instruction is the a_zero_shot variant from fewshot_ablation_results.md — the
winning variant of the 3-way ablation (best on both exact-match rubric criteria, lowest
error rate after message-history). GEPA mutates this instruction text directly.

The compiled result is NOT wired into the production agent automatically — inspect
optimized.signature.instructions afterwards and manually port the winning instruction
text into assertion_developer/assertion_prompts.py if it beats the hand-written
zero-shot baseline on the held-out set, then re-verify with run_gepa_eval.py against
the full 92-row gold set through the real production pipeline.

Usage:
    python experiments/dspy_gepa_optimize.py --auto light --target gpt-4o-mini --reflection gpt-5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Literal

import dspy
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

sys.path.insert(0, str(REPO_ROOT / "src"))
from assertion_developer.assertion_evaluator import _normalize_structure_code  # noqa: E402
from assertion_developer.assertion_rules import BASIC_CONCEPT_RULES  # noqa: E402
from assertion_developer.io import read_assertion_gold_xlsx  # noqa: E402
from assertion_developer.llm_judge import judge_assertion_alignment  # noqa: E402
from survey_agent_lib.llm_clients.base import BaseLLMClient  # noqa: E402


class DspyLMAsClient(BaseLLMClient):
    """Wraps a dspy.LM as a BaseLLMClient so llm_judge.py (written against the shared
    provider abstraction) can reuse a dspy.LM's automatic per-call cost tracking
    (.history) instead of duplicating it for a separate OpenAIClient instance."""

    def __init__(self, lm: dspy.LM) -> None:
        self.lm = lm

    def generate(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> str:
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        result = self.lm(messages=messages, **kwargs)
        return result[0] if isinstance(result, list) else result


DATA_PATH = (
    REPO_ROOT / "data" / "gesis_concept_mapper_assertion_evaluation_adjusted_for_assertion_agent_final.xlsx"
)

_BASIC_CONCEPTS = tuple(sorted(BASIC_CONCEPT_RULES))

# Seed instruction: the a_zero_shot winner from fewshot_ablation_results.md — i.e. the
# content of assertion_developer/assertion_prompts.py::SYSTEM_PROMPT, trimmed of the
# "Strict Output Rules" / "JSON Schema" sections since DSPy's own adapter enforces the
# output format independently of instruction text (those sections would be redundant
# and could confuse GEPA's rewriting, which edits this text directly) — same trimming
# rationale as the Concept Mapper GEPA script.
SEED_INSTRUCTION = """\
You are a survey methodology expert specialising in assertion development for \
questionnaire design, following the Saris & Gallhofer framework.

Your task: given one CI-level survey indicator (a name and role), produce one formal \
declarative assertion that captures exactly what the indicator measures.

## What is an assertion?

An assertion is a single declarative statement that expresses the content to be \
measured. It is NOT a survey question. It does NOT contain response options, Likert \
scales, or measurement formats. It describes the measurement target, not the \
measurement method.

The respondent may be the grammatical subject when appropriate, for example in \
behavior, feelings, demographics, knowledge, or action tendencies. However, for \
evaluations, norms, policies, rights, causal beliefs, and similar concepts, the \
assertion may be a proposition about an object, actor, policy, group, or state of \
affairs.

Correct:

* "The respondent fears burglary."
* "The respondent voted in the last election."
* "Religion is important in the respondent's life."
* "The government should promote gender equality."
* "Immigrants should adapt to the culture of the receiving country."
* "Democracy in the country works well."

Avoid unnecessary belief wrappers:

* Do NOT write "The respondent believes that the government should promote gender \
equality" when the selected structure is g(H+I)y.
* Instead write: "The government should promote gender equality."
* Do NOT write "The respondent believes democracy works well" when the selected \
structure is xIe.
* Instead write: "Democracy in the country works well."

## Variable types

**Subjective** — captures evaluations, importance, values, feelings, judgments, \
beliefs, preferences, norms, policy attitudes, rights, intentions, or expectations.

**Objective** — captures external or observable facts, attributes, actions, events, \
knowledge, time, place, quantities, or procedures.

## Basic concepts

**Subjective basic concepts** (14):
evaluation, importance, values, feelings, cognitive_judgment, causal_relationship, \
similarity_relationship, preference, norms, policies, rights, action_tendencies, \
expectations_future_events, evaluative_belief

**Objective basic concepts** (8):
behavior, events, demographics, knowledge, time, place, quantities, procedures

## Assertion structures

Three grammatical structures are used in this framework:

**structure_1** — Subject + link verb + subject complement
Form: X is/was Y.
Example: "The government is effective." / "Religion is important in the respondent's \
life."
Use for: evaluation, importance, values, cognitive_judgment, demographics

**structure_2** — Subject + predicator + direct object or proposition
Form: X does/feels/believes/prefers/should-do Y.
Example: "The respondent fears burglary." / "The government should promote gender \
equality."
Use for: feelings, causal_relationship, similarity_relationship, preference, norms, \
policies, rights, action_tendencies, evaluative_belief, knowledge, quantities

**structure_3** — Subject + predicator, usually without an explicit direct object
Form: X happened / X acted / X changed.
Example: "The respondent voted." / "Prices increased."
Use for: behavior, events, time, place, procedures, expectations_future_events

## Rule table (basic_concept -> allowed structure codes)

| basic_concept              | allowed codes   | default  |
| --------------------------- | --------------- | -------- |
| evaluation                 | xIe             | xIe      |
| importance                 | xIi             | xIi      |
| values                     | vIi             | vIi      |
| feelings                   | xIf, xFy, xPf, rFy | rFy   |
| cognitive_judgment         | xIc             | xIc      |
| causal_relationship        | xIca, xCy       | xCy      |
| similarity_relationship    | xIs, xSy        | xSy      |
| preference                 | xIpr, xPRy      | xPRy     |
| norms                      | o(H+I)y, o(H+I) | o(H+I)y  |
| policies                   | g(H+I)y         | g(H+I)y  |
| rights                     | xIri, xHRy      | xHRy     |
| action_tendencies          | rFDy            | rFDy     |
| expectations_future_events | xFDy, xFD       | xFD      |
| evaluative_belief          | xPey, xPye, xPe | xPey     |
| behavior                   | rDy, rD         | rD       |
| events                     | xDy, xD         | xD       |
| demographics               | xId             | xId      |
| knowledge                  | xIsc, xPy, xP   | xPy      |
| time                       | xDti            | xDti     |
| place                      | xDpl            | xDpl     |
| quantities                 | xDqu            | xDqu     |
| procedures                 | xDpl_pro        | xDpl_pro |

Choose structure_code only from the allowed codes for your selected basic_concept. \
basic_concept must belong to the selected variable_type. Do not add "The respondent \
believes..." unless the selected basic concept and structure code require the \
respondent as the grammatical subject. The assertion must not end with '?' and must \
not contain response scale or option language.
"""


class AssertionSignature(dspy.Signature):
    __doc__ = SEED_INSTRUCTION

    parent_concept: str = dspy.InputField(desc="e.g. 'fear of crime'")
    indicator_name: str = dspy.InputField(desc="e.g. 'fear of burglary'")
    indicator_role: str = dspy.InputField(desc="e.g. 'component'")

    variable_type: Literal["subjective", "objective"] = dspy.OutputField()
    basic_concept: Literal[_BASIC_CONCEPTS] = dspy.OutputField()
    domain: str = dspy.OutputField(desc="Short phrase naming the specific domain/topic of the indicator")
    structure_code: str = dspy.OutputField(desc="Must be one of the allowed codes for the chosen basic_concept")
    assertion: str = dspy.OutputField(desc="One declarative statement, not a question, no response scale language")
    rationale: str = dspy.OutputField()
    warnings: list[str] = dspy.OutputField(desc="Any caveats about ambiguity or edge cases; empty list if none.")


def load_gold_examples() -> list[dspy.Example]:
    rows = read_assertion_gold_xlsx(DATA_PATH)
    examples = []
    for row in rows:
        ex = dspy.Example(
            parent_concept=row["input_topic_parent_concept"],
            indicator_name=row["indicator_concept_gold"],
            indicator_role="component",
            gold_basic_concept=row["basic_concept_key"],
            gold_structure_code=_normalize_structure_code(row.get("structure_code_gold")),
            gold_assertion=row["gold_assertion"],
        ).with_inputs("parent_concept", "indicator_name", "indicator_role")
        examples.append(ex)
    return examples


def split_train_val(
    examples: list[dspy.Example], val_fraction: float = 0.3, seed: int = 0
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def make_metric(judge_client: BaseLLMClient | None):
    """Build the GEPA metric. Weights: 0.35 basic_concept exact match, 0.35
    structure_code exact match (falls back to "in allowed_codes" when gold code is
    missing, mirroring assertion_evaluator.evaluate_single_against_gold), 0.30
    concept-assertion alignment (LLM-judge against gold_assertion). Without a
    judge_client the 0.30 share is dropped and the other two are rescaled to sum to 1.0.
    """

    def metric(
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace=None,
        pred_name: str | None = None,
        pred_trace=None,
        program_trace=None,
    ) -> dspy.Prediction:
        feedback_parts: list[str] = []
        w_bc, w_sc, w_al = (0.35, 0.35, 0.30) if judge_client is not None else (0.5, 0.5, 0.0)
        score = 0.0

        pred_bc = getattr(pred, "basic_concept", None)
        pred_vt = getattr(pred, "variable_type", None)

        # Gate: basic_concept/variable_type must be a valid pairing per BASIC_CONCEPT_RULES.
        # This is exactly what AssertionOutput's Pydantic validator enforces in production —
        # a mismatch here means the record fails validation entirely (no assertion is ever
        # produced), so no credit should be given for structure_code or alignment either.
        # (A prior run of this script omitted this check: it let GEPA freely trade away
        # variable_type consistency for gains on the other two criteria, since nothing during
        # training penalized the combination that actually crashes the production pipeline —
        # see "Root cause" in gepa_results.md.)
        rule_for_pred_bc = BASIC_CONCEPT_RULES.get(pred_bc)
        if rule_for_pred_bc is None or pred_vt != rule_for_pred_bc["variable_type"]:
            expected_vt = rule_for_pred_bc["variable_type"] if rule_for_pred_bc else "?"
            feedback = (
                f"Invalid basic_concept/variable_type combination for indicator "
                f"{gold.indicator_name!r} (parent concept {gold.parent_concept!r}): "
                f"basic_concept={pred_bc!r} requires variable_type={expected_vt!r}, but "
                f"got variable_type={pred_vt!r}. This combination fails production schema "
                f"validation entirely (no assertion would be produced), so no credit is "
                f"given for basic_concept, structure_code, or alignment on this row."
            )
            return dspy.Prediction(score=0.0, feedback=feedback)

        if pred_bc == gold.gold_basic_concept:
            score += w_bc
        else:
            feedback_parts.append(
                f"Wrong basic_concept for indicator {gold.indicator_name!r} (parent "
                f"concept {gold.parent_concept!r}): predicted {pred_bc!r}, gold is "
                f"{gold.gold_basic_concept!r}."
            )

        pred_sc = getattr(pred, "structure_code", None)
        if gold.gold_structure_code:
            if pred_sc == gold.gold_structure_code:
                score += w_sc
            else:
                feedback_parts.append(
                    f"Wrong structure_code: predicted {pred_sc!r}, gold is "
                    f"{gold.gold_structure_code!r}."
                )
        else:
            rule = BASIC_CONCEPT_RULES.get(pred_bc)
            if rule is not None and pred_sc in rule["allowed_codes"]:
                score += w_sc
            else:
                feedback_parts.append(
                    f"structure_code {pred_sc!r} is not an allowed code for basic_concept "
                    f"{pred_bc!r} (no gold structure_code available for this row, so "
                    f"checked against the allowed-codes rule table instead): allowed are "
                    f"{rule['allowed_codes'] if rule else '?'}."
                )

        if judge_client is not None:
            pred_assertion = getattr(pred, "assertion", "") or ""
            judge_result = judge_assertion_alignment(
                judge_client, gold.parent_concept, gold.indicator_name, gold.gold_assertion, pred_assertion
            )
            if judge_result["score"] is not None:
                score += w_al * judge_result["score"]
                feedback_parts.append(
                    f"Concept-assertion alignment ({judge_result['alignment_score']}/5): "
                    f"{judge_result['feedback']}"
                )

        feedback = " ".join(feedback_parts) if feedback_parts else "Correct."
        return dspy.Prediction(score=score, feedback=feedback)

    return metric


def plain_accuracy(program: dspy.Module, examples: list[dspy.Example]) -> dict:
    n = len(examples)
    bc_correct = 0
    sc_correct = 0
    sc_total = 0
    errors = 0
    for ex in examples:
        try:
            pred = program(
                parent_concept=ex.parent_concept,
                indicator_name=ex.indicator_name,
                indicator_role=ex.indicator_role,
            )
        except Exception:
            errors += 1
            continue
        pred_bc = getattr(pred, "basic_concept", None)
        pred_vt = getattr(pred, "variable_type", None)
        rule_for_pred_bc = BASIC_CONCEPT_RULES.get(pred_bc)
        if rule_for_pred_bc is None or pred_vt != rule_for_pred_bc["variable_type"]:
            # Would fail AssertionOutput's Pydantic validator in production — count as an
            # error here too, so this DSPy-internal number stops hiding the same blind
            # spot the training metric used to have.
            errors += 1
            continue
        if pred_bc == ex.gold_basic_concept:
            bc_correct += 1
        pred_sc = getattr(pred, "structure_code", None)
        if ex.gold_structure_code:
            sc_total += 1
            if pred_sc == ex.gold_structure_code:
                sc_correct += 1
        else:
            rule = BASIC_CONCEPT_RULES.get(getattr(pred, "basic_concept", None))
            sc_total += 1
            if rule is not None and pred_sc in rule["allowed_codes"]:
                sc_correct += 1
    n_valid = n - errors
    return {
        "n": n,
        "errors": errors,
        "basic_concept_accuracy": bc_correct / n_valid if n_valid else None,
        "structure_code_accuracy": sc_correct / sc_total if sc_total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="gpt-4o-mini", help="Model being optimized (the agent's runtime model)")
    parser.add_argument("--reflection", default="gpt-5", help="Stronger model GEPA uses to propose new instructions")
    parser.add_argument(
        "--judge",
        default="gpt-4.1-mini",
        help="Model used to score concept-assertion alignment against gold_assertion. "
        "Pass --no-judge to disable and fall back to the basic_concept + structure_code-only metric.",
    )
    parser.add_argument("--no-judge", action="store_true", help="Disable the alignment judge.")
    parser.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    parser.add_argument("--max-metric-calls", type=int, default=None, help="Hard cap on LLM calls; overrides --auto for smoke tests")
    parser.add_argument("--val-fraction", type=float, default=0.3)
    args = parser.parse_args()

    examples = load_gold_examples()
    trainset, valset = split_train_val(examples, val_fraction=args.val_fraction)
    print(f"Loaded {len(examples)} gold rows -> {len(trainset)} train / {len(valset)} val")

    target_lm = dspy.LM(f"openai/{args.target}", temperature=0.0)
    reflection_lm = dspy.LM(f"openai/{args.reflection}", temperature=1.0)
    dspy.configure(lm=target_lm)

    judge_lm = None
    judge_client = None
    if not args.no_judge:
        judge_lm = dspy.LM(f"openai/{args.judge}", temperature=0.0)
        judge_client = DspyLMAsClient(judge_lm)

    baseline = dspy.Predict(AssertionSignature)
    print("\n=== Baseline (hand-written zero-shot seed prompt) on val set ===")
    print(plain_accuracy(baseline, valset))

    gepa_kwargs = dict(metric=make_metric(judge_client), reflection_lm=reflection_lm, track_stats=True)
    if args.max_metric_calls is not None:
        gepa_kwargs["max_metric_calls"] = args.max_metric_calls
    else:
        gepa_kwargs["auto"] = args.auto

    optimizer = dspy.GEPA(**gepa_kwargs)
    optimized = optimizer.compile(dspy.Predict(AssertionSignature), trainset=trainset, valset=valset)

    print("\n=== Optimized program on val set ===")
    print(plain_accuracy(optimized, valset))

    print("\n=== Optimized instruction text ===")
    print(optimized.signature.instructions)

    out_path = REPO_ROOT / "experiments" / "gepa_optimized_assertion_developer.json"
    optimized.save(str(out_path))
    print(f"\nSaved compiled program to {out_path}")

    instruction_path = REPO_ROOT / "experiments" / "gepa_optimized_instruction.txt"
    instruction_path.write_text(optimized.signature.instructions)
    print(f"Saved optimized instruction text to {instruction_path}")

    target_cost = sum(h.get("cost") or 0 for h in target_lm.history)
    reflection_cost = sum(h.get("cost") or 0 for h in reflection_lm.history)
    judge_cost = sum(h.get("cost") or 0 for h in judge_lm.history) if judge_lm is not None else 0.0
    print(f"\n=== Cost ===")
    print(f"target ({args.target}): {len(target_lm.history)} calls, ${target_cost:.4f}")
    print(f"reflection ({args.reflection}): {len(reflection_lm.history)} calls, ${reflection_cost:.4f}")
    if judge_lm is not None:
        print(f"judge ({args.judge}): {len(judge_lm.history)} calls, ${judge_cost:.4f}")
    print(f"total: ${target_cost + reflection_cost + judge_cost:.4f}")


if __name__ == "__main__":
    main()
