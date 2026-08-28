"""GEPA-based prompt optimization for the Concept Mapper agent's system instruction.

Optimizes against the human-relabeled gold data (concept_level_ci_cp_leo /
concept_level_indicatory_leo columns) in
data/gesis_concept_mapper_assertion_evaluation_adjusted.xlsx, "Concept Mapper Gold" sheet.

The compiled result is NOT wired into the production agent automatically — inspect
optimized.signature.instructions afterwards and manually port the winning instruction
text into concept_mapper/prompts.py if it beats the hand-written baseline on the held-out set.

Usage:
    python experiments/dspy_gepa_optimize.py --auto light --target gpt-4o-mini --reflection gpt-5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Literal

import dspy
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

import sys

sys.path.insert(0, str(REPO_ROOT / "src"))
from concept_mapper.llm_judge import judge_indicator_quality  # noqa: E402
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

DATA_PATH = REPO_ROOT / "data" / "gesis_concept_mapper_assertion_evaluation_adjusted.xlsx"

# Seed instruction: content of concept_mapper/prompts.py::SYSTEM_PROMPT, trimmed of the
# "Strict Output Rules" / "JSON Schema" sections since DSPy's own adapter enforces the
# output format independently of instruction text (those sections would be redundant
# and could confuse GEPA's rewriting, which edits this text directly).
SEED_INSTRUCTION = """\
You are a survey methodology expert specialising in conceptual mapping for questionnaire design.

Your task: given a broad survey topic, produce a structured concept map that identifies whether \
the concept is a Concept-by-Intuition (CI) or Concept-by-Postulation (CP), and — for CPs — \
specifies the indicator model and key indicators.

## Definitions

**Concept-by-Intuition (CI)**
A concept whose meaning is immediately clear and can be captured by a single survey question. \
No explicit definition or decomposition is needed.
Examples: age, gender, country of birth, income.

**Concept-by-Postulation (CP)**
A complex, abstract construct that requires an explicit definition and more than one indicator. \
Examples: fear of crime, political trust, social cohesion, burnout, loneliness.

**Indicator models (for CPs only)**
- **formative**: Indicators are components that together define and build the construct. Each \
covers a distinct facet. Removing one changes the coverage of the construct.
- **reflective**: Indicators are manifestations of an underlying latent construct. They are \
expected to correlate because they all reflect the same disposition.
- **mixed**: The CP needs both component-like and manifestation-like indicators, or the \
literature uses both types.

Rules: for CI, set indicator_model to "NA" and indicators to an empty list. For CP, provide at \
least 2 indicators, each with a name, definition, and role. Never write survey questions, \
response options, or rating scales — only conceptual mapping.
"""


class ConceptMapperSignature(dspy.Signature):
    __doc__ = SEED_INSTRUCTION

    input_topic: str = dspy.InputField(desc="A broad survey topic, e.g. 'fear of crime'")
    ci_or_cp: Literal["CI", "CP"] = dspy.OutputField()
    indicator_model: Literal["NA", "formative", "reflective", "mixed"] = dspy.OutputField()
    construct_definition: str = dspy.OutputField()
    indicators: list[dict] = dspy.OutputField(
        desc='List of {"name": str, "definition": str, "role": str}; empty list for CI topics.'
    )
    rationale: str = dspy.OutputField()
    warnings: list[str] = dspy.OutputField(
        desc="Any caveats about ambiguity or edge cases; empty list if none."
    )


def load_gold_examples() -> list[dspy.Example]:
    df = pd.read_excel(DATA_PATH, sheet_name="Concept Mapper Gold")
    df = df.dropna(subset=["concept_id"])
    df["concept_level_indicatory_leo"] = df["concept_level_indicatory_leo"].fillna("NA")

    examples = []
    for _, row in df.iterrows():
        ex = dspy.Example(
            input_topic=row["input_topic_parent_concept"],
            ci_or_cp_gold=row["concept_level_ci_cp_leo"],
            indicator_model_gold=row["concept_level_indicatory_leo"],
            gold_indicators_text=str(row.get("gold_indicators_conceptual") or ""),
        ).with_inputs("input_topic")
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
    """Build the GEPA metric. Weights: 0.4 ci_or_cp, 0.3 indicator_model, 0.3 indicator
    quality (LLM-judge coverage+distinctiveness against gold_indicators_conceptual, CP
    rows only — CI rows get the 0.3 for free iff indicators == [], no judge call needed).
    Without a judge_client the 0.3 share is dropped and the other two are rescaled to
    sum to 1.0, so the metric still works (just blind to indicator quality, as before).
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
        w_ci_cp, w_im, w_ind = (0.4, 0.3, 0.3) if judge_client is not None else (0.6, 0.4, 0.0)
        score = 0.0

        pred_ci_cp = getattr(pred, "ci_or_cp", None)
        if pred_ci_cp == gold.ci_or_cp_gold:
            score += w_ci_cp
        else:
            feedback_parts.append(
                f"Wrong CI/CP classification for topic {gold.input_topic!r}: "
                f"predicted {pred_ci_cp!r}, gold is {gold.ci_or_cp_gold!r}."
            )

        pred_im = getattr(pred, "indicator_model", None)
        if gold.ci_or_cp_gold == "CP":
            if pred_im == gold.indicator_model_gold:
                score += w_im
            else:
                feedback_parts.append(
                    f"Wrong indicator_model: predicted {pred_im!r}, gold is "
                    f"{gold.indicator_model_gold!r} (formative = components build the construct; "
                    f"reflective = manifestations of one latent disposition)."
                )
        else:
            if pred_im == "NA":
                score += w_im
            else:
                feedback_parts.append(
                    f"Topic is CI (gold), so indicator_model must be 'NA', got {pred_im!r}."
                )

        if judge_client is not None:
            pred_indicators = getattr(pred, "indicators", None) or []
            if gold.ci_or_cp_gold == "CI":
                if pred_indicators:
                    feedback_parts.append(
                        f"Topic is CI (gold), so indicators must be [], got {len(pred_indicators)} indicators."
                    )
                else:
                    score += w_ind
            else:
                judge_result = judge_indicator_quality(
                    judge_client,
                    topic=gold.input_topic,
                    gold_indicators=gold.gold_indicators_text,
                    predicted_indicators=pred_indicators,
                )
                if judge_result["score"] is not None:
                    score += w_ind * judge_result["score"]
                    feedback_parts.append(
                        f"Indicator quality (coverage={judge_result['coverage_score']}/5, "
                        f"distinctiveness={judge_result['distinctiveness_score']}/5): "
                        f"{judge_result['feedback']}"
                    )

        feedback = " ".join(feedback_parts) if feedback_parts else "Correct."
        return dspy.Prediction(score=score, feedback=feedback)

    return metric


def plain_accuracy(program: dspy.Module, examples: list[dspy.Example]) -> dict:
    n = len(examples)
    ci_cp_correct = 0
    im_correct = 0
    im_total = 0
    errors = 0
    for ex in examples:
        try:
            pred = program(input_topic=ex.input_topic)
        except Exception:
            errors += 1
            continue
        if getattr(pred, "ci_or_cp", None) == ex.ci_or_cp_gold:
            ci_cp_correct += 1
        expected_im = ex.indicator_model_gold
        pred_im = getattr(pred, "indicator_model", None)
        if ex.ci_or_cp_gold == "CP":
            im_total += 1
            if pred_im == expected_im:
                im_correct += 1
        else:
            if pred_im == "NA":
                im_correct += 1
            im_total += 1
    return {
        "n": n,
        "errors": errors,
        "ci_cp_accuracy": ci_cp_correct / n if n else None,
        "indicator_model_accuracy": im_correct / im_total if im_total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="gpt-4o-mini", help="Model being optimized (the agent's runtime model)")
    parser.add_argument("--reflection", default="gpt-5", help="Stronger model GEPA uses to propose new instructions")
    parser.add_argument(
        "--judge",
        default="gpt-4.1-mini",
        help="Model used to score indicator coverage/distinctiveness against gold_indicators_conceptual "
        "(CP rows only). Pass --no-judge to disable and fall back to the CI/CP + indicator_model-only metric.",
    )
    parser.add_argument("--no-judge", action="store_true", help="Disable the indicator-quality judge.")
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

    baseline = dspy.Predict(ConceptMapperSignature)
    print("\n=== Baseline (hand-written seed prompt) on val set ===")
    print(plain_accuracy(baseline, valset))

    gepa_kwargs = dict(metric=make_metric(judge_client), reflection_lm=reflection_lm, track_stats=True)
    if args.max_metric_calls is not None:
        gepa_kwargs["max_metric_calls"] = args.max_metric_calls
    else:
        gepa_kwargs["auto"] = args.auto

    optimizer = dspy.GEPA(**gepa_kwargs)
    optimized = optimizer.compile(dspy.Predict(ConceptMapperSignature), trainset=trainset, valset=valset)

    print("\n=== Optimized program on val set ===")
    print(plain_accuracy(optimized, valset))

    print("\n=== Optimized instruction text ===")
    print(optimized.signature.instructions)

    out_path = REPO_ROOT / "experiments" / "gepa_optimized_concept_mapper.json"
    optimized.save(str(out_path))
    print(f"\nSaved compiled program to {out_path}")

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
