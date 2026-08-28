"""LLM-as-judge evaluation of Concept Mapper indicator quality.

evaluator.py checks CI/CP and indicator_model against gold labels exactly, but has no
way to score whether the *indicators themselves* are good — there is no single correct
indicator list, so exact/count-based matching doesn't work (see the rubric's own
"Indicator distinctiveness" and "Coverage of the construct domain" criteria, both rated
non-fully-objective). This module asks an LLM to grade those two axes against the gold
indicator list, on the same 1-5 scale the rubric uses.

Provider-neutral by design (takes a BaseLLMClient), so the same judge can score
predictions from any provider — OpenAI, Ollama, or Transformers/LRZ — using a fixed
judge model, which keeps cross-model comparisons on LRZ apples-to-apples.
"""

from __future__ import annotations

import json
from typing import Any

from survey_agent_lib.llm_clients.base import BaseLLMClient
from survey_agent_lib.parser import extract_json_object

JUDGE_SYSTEM_PROMPT = """\
You are a survey methodology expert grading the indicators produced by a Concept Mapper \
agent against a gold-standard indicator list for the same construct.

You will be given:
- The parent topic/construct.
- A gold indicator list (human-authored reference; may be terse or incomplete, not \
necessarily exhaustive).
- The agent's predicted indicators (name, definition, role for each).

Grade two axes, each on a 1-5 integer scale:

**coverage_score** — Do the predicted indicators, taken together, cover the same \
conceptual facets as the gold list? 5 = covers all gold facets and nothing clearly \
outside the construct; 3 = covers some facets, misses or adds a few; 1 = mostly \
misses the gold facets or covers a different construct entirely.

**distinctiveness_score** — Are the predicted indicators non-redundant with each other \
(each captures a genuinely different facet, not paraphrases of the same idea)? 5 = all \
indicators clearly distinct; 3 = some overlap; 1 = indicators are mostly restatements \
of each other.

Output ONLY a JSON object, no prose, no markdown:
{
  "coverage_score": integer 1-5,
  "distinctiveness_score": integer 1-5,
  "feedback": "one or two sentences explaining the scores, naming specific gaps or overlaps"
}
"""


def _format_predicted_indicators(indicators: list[dict]) -> str:
    if not indicators:
        return "(none)"
    lines = []
    for ind in indicators:
        name = ind.get("name", "?")
        definition = ind.get("definition", "")
        role = ind.get("role", "?")
        lines.append(f"- {name} ({role}): {definition}")
    return "\n".join(lines)


def build_judge_messages(
    topic: str,
    gold_indicators: str,
    predicted_indicators: list[dict],
) -> list[dict]:
    user_content = (
        f'Parent topic/construct: "{topic}"\n\n'
        f"Gold indicators (reference):\n{gold_indicators or '(none provided)'}\n\n"
        f"Predicted indicators:\n{_format_predicted_indicators(predicted_indicators)}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def judge_indicator_quality(
    client: BaseLLMClient,
    topic: str,
    gold_indicators: str,
    predicted_indicators: list[dict],
    **generate_kwargs: Any,
) -> dict[str, Any]:
    """Returns a dict with coverage_score, distinctiveness_score (1-5 each, or None on
    parse failure), a combined `score` normalised to [0, 1], `feedback`, and the raw
    LLM response for debugging.
    """
    messages = build_judge_messages(topic, gold_indicators, predicted_indicators)
    raw = client.generate(messages, **generate_kwargs)

    try:
        obj = json.loads(extract_json_object(raw))
        coverage = int(obj["coverage_score"])
        distinctiveness = int(obj["distinctiveness_score"])
        coverage = max(1, min(5, coverage))
        distinctiveness = max(1, min(5, distinctiveness))
        feedback = str(obj.get("feedback", ""))
        return {
            "coverage_score": coverage,
            "distinctiveness_score": distinctiveness,
            "score": (coverage + distinctiveness) / 10.0,
            "feedback": feedback,
            "raw": raw,
        }
    except Exception as exc:
        return {
            "coverage_score": None,
            "distinctiveness_score": None,
            "score": None,
            "feedback": f"Judge response could not be parsed: {exc}",
            "raw": raw,
        }
