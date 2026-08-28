"""LLM-as-judge evaluation of Assertion Developer output quality.

assertion_evaluator.py's rule-based checks (basic_concept known, structure_code
allowed, no question mark, no scale markers) and the gold-comparison checks in
this module's sibling (basic_concept exact match, structure_code exact match)
are all fully deterministic. The one criterion in the professor's rubric that
genuinely isn't — "Concept–assertion alignment: does the assertion accurately
represent the indicator from the previous step?" (rated 4/5 objectivity, not 5/5)
— has no single correct surface form (see fewshot_ablation_results.md in the
concept-mapper-agent for the same issue with indicator lists). This module grades
that one axis against gold_assertion, on the same 1-5 scale the rubric uses.
"""

from __future__ import annotations

import json
from typing import Any

from survey_agent_lib.llm_clients.base import BaseLLMClient
from survey_agent_lib.parser import extract_json_object

JUDGE_SYSTEM_PROMPT = """\
You are a survey methodology expert grading whether a generated assertion accurately \
represents a survey indicator, against a gold-standard reference assertion.

You will be given:
- The parent concept/construct.
- The indicator being operationalized.
- A gold assertion (human-authored reference).
- A generated assertion (to be graded).

Grade on a 1-5 integer scale:

**alignment_score** — Does the generated assertion express the same measurement \
target as the gold assertion, for this indicator? 5 = same meaning, any reasonable \
paraphrase; 3 = captures the general idea but drifts in scope, target, or emphasis; \
1 = expresses a different indicator or misrepresents the construct entirely.

Do NOT penalize differences in wording, sentence structure, or which entity is the \
grammatical subject — only judge whether the measurement target is the same.

Output ONLY a JSON object, no prose, no markdown:
{
  "alignment_score": integer 1-5,
  "feedback": "one sentence explaining the score, naming the specific drift if any"
}
"""


def build_judge_messages(
    parent_concept: str,
    indicator_name: str,
    gold_assertion: str,
    predicted_assertion: str,
) -> list[dict]:
    user_content = (
        f'Parent concept: "{parent_concept}"\n'
        f'Indicator: "{indicator_name}"\n\n'
        f"Gold assertion: {gold_assertion}\n"
        f"Generated assertion: {predicted_assertion}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def judge_assertion_alignment(
    client: BaseLLMClient,
    parent_concept: str,
    indicator_name: str,
    gold_assertion: str,
    predicted_assertion: str,
    **generate_kwargs: Any,
) -> dict[str, Any]:
    """Returns {"alignment_score": 1-5 or None, "score": normalised [0,1] or None,
    "feedback": str, "raw": str}.
    """
    messages = build_judge_messages(parent_concept, indicator_name, gold_assertion, predicted_assertion)
    raw = client.generate(messages, **generate_kwargs)

    try:
        obj = json.loads(extract_json_object(raw))
        alignment = int(obj["alignment_score"])
        alignment = max(1, min(5, alignment))
        feedback = str(obj.get("feedback", ""))
        return {
            "alignment_score": alignment,
            "score": (alignment - 1) / 4.0,
            "feedback": feedback,
            "raw": raw,
        }
    except Exception as exc:
        return {
            "alignment_score": None,
            "score": None,
            "feedback": f"Judge response could not be parsed: {exc}",
            "raw": raw,
        }
