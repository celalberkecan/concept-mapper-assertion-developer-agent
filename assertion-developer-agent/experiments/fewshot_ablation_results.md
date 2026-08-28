# Few-shot format ablation — Assertion Developer

Compares three prompt variants that hold the underlying instruction (the production `SYSTEM_PROMPT` from `assertion_developer/assertion_prompts.py`) constant and vary only how, or whether, worked examples are shown. Evaluated against the full 92-row CP-parent gold set (`data/gesis_concept_mapper_assertion_evaluation_adjusted_for_assertion_agent_final.xlsx`, `Source Items + Assertions (cor)` sheet — Leo's final gold revision) via the gold-based evaluator (`assertion_evaluator.evaluate_batch_against_gold`), covering all three rubric criteria: basic concept identification, correct semantic structure, and concept-assertion alignment (LLM-judge, gpt-4.1-mini, distinct from the gpt-4o-mini generation model).

**Note on data currency**: all three variants below were run (or re-run) *after* a rule-table correction discovered during the GEPA phase (see [`gepa_results.md`](gepa_results.md)) — `assertion_rules.py`'s `values` entry used an incorrect structure code (`xIv`) that turned out to not exist in the source material at all; the correct code, confirmed against the professor's original table, is `vIi`. All numbers here reflect the corrected rule table.

## Motivation

This mirrors the Concept Mapper few-shot ablation, but the starting point differed: no verbatim-copying bug was found for Assertion Developer during manual testing. Instead, a gold-based evaluator was built first, and the current production prompt (message-history few-shot, 6 examples) was measured as a baseline. This ablation fills in the two missing data points — (a) zero-shot and (b) prose few-shot — using the exact same 6 worked examples as production, so the comparison isolates *format* (prose vs. message-history) and *presence* (zero-shot vs. either few-shot form) rather than example content.

Unlike the Concept Mapper ablation, the worked examples are NOT topic-disjoint from the gold set (`fear of crime`, `political participation`, `religiosity` all recur as gold parent concepts — expected, since these are common survey topics) — but since (b) and (c) share identical example content, this does not bias the format comparison. As a direct copying check, every generated assertion is compared verbatim against the 6 example assertions (`_example_leak_markers`), independent of the accuracy metrics.

## Variants

| Variant | Description |
|---|---|
| `a_zero_shot` | System prompt only, no worked examples. |
| `b_prose_fewshot` | Same system prompt + the same 6 worked examples appended as JSON text inside the single system message. |
| `c_message_history_fewshot` | The same 6 examples as separate user/assistant conversation turns — the format `assertion_prompts.py` actually uses in production. Re-run via `run-batch-from-gold` + `evaluate-gold` after the `vIi` rule-table fix (`outputs/gold_predictions.jsonl` / `outputs/gold_eval.csv`). |

## Results

| Metric | a_zero_shot | b_prose_fewshot | c_message_history_fewshot |
|---|---|---|---|

| Basic concept accuracy | 0.6522 | 0.6087 | 0.6087 |
| Structure code accuracy | 0.4891 | 0.4565 | 0.4674 |
| Mean alignment score (1-5, judge) | 4.3933 | 4.5119 | 4.4023 |
| Errors / missing | 3 | 8 | 5 |

## Example-leak check (verbatim copy of a worked example's assertion)

**b_prose_fewshot**:
- 'fear of burglary': markers ['The respondent fears that their home may be burglarized.']
- 'responsibility to promote gender equality': markers ['The government should promote gender equality.']

## Per-indicator breakdown (basic_concept + structure_code correctness)

| Indicator | Gold basic_concept | a zero-shot | b prose | c msg-history |
|---|---|---|---|---|
| alarm about environmental disaster | feelings | ✗concept | ✓ | ✓ |
| fear of terrorism | feelings | ✓ | ✓ | ✓ |
| fear of burglary | feelings | ✓ | ✓ | ✓ |
| fear of fraud | feelings | ✓ | ✓ | ✓ |
| fear of sexual harassment | feelings | ✓ | ✓ | ✓ |
| responsibility to promote gender equality | norms | ✗concept | ✗concept | ✗concept |
| gender equality as a development goal | policies | ✓ | ✓ | ✓ |
| gender equality as a country characteristic | cognitive_judgment | ✗concept | ✗concept | ERR |
| direction of immigration policy | policies | ✓ | ✓ | ✓ |
| immigration as good for the national labour market | evaluative_belief | ✓ | ✓ | ✓ |
| immigration as good for filling labour shortages | evaluative_belief | ✓ | ✓ | ✓ |
| immigration as good for the economy | evaluative_belief | ✓ | ✓ | ✓ |
| primary territorial attachment | cognitive_judgment | ✗concept | ✗concept | ✗concept |
| attachment to the nation as a value | values | ✓ | ✓ | ✓ |
| national rather than regional identity | cognitive_judgment | ✗concept | ✗concept | ✗concept |
| participation in society as part of national identity | cognitive_judgment | ✗concept | ✗concept | ✗concept |
| trust in political parties | evaluation | ✓ | ✗concept | ✗concept |
| trust in political institutions overall | evaluation | ✓ | ✗concept | ✗concept |
| trust in political parties to keep their promises | evaluation | ✗concept | ✗concept | ✗concept |
| trust in the political system | evaluation | ✓ | ✗concept | ✗concept |
| religion as a personal value | values | ✓ | ✓ | ✓ |
| importance of religion in daily life | importance | ✓ | ✓ | ✓ |
| self-description as religious or non-religious | cognitive_judgment | ✗concept | ERR | ✗concept |
| self-description as agnostic or atheist | cognitive_judgment | ✗concept | ✗concept | ERR |
| trust in people in general | evaluation | ✓ | ✗concept | ✗concept |
| trust in the social security system | evaluation | ✓ | ✗concept | ✗concept |
| trust in social media | evaluation | ✓ | ✗concept | ✗concept |
| trust that most people are fair | evaluation | ✗concept | ✗concept | ✗concept |
| democracy as an ideal | values | ✓ | ✓ | ✓ |
| external support for democratisation | policies | ✓ | ✓ | ✓ |
| use of social media as a medium | behavior | ~structure | ~structure | ~structure |
| use of the internet as a news medium | behavior | ~structure | ~structure | ~structure |
| use of television as a medium | behavior | ~structure | ~structure | ~structure |
| use of radio as a medium | behavior | ~structure | ~structure | ~structure |
| participation in politics | behavior | ✓ | ✓ | ✓ |
| engagement in political consumerism | behavior | ✗concept | ✓ | ✓ |
| intention to participate politically in society | action_tendencies | ✓ | ✓ | ✓ |
| participation by signing petitions | behavior | ✓ | ✓ | ✓ |
| intention to join a protest demonstration | action_tendencies | ✓ | ✓ | ✓ |
| past participation in a protest or demonstration | behavior | ✓ | ✓ | ✓ |
| violent protest as politically effective | evaluative_belief | ✓ | ✓ | ✓ |
| illegal protest as politically effective | evaluative_belief | ✓ | ✓ | ✓ |
| non-use of social media | behavior | ~structure | ~structure | ~structure |
| non-use of social media for political purposes | behavior | ✗concept | ~structure | ~structure |
| use of Netlog | behavior | ~structure | ~structure | ~structure |
| use of Google+ | behavior | ~structure | ~structure | ~structure |
| volunteering in sport | importance | ✗concept | ✗concept | ✗concept |
| willingness to volunteer for developing countries | action_tendencies | ✓ | ✓ | ✓ |
| volunteering in areas other than those listed | cognitive_judgment | ✗concept | ✗concept | ✗concept |
| volunteering mattering in no area | cognitive_judgment | ✗concept | ✗concept | ✗concept |
| knowledge of what the European Union is | knowledge | ~structure | ~structure | ~structure |
| knowledge of EU enlargement | knowledge | ~structure | ~structure | ~structure |
| knowledge of the number of EU member states | knowledge | ~structure | ~structure | ~structure |
| knowledge of the rotating EU presidency | knowledge | ~structure | ~structure | ~structure |
| knowledge of EU customs health and environmental enforcement | knowledge | ~structure | ~structure | ~structure |
| self-assessed knowledge of health topics | knowledge | ERR | ERR | ERR |
| knowledge of Al Gore's health care position | knowledge | ERR | ~structure | ~structure |
| knowledge of Bill Bradley's health care position | knowledge | ERR | ~structure | ~structure |
| knowledge of the European political groups | knowledge | ~structure | ~structure | ~structure |
| time taken for the political knowledge quiz | time | ✓ | ✓ | ✓ |
| knowledge of a current political fact | knowledge | ~structure | ~structure | ✓ |
| knowledge of the current European political groups | knowledge | ~structure | ~structure | ~structure |
| expected national economic situation | expectations_future_events | ✓ | ✓ | ✓ |
| expected economic situation over the coming year | expectations_future_events | ✓ | ✓ | ✓ |
| expected world economic situation | expectations_future_events | ✓ | ✓ | ✓ |
| expected EU economic situation | expectations_future_events | ✓ | ✓ | ✓ |
| outlook on future occupational success | evaluative_belief | ✗concept | ✗concept | ✗concept |
| outlook on the future in general | expectations_future_events | ✓ | ✓ | ✓ |
| expected level of future income | evaluative_belief | ✗concept | ✗concept | ✗concept |
| expected satisfaction with future income | evaluative_belief | ✓ | ✗concept | ✗concept |
| a minimum income for the future of Europe | importance | ✗concept | ✗concept | ✗concept |
| expected share of household income spent on care for parents | quantities | ✗concept | ✓ | ✗concept |
| job security when choosing a job | importance | ✓ | ✓ | ✓ |
| outlook on own job security | evaluative_belief | ✓ | ✓ | ✗concept |
| job security as a quality of a job | importance | ✗concept | ✓ | ✗concept |
| job security as a source of motivation | importance | ✗concept | ✗concept | ✗concept |
| civil rights protection in 1977 | rights | ✗concept | ✗concept | ✗concept |
| civil rights protection in 1978 | rights | ✗concept | ✗concept | ✗concept |
| satisfaction with civil rights and liberties | evaluation | ✓ | ✓ | ✓ |
| civil rights protection in 1974 | rights | ✗concept | ✗concept | ✗concept |
| actor position on the restrictiveness of immigration policy | policies | ✓ | ✗concept | ✗concept |
| restrictiveness of immigration policy | policies | ✗concept | ERR | ERR |
| actor support for restrictive immigration policy | policies | ✓ | ✓ | ✓ |
| governance level of immigration policy | policies | ✗concept | ERR | ✗concept |
| economic policy preference | preference | ~structure | ERR | ✓ |
| foreign policy preference | preference | ~structure | ERR | ✓ |
| defence policy preference | preference | ~structure | ERR | ✓ |
| financial crisis policy preference | preference | ✗concept | ERR | ERR |
| salience of welfare policy | importance | ✓ | ✓ | ✓ |
| governance level of social welfare policy | policies | ✗concept | ✗concept | ✗concept |
| governance level of health and welfare policy | policies | ✗concept | ✗concept | ✗concept |
| responsibility for health policy | policies | ✓ | ✓ | ✓ |

## Error detail

All `values`-labeled rows (`attachment to the nation as a value`, `religion as a personal value`, `democracy as an ideal`) now score `✓` across all three variants — confirming the `vIi` fix resolved cleanly for the hand-written prompt as well, not just the GEPA-optimized one. Remaining errors (3/92 zero-shot, 8/92 prose, 5/92 message-history) are all the previously-documented failure modes: hallucinated basic_concept labels not in the 22-item table (`attributes`, `preferences` — prose few-shot's dominant error source) and `variable_type`/`basic_concept` pairing mistakes (`demographics`, `knowledge`, `policies` emitted with the wrong subjective/objective pairing — spread across all three variants). Prose few-shot again has the highest error count of the three, consistent with every prior run of this ablation.

## Cost

See OpenAI usage dashboard for exact $ (OpenAIClient does not expose per-call cost). Generation model: gpt-4o-mini. Judge model: gpt-4.1-mini. 92 rows x 3 variants (a, b, c all re-run after the rule-table fix) = 276 generation calls + judge calls on all non-error rows per variant.

## Decision

**Carry forward `a_zero_shot` as the DSPy/GEPA seed** — confirmed again after the rule-table fix; the ranking is unchanged and, if anything, more decisive now that the `vIi` issue no longer muddies the comparison.

- Best of the three on both 5/5-objectivity rubric criteria: basic_concept_accuracy=0.6522 (highest) and structure_code_accuracy=0.4891 (highest).
- Lowest error rate (3/92 = 3.3%), versus prose (8/92 = 8.7%) and message-history (5/92 = 5.4%).
- Its only weakness is alignment (4.39 vs. 4.51 prose / 4.40 message-history) — the 4/5-objectivity, judge-scored criterion, where a ~0.1-point gap on a 1-5 scale doesn't outweigh a 3-4pp gap on two exact-match criteria.
- This is exactly the `a_zero_shot` seed that was carried into the GEPA optimization documented in [`gepa_results.md`](gepa_results.md) — no change needed there as a result of this re-run.
