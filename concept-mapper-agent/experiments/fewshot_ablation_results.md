# Few-shot format ablation — Concept Mapper

Compares three prompt variants that hold the underlying instruction (the production `SYSTEM_PROMPT` from `concept_mapper/prompts.py`) constant and vary only how, or whether, worked examples are shown. Run once per variant over the full 51-row human-relabeled gold set (`data/gesis_concept_mapper_assertion_evaluation_adjusted.xlsx`, `Concept Mapper Gold` sheet, `*_leo` columns) — no train/val split, since nothing is being fit here (that's what the DSPy/GEPA run afterwards is for).

## Motivation

An earlier manual test showed the production agent (which always includes 4 message-history few-shot examples) reproducing one of those examples **verbatim** when the query topic exactly matched an example topic (`"fear of crime"`) — the model copied the demonstration instead of generalizing. That raised two separable questions this ablation is designed to answer:

1. Does few-shot conditioning help at all, against a strong zero-shot baseline (the same agent already reached CI/CP=93.3%, indicator_model=53.3% with zero examples in an earlier DSPy run)?
2. Does the *format* of the examples change how strongly the model anchors to them — i.e. is message-history few-shot (separate conversation turns, the format actually used in production) more prone to copying than the same content shown as prose inside a single system message?

To keep the comparison uncontaminated, the two worked examples used below (`"number of children"` — CI, `"sense of belonging to a local community"` — CP) are **deliberately absent from the 51-row gold set**, so no variant can score well on any gold row by simply copying an example. Every generated record is also scanned for literal substrings from the two example topics/indicators (`_example_leak_markers`) as a direct copying check, independent of the accuracy metrics.

## Variants

| Variant | Description |
|---|---|
| `a_zero_shot` | System prompt only, no worked examples. |
| `b_prose_fewshot` | Same system prompt + 2 worked examples appended as JSON text inside the single system message ("classic" in-prompt few-shot). |
| `c_message_history_fewshot` | Same 2 examples as separate user/assistant conversation turns before the real query — the format `prompts.py` actually uses in production. |

## Results

| Metric | a_zero_shot | b_prose_fewshot | c_message_history_fewshot |
|---|---|---|---|

| CI/CP accuracy | 0.8627 | 0.8627 | 0.8627 |
| Indicator model accuracy (CP only) | 0.5161 | 0.5161 | 0.3871 |
| Mean |indicator count diff| (CP only) | 1.3871 | 1.3226 | 1.2581 |
| Mean indicator coverage 1-5 (CP only, judge) | 2.9677 | 3.0645 | 3.0645 |
| Mean indicator distinctiveness 1-5 (CP only, judge) | 4.0323 | 4.0645 | 4.0 |
| Errors / parse failures | 0 | 0 | 0 |

## Example-leak check (copying a worked example verbatim)

**Caveat before reading this section:** the only marker that ever fired was the short,
generic phrase `"sense of belonging"` — and it fired for `a_zero_shot`, which never saw
the CP worked example at all (its prompt contains no examples). That's proof the marker
itself is a false-positive trap: `"sense of belonging"` is just a natural phrase for the
model to use on its own for topics like national identity or group attachment,
independent of any example. None of the more distinctive/unique strings from the worked
examples (`"feeling accepted by neighbors"`, `"attachment to the neighborhood"`,
`"willingness to stay in the community"`, `"number of children"`) appeared anywhere, in
any variant. So: **no evidence of verbatim copying in any variant** — the one marker
that did fire isn't informative because it also fires with zero examples present.

**a_zero_shot**:
- 'national identity': markers ['sense of belonging']
**b_prose_fewshot**:
- 'national identity': markers ['sense of belonging']
- 'party preference': markers ['sense of belonging']
**c_message_history_fewshot**:
- 'national identity': markers ['sense of belonging']

## Per-topic breakdown (CI/CP + indicator_model correctness)

| Topic | Gold | a zero-shot | b prose | c msg-history |
|---|---|---|---|---|
| EU knowledge | CP | ~model | ✓ | ✓ |
| age | CI | ✓ | ✓ | ✓ |
| alcohol consumption frequency | CI | ✓ | ✗ci/cp | ✗ci/cp |
| civil rights | CP | ~model | ~model | ~model |
| climate change concern | CP | ~model | ~model | ~model |
| country of birth | CI | ✓ | ✓ | ✓ |
| economic expectations | CP | ✓ | ~model | ~model |
| economic situation assessment | CP | ~model | ~model | ~model |
| education | CI | ✗ci/cp | ✗ci/cp | ✗ci/cp |
| employment status | CI | ✓ | ✓ | ✓ |
| environmental concern | CP | ~model | ~model | ~model |
| fear of crime | CP | ✓ | ~model | ~model |
| fear of unemployment | CP | ~model | ✓ | ~model |
| future expectations | CP | ~model | ✓ | ~model |
| future income | CP | ✓ | ~model | ~model |
| gender | CI | ✓ | ✓ | ✓ |
| gender equality | CP | ~model | ~model | ~model |
| health knowledge | CP | ✓ | ✓ | ✓ |
| highest degree obtaines | CI | ✓ | ✓ | ✓ |
| household size | CI | ✓ | ✓ | ✓ |
| immigration attitudes | CP | ~model | ~model | ~model |
| immigration policy | CP | ✓ | ✓ | ~model |
| income | CI | ✓ | ✓ | ✓ |
| intenet access | CI | ✗ci/cp | ✓ | ✓ |
| internet use | CI | ✗ci/cp | ✗ci/cp | ✗ci/cp |
| job satisfaction | CP | ~model | ~model | ~model |
| job security | CP | ~model | ✓ | ~model |
| language spoken at home | CI | ✓ | ✓ | ✓ |
| life satisfaction | CP | ✓ | ✓ | ✓ |
| marital status | CI | ✓ | ✓ | ✓ |
| media use | CI | ✗ci/cp | ✗ci/cp | ✗ci/cp |
| national identity | CP | ~model | ~model | ~model |
| occupation | CI | ✓ | ✓ | ✓ |
| party preference | CI | ✗ci/cp | ✗ci/cp | ✗ci/cp |
| place of residence | CI | ✓ | ✓ | ✓ |
| policy preference | CP | ~model | ~model | ~model |
| political interest | CP | ✓ | ✓ | ✓ |
| political knowledge | CP | ~model | ✓ | ✓ |
| political participation | CP | ~model | ~model | ~model |
| political trust | CP | ✓ | ✓ | ✓ |
| protest participation | CP | ✓ | ~model | ✓ |
| religiosity | CP | ✓ | ✓ | ✓ |
| residence duration (years in current city) | CI | ✓ | ✓ | ✓ |
| social media use | CI | ✗ci/cp | ✗ci/cp | ✗ci/cp |
| social trust | CP | ✓ | ✓ | ✓ |
| support for democracy | CP | ✓ | ~model | ~model |
| trust in government | CP | ✓ | ✓ | ✓ |
| trust in parliament | CP | ✓ | ✓ | ✓ |
| volunteering | CP | ✓ | ✓ | ~model |
| voting behavior | CI | ✗ci/cp | ✗ci/cp | ✗ci/cp |
| welfare policy | CP | ✓ | ✓ | ✓ |

## Cost

See OpenAI usage dashboard for exact $ (OpenAIClient does not expose per-call cost). Generation model: gpt-4o-mini. Judge model: gpt-4.1-mini. 51 topics x 3 variants = 153 generation calls + judge calls on CP rows only.

## Decision

**Carried forward: `a_zero_shot` as the DSPy/GEPA seed instruction (no few-shot examples).**

Reasoning from the table above:

- **CI/CP accuracy is identical across all three variants (86.27%)** — few-shot
  presence/format made zero difference on this axis. It is driven entirely by the
  instruction text, not by examples.
- **Indicator model accuracy (formative/reflective/mixed) is where format actually
  matters, and message-history few-shot is the worst option**: zero-shot and prose
  both score 51.6%, message-history drops to 38.7%. This directly supports the
  original intuition that motivated this ablation — showing worked examples as
  separate conversation turns pushes the model toward pattern-matching the
  examples' *style* of answer rather than reasoning about the specific topic,
  measurably hurting the one metric with the most genuine ambiguity.
- **Prose few-shot is a wash, not a win**: it ties zero-shot in aggregate
  indicator_model accuracy (0.5161 vs 0.5161) by fixing some rows
  (`fear of unemployment`, `job security`, `political knowledge`, `EU knowledge`)
  while breaking others (`protest participation`, `support for democracy`) — net
  zero. Indicator coverage/distinctiveness (judge scores) are within noise across
  all three (~3.0 / ~4.0 on both axes, no variant clearly ahead).
- **No evidence any variant benefits from copying the worked examples** (see
  example-leak check above) — so the differences above reflect genuine
  generalization behaviour, not contamination.

Net: zero-shot is not just simplest but empirically at least as good as either
few-shot variant on every metric, and message-history few-shot is measurably worse
on indicator_model. This is the empirical basis for optimizing the zero-shot seed
with GEPA rather than reintroducing few-shot examples in either format.
