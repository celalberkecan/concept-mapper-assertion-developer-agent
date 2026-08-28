# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This is a monorepo of three independently-installed Python packages, each with its own `.venv` and `pyproject.toml`:

- `shared/` — `survey-agent-lib`: LLM client abstractions (`llm_clients/base.py::BaseLLMClient` + OpenAI/Ollama/Transformers/Fake implementations) and shared JSON/JSONL utilities. Both agents depend on it.
- `concept-mapper-agent/` — Stage 1 agent (`concept_mapper` package).
- `assertion-developer-agent/` — Stage 2 agent (`assertion_developer` package); depends on both `survey-agent-lib` and `concept-mapper` (needed for `run-pipeline`).
- `app.py` (repo root) — Streamlit UI chaining both agents; run from the `assertion-developer-agent` venv since it has both packages installed.

Each package must be installed with `pip install -e` into its own venv before use — see Setup below. `concept-mapper-agent/src/concept_mapper/llm_clients/` are thin re-exports of the shared lib kept only so old imports don't break.

## Setup

```bash
# shared lib
cd shared && python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

# concept-mapper-agent
cd ../concept-mapper-agent && python3.11 -m venv .venv && source .venv/bin/activate
pip install -e "../../shared" && pip install -e ".[dev]"
cp .env.example .env   # set OPENAI_API_KEY

# assertion-developer-agent
cd ../assertion-developer-agent && python3.11 -m venv .venv && source .venv/bin/activate
pip install -e "../../shared" && pip install -e "../concept-mapper-agent" && pip install -e ".[dev]"
cp ../concept-mapper-agent/.env .env
```

## Commands

Run from inside the relevant package directory with its `.venv` activated.

```bash
# Tests (each package has its own suite; no API key needed — all use FakeLLMClient)
pytest -v                                  # full suite
pytest tests/test_assertion_schemas.py -v  # single file
pytest tests/test_assertion_schemas.py::test_name -v  # single test

# Smoke tests (no API key, canned FakeLLMClient responses)
python -m concept_mapper smoke-test
python -m assertion_developer smoke-test

# Single-concept / single-indicator runs (need --provider + --config for real LLM calls)
python -m concept_mapper map-one --topic "political trust" --provider openai --config configs/openai.yaml
python -m assertion_developer develop-assertion --parent-concept "fear of crime" --indicator-name "fear of burglary" --indicator-definition "..." --provider openai --config configs/openai.yaml

# Full pipeline (run from assertion-developer-agent/)
python -m assertion_developer run-pipeline --topic "fear of crime" --provider fake
python -m assertion_developer run-pipeline --topic "fear of crime" --provider openai --config configs/openai.yaml --output outputs/pipeline.jsonl

# Batch + evaluation
python -m concept_mapper run-batch --input data/gold.xlsx --provider openai --config configs/openai.yaml --output outputs/results.jsonl
python -m concept_mapper evaluate --gold data/gold.xlsx --predictions outputs/results.jsonl --output outputs/eval.csv
python -m concept_mapper show-results --results outputs/eval.csv
python -m assertion_developer evaluate-assertions --input outputs/assertions.jsonl

# Streamlit UI (from repo root, using the assertion-developer-agent venv)
source assertion-developer-agent/.venv/bin/activate && streamlit run app.py
```

There is no configured linter or type checker (no ruff/flake8/black/mypy config) — pytest is the only quality gate.

## Architecture

The pipeline: `Topic string → Concept Mapper Agent → Concept Map (CI or CP) → Assertion Developer Agent → one formal declarative assertion per indicator`.

**Both agents follow the same shape** (`agent.py` / `assertion_agent.py`): a stateless class wrapping a single LLM call — build few-shot messages → call `client.generate()` → parse/validate the JSON response into a Pydantic model. On a parse/validation failure, it retries exactly once by appending the failed response plus a repair instruction to the message list; a second failure propagates the exception. There is no LangChain/LangGraph — this retry loop is hand-rolled and identical in both agents.

**Provider-neutral plain text, not structured output.** `OpenAIClient` deliberately does not use OpenAI's structured-output/JSON mode — every provider (including `FakeLLMClient`) returns plain text, and the same brace-balanced `extract_json_object()` parser (in `shared/.../parser.py`, duplicated in `concept_mapper/parser.py`) is used to pull the first complete `{...}` object out of it regardless of provider. This is why prose or code fences around the JSON are tolerated.

**Concept Mapper (Stage 1)** classifies a topic as:
- **CI** (Concept-by-Intuition, e.g. age): `indicator_model` must be `"NA"` and `indicators` must be `[]`.
- **CP** (Concept-by-Postulation, e.g. fear of crime): `indicator_model` is one of `formative`/`reflective`/`mixed`, and there must be ≥2 indicators.

This CI/CP ↔ indicator_model ↔ indicators consistency is enforced by a Pydantic `@model_validator` on `ConceptMap` (`schemas.py`), not by the prompt alone — malformed LLM output raises `ValueError` and triggers the repair retry.

**Assertion Developer (Stage 2)** turns each indicator into one Saris & Gallhofer (2007) formal assertion. The critical design point: `assertion_rules.py::BASIC_CONCEPT_RULES` is a hand-authored table mapping each of the 22 basic concepts (14 subjective, 8 objective) to `variable_type`, `allowed_codes` (structure codes), and `structure_id` (`structure_1`/`structure_2`/`structure_3`). The LLM only ever chooses `variable_type`, `basic_concept`, and `structure_code`; `AssertionOutput`'s `@model_validator` (`assertion_schemas.py`) looks up the rule table and **overwrites** `structure_id` programmatically — the model must never be trusted to emit it. The same validator also rejects assertions that end in `?` or contain response-scale/option language (see `_SCALE_MARKERS`), keeping "assertion" (measurement target) strictly separate from "survey question" (measurement instrument).

**Evaluation is rule-based, not gold-label-based**, for the Assertion Developer (`assertion_evaluator.py`): it re-checks the same structural invariants (valid concept, type consistency, no question mark, no scale markers) against already-produced output. The Concept Mapper's evaluator (`evaluator.py`) *is* gold-based, comparing predictions to `data/*.xlsx` via `io.py::read_concept_mapper_gold_xlsx`.

**`FakeLLMClient`** (`shared/.../llm_clients/fake_client.py`) dispatches on the literal prefix of the last user message (`"Map this concept:"` vs `"Develop assertion for indicator:"`) and returns canned JSON keyed by the quoted topic/indicator name, falling back to a default if unrecognized. It backs every test in both suites and the `smoke-test` / `--provider fake` CLI paths — no network calls, deterministic.

**Transformers client** (`transformers_client.py`) imports `torch`/`transformers` lazily so the packages install and the test suite runs without those (large, GPU-oriented) dependencies present; they're only needed on LRZ GPU servers via the `lrz` extra.
