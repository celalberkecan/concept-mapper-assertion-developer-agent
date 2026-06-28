# Concept Mapper Agent

First agent in the LMU seminar survey questionnaire design pipeline.

The Concept Mapper takes a **broad survey topic** as input — such as `"fear of crime"`, `"political trust"`, or `"age"` — and outputs a structured **concept map** that classifies the topic, defines the construct, and (for complex constructs) lists its key indicators.

## What it does

| Input | `"fear of crime"` |
|-------|-------------------|
| Output | CI or CP classification, indicator model (formative / reflective / mixed), construct definition, indicator list, rationale |

**Topic-only mode**: the agent receives *only* the raw topic string. No source variable labels, no gold indicators, no example questions are passed to the prompt.

**No LangChain / LangGraph**: the agent is a stateless function around a single LLM call, a strict prompt with few-shot examples, JSON parsing, and Pydantic validation. This keeps it simple, debuggable, and provider-neutral.

---

## Project structure

```
concept-mapper-agent/
├── configs/             # Per-provider YAML configs
├── data/                # Input data (Excel sheets)
├── outputs/             # Prediction outputs
├── src/concept_mapper/
│   ├── schemas.py       # Pydantic models (ConceptMap, Indicator)
│   ├── prompts.py       # System prompt + few-shot message builder
│   ├── parser.py        # Robust JSON extraction from raw LLM output
│   ├── agent.py         # ConceptMapperAgent (map_concept + retry)
│   ├── cli.py           # Typer CLI (map-one, smoke-test)
│   └── llm_clients/
│       ├── base.py
│       ├── openai_client.py       # OpenAI Responses API
│       ├── ollama_client.py       # Local Ollama HTTP API
│       ├── transformers_client.py # HuggingFace (LRZ servers)
│       └── fake_client.py         # No-API stub for tests
└── tests/
    ├── test_schema.py
    ├── test_parser.py
    └── test_agent_with_fake_client.py
```

---

## Installation (local)

```bash
cd concept-mapper-agent

# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode (includes all dependencies)
pip install -e ".[dev]"
```

---

## Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and paste your key

# Or export directly
export OPENAI_API_KEY=sk-...
```

The key is read from the environment variable `OPENAI_API_KEY`. `python-dotenv` is used; add a `.env` file at the project root and it will be loaded automatically if you integrate it (see `.env.example`).

---

## Running the agent

### Smoke test (no API key required)

Verifies the full pipeline using a local fake client:

```bash
python -m concept_mapper smoke-test
```

### Map a single concept — OpenAI

```bash
python -m concept_mapper map-one \
  --topic "fear of crime" \
  --provider openai \
  --config configs/openai.yaml
```

Add `--raw` to also print the raw LLM response. Add `--output result.json` to save.

### Map a single concept — Ollama

First make sure Ollama is running and the model is pulled:

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

Then:

```bash
python -m concept_mapper map-one \
  --topic "political trust" \
  --provider ollama \
  --config configs/ollama.yaml
```

---

## Running tests

```bash
pytest
```

Tests do not require any API keys or running services. All tests use `FakeLLMClient`.

```bash
pytest -v            # verbose output
pytest --tb=short    # shorter tracebacks
```

---

## LRZ / Transformers (LMU servers)

Install the LRZ requirements:

```bash
pip install -r requirements-lrz.txt
```

Edit `configs/transformers.yaml` and set `model_path` to the local checkpoint directory:

```yaml
model_path: /path/to/Qwen2.5-7B-Instruct
```

Then run:

```bash
python -m concept_mapper map-one \
  --topic "fear of crime" \
  --provider transformers \
  --config configs/transformers.yaml
```

`torch` and `transformers` are imported lazily inside `TransformersClient.__init__`, so the package installs and runs locally without them.

---

## Output schema

```json
{
  "input_topic": "fear of crime",
  "ci_or_cp": "CP",
  "indicator_model": "formative",
  "construct_definition": "...",
  "indicators": [
    { "name": "fear of burglary", "definition": "...", "role": "component" }
  ],
  "rationale": "...",
  "warnings": []
}
```

| Field | Values |
|-------|--------|
| `ci_or_cp` | `"CI"` or `"CP"` |
| `indicator_model` | `"NA"` (CI), `"formative"`, `"reflective"`, `"mixed"` (CP) |
| `role` | `"component"`, `"manifestation"`, `"mixed"`, `"direct"`, `"other"` |

---

## What the agent must NOT output

- Survey questions or question wording
- Response options (Likert scales, 0–10 scales, yes/no formats)
- Any measurement format

The Concept Mapper produces only **conceptual mapping**.
