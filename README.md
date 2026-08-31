# Survey Questionnaire Design — LLM Agent Pipeline

A monorepo of LLM-based agents for automated survey questionnaire design, developed as part of an LMU seminar on NLP for Computational Social Science.

The pipeline takes a raw survey **topic** as input and produces formal, linguistically grounded **assertions** that can serve as the basis for questionnaire items — without generating question wording or response scales.

---

## Pipeline Overview

```
Topic string  ──►  Concept Mapper Agent  ──►  Concept Map (CI or CP)
                                                       │
                                                       ▼
                                         Assertion Developer Agent
                                                       │
                                                       ▼
                                         Formal declarative assertion(s)
                                         with structure code & structure_id
```

**Stage 1 — Concept Mapper** classifies the topic as a concept-by-intuition (CI) or a concept-by-postulation (CP), defines the construct, and extracts indicators.

**Stage 2 — Assertion Developer** takes each indicator and produces one formal declarative assertion using the Saris & Gallhofer linguistic structures, validated against a rule table of 22 basic concepts.

Both stages can be run independently or chained together with a single `run-pipeline` command.

---

## Repository Layout

```
concept-mapper-assertion-developer-agent/
├── shared/                          # Shared library (survey-agent-lib)
│   └── src/survey_agent_lib/
│       ├── llm_clients/             # BaseLLMClient + provider implementations
│       │   ├── base.py
│       │   ├── openai_client.py     # OpenAI Responses API
│       │   ├── ollama_client.py     # Local Ollama HTTP API
│       │   ├── transformers_client.py  # HuggingFace (LRZ servers)
│       │   └── fake_client.py       # No-API stub for tests
│       ├── io.py                    # Shared JSONL read/write utilities
│       └── parser.py                # Brace-balanced JSON extractor
│
├── concept-mapper-agent/            # Agent 1: Concept Mapping
│   ├── configs/                     # Per-provider YAML configs
│   ├── src/concept_mapper/
│   │   ├── schemas.py               # ConceptMap, Indicator Pydantic models
│   │   ├── prompts.py               # System prompt + few-shot builder
│   │   ├── parser.py                # JSON extraction
│   │   ├── agent.py                 # ConceptMapperAgent
│   │   ├── cli.py                   # Typer CLI
│   │   └── llm_clients/             # Thin re-exports from survey-agent-lib
│   └── tests/
│
└── assertion-developer-agent/       # Agent 2: Assertion Development
    ├── configs/                     # Per-provider YAML configs
    ├── src/assertion_developer/
    │   ├── assertion_rules.py       # Rule table: 22 basic concepts → structure codes
    │   ├── assertion_schemas.py     # AssertionOutput Pydantic model (derived structure_id)
    │   ├── assertion_prompts.py     # System prompt + 6 few-shot examples
    │   ├── assertion_parser.py      # JSON extraction → AssertionOutput
    │   ├── assertion_agent.py       # AssertionDeveloperAgent (with repair retry)
    │   ├── assertion_evaluator.py   # Gold-based + rule-based evaluation
    │   └── cli.py                   # Typer CLI (develop-assertion, run-pipeline, …)
    └── tests/
```

---

## Results and Evaluation Artefacts

| File | What it holds |
|------|---------------|
| `BENCHMARK_RESULTS.md` | Consolidated results, four prompting techniques by four models, both agents. |
| `*/experiments/fewshot_ablation_results.md` | Zero-shot vs prose few-shot vs message-history few-shot. |
| `*/experiments/gepa_results.md` | DSPy/GEPA prompt optimization and its evaluation. |
| `*/experiments/outputs/*.jsonl` | Raw per-item predictions for every model and condition. |
| `*/experiments/outputs/*_summary.json` | Aggregate metrics per model. |

The authoritative gold annotation is
`assertion-developer-agent/data/gesis_concept_mapper_assertion_evaluation_adjusted_for_assertion_agent_final.xlsx`.
The `Concept Mapper Gold` sheet holds 46 topics and the `Source Items + Assertions (cor)`
sheet holds 151 GESIS source items, of which the 92 under CP parent topics form the
Assertion Developer evaluation set.

---

## Theoretical Background

### Concept Types (Stage 1)

| Term | Meaning |
|------|---------|
| **CI** — concept-by-intuition | The meaning is immediately intelligible and one direct item can measure it (e.g., *age*, *income*). One assertion is generated directly. |
| **CP** — concept-by-postulation | The topic is a construct that acquires meaning from theory and is measured through several indicators (e.g., *fear of crime* → fear of terrorism, burglary, fraud, sexual harassment). One assertion per indicator. |
| **Formative** | Indicators define the construct — removing one changes the construct. |
| **Reflective** | Indicators are manifestations of the construct — they should correlate. |

### Assertion Linguistic Structures (Saris & Gallhofer)

The Assertion Developer maps each indicator to one of three formal linguistic structures:

| Structure | Form | Example |
|-----------|------|---------|
| **structure_1** | Subject + link verb + complement | *The respondent is of a specific chronological age.* |
| **structure_2** | Subject + predicator + direct object | *The respondent fears that their home may be burglarized.* |
| **structure_3** | Subject + predicator | *The respondent voted in the last election.* |

### 22 Basic Concepts

**Subjective (14):** evaluation, importance, values, feelings, cognitive_judgment, causal_relationship, similarity_relationship, preference, norms, policies, rights, action_tendencies, expectations_future_events, evaluative_belief

**Objective (8):** behavior, events, demographics, knowledge, time, place, quantities, procedures

Each basic concept maps to a fixed `structure_id` and a set of allowed assertion structure codes. The `structure_id` is always derived programmatically from the rule table — never freely generated by the model.

---

## Quickstart

### Prerequisites

- Python 3.11+
- An OpenAI API key (for the `openai` provider)
- Ollama (optional, for local models)

### 1. Install the shared library

```bash
cd shared
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Install Concept Mapper Agent

```bash
cd ../concept-mapper-agent
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e "../../shared"   # install shared lib first
pip install -e ".[dev]"
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...
```

### 3. Install Assertion Developer Agent

```bash
cd ../assertion-developer-agent
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e "../../shared"
pip install -e "../concept-mapper-agent"   # needed for run-pipeline
pip install -e ".[dev]"
cp ../concept-mapper-agent/.env .env      # reuse existing key
```

---

## Running the Full Pipeline

Run from the `assertion-developer-agent/` directory with its `.venv` active:

```bash
cd assertion-developer-agent
source .venv/bin/activate

# Smoke test — no API key needed
python -m assertion_developer run-pipeline \
  --topic "fear of crime" \
  --provider fake

# With OpenAI
python -m assertion_developer run-pipeline \
  --topic "fear of crime" \
  --provider openai \
  --config configs/openai.yaml \
  --output outputs/pipeline_fear_of_crime.jsonl
```

### Example terminal output

```
─────────────────────────── Step 1 · Concept Mapping ───────────────────────────
  Topic : 'fear of crime'  provider=openai
  Type  : CP  (formative)
  3 indicators: fear of burglary, fear of physical assault, fear of theft

─────────────────────── Step 2 · Developing 3 Assertions ───────────────────────
  [1/3]  fear of burglary
         variable_type  subjective
         basic_concept  feelings
         structure      rFy  → structure_2
╭──────────────────────────────────────────────────────────────────────────────╮
│  The respondent fears that their home may be burglarized.                    │
╰──────────────────────────────────────────────────────────────────────────────╯
  ...

────────────────────────────── Pipeline Complete ───────────────────────────────
  Topic      : 'fear of crime'
  Assertions : 3 generated
```

> The transcript above is illustrative output, not a gold label. In the annotated
> reference data `fear of crime` is a CP with a **reflective** indicator model.
> Distinguishing formative from reflective is the least stable decision the
> Concept Mapper makes, which is reported in `BENCHMARK_RESULTS.md`.

### Pipeline output schema (JSONL)

Each line is one assertion. Fields:

```json
{
  "topic": "fear of crime",
  "ci_or_cp": "CP",
  "indicator_model": "formative",
  "indicator_index": 1,
  "parent_concept": "fear of crime",
  "input_indicator": "fear of burglary",
  "indicator_definition": "...",
  "variable_type": "subjective",
  "basic_concept": "feelings",
  "domain": "burglary / home victimization",
  "structure_code": "rFy",
  "structure_id": "structure_2",
  "assertion": "The respondent fears that their home may be burglarized.",
  "rationale": "...",
  "warnings": []
}
```

---

## Running Agents Individually

### Concept Mapper

```bash
cd concept-mapper-agent
source .venv/bin/activate

# Smoke test
python -m concept_mapper smoke-test

# Map a single concept
python -m concept_mapper map-one \
  --topic "political trust" \
  --provider openai \
  --config configs/openai.yaml
```

### Assertion Developer

```bash
cd assertion-developer-agent
source .venv/bin/activate

# Smoke test
python -m assertion_developer smoke-test

# Develop a single assertion
python -m assertion_developer develop-assertion \
  --parent-concept "fear of crime" \
  --indicator "fear of burglary" \
  --definition "Worry or fear that one's home may be broken into." \
  --role component \
  --provider openai \
  --config configs/openai.yaml

# Run assertions from an existing concept map JSONL
python -m assertion_developer run-assertions-from-concept-maps \
  --input concept-mapper-agent/outputs/results.jsonl \
  --output outputs/assertions.jsonl \
  --provider openai \
  --config configs/openai.yaml

# Evaluate assertion outputs
python -m assertion_developer evaluate-assertions \
  --input outputs/assertions.jsonl
```

---

## Supported LLM Providers

| Provider | Flag | Notes |
|----------|------|-------|
| OpenAI | `--provider openai` | Requires `OPENAI_API_KEY` in `.env` |
| Ollama | `--provider ollama` | Local inference; configure model in YAML |
| Transformers | `--provider transformers` | For LRZ GPU servers; set `model_path` in YAML |
| Fake | `--provider fake` | No API key; deterministic canned responses for tests |

---

## Running Tests

Each package has its own test suite. No API key is required — all tests use `FakeLLMClient`.

```bash
# Concept Mapper (35 tests)
cd concept-mapper-agent
source .venv/bin/activate
pytest -v

# Assertion Developer (32 tests)
cd assertion-developer-agent
source .venv/bin/activate
pytest -v
```

Total: **67 tests**, all passing.

---

## LRZ / Transformers (LMU GPU Servers)

The `transformers` provider targets the LRZ GPU servers. The reported open-model
benchmark in `BENCHMARK_RESULTS.md` was **not** run this way. All three open models
were run locally through Ollama with GGUF weights at Q4\_K\_M quantization, so that
no cross-model comparison mixes two quantization backends. Use the section below
only if you want to reproduce on LRZ instead.

Install the LRZ extras:

```bash
pip install -e ".[lrz]"
```

Edit the relevant `configs/transformers.yaml` and set `model_path` to the local checkpoint:

```yaml
model_path: /path/to/Qwen3-8B
```

`torch` and `transformers` are imported lazily, so the package installs and runs locally without them.

---

## Key Design Decisions

- **No LangChain / LangGraph**: each agent is a stateless function around a single LLM call with a structured prompt, JSON parsing, and Pydantic validation. Simple, debuggable, and provider-neutral.
- **Rule-based `structure_id`**: the model chooses a `structure_code`; the code derives and attaches `structure_id` from the rule table. The model can never emit an inconsistent structure pair.
- **Repair retry**: both agents attempt one automatic repair pass if the first LLM response cannot be parsed. A second failure raises an exception.
- **No gold-label evaluation**: the assertion evaluator uses rule-based checks only (valid concept, consistent type, no question marks, no scale markers).
- **Shared library**: `survey-agent-lib` (in `shared/`) holds all LLM client code and utilities. Both agents depend on it; the concept-mapper's client files are thin re-exports so no existing code or tests needed to change.

---

## References

Saris, W. E., & Gallhofer, I. (2004). Operationalization of social science concepts by intuition. *Quality & Quantity*, 38(3), 235-258. https://doi.org/10.1023/B:QUQU.0000031328.25370.e9

Saris, W. E., & Gallhofer, I. N. (2014). *Design, Evaluation, and Analysis of Questionnaires for Survey Research* (2nd ed.). Wiley. https://doi.org/10.1002/9780470165195
