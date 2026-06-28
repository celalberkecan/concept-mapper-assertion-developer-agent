"""
Streamlit UI for the Survey Questionnaire Design Pipeline.

Run from the assertion-developer-agent venv (it has both packages installed):

    cd nlp-css-group1
    source assertion-developer-agent/.venv/bin/activate
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load .env from assertion-developer-agent/ or root, whichever exists
for _env in [
    Path(__file__).parent / "assertion-developer-agent" / ".env",
    Path(__file__).parent / ".env",
]:
    if _env.exists():
        load_dotenv(_env, override=True)
        break

# ──────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Survey Pipeline · LMU NLP for CSS",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .badge {
        display: inline-block;
        padding: 3px 11px;
        border-radius: 12px;
        font-size: 0.80em;
        font-weight: 600;
        margin-right: 4px;
        line-height: 1.6;
    }
    .assertion-box {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        font-size: 1.08em;
        color: #14532d;
        font-style: italic;
        margin: 8px 0 6px;
    }
    .indicator-def {
        color: #94a3b8;
        font-size: 0.85em;
        margin-left: 1.5em;
    }
    .section-divider {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 18px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Badge helper
# ──────────────────────────────────────────────────────────────────────────────

def _badge(text: str, bg: str, fg: str = "#fff") -> str:
    return (
        f'<span class="badge" style="background:{bg};color:{fg}">{text}</span>'
    )


_CI_COLOR      = "#0ea5e9"
_CP_COLOR      = "#a855f7"
_SUBJ_COLOR    = "#f59e0b"
_OBJ_COLOR     = "#3b82f6"
_S1_COLOR      = "#10b981"
_S2_COLOR      = "#6366f1"
_S3_COLOR      = "#ec4899"
_ROLE_COLOR    = "#475569"
_MODEL_COLOR   = "#7c3aed"
_CONCEPT_COLOR = "#0f172a"
_CONCEPT_FG    = "#e2e8f0"

_STRUCTURE_COLORS = {
    "structure_1": _S1_COLOR,
    "structure_2": _S2_COLOR,
    "structure_3": _S3_COLOR,
}

_INDICATOR_MODEL_LABELS = {
    "formative": "formative",
    "reflective": "reflective",
    "mixed": "mixed",
    "NA": "—",
}


# ──────────────────────────────────────────────────────────────────────────────
# LLM client factory
# ──────────────────────────────────────────────────────────────────────────────

def _build_client(provider: str, model: str, api_key: str | None):
    if provider == "openai":
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        from survey_agent_lib.llm_clients.openai_client import OpenAIClient
        return OpenAIClient(model=model, temperature=0.0, max_tokens=800)
    if provider == "ollama":
        from survey_agent_lib.llm_clients.ollama_client import OllamaClient
        return OllamaClient(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.0,
            max_tokens=800,
        )
    if provider == "fake":
        from survey_agent_lib.llm_clients.fake_client import FakeLLMClient
        return FakeLLMClient()
    raise ValueError(f"Unknown provider: {provider!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    provider = st.selectbox(
        "LLM Provider",
        ["openai", "ollama", "fake"],
        index=0,
    )

    model = ""
    api_key: str | None = None

    if provider == "openai":
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
        key_from_env = os.environ.get("OPENAI_API_KEY", "")
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=key_from_env,
            type="password",
            help="Reads from your .env file automatically if left blank.",
        )
        api_key = api_key_input or key_from_env or None
        if not api_key:
            st.warning("No API key found. Set OPENAI_API_KEY in your .env or enter it above.")

    elif provider == "ollama":
        model = st.text_input("Model name", value="qwen2.5:7b-instruct")
        st.info("Make sure Ollama is running locally on port 11434.")

    elif provider == "fake":
        st.info(
            "**Fake provider** returns canned responses instantly — no API key needed.  \n"
            "Perfect for demos and testing the UI."
        )

    st.divider()
    st.markdown(
        "<div style='font-size:0.78em;color:#94a3b8'>"
        "LMU NLP for CSS · Seminar Project<br>"
        "Saris & Gallhofer (2007) assertion framework"
        "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("# 🔬 Survey Questionnaire Design Pipeline")
st.markdown(
    "Enter a survey topic and run the full pipeline: "
    "**Concept Mapper** classifies the construct and extracts indicators, "
    "then **Assertion Developer** produces formal declarative assertions "
    "grounded in Saris & Gallhofer (2007) linguistic structures."
)
st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# Input
# ──────────────────────────────────────────────────────────────────────────────

col_topic, col_run = st.columns([5, 1])
with col_topic:
    topic = st.text_input(
        "Topic",
        placeholder="e.g. fear of crime · political trust · immigration attitudes · age",
        label_visibility="collapsed",
    )
with col_run:
    run_clicked = st.button(
        "▶ Run",
        type="primary",
        use_container_width=True,
        disabled=not topic.strip(),
    )

# Empty state hint
if not topic.strip():
    st.markdown(
        "<div style='color:#94a3b8;font-size:0.9em;margin-top:6px'>"
        "Try: <b>fear of crime</b>, <b>political trust</b>, <b>immigration attitudes</b>, <b>age</b>"
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ──────────────────────────────────────────────────────────────────────────────

if run_clicked and topic.strip():
    topic = topic.strip()

    # ── Build client ─────────────────────────────────────────────────────────
    try:
        client = _build_client(provider, model, api_key)
    except Exception as exc:
        st.error(f"Could not initialise LLM client: {exc}")
        st.stop()

    records: list[dict] = []

    # ── Step 1: Concept Mapping ───────────────────────────────────────────────
    with st.status("**Step 1 · Concept Mapping** — calling LLM…", expanded=True) as s1_status:

        st.markdown(
            f"Topic: **{topic}** &nbsp;·&nbsp; provider: `{provider}`"
            + (f" &nbsp;·&nbsp; model: `{model}`" if model else ""),
            unsafe_allow_html=False,
        )

        try:
            from concept_mapper.agent import ConceptMapperAgent

            cm_agent = ConceptMapperAgent(client)
            concept_map = cm_agent.map_concept(topic)
        except Exception as exc:
            st.error(f"Concept Mapper failed: {exc}")
            s1_status.update(label="**Step 1 · Concept Mapping** — failed", state="error")
            st.stop()

        # ── Concept map display ───────────────────────────────────────────────
        ci_or_cp = concept_map.ci_or_cp
        type_badge  = _badge(ci_or_cp, _CI_COLOR if ci_or_cp == "CI" else _CP_COLOR)
        model_badge = (
            _badge(concept_map.indicator_model, _MODEL_COLOR)
            if ci_or_cp == "CP"
            else ""
        )

        st.markdown(
            f"**Type:** {type_badge} {model_badge}",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Construct definition:**  \n{concept_map.construct_definition}")

        if ci_or_cp == "CP":
            st.markdown(f"**Indicators &nbsp;({len(concept_map.indicators)})**")
            for i, ind in enumerate(concept_map.indicators, 1):
                role_badge = _badge(ind.role, _ROLE_COLOR)
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;**{i}.** &nbsp;{ind.name}&nbsp;&nbsp;{role_badge}"
                    f"<br><span class='indicator-def'>{ind.definition}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<span style='color:#94a3b8'>CI concept — one assertion will be developed "
                "for the topic itself.</span>",
                unsafe_allow_html=True,
            )

        if concept_map.rationale:
            with st.expander("Rationale from Concept Mapper"):
                st.write(concept_map.rationale)

        for w in concept_map.warnings:
            st.warning(w)

        n_indicators = len(concept_map.indicators) if ci_or_cp == "CP" else 1
        s1_status.update(
            label=(
                f"**Step 1 · Concept Mapping** — "
                f"{ci_or_cp} · {n_indicators} indicator{'s' if n_indicators != 1 else ''}"
            ),
            state="complete",
        )

    # ── Step 2: Assertion Development ────────────────────────────────────────
    if ci_or_cp == "CI":
        jobs = [(topic, concept_map.construct_definition, "direct", 0)]
    else:
        jobs = [
            (ind.name, ind.definition, ind.role, idx)
            for idx, ind in enumerate(concept_map.indicators)
        ]

    n_total = len(jobs)

    with st.status(
        f"**Step 2 · Assertion Development** — 0 / {n_total} complete…",
        expanded=True,
    ) as s2_status:

        from assertion_developer.assertion_agent import AssertionDeveloperAgent

        ad_agent = AssertionDeveloperAgent(client)
        progress_bar = st.progress(0)
        n_ok = 0
        n_err = 0

        for ind_name, ind_defn, ind_role, idx in jobs:
            n = idx + 1

            progress_bar.progress(
                (n - 1) / n_total,
                text=f"Calling LLM for indicator {n}/{n_total}: {ind_name}…",
            )
            s2_status.update(
                label=f"**Step 2 · Assertion Development** — {n - 1} / {n_total} complete…"
            )

            try:
                assertion = ad_agent.develop_assertion(topic, ind_name, ind_defn, ind_role)

                vt_badge  = _badge(assertion.variable_type, _SUBJ_COLOR if assertion.variable_type == "subjective" else _OBJ_COLOR)
                bc_badge  = _badge(assertion.basic_concept, _CONCEPT_COLOR, _CONCEPT_FG)
                sid_color = _STRUCTURE_COLORS.get(assertion.structure_id, "#64748b")
                sc_badge  = _badge(f"{assertion.structure_code} → {assertion.structure_id}", sid_color)

                st.markdown(
                    f"**[{n}/{n_total}]** &nbsp; {ind_name}",
                    unsafe_allow_html=False,
                )
                st.markdown(
                    f"{vt_badge} {bc_badge} {sc_badge}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="assertion-box">{assertion.assertion}</div>',
                    unsafe_allow_html=True,
                )
                if assertion.rationale:
                    st.caption(f"Rationale: {assertion.rationale}")

                for w in assertion.warnings:
                    st.warning(w)

                st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

                records.append({
                    "topic": topic,
                    "ci_or_cp": ci_or_cp,
                    "indicator_model": concept_map.indicator_model,
                    "construct_definition": concept_map.construct_definition,
                    "indicator_index": idx,
                    **assertion.model_dump(),
                })
                n_ok += 1

            except Exception as exc:
                st.error(f"**[{n}/{n_total}] {ind_name}** — {exc}")
                records.append({
                    "topic": topic,
                    "ci_or_cp": ci_or_cp,
                    "indicator_index": idx,
                    "input_indicator": ind_name,
                    "parent_concept": topic,
                    "error": str(exc),
                })
                n_err += 1

        progress_bar.progress(1.0, text="Done.")
        s2_status.update(
            label=(
                f"**Step 2 · Assertion Development** — {n_ok} assertion{'s' if n_ok != 1 else ''} generated"
                + (f", {n_err} failed" if n_err else "")
            ),
            state="complete" if n_err == 0 else "error",
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    st.divider()
    col_summary, col_download = st.columns([3, 1])

    with col_summary:
        icon = "✅" if n_err == 0 else "⚠️"
        color = "#16a34a" if n_err == 0 else "#b45309"
        st.markdown(
            f"<h3 style='color:{color};margin-bottom:8px'>{icon} Pipeline Complete</h3>",
            unsafe_allow_html=True,
        )
        type_badge_sm = _badge(ci_or_cp, _CI_COLOR if ci_or_cp == "CI" else _CP_COLOR)
        st.markdown(
            f"**Topic:** {topic!r} &nbsp; {type_badge_sm}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Assertions generated:** {n_ok}"
            + (f" &nbsp;·&nbsp; **Errors:** {n_err}" if n_err else "")
        )

    with col_download:
        jsonl_bytes = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
        safe_name = topic.replace(" ", "_")[:40]
        st.download_button(
            label="⬇ Download JSONL",
            data=jsonl_bytes,
            file_name=f"pipeline_{safe_name}.jsonl",
            mime="application/jsonl",
            use_container_width=True,
            type="secondary",
        )
