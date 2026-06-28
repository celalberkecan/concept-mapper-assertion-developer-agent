"""Typer CLI for the Assertion Developer agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text

load_dotenv(find_dotenv(usecwd=True), override=True)

from survey_agent_lib.llm_clients.base import BaseLLMClient
from survey_agent_lib.io import read_jsonl, write_jsonl

from .assertion_agent import AssertionDeveloperAgent

app = typer.Typer(
    name="assertion-developer",
    help="Assertion Developer Agent — produces formal declarative assertions from survey indicators.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(config_path: Optional[Path]) -> dict:
    if config_path is None:
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _build_client(provider: str, cfg: dict) -> BaseLLMClient:
    if provider == "openai":
        from survey_agent_lib.llm_clients.openai_client import OpenAIClient

        return OpenAIClient(
            model=cfg.get("model", "gpt-4o-mini"),
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 800),
        )
    elif provider == "ollama":
        from survey_agent_lib.llm_clients.ollama_client import OllamaClient

        return OllamaClient(
            model=cfg.get("model", "qwen2.5:7b-instruct"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 800),
        )
    elif provider == "transformers":
        from survey_agent_lib.llm_clients.transformers_client import TransformersClient

        model_path = cfg.get("model_path")
        if not model_path:
            console.print("[red]Error:[/red] 'model_path' is required in the transformers config.")
            raise typer.Exit(1)
        return TransformersClient(
            model_path=model_path,
            torch_dtype=cfg.get("torch_dtype", "bfloat16"),
            device_map=cfg.get("device_map", "auto"),
            max_new_tokens=cfg.get("max_new_tokens", 800),
            temperature=cfg.get("temperature", 0.0),
        )
    elif provider == "fake":
        from survey_agent_lib.llm_clients.fake_client import FakeLLMClient

        return FakeLLMClient()
    else:
        console.print(
            f"[red]Error:[/red] Unknown provider {provider!r}. "
            "Choose one of: openai, ollama, transformers."
        )
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("develop-assertion")
def develop_assertion(
    parent_concept: str = typer.Option(..., "--parent-concept", help="The parent concept (e.g. 'fear of crime')."),
    indicator_name: str = typer.Option(..., "--indicator-name", help="The indicator name (e.g. 'fear of burglary')."),
    indicator_definition: str = typer.Option(..., "--indicator-definition", help="The indicator definition."),
    indicator_role: str = typer.Option("component", "--indicator-role", help="The indicator role (component/manifestation/direct/other)."),
    provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider: openai | ollama | transformers"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to a YAML config file."),
    show_raw: bool = typer.Option(False, "--raw", help="Also print the raw LLM response."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write the JSON result to this file."),
) -> None:
    """Develop a formal declarative assertion for a single indicator."""
    cfg = _load_config(config)
    client = _build_client(provider, cfg)
    agent = AssertionDeveloperAgent(client)

    console.print(
        f"[bold]Developing assertion[/bold]  "
        f"[cyan]{indicator_name!r}[/cyan]  [dim]→ parent: {parent_concept!r}  provider={provider}[/dim]\n"
    )

    with console.status("Calling LLM..."):
        assertion, raw_text = agent.develop_assertion_with_raw(
            parent_concept, indicator_name, indicator_definition, indicator_role
        )

    if show_raw:
        console.print(Panel(raw_text, title="Raw LLM response", border_style="dim"))

    console.print(
        Panel(
            Pretty(assertion.model_dump()),
            title="[bold]Assertion Output[/bold]",
            border_style="green",
        )
    )

    if output:
        output.write_text(json.dumps(assertion.model_dump(), indent=2))
        console.print(f"\n[green]Saved to {output}[/green]")


@app.command("smoke-test")
def smoke_test() -> None:
    """Run a self-contained pipeline test using FakeLLMClient (no API calls needed)."""
    from survey_agent_lib.llm_clients.fake_client import FakeLLMClient

    console.print("[bold]Smoke test[/bold] [dim](no API calls)[/dim]\n")

    agent = AssertionDeveloperAgent(FakeLLMClient())
    cases = [
        ("fear of crime", "fear of burglary", "Worry or fear that one's home may be broken into.", "component"),
        ("age", "age", "The respondent's chronological age in years.", "direct"),
        ("electoral participation", "voted in last election", "Whether the respondent cast a vote in the most recent election.", "component"),
    ]
    failures: list[str] = []

    for parent, name, defn, role in cases:
        try:
            a = agent.develop_assertion(parent, name, defn, role)
            console.print(
                Panel(
                    Pretty(a.model_dump()),
                    title=f"[green]{name}[/green]",
                    border_style="green",
                )
            )
        except Exception as exc:
            console.print(f"[red]FAILED[/red] {name!r}: {exc}")
            failures.append(name)

    if failures:
        console.print(f"\n[bold red]{len(failures)} test(s) failed.[/bold red]")
        raise typer.Exit(1)
    else:
        console.print("\n[bold green]All smoke tests passed.[/bold green]")


@app.command("run-assertions-from-concept-maps")
def run_assertions_from_concept_maps(
    input: Path = typer.Option(..., "--input", "-i", help="JSONL file from concept-mapper run-batch."),
    provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider: openai | ollama | transformers"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to a YAML config file."),
    output: Path = typer.Option(..., "--output", "-o", help="Output JSONL file path."),
) -> None:
    """Generate one assertion per indicator from a Concept Mapper output JSONL.

    For CI concept maps: one assertion for the parent concept itself.
    For CP concept maps: one assertion per indicator.
    """
    cfg = _load_config(config)
    client = _build_client(provider, cfg)
    agent = AssertionDeveloperAgent(client)

    concept_maps = read_jsonl(input)

    # Count total assertions to generate
    total_jobs: list[tuple[dict, str, str, str, str, int]] = []
    for cm in concept_maps:
        if cm.get("error"):
            continue
        ci_or_cp = cm.get("ci_or_cp", "")
        concept_id = cm.get("concept_id", "")
        parent = cm.get("input_topic", "")

        if ci_or_cp == "CI":
            total_jobs.append((
                cm, concept_id, parent,
                parent,
                cm.get("construct_definition", ""),
                "direct",
                0,
            ))
        elif ci_or_cp == "CP":
            for idx, ind in enumerate(cm.get("indicators", [])):
                total_jobs.append((
                    cm, concept_id, parent,
                    ind.get("name", ""),
                    ind.get("definition", ""),
                    ind.get("role", "component"),
                    idx,
                ))

    console.print(
        f"[bold]Assertion batch[/bold] — {len(total_jobs)} indicators  "
        f"[dim]provider={provider}  output={output}[/dim]\n"
    )

    records: list[dict] = []
    for i, job in enumerate(total_jobs, 1):
        cm, concept_id, parent, ind_name, ind_defn, ind_role, idx = job
        console.print(f"[{i}/{len(total_jobs)}] {concept_id}  {ind_name!r}", end=" ")

        try:
            assertion = agent.develop_assertion(parent, ind_name, ind_defn, ind_role)
            record = {
                "source_concept_id": concept_id,
                "source_ci_or_cp": cm.get("ci_or_cp"),
                "indicator_index": idx,
                **assertion.model_dump(),
            }
            console.print("[green]OK[/green]")
        except Exception as exc:
            record = {
                "source_concept_id": concept_id,
                "source_ci_or_cp": cm.get("ci_or_cp"),
                "indicator_index": idx,
                "input_indicator": ind_name,
                "parent_concept": parent,
                "error": str(exc),
            }
            console.print(f"[red]ERROR[/red] {exc}")

        records.append(record)

    write_jsonl(output, records)
    n_ok = sum(1 for r in records if "error" not in r)
    n_err = len(records) - n_ok
    console.print(f"\n[green]Done.[/green] {n_ok} OK, {n_err} errors → {output}")


@app.command("evaluate-assertions")
def evaluate_assertions(
    input: Path = typer.Option(..., "--input", "-i", help="JSONL file from run-assertions-from-concept-maps."),
) -> None:
    """Run rule-based quality checks on assertion JSONL output and print a summary."""
    from .assertion_evaluator import compute_summary, evaluate_batch

    if not input.exists():
        console.print(f"[red]File not found:[/red] {input}")
        raise typer.Exit(1)

    records = read_jsonl(input)
    eval_results = evaluate_batch(records)
    summary = compute_summary(eval_results)

    total = summary["total_records"]
    all_pass = summary["n_all_pass"]
    any_fail = summary["n_any_fail"]

    def _pct(rate: float | None) -> str:
        return f"{rate:.1%}" if rate is not None else "—"

    summary_lines = [
        f"[bold]Total assertions[/bold]       {total}",
        f"[bold]All checks pass[/bold]        [{'green' if all_pass == total else 'yellow'}]{all_pass}/{total}  ({_pct(summary['all_pass_rate'])})[/]",
        f"[bold]Any check failing[/bold]      [{'red' if any_fail else 'green'}]{any_fail}[/]",
        "",
        f"  valid_variable_type      {_pct(summary['valid_variable_type_rate'])}",
        f"  valid_basic_concept      {_pct(summary['valid_basic_concept_rate'])}",
        f"  variable_type_consistent {_pct(summary['variable_type_consistent_rate'])}",
        f"  valid_structure_code     {_pct(summary['valid_structure_code_rate'])}",
        f"  not_a_question           {_pct(summary['not_a_question_rate'])}",
        f"  no_scale_markers         {_pct(summary['no_scale_markers_rate'])}",
    ]
    console.print()
    console.print(Panel(
        "\n".join(summary_lines),
        title="[bold white] Assertion Quality Report [/bold white]",
        border_style="blue",
        padding=(0, 2),
    ))

    # Per-record table for failing records
    failing = [r for r in eval_results if r.get("all_pass") is False]
    if failing:
        table = Table(title="Failing Assertions", border_style="red", show_lines=True)
        table.add_column("Indicator", width=30)
        table.add_column("Parent", width=20)
        table.add_column("vt?", justify="center", width=5)
        table.add_column("bc?", justify="center", width=5)
        table.add_column("vt=bc?", justify="center", width=7)
        table.add_column("code?", justify="center", width=6)
        table.add_column("!?", justify="center", width=4)
        table.add_column("!scale?", justify="center", width=7)

        def _cell(val: bool | None) -> Text:
            if val is True:
                return Text("✓", style="green")
            if val is False:
                return Text("✗", style="bold red")
            return Text("—", style="dim")

        for r in failing:
            table.add_row(
                str(r.get("input_indicator", ""))[:30],
                str(r.get("parent_concept", ""))[:20],
                _cell(r.get("valid_variable_type")),
                _cell(r.get("valid_basic_concept")),
                _cell(r.get("variable_type_consistent")),
                _cell(r.get("valid_structure_code")),
                _cell(r.get("not_a_question")),
                _cell(r.get("no_scale_markers")),
            )

        console.print()
        console.print(table)
    console.print()


@app.command("run-pipeline")
def run_pipeline(
    topic: str = typer.Option(..., "--topic", "-t", help="The survey topic to process end-to-end."),
    provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider: openai | ollama | transformers"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to a YAML config file."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save pipeline results to this JSONL file."),
) -> None:
    """Full pipeline: Concept Mapper → Assertion Developer for a single topic.

    Produces one assertion per indicator (CP) or one assertion for the topic
    itself (CI), printing each result to the terminal as it completes.
    """
    from concept_mapper.agent import ConceptMapperAgent
    from rich.rule import Rule

    cfg = _load_config(config)
    client = _build_client(provider, cfg)

    # -----------------------------------------------------------------------
    # Step 1 — Concept Mapping
    # -----------------------------------------------------------------------
    console.print()
    console.print(Rule("[bold blue]Step 1 · Concept Mapping[/bold blue]", style="blue"))
    console.print(f"\n  Topic : [bold]{topic!r}[/bold]  [dim]provider={provider}[/dim]\n")

    with console.status("Calling Concept Mapper..."):
        cm_agent = ConceptMapperAgent(client)
        concept_map = cm_agent.map_concept(topic)

    ci_or_cp = concept_map.ci_or_cp
    type_color = "cyan" if ci_or_cp == "CI" else "magenta"
    model_str = (
        f"  [dim]({concept_map.indicator_model})[/dim]"
        if ci_or_cp == "CP"
        else "  [dim](no decomposition needed)[/dim]"
    )

    console.print(f"  Type  : [{type_color}]{ci_or_cp}[/{type_color}]{model_str}")
    console.print(f"  Def.  : [dim]{concept_map.construct_definition[:120]}[/dim]")

    if ci_or_cp == "CP":
        console.print(f"\n  [bold]{len(concept_map.indicators)} indicators:[/bold]")
        for i, ind in enumerate(concept_map.indicators, 1):
            console.print(f"    {i}. [cyan]{ind.name}[/cyan]  [dim][{ind.role}][/dim]")
        n_assertions = len(concept_map.indicators)
    else:
        console.print(f"\n  [dim]→ 1 assertion will be developed for the concept itself.[/dim]")
        n_assertions = 1

    if concept_map.warnings:
        for w in concept_map.warnings:
            console.print(f"  [yellow]⚠ {w}[/yellow]")

    # -----------------------------------------------------------------------
    # Step 2 — Assertion Development
    # -----------------------------------------------------------------------
    console.print()
    console.print(Rule(
        f"[bold blue]Step 2 · Developing {n_assertions} Assertion{'s' if n_assertions != 1 else ''}[/bold blue]",
        style="blue",
    ))
    console.print()

    ad_agent = AssertionDeveloperAgent(client)

    # Build the list of (indicator_name, indicator_definition, indicator_role, index)
    if ci_or_cp == "CI":
        jobs = [(topic, concept_map.construct_definition, "direct", 0)]
    else:
        jobs = [
            (ind.name, ind.definition, ind.role, idx)
            for idx, ind in enumerate(concept_map.indicators)
        ]

    records: list[dict] = []
    for ind_name, ind_defn, ind_role, idx in jobs:
        n = idx + 1
        console.print(f"  [[bold]{n}/{n_assertions}[/bold]]  [cyan]{ind_name}[/cyan]")

        try:
            with console.status(f"  Developing assertion for {ind_name!r}..."):
                assertion = ad_agent.develop_assertion(topic, ind_name, ind_defn, ind_role)

            vt_color = "yellow" if assertion.variable_type == "subjective" else "blue"
            console.print(f"         variable_type  [{vt_color}]{assertion.variable_type}[/{vt_color}]")
            console.print(f"         basic_concept  [bold]{assertion.basic_concept}[/bold]")
            console.print(
                f"         structure      [bold]{assertion.structure_code}[/bold]"
                f"  [dim]→ {assertion.structure_id}[/dim]"
            )
            console.print(
                Panel(
                    f"[bold white]{assertion.assertion}[/bold white]",
                    border_style="green",
                    padding=(0, 2),
                )
            )
            if assertion.rationale:
                console.print(f"         [dim]{assertion.rationale[:140]}[/dim]")
            if assertion.warnings:
                for w in assertion.warnings:
                    console.print(f"         [yellow]⚠ {w}[/yellow]")
            console.print()

            records.append({
                "topic": topic,
                "ci_or_cp": ci_or_cp,
                "indicator_model": concept_map.indicator_model,
                "construct_definition": concept_map.construct_definition,
                "indicator_index": idx,
                **assertion.model_dump(),
            })

        except Exception as exc:
            console.print(f"         [red]ERROR: {exc}[/red]\n")
            records.append({
                "topic": topic,
                "ci_or_cp": ci_or_cp,
                "indicator_index": idx,
                "input_indicator": ind_name,
                "parent_concept": topic,
                "error": str(exc),
            })

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    n_ok = sum(1 for r in records if "error" not in r)
    n_err = len(records) - n_ok
    console.print(Rule("[bold green]Pipeline Complete[/bold green]", style="green"))
    console.print(
        f"\n  Topic      : [bold]{topic!r}[/bold]\n"
        f"  CI/CP      : [{type_color}]{ci_or_cp}[/{type_color}]\n"
        f"  Assertions : [green]{n_ok} generated[/green]"
        + (f"  [red]{n_err} failed[/red]" if n_err else "")
    )

    if output:
        write_jsonl(output, records)
        console.print(f"  Saved to   : [dim]{output}[/dim]")
    console.print()


if __name__ == "__main__":
    app()
