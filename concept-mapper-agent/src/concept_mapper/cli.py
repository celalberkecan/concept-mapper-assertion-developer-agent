"""Typer CLI for the Concept Mapper agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from dotenv import find_dotenv, load_dotenv
from rich.console import Console

load_dotenv(find_dotenv(usecwd=True), override=True)
from rich.panel import Panel
from rich.pretty import Pretty

from .agent import ConceptMapperAgent
from .llm_clients.base import BaseLLMClient

app = typer.Typer(
    name="concept-mapper",
    help="Concept Mapper Agent — maps survey topics to structured concept maps.",
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
        from .llm_clients.openai_client import OpenAIClient

        return OpenAIClient(
            model=cfg.get("model", "gpt-4o-mini"),
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 1200),
        )
    elif provider == "ollama":
        from .llm_clients.ollama_client import OllamaClient

        return OllamaClient(
            model=cfg.get("model", "qwen2.5:7b-instruct"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 1200),
        )
    elif provider == "transformers":
        from .llm_clients.transformers_client import TransformersClient

        model_path = cfg.get("model_path")
        if not model_path:
            console.print(
                "[red]Error:[/red] 'model_path' is required in the transformers config."
            )
            raise typer.Exit(1)
        return TransformersClient(
            model_path=model_path,
            torch_dtype=cfg.get("torch_dtype", "bfloat16"),
            device_map=cfg.get("device_map", "auto"),
            max_new_tokens=cfg.get("max_new_tokens", 1200),
            temperature=cfg.get("temperature", 0.0),
            trust_remote_code=cfg.get("trust_remote_code", False),
            load_in_4bit=cfg.get("load_in_4bit", False),
            load_in_8bit=cfg.get("load_in_8bit", False),
        )
    else:
        console.print(
            f"[red]Error:[/red] Unknown provider {provider!r}. "
            "Choose one of: openai, ollama, transformers."
        )
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("map-one")
def map_one(
    topic: str = typer.Option(..., "--topic", "-t", help="The survey concept to map."),
    provider: str = typer.Option(
        "openai", "--provider", "-p", help="LLM provider: openai | ollama | transformers"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to a YAML config file (e.g. configs/openai.yaml)."
    ),
    show_raw: bool = typer.Option(False, "--raw", help="Also print the raw LLM response."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write the JSON result to this file."
    ),
) -> None:
    """Map a single survey concept and print the structured concept map."""
    cfg = _load_config(config)
    client = _build_client(provider, cfg)
    agent = ConceptMapperAgent(client)

    console.print(f"[bold]Mapping:[/bold] {topic!r}  [dim]provider={provider}[/dim]\n")

    with console.status("Calling LLM..."):
        concept_map, raw_text = agent.map_concept_with_raw(topic)

    if show_raw:
        console.print(Panel(raw_text, title="Raw LLM response", border_style="dim"))

    console.print(
        Panel(Pretty(concept_map.model_dump()), title="[bold]Concept Map[/bold]", border_style="green")
    )

    if output:
        output.write_text(json.dumps(concept_map.model_dump(), indent=2))
        console.print(f"\n[green]Saved to {output}[/green]")


@app.command("smoke-test")
def smoke_test() -> None:
    """Run a self-contained pipeline test using FakeLLMClient (no API calls needed)."""
    from .llm_clients.fake_client import FakeLLMClient

    console.print("[bold]Smoke test[/bold] [dim](no API calls)[/dim]\n")

    agent = ConceptMapperAgent(FakeLLMClient())
    topics = ["fear of crime", "age", "political trust"]
    failures: list[str] = []

    for topic in topics:
        try:
            cm = agent.map_concept(topic)
            console.print(
                Panel(
                    Pretty(cm.model_dump()),
                    title=f"[green]{topic}[/green]",
                    border_style="green",
                )
            )
        except Exception as exc:
            console.print(f"[red]FAILED[/red] {topic!r}: {exc}")
            failures.append(topic)

    if failures:
        console.print(f"\n[bold red]{len(failures)} test(s) failed.[/bold red]")
        raise typer.Exit(1)
    else:
        console.print("\n[bold green]All smoke tests passed.[/bold green]")


@app.command("run-batch")
def run_batch(
    input: Path = typer.Option(..., "--input", "-i", help="Path to the gold Excel file."),
    sheet: str = typer.Option("Concept Mapper Gold", "--sheet", "-s", help="Sheet name to read."),
    provider: str = typer.Option(
        "openai", "--provider", "-p", help="LLM provider: openai | ollama | transformers"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to a YAML config file."
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output JSONL file path."),
) -> None:
    """Run the agent over every row in the gold Excel sheet and save predictions to JSONL."""
    from .io import read_concept_mapper_gold_xlsx, write_jsonl

    cfg = _load_config(config)
    client = _build_client(provider, cfg)
    agent = ConceptMapperAgent(client)

    gold_rows = read_concept_mapper_gold_xlsx(input, sheet_name=sheet)
    console.print(
        f"[bold]Batch run[/bold] — {len(gold_rows)} topics  "
        f"[dim]provider={provider}  output={output}[/dim]\n"
    )

    records: list[dict] = []
    for i, row in enumerate(gold_rows, 1):
        topic = row["input_topic_parent_concept"]
        concept_id = row["concept_id"]
        console.print(f"[{i}/{len(gold_rows)}] {concept_id}  {topic!r}", end=" ")

        try:
            cm = agent.map_concept(topic)
            record = {"concept_id": concept_id, "input_topic": topic, **cm.model_dump()}
            console.print("[green]OK[/green]")
        except Exception as exc:
            record = {"concept_id": concept_id, "input_topic": topic, "error": str(exc)}
            console.print(f"[red]ERROR[/red] {exc}")

        records.append(record)

    write_jsonl(output, records)
    n_ok = sum(1 for r in records if "error" not in r)
    n_err = len(records) - n_ok
    console.print(f"\n[green]Done.[/green] {n_ok} OK, {n_err} errors → {output}")


@app.command("evaluate")
def evaluate(
    gold: Path = typer.Option(..., "--gold", "-g", help="Path to the gold Excel file."),
    sheet: str = typer.Option("Concept Mapper Gold", "--sheet", "-s", help="Sheet name to read."),
    predictions: Path = typer.Option(..., "--predictions", "-p", help="JSONL predictions file."),
    output: Path = typer.Option(..., "--output", "-o", help="Output CSV file path."),
    judge_provider: Optional[str] = typer.Option(
        None,
        "--judge-provider",
        help="If set (openai | ollama | transformers), scores indicator coverage/"
        "distinctiveness against gold_indicators_conceptual via LLM-as-judge "
        "(llm_judge.py). Adds one LLM call per CP row. Omit to skip (free, deterministic-only).",
    ),
    judge_config: Optional[Path] = typer.Option(
        None, "--judge-config", help="YAML config for the judge client (model, temperature, ...)."
    ),
) -> None:
    """Evaluate predictions against gold labels and save metrics to CSV."""
    import pandas as pd

    from .evaluator import compute_summary, evaluate_batch
    from .io import read_concept_mapper_gold_xlsx, read_jsonl

    gold_rows = read_concept_mapper_gold_xlsx(gold, sheet_name=sheet)
    pred_records = read_jsonl(predictions)

    judge_client = None
    if judge_provider:
        judge_client = _build_client(judge_provider, _load_config(judge_config))
        console.print(f"[dim]Using {judge_provider} as indicator-quality judge[/dim]\n")

    eval_records = evaluate_batch(gold_rows, pred_records, judge_client=judge_client)

    df = pd.DataFrame(eval_records)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    summary = compute_summary(eval_records)

    console.print(f"\n[bold]Evaluation complete[/bold] → {output}\n")
    console.print(f"  Total rows          : {summary['total_rows']}")
    console.print(f"  Gold CI / CP        : {summary['n_gold_ci']} / {summary['n_gold_cp']}")
    console.print(f"  CI/CP accuracy      : {summary['ci_cp_accuracy']}")
    console.print(f"  Indicator model acc : {summary['indicator_model_accuracy_cp_only']}  (CP rows only)")
    console.print(f"  Mean count abs diff : {summary['mean_indicator_count_abs_diff_cp_only']}  (CP rows only)")
    if judge_client is not None:
        console.print(f"  Mean coverage (1-5)       : {summary['mean_indicator_coverage_1to5_cp_only']}  (CP rows only, LLM-judge)")
        console.print(f"  Mean distinctiveness (1-5): {summary['mean_indicator_distinctiveness_1to5_cp_only']}  (CP rows only, LLM-judge)")
    if summary["n_errors"]:
        console.print(f"  [yellow]Errors / missing    : {summary['n_errors']}[/yellow]")


@app.command("show-results")
def show_results(
    results: Path = typer.Option(..., "--results", "-r", help="Evaluation CSV file (from evaluate command)."),
) -> None:
    """Display a rich interactive summary of evaluation results from a CSV file."""
    import math

    import pandas as pd
    from rich.table import Table
    from rich.text import Text

    if not results.exists():
        console.print(f"[red]File not found:[/red] {results}")
        raise typer.Exit(1)

    # keep_default_na=False prevents pandas from silently converting the
    # string "NA" (our indicator_model value for CI rows) back to float NaN.
    df = pd.read_csv(results, keep_default_na=False)

    # Coerce numeric columns that become strings with keep_default_na=False
    df["indicator_count_abs_diff"] = pd.to_numeric(df["indicator_count_abs_diff"], errors="coerce")
    df["pred_indicator_count"] = pd.to_numeric(df["pred_indicator_count"], errors="coerce")
    df["gold_indicator_count"] = pd.to_numeric(df["gold_indicator_count"], errors="coerce")

    def _bool(val) -> bool | None:
        if val is None or val == "" or (isinstance(val, float) and math.isnan(val)):
            return None
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() == "true"

    def _str(val) -> str:
        if val is None or val == "" or (isinstance(val, float) and math.isnan(val)):
            return ""
        return str(val)

    # -----------------------------------------------------------------------
    # 1. Summary panel
    # -----------------------------------------------------------------------
    total = len(df)
    n_ci = int((df["gold_ci_cp"] == "CI").sum())
    n_cp = int((df["gold_ci_cp"] == "CP").sum())

    ci_cp_correct = df["ci_cp_correct"].apply(_bool)
    n_ci_cp_correct = int(ci_cp_correct.sum())
    ci_cp_acc = n_ci_cp_correct / total

    cp_rows = df[df["gold_ci_cp"] == "CP"]
    im_correct = cp_rows["indicator_model_correct"].apply(_bool)
    n_im_correct = int(im_correct.sum())
    im_acc = n_im_correct / len(cp_rows) if len(cp_rows) > 0 else None

    count_diffs = cp_rows["indicator_count_abs_diff"].dropna()
    mean_diff = float(count_diffs.mean()) if len(count_diffs) > 0 else None

    n_errors = int((df["error"] != "").sum())

    summary_lines = [
        f"[bold]Total rows[/bold]          {total}  ([cyan]CI={n_ci}[/cyan]  [magenta]CP={n_cp}[/magenta])",
        f"[bold]CI/CP accuracy[/bold]      [{'green' if ci_cp_acc >= 0.8 else 'yellow' if ci_cp_acc >= 0.6 else 'red'}]{n_ci_cp_correct}/{total}  ({ci_cp_acc:.1%})[/]",
        f"[bold]Indicator model acc[/bold] [{'green' if im_acc and im_acc >= 0.7 else 'yellow' if im_acc and im_acc >= 0.5 else 'red'}]{n_im_correct}/{len(cp_rows)}  ({im_acc:.1%})[/]  [dim](CP rows only)[/dim]" if im_acc is not None else f"[bold]Indicator model acc[/bold] —",
        f"[bold]Mean count abs diff[/bold] {mean_diff:.2f}  [dim](CP rows only)[/dim]" if mean_diff is not None else f"[bold]Mean count abs diff[/bold] —",
        f"[bold]Errors[/bold]              [{'red' if n_errors else 'green'}]{n_errors}[/]",
    ]
    console.print()
    console.print(Panel("\n".join(summary_lines), title="[bold white] Summary [/bold white]", border_style="blue", padding=(0, 2)))

    # -----------------------------------------------------------------------
    # 2. CI/CP confusion matrix
    # -----------------------------------------------------------------------
    pred_labels = ["CI", "CP"]
    matrix: dict[tuple[str, str], int] = {}
    for gold_label in pred_labels:
        for pred_label in pred_labels:
            mask = (df["gold_ci_cp"] == gold_label) & (df["pred_ci_cp"] == pred_label)
            matrix[(gold_label, pred_label)] = int(mask.sum())

    cm_table = Table(title="CI/CP Confusion Matrix", border_style="blue", show_lines=True)
    cm_table.add_column("", style="bold", width=14)
    cm_table.add_column("Pred CI", justify="center", width=10)
    cm_table.add_column("Pred CP", justify="center", width=10)

    for gold_label in pred_labels:
        row_cells = [f"Gold {gold_label}"]
        for pred_label in pred_labels:
            count = matrix[(gold_label, pred_label)]
            is_diagonal = gold_label == pred_label
            cell = Text(str(count), style="bold green" if is_diagonal else "bold red")
            row_cells.append(cell)
        cm_table.add_row(*row_cells)

    console.print()
    console.print(cm_table)

    # -----------------------------------------------------------------------
    # 3. Indicator model breakdown (CP rows only)
    # -----------------------------------------------------------------------
    im_labels = ["formative", "reflective", "mixed"]

    im_table = Table(title="Indicator Model Breakdown  [dim](CP rows only)[/dim]", border_style="blue", show_lines=True)
    im_table.add_column("Gold \\ Predicted", style="bold", width=16)
    for pl in im_labels:
        im_table.add_column(pl.capitalize(), justify="center", width=12)
    im_table.add_column("(other)", justify="center", width=10)

    for gold_m in im_labels:
        gold_mask = cp_rows["gold_indicator_model"] == gold_m
        row_cells = [gold_m.capitalize()]
        for pred_m in im_labels:
            count = int((gold_mask & (cp_rows["pred_indicator_model"] == pred_m)).sum())
            is_diagonal = gold_m == pred_m
            row_cells.append(Text(str(count), style="bold green" if is_diagonal else ("bold red" if count > 0 else "dim")))
        # "other" = predicted something not in the three labels
        other_count = int((gold_mask & ~cp_rows["pred_indicator_model"].isin(im_labels)).sum())
        row_cells.append(Text(str(other_count), style="bold red" if other_count > 0 else "dim"))
        im_table.add_row(*row_cells)

    console.print()
    console.print(im_table)

    # -----------------------------------------------------------------------
    # 4. Per-topic result table
    # -----------------------------------------------------------------------
    topic_table = Table(
        title="Per-Topic Results",
        border_style="blue",
        show_lines=False,
        header_style="bold white on dark_blue",
        expand=False,
    )
    topic_table.add_column("ID", style="dim", width=5, no_wrap=True)
    topic_table.add_column("Topic", width=28, no_wrap=True)
    topic_table.add_column("G", justify="center", width=4, no_wrap=True)
    topic_table.add_column("P", justify="center", width=4, no_wrap=True)
    topic_table.add_column("Gold model", justify="center", width=11, no_wrap=True)
    topic_table.add_column("Pred model", justify="center", width=11, no_wrap=True)
    topic_table.add_column("Cnt Δ", justify="center", width=6, no_wrap=True)
    topic_table.add_column("Status", justify="center", width=9, no_wrap=True)

    for _, row in df.iterrows():
        correct_ci_cp = _bool(row["ci_cp_correct"])
        correct_im = _bool(row["indicator_model_correct"])
        is_cp = row["gold_ci_cp"] == "CP"
        count_diff = row["indicator_count_abs_diff"]

        if correct_ci_cp is False:
            row_style = "red"
            status = Text("✗ CI/CP", style="bold red")
        elif is_cp and correct_im is False:
            row_style = "yellow"
            status = Text("~ model", style="bold yellow")
        else:
            row_style = "green"
            status = Text("✓", style="bold green")

        count_diff_str = f"{int(count_diff):+d}" if (is_cp and not (isinstance(count_diff, float) and math.isnan(count_diff))) else "—"

        topic_table.add_row(
            _str(row["concept_id"]),
            _str(row["input_topic"])[:30],
            Text(row["gold_ci_cp"], style="cyan"),
            Text(_str(row["pred_ci_cp"]), style="magenta"),
            Text(_str(row["gold_indicator_model"]), style="dim"),
            Text(_str(row["pred_indicator_model"]), style="dim"),
            count_diff_str,
            status,
            style=row_style,
        )

    console.print()
    console.print(topic_table)

    # -----------------------------------------------------------------------
    # 5. Wrong predictions summary
    # -----------------------------------------------------------------------
    wrong_ci_cp = df[df["ci_cp_correct"].apply(_bool) == False]  # noqa: E712
    # Only flag indicator model as wrong when the gold actually has a model specified
    wrong_im = df[
        (df["gold_ci_cp"] == "CP")
        & (df["gold_indicator_model"].isin(["formative", "reflective", "mixed"]))
        & (df["indicator_model_correct"].apply(_bool) == False)  # noqa: E712
    ]

    if len(wrong_ci_cp) or len(wrong_im):
        console.print()
        console.print(Panel(
            _format_wrong(wrong_ci_cp, wrong_im),
            title="[bold red] Misclassifications [/bold red]",
            border_style="red",
            padding=(0, 2),
        ))
    console.print()


def _format_wrong(wrong_ci_cp, wrong_im) -> str:
    lines: list[str] = []
    if len(wrong_ci_cp):
        lines.append(f"[bold]CI/CP wrong ({len(wrong_ci_cp)}):[/bold]")
        for _, r in wrong_ci_cp.iterrows():
            lines.append(f"  {r['concept_id']}  [cyan]{r['input_topic']}[/cyan]  gold=[green]{r['gold_ci_cp']}[/green] → pred=[red]{r['pred_ci_cp']}[/red]")
    if len(wrong_im):
        if lines:
            lines.append("")
        lines.append(f"[bold]Indicator model wrong ({len(wrong_im)}):[/bold]")
        for _, r in wrong_im.iterrows():
            lines.append(f"  {r['concept_id']}  [cyan]{r['input_topic']}[/cyan]  gold=[green]{r['gold_indicator_model']}[/green] → pred=[red]{r['pred_indicator_model']}[/red]")
    return "\n".join(lines)


if __name__ == "__main__":
    app()
