"""Build the two Concept Mapper figures used in the paper.

Figure 1 (indicator_model_confusion.pdf)
    Where the indicator model predictions go, pooled over all 16 model x
    condition cells and the 25 CP topics of the gold coding (n = 400). Cells are
    shaded by row proportion so the three gold rows are comparable, and the raw
    counts are printed.

Figure 2 (coding_comparison.pdf)
    CI/CP accuracy of the same 16 sets of predictions scored under both codings
    of the same 46 topics. One row per model and condition, with the gold coding
    and the theory-led coding as the two endpoints.

Both figures read the predictions from experiments/outputs/*.jsonl. Neither
regenerates anything, so no API key and no model call is involved.

Data sources
    Gold coding       assertion-developer-agent/data/..._final.xlsx
    Theory-led coding concept-mapper-agent/data/..._adjusted.xlsx, whose *_leo
                      columns hold that coding. read_concept_mapper_gold_xlsx
                      prefers those columns when they are present, which is what
                      we want here and only here.

Usage (run from concept-mapper-agent/):
    python experiments/make_paper_figures.py [--outdir PATH]

Colours follow the validated palette used across the paper's figures: blue
#2a78d6 and orange #eb6834 for the two codings, a single-hue blue ramp for the
heatmap. Text stays in ink colours rather than series colours.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from concept_mapper.io import read_concept_mapper_gold_xlsx, read_jsonl  # noqa: E402

GOLD_PATH = (
    REPO_ROOT.parent
    / "assertion-developer-agent"
    / "data"
    / "gesis_concept_mapper_assertion_evaluation_adjusted_for_assertion_agent_final.xlsx"
)
THEORY_PATH = REPO_ROOT / "data" / "gesis_concept_mapper_assertion_evaluation_adjusted.xlsx"
OUT_DIR_DEFAULT = REPO_ROOT.parent.parent.parent / "paper" / "figures"

MODELS: list[tuple[str, str]] = [
    ("", "gpt-4o-mini"),
    ("llama-3.1-8b-instruct_", "Llama-3.1-8B"),
    ("qwen3-8b_", "Qwen3-8B"),
    ("granite-4.2-8b_", "Granite-4.2-8B"),
]
VARIANTS: list[tuple[str, str]] = [
    ("a_zero_shot", "zero-shot"),
    ("b_prose_fewshot", "prose few-shot"),
    ("c_message_history_fewshot", "message-history"),
    ("d_gepa_optimized", "GEPA-optimized"),
]

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GOLD_BLUE = "#2a78d6"
THEORY_ORANGE = "#eb6834"
CONNECTOR = "#c9c8c4"
# Sequential blue ramp, lightest to darkest, from the reference palette.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

PRED_LABELS = ["formative", "reflective", "mixed", "NA"]
GOLD_LABELS = ["formative", "reflective", "mixed"]


def _style() -> None:
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.size": 8.5,
        "text.color": INK,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def load_predictions() -> dict[tuple[str, str], dict[str, dict]]:
    """Return {(model_prefix, variant): {concept_id: prediction}}."""
    out: dict[tuple[str, str], dict[str, dict]] = {}
    for prefix, _ in MODELS:
        for variant, _ in VARIANTS:
            path = REPO_ROOT / "experiments" / "outputs" / f"{prefix}{variant}.jsonl"
            records = read_jsonl(path)
            out[(prefix, variant)] = {r["concept_id"]: r for r in records if "concept_id" in r}
    return out


def figure_confusion(preds, gold_rows, out_path: Path) -> None:
    gold = {r["concept_id"]: r for r in gold_rows}
    cp_ids = [c for c, r in gold.items() if r["concept_level_ci_cp_gold"] == "CP"]

    counts = {g: {p: 0 for p in PRED_LABELS} for g in GOLD_LABELS}
    for cell in preds.values():
        for cid in cp_ids:
            if cid not in cell:
                continue
            g = gold[cid]["concept_level_indicator_model_gold"] or "NA"
            p = cell[cid].get("indicator_model") or "NA"
            if g in counts and p in counts[g]:
                counts[g][p] += 1

    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(5.1, 2.35))

    for i, g in enumerate(GOLD_LABELS):
        total = sum(counts[g].values())
        for j, p in enumerate(PRED_LABELS):
            n = counts[g][p]
            share = n / total if total else 0.0
            ax.add_patch(plt.Rectangle(
                (j + 0.01, i + 0.01), 0.98, 0.98,
                facecolor=cmap(share), edgecolor="white", linewidth=1.4,
            ))
            ax.text(j + 0.5, i + 0.5, str(n), ha="center", va="center",
                    fontsize=9, color="white" if share > 0.42 else INK)

    ax.set_xlim(0, len(PRED_LABELS))
    ax.set_ylim(len(GOLD_LABELS), 0)
    ax.set_xticks([j + 0.5 for j in range(len(PRED_LABELS))])
    ax.set_xticklabels(PRED_LABELS)
    ax.set_yticks([i + 0.5 for i in range(len(GOLD_LABELS))])
    ax.set_yticklabels([f"{g}\n(n={sum(counts[g].values())})" for g in GOLD_LABELS])
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("predicted indicator model", color=INK_MUTED, labelpad=8)
    ax.set_ylabel("gold indicator model", color=INK_MUTED, labelpad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")
    return counts


def figure_coding(preds, gold_rows, theory_rows, out_path: Path) -> None:
    gold = {r["concept_id"]: r["concept_level_ci_cp_gold"] for r in gold_rows}
    theory = {r["concept_id"]: r["concept_level_ci_cp_gold"] for r in theory_rows}
    ids = list(gold)

    rows = []
    for prefix, model_label in MODELS:
        for variant, variant_label in VARIANTS:
            cell = preds[(prefix, variant)]
            g = sum(1 for c in ids if c in cell and cell[c]["ci_or_cp"] == gold[c]) / len(ids)
            t = sum(1 for c in ids if c in cell and cell[c]["ci_or_cp"] == theory[c]) / len(ids)
            rows.append((model_label, variant_label, g * 100, t * 100))

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ypos = list(range(len(rows)))[::-1]

    for y, (_, _, g, t) in zip(ypos, rows):
        ax.plot([g, t], [y, y], color=CONNECTOR, linewidth=1.8, solid_capstyle="round", zorder=1)
    ax.scatter([r[2] for r in rows], ypos, s=34, color=GOLD_BLUE, zorder=3,
               edgecolor="white", linewidth=0.8, label="gold coding")
    ax.scatter([r[3] for r in rows], ypos, s=34, color=THEORY_ORANGE, zorder=3,
               edgecolor="white", linewidth=0.8, label="theory-led coding")

    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{m}, {v}" for m, v, _, _ in rows])
    ax.set_xlabel("CI/CP accuracy (percent)", color=INK_MUTED)
    ax.set_xlim(69, 98)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.grid(axis="x", color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)
    # Faint separators between the four model groups.
    for k in range(1, len(MODELS)):
        ax.axhline(len(rows) - k * len(VARIANTS) - 0.5,
                   color="#e8e7e3", linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    # Legend sits above the axes so it cannot overlap the rightmost markers.
    ax.legend(frameon=False, ncol=2, loc="lower left",
              bbox_to_anchor=(0.0, 1.01), handletextpad=0.4, borderpad=0.0)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(OUT_DIR_DEFAULT))
    args = parser.parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _style()
    preds = load_predictions()
    gold_rows = read_concept_mapper_gold_xlsx(GOLD_PATH)
    theory_rows = read_concept_mapper_gold_xlsx(THEORY_PATH)

    counts = figure_confusion(preds, gold_rows, out_dir / "indicator_model_confusion.pdf")
    rows = figure_coding(preds, gold_rows, theory_rows, out_dir / "coding_comparison.pdf")

    print("\nFigure 1 row totals:", {g: sum(v.values()) for g, v in counts.items()})
    print("Figure 1 predicted 'mixed' total:", sum(v["mixed"] for v in counts.values()))
    print("Figure 2 gold-coding range: "
          f"{min(r[2] for r in rows):.2f} to {max(r[2] for r in rows):.2f}")
    print("Figure 2 theory-led range:   "
          f"{min(r[3] for r in rows):.2f} to {max(r[3] for r in rows):.2f}")
    print("Figure 2 cells where theory-led is higher: "
          f"{sum(1 for r in rows if r[3] > r[2])} of {len(rows)}")


if __name__ == "__main__":
    main()
