#!/usr/bin/env python3
"""Create the six-panel BLASTp summary figure used in the ASAMP study.

The input is a headerless BLAST tabular file with these columns:
qseqid, sseqid, pident, length, mismatch, gapopen, qstart, qend,
sstart, send, evalue, bitscore and qcovs.

Example
-------
python scripts/04_plot_blastp_results.py \
    --input supplementary/tables11_alignment-HitTable.csv \
    --output-prefix figures/Figure_19_BLASTp_Results \
    --total-candidates 243 \
    --highlight-query AMP_0206

The script writes a vector PDF, a 500-dpi PNG and a candidate-level summary CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

BLAST_COLUMNS = [
    "qseqid",
    "sseqid",
    "pident",
    "length",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
    "qcovs",
]

C_NO_HIT = "#4C72B0"
C_HIT = "#DD8452"
C_HIGH = "#55A868"
C_MODERATE = "#C44E52"
C_LOW = "#8172B2"
C_HISTOGRAM = "#4C72B0"
C_MEDIAN = "#C44E52"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ASAMP BLASTp summary panels.")
    parser.add_argument("--input", required=True, type=Path, help="BLASTp hit table CSV.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("Figure_19_BLASTp_Results"),
        help="Output path without a file extension.",
    )
    parser.add_argument(
        "--total-candidates",
        type=int,
        default=243,
        help="Total number of candidates submitted to BLASTp.",
    )
    parser.add_argument(
        "--highlight-query",
        default="AMP_0206",
        help="Query identifier to highlight in panel F when present.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=500,
        help="Resolution of the PNG output.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_blast_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"BLASTp table not found: {path}")

    data = pd.read_csv(path, header=None, names=BLAST_COLUMNS)
    if data.empty:
        raise ValueError("The BLASTp table contains no alignments.")

    numeric_columns = [
        "pident",
        "length",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
        "qcovs",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if data["qseqid"].isna().any() or data["sseqid"].isna().any():
        raise ValueError("Query and subject identifiers must not be missing.")
    if (data["evalue"] < 0).any():
        raise ValueError("E-values must be non-negative.")

    data["log10_evalue"] = np.log10(data["evalue"].clip(lower=1e-300))
    return data


def select_best_hits(data: pd.DataFrame) -> pd.DataFrame:
    ordered = data.sort_values(
        ["qseqid", "evalue", "bitscore", "pident", "qcovs"],
        ascending=[True, True, False, False, False],
    )
    return ordered.groupby("qseqid", as_index=False).first()


def identity_category(identity: float) -> str:
    if identity >= 90:
        return "High identity\n(≥90%)"
    if identity >= 50:
        return "Moderate identity\n(50–89%)"
    return "Low identity\n(<50%)"


def identity_colour(identity: float) -> str:
    if identity >= 90:
        return C_HIGH
    if identity >= 50:
        return C_MODERATE
    return C_LOW


def add_panel_labels(axes: list[plt.Axes]) -> None:
    for axis, label in zip(axes, "ABCDEF"):
        axis.text(
            -0.14,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
        )


def create_figure(
    data: pd.DataFrame,
    total_candidates: int,
    highlight_query: str,
) -> tuple[plt.Figure, pd.DataFrame]:
    best = select_best_hits(data)
    matched_candidates = best["qseqid"].nunique()
    no_hit_candidates = total_candidates - matched_candidates

    if no_hit_candidates < 0:
        raise ValueError(
            "The number of matched queries exceeds --total-candidates. "
            "Check the input table and total-candidate value."
        )

    best["identity_category"] = best["pident"].map(identity_category)
    category_order = [
        "High identity\n(≥90%)",
        "Moderate identity\n(50–89%)",
        "Low identity\n(<50%)",
    ]
    category_counts = (
        best["identity_category"].value_counts().reindex(category_order, fill_value=0)
    )

    figure = plt.figure(figsize=(10, 8.5), facecolor="white")
    grid = gridspec.GridSpec(
        2,
        3,
        figure=figure,
        hspace=0.48,
        wspace=0.42,
        left=0.08,
        right=0.97,
        top=0.96,
        bottom=0.09,
    )

    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[0, 2])
    ax_d = figure.add_subplot(grid[1, 0])
    ax_e = figure.add_subplot(grid[1, 1])
    ax_f = figure.add_subplot(grid[1, 2])
    axes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]

    sizes = [no_hit_candidates, matched_candidates]
    colours = [C_NO_HIT, C_HIT]
    _, _, percentage_labels = ax_a.pie(
        sizes,
        colors=colours,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 1.5},
        pctdistance=0.72,
        textprops={"fontsize": 8},
    )
    for label in percentage_labels:
        label.set_color("white")
        label.set_fontweight("bold")

    no_hit_percentage = 100 * no_hit_candidates / total_candidates
    hit_percentage = 100 * matched_candidates / total_candidates
    ax_a.legend(
        handles=[
            mpatches.Patch(
                color=C_NO_HIT,
                label=f"No significant match (n = {no_hit_candidates}, {no_hit_percentage:.1f}%)",
            ),
            mpatches.Patch(
                color=C_HIT,
                label=f"Significant match (n = {matched_candidates}, {hit_percentage:.1f}%)",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        fontsize=7.5,
        frameon=False,
    )
    ax_a.set_title(
        f"Database match status\n(n = {total_candidates} candidates)",
        fontsize=9,
        fontweight="bold",
        pad=6,
    )

    identity_bins = np.arange(
        np.floor(data["pident"].min() / 5) * 5,
        np.ceil(data["pident"].max() / 5) * 5 + 5,
        5,
    )
    ax_b.hist(
        data["pident"],
        bins=identity_bins,
        color=C_HISTOGRAM,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )
    median_identity = data["pident"].median()
    ax_b.axvline(
        median_identity,
        color=C_MEDIAN,
        linewidth=1.5,
        linestyle="--",
        label=f"Median = {median_identity:.1f}%",
    )
    ax_b.set_xlabel("Percentage identity (%)", fontsize=9)
    ax_b.set_ylabel("Number of alignments", fontsize=9)
    ax_b.set_title("Alignment identity\ndistribution", fontsize=9, fontweight="bold")
    ax_b.legend(fontsize=8, frameon=False)
    ax_b.yaxis.set_major_locator(MaxNLocator(integer=True))

    evalue_min = np.floor(data["log10_evalue"].min())
    evalue_max = np.ceil(data["log10_evalue"].max())
    evalue_bins = np.linspace(evalue_min, evalue_max, 14)
    ax_c.hist(
        data["log10_evalue"],
        bins=evalue_bins,
        color=C_HIGH,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )
    median_log_evalue = data["log10_evalue"].median()
    ax_c.axvline(
        median_log_evalue,
        color=C_MEDIAN,
        linewidth=1.5,
        linestyle="--",
        label=f"Median = {median_log_evalue:.1f}",
    )
    ax_c.set_xlabel("log₁₀(E-value)", fontsize=9)
    ax_c.set_ylabel("Number of alignments", fontsize=9)
    ax_c.set_title("E-value distribution\n(log scale)", fontsize=9, fontweight="bold")
    ax_c.legend(fontsize=8, frameon=False)
    ax_c.yaxis.set_major_locator(MaxNLocator(integer=True))

    coverage_bins = np.arange(
        np.floor(data["qcovs"].min() / 5) * 5,
        np.ceil(data["qcovs"].max() / 5) * 5 + 5,
        5,
    )
    ax_d.hist(
        data["qcovs"],
        bins=coverage_bins,
        color=C_LOW,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )
    median_coverage = data["qcovs"].median()
    ax_d.axvline(
        median_coverage,
        color=C_MEDIAN,
        linewidth=1.5,
        linestyle="--",
        label=f"Median = {median_coverage:.1f}%",
    )
    ax_d.set_xlabel("Query coverage (%)", fontsize=9)
    ax_d.set_ylabel("Number of alignments", fontsize=9)
    ax_d.set_title("Query coverage\ndistribution", fontsize=9, fontweight="bold")
    ax_d.legend(fontsize=8, frameon=False)
    ax_d.yaxis.set_major_locator(MaxNLocator(integer=True))

    bars = ax_e.bar(
        category_order,
        category_counts.to_numpy(),
        color=[C_HIGH, C_MODERATE, C_LOW],
        edgecolor="white",
        linewidth=0.8,
        width=0.55,
    )
    for bar, value in zip(bars, category_counts.to_numpy()):
        ax_e.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax_e.set_ylabel("Number of candidates\n(best hit per query)", fontsize=9)
    ax_e.set_title(
        "Sequence-similarity category\n(best hit per query)",
        fontsize=9,
        fontweight="bold",
    )
    ax_e.set_ylim(0, max(1, category_counts.max()) * 1.18)
    ax_e.yaxis.set_major_locator(MaxNLocator(integer=True))

    point_colours = best["pident"].map(identity_colour)
    ax_f.scatter(
        best["pident"],
        best["qcovs"],
        c=point_colours,
        s=28,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.4,
    )

    highlighted = best.loc[best["qseqid"] == highlight_query]
    if highlighted.empty:
        highlighted = best.loc[
            (best["pident"] == best["pident"].max())
            & (best["qcovs"] == best["qcovs"].max())
        ].head(1)

    if not highlighted.empty:
        row = highlighted.iloc[0]
        ax_f.scatter(
            [row["pident"]],
            [row["qcovs"]],
            color="gold",
            s=80,
            zorder=5,
            edgecolors="#333333",
            linewidths=0.8,
        )
        ax_f.annotate(
            str(row["qseqid"]),
            (row["pident"], row["qcovs"]),
            xytext=(-36, 10),
            textcoords="offset points",
            fontsize=7.5,
            arrowprops={"arrowstyle": "-", "linewidth": 0.7, "color": "#555555"},
        )

    ax_f.set_xlabel("Best-hit identity (%)", fontsize=9)
    ax_f.set_ylabel("Query coverage (%)", fontsize=9)
    ax_f.set_title(
        "Identity vs. query coverage\n(best hit per query)",
        fontsize=9,
        fontweight="bold",
    )
    ax_f.legend(
        handles=[
            mpatches.Patch(
                color=C_HIGH,
                label=f"≥90% identity (n = {category_counts.iloc[0]})",
            ),
            mpatches.Patch(
                color=C_MODERATE,
                label=f"50–89% identity (n = {category_counts.iloc[1]})",
            ),
            mpatches.Patch(
                color=C_LOW,
                label=f"<50% identity (n = {category_counts.iloc[2]})",
            ),
        ],
        fontsize=7,
        frameon=False,
        loc="lower right",
        handlelength=1.0,
    )

    for axis in [ax_b, ax_c, ax_d, ax_e, ax_f]:
        axis.tick_params(labelsize=8)

    add_panel_labels(axes)

    candidate_summary = best[
        ["qseqid", "sseqid", "pident", "qcovs", "evalue", "bitscore", "identity_category"]
    ].copy()
    return figure, candidate_summary


def save_outputs(
    figure: plt.Figure,
    candidate_summary: pd.DataFrame,
    output_prefix: Path,
    dpi: int,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    summary_path = output_prefix.with_name(output_prefix.name + "_best_hits.csv")

    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    candidate_summary.to_csv(summary_path, index=False)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path} ({dpi} dpi)")
    print(f"Saved: {summary_path}")


def main() -> None:
    args = parse_args()
    if args.total_candidates <= 0:
        raise ValueError("--total-candidates must be greater than zero.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero.")

    configure_matplotlib()
    data = load_blast_table(args.input)
    figure, candidate_summary = create_figure(
        data=data,
        total_candidates=args.total_candidates,
        highlight_query=args.highlight_query,
    )
    save_outputs(figure, candidate_summary, args.output_prefix, args.dpi)
    plt.close(figure)


if __name__ == "__main__":
    main()
