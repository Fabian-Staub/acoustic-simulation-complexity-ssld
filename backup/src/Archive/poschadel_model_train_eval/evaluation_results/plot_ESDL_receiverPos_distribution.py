#!/usr/bin/env python3
"""Plot receiver-position ESDL distributions for every dataset in one CSV.

The input is a ``metrics_receiverPos_<model>.csv`` file. One histogram panel is
created per ``prediction_dataset`` using common bin edges and a common x-axis,
so the distributions are directly comparable. Non-finite ESDL values (empty,
``nan``, or ``inf``) are reported and excluded.

Example:
    python plot_esdl_distribution.py \
        --metrics metrics_receiverPos_my_model.csv

Default output:
    esdl_distribution_metrics_receiverPos_my_model.png

Requirements:
    pip install numpy matplotlib
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics_positional",
        nargs="?",
        type=Path,
        help="Receiver-position metrics CSV.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Named alternative to the positional metrics CSV argument.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. By default it is written beside the input CSV.",
    )
    parser.add_argument(
        "--dataset-column",
        default="prediction_dataset",
        help="CSV column containing the dataset name.",
    )
    parser.add_argument(
        "--score-column",
        default="ESDL",
        help="CSV column containing the score to plot.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="NAME",
        help="Plot only this dataset; repeat to select several datasets.",
    )
    parser.add_argument(
        "--bins",
        default="auto",
        help="Histogram bins: an integer or a NumPy rule such as auto, fd, or sturges.",
    )
    parser.add_argument(
        "--counts",
        action="store_true",
        help="Plot receiver-position counts instead of probability density.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--title",
        default="ESDL distribution by dataset",
        help="Figure title.",
    )
    return parser.parse_args()


def normalized_name(name: str) -> str:
    return name.replace("\\", "").strip().strip("*").lower()


def resolve_column(fieldnames: Sequence[str], wanted: str) -> str:
    wanted_normalized = normalized_name(wanted)
    for name in fieldnames:
        if normalized_name(name) == wanted_normalized:
            return name
    raise ValueError(f"Missing column {wanted!r}; found {list(fieldnames)}")


def resolve_input(args: argparse.Namespace) -> Path:
    if args.metrics_positional is not None and args.metrics is not None:
        raise ValueError("Pass the metrics CSV positionally or with --metrics, not both")
    path = args.metrics or args.metrics_positional
    if path is None:
        raise ValueError("A receiver-position metrics CSV is required")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")
    return path


def default_output_path(metrics_path: Path) -> Path:
    return metrics_path.with_name(f"esdl_distribution_{metrics_path.stem}.png")


def parse_bins(raw: str) -> int | str:
    try:
        bins = int(raw)
    except ValueError:
        bins = raw.lower()
        allowed = {"auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"}
        if bins not in allowed:
            raise ValueError(
                f"Unknown --bins value {raw!r}; use a positive integer or "
                f"one of {sorted(allowed)}"
            )
        return bins
    if bins < 1:
        raise ValueError("--bins must be at least 1")
    return bins


def load_scores(
    path: Path,
    dataset_column_wanted: str,
    score_column_wanted: str,
    selected_datasets: Sequence[str] | None,
) -> tuple[OrderedDict[str, np.ndarray], int]:
    scores: OrderedDict[str, list[float]] = OrderedDict()
    invalid_scores = 0
    selected = set(selected_datasets) if selected_datasets else None

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        dataset_column = resolve_column(reader.fieldnames, dataset_column_wanted)
        score_column = resolve_column(reader.fieldnames, score_column_wanted)

        for row_number, row in enumerate(reader, start=2):
            dataset = (row.get(dataset_column) or "").strip()
            if not dataset:
                raise ValueError(f"Empty dataset name in {path} row {row_number}")
            if selected is not None and dataset not in selected:
                continue
            raw_score = (row.get(score_column) or "").strip()
            try:
                score = float(raw_score)
            except ValueError:
                invalid_scores += 1
                continue
            if not math.isfinite(score):
                invalid_scores += 1
                continue
            scores.setdefault(dataset, []).append(score)

    if selected is not None:
        missing = [dataset for dataset in selected_datasets or [] if dataset not in scores]
        if missing:
            raise ValueError(
                "Selected dataset(s) have no finite scores or are absent: "
                + ", ".join(missing)
            )
        # Honor the user's repeated --dataset order.
        scores = OrderedDict((dataset, scores[dataset]) for dataset in selected_datasets or [])
    if not scores:
        raise ValueError("No finite scores were found for any selected dataset")
    return OrderedDict(
        (dataset, np.asarray(values, dtype=np.float64))
        for dataset, values in scores.items()
    ), invalid_scores


def common_bin_edges(arrays: Sequence[np.ndarray], bins: int | str) -> np.ndarray:
    all_scores = np.concatenate(arrays)
    if np.all(all_scores == all_scores[0]):
        center = float(all_scores[0])
        half_width = max(abs(center) * 0.05, 0.01)
        count = bins if isinstance(bins, int) else 10
        return np.linspace(center - half_width, center + half_width, count + 1)
    return np.histogram_bin_edges(all_scores, bins=bins)


def plot_distributions(
    scores: OrderedDict[str, np.ndarray],
    output_path: Path,
    bins: int | str,
    density: bool,
    dpi: int,
    title: str,
) -> None:
    dataset_count = len(scores)
    ncols = min(3, max(1, math.ceil(math.sqrt(dataset_count))))
    nrows = math.ceil(dataset_count / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.8 * nrows),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    edges = common_bin_edges(list(scores.values()), bins)
    colors = plt.get_cmap("tab10").colors

    for index, (dataset, values) in enumerate(scores.items()):
        axis = flat_axes[index]
        color = colors[index % len(colors)]
        axis.hist(
            values,
            bins=edges,
            density=density,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.7,
        )
        mean = float(np.mean(values))
        median = float(np.median(values))
        axis.axvline(
            mean,
            color="#202020",
            linestyle="--",
            linewidth=1.5,
            label=f"mean = {mean:.3f}",
        )
        axis.axvline(
            median,
            color="#202020",
            linestyle=":",
            linewidth=1.5,
            label=f"median = {median:.3f}",
        )
        axis.set_title(f"{dataset}\n$n={len(values):,}$", fontsize=11)
        axis.set_ylabel("Density" if density else "Receiver positions")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)

    for axis in flat_axes[dataset_count:]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("ESDL")

    fig.suptitle(title, fontsize=15, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics_path = resolve_input(args)
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(metrics_path)
    )
    scores, invalid_scores = load_scores(
        metrics_path,
        args.dataset_column,
        args.score_column,
        args.dataset,
    )
    plot_distributions(
        scores=scores,
        output_path=output_path,
        bins=parse_bins(args.bins),
        density=not args.counts,
        dpi=args.dpi,
        title=args.title,
    )
    summary = ", ".join(
        f"{dataset}: {len(values):,}" for dataset, values in scores.items()
    )
    print(f"Finite scores by dataset: {summary}")
    if invalid_scores:
        print(f"Skipped non-finite/invalid scores: {invalid_scores:,}")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
