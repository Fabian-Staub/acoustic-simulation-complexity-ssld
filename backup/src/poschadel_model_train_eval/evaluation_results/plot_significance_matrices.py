#!/usr/bin/env python3
r"""Plot square significance matrices in one PNG.

Expected input (wide CSV)::

    dataset,metric,model A vs model B,model A vs model C,...
    arni-dataset,LE,0.012,0.34,...

Each ``model A vs model B`` column is converted to a model-by-model matrix. The
wide CSV supplies Holm-adjusted p-values. The sibling ``pairwise_significance.csv``
supplies the ``better_model`` value needed to show direction. Green means the
row model is significantly better, orange means the column model is
significantly better, and light gray means the difference is not significant.
The diagonal and missing/non-numeric cells are white.

The script also tolerates Markdown artifacts found in pasted CSV data, such as
escaped underscores (``count\_MAE``) and surrounding ``**`` markers.

Example:
    python plot_significance_matrices.py \
        --csv_path significance_matrix.csv \
        --pairwise-csv pairwise_significance.csv

Optional:
    python plot_significance_matrices.py \
        --csv_path significance_matrix.csv \
        --alpha 0.05 --annotate-p-values

Output:
    checkpoints/evaluation_results/significance_test/significance_matrices.png
    checkpoints/evaluation_results/significance_test/model_significance_ranking.csv

The ranking uses a Copeland-style score: significant wins minus significant
losses. Higher score ranks first; ties are broken by more wins, then fewer
losses. Both an overall ranking and one ranking per dataset are written. Each
pairwise test is counted once, even though it appears twice in a square matrix.

Requirements:
    pip install numpy matplotlib
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np


DEFAULT_DATASETS = ("arni", "ivan", "motus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path(
            "checkpoints/evaluation_results/significance_test/"
            "significance_matrix.csv"
        ),
        help="Wide significance CSV",
    )
    parser.add_argument(
        "--pairwise-csv",
        type=Path,
        default=Path(
            "checkpoints/evaluation_results/significance_test/"
            "pairwise_significance.csv"
        ),
        help=(
            "Long pairwise CSV containing better_model. By default, use "
            "pairwise_significance.csv beside --csv_path"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/evaluation_results/significance_test"),
        help="Directory for the output PNG",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to plot (default: arni ivan motus)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold; significant means p < alpha (default: 0.05)",
    )
    parser.add_argument(
        "--annotate-p-values",
        action="store_true",
        help="Write p-values inside the matrix cells",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution (default: 200)",
    )
    parser.add_argument(
        "--ranking-csv",
        type=Path,
        default=None,
        help=(
            "Model-ranking CSV (default: model_significance_ranking.csv in "
            "--output-dir)"
        ),
    )
    return parser.parse_args()


def clean_markdown_artifacts(value: str) -> str:
    """Remove formatting artifacts without changing ordinary model names."""
    return value.strip().strip("*").replace("\\_", "_")


def canonical_dataset(value: str) -> str:
    """Map values such as 'arni-dataset' and 'Arni_dataset' to 'arni'."""
    value = clean_markdown_artifacts(value).casefold().replace("_", "-")
    if value.endswith("-dataset"):
        value = value[: -len("-dataset")]
    return value


def parse_p_value(value: str) -> float:
    value = clean_markdown_artifacts(value)
    if not value:
        return math.nan
    try:
        result = float(value)
    except ValueError:
        return math.nan
    return result if math.isfinite(result) else math.nan


def read_significance_csv(
    path: Path,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, list[float]]]]:
    """Return model pairs and rows of (dataset, metric, p-values)."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input CSV is empty: {path}") from exc

        header = [clean_markdown_artifacts(cell) for cell in raw_header]
        if len(header) < 3:
            raise ValueError(
                "Expected at least three columns: dataset, metric, and one model pair"
            )
        if header[0].casefold() != "dataset" or header[1].casefold() != "metric":
            raise ValueError(
                f"First two columns must be dataset and metric; found {header[:2]}"
            )

        pair_labels = header[2:]
        model_pairs: list[tuple[str, str]] = []
        for label in pair_labels:
            parts = label.split(" vs ")
            if len(parts) != 2 or not all(part.strip() for part in parts):
                raise ValueError(
                    f"Could not parse comparison column {label!r}; "
                    "expected 'model A vs model B'"
                )
            model_pairs.append((parts[0].strip(), parts[1].strip()))
        if len(model_pairs) != len(set(model_pairs)):
            raise ValueError("The CSV contains duplicate model-comparison columns")
        rows: list[tuple[str, str, list[float]]] = []
        expected_columns = len(header)
        for line_number, raw_row in enumerate(reader, start=2):
            if not raw_row or not any(cell.strip() for cell in raw_row):
                continue
            if len(raw_row) != expected_columns:
                raise ValueError(
                    f"Line {line_number} has {len(raw_row)} columns; "
                    f"expected {expected_columns}"
                )
            dataset = canonical_dataset(raw_row[0])
            metric = clean_markdown_artifacts(raw_row[1])
            rows.append((dataset, metric, [parse_p_value(x) for x in raw_row[2:]]))

    if not rows:
        raise ValueError(f"Input CSV has a header but no data rows: {path}")
    return model_pairs, rows


def read_better_models(
    path: Path,
) -> dict[tuple[str, str, str, str], str]:
    """Read winners from the long pairwise-significance CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Pairwise CSV is empty: {path}")

        original_by_clean = {
            clean_markdown_artifacts(name).casefold(): name
            for name in reader.fieldnames
        }
        required = ("dataset", "metric", "model_a", "model_b", "better_model")
        missing = [name for name in required if name not in original_by_clean]
        if missing:
            raise ValueError(
                f"Pairwise CSV {path} is missing columns {missing}. "
                f"Found: {reader.fieldnames}"
            )

        winners: dict[tuple[str, str, str, str], str] = {}
        for line_number, row in enumerate(reader, start=2):
            dataset = canonical_dataset(row[original_by_clean["dataset"]])
            metric = clean_markdown_artifacts(row[original_by_clean["metric"]])
            model_a = clean_markdown_artifacts(row[original_by_clean["model_a"]])
            model_b = clean_markdown_artifacts(row[original_by_clean["model_b"]])
            winner = clean_markdown_artifacts(
                row[original_by_clean["better_model"]]
            )
            key = (dataset, metric, model_a, model_b)
            if key in winners:
                raise ValueError(
                    f"Duplicate pairwise row at line {line_number}: {key}"
                )
            if winner and winner not in (model_a, model_b):
                raise ValueError(
                    f"Invalid better_model {winner!r} at line {line_number}; "
                    f"expected {model_a!r}, {model_b!r}, or an empty value"
                )
            winners[key] = winner

    if not winners:
        raise ValueError(f"Pairwise CSV has no data rows: {path}")
    return winners


def model_order(model_pairs: Sequence[tuple[str, str]]) -> list[str]:
    """Preserve the first-seen model order from the comparison columns."""
    models: list[str] = []
    for model_a, model_b in model_pairs:
        for model in (model_a, model_b):
            if model not in models:
                models.append(model)
    return models


def shortened_model_labels(models: Sequence[str]) -> list[str]:
    """Remove a shared prefix ending in '_' or '-' to keep axes readable."""
    prefix = os.path.commonprefix(list(models))
    cut = max(prefix.rfind("_"), prefix.rfind("-")) + 1
    if cut >= 3 and all(len(model) > cut for model in models):
        return [model[cut:] for model in models]
    return list(models)


def square_matrices(
    values: Sequence[float],
    model_pairs: Sequence[tuple[str, str]],
    models: Sequence[str],
    dataset: str,
    metric: str,
    better_models: dict[tuple[str, str, str, str], str],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build square p-value and directional-significance matrices.

    Directional states use the cell's perspective:
      1 = row model is significantly better
      0 = not significant
     -1 = column model is significantly better
    """
    if len(values) != len(model_pairs):
        raise ValueError("The number of p-values does not match the model pairs")
    indices = {model: index for index, model in enumerate(models)}
    p_matrix = np.full((len(models), len(models)), np.nan, dtype=float)
    direction_matrix = np.full((len(models), len(models)), np.nan, dtype=float)
    for p_value, (model_a, model_b) in zip(values, model_pairs):
        row = indices[model_a]
        column = indices[model_b]
        p_matrix[row, column] = p_value
        p_matrix[column, row] = p_value
        if not math.isfinite(p_value):
            continue
        if p_value >= alpha:
            direction_matrix[row, column] = 0.0
            direction_matrix[column, row] = 0.0
            continue

        winner_key = (dataset, metric, model_a, model_b)
        winner = better_models.get(winner_key)
        if winner is None:
            reverse_key = (dataset, metric, model_b, model_a)
            winner = better_models.get(reverse_key)
        if not winner:
            raise ValueError(
                "A significant comparison has no better_model entry: "
                f"dataset={dataset!r}, metric={metric!r}, "
                f"models={model_a!r} vs {model_b!r}, p={p_value:g}"
            )

        row_wins = winner == model_a
        direction_matrix[row, column] = 1.0 if row_wins else -1.0
        direction_matrix[column, row] = -1.0 if row_wins else 1.0
    return p_matrix, direction_matrix


def collect_plot_data(
    requested_datasets: Sequence[str],
    rows: Sequence[tuple[str, str, list[float]]],
) -> tuple[list[str], list[str], dict[tuple[str, str], list[float]]]:
    datasets = [canonical_dataset(value) for value in requested_datasets]
    available = sorted({name for name, _, _ in rows})
    missing = [dataset for dataset in datasets if dataset not in available]
    if missing:
        raise ValueError(
            f"Dataset(s) {missing} not found. Available datasets: {available}"
        )

    lookup: dict[tuple[str, str], list[float]] = {}
    metrics: list[str] = []
    for dataset, metric, values in rows:
        if dataset not in datasets:
            continue
        key = (dataset, metric)
        if key in lookup:
            raise ValueError(f"Duplicate row for dataset={dataset!r}, metric={metric!r}")
        lookup[key] = values
        if dataset == datasets[0]:
            metrics.append(metric)

    for dataset in datasets:
        dataset_metrics = {metric for name, metric, _ in rows if name == dataset}
        if dataset_metrics != set(metrics):
            raise ValueError(
                f"Dataset {dataset!r} has different metrics: {sorted(dataset_metrics)}; "
                f"expected {metrics}"
            )
    return datasets, metrics, lookup


def significant_outcomes(
    datasets: Sequence[str],
    metrics: Sequence[str],
    model_pairs: Sequence[tuple[str, str]],
    values_by_dataset_metric: dict[tuple[str, str], list[float]],
    better_models: dict[tuple[str, str, str, str], str],
    alpha: float,
) -> list[tuple[str, str, str, str]]:
    """Return (dataset, metric, winner, loser) for significant pairwise tests."""
    outcomes: list[tuple[str, str, str, str]] = []
    for dataset in datasets:
        for metric in metrics:
            values = values_by_dataset_metric[(dataset, metric)]
            for p_value, (model_a, model_b) in zip(values, model_pairs):
                if not math.isfinite(p_value) or p_value >= alpha:
                    continue
                winner = better_models.get((dataset, metric, model_a, model_b))
                if winner is None:
                    winner = better_models.get((dataset, metric, model_b, model_a))
                if not winner:
                    raise ValueError(
                        "A significant comparison has no better_model entry: "
                        f"dataset={dataset!r}, metric={metric!r}, "
                        f"models={model_a!r} vs {model_b!r}, p={p_value:g}"
                    )
                loser = model_b if winner == model_a else model_a
                outcomes.append((dataset, metric, winner, loser))
    return outcomes


def rank_models(
    models: Sequence[str],
    datasets: Sequence[str],
    metrics: Sequence[str],
    model_pairs: Sequence[tuple[str, str]],
    values_by_dataset_metric: dict[tuple[str, str], list[float]],
    outcomes: Sequence[tuple[str, str, str, str]],
) -> list[dict[str, object]]:
    """Create overall and per-dataset Copeland rankings."""
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, str, set[str]]] = [
        ("overall", "all", set(datasets))
    ] + [("dataset", dataset, {dataset}) for dataset in datasets]

    for scope, dataset_label, included_datasets in scopes:
        wins: defaultdict[str, int] = defaultdict(int)
        losses: defaultdict[str, int] = defaultdict(int)
        total: defaultdict[str, int] = defaultdict(int)

        for dataset in included_datasets:
            for metric in metrics:
                values = values_by_dataset_metric[(dataset, metric)]
                for p_value, (model_a, model_b) in zip(values, model_pairs):
                    if math.isfinite(p_value):
                        total[model_a] += 1
                        total[model_b] += 1

        for dataset, _metric, winner, loser in outcomes:
            if dataset in included_datasets:
                wins[winner] += 1
                losses[loser] += 1

        ordered = sorted(
            models,
            key=lambda model: (
                -(wins[model] - losses[model]),
                -wins[model],
                losses[model],
                model.casefold(),
            ),
        )
        previous_key: tuple[int, int, int] | None = None
        rank = 0
        for position, model in enumerate(ordered, start=1):
            ranking_key = (
                wins[model] - losses[model],
                wins[model],
                -losses[model],
            )
            if ranking_key != previous_key:
                rank = position
                previous_key = ranking_key
            significant = wins[model] + losses[model]
            rows.append(
                {
                    "scope": scope,
                    "dataset": dataset_label,
                    "rank": rank,
                    "model": model,
                    "significant_wins": wins[model],
                    "significant_losses": losses[model],
                    "net_significant_wins": wins[model] - losses[model],
                    "significant_comparisons": significant,
                    "total_comparisons": total[model],
                    "significance_rate": (
                        significant / total[model] if total[model] else math.nan
                    ),
                }
            )
    return rows


def write_ranking_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write ranking rows with a stable, analysis-friendly schema."""
    fieldnames = [
        "scope",
        "dataset",
        "rank",
        "model",
        "significant_wins",
        "significant_losses",
        "net_significant_wins",
        "significant_comparisons",
        "total_comparisons",
        "significance_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = dict(row)
            rate = output_row["significance_rate"]
            output_row["significance_rate"] = (
                f"{rate:.6f}" if isinstance(rate, float) and math.isfinite(rate) else ""
            )
            writer.writerow(output_row)


def print_overall_ranking(rows: Sequence[dict[str, object]]) -> None:
    """Print a compact overall ranking to stdout."""
    overall = [row for row in rows if row["scope"] == "overall"]
    print("Overall model ranking (significant wins - losses):")
    for row in overall:
        print(
            f"  {row['rank']:>2}. {row['model']}: "
            f"score={row['net_significant_wins']:+d}, "
            f"wins={row['significant_wins']}, losses={row['significant_losses']}"
        )


def plot_all_matrices(
    output_path: Path,
    datasets: Sequence[str],
    metrics: Sequence[str],
    model_pairs: Sequence[tuple[str, str]],
    models: Sequence[str],
    values_by_dataset_metric: dict[tuple[str, str], list[float]],
    better_models: dict[tuple[str, str, str, str], str],
    alpha: float,
    annotate_p_values: bool,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        len(datasets),
        len(metrics),
        figsize=(max(16.0, 3.55 * len(metrics)), max(10.0, 4.25 * len(datasets))),
        constrained_layout=True,
        squeeze=False,
    )

    # Values displayed from each cell's perspective:
    # -1 = column model wins; 0 = not significant; 1 = row model wins.
    color_map = ListedColormap(["#d95f02", "#e5e7eb", "#1a9850"])
    color_map.set_bad("white")
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], color_map.N)

    display_labels = shortened_model_labels(models)
    number_of_models = len(models)
    for dataset_index, dataset in enumerate(datasets):
        for metric_index, metric in enumerate(metrics):
            axis = axes[dataset_index, metric_index]
            p_matrix, direction_matrix = square_matrices(
                values=values_by_dataset_metric[(dataset, metric)],
                model_pairs=model_pairs,
                models=models,
                dataset=dataset,
                metric=metric,
                better_models=better_models,
                alpha=alpha,
            )
            axis.imshow(
                direction_matrix, cmap=color_map, norm=norm, aspect="equal"
            )

            axis.set_xticks(np.arange(number_of_models))
            axis.set_yticks(np.arange(number_of_models))
            if dataset_index == len(datasets) - 1:
                axis.set_xticklabels(display_labels, rotation=90, fontsize=7)
            else:
                axis.set_xticklabels([])
            if metric_index == 0:
                axis.set_yticklabels(display_labels, fontsize=7)
                axis.set_ylabel(dataset.capitalize(), fontweight="bold", labelpad=10)
            else:
                axis.set_yticklabels([])
            if dataset_index == 0:
                axis.set_title(metric, fontsize=12, fontweight="bold")

            axis.set_xticks(np.arange(-0.5, number_of_models, 1), minor=True)
            axis.set_yticks(np.arange(-0.5, number_of_models, 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.55)
            axis.tick_params(which="minor", bottom=False, left=False)
            axis.tick_params(which="major", length=2)

            if annotate_p_values:
                for row_index in range(number_of_models):
                    for column_index in range(number_of_models):
                        p_value = p_matrix[row_index, column_index]
                        if math.isfinite(p_value):
                            axis.text(
                                column_index,
                                row_index,
                                f"{p_value:.2g}",
                                ha="center",
                                va="center",
                                fontsize=4.2,
                                color=(
                                    "white"
                                    if direction_matrix[row_index, column_index] != 0
                                    else "#111827"
                                ),
                            )

    figure.suptitle(
        f"Pairwise significance matrices (p < {alpha:g})",
        fontsize=18,
        fontweight="bold",
    )
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.055, 1.0, 0.925))
    figure.legend(
        handles=[
            Patch(
                facecolor="#1a9850",
                label=f"Row model better (p < {alpha:g})",
            ),
            Patch(
                facecolor="#d95f02",
                label=f"Column model better (p < {alpha:g})",
            ),
            Patch(facecolor="#e5e7eb", label=f"Not significant (p â‰¥ {alpha:g})"),
            Patch(
                facecolor="white",
                edgecolor="#9ca3af",
                label="Diagonal / missing",
            ),
        ],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.006),
        frameon=False,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")
    if not args.csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    pairwise_csv = (
        args.pairwise_csv
        if args.pairwise_csv is not None
        else args.csv_path.with_name("pairwise_significance.csv")
    )
    if not pairwise_csv.is_file():
        raise FileNotFoundError(
            f"Pairwise CSV not found: {pairwise_csv}. The wide significance "
            "matrix contains p-values but cannot identify the better model. "
            "Provide --pairwise-csv with a CSV containing the better_model column."
        )

    model_pairs, rows = read_significance_csv(args.csv_path)
    better_models = read_better_models(pairwise_csv)
    models = model_order(model_pairs)
    datasets, metrics, values_by_dataset_metric = collect_plot_data(
        args.datasets, rows
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "significance_matrices.png"
    plot_all_matrices(
        output_path=output_path,
        datasets=datasets,
        metrics=metrics,
        model_pairs=model_pairs,
        models=models,
        values_by_dataset_metric=values_by_dataset_metric,
        better_models=better_models,
        alpha=args.alpha,
        annotate_p_values=args.annotate_p_values,
        dpi=args.dpi,
    )
    outcomes = significant_outcomes(
        datasets=datasets,
        metrics=metrics,
        model_pairs=model_pairs,
        values_by_dataset_metric=values_by_dataset_metric,
        better_models=better_models,
        alpha=args.alpha,
    )
    ranking_rows = rank_models(
        models=models,
        datasets=datasets,
        metrics=metrics,
        model_pairs=model_pairs,
        values_by_dataset_metric=values_by_dataset_metric,
        outcomes=outcomes,
    )
    ranking_path = (
        args.ranking_csv
        if args.ranking_csv is not None
        else args.output_dir / "model_significance_ranking.csv"
    )
    write_ranking_csv(ranking_path, ranking_rows)
    significant_count = sum(
        int(np.count_nonzero(np.isfinite(values) & (np.asarray(values) < args.alpha)))
        for values in values_by_dataset_metric.values()
    )
    finite_count = sum(
        int(np.count_nonzero(np.isfinite(values)))
        for values in values_by_dataset_metric.values()
    )
    print(
        f"Saved {output_path}: {len(datasets) * len(metrics)} square subplots, "
        f"{significant_count}/{finite_count} significant pairwise tests"
    )
    print(f"Saved {ranking_path}")
    print_overall_ranking(ranking_rows)


if __name__ == "__main__":
    main()