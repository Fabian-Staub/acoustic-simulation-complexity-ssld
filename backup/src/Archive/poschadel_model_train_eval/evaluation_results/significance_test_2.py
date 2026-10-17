#!/usr/bin/env python3
"""Paired significance tests for precomputed receiver-position metric CSVs.

Each input is a ``metrics_receiverPos_<MODEL>.csv`` export.  Every included
row is one recording/receiver position and already contains its metric scores.
The script pairs rows across models by dataset and receiver-position key, then
uses the position-level scores directly for:

* a paired bootstrap percentile confidence interval for the mean difference;
* a two-sided paired randomization (label-permutation) p-value; and
* a Holm correction within each dataset-and-metric matrix by default.

The correction family matters when many models are compared. With M models,
one square matrix contains M*(M-1)/2 hypotheses. Pooling all metrics into the
same family multiplies that number by the number of metrics and can make a
Monte Carlo permutation test too coarse to reject anything. Use
``--holm-family`` to select the intended family explicitly.

Examples:
    python compare_receiver_position_metrics.py \
        --csvs \
        evaluation_metrics/metrics_receiverPos_shoeBox_SH_Direct.csv \
        evaluation_metrics/metrics_receiverPos_shoeBox_SH_ISMOrder12.csv \
        --output-dir significance_results

    python compare_receiver_position_metrics.py \
        --csvs \
        "Direct=metrics_receiverPos_shoeBox_SH_Direct.csv" \
        "ISM12=metrics_receiverPos_shoeBox_SH_ISMOrder12.csv"

The input CSV filenames/paths are supplied through the required ``--csvs``
argument. Their command-line order is
preserved everywhere: model summaries, pairwise-comparison columns, heatmaps,
and the row/column order of every square significance matrix.

Outputs:
    model_metrics.csv
    pairwise_significance.csv
    significance_matrix.csv
    significance_heatmap.png
    significance_matrices.png
    significance_results.json

Requirements:
    pip install numpy matplotlib
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np


EPS = 1e-12
DEFAULT_METRICS = (
    "LE",
    "LR",
    "F15",
    "ER15",
    "ESDL",
    "count_MAE",
    "count_accuracy",
)
EXCLUDED_PLOT_METRICS = {"count_MAE", "count_accuracy"}
LOWER_IS_BETTER = {"LE", "ER15", "ESDL", "count_MAE"}
MODEL_FILENAME_PREFIX = "metrics_receiverPos_"
MATRIX_LABEL_COLUMN_HEADER = "Mic. | Order  | L | Corn | Rand"
MATRIX_LABEL_DEFINITIONS = (
    "Mic.=Mic. Sim.; Order=ISM Order; L=Diffuse Late reverb; "
    "Corn=Corner Displacement (m); "
    "Rand=Random image-source displacement (m)"
)
DISPLACEMENT_METERS_BY_TOKEN = {
    "025": "0.25",
    "05": "0.5",
    "1": "1",
    "3": "3",
    "5": "5",
    "10": "10",
}


@dataclass
class ModelData:
    label: str
    path: Path
    # dataset -> position key -> metric -> value
    scores: dict[str, dict[str, dict[str, float]]]


@dataclass
class ComparisonRow:
    dataset: str
    model_a: str
    model_b: str
    metric: str
    n_paired_positions: int
    estimate_a: float
    estimate_b: float
    difference_a_minus_b: float
    ci_low: float
    ci_high: float
    p_value: float
    p_value_holm: float = math.nan
    significant: bool = False
    better_model: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        metavar="[LABEL=]CSV",
        help=(
            "Full filenames/paths of two or more receiver-position metric "
            "CSVs, in the exact order they should appear in tables and plots. "
            "An optional LABEL= prefix overrides the label derived from a "
            "filename."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("significance_results"),
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=["checkpoint_dataset"],
        metavar="NAME",
        help="prediction_dataset value to skip; repeat to exclude more.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=DEFAULT_METRICS,
        default=list(DEFAULT_METRICS),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Paired receiver-position bootstrap samples for the 95%% CI.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10_000,
        help="Paired receiver-position label permutations per p-value.",
    )
    parser.add_argument(
        "--holm-family",
        choices=("dataset-metric", "dataset", "global", "none"),
        default="dataset-metric",
        help=(
            "Family used for Holm correction. 'dataset-metric' corrects the "
            "model pairs in each square significance matrix separately and "
            "is the default. 'dataset' also pools metrics; 'global' pools all "
            "tests; 'none' reports raw p-values as adjusted p-values."
        ),
    )
    parser.add_argument(
        "--resampling-batch-size",
        type=int,
        default=2048,
        help="Bootstrap/permutation replicates processed per batch.",
    )
    parser.add_argument(
        "--annotate-p-values",
        action="store_true",
        help=(
            "Deprecated compatibility option. Significance-plot cells are "
            "always annotated with signed mean differences; combined-row "
            "cells show signed net scores."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution for both significance plots (default: %(default)s).",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def split_model_argument(argument: str) -> tuple[str | None, Path]:
    label: str | None = None
    path_text = argument
    if "=" in argument:
        possible_label, possible_path = argument.split("=", 1)
        # Treat the prefix as a label only when the complete argument is not
        # itself an existing path. This keeps filenames containing '=' valid.
        if possible_label.strip() and not Path(argument).expanduser().exists():
            label = possible_label.strip()
            path_text = possible_path
    return label, Path(path_text).expanduser().resolve()


def label_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith(MODEL_FILENAME_PREFIX):
        stem = stem[len(MODEL_FILENAME_PREFIX) :]
    if not stem:
        raise ValueError(f"Could not derive a model label from {path.name!r}")
    return stem


def displacement_from_token(token: str, path: Path) -> str:
    try:
        return DISPLACEMENT_METERS_BY_TOKEN[token]
    except KeyError as error:
        raise ValueError(
            f"Unknown displacement token {token!r} in {path.name!r}. "
            "Add its value to DISPLACEMENT_METERS_BY_TOKEN."
        ) from error


def matrix_parameter_label(path: Path) -> str:
    """Convert an experiment filename into the compact matrix-table row."""
    stem = path.stem
    if stem.startswith(MODEL_FILENAME_PREFIX):
        stem = stem[len(MODEL_FILENAME_PREFIX) :]
    # Ignore copy suffixes added by file managers, e.g. ``...(1).csv``.
    stem = re.sub(r"\(\d+\)$", "", stem)

    microphone_match = re.search(r"_(SH|em32)_(?:Direct|ISMOrder)", stem)
    if microphone_match is None:
        raise ValueError(
            f"Cannot determine microphone simulation from {path.name!r}; "
            "expected '_SH_' or '_em32_' before Direct/ISMOrder."
        )
    microphone_simulation = microphone_match.group(1)

    order_match = re.search(r"_ISMOrder(\d+)(?:-Late)?$", stem)
    if order_match is not None:
        ism_order = order_match.group(1)
    elif re.search(r"_Direct(?:-Late)?$", stem):
        ism_order = "Direct"
    else:
        raise ValueError(
            f"Cannot determine ISM order from {path.name!r}; expected "
            "'_Direct', '_ISMOrder2', or '_ISMOrder12', optionally '-Late'."
        )

    diffuse_late_reverb = "y" if stem.endswith("-Late") else "n"

    corner_match = re.search(r"non-shoeBox_disp-corners-([^_]+)_", stem)
    corner_displacement = (
        displacement_from_token(corner_match.group(1), path)
        if corner_match is not None
        else "0"
    )

    random_match = re.search(r"_rand-disp-([^_]+)_", stem)
    random_displacement = (
        displacement_from_token(random_match.group(1), path)
        if random_match is not None
        else "0"
    )

    # Fixed-width fields plus a monospaced tick font make the table columns
    # line up across every row and column label.
    return (
        f"{microphone_simulation:<4} | "
        f"{ism_order:<6} | "
        f"{diffuse_late_reverb:^1} | "
        f"{corner_displacement:>4} | "
        f"{random_displacement:>4}"
    )


def collect_model_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    # argparse preserves the order of values passed to --csvs. Do not sort this
    # list: it is the authoritative display order for all downstream outputs.
    raw_specs = [split_model_argument(value) for value in args.csvs]

    specs: list[tuple[str, Path]] = []
    seen_paths: set[Path] = set()
    seen_labels: set[str] = set()
    for explicit_label, path in raw_specs:
        if path in seen_paths:
            raise ValueError(f"CSV path was specified more than once: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Metric CSV not found: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got {path.name!r}")
        label = explicit_label or label_from_path(path)
        if label in seen_labels:
            raise ValueError(
                f"Duplicate model label {label!r}; use LABEL=PATH to disambiguate"
            )
        seen_paths.add(path)
        seen_labels.add(label)
        specs.append((label, path))

    if len(specs) < 2:
        raise ValueError(
            "--csvs requires at least two metric CSV filenames/paths."
        )
    return specs


def normalized_column(fieldnames: Sequence[str], wanted: str) -> str | None:
    for fieldname in fieldnames:
        normalized = fieldname.replace("\\", "").strip().strip("*").lower()
        if normalized == wanted.lower():
            return fieldname
    return None


def require_column(fieldnames: Sequence[str], wanted: str) -> str:
    column = normalized_column(fieldnames, wanted)
    if column is None:
        raise ValueError(f"Missing column {wanted!r}; found {list(fieldnames)}")
    return column


def finite_float(value: str, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value in {context}: {value!r}") from error
    return parsed


def position_key(row: dict[str, str], columns: dict[str, str | None], context: str) -> str:
    receiver_key_column = columns["receiver_position_key"]
    if receiver_key_column:
        receiver_key = row.get(receiver_key_column, "").strip()
        if receiver_key:
            return receiver_key

    source_column = columns["prediction_source_csv"]
    position_column = columns["receiver_position"]
    source = row.get(source_column, "").strip() if source_column else ""
    position = row.get(position_column, "").strip() if position_column else ""
    if source and position:
        return json.dumps((source, position), ensure_ascii=False, separators=(",", ":"))
    raise ValueError(
        f"No usable receiver_position_key or prediction_source_csv/"
        f"receiver_position pair in {context}"
    )


def load_model(
    label: str,
    path: Path,
    metrics: Sequence[str],
    excluded_datasets: set[str],
) -> ModelData:
    scores: dict[str, dict[str, dict[str, float]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = reader.fieldnames
        dataset_column = require_column(fieldnames, "prediction_dataset")
        metric_columns = {metric: require_column(fieldnames, metric) for metric in metrics}
        columns = {
            "receiver_position_key": normalized_column(fieldnames, "receiver_position_key"),
            "prediction_source_csv": normalized_column(fieldnames, "prediction_source_csv"),
            "receiver_position": normalized_column(fieldnames, "receiver_position"),
            "status": normalized_column(fieldnames, "status"),
        }

        for row_number, row in enumerate(reader, start=2):
            dataset = row.get(dataset_column, "").strip()
            if not dataset:
                raise ValueError(f"Empty prediction_dataset in {path} row {row_number}")
            if dataset in excluded_datasets:
                continue
            status_column = columns["status"]
            if status_column:
                status = row.get(status_column, "").strip()
                if status and status.lower() != "success":
                    raise ValueError(
                        f"Non-success status {status!r} in {path} row {row_number}"
                    )
            key = position_key(row, columns, f"{path} row {row_number}")
            dataset_scores = scores.setdefault(dataset, {})
            if key in dataset_scores:
                raise ValueError(
                    f"Duplicate receiver-position key in {path} row {row_number}: {key}"
                )
            dataset_scores[key] = {
                metric: finite_float(
                    row.get(column, ""), f"{path} row {row_number}, column {metric}"
                )
                for metric, column in metric_columns.items()
            }

    if not scores:
        raise ValueError(f"No included receiver-position rows found in {path}")
    return ModelData(label=label, path=path, scores=scores)


def validate_dataset_sets(models: Sequence[ModelData]) -> list[str]:
    reference = set(models[0].scores)
    for model in models[1:]:
        current = set(model.scores)
        if current != reference:
            raise ValueError(
                f"Dataset mismatch for {model.label}: "
                f"missing={sorted(reference-current)}, extra={sorted(current-reference)}"
            )
    return sorted(reference)


def validate_positions(models: Sequence[ModelData], dataset: str) -> list[str]:
    reference = set(models[0].scores[dataset])
    for model in models[1:]:
        current = set(model.scores[dataset])
        if current != reference:
            raise ValueError(
                f"Receiver-position mismatch for {model.label} in {dataset}: "
                f"missing={sorted(reference-current)[:5]}, "
                f"extra={sorted(current-reference)[:5]}"
            )
    if not reference:
        raise ValueError(f"Dataset {dataset!r} has no receiver positions")
    return sorted(reference)


def finite_pairs(values_a: np.ndarray, values_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values_a) & np.isfinite(values_b)
    return values_a[valid], values_b[valid]


def finite_percentile(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    low, high = np.percentile(values, (2.5, 97.5))
    return float(low), float(high)


def compare_pair(
    model_a: ModelData,
    model_b: ModelData,
    dataset: str,
    positions: Sequence[str],
    metrics: Sequence[str],
    bootstrap_samples: int,
    permutations: int,
    batch_size: int,
    rng: np.random.Generator,
) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    for metric in metrics:
        values_a = np.asarray(
            [model_a.scores[dataset][position][metric] for position in positions],
            dtype=np.float64,
        )
        values_b = np.asarray(
            [model_b.scores[dataset][position][metric] for position in positions],
            dtype=np.float64,
        )
        values_a, values_b = finite_pairs(values_a, values_b)
        count = len(values_a)
        if count < 2:
            raise ValueError(
                f"Need at least two finite paired values for {dataset}, {metric}, "
                f"{model_a.label} vs {model_b.label}; found {count}"
            )

        differences = values_a - values_b
        observed = float(np.mean(differences))
        bootstrap_differences = np.empty(bootstrap_samples, dtype=np.float64)
        effective_batch = min(batch_size, max(1, 250_000 // count))
        for start in range(0, bootstrap_samples, effective_batch):
            stop = min(start + effective_batch, bootstrap_samples)
            indices = rng.integers(0, count, size=(stop - start, count))
            bootstrap_differences[start:stop] = differences[indices].mean(axis=1)

        extreme = 0
        valid_permutations = 0
        for start in range(0, permutations, effective_batch):
            stop = min(start + effective_batch, permutations)
            signs = rng.integers(0, 2, size=(stop - start, count), dtype=np.int8)
            signs = signs * 2 - 1
            permuted = (signs * differences).mean(axis=1)
            valid = np.isfinite(permuted)
            valid_permutations += int(np.count_nonzero(valid))
            extreme += int(
                np.count_nonzero(valid & (np.abs(permuted) >= abs(observed) - EPS))
            )

        ci_low, ci_high = finite_percentile(bootstrap_differences)
        p_value = (
            math.nan
            if valid_permutations == 0
            else (extreme + 1.0) / (valid_permutations + 1.0)
        )
        rows.append(
            ComparisonRow(
                dataset=dataset,
                model_a=model_a.label,
                model_b=model_b.label,
                metric=metric,
                n_paired_positions=count,
                estimate_a=float(np.mean(values_a)),
                estimate_b=float(np.mean(values_b)),
                difference_a_minus_b=observed,
                ci_low=ci_low,
                ci_high=ci_high,
                p_value=p_value,
            )
        )
    return rows


def correction_family_key(row: ComparisonRow, family: str) -> tuple[str, ...]:
    if family == "dataset-metric":
        return row.dataset, row.metric
    if family == "dataset":
        return (row.dataset,)
    if family == "global":
        return ("all-datasets-and-metrics",)
    if family == "none":
        return row.dataset, row.metric, row.model_a, row.model_b
    raise ValueError(f"Unknown Holm family: {family!r}")


def apply_holm_correction(
    rows: list[ComparisonRow],
    alpha: float,
    family: str,
    permutations: int,
) -> None:
    """Adjust p-values inside explicit hypothesis families.

    The default family is one dataset/metric square matrix. This matches the
    plotted inferential question: for a fixed dataset and metric, which model
    pairs differ? Broader families remain available through --holm-family.
    """
    for row in rows:
        row.p_value_holm = math.nan
        row.significant = False
        row.better_model = ""

    families: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if math.isfinite(row.p_value):
            families[correction_family_key(row, family)].append(index)

    if family == "none":
        for indices in families.values():
            index = indices[0]
            rows[index].p_value_holm = rows[index].p_value
    else:
        for indices in families.values():
            ordered = sorted(indices, key=lambda index: rows[index].p_value)
            running_max = 0.0
            number_of_tests = len(ordered)
            for rank, index in enumerate(ordered):
                adjusted = min(
                    1.0,
                    rows[index].p_value * (number_of_tests - rank),
                )
                running_max = max(running_max, adjusted)
                rows[index].p_value_holm = running_max

    # A Monte Carlo p-value with the plus-one correction cannot be smaller
    # than 1/(B+1). Holm's smallest ordered adjusted p is therefore bounded by
    # family_size/(B+1). Report when the requested setup cannot possibly cross
    # alpha, rather than silently drawing an all-gray matrix.
    if family != "none":
        p_floor = 1.0 / (permutations + 1.0)
        impossible = [
            (key, len(indices), min(1.0, len(indices) * p_floor))
            for key, indices in families.items()
            if len(indices) * p_floor >= alpha
        ]
        if impossible:
            largest_size = max(size for _, size, _ in impossible)
            required = math.floor(largest_size / alpha)
            print(
                "WARNING: Monte Carlo resolution makes significance impossible "
                f"in {len(impossible)} Holm family/families. The largest family "
                f"has {largest_size} tests; with {permutations:,} permutations, "
                f"its smallest possible first Holm p is "
                f"{largest_size}/({permutations}+1) = "
                f"{largest_size * p_floor:.6g}. Use at least {required:,} "
                "permutations, or choose a scientifically appropriate narrower "
                "--holm-family.",
                flush=True,
            )

    for row in rows:
        row.significant = bool(
            math.isfinite(row.p_value_holm) and row.p_value_holm < alpha
        )
        if not row.significant or row.difference_a_minus_b == 0:
            continue
        if row.metric in LOWER_IS_BETTER:
            row.better_model = (
                row.model_a if row.difference_a_minus_b < 0 else row.model_b
            )
        else:
            row.better_model = (
                row.model_a if row.difference_a_minus_b > 0 else row.model_b
            )


def json_number(value: float) -> float | None:
    return None if not math.isfinite(value) else float(value)


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pair_label(model_a: str, model_b: str) -> str:
    return f"{model_a} vs {model_b}"


def write_significance_matrix(
    path: Path,
    comparison_rows: Sequence[ComparisonRow],
    datasets: Sequence[str],
    metrics: Sequence[str],
    model_pairs: Sequence[tuple[str, str]],
) -> None:
    pair_columns = [pair_label(*pair) for pair in model_pairs]
    lookup = {
        (row.dataset, row.metric, row.model_a, row.model_b): row
        for row in comparison_rows
    }
    matrix_rows: list[dict[str, object]] = []
    for dataset in datasets:
        for metric in metrics:
            output: dict[str, object] = {"dataset": dataset, "metric": metric}
            for pair, column in zip(model_pairs, pair_columns):
                row = lookup[(dataset, metric, *pair)]
                output[column] = json_number(row.p_value_holm)
            matrix_rows.append(output)
    write_csv(path, matrix_rows, ("dataset", "metric", *pair_columns))


def plot_significance_heatmap(
    path: Path,
    comparison_rows: Sequence[ComparisonRow],
    datasets: Sequence[str],
    metrics: Sequence[str],
    model_pairs: Sequence[tuple[str, str]],
    dpi: int,
    p_value_label: str,
) -> None:
    lookup = {
        (row.dataset, row.metric, row.model_a, row.model_b): row
        for row in comparison_rows
    }
    row_keys = [(dataset, metric) for dataset in datasets for metric in metrics]
    values = np.full((len(row_keys), len(model_pairs)), np.nan)
    annotations = [["NA" for _ in model_pairs] for _ in row_keys]
    for row_index, (dataset, metric) in enumerate(row_keys):
        for column_index, pair in enumerate(model_pairs):
            row = lookup[(dataset, metric, *pair)]
            if not math.isfinite(row.p_value_holm):
                continue
            if row.significant and row.better_model:
                direction = 1.0 if row.better_model == pair[0] else -1.0
                values[row_index, column_index] = direction * -math.log10(
                    max(row.p_value_holm, np.finfo(float).tiny)
                )
                winner = "A" if direction > 0 else "B"
                annotations[row_index][column_index] = (
                    f"{winner}*\nÎ”={row.difference_a_minus_b:+.3g}"
                )
            else:
                values[row_index, column_index] = 0.0
                annotations[row_index][column_index] = (
                    f"ns\nÎ”={row.difference_a_minus_b:+.3g}"
                )

    finite = np.abs(values[np.isfinite(values)])
    limit = max(1.5, float(finite.max(initial=0.0)))
    figure, axis = plt.subplots(
        figsize=(max(8.0, 2.4 * len(model_pairs) + 4.0), max(5.0, 0.56 * len(row_keys) + 2.2)),
        constrained_layout=True,
    )
    image = axis.imshow(
        np.ma.masked_invalid(values),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    axis.set_xticks(range(len(model_pairs)))
    axis.set_xticklabels([pair_label(*pair) for pair in model_pairs], rotation=30, ha="right")
    axis.set_yticks(range(len(row_keys)))
    axis.set_yticklabels([f"{dataset} | {metric}" for dataset, metric in row_keys])
    axis.set_title("Per-dataset paired significance (receiver-position scores)")
    axis.set_xlabel("Model A vs model B")
    axis.set_ylabel("Dataset and metric")
    for boundary in range(len(metrics), len(row_keys), len(metrics)):
        axis.axhline(boundary - 0.5, color="black", linewidth=1.2)
    for row_index in range(len(row_keys)):
        for column_index in range(len(model_pairs)):
            value = values[row_index, column_index]
            color = "white" if math.isfinite(value) and abs(value) > limit * 0.55 else "black"
            axis.text(
                column_index,
                row_index,
                annotations[row_index][column_index],
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.75)
    colorbar.set_label(
        f"Signed -log10({p_value_label}): red = A better, blue = B better"
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def square_significance_matrices(
    comparison_rows: Sequence[ComparisonRow],
    dataset: str,
    metric: str,
    models: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build p-value, directional, and signed mean-difference matrices.

    Direction is expressed from each cell's perspective:
      1 = row model is significantly better
      0 = not significant
     -1 = column model is significantly better
    """
    indices = {model: index for index, model in enumerate(models)}
    p_matrix = np.full((len(models), len(models)), np.nan, dtype=np.float64)
    direction_matrix = np.full(
        (len(models), len(models)), np.nan, dtype=np.float64
    )
    difference_matrix = np.full(
        (len(models), len(models)), np.nan, dtype=np.float64
    )
    matching_rows = [
        row
        for row in comparison_rows
        if row.dataset == dataset and row.metric == metric
    ]
    expected_pairs = len(models) * (len(models) - 1) // 2
    if len(matching_rows) != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} comparisons for {dataset}, {metric}; "
            f"found {len(matching_rows)}"
        )

    for row in matching_rows:
        row_index = indices[row.model_a]
        column_index = indices[row.model_b]
        p_matrix[row_index, column_index] = row.p_value_holm
        p_matrix[column_index, row_index] = row.p_value_holm
        # Each cell is expressed from the row model's perspective.
        difference_matrix[row_index, column_index] = row.difference_a_minus_b
        difference_matrix[column_index, row_index] = -row.difference_a_minus_b
        if not math.isfinite(row.p_value_holm):
            continue
        if not row.significant:
            direction_matrix[row_index, column_index] = 0.0
            direction_matrix[column_index, row_index] = 0.0
            continue
        if not row.better_model:
            raise ValueError(
                f"Significant comparison has no better_model: {dataset}, "
                f"{metric}, {row.model_a} vs {row.model_b}"
            )
        model_a_wins = row.better_model == row.model_a
        direction_matrix[row_index, column_index] = 1.0 if model_a_wins else -1.0
        direction_matrix[column_index, row_index] = -1.0 if model_a_wins else 1.0
    return p_matrix, direction_matrix, difference_matrix


def combined_dataset_score_matrix(
    comparison_rows: Sequence[ComparisonRow],
    datasets: Sequence[str],
    metric: str,
    models: Sequence[str],
) -> np.ndarray:
    """Sum significant directions across datasets from each cell's perspective.

    A significant row-model win contributes +1, a significant row-model loss
    contributes -1, and a non-significant comparison contributes 0. A combined
    cell is missing when any constituent dataset cell is missing.
    """
    dataset_matrices = [
        square_significance_matrices(
            comparison_rows,
            dataset,
            metric,
            models,
        )[1]
        for dataset in datasets
    ]
    stacked = np.stack(dataset_matrices)
    all_available = np.all(np.isfinite(stacked), axis=0)
    combined = np.sum(np.where(np.isfinite(stacked), stacked, 0.0), axis=0)
    combined[~all_available] = np.nan
    return combined


def plot_significance_matrices(
    path: Path,
    comparison_rows: Sequence[ComparisonRow],
    datasets: Sequence[str],
    metrics: Sequence[str],
    models: Sequence[str],
    model_display_labels: Sequence[str],
    alpha: float,
    annotate_p_values: bool,
    dpi: int,
    p_value_label: str,
) -> None:
    """Plot dataset matrices plus a net significant-dataset score row."""
    number_of_plot_rows = len(datasets) + 1
    if len(model_display_labels) != len(models):
        raise ValueError("Model labels and matrix display labels have different lengths")
    display_labels = list(model_display_labels)
    number_of_models = len(models)

    # Scale every square panel with the model count so each annotated cell has
    # enough physical space. Cap the panel size to avoid impractically large
    # image files for unusually large model collections.
    matrix_panel_inches = max(5.6, min(13.0, 0.65 * number_of_models))
    figure, axes = plt.subplots(
        number_of_plot_rows,
        len(metrics),
        figsize=(
            matrix_panel_inches * len(metrics),
            matrix_panel_inches * number_of_plot_rows,
        ),
        constrained_layout=True,
        squeeze=False,
    )

    # The combined row ranges from -N to +N for N datasets. Orange means the
    # row model has a net loss, green a net win, and stronger shades mean a
    # larger magnitude. Ordinary dataset rows naturally use only -1, 0, +1.
    maximum_score = len(datasets)
    orange_map = plt.get_cmap("Oranges")
    green_map = plt.get_cmap("Greens")
    orange_colors = [
        orange_map(value)
        for value in np.linspace(0.88, 0.42, maximum_score)
    ]
    green_colors = [
        green_map(value)
        for value in np.linspace(0.42, 0.88, maximum_score)
    ]
    score_colors = [*orange_colors, "#e5e7eb", *green_colors]
    color_map = ListedColormap(score_colors)
    color_map.set_bad("white")
    norm = BoundaryNorm(
        np.arange(-maximum_score - 0.5, maximum_score + 1.5, 1.0),
        color_map.N,
    )
    plot_rows: list[str | None] = [*datasets, None]
    for plot_row_index, dataset in enumerate(plot_rows):
        is_combined = dataset is None
        for metric_index, metric in enumerate(metrics):
            axis = axes[plot_row_index, metric_index]
            if is_combined:
                p_matrix = np.full(
                    (number_of_models, number_of_models),
                    np.nan,
                    dtype=np.float64,
                )
                direction_matrix = combined_dataset_score_matrix(
                    comparison_rows,
                    datasets,
                    metric,
                    models,
                )
            else:
                assert dataset is not None
                p_matrix, direction_matrix, difference_matrix = square_significance_matrices(
                    comparison_rows,
                    dataset,
                    metric,
                    models,
                )
            axis.imshow(
                direction_matrix,
                cmap=color_map,
                norm=norm,
                aspect="equal",
            )
            axis.set_xticks(np.arange(number_of_models))
            axis.set_yticks(np.arange(number_of_models))
            if plot_row_index == number_of_plot_rows - 1:
                axis.set_xticklabels(
                    display_labels,
                    rotation=90,
                    fontsize=7,
                    fontfamily="monospace",
                )
            else:
                axis.set_xticklabels([])
            if metric_index == 0:
                axis.set_yticklabels(
                    display_labels,
                    fontsize=7,
                    fontfamily="monospace",
                )
                # Right-align the fixed-width header with the fixed-width tick
                # rows, creating an actual table header for each matrix row.
                axis.text(
                    -0.02,
                    1.015,
                    MATRIX_LABEL_COLUMN_HEADER,
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    fontfamily="monospace",
                    fontweight="bold",
                    clip_on=False,
                )
                row_label = (
                    "Combined\n(net significant datasets)"
                    if is_combined
                    else str(dataset)
                )
                axis.set_ylabel(row_label, fontweight="bold", labelpad=10)
            else:
                axis.set_yticklabels([])
            if plot_row_index == 0:
                axis.set_title(metric, fontsize=12, fontweight="bold")

            axis.set_xticks(np.arange(-0.5, number_of_models, 1), minor=True)
            axis.set_yticks(np.arange(-0.5, number_of_models, 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.55)
            axis.tick_params(which="minor", bottom=False, left=False)
            axis.tick_params(which="major", length=2)

            for row_index in range(number_of_models):
                for column_index in range(number_of_models):
                    if is_combined:
                        displayed_value = direction_matrix[row_index, column_index]
                        if not math.isfinite(displayed_value):
                            continue
                        annotation = f"{int(displayed_value):+d}"
                    else:
                        displayed_value = difference_matrix[row_index, column_index]
                        if not math.isfinite(displayed_value):
                            continue
                        annotation = f"{displayed_value:+.2g}"
                    axis.text(
                        column_index,
                        row_index,
                        annotation,
                        ha="center",
                        va="center",
                        fontsize=7.0,
                        color=(
                            "white"
                            if abs(direction_matrix[row_index, column_index]) >= 2
                            else "#111827"
                        ),
                    )

    figure.suptitle(
        f"Pairwise significance matrices ({p_value_label} < {alpha:g}); "
        "combined row = net significant datasets",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.955,
        MATRIX_LABEL_DEFINITIONS,
        ha="center",
        va="top",
        fontsize=9,
        fontfamily="monospace",
    )
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.055, 1.0, 0.89))
    legend_scores = [
        *range(maximum_score, 0, -1),
        0,
        *range(-1, -maximum_score - 1, -1),
    ]
    legend_handles = [
        Patch(
            facecolor=score_colors[score + maximum_score],
            label=(
                f"+{score} net row-model dataset "
                f"win{'s' if score != 1 else ''}"
                if score > 0
                else f"{score} net row-model dataset "
                f"loss{'es' if score != -1 else ''}"
                if score < 0
                else "0 net dataset wins"
            ),
        )
        for score in legend_scores
    ]
    legend_handles.append(
        Patch(
            facecolor="white",
            edgecolor="#9ca3af",
            label="Diagonal / missing",
        )
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(8, len(legend_handles)),
        bbox_to_anchor=(0.5, 0.006),
        frameon=False,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1 or args.permutations < 1:
        raise ValueError("--bootstrap-samples and --permutations must be positive")
    if args.resampling_batch_size < 1:
        raise ValueError("--resampling-batch-size must be positive")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    specs = collect_model_specs(args)
    excluded = {value.strip() for value in args.exclude_dataset if value.strip()}
    models = [load_model(label, path, args.metrics, excluded) for label, path in specs]
    datasets = validate_dataset_sets(models)
    labels = [model.label for model in models]
    matrix_display_labels = [matrix_parameter_label(model.path) for model in models]
    model_pairs = list(combinations(labels, 2))
    rng = np.random.default_rng(args.seed)
    comparisons: list[ComparisonRow] = []
    metric_rows: list[dict[str, object]] = []
    dataset_payload: dict[str, object] = {}

    for dataset in datasets:
        positions = validate_positions(models, dataset)
        print(f"Dataset {dataset}: {len(positions)} paired receiver positions.")
        dataset_comparisons: list[ComparisonRow] = []
        for model_a, model_b in combinations(models, 2):
            print(f"  Comparing {model_a.label} vs {model_b.label}...")
            dataset_comparisons.extend(
                compare_pair(
                    model_a,
                    model_b,
                    dataset,
                    positions,
                    args.metrics,
                    args.bootstrap_samples,
                    args.permutations,
                    args.resampling_batch_size,
                    rng,
                )
            )
        comparisons.extend(dataset_comparisons)

        models_payload: dict[str, object] = {}
        for model in models:
            metrics_payload: dict[str, object] = {}
            for metric in args.metrics:
                values = np.asarray(
                    [model.scores[dataset][position][metric] for position in positions]
                )
                finite = values[np.isfinite(values)]
                mean = float(np.mean(finite)) if len(finite) else math.nan
                metrics_payload[metric] = {
                    "mean_receiver_position_score": json_number(mean),
                    "number_of_finite_positions": int(len(finite)),
                }
                metric_rows.append(
                    {
                        "dataset": dataset,
                        "model": model.label,
                        "metric": metric,
                        "mean_receiver_position_score": mean,
                        "number_of_finite_positions": len(finite),
                    }
                )
            models_payload[model.label] = {
                "metric_file": str(model.path),
                "matrix_parameter_label": matrix_parameter_label(model.path),
                "metrics": metrics_payload,
            }
        dataset_payload[dataset] = {
            "number_of_paired_receiver_positions": len(positions),
            "models": models_payload,
        }

    apply_holm_correction(
        comparisons,
        args.alpha,
        args.holm_family,
        args.permutations,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dicts = [asdict(row) for row in comparisons]
    write_csv(
        args.output_dir / "model_metrics.csv",
        metric_rows,
        (
            "dataset",
            "model",
            "metric",
            "mean_receiver_position_score",
            "number_of_finite_positions",
        ),
    )
    write_csv(
        args.output_dir / "pairwise_significance.csv",
        comparison_dicts,
        tuple(ComparisonRow.__dataclass_fields__),
    )
    write_significance_matrix(
        args.output_dir / "significance_matrix.csv",
        comparisons,
        datasets,
        args.metrics,
        model_pairs,
    )
    p_value_label = "raw p" if args.holm_family == "none" else "Holm p"
    plot_metrics = [
        metric for metric in args.metrics if metric not in EXCLUDED_PLOT_METRICS
    ]
    if plot_metrics:
        plot_significance_heatmap(
            args.output_dir / "significance_heatmap.png",
            comparisons,
            datasets,
            plot_metrics,
            model_pairs,
            args.dpi,
            p_value_label,
        )
        plot_significance_matrices(
            args.output_dir / "significance_matrices.png",
            comparisons,
            datasets,
            plot_metrics,
            labels,
            matrix_display_labels,
            args.alpha,
            args.annotate_p_values,
            args.dpi,
            p_value_label,
        )
    else:
        print("No plottable metrics selected; significance plots were skipped.")

    payload = {
        "method": {
            "input": "precomputed per-receiver-position metric scores",
            "resampling_unit": "receiver position",
            "point_estimate": "unweighted mean of receiver-position scores",
            "bootstrap_samples": args.bootstrap_samples,
            "permutations": args.permutations,
            "confidence_interval": "paired receiver-position bootstrap percentile 95%",
            "p_value": "two-sided paired receiver-position label-permutation test",
            "multiple_testing": (
                "No multiplicity correction"
                if args.holm_family == "none"
                else f"Holm correction with family={args.holm_family}"
            ),
            "holm_family": args.holm_family,
            "minimum_monte_carlo_p_value": 1.0 / (args.permutations + 1.0),
            "combined_matrix_score": (
                "+1 for each dataset where the row model is significantly "
                "better, -1 where significantly worse, and 0 where not "
                "significant; summed across all datasets"
            ),
            "excluded_datasets": sorted(excluded),
            "alpha": args.alpha,
            "seed": args.seed,
        },
        "number_of_models": len(models),
        "number_of_datasets": len(datasets),
        "datasets": dataset_payload,
        "pairwise_comparisons": [
            {
                key: json_number(value) if isinstance(value, float) else value
                for key, value in row.items()
            }
            for row in comparison_dicts
        ],
    }
    with (args.output_dir / "significance_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)

    print(f"Saved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()