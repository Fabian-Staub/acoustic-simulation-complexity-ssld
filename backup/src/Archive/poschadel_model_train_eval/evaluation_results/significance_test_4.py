#!/usr/bin/env python3
"""Jackknife significance analysis for raw prediction CSVs from eval_5_fast.py.

Each ``--csvs`` input is one model's augmented prediction CSV.  Metrics are
recomputed from the reference and predicted ACCDOA vectors.  A unique room and
receiver-position pair parsed from ``sample_id`` is one independent sample;
for example, ``R1800_P00`` becomes room ``R1800`` and position ``P00``.  Room
names are not restricted to ``R`` followed by a number.

Overall metrics and metrics conditioned on exactly 1, 2, or 3 active
ground-truth sources are reported for every dataset. ``checkpoint_dataset`` is
reported as ``simulated-dataset`` and is metrics-only: it is never included in
pairwise significance tests, multiple-comparison correction, or plots.

For every model metric and every paired model difference, this script uses
the delete-one jackknife construction described by Mesaros et al., "Sound
Event Detection in the DCASE 2017 Challenge": full-sample and leave-one-out
estimates, pseudo-values, the bias-corrected jackknife estimate, its standard
error, and a Student-t confidence interval with N-1 degrees of freedom. The
standard error uses the standard delete-one jackknife scaling; the printed
leave-one-out form of Equation (6) in that paper is missing a factor of
(N-1)^2 and would make standard errors N-1 times too small.

A paired difference is
Raw two-sided p-values are obtained from the jackknife t statistic. By default,
Holm's method corrects all model-pair tests within each dataset/metric family;
the correction method and family scope are configurable on the command line.

Examples:
    python compare_receiver_position_metrics.py \
        --csvs \
        evaluation_results/predictions_shoeBox_SH_Direct.csv \
        evaluation_results/predictions_shoeBox_SH_ISMOrder12.csv \
        --output-dir significance_results

    python compare_receiver_position_metrics.py \
        --csvs \
        "Direct=predictions_shoeBox_SH_Direct.csv" \
        "ISM12=predictions_shoeBox_SH_ISMOrder12.csv"

The input CSV filenames/paths are supplied through the required ``--csvs``
argument. Their command-line order is
preserved everywhere: model summaries, pairwise-comparison columns, heatmaps,
and the row/column order of every square significance matrix.

Outputs:
    model_metrics.csv
    pairwise_significance.csv
    significance_matrix.csv
    significance_heatmap.png
    significance_matrices_a4_<METRIC>.png
    significance_results.json

Requirements:
    pip install numpy scipy matplotlib
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy.stats import t as student_t
from scipy.optimize import linear_sum_assignment


EPS = 1e-12
SPATIAL_THRESHOLD_DEG = 15.0
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
MODEL_FILENAME_PREFIXES = ("metrics_receiverPos_", "predictions_")
MATRIX_LABEL_COLUMN_HEADER = (
    f"{'Mic.':<4} | "
    f"{'Order':<5} | "
    f"{'L':^1} | "
    f"{'Corn':>4} | "
    f"{'Rand':>4}"
)
MATRIX_LABEL_DEFINITIONS = (
    "Mic. = Mic. Sim. | Order = ISM Order | L = Diffuse Late reverb | Corn = Corner Displacement (m) | Rand = Random image-source displacement (m)"
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
    # dataset -> room/receiver-position key -> additive sufficient statistics
    samples: dict[str, dict[str, "MetricAccumulator"]]
    # dataset -> active reference-source count -> position -> statistics
    samples_by_gt_source_count: dict[
        str, dict[int, dict[str, "MetricAccumulator"]]
    ]


@dataclass
class MetricAccumulator:
    num_frames: int = 0
    num_reference_events: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tp_spatial: int = 0
    fp_spatial: int = 0
    localization_error_sum: float = 0.0
    localization_matches: int = 0
    exact_count_frames: int = 0
    count_abs_error_sum: int = 0

    def __add__(self, other: "MetricAccumulator") -> "MetricAccumulator":
        return MetricAccumulator(**{
            name: getattr(self, name) + getattr(other, name)
            for name in self.__dataclass_fields__
        })

    def __sub__(self, other: "MetricAccumulator") -> "MetricAccumulator":
        return MetricAccumulator(**{
            name: getattr(self, name) - getattr(other, name)
            for name in self.__dataclass_fields__
        })


@dataclass(frozen=True)
class JackknifeEstimate:
    full_sample: float
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float


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
    jackknife_standard_error: float
    p_value: float
    adjusted_p_value: float = math.nan
    correction_method: str = "none"
    correction_family: str = ""
    p_value_display: float = math.nan
    significant: bool = False
    better_model: str = ""


@dataclass
class ModelJackknifeData:
    full_metrics: dict[str, float]
    leave_one_out: dict[str, np.ndarray]
    estimates: dict[str, JackknifeEstimate]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        metavar="[LABEL=]CSV",
        help=(
            "Full filenames/paths of two or more prediction CSVs written by "
            "the eval script, in display order. "
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
        default=[],
        metavar="NAME",
        help=(
            "prediction_dataset value to skip entirely; repeat to exclude more. "
            "The checkpoint_dataset is retained as simulated-dataset for metrics "
            "but is always omitted from significance comparisons."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=DEFAULT_METRICS,
        default=list(DEFAULT_METRICS),
    )
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=0.5,
        help="Predicted ACCDOA norm required for an active source.",
    )
    parser.add_argument(
        "--reference-activity-threshold",
        type=float,
        default=1e-6,
        help="Reference ACCDOA norm required for an active source.",
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
        "--example-plot",
        action="store_true",
        help=(
            "Quick plotting test: do not read metric values from the CSVs. "
            "Use the supplied CSV filenames only to determine matrix row/column labels, "
            "and generate deterministic example matrix values."
        ),
    )
    parser.add_argument(
        "--no-cell-annotations",
        action="store_true",
        help=(
            "Do not draw a text value in every matrix cell. This can make PDF "
            "generation much faster for many models. Example plots disable cell "
            "annotations automatically."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution for both significance plots (default: %(default)s).",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--multiple-comparison-correction",
        choices=("none", "bonferroni", "holm", "fdr_bh"),
        default="holm",
        help=(
            "Correction applied to pairwise p-values (default: %(default)s). "
            "Holm and Bonferroni control the family-wise error rate; fdr_bh "
            "controls the false discovery rate."
        ),
    )
    parser.add_argument(
        "--correction-scope",
        choices=("dataset_metric", "dataset", "metric", "global"),
        default="dataset_metric",
        help=(
            "Tests grouped into one correction family (default: %(default)s). "
            "dataset_metric corrects all model pairs separately for each "
            "dataset and metric."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Processes used to parse independent CSVs. 0 chooses up to 4 "
            "workers automatically; use 1 for sequential I/O."
        ),
    )
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
    for prefix in MODEL_FILENAME_PREFIXES:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
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
    for prefix in MODEL_FILENAME_PREFIXES:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
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
        ism_order = "0"
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
        f"{ism_order:<5} | "
        f"{diffuse_late_reverb:^1} | "
        f"{corner_displacement:>4} | "
        f"{random_displacement:>4}"
    )


def matrix_parameter_values(path: Path) -> tuple[str, str, str, str, str]:
    """Return the five compact parameter values used in matrix labels."""
    label = matrix_parameter_label(path)
    parts = [part.strip() for part in label.split("|")]
    if len(parts) != 5:
        raise ValueError(f"Unexpected matrix parameter label: {label!r}")
    return tuple(parts)  # type: ignore[return-value]


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


def receiver_position_key(sample_id: str, context: str) -> str:
    """Extract the final ``<room>_P<digits>`` prefix from a sample id."""
    matches = list(re.finditer(r"_P(\d+)(?=$|_)", sample_id))
    if not matches:
        raise ValueError(
            f"Cannot extract room/receiver position from sample_id {sample_id!r} "
            f"in {context}; expected '<room>_P<digits>'."
        )
    match = matches[-1]
    room = sample_id[: match.start()]
    if not room:
        raise ValueError(f"Empty room name in sample_id {sample_id!r} ({context})")
    position = f"P{match.group(1)}"
    return json.dumps((room, position), ensure_ascii=False, separators=(",", ":"))


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def finalize_metrics(accumulator: MetricAccumulator) -> dict[str, float]:
    precision_15 = safe_divide(
        accumulator.tp_spatial,
        accumulator.tp_spatial + accumulator.fp_spatial + accumulator.fp,
    )
    recall_15 = safe_divide(
        accumulator.tp_spatial, accumulator.tp_spatial + accumulator.fn
    )
    f15 = safe_divide(2.0 * precision_15 * recall_15, precision_15 + recall_15)
    er15 = safe_divide(
        accumulator.fp + accumulator.fp_spatial + accumulator.fn,
        accumulator.num_reference_events,
    )
    le = safe_divide(
        accumulator.localization_error_sum, accumulator.localization_matches
    )
    lr = safe_divide(accumulator.tp, accumulator.tp + accumulator.fn)
    esdl = 0.25 * (er15 + (1.0 - f15) + le / 180.0 + (1.0 - lr))
    return {
        "LE": le,
        "LR": lr,
        "F15": f15,
        "ER15": er15,
        "ESDL": esdl,
        "count_MAE": safe_divide(
            accumulator.count_abs_error_sum, accumulator.num_frames
        ),
        "count_accuracy": safe_divide(
            accumulator.exact_count_frames, accumulator.num_frames
        ),
    }


def accumulate_frame(
    accumulator: MetricAccumulator,
    references: np.ndarray,
    predictions: np.ndarray,
    activity_threshold: float,
    reference_activity_threshold: float,
) -> int:
    reference_norm = np.linalg.norm(references, axis=1)
    prediction_norm = np.linalg.norm(predictions, axis=1)
    reference_slots = np.flatnonzero(reference_norm > reference_activity_threshold)
    prediction_slots = np.flatnonzero(prediction_norm >= activity_threshold)
    reference_count = len(reference_slots)
    prediction_count = len(prediction_slots)
    matched_count = min(reference_count, prediction_count)
    selected_angles = np.empty(0, dtype=np.float64)

    if matched_count:
        reference_unit = references[reference_slots] / reference_norm[reference_slots, None]
        prediction_unit = predictions[prediction_slots] / prediction_norm[prediction_slots, None]
        angular_cost = np.degrees(
            np.arccos(np.clip(reference_unit @ prediction_unit.T, -1.0, 1.0))
        )
        # Exact rectangular assignment in O(n^3), without constructing every
        # subset/permutation and all of its temporary arrays.
        row_indices, column_indices = linear_sum_assignment(angular_cost)
        selected_angles = angular_cost[row_indices, column_indices]

    spatial_tp = int(np.count_nonzero(selected_angles <= SPATIAL_THRESHOLD_DEG))
    accumulator.num_frames += 1
    accumulator.num_reference_events += reference_count
    accumulator.count_abs_error_sum += abs(reference_count - prediction_count)
    accumulator.exact_count_frames += int(reference_count == prediction_count)
    accumulator.tp += matched_count
    accumulator.fn += reference_count - matched_count
    accumulator.fp += prediction_count - matched_count
    accumulator.tp_spatial += spatial_tp
    accumulator.fp_spatial += matched_count - spatial_tp
    accumulator.localization_error_sum += float(np.sum(selected_angles))
    accumulator.localization_matches += matched_count
    return reference_count


def canonical_dataset_name(dataset: str) -> str:
    """Use the requested public name for predictions from the training dataset."""
    return "checkpoint_dataset" if dataset == "checkpoint_dataset" else dataset


def load_model(
    label: str,
    path: Path,
    excluded_datasets: set[str],
    activity_threshold: float,
    reference_activity_threshold: float,
) -> ModelData:
    samples: dict[str, dict[str, MetricAccumulator]] = {}
    samples_by_gt_source_count: dict[
        str, dict[int, dict[str, MetricAccumulator]]
    ] = {}
    position_cache: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            fieldnames = next(reader)
        except StopIteration:
            raise ValueError(f"CSV has no header: {path}")
        dataset_column = require_column(fieldnames, "prediction_dataset")
        named_columns = {
            name: require_column(fieldnames, name)
            for name in (
                "sample_id", "segment_idx", "frame_idx", "slot", "accdoa_x", "accdoa_y",
                "accdoa_z", "prediction_accdoa_x", "prediction_accdoa_y",
                "prediction_accdoa_z",
            )
        }
        index_by_name = {name: index for index, name in enumerate(fieldnames)}
        dataset_index = index_by_name[dataset_column]
        columns = {name: index_by_name[column] for name, column in named_columns.items()}
        current_frame: tuple[str, str, str, str, str] | None = None
        frame_rows: list[tuple[int, float, float, float, float, float, float]] = []

        def flush_frame() -> None:
            nonlocal frame_rows
            if current_frame is None:
                return
            dataset, position, _sample_id, _segment, _frame = current_frame
            ordered = sorted(frame_rows, key=lambda item: item[0])
            slots = [item[0] for item in ordered]
            if slots != list(range(len(slots))):
                raise ValueError(
                    f"Frame slots must be contiguous from zero in {path}: {slots}"
                )
            accumulator = samples.setdefault(dataset, {}).setdefault(
                position, MetricAccumulator()
            )
            values = np.asarray(ordered, dtype=np.float64)
            reference_count = accumulate_frame(
                accumulator,
                values[:, 1:4],
                values[:, 4:7],
                activity_threshold,
                reference_activity_threshold,
            )
            # Create every requested slice for every receiver position so that
            # delete-one jackknife samples remain aligned, including positions
            # with no frames of a particular source count.
            grouped = samples_by_gt_source_count.setdefault(
                dataset, {count: {} for count in (1, 2, 3)}
            )
            for count in (1, 2, 3):
                grouped[count].setdefault(position, MetricAccumulator())
            if reference_count in grouped:
                accumulate_frame(
                    grouped[reference_count][position],
                    values[:, 1:4],
                    values[:, 4:7],
                    activity_threshold,
                    reference_activity_threshold,
                )
            frame_rows = []

        for row_number, row in enumerate(reader, start=2):
            try:
                dataset = canonical_dataset_name(row[dataset_index].strip())
            except IndexError as error:
                raise ValueError(f"Short CSV row in {path} row {row_number}") from error
            if not dataset:
                raise ValueError(f"Empty prediction_dataset in {path} row {row_number}")
            if dataset in excluded_datasets:
                continue
            sample_id = row[columns["sample_id"]].strip()
            position = position_cache.get(sample_id)
            if position is None:
                position = receiver_position_key(sample_id, f"{path} row {row_number}")
                position_cache[sample_id] = position
            segment = row[columns["segment_idx"]].strip()
            frame = row[columns["frame_idx"]].strip()
            frame_key = (dataset, position, sample_id, segment, frame)
            if current_frame is not None and frame_key != current_frame:
                flush_frame()
            current_frame = frame_key
            try:
                frame_rows.append((
                    int(row[columns["slot"]]),
                    float(row[columns["accdoa_x"]]),
                    float(row[columns["accdoa_y"]]),
                    float(row[columns["accdoa_z"]]),
                    float(row[columns["prediction_accdoa_x"]]),
                    float(row[columns["prediction_accdoa_y"]]),
                    float(row[columns["prediction_accdoa_z"]]),
                ))
            except (IndexError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid value in {path} row {row_number}") from error
        flush_frame()

    if not samples:
        raise ValueError(f"No included prediction rows found in {path}")
    return ModelData(
        label=label,
        path=path,
        samples=samples,
        samples_by_gt_source_count=samples_by_gt_source_count,
    )


def validate_dataset_sets(models: Sequence[ModelData]) -> list[str]:
    reference = set(models[0].samples)
    for model in models[1:]:
        current = set(model.samples)
        if current != reference:
            raise ValueError(
                f"Dataset mismatch for {model.label}: "
                f"missing={sorted(reference-current)}, extra={sorted(current-reference)}"
            )
    return sorted(reference)


def validate_positions(models: Sequence[ModelData], dataset: str) -> list[str]:
    reference = set(models[0].samples[dataset])
    for model in models[1:]:
        current = set(model.samples[dataset])
        if current != reference:
            raise ValueError(
                f"Receiver-position mismatch for {model.label} in {dataset}: "
                f"missing={sorted(reference-current)[:5]}, "
                f"extra={sorted(current-reference)[:5]}"
            )
    if not reference:
        raise ValueError(f"Dataset {dataset!r} has no receiver positions")
    return sorted(reference)


def sum_samples(samples: Iterable[MetricAccumulator]) -> MetricAccumulator:
    total = MetricAccumulator()
    for sample in samples:
        total = total + sample
    return total


def jackknife_from_full_and_loo(
    full_sample: float,
    leave_one_out: np.ndarray,
    alpha: float,
) -> JackknifeEstimate:
    count = len(leave_one_out)
    if count < 2 or not np.all(np.isfinite(leave_one_out)) or not math.isfinite(full_sample):
        raise ValueError("Jackknife requires at least two finite leave-one-out estimates")
    pseudo_values = count * full_sample - (count - 1) * leave_one_out
    estimate = float(np.mean(pseudo_values))
    mean_loo = float(np.mean(leave_one_out))
    # Standard delete-one jackknife SE:
    #
    #   sqrt((N - 1) / N * sum_i(theta_(i) - mean(theta_(.)))**2)
    #
    # This is also std(pseudo_values, ddof=1) / sqrt(N).  Do not divide the
    # leave-one-out sum of squares by N(N-1): that expression omits the
    # (N-1)^2 scaling induced by the pseudo-values and makes the SE exactly
    # N-1 times too small.
    standard_error = math.sqrt(
        ((count - 1) / count)
        * float(np.sum((leave_one_out - mean_loo) ** 2))
    )
    critical = float(student_t.ppf(1.0 - alpha / 2.0, df=count - 1))
    return JackknifeEstimate(
        full_sample=full_sample,
        estimate=estimate,
        standard_error=standard_error,
        ci_low=estimate - critical * standard_error,
        ci_high=estimate + critical * standard_error,
    )


def model_jackknife(
    model: ModelData,
    dataset: str,
    positions: Sequence[str],
    metric: str,
    alpha: float,
) -> JackknifeEstimate:
    total = sum_samples(model.samples[dataset][position] for position in positions)
    full = finalize_metrics(total)[metric]
    leave_one_out = np.asarray([
        finalize_metrics(total - model.samples[dataset][position])[metric]
        for position in positions
    ], dtype=np.float64)
    return jackknife_from_full_and_loo(full, leave_one_out, alpha)


def prepare_model_jackknife(
    model: ModelData,
    dataset: str,
    positions: Sequence[str],
    metrics: Sequence[str],
    alpha: float,
) -> ModelJackknifeData:
    """Compute every reusable model/position metric exactly once."""
    return prepare_accumulator_jackknife(
        model.samples[dataset], positions, metrics, alpha
    )


def prepare_accumulator_jackknife(
    samples_by_position: dict[str, MetricAccumulator],
    positions: Sequence[str],
    metrics: Sequence[str],
    alpha: float,
) -> ModelJackknifeData:
    """Compute jackknife metrics for one overall or source-count slice."""
    total = sum_samples(samples_by_position[position] for position in positions)
    full_metrics = finalize_metrics(total)
    loo_metric_rows = [
        finalize_metrics(total - samples_by_position[position])
        for position in positions
    ]
    leave_one_out = {
        metric: np.fromiter(
            (row[metric] for row in loo_metric_rows),
            dtype=np.float64,
            count=len(loo_metric_rows),
        )
        for metric in metrics
    }
    estimates = {
        metric: jackknife_from_full_and_loo(
            full_metrics[metric], leave_one_out[metric], alpha
        )
        for metric in metrics
    }
    return ModelJackknifeData(full_metrics, leave_one_out, estimates)


def compare_pair(
    model_a: ModelData,
    model_b: ModelData,
    dataset: str,
    positions: Sequence[str],
    metrics: Sequence[str],
    alpha: float,
    prepared_a: ModelJackknifeData | None = None,
    prepared_b: ModelJackknifeData | None = None,
) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    prepared_a = prepared_a or prepare_model_jackknife(
        model_a, dataset, positions, metrics, alpha
    )
    prepared_b = prepared_b or prepare_model_jackknife(
        model_b, dataset, positions, metrics, alpha
    )
    for metric in metrics:
        full_difference = (
            prepared_a.full_metrics[metric] - prepared_b.full_metrics[metric]
        )
        leave_one_out = (
            prepared_a.leave_one_out[metric] - prepared_b.leave_one_out[metric]
        )
        jackknife = jackknife_from_full_and_loo(full_difference, leave_one_out, alpha)
        if jackknife.standard_error == 0.0:
            p_value = 1.0 if jackknife.estimate == 0.0 else 0.0
        else:
            statistic = abs(jackknife.estimate / jackknife.standard_error)
            p_value = float(2.0 * student_t.sf(statistic, df=len(positions) - 1))
        estimate_a = prepared_a.estimates[metric]
        estimate_b = prepared_b.estimates[metric]
        significant = jackknife.ci_low > 0.0 or jackknife.ci_high < 0.0
        better_model = ""
        if significant and jackknife.estimate != 0.0:
            if metric in LOWER_IS_BETTER:
                better_model = model_a.label if jackknife.estimate < 0 else model_b.label
            else:
                better_model = model_a.label if jackknife.estimate > 0 else model_b.label
        rows.append(
            ComparisonRow(
                dataset=dataset,
                model_a=model_a.label,
                model_b=model_b.label,
                metric=metric,
                n_paired_positions=len(positions),
                estimate_a=estimate_a.estimate,
                estimate_b=estimate_b.estimate,
                difference_a_minus_b=jackknife.estimate,
                ci_low=jackknife.ci_low,
                ci_high=jackknife.ci_high,
                jackknife_standard_error=jackknife.standard_error,
                p_value=p_value,
                p_value_display=p_value,
                significant=significant,
                better_model=better_model,
            )
        )
    return rows


def adjust_p_values(p_values: Sequence[float], method: str) -> np.ndarray:
    """Return multiplicity-adjusted p-values in the original input order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p-values must be one-dimensional")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be finite and between zero and one")
    count = len(values)
    if count == 0 or method == "none":
        return values.copy()
    if method == "bonferroni":
        return np.minimum(values * count, 1.0)

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    if method == "holm":
        adjusted_ordered = np.maximum.accumulate(
            (count - np.arange(count)) * ordered
        )
    elif method == "fdr_bh":
        ranks = np.arange(1, count + 1)
        adjusted_ordered = np.minimum.accumulate(
            (count * ordered / ranks)[::-1]
        )[::-1]
    else:
        raise ValueError(f"Unknown multiple-comparison correction: {method!r}")

    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted


def correction_family_key(row: ComparisonRow, scope: str) -> tuple[str, ...]:
    if scope == "dataset_metric":
        return row.dataset, row.metric
    if scope == "dataset":
        return (row.dataset,)
    if scope == "metric":
        return (row.metric,)
    if scope == "global":
        return ("all",)
    raise ValueError(f"Unknown correction scope: {scope!r}")


def apply_multiple_comparison_correction(
    rows: Sequence[ComparisonRow],
    method: str,
    scope: str,
    alpha: float,
) -> None:
    """Adjust p-values by family and update every significance-dependent field."""
    families: dict[tuple[str, ...], list[ComparisonRow]] = {}
    for row in rows:
        families.setdefault(correction_family_key(row, scope), []).append(row)

    for key, family in families.items():
        adjusted = adjust_p_values([row.p_value for row in family], method)
        family_label = f"{scope}:" + "/".join(key)
        for row, adjusted_p_value in zip(family, adjusted):
            row.adjusted_p_value = float(adjusted_p_value)
            row.correction_method = method
            row.correction_family = family_label
            row.p_value_display = row.adjusted_p_value
            row.significant = row.adjusted_p_value <= alpha
            row.better_model = ""
            if row.significant and row.difference_a_minus_b != 0.0:
                if row.metric in LOWER_IS_BETTER:
                    row.better_model = (
                        row.model_a if row.difference_a_minus_b < 0.0 else row.model_b
                    )
                else:
                    row.better_model = (
                        row.model_a if row.difference_a_minus_b > 0.0 else row.model_b
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
                output[column] = json_number(row.p_value_display)
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
            if not math.isfinite(row.p_value_display):
                continue
            if row.significant and row.better_model:
                direction = 1.0 if row.better_model == pair[0] else -1.0
                values[row_index, column_index] = direction * -math.log10(
                    max(row.p_value_display, np.finfo(float).tiny)
                )
                winner = "A" if direction > 0 else "B"
                annotations[row_index][column_index] = (
                    f"{winner}*\nd={row.difference_a_minus_b:+.3g}"
                )
            else:
                values[row_index, column_index] = 0.0
                annotations[row_index][column_index] = (
                    f"ns\nd={row.difference_a_minus_b:+.3g}"
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
    axis.set_title("Per-dataset paired jackknife significance")
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
        p_matrix[row_index, column_index] = row.p_value_display
        p_matrix[column_index, row_index] = row.p_value_display
        # Each cell is expressed from the row model's perspective.
        difference_matrix[row_index, column_index] = row.difference_a_minus_b
        difference_matrix[column_index, row_index] = -row.difference_a_minus_b
        if not math.isfinite(row.p_value_display):
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


def combined_score_from_cached_directions(
    direction_matrices: Sequence[np.ndarray],
) -> np.ndarray:
    """Combine already-built dataset direction matrices without recomputing them."""
    stacked = np.stack(direction_matrices)
    all_available = np.all(np.isfinite(stacked), axis=0)
    combined = np.sum(np.where(np.isfinite(stacked), stacked, 0.0), axis=0)
    combined[~all_available] = np.nan
    return combined


def formatted_metric_name(metric: str) -> str:
    """Format metric names for plot titles."""

    if metric == "ESDL":
        return r"$\mathbf{E}_{\mathbf{SDL}}$"

    match = re.fullmatch(r"(.+?)(\d+)", metric)
    if match is None:
        return metric

    base, digits = match.groups()
    escaped_base = base.replace("_", r"\_")
    return rf"$\mathbf{{{escaped_base}}}_{{\mathbf{{{digits}}}}}$"


def datasets_in_plot_order(datasets: Sequence[str]) -> list[str]:
    preferred_order = [
        "arni-dataset",
        "motus-dataset",
        "ivan-dataset",
    ]
    return [
        dataset
        for dataset in preferred_order
        if dataset in datasets
    ]


def plot_significance_matrices(
    path: Path,
    comparison_rows: Sequence[ComparisonRow],
    datasets: Sequence[str],
    metrics: Sequence[str],
    models: Sequence[str],
    model_display_labels: Sequence[str],
    model_parameter_values: Sequence[tuple[str, str, str, str, str]],
    alpha: float,
    show_cell_annotations: bool,
    dpi: int,
    p_value_label: str,
) -> list[Path]:
    """Write one DIN-A4 portrait PNG per metric.

    Labeling follows the multi-metric matrix plot:
    * the compact model descriptor is used directly as the y tick label;
    * the same descriptor is used for x tick labels on the bottom matrix;
    * the fixed-width parameter header is shown above the y labels;
    * dataset names are regular y-axis labels.
    """
    if len(model_display_labels) != len(models):
        raise ValueError("Model labels and matrix display labels have different lengths")
    if len(model_parameter_values) != len(models):
        raise ValueError("Model labels and matrix parameter values have different lengths")

    display_labels = list(model_display_labels)
    number_of_models = len(models)
    maximum_score = max(1, len(datasets))

    orange_map = plt.get_cmap("Oranges")
    green_map = plt.get_cmap("Greens")
    orange_colors = [
        orange_map(value) for value in np.linspace(0.88, 0.30, maximum_score)
    ]
    green_colors = [
        green_map(value) for value in np.linspace(0.30, 0.88, maximum_score)
    ]
    score_colors = [*orange_colors, "#e5e7eb", *green_colors]
    color_map = ListedColormap(score_colors)
    color_map.set_bad("white")
    norm = BoundaryNorm(
        np.arange(-maximum_score - 0.5, maximum_score + 1.5, 1.0),
        color_map.N,
    )

    panels = [*datasets, None]
    number_of_panels = len(panels)
    base_path = path.with_suffix("")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    a4_portrait = (8.27, 11.69)

    for metric in metrics:
        cached_matrices = {
            dataset: square_significance_matrices(
                comparison_rows, dataset, metric, models
            )
            for dataset in datasets
        }
        combined_direction_matrix = combined_score_from_cached_directions(
            [cached_matrices[dataset][1] for dataset in datasets]
        )

        figure, axes = plt.subplots(
            number_of_panels,
            1,
            figsize=a4_portrait,
            squeeze=False,
        )
        flat_axes = list(axes.flat)

        tick_font_size = max(4.0, min(6.4, 8.0 - 0.16 * number_of_models))
        annotation_font_size = max(3.6, min(6.6, 54.0 / max(1, number_of_models)))

        for panel_index, dataset in enumerate(panels):
            axis = flat_axes[panel_index]
            is_combined = dataset is None

            if is_combined:
                direction_matrix = combined_direction_matrix
                difference_matrix = np.full(
                    (number_of_models, number_of_models), np.nan, dtype=np.float64
                )
                dataset_label = "Summary"
            else:
                _, direction_matrix, difference_matrix = cached_matrices[str(dataset)]
                dataset_label = str(dataset)

            axis.imshow(
                direction_matrix,
                cmap=color_map,
                norm=norm,
                aspect="auto",
                interpolation="nearest",
            )

            axis.set_xticks(np.arange(number_of_models))
            axis.set_yticks(np.arange(number_of_models))

            # Same matrix labeling style as the multi-metric version:
            # fixed-width parameter rows directly beside the matrix.
            axis.set_yticklabels(
                display_labels,
                fontsize=tick_font_size,
                fontfamily="monospace",
            )

            # Only the final matrix gets bottom labels.  Instead of rotating
            # the full fixed-width descriptor, draw a transposed parameter table:
            # one model per matrix column, with the five parameter fields stacked
            # below it and a shared field-name header at the left.
            if panel_index == number_of_panels - 1:
                axis.set_xticklabels([])
                bottom_header = ("Mic.", "Order", "L", "Corn", "Rand")
                # Keep table rows evenly spaced in physical units.  Axes-fraction
                # offsets made the gaps change whenever the panel layout changed.
                table_top_points = -9.0
                row_step_points = 8.0

                # Field-name header/row labels, analogous to the header above
                # the fixed-width y-label table on the left.
                for field_index, field_name in enumerate(bottom_header):
                    axis.annotate(
                        field_name,
                        xy=(0.0, 0.0),
                        xycoords="axes fraction",
                        xytext=(-4.0, table_top_points - field_index * row_step_points),
                        textcoords="offset points",
                        ha="right",
                        va="top",
                        fontsize=tick_font_size,
                        fontfamily="monospace",
                        fontweight="bold",
                        annotation_clip=False,
                    )

                # Each matrix column gets one value in every parameter row.
                # x is expressed in axes coordinates so values stay centered
                # directly under their corresponding matrix column.
                for model_index, values in enumerate(model_parameter_values):
                    x = (model_index + 0.5) / number_of_models
                    for field_index, value in enumerate(values):
                        axis.annotate(
                            value,
                            xy=(x, 0.0),
                            xycoords="axes fraction",
                            xytext=(0.0, table_top_points - field_index * row_step_points),
                            textcoords="offset points",
                            ha="center",
                            va="top",
                            fontsize=tick_font_size,
                            fontfamily="monospace",
                            annotation_clip=False,
                        )
            else:
                axis.set_xticklabels([])

            # Header for the fixed-width row-label table.  Use the same
            # right-aligned anchor as the y tick labels so every separator
            # column is vertically identical in header and rows.
            if panel_index == 0:
                axis.annotate(
                    MATRIX_LABEL_COLUMN_HEADER,
                    xy=(0.0, 1.0),
                    xycoords="axes fraction",
                    xytext=(-4.0, 5.0),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    fontsize=tick_font_size,
                    fontfamily="monospace",
                    fontweight="bold",
                    annotation_clip=False,
                )

            axis.set_xticks(np.arange(-0.5, number_of_models, 1), minor=True)
            axis.set_yticks(np.arange(-0.5, number_of_models, 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.45)
            axis.tick_params(which="minor", bottom=False, left=False)
            axis.tick_params(which="major", length=2)
            # Keep y labels snug to the matrix while preserving the exact
            # monospaced table spacing encoded in matrix_parameter_label().
            axis.tick_params(axis="y", which="major", pad=2)
            axis.tick_params(axis="x", which="major", pad=3)

            if show_cell_annotations:
                for row_index in range(number_of_models):
                    for column_index in range(number_of_models):
                        if is_combined:
                            displayed_value = direction_matrix[row_index, column_index]
                            if not math.isfinite(displayed_value):
                                continue
                            annotation = f"{int(displayed_value):d}"
                        else:
                            displayed_value = difference_matrix[row_index, column_index]
                            if not math.isfinite(displayed_value):
                                continue
                            annotation = f"{(displayed_value):+.2g}"

                        axis.text(
                            column_index,
                            row_index,
                            annotation,
                            ha="center",
                            va="center",
                            fontsize=annotation_font_size,
                            color=(
                                "white"
                                if abs(direction_matrix[row_index, column_index]) >= 2
                                else "#111827"
                            ),
                        )

        # First establish the final matrix geometry.  The headings below are
        # centered on the matrix area (not on the full page, which also contains
        # the wide parameter table on the left).
        figure.subplots_adjust(
            left=0.205,
            right=0.965,
            top=0.935,
            bottom=0.06,
            hspace=0.060,
        )

        first_matrix_box = flat_axes[0].get_position()
        first_matrix_top = first_matrix_box.y1

        # Shade each complete panel, including its matrix and left-side labels.
        # A stronger strip behind the dataset name visually connects the
        # rotated heading with its matrix and removes the large empty-looking
        # horizontal gap present in the unshaded layout.
        #panel_shades = ("#edf5fb", "#eef8f1", "#fff4e8", "#f1f3f6")
        #label_shades = ("#c8e0f2", "#cce9d5", "#f6d5ae", "#d8dde6")
        panel_shades = (
            "#e4eff6",  # arni-dataset  - light blue
            "#f2edf7",  # motus-dataset - light purple
            "#fff0e2",  # ivan-dataset  - light orange
            "#f1f3f6",  # Combined
        )

        label_shades = (
            "#bcd6e8",  # arni-dataset
            "#dfd1eb",  # motus-dataset
            "#ffd9b7",  # ivan-dataset
            "#d8dde6",  # Combined
        )
        panel_left = 0.018
        panel_right = 0.982
        label_left = 0.020
        label_right = 0.04
        panel_vertical_padding = 0.004

        for panel_index, (axis, dataset) in enumerate(zip(flat_axes, panels)):
            axis_box = axis.get_position()
            panel_bottom = axis_box.y0 - panel_vertical_padding
            panel_height = axis_box.height + 2.0 * panel_vertical_padding
            figure.add_artist(
                Rectangle(
                    (panel_left, panel_bottom),
                    panel_right - panel_left,
                    panel_height,
                    transform=figure.transFigure,
                    facecolor=panel_shades[panel_index % len(panel_shades)],
                    edgecolor="none",
                    zorder=-10,
                    clip_on=False,
                )
            )
            figure.add_artist(
                Rectangle(
                    (label_left, panel_bottom),
                    label_right - label_left,
                    panel_height,
                    transform=figure.transFigure,
                    facecolor=label_shades[panel_index % len(label_shades)],
                    edgecolor="none",
                    zorder=-9,
                    clip_on=False,
                )
            )
            axis.set_zorder(1)
            axis.patch.set_alpha(0.0)
            figure.text(
                0.5 * (label_left + label_right),
                0.5 * (axis_box.y0 + axis_box.y1),
                "Summary" if dataset is None else str(dataset),
                ha="center",
                va="center",
                rotation=90,
                fontsize=7.5,
                fontweight="bold",
                zorder=2,
            )

        # Center the title and definition line on the complete A4 page/content
        # band.  Centering on the matrix alone shifts them right because the
        # parameter table and dataset-name gutter occupy only the left side.
        figure.suptitle(
            formatted_metric_name(metric),
            x=0.5,
            y=0.995,
            fontsize=14,
            fontweight="bold",
            ha="center",
        )
        figure.text(
            0.5,
            0.965,
            MATRIX_LABEL_DEFINITIONS,
            ha="center",
            va="center",
            fontsize=6.0,
            fontfamily="monospace",
        )

        # Put the legend directly above the first matrix and center it on the
        # matrix width.  "lower center" makes the anchor the legend's bottom edge.
        legend_handles = [
            Patch(facecolor=score_colors[maximum_score], label="not significant"),
            Patch(facecolor=green_colors[0], label="row model significantly better"),
            Patch(facecolor=orange_colors[-1], label="row model significantly worse"),
            Patch(facecolor="none", edgecolor="none", label=r"Cell value = $\hat{\Delta}_{\mathrm{row,col,jack}}$"),
        ]
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, first_matrix_top + 0.006),
            frameon=False,
            fontsize=6.3,
            columnspacing=1.2,
            handletextpad=0.5,
            borderaxespad=0.0,
        )

        safe_metric = re.sub(r"[^A-Za-z0-9._-]+", "_", metric).strip("_") or "metric"
        output_path = base_path.parent / f"{base_path.name}_{safe_metric}.pdf"
        # Preserve the exact A4 canvas.  Tight bounding boxes recrop around the
        # external labels and shift all visually centered elements sideways.
        figure.savefig(output_path, dpi=dpi, facecolor="white")
        plt.close(figure)
        output_paths.append(output_path)

    return output_paths

def make_example_comparisons(
    models: Sequence[str],
    datasets: Sequence[str],
    metrics: Sequence[str],
    seed: int,
) -> list[ComparisonRow]:
    """Create deterministic example values for a fast plot-layout check."""
    rng = np.random.default_rng(seed)
    rows: list[ComparisonRow] = []
    for dataset_index, dataset in enumerate(datasets):
        for metric_index, metric in enumerate(metrics):
            for pair_index, (model_a, model_b) in enumerate(combinations(models, 2)):
                difference = float(rng.normal(loc=0.0, scale=0.25))
                significant = ((pair_index + dataset_index + metric_index) % 3) != 0
                example_p_value = 0.01 if significant else 0.2
                if significant and difference == 0:
                    difference = 0.1
                if significant:
                    if metric in LOWER_IS_BETTER:
                        better_model = model_a if difference < 0 else model_b
                    else:
                        better_model = model_a if difference > 0 else model_b
                else:
                    better_model = ""
                rows.append(
                    ComparisonRow(
                        dataset=dataset,
                        model_a=model_a,
                        model_b=model_b,
                        metric=metric,
                        n_paired_positions=12,
                        estimate_a=1.0 + difference / 2.0,
                        estimate_b=1.0 - difference / 2.0,
                        difference_a_minus_b=difference,
                        ci_low=difference - 0.1,
                        ci_high=difference + 0.1,
                        jackknife_standard_error=0.05,
                        p_value=example_p_value,
                        p_value_display=example_p_value,
                        significant=significant,
                        better_model=better_model,
                    )
                )
    return rows


def main() -> None:
    args = parse_args()
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if args.workers < 0:
        raise ValueError("--workers must be zero or positive")

    specs = collect_model_specs(args)
    excluded = {
        canonical_dataset_name(value.strip())
        for value in args.exclude_dataset
        if value.strip()
    }

    if args.example_plot:
        labels = [label for label, _ in specs]
        matrix_display_labels = [matrix_parameter_label(path) for _, path in specs]
        matrix_parameter_rows = [matrix_parameter_values(path) for _, path in specs]
        datasets = ["Arni", "Motus", "Ivan"]
        plot_metrics = [
            metric for metric in args.metrics if metric not in EXCLUDED_PLOT_METRICS
        ]
        if not plot_metrics:
            raise ValueError("--example-plot needs at least one plottable metric")
        comparisons = make_example_comparisons(labels, datasets, plot_metrics, 20260809)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_significance_matrices(
            args.output_dir / "significance_matrices_a4.pdf",
            comparisons,
            datasets,
            plot_metrics,
            labels,
            matrix_display_labels,
            matrix_parameter_rows,
            args.alpha,
            True,  # show deterministic example mean differences in matrix cells
            args.dpi,
            "example jackknife p",
        )
        print(
            f"Saved example A4 matrix PNGs with basename: "
            f"{(args.output_dir / 'significance_matrices_a4').resolve()}"
        )
        return

    worker_count = args.workers or min(len(specs), os.cpu_count() or 1, 4)
    worker_count = min(worker_count, len(specs))
    load_arguments = [
        (label, path, excluded, args.activity_threshold,
         args.reference_activity_threshold)
        for label, path in specs
    ]
    if worker_count == 1:
        models = [load_model(*values) for values in load_arguments]
    else:
        print(f"Parsing {len(specs)} CSVs with {worker_count} worker processes...")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(load_model, *values) for values in load_arguments]
            # Resolve in input order so all output ordering remains deterministic.
            models = [future.result() for future in futures]
    datasets = validate_dataset_sets(models)
    significance_datasets = [
        dataset for dataset in datasets if dataset != "simulated-dataset"
    ]
    labels = [model.label for model in models]
    matrix_display_labels = [matrix_parameter_label(model.path) for model in models]
    matrix_parameter_rows = [matrix_parameter_values(model.path) for model in models]
    model_pairs = list(combinations(labels, 2))
    comparisons: list[ComparisonRow] = []
    metric_rows: list[dict[str, object]] = []
    dataset_payload: dict[str, object] = {}

    for dataset in datasets:
        positions = validate_positions(models, dataset)
        comparison_enabled = dataset in significance_datasets
        comparison_note = "" if comparison_enabled else " (metrics only)"
        print(
            f"Dataset {dataset}: {len(positions)} paired receiver positions"
            f"{comparison_note}."
        )
        prepared = {
            model.label: prepare_model_jackknife(
                model, dataset, positions, args.metrics, args.alpha
            )
            for model in models
        }
        prepared_by_source_count = {
            model.label: {
                count: prepare_accumulator_jackknife(
                    model.samples_by_gt_source_count[dataset][count],
                    positions,
                    args.metrics,
                    args.alpha,
                )
                for count in (1, 2, 3)
            }
            for model in models
        }
        dataset_comparisons: list[ComparisonRow] = []
        if comparison_enabled:
            for model_a, model_b in combinations(models, 2):
                print(f"  Comparing {model_a.label} vs {model_b.label}...")
                dataset_comparisons.extend(
                    compare_pair(
                        model_a,
                        model_b,
                        dataset,
                        positions,
                        args.metrics,
                        args.alpha,
                        prepared[model_a.label],
                        prepared[model_b.label],
                    )
                )
        comparisons.extend(dataset_comparisons)

        models_payload: dict[str, object] = {}
        for model in models:
            metrics_payload: dict[str, object] = {}
            for metric in args.metrics:
                estimate = prepared[model.label].estimates[metric]
                metrics_payload[metric] = {
                    key: json_number(value)
                    for key, value in asdict(estimate).items()
                }
                metric_rows.append(
                    {
                        "dataset": dataset,
                        "model": model.label,
                        "gt_source_count": "all",
                        "metric": metric,
                        "number_of_receiver_positions": len(positions),
                        **asdict(estimate),
                    }
                )
            by_source_count_payload: dict[str, object] = {}
            for count in (1, 2, 3):
                count_metrics_payload: dict[str, object] = {}
                for metric in args.metrics:
                    estimate = prepared_by_source_count[model.label][count].estimates[
                        metric
                    ]
                    count_metrics_payload[metric] = {
                        key: json_number(value)
                        for key, value in asdict(estimate).items()
                    }
                    metric_rows.append(
                        {
                            "dataset": dataset,
                            "model": model.label,
                            "gt_source_count": count,
                            "metric": metric,
                            "number_of_receiver_positions": len(positions),
                            **asdict(estimate),
                        }
                    )
                by_source_count_payload[str(count)] = count_metrics_payload
            models_payload[model.label] = {
                "prediction_file": str(model.path),
                "matrix_parameter_label": matrix_parameter_label(model.path),
                "metrics": metrics_payload,
                "metrics_by_gt_source_count": by_source_count_payload,
            }
        dataset_payload[dataset] = {
            "number_of_paired_receiver_positions": len(positions),
            "included_in_significance_comparison": comparison_enabled,
            "models": models_payload,
        }

    apply_multiple_comparison_correction(
        comparisons,
        args.multiple_comparison_correction,
        args.correction_scope,
        args.alpha,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dicts = [asdict(row) for row in comparisons]
    write_csv(
        args.output_dir / "model_metrics.csv",
        metric_rows,
        (
            "dataset",
            "model",
            "gt_source_count",
            "metric",
            "number_of_receiver_positions",
            "full_sample",
            "estimate",
            "standard_error",
            "ci_low",
            "ci_high",
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
        significance_datasets,
        args.metrics,
        model_pairs,
    )
    if args.multiple_comparison_correction == "none":
        p_value_label = "raw jackknife t p"
    else:
        p_value_label = (
            f"{args.multiple_comparison_correction}-adjusted jackknife t p"
        )
    plot_metrics = [
        metric for metric in args.metrics if metric not in EXCLUDED_PLOT_METRICS
    ]
    if plot_metrics and significance_datasets:
        plot_datasets = datasets_in_plot_order(significance_datasets)
        plot_significance_heatmap(
            args.output_dir / "significance_heatmap.pdf",
            comparisons,
            plot_datasets,
            plot_metrics,
            model_pairs,
            args.dpi,
            p_value_label,
        )
        plot_significance_matrices(
            args.output_dir / "significance_matrices_a4.pdf",
            comparisons,
            plot_datasets,
            plot_metrics,
            labels,
            matrix_display_labels,
            matrix_parameter_rows,
            args.alpha,
            not args.no_cell_annotations,
            args.dpi,
            p_value_label,
        )
    elif not plot_metrics:
        print("No plottable metrics selected; significance plots were skipped.")
    else:
        print("No significance datasets found; significance plots were skipped.")

    payload = {
        "method": {
            "input": "raw augmented prediction CSVs from the evaluation script",
            "resampling_unit": "unique room and receiver position parsed from sample_id",
            "point_estimate": "bias-corrected jackknife pseudo-value mean",
            "confidence_interval": (
                "Student-t jackknife interval using N-1 degrees of freedom, "
                "following Equations (1)-(7) of Mesaros et al."
            ),
            "paired_significance": (
                "adjusted two-sided jackknife t p-value is less than or equal "
                "to alpha; reported confidence intervals remain unadjusted"
            ),
            "p_value": (
                "raw two-sided Student-t p-value from jackknife estimate / "
                "standard error"
            ),
            "adjusted_p_value": (
                f"{args.multiple_comparison_correction} correction with "
                f"{args.correction_scope} families"
            ),
            "multiple_testing": {
                "method": args.multiple_comparison_correction,
                "scope": args.correction_scope,
            },
            "combined_matrix_score": (
                "+1 for each dataset where the row model is significantly "
                "better, -1 where significantly worse, and 0 where not "
                "significant; summed across all datasets"
            ),
            "excluded_datasets": sorted(excluded),
            "metrics_only_datasets": (
                ["simulated-dataset"] if "simulated-dataset" in datasets else []
            ),
            "alpha": args.alpha,
            "activity_threshold": args.activity_threshold,
            "reference_activity_threshold": args.reference_activity_threshold,
        },
        "number_of_models": len(models),
        "number_of_datasets": len(datasets),
        "number_of_significance_datasets": len(significance_datasets),
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
