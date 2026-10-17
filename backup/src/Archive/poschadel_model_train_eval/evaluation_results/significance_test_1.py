#!/usr/bin/env python3
"""Per-dataset paired statistical comparison of SDL model prediction exports.

Inputs are combined ``predictions_<checkpoint>.csv`` files produced by the
evaluation script. Rows are split using ``prediction_dataset`` before any
statistics are calculated. The checkpoint dataset is excluded by default.
ACCDOA magnitude is used only to threshold predicted/reference activity.
Comparisons are performed on thresholded source counts and Hungarian-matched
direction-of-arrival (DOA) angles.

The resampling unit is a recording position, not a whole source CSV and not an
individual frame. A position is identified by ``recording_position`` when that
column exists; otherwise ``base_name`` or ``sample_id``/``segment_idx`` is
used. The source CSV remains part of the key so equally named positions in
different source files cannot collide. This preserves within-position
dependence while avoiding both CSV-level aggregation and frame-level
pseudo-replication.

Example:
    python compare_model_predictions.py \
        --model evaluation_a/predictions_shoeBox_SH_ISMOrder12-Late.csv \
        --model evaluation_b/predictions_improved.csv \
        --output-dir significance_results

Outputs:
    model_metrics.csv
    pairwise_significance.csv
    significance_matrix.csv
    significance_heatmap.png
    significance_results.json

Requirements:
    pip install numpy scipy matplotlib
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations, permutations as iter_permutations
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.optimize import linear_sum_assignment


EPS = 1e-12
# Reference coordinates are metadata written through CSV and may differ by a
# few last decimal places between otherwise equivalent exports. Pairing uses
# this precision only for validation; metrics still use each file's full values.
REFERENCE_PAIRING_DECIMALS = 8
# Emit a heartbeat often enough that loading large CSV exports does not look
# stalled.
LOAD_PROGRESS_ROWS = 100_000
DEFAULT_METRICS = (
    "LE",
    "LR",
    "F15",
    "ER15",
    "ESDL",
    "count_MAE",
    "count_accuracy",
)
LOWER_IS_BETTER = {"LE", "ER15", "ESDL", "count_MAE"}


@dataclass
class SufficientStatistics:
    frames: int = 0
    reference_events: int = 0
    prediction_events: int = 0
    localization_matches: int = 0
    localization_error_sum: float = 0.0
    source_true_positives: int = 0
    source_false_positives: int = 0
    source_false_negatives: int = 0
    spatial_true_positives: int = 0
    spatial_false_positives: int = 0
    spatial_false_negatives: int = 0
    spatial_substitutions: int = 0
    count_absolute_error_sum: int = 0
    exact_count_frames: int = 0

    def add(self, other: "SufficientStatistics") -> None:
        for field_name in self.__dataclass_fields__:
            setattr(
                self,
                field_name,
                getattr(self, field_name) + getattr(other, field_name),
            )

    def as_array(self) -> np.ndarray:
        return np.asarray(list(asdict(self).values()), dtype=np.float64)

    @classmethod
    def from_array(cls, values: np.ndarray) -> "SufficientStatistics":
        names = list(cls.__dataclass_fields__)
        return cls(**dict(zip(names, values.tolist())))


@dataclass
class ModelData:
    label: str
    dataset: str
    path: Path
    statistics_by_position: dict[str, SufficientStatistics]
    signatures_by_position: dict[str, str]


@dataclass
class ComparisonRow:
    dataset: str
    model_a: str
    model_b: str
    metric: str
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
        "--model",
        action="append",
        required=True,
        metavar="PATH",
        help=(
            "Prediction export; repeat for every model. The model name is "
            "derived from predictions_<MODEL>.csv."
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
        help=(
            "prediction_dataset value to skip; repeat to exclude more than "
            "one. The eval export's checkpoint_dataset is skipped by default."
        ),
    )
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=0.5,
        help="ACCDOA magnitude used only to decide predicted activity.",
    )
    parser.add_argument(
        "--reference-activity-threshold",
        type=float,
        default=1e-6,
        help="Reference ACCDOA magnitude used only to decide activity.",
    )
    parser.add_argument(
        "--spatial-threshold-deg",
        type=float,
        default=15.0,
        help="Maximum angular error for a spatial true positive.",
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
        help="Paired recording-position-level bootstrap samples for the 95%% CI.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10_000,
        help="Paired recording-position-level label permutations for each p-value.",
    )
    parser.add_argument(
        "--resampling-batch-size",
        type=int,
        default=2048,
        help=(
            "Number of bootstrap/permutation replicates processed together "
            "with NumPy (default: 2048). Reduce only if memory is constrained."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def parse_model_arguments(arguments: Sequence[str]) -> list[tuple[str, Path]]:
    models: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for argument in arguments:
        path = Path(argument).expanduser().resolve()
        prefix = "predictions_"
        if path.suffix.lower() != ".csv" or not path.stem.startswith(prefix):
            raise ValueError(
                "--model filenames must use predictions_<MODEL>.csv, got "
                f"{path.name!r}"
            )
        label = path.stem[len(prefix):]
        if not label:
            raise ValueError(f"Empty model name in prediction file {path.name!r}")
        if label in labels:
            raise ValueError(
                f"Duplicate model name derived from prediction filenames: {label!r}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        labels.add(label)
        models.append((label, path))
    if len(models) < 2:
        raise ValueError("At least two --model arguments are required.")
    return models


def open_text_csv(path: Path):
    return path.open("r", encoding="utf-8-sig", newline="")


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


def column_index(fieldnames: Sequence[str], wanted: str) -> int:
    return list(fieldnames).index(require_column(fieldnames, wanted))


def optional_column_index(fieldnames: Sequence[str], wanted: str) -> int | None:
    column = normalized_column(fieldnames, wanted)
    return None if column is None else list(fieldnames).index(column)


def pairing_identity_columns(
    fieldnames: Sequence[str],
) -> tuple[int | None, int | None, int | None, int | None]:
    """Locate every supported identity column in a combined eval export."""
    columns = (
        optional_column_index(fieldnames, "recording_position"),
        optional_column_index(fieldnames, "base_name"),
        optional_column_index(fieldnames, "sample_id"),
        optional_column_index(fieldnames, "segment_idx"),
    )
    if any(index is not None for index in columns):
        return columns
    raise ValueError(
        f"Prediction CSV must contain recording_position, base_name, sample_id, "
        f"or segment_idx; "
        f"found {list(fieldnames)}"
    )


def pairing_identity(
    row: Sequence[str],
    columns: tuple[int | None, int | None, int | None, int | None],
    context: str,
) -> tuple[str, ...]:
    """Choose an identity per row, supporting heterogeneous dataset schemas."""
    (
        recording_position_index,
        base_name_index,
        sample_id_index,
        segment_idx_index,
    ) = columns
    recording_position = (
        row[recording_position_index].strip()
        if recording_position_index is not None
        else ""
    )
    if recording_position:
        return (recording_position,)

    base_name = (
        row[base_name_index].strip() if base_name_index is not None else ""
    )
    if base_name:
        # sample_id/segment_idx may be regenerated between otherwise identical
        # exports, so they are not part of the key when base_name is populated.
        return (base_name,)

    sample_id = row[sample_id_index].strip() if sample_id_index is not None else ""
    segment_idx = (
        row[segment_idx_index].strip() if segment_idx_index is not None else ""
    )
    if sample_id and segment_idx:
        return (sample_id, segment_idx)
    if sample_id:
        return (sample_id,)
    if segment_idx:
        return (segment_idx,)
    raise ValueError(
        f"Empty recording-position identity in {context}; expected a populated "
        "recording_position, base_name, or sample_id/segment_idx value"
    )


def recording_position_key(
    source_csv: str,
    identity: tuple[str, ...],
) -> str:
    """Build a collision-safe, stable key for one recording position."""
    return json.dumps((source_csv, *identity), ensure_ascii=False, separators=(",", ":"))


def canonical_reference_bytes(reference: np.ndarray) -> bytes:
    """Return a stable pairing representation tolerant of CSV float noise."""
    rounded = np.round(reference, decimals=REFERENCE_PAIRING_DECIMALS)
    # IEEE -0.0 and +0.0 compare equal but have different byte encodings.
    rounded[rounded == 0.0] = 0.0
    return rounded.astype("<f8", copy=False).tobytes()


def finite_vector_cells(
    row: Sequence[str],
    indices: Sequence[int],
    context: str,
) -> np.ndarray:
    try:
        vector = np.asarray([float(row[index]) for index in indices], dtype=np.float64)
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid vector in {context}") from error
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Non-finite 3D vector in {context}: {vector}")
    return vector


def iter_prediction_frames(
    path: Path,
    excluded_datasets: set[str],
) -> Iterator[
    tuple[str, str, tuple[str, ...], list[np.ndarray], list[np.ndarray], bytes]
]:
    """Yield frames grouped under a recording-position resampling key."""
    size_mib = path.stat().st_size / (1024 * 1024)
    print(
        f"  Opening prediction CSV {path} ({size_mib:.1f} MiB)...",
        flush=True,
    )
    with open_text_csv(path) as handle:
        reader = csv.reader(handle)
        fieldnames = next(reader, None)
        if not fieldnames:
            raise ValueError(f"Prediction CSV has no header: {path}")

        dataset_index = column_index(fieldnames, "prediction_dataset")
        source_index = column_index(fieldnames, "prediction_source_csv")
        frame_index_column = column_index(fieldnames, "frame_idx")
        slot_index = column_index(fieldnames, "slot")
        identity_columns = pairing_identity_columns(fieldnames)
        reference_indices = [
            column_index(fieldnames, name)
            for name in ("accdoa_x", "accdoa_y", "accdoa_z")
        ]
        prediction_indices = [
            column_index(fieldnames, name)
            for name in (
                "prediction_accdoa_x",
                "prediction_accdoa_y",
                "prediction_accdoa_z",
            )
        ]

        current_key: tuple[str, ...] | None = None
        current_sample_key: tuple[str, ...] | None = None
        current_frame_index = -1
        current_dataset = ""
        current_position = ""
        references: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        signature_parts: list[bytes] = []
        seen_slots: set[int] = set()
        completed_sample_keys: set[tuple[str, ...]] = set()
        rows_processed = 0
        frames_processed = 0
        excluded_rows = 0

        for row_number, row in enumerate(reader, start=2):
            rows_processed += 1
            if rows_processed % LOAD_PROGRESS_ROWS == 0:
                print(
                    f"  {path.name}: decompressed and parsed "
                    f"{rows_processed:,} CSV rows...",
                    flush=True,
                )
            try:
                dataset = row[dataset_index].strip()
                source_csv = row[source_index].strip()
                frame_index = int(row[frame_index_column])
            except (IndexError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid identity in {path} row {row_number}") from error
            if not dataset:
                raise ValueError(f"Empty prediction_dataset in {path} row {row_number}")
            if dataset in excluded_datasets:
                excluded_rows += 1
                continue
            if not source_csv:
                raise ValueError(
                    f"Empty prediction_source_csv in {path} row {row_number}"
                )
            try:
                identity = pairing_identity(
                    row,
                    identity_columns,
                    f"{path} row {row_number}",
                )
            except (IndexError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid identity in {path} row {row_number}") from error
            position = recording_position_key(source_csv, identity)
            sample_key = (dataset, source_csv, *identity)
            key = (*sample_key, str(frame_index))

            if current_key is not None and key != current_key:
                assert current_sample_key is not None
                if sample_key == current_sample_key:
                    if frame_index <= current_frame_index:
                        raise ValueError(
                            f"Frame index is repeated or decreasing for sample "
                            f"{sample_key} in {path} row {row_number}"
                        )
                else:
                    completed_sample_keys.add(current_sample_key)
                    if sample_key in completed_sample_keys:
                        raise ValueError(
                            f"Sample {sample_key} is not contiguous in {path} "
                            f"(row {row_number})"
                        )
                frames_processed += 1
                yield (
                    current_dataset,
                    current_position,
                    current_key,
                    references,
                    predictions,
                    b"".join(signature_parts),
                )
                references, predictions, signature_parts = [], [], []
                seen_slots = set()

            if current_key is None or key != current_key:
                current_key = key
                current_sample_key = sample_key
                current_frame_index = frame_index
                current_dataset = dataset
                current_position = position
                signature_parts.append(
                    json.dumps(key, separators=(",", ":")).encode("utf-8")
                )

            try:
                slot = int(row[slot_index])
            except (IndexError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid slot in {path} row {row_number}") from error
            if slot in seen_slots:
                raise ValueError(f"Duplicate slot {slot} for frame {key} in {path}")
            seen_slots.add(slot)

            reference = finite_vector_cells(
                row, reference_indices, f"{path} row {row_number} reference"
            )
            prediction = finite_vector_cells(
                row, prediction_indices, f"{path} row {row_number} prediction"
            )
            references.append(reference)
            predictions.append(prediction)
            signature_parts.append(
                slot.to_bytes(4, byteorder="little", signed=True)
                + canonical_reference_bytes(reference)
            )

        if current_key is not None:
            frames_processed += 1
            yield (
                current_dataset,
                current_position,
                current_key,
                references,
                predictions,
                b"".join(signature_parts),
            )

        print(
            f"  Finished {path.name}: {rows_processed:,} rows, "
            f"{frames_processed:,} included frames, "
            f"{excluded_rows:,} excluded rows.",
            flush=True,
        )


def angular_cost(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    reference_unit = reference / np.maximum(
        np.linalg.norm(reference, axis=1, keepdims=True), EPS
    )
    prediction_unit = prediction / np.maximum(
        np.linalg.norm(prediction, axis=1, keepdims=True), EPS
    )
    cosine = np.clip(reference_unit @ prediction_unit.T, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


@lru_cache(maxsize=None)
def small_assignment_candidates(
    number_of_references: int,
    number_of_predictions: int,
) -> np.ndarray:
    """Precompute every rectangular assignment for at most three tracks."""
    matches = min(number_of_references, number_of_predictions)
    candidates: list[list[tuple[int, int]]] = []
    if number_of_references <= number_of_predictions:
        for prediction_order in iter_permutations(
            range(number_of_predictions),
            matches,
        ):
            candidates.append(
                list(zip(range(number_of_references), prediction_order))
            )
    else:
        for reference_order in iter_permutations(
            range(number_of_references),
            matches,
        ):
            candidates.append(
                list(zip(reference_order, range(number_of_predictions)))
            )
    return np.asarray(candidates, dtype=np.int64)


def minimum_assignment_angles(cost: np.ndarray) -> np.ndarray:
    """Solve common 1-3 track cases without a per-frame SciPy call."""
    number_of_references, number_of_predictions = cost.shape
    if max(number_of_references, number_of_predictions) > 3:
        reference_indices, prediction_indices = linear_sum_assignment(cost)
        return cost[reference_indices, prediction_indices]

    candidates = small_assignment_candidates(
        number_of_references,
        number_of_predictions,
    )
    candidate_costs = cost[
        candidates[:, :, 0],
        candidates[:, :, 1],
    ].sum(axis=1)
    best = candidates[int(np.argmin(candidate_costs))]
    return cost[best[:, 0], best[:, 1]]


def evaluate_frame(
    references: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    activity_threshold: float,
    reference_activity_threshold: float,
    spatial_threshold_deg: float,
) -> SufficientStatistics:
    reference_array = np.asarray(references, dtype=np.float64)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    active_reference = reference_array[
        np.linalg.norm(reference_array, axis=1) > reference_activity_threshold
    ]
    active_prediction = prediction_array[
        np.linalg.norm(prediction_array, axis=1) >= activity_threshold
    ]
    num_reference = len(active_reference)
    num_prediction = len(active_prediction)

    stats = SufficientStatistics(
        frames=1,
        reference_events=num_reference,
        prediction_events=num_prediction,
        count_absolute_error_sum=abs(num_reference - num_prediction),
        exact_count_frames=int(num_reference == num_prediction),
    )

    if num_reference == 0 or num_prediction == 0:
        stats.source_false_positives = num_prediction
        stats.source_false_negatives = num_reference
        stats.spatial_false_positives = num_prediction
        stats.spatial_false_negatives = num_reference
        return stats

    cost = angular_cost(active_reference, active_prediction)
    matched_angles = minimum_assignment_angles(cost)
    matches = len(matched_angles)
    close_matches = int(np.sum(matched_angles <= spatial_threshold_deg))
    far_matches = matches - close_matches
    unmatched_reference = num_reference - matches
    unmatched_prediction = num_prediction - matches

    stats.localization_matches = matches
    stats.localization_error_sum = float(np.sum(matched_angles))
    stats.source_true_positives = matches
    stats.source_false_positives = unmatched_prediction
    stats.source_false_negatives = unmatched_reference
    stats.spatial_true_positives = close_matches
    stats.spatial_false_positives = far_matches + unmatched_prediction
    stats.spatial_false_negatives = far_matches + unmatched_reference
    stats.spatial_substitutions = far_matches
    return stats


def load_model(
    label: str,
    path: Path,
    activity_threshold: float,
    reference_activity_threshold: float,
    spatial_threshold_deg: float,
    excluded_datasets: set[str],
) -> dict[str, ModelData]:
    stats_by_dataset: dict[str, dict[str, SufficientStatistics]] = {}
    hashers_by_dataset: dict[str, dict[str, hashlib._Hash]] = {}

    for (
        dataset,
        position,
        _,
        references,
        predictions,
        signature,
    ) in iter_prediction_frames(path, excluded_datasets):
        stats_by_position = stats_by_dataset.setdefault(dataset, {})
        hashers = hashers_by_dataset.setdefault(dataset, {})
        position_stats = stats_by_position.setdefault(
            position, SufficientStatistics()
        )
        position_stats.add(
            evaluate_frame(
                references,
                predictions,
                activity_threshold,
                reference_activity_threshold,
                spatial_threshold_deg,
            )
        )
        hasher = hashers.setdefault(position, hashlib.sha256())
        hasher.update(signature)
        hasher.update(b"\n--frame--\n")

    if not stats_by_dataset:
        raise ValueError(f"No prediction frames found in {path}")
    return {
        dataset: ModelData(
            label=label,
            dataset=dataset,
            path=path,
            statistics_by_position=statistics,
            signatures_by_position={
                key: value.hexdigest()
                for key, value in hashers_by_dataset[dataset].items()
            },
        )
        for dataset, statistics in stats_by_dataset.items()
    }


def validate_dataset_sets(
    models_by_label: dict[str, dict[str, ModelData]],
) -> list[str]:
    """Require every model export to contain exactly the same datasets."""
    labels = list(models_by_label)
    reference_label = labels[0]
    reference_datasets = set(models_by_label[reference_label])
    for label in labels[1:]:
        datasets = set(models_by_label[label])
        if datasets != reference_datasets:
            missing = sorted(reference_datasets - datasets)
            extra = sorted(datasets - reference_datasets)
            raise ValueError(
                f"Dataset mismatch for {label}: missing={missing}, extra={extra}"
            )
    return sorted(reference_datasets)


def validate_pairing(models: Sequence[ModelData]) -> list[str]:
    reference = models[0]
    if any(model.dataset != reference.dataset for model in models[1:]):
        raise ValueError("validate_pairing received models from different datasets")
    reference_positions = set(reference.statistics_by_position)
    for model in models[1:]:
        positions = set(model.statistics_by_position)
        if positions != reference_positions:
            missing = sorted(reference_positions - positions)[:5]
            extra = sorted(positions - reference_positions)[:5]
            raise ValueError(
                f"Recording-position mismatch for {model.label} in dataset "
                f"{reference.dataset}: missing={missing}, extra={extra}"
            )
        mismatched = [
            position
            for position in reference_positions
            if model.signatures_by_position[position]
            != reference.signatures_by_position[position]
        ]
        if mismatched:
            raise ValueError(
                f"Reference metadata/frame mismatch for {model.label} in dataset "
                f"{reference.dataset}, "
                f"recording position(s): {sorted(mismatched)[:5]}"
            )
    return sorted(reference_positions)


def safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def metrics_from_statistics(stats: SufficientStatistics) -> dict[str, float]:
    le = (
        math.nan
        if stats.localization_matches == 0
        else stats.localization_error_sum / stats.localization_matches
    )
    lr = safe_ratio(
        stats.source_true_positives,
        stats.source_true_positives + stats.source_false_negatives,
    )
    precision_15 = safe_ratio(
        stats.spatial_true_positives,
        stats.spatial_true_positives + stats.spatial_false_positives,
    )
    recall_15 = safe_ratio(
        stats.spatial_true_positives,
        stats.spatial_true_positives + stats.source_false_negatives,
    )
    f15 = safe_ratio(2.0 * precision_15 * recall_15, precision_15 + recall_15)
    er15 = safe_ratio(
        stats.spatial_substitutions
        + stats.source_false_negatives
        + stats.source_false_positives,
        stats.reference_events,
    )
    esdl = math.nan if math.isnan(le) else 0.25 * (
        er15 + (1.0 - f15) + le / 180.0 + (1.0 - lr)
    )
    return {
        "LE": le,
        "LR": lr,
        "F15": f15,
        "ER15": er15,
        "ESDL": esdl,
        "count_MAE": safe_ratio(stats.count_absolute_error_sum, stats.frames),
        "count_accuracy": safe_ratio(stats.exact_count_frames, stats.frames),
    }


def aggregate_arrays(arrays: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
    return arrays.sum(axis=0) if indices is None else arrays[indices].sum(axis=0)


def metric_from_array(values: np.ndarray, metric: str) -> float:
    return metrics_from_statistics(SufficientStatistics.from_array(values))[metric]


def safe_ratio_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Vectorized safe division with the scalar implementation's zero rule."""
    numerator_array, denominator_array = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    result = np.zeros(numerator_array.shape, dtype=np.float64)
    np.divide(
        numerator_array,
        denominator_array,
        out=result,
        where=denominator_array != 0,
    )
    return result


def metrics_from_arrays(values: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate every metric for one or many sufficient-statistic vectors."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != len(SufficientStatistics.__dataclass_fields__):
        raise ValueError(f"Unexpected statistics array shape: {values.shape}")

    frames = values[..., 0]
    reference_events = values[..., 1]
    localization_matches = values[..., 3]
    localization_error_sum = values[..., 4]
    source_true_positives = values[..., 5]
    source_false_positives = values[..., 6]
    source_false_negatives = values[..., 7]
    spatial_true_positives = values[..., 8]
    spatial_false_positives = values[..., 9]
    spatial_false_negatives = values[..., 10]
    spatial_substitutions = values[..., 11]
    count_absolute_error_sum = values[..., 12]
    exact_count_frames = values[..., 13]

    le = np.full(np.shape(localization_matches), np.nan, dtype=np.float64)
    np.divide(
        localization_error_sum,
        localization_matches,
        out=le,
        where=localization_matches != 0,
    )
    lr = safe_ratio_array(
        source_true_positives,
        source_true_positives + source_false_negatives,
    )
    precision_15 = safe_ratio_array(
        spatial_true_positives,
        spatial_true_positives + spatial_false_positives,
    )
    recall_15 = safe_ratio_array(
        spatial_true_positives,
        spatial_true_positives + source_false_negatives,
    )
    f15 = safe_ratio_array(
        2.0 * precision_15 * recall_15,
        precision_15 + recall_15,
    )
    er15 = safe_ratio_array(
        spatial_substitutions + source_false_negatives + source_false_positives,
        reference_events,
    )
    esdl = 0.25 * (er15 + (1.0 - f15) + le / 180.0 + (1.0 - lr))
    return {
        "LE": le,
        "LR": lr,
        "F15": f15,
        "ER15": er15,
        "ESDL": esdl,
        "count_MAE": safe_ratio_array(count_absolute_error_sum, frames),
        "count_accuracy": safe_ratio_array(exact_count_frames, frames),
    }


def finite_percentile(values: Sequence[float], percentiles: tuple[float, float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return math.nan, math.nan
    low, high = np.percentile(finite, percentiles)
    return float(low), float(high)


def compare_pair(
    model_a: ModelData,
    model_b: ModelData,
    positions: Sequence[str],
    metrics: Sequence[str],
    bootstrap_samples: int,
    permutations: int,
    resampling_batch_size: int,
    rng: np.random.Generator,
) -> list[ComparisonRow]:
    if model_a.dataset != model_b.dataset:
        raise ValueError("Cannot compare models from different datasets")
    arrays_a = np.stack(
        [model_a.statistics_by_position[position].as_array() for position in positions]
    )
    arrays_b = np.stack(
        [model_b.statistics_by_position[position].as_array() for position in positions]
    )
    aggregate_a = arrays_a.sum(axis=0)
    aggregate_b = arrays_b.sum(axis=0)
    number_of_positions = len(positions)
    rows: list[ComparisonRow] = []

    # Keep temporary bootstrap indexing arrays bounded even if a future export
    # uses thousands of recording-position resampling units.
    memory_limited_batch_size = max(1, 250_000 // number_of_positions)
    effective_batch_size = min(resampling_batch_size, memory_limited_batch_size)

    bootstrap_differences = {
        metric: np.empty(bootstrap_samples, dtype=np.float64)
        for metric in metrics
    }
    for start in range(0, bootstrap_samples, effective_batch_size):
        stop = min(start + effective_batch_size, bootstrap_samples)
        batch_size = stop - start
        sample = rng.integers(
            0,
            number_of_positions,
            size=(batch_size, number_of_positions),
        )
        sampled_a = arrays_a[sample].sum(axis=1)
        sampled_b = arrays_b[sample].sum(axis=1)
        sampled_metrics_a = metrics_from_arrays(sampled_a)
        sampled_metrics_b = metrics_from_arrays(sampled_b)
        for metric in metrics:
            bootstrap_differences[metric][start:stop] = (
                sampled_metrics_a[metric] - sampled_metrics_b[metric]
            )

    aggregate_metrics_a = metrics_from_arrays(aggregate_a)
    aggregate_metrics_b = metrics_from_arrays(aggregate_b)
    observed = {
        metric: float(aggregate_metrics_a[metric] - aggregate_metrics_b[metric])
        for metric in metrics
    }
    extreme_counts = {metric: 0 for metric in metrics}
    valid_counts = {metric: 0 for metric in metrics}
    position_differences = arrays_b - arrays_a
    for start in range(0, permutations, effective_batch_size):
        stop = min(start + effective_batch_size, permutations)
        batch_size = stop - start
        swap = rng.integers(
            0,
            2,
            size=(batch_size, number_of_positions),
        )
        swapped_differences = swap.astype(np.float64) @ position_differences
        permuted_a = aggregate_a + swapped_differences
        permuted_b = aggregate_b - swapped_differences
        permuted_metrics_a = metrics_from_arrays(permuted_a)
        permuted_metrics_b = metrics_from_arrays(permuted_b)
        for metric in metrics:
            differences = permuted_metrics_a[metric] - permuted_metrics_b[metric]
            valid = np.isfinite(differences)
            if math.isfinite(observed[metric]):
                valid_counts[metric] += int(np.count_nonzero(valid))
                extreme_counts[metric] += int(
                    np.count_nonzero(
                        valid
                        & (np.abs(differences) >= abs(observed[metric]) - EPS)
                    )
                )

    for metric in metrics:
        estimate_a = float(aggregate_metrics_a[metric])
        estimate_b = float(aggregate_metrics_b[metric])
        ci_low, ci_high = finite_percentile(
            bootstrap_differences[metric], (2.5, 97.5)
        )
        valid = valid_counts[metric]
        p_value = (
            math.nan
            if valid == 0
            else (extreme_counts[metric] + 1.0) / (valid + 1.0)
        )
        rows.append(
            ComparisonRow(
                dataset=model_a.dataset,
                model_a=model_a.label,
                model_b=model_b.label,
                metric=metric,
                estimate_a=estimate_a,
                estimate_b=estimate_b,
                difference_a_minus_b=observed[metric],
                ci_low=ci_low,
                ci_high=ci_high,
                p_value=p_value,
            )
        )
    return rows


def apply_holm_correction(rows: list[ComparisonRow], alpha: float) -> None:
    valid = [(index, row.p_value) for index, row in enumerate(rows) if math.isfinite(row.p_value)]
    valid.sort(key=lambda item: item[1])
    number_of_tests = len(valid)
    running_max = 0.0
    for rank, (index, p_value) in enumerate(valid):
        adjusted = min(1.0, p_value * (number_of_tests - rank))
        running_max = max(running_max, adjusted)
        rows[index].p_value_holm = running_max

    for row in rows:
        row.significant = bool(
            math.isfinite(row.p_value_holm) and row.p_value_holm < alpha
        )
        if not row.significant or row.difference_a_minus_b == 0:
            row.better_model = ""
        elif row.metric in LOWER_IS_BETTER:
            row.better_model = (
                row.model_a if row.difference_a_minus_b < 0 else row.model_b
            )
        else:
            row.better_model = (
                row.model_a if row.difference_a_minus_b > 0 else row.model_b
            )


def aggregate_model(model: ModelData, positions: Sequence[str]) -> SufficientStatistics:
    aggregate = SufficientStatistics()
    for position in positions:
        aggregate.add(model.statistics_by_position[position])
    return aggregate


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
    """Write a wide matrix of per-dataset Holm-adjusted p-values."""
    pair_columns = [pair_label(model_a, model_b) for model_a, model_b in model_pairs]
    lookup = {
        (row.dataset, row.metric, row.model_a, row.model_b): row
        for row in comparison_rows
    }
    matrix_rows: list[dict[str, object]] = []
    for dataset in datasets:
        for metric in metrics:
            output_row: dict[str, object] = {
                "dataset": dataset,
                "metric": metric,
            }
            for (model_a, model_b), column in zip(model_pairs, pair_columns):
                row = lookup.get((dataset, metric, model_a, model_b))
                output_row[column] = (
                    ""
                    if row is None or not math.isfinite(row.p_value_holm)
                    else row.p_value_holm
                )
            matrix_rows.append(output_row)
    write_csv(path, matrix_rows, ("dataset", "metric", *pair_columns))


def plot_significance_heatmap(
    path: Path,
    comparison_rows: Sequence[ComparisonRow],
    datasets: Sequence[str],
    metrics: Sequence[str],
    model_pairs: Sequence[tuple[str, str]],
) -> None:
    """
    Plot significant direction and strength for every dataset/metric/model pair.

    Positive (red) cells mean model A is significantly better; negative (blue)
    cells mean model B is significantly better. Non-significant cells are zero
    and annotated ``ns``. Strength is -log10(Holm-adjusted p).
    """
    lookup = {
        (row.dataset, row.metric, row.model_a, row.model_b): row
        for row in comparison_rows
    }
    row_keys = [(dataset, metric) for dataset in datasets for metric in metrics]
    values = np.full((len(row_keys), len(model_pairs)), np.nan, dtype=np.float64)
    annotations: list[list[str]] = [
        ["NA" for _ in model_pairs]
        for _ in row_keys
    ]

    for row_index, (dataset, metric) in enumerate(row_keys):
        for column_index, (model_a, model_b) in enumerate(model_pairs):
            row = lookup.get((dataset, metric, model_a, model_b))
            if row is None or not math.isfinite(row.p_value_holm):
                continue
            adjusted_p = max(row.p_value_holm, np.finfo(np.float64).tiny)
            if row.significant and row.better_model:
                direction = 1.0 if row.better_model == model_a else -1.0
                values[row_index, column_index] = direction * -math.log10(adjusted_p)
                winner = "A" if direction > 0 else "B"
                annotations[row_index][column_index] = f"{winner}*\np={row.p_value_holm:.3g}"
            else:
                values[row_index, column_index] = 0.0
                annotations[row_index][column_index] = f"ns\np={row.p_value_holm:.3g}"

    finite_strength = np.abs(values[np.isfinite(values)])
    color_limit = max(1.5, float(finite_strength.max(initial=0.0)))
    figure_width = max(8.0, 2.4 * len(model_pairs) + 4.0)
    figure_height = max(5.0, 0.56 * len(row_keys) + 2.2)
    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )
    image = axis.imshow(
        np.ma.masked_invalid(values),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit),
        aspect="auto",
    )
    axis.set_xticks(range(len(model_pairs)))
    axis.set_xticklabels(
        [pair_label(model_a, model_b) for model_a, model_b in model_pairs],
        rotation=30,
        ha="right",
    )
    axis.set_yticks(range(len(row_keys)))
    axis.set_yticklabels([f"{dataset} | {metric}" for dataset, metric in row_keys])
    axis.set_title("Per-dataset pairwise significance")
    axis.set_xlabel("Model A vs model B")
    axis.set_ylabel("Dataset and metric")
    for boundary in range(len(metrics), len(row_keys), len(metrics)):
        axis.axhline(boundary - 0.5, color="black", linewidth=1.2)

    for row_index in range(len(row_keys)):
        for column_index in range(len(model_pairs)):
            value = values[row_index, column_index]
            text_color = "white" if math.isfinite(value) and abs(value) > color_limit * 0.55 else "black"
            axis.text(
                column_index,
                row_index,
                annotations[row_index][column_index],
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    colorbar = figure.colorbar(image, ax=axis, shrink=0.75)
    colorbar.set_label(
        "Signed -log10(Holm p): red = model A better, blue = model B better"
    )
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1 or args.permutations < 1:
        raise ValueError("--bootstrap-samples and --permutations must be positive")
    if args.resampling_batch_size < 1:
        raise ValueError("--resampling-batch-size must be positive")
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("--alpha must be between 0 and 1")

    model_specs = parse_model_arguments(args.model)
    excluded_datasets = {name.strip() for name in args.exclude_dataset if name.strip()}
    if excluded_datasets:
        print(
            "Excluding prediction dataset(s): "
            + ", ".join(sorted(excluded_datasets)),
            flush=True,
        )
    models_by_label: dict[str, dict[str, ModelData]] = {}
    for model_number, (label, path) in enumerate(model_specs, start=1):
        print(
            f"Loading model {model_number}/{len(model_specs)}: {label}",
            flush=True,
        )
        models_by_label[label] = load_model(
            label,
            path,
            args.activity_threshold,
            args.reference_activity_threshold,
            args.spatial_threshold_deg,
            excluded_datasets,
        )
        print(f"Finished loading model: {label}", flush=True)
    datasets = validate_dataset_sets(models_by_label)
    model_labels = [label for label, _ in model_specs]
    model_pairs = list(combinations(model_labels, 2))
    print(
        f"Loaded {len(model_labels)} models with {len(datasets)} shared "
        "datasets."
    )

    rng = np.random.default_rng(args.seed)
    comparison_rows: list[ComparisonRow] = []
    model_metric_rows: list[dict[str, object]] = []
    dataset_payload: dict[str, object] = {}

    for dataset in datasets:
        dataset_models = [models_by_label[label][dataset] for label in model_labels]
        positions = validate_pairing(dataset_models)
        print(
            f"Dataset {dataset}: {len(positions)} paired recording positions."
        )

        dataset_comparisons: list[ComparisonRow] = []
        for model_a, model_b in combinations(dataset_models, 2):
            print(
                f"  Comparing {model_a.label} vs {model_b.label} on "
                f"{dataset}..."
            )
            dataset_comparisons.extend(
                compare_pair(
                    model_a,
                    model_b,
                    positions,
                    args.metrics,
                    args.bootstrap_samples,
                    args.permutations,
                    args.resampling_batch_size,
                    rng,
                )
            )

        # Each dataset is a separate family of hypotheses, as requested.
        apply_holm_correction(dataset_comparisons, args.alpha)
        comparison_rows.extend(dataset_comparisons)

        per_model_payload: dict[str, object] = {}
        for model in dataset_models:
            stats = aggregate_model(model, positions)
            metric_values = metrics_from_statistics(stats)
            per_model_payload[model.label] = {
                "prediction_file": str(model.path),
                "metrics": {
                    key: json_number(value)
                    for key, value in metric_values.items()
                },
                "sufficient_statistics": asdict(stats),
            }
            for metric, value in metric_values.items():
                model_metric_rows.append(
                    {
                        "dataset": dataset,
                        "model": model.label,
                        "metric": metric,
                        "value": value,
                    }
                )

        dataset_payload[dataset] = {
            "number_of_paired_recording_positions": len(positions),
            "models": per_model_payload,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dicts = [asdict(row) for row in comparison_rows]
    model_metrics_path = args.output_dir / "model_metrics.csv"
    pairwise_path = args.output_dir / "pairwise_significance.csv"
    matrix_path = args.output_dir / "significance_matrix.csv"
    heatmap_path = args.output_dir / "significance_heatmap.png"
    json_path = args.output_dir / "significance_results.json"
    write_csv(
        model_metrics_path,
        model_metric_rows,
        ("dataset", "model", "metric", "value"),
    )
    write_csv(pairwise_path, comparison_dicts, tuple(ComparisonRow.__dataclass_fields__))
    write_significance_matrix(
        matrix_path,
        comparison_rows,
        datasets,
        args.metrics,
        model_pairs,
    )
    plot_significance_heatmap(
        heatmap_path,
        comparison_rows,
        datasets,
        args.metrics,
        model_pairs,
    )

    payload = {
        "method": {
            "activity_threshold": args.activity_threshold,
            "reference_activity_threshold": args.reference_activity_threshold,
            "spatial_threshold_deg": args.spatial_threshold_deg,
            "excluded_datasets": sorted(excluded_datasets),
            "resampling_unit": "recording position within prediction_source_csv",
            "bootstrap_samples": args.bootstrap_samples,
            "permutations": args.permutations,
            "resampling_batch_size": args.resampling_batch_size,
            "confidence_interval": (
                "paired recording-position bootstrap percentile 95%"
            ),
            "p_value": (
                "two-sided paired recording-position label-permutation test"
            ),
            "multiple_testing": (
                "Holm correction across all pairwise model/metric tests, "
                "applied separately within each dataset"
            ),
            "alpha": args.alpha,
            "seed": args.seed,
        },
        "number_of_models": len(model_labels),
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
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)

    print(f"Saved model metrics:          {model_metrics_path}")
    print(f"Saved pairwise comparisons:   {pairwise_path}")
    print(f"Saved significance matrix:    {matrix_path}")
    print(f"Saved significance heatmap:   {heatmap_path}")
    print(f"Saved full results:           {json_path}")


if __name__ == "__main__":
    main()
