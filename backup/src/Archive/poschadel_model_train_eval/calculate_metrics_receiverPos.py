#!/usr/bin/env python3
"""Calculate SDL metrics for every receiver position in a prediction export.

The input is one combined ``predictions_<model>.csv`` written by the evaluation
script. Metric definitions follow the supplied ``significance_test`` script
exactly. All segments at the same receiver position are accumulated before the
metrics are calculated. One output row is written per ``prediction_dataset`` /
receiver-position combination.

The output contains:

* LE, LR, F15, ER15, ESDL, count_MAE, and count_accuracy;
* the underlying counts used to calculate those metrics;
* constant, non-framewise columns from the original CSV (room metadata is
  therefore carried through without requiring hard-coded column names); and
* ``mean_rt60``, averaged over all three sources; and
* the number of aggregated segments and source CSV files.

RT60 layouts supported automatically:

1. Three columns whose names contain ``rt60`` (one column per source).
2. One ``rt60`` column, with the three sources distinguished by ``slot``.

If ``rt60_s`` exists, it is preferred over frequency-band columns such as
``rt60_125Hz``. Use ``--rt60-column`` to override that choice.

Receiver-position identity is selected in this order:

1. ``recording_position`` when present;
2. room dimensions plus microphone coordinates
   (``x_room/y_room/z_room/x_mic/y_mic/z_mic``); or
3. ``base_name`` as a compatibility fallback.

Use repeated ``--receiver-position-column`` arguments to override the identity
columns. Segment identity remains separate, so repeated frame indices in
different segments cannot collide.

Example:
    python metrics_receiver_position.py \
        evaluation_results/predictions_my_model.csv

This creates:
    evaluation_results/metrics_receiverPos_my_model.csv

Requirements:
    pip install numpy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from typing import Sequence

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

# These columns are observations rather than position/room metadata.  They are
# deliberately excluded even when a particular receiver position happens to
# contain only one value for them.
NON_METADATA_NAMES = {
    "frame_idx",
    "slot",
    "accdoa_x",
    "accdoa_y",
    "accdoa_z",
    "prediction_accdoa_x",
    "prediction_accdoa_y",
    "prediction_accdoa_z",
    "prediction_activity",
    "prediction_active",
    "prediction_az_deg",
    "prediction_el_deg",
}


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
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class PositionData:
    dataset: str
    identity_columns: tuple[str, ...]
    identity: tuple[str, ...]
    statistics: SufficientStatistics
    constant_metadata: dict[str, str]
    varying_metadata: set[str]
    source_csvs: set[str]
    segments: set[tuple[str, ...]]
    # Values are accumulated by RT60 column and slot.  Averaging within a slot
    # removes frame duplication before the final across-source mean.
    rt60_by_column_slot: dict[str, dict[str, "MeanAccumulator"]]


@dataclass
class MeanAccumulator:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        self.total += value
        self.count += 1

    def mean(self) -> float:
        if self.count == 0:
            return math.nan
        return self.total / self.count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions",
        nargs="?",
        type=Path,
        help="Combined predictions_<model>.csv file.",
    )
    parser.add_argument(
        "--predictions",
        dest="predictions_option",
        type=Path,
        help="Named alternative to the positional predictions CSV argument.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV. By default it is written beside the input as "
            "metrics_receiverPos_<model>.csv."
        ),
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=[],
        metavar="NAME",
        help="prediction_dataset value to skip; repeat to exclude more.",
    )
    parser.add_argument("--activity-threshold", type=float, default=0.5)
    parser.add_argument("--reference-activity-threshold", type=float, default=1e-6)
    parser.add_argument("--spatial-threshold-deg", type=float, default=15.0)
    parser.add_argument(
        "--rt60-column",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "RT60 column; repeat for each source. If omitted, columns whose "
            "normalized names contain 'rt60' are detected automatically."
        ),
    )
    parser.add_argument(
        "--receiver-position-column",
        action="append",
        default=["sample_id"],
        metavar="NAME",
        help=(
            "Column forming the receiver-position key; repeat for compound "
            "keys. By default, recording_position or room/mic coordinates are "
            "used."
        ),
    )
    parser.add_argument(
        "--expected-rt60-sources",
        type=int,
        default=3,
        help="Required number of RT60 source values per receiver position.",
    )
    parser.add_argument(
        "--allow-missing-rt60",
        action="store_true",
        help="Write an empty mean_rt60 instead of failing when RT60 is unavailable.",
    )
    return parser.parse_args()


def normalized_name(name: str) -> str:
    return name.replace("\\", "").strip().strip("*").lower()


def resolve_column(fieldnames: Sequence[str], wanted: str) -> str | None:
    wanted_normalized = normalized_name(wanted)
    for name in fieldnames:
        if normalized_name(name) == wanted_normalized:
            return name
    return None


def require_column(fieldnames: Sequence[str], wanted: str) -> str:
    result = resolve_column(fieldnames, wanted)
    if result is None:
        raise ValueError(f"Missing column {wanted!r}; found {list(fieldnames)}")
    return result


def resolve_receiver_identity_columns(
    fieldnames: Sequence[str], requested: Sequence[str] | None
) -> tuple[tuple[str, ...], ...]:
    """Resolve ordered, row-level strategies for receiver identity.

    Combined prediction exports can have a union header: coordinate columns
    may exist in the header while being empty for an entire constituent
    dataset. Returning multiple strategies lets each row use the richest
    populated identity available.
    """
    if requested:
        columns: list[str] = []
        for wanted in requested:
            column = resolve_column(fieldnames, wanted)
            if column is None:
                raise ValueError(
                    f"Requested receiver-position column {wanted!r} was not found"
                )
            if column not in columns:
                columns.append(column)
        return (tuple(columns),)

    strategies: list[tuple[str, ...]] = []
    recording_position = resolve_column(fieldnames, "recording_position")
    if recording_position is not None:
        strategies.append((recording_position,))

    mic_columns = tuple(
        resolve_column(fieldnames, name) for name in ("x_mic", "y_mic", "z_mic")
    )
    if all(mic_columns):
        room_columns = tuple(
            resolve_column(fieldnames, name)
            for name in ("x_room", "y_room", "z_room")
        )
        if all(room_columns):
            strategies.append(tuple((*room_columns, *mic_columns)))
        strategies.append(tuple(mic_columns))

    base_name = resolve_column(fieldnames, "base_name")
    if base_name is not None:
        strategies.append((base_name,))
    if not strategies:
        raise ValueError(
            "Cannot identify receiver positions. Expected recording_position, "
            "x_mic/y_mic/z_mic, or base_name, or pass "
            "--receiver-position-column explicitly."
        )
    return tuple(strategies)


def canonical_identity_value(raw: str) -> str:
    """Normalize equivalent coordinate spellings without losing text IDs."""
    value = raw.strip()
    if not value:
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if not math.isfinite(number):
        return value
    return format(number, ".12g")


def receiver_identity(
    row: dict[str, str], strategies: Sequence[Sequence[str]], context: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    for columns in strategies:
        values = tuple(
            canonical_identity_value(row.get(column) or "") for column in columns
        )
        if values and all(values):
            return tuple(columns), values
    attempted = [list(columns) for columns in strategies]
    raise ValueError(
        f"Empty receiver-position value in {context}; tried={attempted}"
    )


def resolve_segment_columns(fieldnames: Sequence[str]) -> tuple[str, ...]:
    columns = tuple(
        column
        for column in (
            resolve_column(fieldnames, "sample_id"),
            resolve_column(fieldnames, "segment_idx"),
            resolve_column(fieldnames, "base_name"),
        )
        if column is not None
    )
    if not columns:
        raise ValueError(
            "Prediction CSV must contain sample_id, segment_idx, or base_name "
            "to distinguish segments."
        )
    return columns


def segment_identity(
    row: dict[str, str], source_csv: str, columns: Sequence[str], context: str
) -> tuple[str, ...]:
    values = tuple((row.get(column) or "").strip() for column in columns)
    if not any(values):
        raise ValueError(f"Empty segment identity in {context}")
    return (source_csv, *values)


def finite_vector(row: dict[str, str], columns: Sequence[str], context: str) -> np.ndarray:
    try:
        vector = np.asarray([float(row[name]) for name in columns], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid vector in {context}") from error
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Non-finite 3D vector in {context}: {vector}")
    return vector


def angular_cost(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    reference_unit = reference / np.maximum(
        np.linalg.norm(reference, axis=1, keepdims=True), EPS
    )
    prediction_unit = prediction / np.maximum(
        np.linalg.norm(prediction, axis=1, keepdims=True), EPS
    )
    return np.degrees(np.arccos(np.clip(reference_unit @ prediction_unit.T, -1.0, 1.0)))


@lru_cache(maxsize=None)
def assignment_candidates(n_reference: int, n_prediction: int) -> np.ndarray:
    matches = min(n_reference, n_prediction)
    candidates: list[list[tuple[int, int]]] = []
    if n_reference <= n_prediction:
        for prediction_order in permutations(range(n_prediction), matches):
            candidates.append(list(zip(range(n_reference), prediction_order)))
    else:
        for reference_order in permutations(range(n_reference), matches):
            candidates.append(list(zip(reference_order, range(n_prediction))))
    return np.asarray(candidates, dtype=np.int64)


def minimum_assignment_angles(cost: np.ndarray) -> np.ndarray:
    """Return angles from the minimum-cost rectangular assignment.

    Prediction exports normally contain three tracks.  A small exhaustive
    solver keeps this script's only dependency at NumPy.  For an unusual export
    with more than eight tracks, fail explicitly rather than allocating a very
    large permutation table.
    """
    n_reference, n_prediction = cost.shape
    if max(n_reference, n_prediction) > 8:
        raise ValueError(
            "More than 8 active tracks are not supported; use a three-track "
            "ACCDOA prediction export or add scipy.optimize.linear_sum_assignment."
        )
    candidates = assignment_candidates(n_reference, n_prediction)
    candidate_costs = cost[candidates[:, :, 0], candidates[:, :, 1]].sum(axis=1)
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
    n_reference = len(active_reference)
    n_prediction = len(active_prediction)
    stats = SufficientStatistics(
        frames=1,
        reference_events=n_reference,
        prediction_events=n_prediction,
        count_absolute_error_sum=abs(n_reference - n_prediction),
        exact_count_frames=int(n_reference == n_prediction),
    )
    if n_reference == 0 or n_prediction == 0:
        stats.source_false_positives = n_prediction
        stats.source_false_negatives = n_reference
        stats.spatial_false_positives = n_prediction
        stats.spatial_false_negatives = n_reference
        return stats

    matched_angles = minimum_assignment_angles(angular_cost(active_reference, active_prediction))
    matches = len(matched_angles)
    close_matches = int(np.sum(matched_angles <= spatial_threshold_deg))
    far_matches = matches - close_matches
    unmatched_reference = n_reference - matches
    unmatched_prediction = n_prediction - matches
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
    # This deliberately mirrors significance_test: source FN is used here.
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


def default_output_path(predictions: Path) -> Path:
    prefix = "predictions_"
    model = predictions.stem[len(prefix):] if predictions.stem.startswith(prefix) else predictions.stem
    return predictions.with_name(f"metrics_receiverPos_{model}.csv")


def detect_rt60_columns(
    fieldnames: Sequence[str], requested: Sequence[str] | None
) -> list[str]:
    if requested:
        columns = []
        for name in requested:
            column = resolve_column(fieldnames, name)
            if column is None:
                raise ValueError(f"Requested RT60 column {name!r} was not found")
            if column not in columns:
                columns.append(column)
        return columns
    broadband = resolve_column(fieldnames, "rt60_s")
    if broadband is not None:
        return [broadband]
    return [name for name in fieldnames if "rt60" in normalized_name(name)]


def update_constant_metadata(
    position: PositionData, row: dict[str, str], candidates: Sequence[str]
) -> None:
    for name in candidates:
        if name in position.varying_metadata:
            continue
        value = (row.get(name) or "").strip()
        if name not in position.constant_metadata:
            position.constant_metadata[name] = value
        elif position.constant_metadata[name] != value:
            position.constant_metadata.pop(name, None)
            position.varying_metadata.add(name)


def update_rt60(
    position: PositionData,
    row: dict[str, str],
    rt60_columns: Sequence[str],
    slot_column: str,
    context: str,
) -> None:
    slot = (row.get(slot_column) or "").strip()
    for column in rt60_columns:
        raw = (row.get(column) or "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError(f"Non-numeric RT60 value {raw!r} in {context}") from error
        if not math.isfinite(value):
            raise ValueError(f"Non-finite RT60 value in {context}")
        accumulator = position.rt60_by_column_slot.setdefault(column, {}).setdefault(
            slot, MeanAccumulator()
        )
        accumulator.add(value)


def mean_rt60(position: PositionData, expected_sources: int) -> float:
    by_column = position.rt60_by_column_slot
    if not by_column:
        return math.nan

    if len(by_column) == expected_sources:
        # Three source-specific columns: collapse repeated frame/slot rows in
        # each column, then average one value per source.
        source_values = [
            safe_ratio(
                sum(accumulator.total for accumulator in slots.values()),
                sum(accumulator.count for accumulator in slots.values()),
            )
            for slots in by_column.values()
        ]
    elif len(by_column) == 1:
        # One RT60 column repeated down the CSV: one mean per source/slot.
        slots = next(iter(by_column.values()))
        source_values = [accumulator.mean() for _, accumulator in sorted(slots.items())]
    else:
        raise ValueError(
            f"Receiver position {position.identity!r} has {len(by_column)} RT60 "
            f"columns; expected either 1 or {expected_sources}."
        )
    if len(source_values) != expected_sources:
        raise ValueError(
            f"Receiver position {position.identity!r} supplies {len(source_values)} "
            f"RT60 source values; expected {expected_sources}."
        )
    return float(np.mean(source_values))


def calculate(args: argparse.Namespace) -> Path:
    if args.predictions is not None and args.predictions_option is not None:
        raise ValueError("Pass predictions either positionally or with --predictions, not both")
    predictions_argument = args.predictions_option or args.predictions
    if predictions_argument is None:
        raise ValueError("A predictions CSV is required")
    predictions_path = predictions_argument.expanduser().resolve()
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {predictions_path}")
    output_path = (
        args.output.expanduser().resolve() if args.output else default_output_path(predictions_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions: dict[
        tuple[str, tuple[str, ...], tuple[str, ...]], PositionData
    ] = {}
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Prediction CSV has no header: {predictions_path}")
        fieldnames = list(reader.fieldnames)
        dataset_column = require_column(fieldnames, "prediction_dataset")
        source_column = require_column(fieldnames, "prediction_source_csv")
        frame_column = require_column(fieldnames, "frame_idx")
        slot_column = require_column(fieldnames, "slot")
        identity_strategies = resolve_receiver_identity_columns(
            fieldnames, args.receiver_position_column
        )
        segment_columns = resolve_segment_columns(fieldnames)
        reference_columns = [require_column(fieldnames, name) for name in ("accdoa_x", "accdoa_y", "accdoa_z")]
        prediction_columns = [
            require_column(fieldnames, name)
            for name in ("prediction_accdoa_x", "prediction_accdoa_y", "prediction_accdoa_z")
        ]
        rt60_columns = detect_rt60_columns(fieldnames, args.rt60_column)
        excluded = set(args.exclude_dataset)
        metadata_candidates = [
            name
            for name in fieldnames
            if normalized_name(name) not in NON_METADATA_NAMES
            and name not in rt60_columns
            and name not in (dataset_column, source_column)
        ]

        current_frame_key: tuple[str, tuple[str, ...], int] | None = None
        current_position: PositionData | None = None
        references: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        seen_slots: set[int] = set()

        def finish_frame() -> None:
            nonlocal references, predictions, seen_slots
            if current_position is not None and references:
                current_position.statistics.add(
                    evaluate_frame(
                        references,
                        predictions,
                        args.activity_threshold,
                        args.reference_activity_threshold,
                        args.spatial_threshold_deg,
                    )
                )
            references, predictions, seen_slots = [], [], set()

        completed_frames: set[tuple[str, tuple[str, ...], int]] = set()
        for row_number, row in enumerate(reader, start=2):
            dataset = (row.get(dataset_column) or "").strip()
            if dataset in excluded:
                continue
            source_csv = (row.get(source_column) or "").strip()
            if not dataset or not source_csv:
                raise ValueError(f"Empty dataset/source identity in row {row_number}")
            identity_columns, identity = receiver_identity(
                row, identity_strategies, f"row {row_number}"
            )
            segment = segment_identity(
                row, source_csv, segment_columns, f"row {row_number}"
            )
            try:
                frame_index = int(row[frame_column])
                slot = int(row[slot_column])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid frame/slot in row {row_number}") from error
            # Include the column names so a textual base_name can never collide
            # with an otherwise identical coordinate value.
            identity_column_key = tuple(
                normalized_name(column) for column in identity_columns
            )
            position_key = (dataset, identity_column_key, identity)
            frame_key = (dataset, segment, frame_index)
            if current_frame_key is not None and frame_key != current_frame_key:
                completed_frames.add(current_frame_key)
                finish_frame()
            if frame_key != current_frame_key:
                if frame_key in completed_frames:
                    raise ValueError(
                        f"Frame {frame_key!r} is not contiguous (row {row_number})"
                    )
                current_frame_key = frame_key
                current_position = positions.get(position_key)
                if current_position is None:
                    current_position = PositionData(
                        dataset=dataset,
                        identity_columns=identity_columns,
                        identity=identity,
                        statistics=SufficientStatistics(),
                        constant_metadata={},
                        varying_metadata=set(),
                        source_csvs=set(),
                        segments=set(),
                        rt60_by_column_slot={},
                    )
                    positions[position_key] = current_position
            if slot in seen_slots:
                raise ValueError(f"Duplicate slot {slot} in frame {frame_key!r}")
            seen_slots.add(slot)
            assert current_position is not None
            current_position.source_csvs.add(source_csv)
            current_position.segments.add(segment)
            references.append(finite_vector(row, reference_columns, f"row {row_number} reference"))
            predictions.append(finite_vector(row, prediction_columns, f"row {row_number} prediction"))
            update_constant_metadata(current_position, row, metadata_candidates)
            update_rt60(current_position, row, rt60_columns, slot_column, f"row {row_number}")
        finish_frame()

    if not positions:
        raise ValueError("No included prediction rows were found")
    if not rt60_columns and not args.allow_missing_rt60:
        raise ValueError(
            "No RT60 column was detected. Use --rt60-column NAME, or "
            "--allow-missing-rt60 if this export has no RT60 metadata."
        )

    rt60_by_position: dict[
        tuple[str, tuple[str, ...], tuple[str, ...]], float
    ] = {}
    for position_key, position in positions.items():
        rt60 = mean_rt60(position, args.expected_rt60_sources) if rt60_columns else math.nan
        if math.isnan(rt60) and not args.allow_missing_rt60:
            raise ValueError(f"RT60 is missing for receiver position {position.identity!r}")
        rt60_by_position[position_key] = rt60

    # Preserve original order for every metadata field that is constant for at
    # least one position. Missing/non-constant fields are empty in that row.
    metadata_output_columns = [
        name
        for name in metadata_candidates
        if any(name in position.constant_metadata for position in positions.values())
    ]
    identity_output_columns = [
        "prediction_dataset",
        "prediction_source_csv",
        "receiver_position_key",
        "receiver_position",
        "receiver_position_columns",
        "segment_count",
        "source_csv_count",
    ]
    statistic_columns = list(SufficientStatistics.__dataclass_fields__)
    output_columns = (
        identity_output_columns
        + metadata_output_columns
        + ["mean_rt60"]
        + list(DEFAULT_METRICS)
        + statistic_columns
    )
    # Avoid duplicate identity/statistic names if they also appeared in the
    # constant metadata set.
    output_columns = list(dict.fromkeys(output_columns))

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns, extrasaction="ignore")
        writer.writeheader()
        for position_key in sorted(positions):
            position = positions[position_key]
            rt60 = rt60_by_position[position_key]
            row: dict[str, object] = {
                "prediction_dataset": position.dataset,
                "prediction_source_csv": ";".join(sorted(position.source_csvs)),
                "receiver_position_key": json.dumps(
                    (
                        position.dataset,
                        *(
                            (normalized_name(column), value)
                            for column, value in zip(
                                position.identity_columns, position.identity
                            )
                        ),
                    ),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "receiver_position": "|".join(
                    f"{normalized_name(column)}={value}"
                    for column, value in zip(
                        position.identity_columns, position.identity
                    )
                ),
                "receiver_position_columns": ";".join(position.identity_columns),
                "segment_count": len(position.segments),
                "source_csv_count": len(position.source_csvs),
                **position.constant_metadata,
                "mean_rt60": "" if math.isnan(rt60) else rt60,
                **metrics_from_statistics(position.statistics),
                **asdict(position.statistics),
            }
            writer.writerow(row)
    return output_path


def main() -> None:
    args = parse_args()
    output = calculate(args)
    print(f"Wrote receiver-position metrics: {output}")


if __name__ == "__main__":
    main()
