#!/usr/bin/env python3
"""Compare ground truth and predictions across one or more datasets.

Edit the configuration below or override it on the command line:

    python plot_room_position_predictions.py \
        --dataset model_a.csv model_b.csv model_c.csv \
        --sample-id R1800_P00 \
        --prediction-dataset checkpoint_dataset \
        --labels "Model A" "Model B" "Model C" \
        --ground-truth-slot-colors '0=#f4a6a6' '1=#a9c8ef' \
        --prediction-slot-colors '0=#e41a1c' '1=#377eb8' \
        --output R1800_P00_comparison.pdf

Each dataset becomes one column, with azimuth above elevation. A translucent
background spans each complete column, and one shared legend explains all traces.
Active predictions that cannot be matched to a ground-truth source are shown in
gray.
A single prediction-dataset value is applied to all files; alternatively,
provide one value per input file.

Supported inputs are CSV, TSV, Parquet, Excel, JSON, JSON Lines, and the
flattened Markdown table format included with the example dataset.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# Edit these values to run the script without command-line arguments.
DATASET_PATH = Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions_shoeBox_em32_ISMOrder12.csv")
DATASET_PATHS: tuple[Path, ...] = (DATASET_PATH,)  # Add more paths here.
DATASET_LABELS: tuple[str, ...] | None = None  # Example: ("Model A", "Model B")
SAMPLE_ID = "R1800_P00"
PREDICTION_DATASET: str | None = "checkpoint_dataset"
OUTPUT_PDF: Path | None = Path("predictions.pdf")
SLOTS: tuple[str, ...] | None = None  # Example: ("0", "2")
# Assign ground-truth and prediction colors independently for each source slot.
# Unlisted slots use tab20's light and dark color pairs, respectively. Colors
# may be hexadecimal values, named colors, or any Matplotlib color.
GROUND_TRUTH_SLOT_COLORS: dict[str, str] = {
    "0": "#5f64adff",
    "1": "#318245FF",
    "2": "#aa5757ff",
}
PREDICTION_SLOT_COLORS: dict[str, str] = {
    "0": "#3d4affff",
    "1": "#68CD63FF",
    "2": "#ff4343ff",
}
ACTIVITY_THRESHOLD = 0.5
CSV_CHUNK_SIZE = 200_000
COLUMN_BACKGROUND_COLORS = (
    "#d0d0d0",
    "#3bc2b9",
    "#ec4e4e",
)
COLUMN_BACKGROUND_ALPHA = 0.15
COLUMN_BACKGROUND_EXTEND_TOP = 0.07
COLUMN_BACKGROUND_EXTEND_BOTTOM = 0.25
COLUMN_BACKGROUND_EXTEND_SIDES = 0.005
FIGURE_TITLE_FONT_SIZE = 20
FIGURE_TITLE_Y = 1.015
SUBPLOT_TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 18
TICK_LABEL_FONT_SIZE = 18
LEGEND_FONT_SIZE = 18
GROUND_TRUTH_MARKER = "o"
GROUND_TRUTH_MARKER_SIZE = 5.5
GROUND_TRUTH_MARKER_EVERY = 5
GROUND_TRUTH_MARKER_EDGE_WIDTH = 1.25
LEGEND_GROUND_TRUTH_MARKER_EDGE_WIDTH = 2
PREDICTION_LINE_WIDTH = 2.7
LEGEND_LINE_WIDTH = 4
GROUND_TRUTH_ALPHA = 0.9
PREDICTION_ALPHA = 1.0
PREDICTION_OUTLINE_WIDTH = 0.0
UNMATCHED_PREDICTION_COLOR = "0.55"
UNMATCHED_PREDICTION_ALPHA = 0.9

# Edit these strings to change the shared legend text. The {slot} placeholder
# is replaced with the corresponding source slot number.
LEGEND_TEXT = {
    "ground_truth": "Ground truth",
    "prediction": "Prediction",
    "slot": "Speaker {slot}",
}
GROUND_TRUTH_COLUMNS = ("active", "az_deg", "el_deg")
PREDICTION_ANGLE_COLUMNS = ("prediction_az_deg", "prediction_el_deg")
PREDICTION_VECTOR_COLUMNS = (
    "prediction_accdoa_x",
    "prediction_accdoa_y",
    "prediction_accdoa_z",
)
PLOT_COLUMNS = {
    "sample_id",
    "prediction_dataset",
    "slot",
    "active",
    "az_deg",
    "el_deg",
    "prediction_active",
    "prediction_activity",
    *PREDICTION_ANGLE_COLUMNS,
    *PREDICTION_VECTOR_COLUMNS,
    "frame_start_s",
    "frame_start_sample",
    "sample_rate",
    "frame_idx",
    "segment_idx",
    "segment_start_s",
}


def parse_flattened_markdown(path: Path) -> pd.DataFrame:
    """Read the one-cell-per-table Markdown export shown in the example."""
    text = path.read_text(encoding="utf-8")
    cells = [
        cell.strip().replace(r"\_", "_")
        for cell in re.split(r"\|\s*\|\s*\n\|\s*-\s*\|\s*\n", text)[1:]
    ]
    final_header = "**prediction_el_deg**"
    try:
        column_count = cells.index(final_header) + 1
    except ValueError as error:
        raise ValueError(
            "Markdown input does not match the supplied flattened table format; "
            "use a CSV/Parquet file or include the prediction_el_deg header."
        ) from error

    columns = [cell.strip("*") for cell in cells[:column_count]]
    values = cells[column_count:]
    if len(values) % column_count:
        raise ValueError(
            f"Markdown table contains {len(values)} data cells, which cannot "
            f"be divided into rows of {column_count} columns."
        )

    rows = [
        [value.strip("*") for value in values[index : index + column_count]]
        for index in range(0, len(values), column_count)
    ]
    return pd.DataFrame(rows, columns=columns)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load one combined ground-truth/prediction dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")

    extension = path.suffix.lower()
    if extension == ".csv":
        return pd.read_csv(path, low_memory=False)
    if extension in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if extension in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as error:
            raise ImportError(
                "Reading Parquet requires pyarrow or fastparquet; install one "
                "with: pip install pyarrow"
            ) from error
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if extension == ".json":
        return pd.read_json(path)
    if extension in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if extension == ".md" or ".md " in path.name.lower():
        return parse_flattened_markdown(path)

    raise ValueError(
        f"Unsupported dataset format {extension!r}; expected CSV, TSV, "
        "Parquet, Excel, JSON, JSON Lines, or the supplied Markdown format."
    )


def load_selected_tabular_dataset(
    path: Path,
    sample_id: str,
    prediction_dataset: str | None,
    slots: Sequence[str] | None = None,
    chunk_size: int = CSV_CHUNK_SIZE,
) -> pd.DataFrame:
    """Stream a CSV/TSV and retain only columns and rows used by the plot."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    if chunk_size < 1:
        raise ValueError("CSV chunk size must be greater than zero.")

    separator = "," if path.suffix.lower() == ".csv" else "\t"
    header = pd.read_csv(path, sep=separator, nrows=0)
    required = {"sample_id", *GROUND_TRUTH_COLUMNS}
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    if prediction_dataset is not None and "prediction_dataset" not in header.columns:
        raise ValueError(
            "--prediction-dataset was specified, but the dataset has no "
            "prediction_dataset column. Set PREDICTION_DATASET = None or "
            "pass --all-prediction-datasets."
        )

    text_dtypes = {"sample_id": "string"}
    if "prediction_dataset" in header.columns:
        text_dtypes["prediction_dataset"] = "string"

    requested_slots = None if slots is None else {str(slot) for slot in slots}
    sample_found = False
    prediction_found = False
    matches: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        path,
        sep=separator,
        chunksize=chunk_size,
        usecols=lambda column: column in PLOT_COLUMNS,
        dtype=text_dtypes,
    ):
        sample_mask = chunk["sample_id"].eq(str(sample_id)).fillna(False)
        if not sample_mask.any():
            continue

        sample_found = True
        selected = chunk.loc[sample_mask]

        if prediction_dataset is not None:
            dataset_mask = (
                selected["prediction_dataset"]
                .eq(str(prediction_dataset))
                .fillna(False)
            )
            selected = selected.loc[dataset_mask]

        if selected.empty:
            continue
        prediction_found = True

        if requested_slots is not None:
            if "slot" in selected.columns:
                selected = selected.loc[
                    selected["slot"].astype("string").isin(requested_slots)
                ]
            elif "0" not in requested_slots:
                continue

        if not selected.empty:
            matches.append(selected.copy())

    if not sample_found:
        raise ValueError(f"No rows were found for sample_id={sample_id!r}.")
    if not prediction_found:
        raise ValueError(
            f"No rows match sample_id={sample_id!r} and "
            f"prediction_dataset={prediction_dataset!r}."
        )
    if not matches:
        raise ValueError(f"No rows match requested slots: {sorted(requested_slots or [])}")

    return pd.concat(matches, ignore_index=True, copy=False)


def numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce")


def binary_column(values: pd.Series) -> pd.Series:
    """Normalize booleans and common CSV string representations to 0/1."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(float)

    text = values.astype("string").str.strip().str.lower()
    normalized = text.replace(
        {"true": "1", "false": "0", "yes": "1", "no": "0"}
    )
    return pd.to_numeric(normalized, errors="coerce").astype(float)


def ensure_prediction_columns(
    frame: pd.DataFrame, activity_threshold: float
) -> pd.DataFrame:
    """Derive missing angle/activity columns from prediction ACCDOA vectors."""
    result = frame.copy()
    has_vectors = all(column in result.columns for column in PREDICTION_VECTOR_COLUMNS)

    if has_vectors:
        x, y, z = (
            numeric_column(result, column).to_numpy(dtype=float)
            for column in PREDICTION_VECTOR_COLUMNS
        )
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        if "prediction_activity" not in result.columns:
            result["prediction_activity"] = magnitude
        if "prediction_az_deg" not in result.columns:
            result["prediction_az_deg"] = np.degrees(np.arctan2(y, x))
        if "prediction_el_deg" not in result.columns:
            result["prediction_el_deg"] = np.degrees(
                np.arctan2(z, np.hypot(x, y))
            )

    if "prediction_activity" not in result.columns:
        if "prediction_active" not in result.columns:
            raise ValueError(
                "Dataset needs prediction_activity, prediction_active, or all "
                "three prediction_accdoa_x/y/z columns."
            )
        result["prediction_activity"] = binary_column(result["prediction_active"])

    if "prediction_active" not in result.columns:
        result["prediction_active"] = (
            numeric_column(result, "prediction_activity") >= activity_threshold
        ).astype(int)

    missing_angles = [
        column for column in PREDICTION_ANGLE_COLUMNS if column not in result.columns
    ]
    if missing_angles:
        raise ValueError(
            f"Missing prediction angle columns: {', '.join(missing_angles)}; "
            "alternatively provide prediction_accdoa_x/y/z."
        )

    return result


def resolve_time_seconds(frame: pd.DataFrame) -> pd.Series:
    """Use absolute frame time, adding segment offsets only if frames reset."""
    if "frame_start_s" in frame.columns:
        frame_time = numeric_column(frame, "frame_start_s")
    elif "frame_start_sample" in frame.columns:
        if "sample_rate" not in frame.columns:
            raise ValueError(
                "frame_start_sample requires a sample_rate column when "
                "frame_start_s is unavailable."
            )
        frame_time = numeric_column(frame, "frame_start_sample") / numeric_column(
            frame, "sample_rate"
        )
    elif "frame_idx" in frame.columns:
        frame_time = numeric_column(frame, "frame_idx")
        print("Warning: frame_start_s is unavailable; using frame_idx as time.")
    else:
        raise ValueError("Dataset needs frame_start_s or frame_idx for the time axis.")

    if "segment_idx" not in frame.columns or "segment_start_s" not in frame.columns:
        return frame_time

    segment_starts = numeric_column(frame, "segment_start_s")
    segment_data = pd.DataFrame(
        {
            "segment": frame["segment_idx"].astype("string"),
            "frame_time": frame_time,
            "segment_start": segment_starts,
        }
    )
    summaries = segment_data.groupby("segment", dropna=False).agg(
        first_frame=("frame_time", "min"),
        segment_start=("segment_start", "min"),
    )
    later_segments = summaries[summaries["segment_start"].abs() > 1e-9]

    if not later_segments.empty:
        relative_error = later_segments["first_frame"].abs().median()
        absolute_error = (
            later_segments["first_frame"] - later_segments["segment_start"]
        ).abs().median()
        if relative_error < absolute_error:
            return frame_time + segment_starts.fillna(0.0)

    return frame_time


def slot_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def without_azimuth_wraparound(
    values: np.ndarray, *, limit_degrees: float = 180.0
) -> np.ndarray:
    """Break lines across the -180/180-degree seam instead of drawing spikes."""
    result = values.astype(float, copy=True)
    jumps = np.flatnonzero(np.abs(np.diff(result)) > limit_degrees)
    result[jumps + 1] = np.nan
    return result


def match_prediction_slots(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Assign active predictions to active GT tracks independently per frame."""
    matched = frame.copy()
    matched["matched_gt_slot"] = pd.Series(pd.NA, index=matched.index, dtype="string")
    matched["matched_angular_distance_deg"] = np.nan

    ground_truth_active = binary_column(matched["active"]).ge(0.5).fillna(False)
    prediction_active = (
        binary_column(matched["prediction_active"]).ge(0.5).fillna(False)
    )
    truth_azimuth = np.radians(numeric_column(matched, "az_deg"))
    truth_elevation = np.radians(numeric_column(matched, "el_deg"))
    prediction_azimuth = np.radians(
        numeric_column(matched, "prediction_az_deg")
    )
    prediction_elevation = np.radians(
        numeric_column(matched, "prediction_el_deg")
    )

    valid_truth = (
        ground_truth_active
        & np.isfinite(truth_azimuth)
        & np.isfinite(truth_elevation)
    )
    valid_prediction = (
        prediction_active
        & np.isfinite(prediction_azimuth)
        & np.isfinite(prediction_elevation)
    )

    for _, frame_rows in matched.groupby(
        ["_matching_segment", "time_s"], sort=False, dropna=False
    ):
        truth_indices = frame_rows.index[valid_truth.loc[frame_rows.index]]
        prediction_indices = frame_rows.index[
            valid_prediction.loc[frame_rows.index]
        ]
        if not len(truth_indices) or not len(prediction_indices):
            continue

        truth_az = truth_azimuth.loc[truth_indices].to_numpy(dtype=float)
        truth_el = truth_elevation.loc[truth_indices].to_numpy(dtype=float)
        prediction_az = prediction_azimuth.loc[prediction_indices].to_numpy(
            dtype=float
        )
        prediction_el = prediction_elevation.loc[prediction_indices].to_numpy(
            dtype=float
        )

        cosine_distance = (
            np.sin(truth_el[:, None]) * np.sin(prediction_el[None, :])
            + np.cos(truth_el[:, None])
            * np.cos(prediction_el[None, :])
            * np.cos(truth_az[:, None] - prediction_az[None, :])
        )
        angular_distance = np.degrees(
            np.arccos(np.clip(cosine_distance, -1.0, 1.0))
        )
        truth_positions, prediction_positions = linear_sum_assignment(
            angular_distance
        )

        assigned_truth_indices = truth_indices.take(truth_positions)
        assigned_prediction_indices = prediction_indices.take(prediction_positions)
        matched.loc[assigned_prediction_indices, "matched_gt_slot"] = matched.loc[
            assigned_truth_indices, "slot"
        ].to_numpy()
        matched.loc[
            assigned_prediction_indices, "matched_angular_distance_deg"
        ] = angular_distance[truth_positions, prediction_positions]

    return matched


def prepare_sample(
    dataset_path: Path,
    sample_id: str,
    prediction_dataset: str | None = PREDICTION_DATASET,
    slots: Sequence[str] | None = SLOTS,
    activity_threshold: float = ACTIVITY_THRESHOLD,
    chunk_size: int = CSV_CHUNK_SIZE,
) -> tuple[pd.DataFrame, list[str], str | None]:
    """Load, filter, and match the tracks for one dataset column."""
    dataset_path = Path(dataset_path)
    if dataset_path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        frame = load_selected_tabular_dataset(
            dataset_path,
            sample_id,
            prediction_dataset,
            # A selected GT slot may correspond to a different predicted slot.
            slots=None,
            chunk_size=chunk_size,
        )
    else:
        frame = load_dataset(dataset_path)
    required = {"sample_id", *GROUND_TRUTH_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    frame = frame[frame["sample_id"].astype("string") == str(sample_id)].copy()
    if frame.empty:
        raise ValueError(f"No rows were found for sample_id={sample_id!r}.")

    if prediction_dataset is not None:
        if "prediction_dataset" not in frame.columns:
            raise ValueError(
                "--prediction-dataset was specified, but the dataset has no "
                "prediction_dataset column. Set PREDICTION_DATASET = None or "
                "pass --all-prediction-datasets."
            )
        frame = frame[
            frame["prediction_dataset"].astype("string") == str(prediction_dataset)
        ].copy()
        if frame.empty:
            raise ValueError(
                f"No rows match sample_id={sample_id!r} and "
                f"prediction_dataset={prediction_dataset!r}."
            )
    elif "prediction_dataset" in frame.columns:
        names = sorted(frame["prediction_dataset"].dropna().astype(str).unique())
        if len(names) > 1:
            raise ValueError(
                "Multiple prediction datasets match this sample; select one "
                f"with --prediction-dataset: {', '.join(names)}"
            )
        if names:
            prediction_dataset = names[0]

    if "slot" not in frame.columns:
        frame["slot"] = "0"
    frame["slot"] = frame["slot"].astype("string")

    frame = ensure_prediction_columns(frame, activity_threshold)
    frame["time_s"] = resolve_time_seconds(frame)
    frame = frame.dropna(subset=["time_s"]).sort_values(
        ["time_s", "slot", *( ["segment_idx"] if "segment_idx" in frame.columns else [] )],
        kind="stable",
    )
    if frame.empty:
        raise ValueError("The selected sample has no valid frame timestamps.")

    if "segment_idx" in frame.columns:
        frame["_matching_segment"] = (
            frame["segment_idx"].astype("string").fillna("<missing>")
        )
    else:
        frame["_matching_segment"] = "all"

    frame = match_prediction_slots(frame)

    selected_slots = sorted(frame["slot"].dropna().unique(), key=slot_sort_key)
    if slots is not None:
        requested_slots = {str(slot) for slot in slots}
        selected_slots = [slot for slot in selected_slots if slot in requested_slots]
        if not selected_slots:
            raise ValueError(f"No rows match requested slots: {sorted(requested_slots)}")

    matching_lines = []
    matched_rows = frame[
        frame["matched_gt_slot"].notna()
        & frame["matched_gt_slot"].isin(selected_slots)
    ]
    for segment, segment_rows in matched_rows.groupby(
        "_matching_segment", sort=False, dropna=False
    ):
        summaries = segment_rows.groupby(
            ["slot", "matched_gt_slot"], sort=False, dropna=False
        )["matched_angular_distance_deg"].agg(["size", "mean"])
        pairs = ", ".join(
            f"prediction {prediction_slot} -> GT {truth_slot} "
            f"({int(summary['size'])} frames, LE {summary['mean']:.1f} deg)"
            for (prediction_slot, truth_slot), summary in summaries.iterrows()
        )
        matching_lines.append(f"segment {segment}: {pairs}")
    for summary in matching_lines[:20]:
        print(f"[{dataset_path.name}] Track matching {summary}")
    if len(matching_lines) > 20:
        print(
            f"[{dataset_path.name}] Track matching: "
            f"{len(matching_lines) - 20} more segments omitted."
        )

    return frame, selected_slots, prediction_dataset


def resolve_slot_colors(
    slots: Sequence[str],
    configured_colors: Mapping[str, str] | None = None,
    *,
    palette_offset: int = 0,
) -> dict[str, tuple[float, float, float, float]]:
    """Apply configured slot colors and use paired tab20 colors as fallback."""
    colors = plt.get_cmap("tab20")
    configured = {
        str(slot): color for slot, color in (configured_colors or {}).items()
    }
    resolved: dict[str, tuple[float, float, float, float]] = {}

    for index, slot in enumerate(slots):
        slot = str(slot)
        color = configured.get(slot, colors((2 * index + palette_offset) % colors.N))
        try:
            resolved[slot] = to_rgba(color)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid color for slot {slot!r}: {color!r}.") from error

    return resolved


def draw_sample_column(
    axes: Sequence[plt.Axes],
    frame: pd.DataFrame,
    selected_slots: Sequence[str],
    ground_truth_slot_colors: dict[str, tuple[float, float, float, float]],
    prediction_slot_colors: dict[str, tuple[float, float, float, float]],
    title: str,
    time_limits: tuple[float, float],
    azimuth_limits: tuple[float, float],
    elevation_limits: tuple[float, float],
    *,
    first_column: bool,
) -> None:
    """Draw one dataset on an existing azimuth/elevation subplot column."""
    azimuth_axis, elevation_axis = axes

    for slot in selected_slots:
        ground_truth_color = ground_truth_slot_colors[str(slot)]
        prediction_color = prediction_slot_colors[str(slot)]
        for segment, segment_frame in frame.groupby(
            "_matching_segment", sort=False, dropna=False
        ):
            truth_source = segment_frame[segment_frame["slot"] == slot].sort_values(
                "time_s", kind="stable"
            )
            truth_time = truth_source["time_s"].to_numpy(dtype=float)
            truth_active = binary_column(truth_source["active"]).to_numpy(dtype=float)

            for axis, truth_column, prediction_column, wrap in (
                (azimuth_axis, "az_deg", "prediction_az_deg", True),
                (elevation_axis, "el_deg", "prediction_el_deg", False),
            ):
                truth = numeric_column(truth_source, truth_column).to_numpy(dtype=float)
                truth[truth_active < 0.5] = np.nan
                if wrap:
                    truth = without_azimuth_wraparound(truth)

                for prediction_slot, prediction_source in segment_frame.groupby(
                    "slot", sort=False, dropna=False
                ):
                    prediction_source = prediction_source.sort_values(
                        "time_s", kind="stable"
                    )
                    prediction_time = prediction_source["time_s"].to_numpy(dtype=float)
                    matched_to_truth = (
                        prediction_source["matched_gt_slot"]
                        .eq(slot)
                        .fillna(False)
                        .to_numpy(dtype=bool)
                    )
                    if not matched_to_truth.any():
                        continue
                    prediction = numeric_column(
                        prediction_source, prediction_column
                    ).to_numpy(dtype=float)
                    prediction[~matched_to_truth] = np.nan
                    if wrap:
                        prediction = without_azimuth_wraparound(prediction)

                    axis.plot(
                        prediction_time,
                        prediction,
                        color=prediction_color,
                        linestyle="-",
                        linewidth=PREDICTION_LINE_WIDTH,
                        alpha=PREDICTION_ALPHA,
                        zorder=4,
                    )

                axis.plot(
                    truth_time,
                    truth,
                    linestyle="none",
                    marker=GROUND_TRUTH_MARKER,
                    markersize=GROUND_TRUTH_MARKER_SIZE,
                    markevery=GROUND_TRUTH_MARKER_EVERY,
                    markerfacecolor="none",
                    markeredgecolor=ground_truth_color,
                    markeredgewidth=GROUND_TRUTH_MARKER_EDGE_WIDTH,
                    alpha=GROUND_TRUTH_ALPHA,
                    zorder=6,
                )

    # Draw active predictions that the per-frame assignment could not pair
    # with any active ground-truth source. This is done outside the GT-slot
    # loop so each unmatched prediction is plotted exactly once.
    for _, segment_frame in frame.groupby(
        "_matching_segment", sort=False, dropna=False
    ):
        for _, prediction_source in segment_frame.groupby(
            "slot", sort=False, dropna=False
        ):
            prediction_source = prediction_source.sort_values("time_s", kind="stable")
            prediction_time = prediction_source["time_s"].to_numpy(dtype=float)
            unmatched_prediction = (
                binary_column(prediction_source["prediction_active"])
                .ge(0.5)
                .fillna(False)
                & prediction_source["matched_gt_slot"].isna()
            ).to_numpy(dtype=bool)
            if not unmatched_prediction.any():
                continue

            for axis, prediction_column, wrap in (
                (azimuth_axis, "prediction_az_deg", True),
                (elevation_axis, "prediction_el_deg", False),
            ):
                prediction = numeric_column(
                    prediction_source, prediction_column
                ).to_numpy(dtype=float)
                prediction[~unmatched_prediction] = np.nan
                if wrap:
                    prediction = without_azimuth_wraparound(prediction)

                axis.plot(
                    prediction_time,
                    prediction,
                    color=UNMATCHED_PREDICTION_COLOR,
                    linestyle="-",
                    linewidth=PREDICTION_LINE_WIDTH,
                    alpha=UNMATCHED_PREDICTION_ALPHA,
                    zorder=3,
                )

    if "segment_idx" in frame.columns:
        starts = frame.groupby("segment_idx", dropna=False)["time_s"].min().sort_values()
        # Avoid filling small figures with boundaries for long recordings.
        if len(starts) <= 40:
            for boundary in starts.iloc[1:]:
                for axis in axes:
                    axis.axvline(
                        boundary,
                        color="0.72",
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.7,
                        zorder=0,
                    )

    azimuth_axis.set(
        ylabel="Azimuth (°)" if first_column else "",
        xlim=time_limits,
        ylim=azimuth_limits,
    )
    elevation_axis.set(
        xlabel="Time (s)",
        ylabel="Elevation (°)" if first_column else "",
        xlim=time_limits,
        ylim=elevation_limits,
    )
    azimuth_axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    elevation_axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    azimuth_axis.set_title(
        title, fontsize=SUBPLOT_TITLE_FONT_SIZE, fontweight="bold", pad=10
    )
    for axis in axes:
        axis.set_facecolor("none")
        axis.grid(True, color="black", linewidth=1.05, alpha=0.1)
        axis.tick_params(labelsize=TICK_LABEL_FONT_SIZE, width=1.2)
        axis.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
        axis.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["bottom"].set_linewidth(1.2)
        axis.spines["left"].set_linewidth(1.2)


def add_column_background(
    figure: plt.Figure,
    axes_in_column: Sequence[plt.Axes],
    color: str,
    alpha: float = COLUMN_BACKGROUND_ALPHA,
    top_extend: float = COLUMN_BACKGROUND_EXTEND_TOP,
    bottom_extend: float = COLUMN_BACKGROUND_EXTEND_BOTTOM,
    side_extend: float = COLUMN_BACKGROUND_EXTEND_SIDES,
) -> None:
    """Shade an entire column and extend beyond its top, bottom, and sides."""
    positions = [axis.get_position() for axis in axes_in_column]
    left = max(0.0, min(position.x0 for position in positions) - side_extend)
    right = min(1.0, max(position.x1 for position in positions) + side_extend)
    bottom = max(0.0, min(position.y0 for position in positions) - bottom_extend)
    top = min(1.0, max(position.y1 for position in positions) + top_extend)

    figure.add_artist(
        Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            transform=figure.transFigure,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=-10,
            clip_on=False,
        )
    )


def calculate_shared_time_limits(
    prepared: Sequence[tuple[pd.DataFrame, Sequence[str], str | None]],
) -> tuple[float, float]:
    """Include every frame timestamp from every model, even inactive frames."""
    timestamps = [
        numeric_column(frame, "time_s").to_numpy(dtype=float)
        for frame, _, _ in prepared
    ]
    finite_timestamps = [values[np.isfinite(values)] for values in timestamps]
    finite_timestamps = [values for values in finite_timestamps if values.size]
    if not finite_timestamps:
        raise ValueError("The selected models have no valid frame timestamps.")

    values = np.concatenate(finite_timestamps)
    start = float(values.min())
    end = float(values.max())
    if start == end:
        return start - 0.5, end + 0.5
    return start, end


def calculate_shared_angle_limits(
    prepared: Sequence[tuple[pd.DataFrame, Sequence[str], str | None]],
    truth_column: str,
    prediction_column: str,
    hard_limits: tuple[float, float],
    minimum_span: float,
) -> tuple[float, float]:
    """Return padded limits from all visible traces in every model column."""
    visible_values: list[np.ndarray] = []

    for frame, selected_slots, _ in prepared:
        selected_slot_strings = {str(slot) for slot in selected_slots}
        truth_mask = (
            binary_column(frame["active"]).ge(0.5).fillna(False)
            & frame["slot"].astype("string").isin(selected_slot_strings)
        )
        prediction_mask = (
            binary_column(frame["prediction_active"]).ge(0.5).fillna(False)
            & (
                frame["matched_gt_slot"].astype("string").isin(selected_slot_strings)
                | frame["matched_gt_slot"].isna()
            )
        )
        visible_values.extend(
            [
                numeric_column(frame.loc[truth_mask], truth_column).to_numpy(dtype=float),
                numeric_column(
                    frame.loc[prediction_mask], prediction_column
                ).to_numpy(dtype=float),
            ]
        )

    finite_arrays = [values[np.isfinite(values)] for values in visible_values]
    finite_arrays = [values for values in finite_arrays if values.size]
    if not finite_arrays:
        return hard_limits

    values = np.concatenate(finite_arrays)
    lower = float(values.min())
    upper = float(values.max())
    span = max(upper - lower, minimum_span)
    center = (lower + upper) / 2.0
    lower = min(lower, center - minimum_span / 2.0)
    upper = max(upper, center + minimum_span / 2.0)
    padding = max(2.0, span * 0.05)

    # Round outward to clean five-degree boundaries, then respect the physical
    # plotting bounds (with five extra degrees so edge traces remain visible).
    lower = np.floor((lower - padding) / 5.0) * 5.0
    upper = np.ceil((upper + padding) / 5.0) * 5.0
    lower = max(hard_limits[0], lower)
    upper = min(hard_limits[1], upper)
    return float(lower), float(upper)


def plot_samples(
    dataset_paths: Sequence[Path],
    sample_id: str,
    prediction_datasets: str | Sequence[str | None] | None = PREDICTION_DATASET,
    output_pdf: Path | None = OUTPUT_PDF,
    labels: Sequence[str] | None = DATASET_LABELS,
    slots: Sequence[str] | None = SLOTS,
    activity_threshold: float = ACTIVITY_THRESHOLD,
    active_only: bool = False,
    show_binary_prediction: bool = False,
    chunk_size: int = CSV_CHUNK_SIZE,
    ground_truth_slot_colors: Mapping[str, str] | None = None,
    prediction_slot_colors: Mapping[str, str] | None = None,
) -> Path:
    """Save a 2-by-N comparison: one azimuth/elevation column per dataset."""
    paths = [Path(path) for path in dataset_paths]
    if not paths:
        raise ValueError("At least one input dataset is required.")
    if prediction_datasets is None or isinstance(prediction_datasets, str):
        selected_prediction_datasets = [prediction_datasets] * len(paths)
    else:
        selected_prediction_datasets = list(prediction_datasets)
        if len(selected_prediction_datasets) == 1:
            selected_prediction_datasets *= len(paths)
        elif len(selected_prediction_datasets) != len(paths):
            raise ValueError(
                "Provide one --prediction-dataset value for all files or "
                f"exactly one per file; got {len(selected_prediction_datasets)} "
                f"values for {len(paths)} files."
            )

    if labels is not None and len(labels) != len(paths):
        raise ValueError(
            f"Provide exactly one label per dataset; got {len(labels)} labels "
            f"for {len(paths)} files."
        )

    prepared = [
        prepare_sample(
            dataset_path=path,
            sample_id=sample_id,
            prediction_dataset=prediction_dataset,
            slots=slots,
            activity_threshold=activity_threshold,
            chunk_size=chunk_size,
        )
        for path, prediction_dataset in zip(paths, selected_prediction_datasets)
    ]

    time_limits = calculate_shared_time_limits(prepared)
    azimuth_limits = calculate_shared_angle_limits(
        prepared,
        truth_column="az_deg",
        prediction_column="prediction_az_deg",
        hard_limits=(-185.0, 185.0),
        minimum_span=60.0,
    )
    elevation_limits = calculate_shared_angle_limits(
        prepared,
        truth_column="el_deg",
        prediction_column="prediction_el_deg",
        hard_limits=(-95.0, 95.0),
        minimum_span=30.0,
    )

    all_slots = sorted(
        {str(slot) for _, selected_slots, _ in prepared for slot in selected_slots},
        key=slot_sort_key,
    )
    resolved_ground_truth_colors = resolve_slot_colors(
        all_slots,
        GROUND_TRUTH_SLOT_COLORS
        if ground_truth_slot_colors is None
        else ground_truth_slot_colors,
        palette_offset=1,
    )
    resolved_prediction_colors = resolve_slot_colors(
        all_slots,
        PREDICTION_SLOT_COLORS
        if prediction_slot_colors is None
        else prediction_slot_colors,
        palette_offset=0,
    )

    column_width = 10.5 if len(paths) == 1 else 6.8
    figure = plt.figure(
        figsize=(column_width * len(paths), 7.2),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        nrows=3,
        ncols=len(paths),
        height_ratios=(0.35, 1.0, 1.0),
    )
    legend_axis = figure.add_subplot(grid[0, :])
    legend_axis.set_axis_off()

    axes = np.empty((2, len(paths)), dtype=object)
    for index in range(len(paths)):
        axes[0, index] = figure.add_subplot(
            grid[1, index],
            sharey=axes[0, 0] if index else None,
        )
        axes[1, index] = figure.add_subplot(
            grid[2, index],
            sharex=axes[0, index],
            sharey=axes[1, 0] if index else None,
        )
        axes[0, index].tick_params(labelbottom=False)
        if index:
            for axis in axes[:, index]:
                axis.tick_params(labelleft=False)

    for index, (path, (frame, selected_slots, resolved_dataset)) in enumerate(
        zip(paths, prepared)
    ):
        if labels is not None:
            title = labels[index]
        elif len(paths) == 1 and resolved_dataset is not None:
            title = resolved_dataset
        else:
            title = path.stem

        draw_sample_column(
            axes[:, index],
            frame,
            selected_slots,
            resolved_ground_truth_colors,
            resolved_prediction_colors,
            title,
            time_limits,
            azimuth_limits,
            elevation_limits,
            first_column=index == 0,
        )

    # Align both y-axis labels to whichever row needs the leftmost position.
    figure.align_ylabels(axes[:, 0])

    figure_title_separator = "\n" if len(paths) == 1 else " "
    figure.suptitle(
        "Predictions vs. Ground Truth"
        f"{figure_title_separator}"
        "(Arni: Reverberant Condition, Receiver Position 65)",
        fontsize=FIGURE_TITLE_FONT_SIZE,
        fontweight="bold",
        y=FIGURE_TITLE_Y,
    )

    legend_handles = [
        Line2D(
            [0], [0],
            color="0.5",
            linestyle="none",
            marker=GROUND_TRUTH_MARKER,
            markersize=GROUND_TRUTH_MARKER_SIZE*1.5,
            markerfacecolor="none",
            markeredgecolor="0.2",
            markeredgewidth=LEGEND_GROUND_TRUTH_MARKER_EDGE_WIDTH,
            alpha=GROUND_TRUTH_ALPHA,
            label=LEGEND_TEXT["ground_truth"],
        ),
        Line2D(
            [0], [0],
            color="0.25",
            linestyle="-",
            linewidth=LEGEND_LINE_WIDTH,
            alpha=PREDICTION_ALPHA,
            label=LEGEND_TEXT["prediction"],
        ),

        *[
            Line2D(
                [0], [0],
                color=resolved_prediction_colors[slot],
                linewidth=LEGEND_LINE_WIDTH,
                label=LEGEND_TEXT["slot"].format(slot=int(slot) + 1),
            )
            for slot in all_slots
        ],
    ]
    legend_axis.legend(
        handles=legend_handles,
        loc="center",
        ncol=min(6, len(legend_handles)),
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        handlelength=2.2,
        numpoints=5,
        columnspacing=1.2,
    )

    # Constrained layout finalizes axes positions only when the canvas is drawn.
    # Draw first so each background follows both rows and the gap between them.
    figure.canvas.draw()
    for index in range(len(paths)):
        add_column_background(
            figure,
            axes[:, index],
            COLUMN_BACKGROUND_COLORS[index % len(COLUMN_BACKGROUND_COLORS)],
        )

    if output_pdf is None:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))
        if len(paths) == 1:
            dataset_label = prepared[0][2] or paths[0].stem
            safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset_label))
            output_pdf = Path(
                f"{safe_id}_{safe_dataset}_ground_truth_vs_prediction.pdf"
            )
        else:
            output_pdf = Path(f"{safe_id}_dataset_comparison.pdf")
    output_pdf = Path(output_pdf)
    if output_pdf.suffix.lower() != ".pdf":
        output_pdf = output_pdf.with_suffix(".pdf")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(figure)

    print(f"Saved {output_pdf.resolve()} ({len(paths)} dataset columns).")
    for path, (frame, selected_slots, _) in zip(paths, prepared):
        segment_count = (
            frame["segment_idx"].nunique() if "segment_idx" in frame.columns else 1
        )
        print(
            f"  {path.name}: {len(frame)} rows, "
            f"{segment_count} segments, {len(selected_slots)} slots."
        )
    return output_pdf


def plot_sample(
    dataset_path: Path,
    sample_id: str,
    prediction_dataset: str | None = PREDICTION_DATASET,
    output_pdf: Path | None = OUTPUT_PDF,
    slots: Sequence[str] | None = SLOTS,
    activity_threshold: float = ACTIVITY_THRESHOLD,
    active_only: bool = False,
    show_binary_prediction: bool = False,
    chunk_size: int = CSV_CHUNK_SIZE,
    ground_truth_slot_colors: Mapping[str, str] | None = None,
    prediction_slot_colors: Mapping[str, str] | None = None,
) -> Path:
    """Keep the original single-dataset Python interface available."""
    return plot_samples(
        dataset_paths=[dataset_path],
        sample_id=sample_id,
        prediction_datasets=prediction_dataset,
        output_pdf=output_pdf,
        labels=None,
        slots=slots,
        activity_threshold=activity_threshold,
        active_only=active_only,
        show_binary_prediction=show_binary_prediction,
        chunk_size=chunk_size,
        ground_truth_slot_colors=ground_truth_slot_colors,
        prediction_slot_colors=prediction_slot_colors,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        type=Path,
        nargs="+",
        action="append",
        default=None,
        metavar="FILE",
        help=(
            "One or more input dataset files. Each file is shown in its own "
            "subplot column; --dataset may also be repeated."
        ),
    )
    parser.add_argument("--sample-id", default=SAMPLE_ID, help="Room-position sample ID.")
    parser.add_argument(
        "--prediction-dataset",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Value to select from the prediction_dataset column; provide one "
            "shared value or one value per input file."
        ),
    )
    parser.add_argument(
        "--all-prediction-datasets",
        action="store_true",
        help="Do not require a prediction_dataset column; one unique dataset is still required.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DATASET_LABELS,
        metavar="LABEL",
        help="Optional subplot titles, one per input dataset.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PDF, help="Output PDF path.")
    parser.add_argument("--slots", nargs="+", default=SLOTS, help="Optional source slots to plot.")
    parser.add_argument(
        "--ground-truth-slot-colors",
        nargs="+",
        default=None,
        metavar="SLOT=COLOR",
        help=(
            "Set ground-truth colors by slot, for example "
            "--ground-truth-slot-colors '0=#f4a6a6' '1=lightblue'. "
            "Unspecified slots use tab20's light colors."
        ),
    )
    parser.add_argument(
        "--prediction-slot-colors",
        nargs="+",
        default=None,
        metavar="SLOT=COLOR",
        help=(
            "Set prediction colors by slot, for example "
            "--prediction-slot-colors '0=#e41a1c' '1=royalblue'. "
            "Unspecified slots use tab20's dark colors."
        ),
    )
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=ACTIVITY_THRESHOLD,
        help="Threshold for deriving predicted binary activity.",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Accepted for compatibility; inactive traces are always hidden.",
    )
    parser.add_argument(
        "--show-binary-prediction",
        action="store_true",
        help="Accepted for compatibility; the compact plot has no activity panel.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CSV_CHUNK_SIZE,
        help="Number of CSV/TSV rows to scan per chunk (default: %(default)s).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    def parse_color_assignments(
        option: str,
        defaults: Mapping[str, str],
        assignments: Sequence[str] | None,
    ) -> dict[str, str]:
        selected_colors = dict(defaults)
        for assignment in assignments or ():
            slot, separator, color = assignment.partition("=")
            if not separator or not slot or not color:
                parser.error(
                    f"{option} entries must use SLOT=COLOR, "
                    "for example '0=#e41a1c'."
                )
            try:
                to_rgba(color)
            except (TypeError, ValueError):
                parser.error(f"Invalid color for slot {slot!r}: {color!r}.")
            selected_colors[slot] = color
        return selected_colors

    selected_ground_truth_colors = parse_color_assignments(
        "--ground-truth-slot-colors",
        GROUND_TRUTH_SLOT_COLORS,
        arguments.ground_truth_slot_colors,
    )
    selected_prediction_colors = parse_color_assignments(
        "--prediction-slot-colors",
        PREDICTION_SLOT_COLORS,
        arguments.prediction_slot_colors,
    )

    dataset_paths = (
        [path for group in arguments.datasets for path in group]
        if arguments.datasets is not None
        else list(DATASET_PATHS)
    )

    if arguments.labels is not None and len(arguments.labels) != len(dataset_paths):
        parser.error("--labels requires exactly one label per input dataset.")
    if (
        not arguments.all_prediction_datasets
        and arguments.prediction_dataset is not None
        and len(arguments.prediction_dataset) not in {1, len(dataset_paths)}
    ):
        parser.error(
            "--prediction-dataset requires one shared value or exactly one "
            "value per input dataset."
        )

    if arguments.all_prediction_datasets:
        prediction_datasets = None
    elif arguments.prediction_dataset is not None:
        prediction_datasets = arguments.prediction_dataset
    else:
        prediction_datasets = PREDICTION_DATASET

    plot_samples(
        dataset_paths=dataset_paths,
        sample_id=arguments.sample_id,
        prediction_datasets=prediction_datasets,
        output_pdf=arguments.output,
        labels=arguments.labels,
        slots=arguments.slots,
        activity_threshold=arguments.activity_threshold,
        active_only=arguments.active_only,
        show_binary_prediction=arguments.show_binary_prediction,
        chunk_size=arguments.chunk_size,
        ground_truth_slot_colors=selected_ground_truth_colors,
        prediction_slot_colors=selected_prediction_colors,
    )


if __name__ == "__main__":
    main()