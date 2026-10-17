#!/usr/bin/env python3
"""Plot model metrics against RT60 in one publication-ready PDF figure.

Examples
--------
python rt60_metrics_with_confidence_intervals.py
python rt60_metrics_with_confidence_intervals.py --metrics LE LR F15
python rt60_metrics_with_confidence_intervals.py --metrics LE,ER15,ESDL --output selected.pdf
python rt60_metrics_with_confidence_intervals.py --all-metrics --music-metrics all
python rt60_metrics_with_confidence_intervals.py --all-models --no-ci

Put ``model_metrics_by_rt60.csv`` next to this script, or supply ``--csv``.
Requires matplotlib, numpy, and pandas.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator


# ============================================================================
# CONFIGURATION: edit these lists, or override them with command-line options.
# ============================================================================

CSV_PATH = Path("model_metrics_by_rt60.csv")
OUTPUT_PATH = Path("plot_metrics_vs_rt60.pdf")

# The default models and their order exactly match the supplied plotting script.
# Set this to None to include all models present in the CSV.
PLOTTED_MODELS = [
    "MUSIC",
    #"shoeBox_SH_Direct",
    "shoeBox_em32_Direct",
    #"shoeBox_SH_Direct-Late",
    "shoeBox_em32_Direct-Late",
    #"shoeBox_SH_ISMOrder2",
    "shoeBox_em32_ISMOrder2",
    #"shoeBox_SH_ISMOrder2-Late",
    "shoeBox_em32_ISMOrder2-Late",
    #"shoeBox_SH_ISMOrder12",
    "shoeBox_em32_ISMOrder12",
    #"shoeBox_SH_ISMOrder12-Late",
    "shoeBox_em32_ISMOrder12-Late",
]

# MUSIC is shown only for these metrics. Set to None to show it on every row.
MUSIC_PLOTTED_METRICS = ["LE"]
PLOTTED_METRICS = ["LE", "ER15", "ESDL"]

# ``full_sample`` matches the source plotting code. ``estimate`` is the
# jackknife estimate around which the supplied confidence intervals are built.
VALUE_COLUMN = "full_sample"
SHOW_CONFIDENCE_INTERVALS = True
CONFIDENCE_ALPHA = 0.16
MODEL_BACKGROUND_ALPHA = 0.15

# Font sizes (points). Change these values to scale individual figure elements.
FONT_SIZE_DEFAULT = 16
FONT_SIZE_FIGURE_TITLE = 28
FONT_SIZE_LEGEND = 22
FONT_SIZE_EXPLANATION = 20
FONT_SIZE_MODEL_PARAMETERS = 22
FONT_SIZE_TICK_LABELS = 16
FONT_SIZE_Y_AXIS_LABEL = 18
FONT_SIZE_NOT_PLOTTED = 15
FONT_SIZE_METRIC_TITLE = 22
FONT_SIZE_X_AXIS_LABEL = 20

# Figure dimensions. Figure sizes are inches; margins and vertical offsets use
# normalized figure coordinates.
FIGURE_MIN_WIDTH = 0
FIGURE_WIDTH_PER_MODEL = 3.4
FIGURE_WIDTH_PADDING = 1
FIGURE_MIN_HEIGHT = 0
FIGURE_HEIGHT_PER_METRIC = 3.15
FIGURE_HEIGHT_PADDING = 5.3

# Vertical layout: these are the main spacing controls, in top-to-bottom order.
# The first three use normalized figure coordinates. The last two are relative
# grid-row sizes (increase a value to add more vertical space).
SPACING_AFTER_TITLE = 0.03
SPACING_AFTER_LEGEND = 0.065
SPACING_AFTER_DESCRIPTION = 0.025
SPACING_AFTER_MIC_ORDER = 0.6
SPACING_BETWEEN_METRICS = 0.6

FIGURE_TITLE_Y = 0.993
LEGEND_CENTER_X = 0.53
DESCRIPTION_CENTER_X = 0.535

GRID_HEADER_HEIGHT_RATIO = 1.6
GRID_METRIC_HEIGHT_RATIO = 2.35
GRID_LEFT_MARGIN_MIN = 0.070
GRID_LEFT_PADDING_INCHES = 2.0
GRID_RIGHT_MARGIN = 0.989
GRID_BOTTOM_MARGIN = 0.085
GRID_HORIZONTAL_SPACING = 0.075
LEGEND_HANDLE_LENGTH = 2.4
LEGEND_COLUMN_SPACING = 2.4
EXPLANATION_LINE_SPACING = 1.6
MODEL_PARAMETER_LABEL_X = -0.13
TICK_LENGTH = 4
Y_AXIS_LABEL_PADDING = 12
METRIC_TITLE_Y_PADDING = 0.004
X_AXIS_LABEL_Y = 0.027
X_AXIS_DATA_PADDING = 0.1
Y_AXIS_DATA_PADDING_RATIO = 0.09

# Derived vertical positions; edit the five SPACING_* values above instead.
LEGEND_Y = FIGURE_TITLE_Y - SPACING_AFTER_TITLE
DESCRIPTION_Y = LEGEND_Y - SPACING_AFTER_LEGEND
GRID_TOP_MARGIN = DESCRIPTION_Y - SPACING_AFTER_DESCRIPTION

DATASET_ORDER = [
    "simulated-dataset",
    "arni-dataset",
    "motus-dataset",
    "ivan-dataset",
]

# Exact RGB values from the supplied plotting code.
KNOWN_DATASET_COLORS = {
    "checkpoint_dataset": np.array([44, 160, 44]) / 255.0,
    "simulated-dataset": np.array([44, 160, 44]) / 255.0,
    "arni-dataset": np.array([31, 119, 180]) / 255.0,
    "ivan-dataset": np.array([255, 127, 14]) / 255.0,
    "motus-dataset": np.array([148, 103, 189]) / 255.0,
}

DATASET_LEGEND_LABELS = {
    "checkpoint_dataset": "Simulation dataset",
    "simulated-dataset": "Simulation dataset",
    "arni-dataset": "Arni dataset",
    "ivan-dataset": "Agrid dataset",
    "motus-dataset": "Motus dataset",
}

# Associate each shade with its model name, so model subsets retain their
# original colors. The blue/green shades come from the alternative model group
# in the attached reference and are used by ``--all-models``.
MODEL_BACKGROUND_COLORS = {
    "MUSIC": "#d0d0d0",
    "shoeBox_SH_Direct": "#006b6b",
    "shoeBox_em32_Direct": "#3bc2b9",
    "shoeBox_SH_Direct-Late": "#62938f",
    "shoeBox_em32_Direct-Late": "#a1d6d1",
    "shoeBox_SH_ISMOrder2": "#8a6a00",
    "shoeBox_em32_ISMOrder2": "#e6b710",
    "shoeBox_SH_ISMOrder2-Late": "#a28e4b",
    "shoeBox_em32_ISMOrder2-Late": "#dfcd88",
    "shoeBox_SH_ISMOrder12": "#951010",
    "shoeBox_em32_ISMOrder12": "#ec4e4e",
    "shoeBox_SH_ISMOrder12-Late": "#a96868",
    "shoeBox_em32_ISMOrder12-Late": "#e7a5a5",
    "non-shoeBox_disp-corners-05_SH_ISMOrder12": "#0108C8",
    "non-shoeBox_disp-corners-1_SH_ISMOrder12": "#5C61E0",
    "non-shoeBox_disp-corners-3_SH_ISMOrder12": "#A9ABF6",
    "shoeBox_rand-ism_rand-disp-025_SH_ISMOrder12": "#00B00F",
    "shoeBox_rand-ism_rand-disp-1_SH_ISMOrder12": "#54D861",
    "shoeBox_rand-ism_rand-disp-5_SH_ISMOrder12": "#8FDC97",
    "shoeBox_rand-ism_rand-disp-10_SH_ISMOrder12": "#CCF2D1",
}

METRIC_LABELS = {
    "LR":   r"$\mathrm{LR}$",
    "LE":   r"$\mathrm{LE}$",
    "ER15": r"$\mathrm{ER}_{15}$",
    "F15":  r"$\mathrm{F}_{15}$",
    "ESDL": r"$\mathrm{E}_{\mathrm{SDL}}$",
    "count_MAE": "Count MAE",
    "count_accuracy": "Count accuracy",
}

METRIC_TITLES = {
    "LR": "Localization Recall (LR) ↑",
    "LE": "Localization Error (LE) ↓",
    "ER15": r"Location-Sensitive Error Rate ($\mathbf{ER}_{\mathbf{15}}$) ↓",
    "F15": r"Location-Sensitive F-Score ($\mathbf{F}_{\mathbf{15}}$) ↑",
    "ESDL": r"$\mathbf{E}_{\mathbf{SDL}}\mathbf{-Score}$ ↓",
    "count_MAE": "Source-count mean absolute error",
    "count_accuracy": "Source-count accuracy",
}

MODEL_PARAMETER_HEADERS = ("Mic", "O", "L", "C", "R")
DISPLACEMENT_METERS_BY_TOKEN = {
    "025": "0.25",
    "05": "0.5",
    "1": "1",
    "3": "3",
    "5": "5",
    "10": "10",
}

REQUIRED_COLUMNS = {
    "dataset",
    "model",
    "metric",
    "rt60_bin_center",
    "full_sample",
    "estimate",
    "ci_low",
    "ci_high",
}


def split_requested_values(values: Sequence[str] | None) -> list[str] | None:
    """Accept both ``--metrics LE LR`` and ``--metrics LE,LR``."""
    if values is None:
        return None
    return [item.strip() for value in values for item in value.split(",") if item.strip()]


def resolve_csv_path(csv_path: str | Path) -> Path:
    """Find the CSV beside the script, in the current directory, or in upload/."""
    path = Path(csv_path).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        candidates = [path]
    else:
        candidates = [Path.cwd() / path, Path(__file__).resolve().parent / path]
        candidates.append(Path.cwd() / "upload" / path)

    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"RT60 metrics CSV not found. Paths checked:\n  {searched}")


def ordered_selection(
    frame: pd.DataFrame,
    column: str,
    requested: Sequence[str] | None,
    description: str,
) -> list[str]:
    """Validate a requested selection while keeping its configured order."""
    available = list(dict.fromkeys(frame[column].dropna().astype(str)))
    if requested is None:
        return available

    selected = list(requested)
    if not selected:
        raise ValueError(f"No {description} were selected.")

    duplicated = [item for item in dict.fromkeys(selected) if selected.count(item) > 1]
    if duplicated:
        raise ValueError(f"Duplicate {description}: {', '.join(duplicated)}")

    missing = [item for item in selected if item not in available]
    if missing:
        raise ValueError(
            f"Selected {description} not found in the CSV: {', '.join(missing)}\n"
            f"Available {description}: {', '.join(available)}"
        )
    return selected


def model_parameter_values(model_name: str) -> tuple[str, str, str, str, str]:
    """Convert the same model-name tokens as the supplied plotting code."""
    if model_name.upper() == "MUSIC":
        return ("MUSIC", "-", "-", "-", "-")

    microphone_match = re.search(r"_(SH|em32)_(?:Direct|ISMOrder)", model_name)
    order_match = re.search(r"_ISMOrder(\d+)(?:-Late)?$", model_name)
    corner_match = re.search(r"non-shoeBox_disp-corners-([^_]+)_", model_name)
    random_match = re.search(r"_rand-disp-([^_]+)_", model_name)

    microphone = microphone_match.group(1) if microphone_match else "?"
    if microphone == "em32":
        microphone = "EM"
    if order_match:
        order = order_match.group(1)
    elif re.search(r"_Direct(?:-Late)?$", model_name):
        order = "0"
    else:
        order = "?"

    def displacement(match: re.Match[str] | None) -> str:
        if match is None:
            return "0"
        return DISPLACEMENT_METERS_BY_TOKEN.get(match.group(1), match.group(1))

    return (
        microphone,
        order,
        "1" if model_name.endswith("-Late") else "0",
        displacement(corner_match),
        displacement(random_match),
    )


def model_background(model_name: str, model_index: int):
    """Return a stable reference color, with tab20 fallback for unknown models."""
    return MODEL_BACKGROUND_COLORS.get(model_name, plt.get_cmap("tab20")(model_index % 20))


def metric_limits(values: pd.Series, metric: str) -> tuple[float, float]:
    """Fit observed values instead of allowing sparse-bin CIs to flatten curves."""
    finite = values[np.isfinite(values.to_numpy(dtype=float))].to_numpy(dtype=float)
    if not len(finite):
        return (0.0, 1.0)

    low = float(np.min(finite))
    high = float(np.max(finite))
    span = max(high - low, abs(high) * 0.08, 0.08)
    padding = span * Y_AXIS_DATA_PADDING_RATIO

    if metric in {"LR", "F15", "count_accuracy"}:
        return (max(0.0, low - padding), min(1.0, high + padding))

    return (max(0.0, low - padding), high + padding)


def order_datasets(datasets: Sequence[str]) -> list[str]:
    """Keep the original reference legend order, followed by unknown datasets."""
    priorities = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    priorities.setdefault("checkpoint_dataset", priorities.get("simulated-dataset", 0))
    first_seen = {dataset: index for index, dataset in enumerate(datasets)}
    return sorted(datasets, key=lambda dataset: (priorities.get(dataset, len(priorities)), first_seen[dataset]))


def add_header(
    fig: plt.Figure,
    header_axes: Sequence[plt.Axes],
    models: Sequence[str],
    datasets: Sequence[str],
    dataset_colors: dict[str, object],
    confidence_intervals: bool,
) -> None:
    """Draw the shared dataset legend and original model-parameter table."""
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=dataset_colors[dataset],
            linewidth=4,
            marker="o",
            markersize=8,
            markeredgewidth=0,
            label=DATASET_LEGEND_LABELS.get(dataset, dataset),
        )
        for dataset in datasets
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(LEGEND_CENTER_X, LEGEND_Y),
        ncol=min(4, len(legend_handles)),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
        handlelength=LEGEND_HANDLE_LENGTH,
        columnspacing=LEGEND_COLUMN_SPACING,
    )

    explanation = (
        r"$\mathbf{Mic}$ = Microphone Simulation | "
        r"$\mathbf{O}$ = ISM Order | "
        r"$\mathbf{L}$ = Diffuse Late Reverb | "
        r"$\mathbf{C}$ = Room Corner Displacement Range (m) | "
        r"$\mathbf{R}$ = Random Image-Source Displacement Range (m)"
    )

    fig.text(
        DESCRIPTION_CENTER_X,
        DESCRIPTION_Y,
        explanation,
        ha="center",
        va="center",
        fontsize=FONT_SIZE_EXPLANATION,
        linespacing=EXPLANATION_LINE_SPACING,
        color="#444444",
    )

    for model_index, (axis, model) in enumerate(zip(header_axes, models)):
        axis.set_facecolor("none")
        axis.patch.set_alpha(0.0)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim(0, 1)
        axis.set_ylim(-0.7, len(MODEL_PARAMETER_HEADERS) - 0.3)
        axis.invert_yaxis()
        axis.axvspan(0, 1, color=model_background(model, model_index), alpha=MODEL_BACKGROUND_ALPHA, linewidth=0)

        for spine in axis.spines.values():
            spine.set_visible(False)

        for field_index, value in enumerate(model_parameter_values(model)):
            axis.text(
                0.5,
                field_index,
                value,
                ha="center",
                va="center",
                fontsize=FONT_SIZE_MODEL_PARAMETERS,
                family="monospace",
                fontweight="normal" if field_index == 0 else "normal",
            )

    first_axis = header_axes[0]
    for field_index, field_name in enumerate(MODEL_PARAMETER_HEADERS):
        first_axis.text(
            MODEL_PARAMETER_LABEL_X,
            field_index,
            field_name,
            ha="right",
            va="center",
            fontsize=FONT_SIZE_MODEL_PARAMETERS,
            family="monospace",
            fontweight="bold",
            clip_on=False,
        )


def plot(
    csv_path: str | Path = CSV_PATH,
    plotted_models: Sequence[str] | None = PLOTTED_MODELS,
    plotted_metrics: Sequence[str] | None = PLOTTED_METRICS,
    output_path: str | Path = OUTPUT_PATH,
    music_plotted_metrics: Sequence[str] | None = MUSIC_PLOTTED_METRICS,
    show_confidence_intervals: bool = SHOW_CONFIDENCE_INTERVALS,
    value_column: str = VALUE_COLUMN,
) -> Path:
    """Save every requested model and metric in a single-page PDF figure."""
    source_path = resolve_csv_path(csv_path)
    frame = pd.read_csv(source_path)

    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"RT60 metrics CSV is missing columns: {', '.join(sorted(missing_columns))}")
    if frame.empty:
        raise ValueError(f"RT60 metrics CSV contains no data: {source_path}")
    if value_column not in {"full_sample", "estimate"}:
        raise ValueError("value_column must be 'full_sample' or 'estimate'.")

    models = ordered_selection(frame, "model", plotted_models, "models")
    selected = frame.loc[frame["model"].isin(models)].copy()
    metrics = ordered_selection(selected, "metric", plotted_metrics, "metrics")
    selected = selected.loc[selected["metric"].isin(metrics)].copy()

    if music_plotted_metrics is None:
        selected_music_metrics = set(metrics)
    else:
        selected_music_metrics = set(music_plotted_metrics).intersection(metrics)

    datasets = order_datasets(ordered_selection(selected, "dataset", None, "datasets"))
    color_map = plt.get_cmap("tab20")
    dataset_colors = {
        dataset: KNOWN_DATASET_COLORS.get(dataset, color_map(index % 20)[:3])
        for index, dataset in enumerate(datasets)
    }

    numeric_columns = ["rt60_bin_center", value_column, "ci_low", "ci_high"]
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    selected = selected.dropna(subset=["rt60_bin_center", value_column])
    if selected.empty:
        raise ValueError("Selected models and metrics contain no finite RT60 values.")

    duplicates = selected.duplicated(subset=["dataset", "model", "metric", "rt60_bin_center"])
    if duplicates.any():
        duplicate = selected.loc[duplicates, ["dataset", "model", "metric", "rt60_bin_center"]].iloc[0]
        raise ValueError(f"Duplicate dataset/model/metric/RT60 combination: {duplicate.to_dict()}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": FONT_SIZE_DEFAULT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    })

    n_models = len(models)
    n_metrics = len(metrics)
    figure_width = max(FIGURE_MIN_WIDTH, n_models * FIGURE_WIDTH_PER_MODEL + FIGURE_WIDTH_PADDING)
    figure_height = max(FIGURE_MIN_HEIGHT, n_metrics * FIGURE_HEIGHT_PER_METRIC + FIGURE_HEIGHT_PADDING)
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    # Explicit spacer rows keep the two kinds of vertical grid spacing
    # independent and make their configuration values easier to understand.
    row_height_ratios = [GRID_HEADER_HEIGHT_RATIO, SPACING_AFTER_MIC_ORDER]
    metric_rows: list[int] = []
    for metric_index in range(n_metrics):
        metric_rows.append(len(row_height_ratios))
        row_height_ratios.append(GRID_METRIC_HEIGHT_RATIO)
        if metric_index < n_metrics - 1:
            row_height_ratios.append(SPACING_BETWEEN_METRICS)

    grid = fig.add_gridspec(
        nrows=len(row_height_ratios),
        ncols=n_models,
        height_ratios=row_height_ratios,
        left=max(GRID_LEFT_MARGIN_MIN, GRID_LEFT_PADDING_INCHES / figure_width),
        right=GRID_RIGHT_MARGIN,
        bottom=GRID_BOTTOM_MARGIN,
        top=GRID_TOP_MARGIN,
        wspace=GRID_HORIZONTAL_SPACING,
        hspace=0,
    )

    fig.suptitle(
        "Model performance versus reverberation time",
        fontsize=FONT_SIZE_FIGURE_TITLE,
        fontweight="bold",
        y=FIGURE_TITLE_Y,
    )
    header_axes = [fig.add_subplot(grid[0, column]) for column in range(n_models)]
    add_header(fig, header_axes, models, datasets, dataset_colors, show_confidence_intervals)

    x_values = selected["rt60_bin_center"].to_numpy(dtype=float)
    x_lower = max(0.0, float(np.min(x_values)) - X_AXIS_DATA_PADDING)
    x_upper = float(np.max(x_values)) + X_AXIS_DATA_PADDING
    x_tick_step = 1.0 if x_upper - x_lower <= 5.5 else 2.0

    axes: list[list[plt.Axes]] = []
    for metric_index, metric in enumerate(metrics):
        metric_frame = selected.loc[selected["metric"] == metric]
        plottable_metric_frame = metric_frame.loc[
            (metric_frame["model"].str.upper() != "MUSIC") | (metric in selected_music_metrics)
        ]
        lower, upper = metric_limits(plottable_metric_frame[value_column], metric)
        metric_axes: list[plt.Axes] = []

        for model_index, model in enumerate(models):
            share_y = metric_axes[0] if metric_axes else None
            share_x = axes[0][model_index] if axes else None
            axis = fig.add_subplot(grid[metric_rows[metric_index], model_index], sharex=share_x, sharey=share_y)
            metric_axes.append(axis)

            axis.set_facecolor("none")
            axis.axvspan(
                x_lower,
                x_upper,
                color=model_background(model, model_index),
                alpha=MODEL_BACKGROUND_ALPHA,
                linewidth=0,
                zorder=0,
            )
            axis.grid(axis="y", color="black", linewidth=0.9, alpha=0.10, zorder=1)
            axis.set_axisbelow(True)
            axis.set_xlim(x_lower, x_upper)
            axis.set_ylim(lower, upper)
            axis.xaxis.set_major_locator(MultipleLocator(x_tick_step))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))

            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            for spine in ("bottom", "left"):
                axis.spines[spine].set_color("#777777")

            axis.tick_params(
                axis="both",
                labelsize=FONT_SIZE_TICK_LABELS,
                length=TICK_LENGTH,
                color="#666666",
            )
            if model_index:
                axis.tick_params(axis="y", left=False, labelleft=False)
                axis.spines["left"].set_visible(False)
            else:
                axis.set_ylabel(
                    METRIC_LABELS.get(metric, metric),
                    fontsize=FONT_SIZE_Y_AXIS_LABEL,
                    fontweight="bold",
                    labelpad=Y_AXIS_LABEL_PADDING,
                )

            if metric_index < n_metrics - 1:
                axis.tick_params(axis="x", labelbottom=False)

            if model.upper() == "MUSIC" and metric not in selected_music_metrics:
                axis.text(
                    0.5,
                    0.5,
                    "",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE_NOT_PLOTTED,
                    color="#777777",
                    style="italic",
                )
                continue

            model_frame = metric_frame.loc[metric_frame["model"] == model]
            for dataset in datasets:
                series = model_frame.loc[model_frame["dataset"] == dataset].sort_values("rt60_bin_center")
                if series.empty:
                    continue

                x = series["rt60_bin_center"].to_numpy(dtype=float)
                y = series[value_column].to_numpy(dtype=float)
                color = dataset_colors[dataset]

                if show_confidence_intervals:
                    low = series["ci_low"].to_numpy(dtype=float)
                    high = series["ci_high"].to_numpy(dtype=float)
                    valid = np.isfinite(low) & np.isfinite(high) & (high >= low)
                    if "jackknife_valid" in series.columns:
                        valid &= series["jackknife_valid"].astype(str).str.lower().isin(["true", "1"]).to_numpy()

                    # A filled band needs at least two adjacent valid points.
                    # Draw a capped vertical interval for every valid point that
                    # has no valid neighbour, including a one-point RT60 series.
                    previous_valid = np.r_[False, valid[:-1]]
                    next_valid = np.r_[valid[1:], False]
                    connected = valid & (previous_valid | next_valid)
                    isolated = valid & ~connected

                    axis.fill_between(
                        x,
                        np.maximum(low, lower),
                        np.minimum(high, upper),
                        where=connected & (high >= lower) & (low <= upper),
                        color=color,
                        alpha=CONFIDENCE_ALPHA,
                        linewidth=0,
                        interpolate=False,
                        zorder=2,
                    )

                    visible_isolated = isolated & (high >= lower) & (low <= upper)
                    if np.any(visible_isolated):
                        clipped_low = np.maximum(low[visible_isolated], lower)
                        clipped_high = np.minimum(high[visible_isolated], upper)
                        interval_center = (clipped_low + clipped_high) / 2.0
                        interval_half_height = (clipped_high - clipped_low) / 2.0
                        axis.errorbar(
                            x[visible_isolated],
                            interval_center,
                            yerr=interval_half_height,
                            fmt="none",
                            ecolor=color,
                            elinewidth=1.5,
                            capsize=3.5,
                            capthick=1.5,
                            alpha=0.78,
                            zorder=2.5,
                        )

                axis.plot(
                    x,
                    y,
                    color=color,
                    linewidth=1.65,
                    marker="o",
                    markersize=3.5,
                    markeredgewidth=0,
                    solid_capstyle="round",
                    zorder=3,
                )

        first_position = metric_axes[0].get_position()
        last_position = metric_axes[-1].get_position()
        fig.text(
            (first_position.x0 + last_position.x1) / 2,
            first_position.y1 + METRIC_TITLE_Y_PADDING,
            METRIC_TITLES.get(metric, metric),
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_METRIC_TITLE,
            fontweight="semibold",
        )
        axes.append(metric_axes)

    fig.supxlabel(
        r"Reverberation time RT (s)",
        fontsize=FONT_SIZE_X_AXIS_LABEL,
        y=X_AXIS_LABEL_Y,
    )

    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != ".pdf":
        raise ValueError("Output filename must end in '.pdf' so all plots are saved in one PDF.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to model_metrics_by_rt60.csv.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Destination PDF filename.")

    metric_group = parser.add_mutually_exclusive_group()
    metric_group.add_argument("--metrics", nargs="+", metavar="METRIC", help="Metrics to plot; comma-separated values also work.")
    metric_group.add_argument("--all-metrics", action="store_true", help="Plot every metric present in the CSV.")

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--models", nargs="+", metavar="MODEL", help="Model names to plot in the specified order.")
    model_group.add_argument("--all-models", action="store_true", help="Plot every model present in the CSV.")

    parser.add_argument(
        "--music-metrics",
        nargs="+",
        metavar="METRIC",
        help="Metrics that include MUSIC; use 'all' to show MUSIC for every selected metric.",
    )
    parser.add_argument("--value-column", choices=["full_sample", "estimate"], default=VALUE_COLUMN)
    parser.add_argument("--no-ci", action="store_true", help="Hide the translucent confidence-interval bands.")
    parser.add_argument("--list-metrics", action="store_true", help="Print available CSV metrics and exit.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_metrics:
        available = pd.read_csv(resolve_csv_path(args.csv), usecols=["metric"])
        print("Available metrics:", ", ".join(available["metric"].dropna().unique()))
        return

    metrics = None if args.all_metrics else split_requested_values(args.metrics) or PLOTTED_METRICS
    models = None if args.all_models else split_requested_values(args.models) or PLOTTED_MODELS

    if args.music_metrics is None:
        music_metrics = MUSIC_PLOTTED_METRICS
    elif len(args.music_metrics) == 1 and args.music_metrics[0].lower() == "all":
        music_metrics = None
    else:
        music_metrics = split_requested_values(args.music_metrics)

    destination = plot(
        csv_path=args.csv,
        plotted_models=models,
        plotted_metrics=metrics,
        output_path=args.output,
        music_plotted_metrics=music_metrics,
        show_confidence_intervals=not args.no_ci,
        value_column=args.value_column,
    )
    print(f"Saved one-page RT60 metrics plot: {destination}")


if __name__ == "__main__":
    main()