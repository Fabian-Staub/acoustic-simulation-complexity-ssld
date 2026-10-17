#!/usr/bin/env python3
"""Plot model metrics against RT60 in one publication-ready PDF figure.

Examples
--------
python plot_rt60_metrics_with_receiver_histogram.py
python plot_rt60_metrics_with_receiver_histogram.py --metrics LE LR F15
python plot_rt60_metrics_with_receiver_histogram.py --metrics LE,ER15,ESDL --output selected.pdf
python plot_rt60_metrics_with_receiver_histogram.py --all-metrics --music-metrics all
python plot_rt60_metrics_with_receiver_histogram.py --all-models --no-ci

Put ``model_metrics_by_rt60.csv`` next to this script, or supply ``--csv``.
Requires matplotlib, numpy, and pandas.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
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

CSV_PATH = Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/significance_test/model_metrics_by_rt60.csv")
OUTPUT_PATH = Path("plot_metrics_vs_rt60_presentation_2.pdf")

# The default models and their order exactly match the supplied plotting script.
# Set this to None to include all models present in the CSV.
PLOTTED_MODELS = [
    "MUSIC",
    "shoeBox_SH_Direct",
    "shoeBox_em32_Direct",
    "shoeBox_SH_Direct-Late",
    "shoeBox_em32_Direct-Late",
    "shoeBox_SH_ISMOrder2",
    "shoeBox_em32_ISMOrder2",
    "shoeBox_SH_ISMOrder2-Late",
    "shoeBox_em32_ISMOrder2-Late",
    "shoeBox_SH_ISMOrder12",
    "shoeBox_em32_ISMOrder12",
    "shoeBox_SH_ISMOrder12-Late",
    "shoeBox_em32_ISMOrder12-Late",
]

# MUSIC is shown only for these metrics. Set to None to show it on every row.
MUSIC_PLOTTED_METRICS = ["LE"]
PLOTTED_METRICS = ["LE", "ER", "ESDL"]

# ``full_sample`` matches the source plotting code. ``estimate`` is the
# jackknife estimate around which the supplied confidence intervals are built.
VALUE_COLUMN = "full_sample"
SHOW_CONFIDENCE_INTERVALS = True
CONFIDENCE_ALPHA = 0.16
MODEL_BACKGROUND_ALPHA = 0.15

# Training histograms average the three source RTs at each receiver position.
# Folder names below follow the supplied example; adjust entries if a training
# dataset uses a different R/P/S, order, sample-rate, or model folder name.
TRAINING_DATA_ROOT = Path(
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/"
    "Projects/Master_Thesis/data/harp"
)
TRAINING_MODEL_DATASET_FOLDERS = {
    "shoeBox_SH_Direct": "R2000_P8_S3_Order3_24kHz_shoeBox_SH_Direct",
    "shoeBox_em32_Direct": "R2000_P8_S3_Order3_24kHz_shoeBox_em32_Direct",
    "shoeBox_SH_Direct-Late": "R2000_P8_S3_Order3_24kHz_shoeBox_SH_Direct-Late",
    "shoeBox_em32_Direct-Late": "R2000_P8_S3_Order3_24kHz_shoeBox_em32_Direct-Late",
    "shoeBox_SH_ISMOrder2": "R2000_P8_S3_Order3_24kHz_shoeBox_SH_ISMOrder2",
    "shoeBox_em32_ISMOrder2": "R2000_P8_S3_Order3_24kHz_shoeBox_em32_ISMOrder2",
    "shoeBox_SH_ISMOrder2-Late": "R2000_P8_S3_Order3_24kHz_shoeBox_SH_ISMOrder2-Late",
    "shoeBox_em32_ISMOrder2-Late": "R2000_P8_S3_Order3_24kHz_shoeBox_em32_ISMOrder2-Late",
    "shoeBox_SH_ISMOrder12": "R2000_P8_S3_Order3_24kHz_shoeBox_SH_ISMOrder12",
    "shoeBox_em32_ISMOrder12": "R2000_P8_S3_Order3_24kHz_shoeBox_em32_ISMOrder12",
    "shoeBox_SH_ISMOrder12-Late": "R2000_P8_S3_Order3_24kHz_shoeBox_SH_ISMOrder12-Late",
    "shoeBox_em32_ISMOrder12-Late": "R2000_P8_S3_Order3_24kHz_shoeBox_em32_ISMOrder12-Late",
    "non-shoeBox_disp-corners-05_SH_ISMOrder12": "R2000_P8_S3_Order3_24kHz_non-shoeBox_disp-corners-05_SH_ISMOrder12",
    "non-shoeBox_disp-corners-1_SH_ISMOrder12": "R2000_P8_S3_Order3_24kHz_non-shoeBox_disp-corners-1_SH_ISMOrder12",
    "non-shoeBox_disp-corners-3_SH_ISMOrder12": "R2000_P8_S3_Order3_24kHz_non-shoeBox_disp-corners-3_SH_ISMOrder12",
    "shoeBox_rand-ism_rand-disp-025_SH_ISMOrder12": "R2000_P8_S3_Order3_24kHz_shoeBox_rand-ism_rand-disp-025_SH_ISMOrder12",
    "shoeBox_rand-ism_rand-disp-1_SH_ISMOrder12": "R2000_P8_S3_Order3_24kHz_shoeBox_rand-ism_rand-disp-1_SH_ISMOrder12",
    "shoeBox_rand-ism_rand-disp-5_SH_ISMOrder12": "R2000_P8_S3_Order3_24kHz_shoeBox_rand-ism_rand-disp-5_SH_ISMOrder12",
    "shoeBox_rand-ism_rand-disp-10_SH_ISMOrder12": "R2000_P8_S3_Order3_24kHz_shoeBox_rand-ism_rand-disp-10_SH_ISMOrder12",
}
# Each model may have multiple training dataset folders. Add further RIRs
# folders to its list to combine their task CSVs. MUSIC has no training dataset.
TRAINING_RT60_PATHS = {
    model: [TRAINING_DATA_ROOT / folder / "RIRs"]
    for model, folder in TRAINING_MODEL_DATASET_FOLDERS.items()
}
TRAINING_RT60_TASK_GLOB = "Generated_HOA_SRIR_data_task_*.csv"
TRAINING_RT60_COLUMN = "RT"
TRAINING_SOURCES_PER_RECEIVER = 3
# Set explicitly to your CSV columns, e.g. ("room_id", "receiver_id").
# None detects common room/receiver identifiers. Receiver coordinates are also
# supported. Room IDs are included when present; IDs are scoped to each file.
# Compound position IDs such as "0_S0" are grouped by their receiver part "0".
TRAINING_RECEIVER_GROUP_COLUMNS = None
RT60_HISTOGRAM_BIN_WIDTH = 0.2  # Seconds; identical edges for all models.
RT60_HISTOGRAM_TITLE = "RT Distribution Training Datasets"
RT60_HISTOGRAM_BAR_COLOR = "#888888"
# Optional override for a single distribution file shared by all models.
# With None, missing training paths raise an error rather than using eval data.
RT60_DISTRIBUTION_CSV = None
RT60_HISTOGRAM_HEIGHT_RATIO = 2 / 3
METRIC_GRID_COLOR = "#777777"
METRIC_GRID_ALPHA = 0.16
METRIC_GRID_LINEWIDTH = 0.6
SIMULATED_DATASETS = {"simulated-dataset", "checkpoint_dataset"}

# Font sizes (points). Change these values to scale individual figure elements.
FONT_SIZE_DEFAULT = 15
FONT_SIZE_FIGURE_TITLE = 28
FONT_SIZE_LEGEND = 50
FONT_SIZE_EXPLANATION = 20
FONT_SIZE_MODEL_PARAMETERS = 50
FONT_SIZE_TICK_LABELS = 44
FONT_SIZE_Y_AXIS_LABEL = 44
FONT_SIZE_NOT_PLOTTED = 15
FONT_SIZE_METRIC_TITLE = 48
FONT_SIZE_X_AXIS_LABEL = 50

# Figure dimensions. Figure sizes are inches; margins and vertical offsets use
# normalized figure coordinates.
FIGURE_MIN_WIDTH = 29
FIGURE_WIDTH_PER_MODEL = 10
FIGURE_WIDTH_PADDING = 0
FIGURE_MIN_HEIGHT = 0
FIGURE_HEIGHT_PER_METRIC = 4.5
FIGURE_HEIGHT_PADDING = 7.2

# Vertical gaps in inches. These controls are applied directly to the layout.
SPACING_AFTER_TITLE = 0.03  # Reserved for an optional figure title.
SPACING_AFTER_LEGEND = 1
SPACING_BETWEEN_MODEL_PARAMETERS = 0
SPACING_AFTER_MODEL_PARAMETERS = 0.6  # Vertical gap below the model settings.
SPACING_AFTER_METRIC_TITLE = 0.05
SPACING_AFTER_METRIC = 0.8
SPACING_AFTER_MODEL_GROUP_HEADING = 0.3

FIGURE_TITLE_Y = 0.993
LEGEND_CENTER_X = 0.53
LEGEND_TOP_PADDING_INCHES = 0.15
GRID_BOTTOM_PADDING_INCHES = 1.4

GRID_METRIC_HEIGHT_RATIO = 5
GRID_LEFT_MARGIN_MIN = 0.070
GRID_LEFT_PADDING_INCHES = 4.6
GRID_RIGHT_MARGIN = 0.945
SPACING_BETWEEN_MODELS = 0.06  # Column gap as a fraction of panel width.
LEGEND_HANDLE_LENGTH = 2.4
LEGEND_COLUMN_SPACING = 1.3
LEGEND_LINEWIDTH = 8.0
LEGEND_FRAME_LINEWIDTH = 0.5
LEGEND_FRAME_COLOR = "#D6D6D6"
LEGEND_BORDER_PADDING = 0.45
EXPLANATION_LINE_SPACING = 1.6
MODEL_PARAMETER_LABEL_X = -0.13
TICK_LENGTH = 10
Y_AXIS_LABEL_PADDING = 12
X_AXIS_LABEL_Y = 0.027
X_AXIS_DATA_PADDING = 0.1
Y_AXIS_DATA_PADDING_RATIO = 0.09
MAX_RT60_SECONDS = 3.0

DATASET_ORDER = [
    "simulated-dataset",
    "arni-dataset",
    "motus-dataset",
    "ivan-dataset",
]

# Exact RGB values from the supplied plotting code.
KNOWN_DATASET_COLORS = {
    "checkpoint_dataset": np.array([100, 100, 100]) / 255.0,
    "simulated-dataset": np.array([150, 150, 150]) / 255.0,
    "arni-dataset": np.array([44, 160, 44]) / 255.0,
    #"arni-dataset": np.array([31, 119, 180]) / 255.0,
    "ivan-dataset": np.array([255, 127, 14]) / 255.0,
    #"motus-dataset": np.array([148, 103, 189]) / 255.0,
    "motus-dataset": np.array([31, 119, 180]) / 255.0,
}

DATASET_LEGEND_LABELS = {
    "checkpoint_dataset": "Simulation dataset",
    "simulated-dataset": "Simulation dataset",
    "arni-dataset": "Arni dataset",
    "ivan-dataset": "AGrid dataset",
    "motus-dataset": "MOTUS dataset",
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
    "shoeBox_rand-ism_rand-disp-025_SH_ISMOrder12": "#01940D",
    "shoeBox_rand-ism_rand-disp-1_SH_ISMOrder12": "#25DA37",
    "shoeBox_rand-ism_rand-disp-5_SH_ISMOrder12": "#7BB381",
    "shoeBox_rand-ism_rand-disp-10_SH_ISMOrder12": "#CCF2D1",
}

METRIC_LABELS = {
    "LR":   r"$\mathrm{LR}$",
    "LE":   r"$\mathrm{LE}$",
    "ER15": r"$\mathrm{ER}_{15}$",
    "F15":  r"$\mathrm{F}_{15}$",
    "ESDL": r"$\mathrm{E}_{\mathrm{SSLD}}$",
    "F":   r"$\mathrm{F}_{1}$",
    "count_MAE": r"SC-MAE",
    "ER":   r"$\mathrm{ER}$",
}

METRIC_TITLES = {
    "LR": "Localization Recall (LR) ↑",
    "LE": "Localization Error (LE) ↓",
    "ER15": r"Location-Sensitive Error Rate ($\mathbf{ER}_{\mathbf{15}}$) ↓",
    "F15": r"Location-Sensitive F-Score ($\mathbf{F}_{\mathbf{15}}$) ↑",
    "ESDL": r"$\mathbf{E}_{\mathbf{SSLD}}\mathbf{-Score}$ ↓",
    "F": r"$\mathbf{F}_{\mathbf{1}}$-Score ↑",
    "count_MAE": r"Source Count Mean Absolute Error (SC-MAE) ↓",
    "ER": r"(Non-Location-Sensitive) Error Rate (ER) ↓",
}

MODEL_SETTING_ROWS = ("Late Reverb", "Array Sim.", "Randomization")
MODEL_GROUP_HEADING_TEXT_SIZE = 48
MODEL_GROUP_SEPARATOR_COLOR = "#666666"
MODEL_GROUP_SEPARATOR_WIDTH = 1.5
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


def displacement_from_token(token, model_name):
    """Translate a model-name displacement token to its value in meters."""
    try:
        return DISPLACEMENT_METERS_BY_TOKEN[token]
    except KeyError as error:
        raise ValueError(
            f"Unknown displacement token {token!r} in model {model_name!r}. "
            "Add its value to DISPLACEMENT_METERS_BY_TOKEN."
        ) from error

def is_music_model(model_name):
    """Return whether a CSV model name identifies the MUSIC baseline."""
    stem = str(model_name).strip()
    if stem.startswith("eval_"):
        stem = stem[5:]
    stem = re.sub(r"\(\d+\)$", "", stem)
    return re.search(r"(?:^|_)MUSIC$", stem, flags=re.IGNORECASE) is not None

def model_heading_values(model_name):
    """Return ISM order, microphone simulation, late reverb, and randomization."""
    stem = str(model_name).strip()
    if is_music_model(stem):
        return ("n/a", "n/a", "n/a")

    if stem.startswith("eval_"):
        stem = stem[5:]
    stem = re.sub(r"\(\d+\)$", "", stem)

    if re.search(r"_Direct(?:-Late)?$", stem):
        ism_order = "0"
    else:
        order_match = re.search(r"_ISMOrder(\d+)(?:-Late)?$", stem)
        if order_match is None:
            raise ValueError(
                f"Cannot determine ISM order from model {model_name!r}; expected "
                "'_Direct' or '_ISMOrder<number>', optionally '-Late'."
            )
        ism_order = str(int(order_match.group(1)))

    microphone_match = re.search(r"_(SH|em32)_(?:Direct|ISMOrder)", stem)
    if microphone_match is None:
        raise ValueError(
            f"Cannot determine microphone simulation from model {model_name!r}; "
            "expected '_SH_' or '_em32_' before Direct/ISMOrder."
        )
    microphone_simulation = microphone_match.group(1)
    if microphone_simulation == "em32":
        microphone_simulation = "EM"

    late_reverb = "\u2714" if stem.endswith("-Late") else "-"

    random_match = re.search(r"_rand-disp-([^_]+)_", stem)
    random_displacement = (
        displacement_from_token(random_match.group(1), model_name)
        if random_match is not None
        else "-"
    )

    return (late_reverb, microphone_simulation, random_displacement)

def ordinal_suffix(number):
    """Return the English suffix for an integer ordinal."""
    number = int(number)
    if 10 <= number % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")

def model_group_heading(model_name):
    """Return the heading spanning the consecutive variants of a model."""
    stem = str(model_name).strip()
    if is_music_model(stem):
        return "MUSIC"
    if stem.startswith("eval_"):
        stem = stem[5:]
    stem = re.sub(r"\(\d+\)$", "", stem)

    if re.search(r"_Direct(?:-Late)?$", stem):
        return "Direct Sound Only"

    order_match = re.search(r"_ISMOrder(\d+)(?:-Late)?$", stem)
    if order_match is None:
        raise ValueError(
            f"Cannot determine heading for model {model_name!r}; expected "
            "'_Direct' or '_ISMOrder<number>', optionally '-Late'."
        )

    order = int(order_match.group(1))
    suffix = ordinal_suffix(order)
    description = "Randomized ISM" if "_rand-disp-" in stem else "ISM"
    return rf"${order}^{{\mathrm{{{suffix}}}}}$-order {description}"

def consecutive_model_groups(models):
    """Return (start, end, heading) runs for the displayed model order."""
    groups = []
    for model_index, model in enumerate(models):
        heading = model_group_heading(model)
        if groups and groups[-1][2] == heading:
            groups[-1] = (groups[-1][0], model_index, heading)
        else:
            groups.append((model_index, model_index, heading))
    return groups

def add_header(fig, header_axes, models, datasets, dataset_colors, confidence_intervals):
    """Match the reference's shared legend, grouped headings, and settings table."""
    handles = [Line2D([0], [0], color=dataset_colors[dataset], linewidth=LEGEND_LINEWIDTH,
                      marker="o", markersize=20, markeredgewidth=0,
                      label=DATASET_LEGEND_LABELS.get(dataset, dataset))
               for dataset in datasets]
    legend = fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(LEGEND_CENTER_X,
                               1 - LEGEND_TOP_PADDING_INCHES / fig.get_figheight()),
               borderaxespad=0,
               ncol=min(4, len(handles)), frameon=True, fancybox=False,
               facecolor="white", edgecolor=LEGEND_FRAME_COLOR, framealpha=1,
               borderpad=LEGEND_BORDER_PADDING,
               fontsize=FONT_SIZE_LEGEND, handlelength=LEGEND_HANDLE_LENGTH,
               columnspacing=LEGEND_COLUMN_SPACING)
    legend.get_frame().set_linewidth(LEGEND_FRAME_LINEWIDTH)
    axis = header_axes[0]
    axis.set_axis_off()
    axis.set_xlim(-0.5, len(models) - 0.5)
    heading_height = MODEL_GROUP_HEADING_TEXT_SIZE / 72 * 1.3
    parameter_height = FONT_SIZE_MODEL_PARAMETERS / 72 * 1.3
    height = (heading_height + SPACING_AFTER_MODEL_GROUP_HEADING
              + len(MODEL_SETTING_ROWS) * parameter_height
              + max(0, len(MODEL_SETTING_ROWS) - 1)
              * SPACING_BETWEEN_MODEL_PARAMETERS)
    axis.set_ylim(height, 0)
    # Match actual model panel positions, including the gaps between columns.
    slot = axis.get_position()
    grid = axis.get_subplotspec().get_gridspec()
    bounds = []
    for index in range(len(models)):
        pos = grid[0, index].get_position(fig)
        bounds.append(((pos.x0 - slot.x0) / slot.width * len(models) - 0.5,
                       (pos.x1 - slot.x0) / slot.width * len(models) - 0.5))
    centers = [(left + right) / 2 for left, right in bounds]
    heading_y = heading_height / 2
    parameter_top = heading_height + SPACING_AFTER_MODEL_GROUP_HEADING
    row_y = [parameter_top + parameter_height / 2
             + i * (parameter_height + SPACING_BETWEEN_MODEL_PARAMETERS)
             for i in range(len(MODEL_SETTING_ROWS))]
    for start, end, heading in consecutive_model_groups(models):
        axis.text((centers[start] + centers[end]) / 2, heading_y, heading, ha="center", va="center",
                  fontsize=MODEL_GROUP_HEADING_TEXT_SIZE)
        if start:
            axis.axvline((bounds[start - 1][1] + bounds[start][0]) / 2, color=MODEL_GROUP_SEPARATOR_COLOR,
                         linewidth=MODEL_GROUP_SEPARATOR_WIDTH)
    for y, label in zip(row_y, MODEL_SETTING_ROWS):
        axis.text(-0.58, y, label, ha="right", va="center",
                  fontsize=FONT_SIZE_MODEL_PARAMETERS, clip_on=False)
    for index, model in enumerate(models):
        axis.axvspan(*bounds[index], ymin=0,
                     ymax=1 - parameter_top / height,
                     color=model_background(model, index),
                     alpha=MODEL_BACKGROUND_ALPHA, linewidth=0)
        for y, value in zip(row_y, model_heading_values(model)):
            axis.text(centers[index], y, value, ha="center", va="center",
                      fontsize=FONT_SIZE_MODEL_PARAMETERS)
    return legend


def receiver_mean_rt60_values(data, rt_column=TRAINING_RT60_COLUMN, context="CSV"):
    """Return one RT mean per receiver, requiring all three valid source RTs."""
    if rt_column not in data:
        raise ValueError(f"{context}: missing RT column {rt_column!r}.")
    if TRAINING_RECEIVER_GROUP_COLUMNS is not None:
        keys = list(TRAINING_RECEIVER_GROUP_COLUMNS)
        if not keys or any(key not in data for key in keys):
            raise ValueError(f"{context}: missing configured grouping columns {keys!r}.")
    else:
        # Match case and punctuation variants without assuming CSV row order.
        names = {re.sub(r"[^a-z0-9]", "", str(col).lower()): col for col in data}
        def find(*aliases):
            return next((names[name] for name in aliases if name in names), None)
        room = find("roomid", "roomidx", "roomindex", "room", "r")
        receiver = find("receiverid", "receiveridx", "receiverindex", "receiver",
                        "receiverposition", "receiverpos", "positionid", "positionidx",
                        "positionindex", "position", "pos", "p")
        if receiver is not None:
            keys = [receiver]
        else:
            coordinates = [find("receiver" + c, "receiverpos" + c,
                                "receiverposition" + c, "rec" + c) for c in "xyz"]
            keys = coordinates if all(c is not None for c in coordinates) else []
        if not keys:
            raise ValueError(
                f"{context}: cannot identify receiver positions. Set "
                "TRAINING_RECEIVER_GROUP_COLUMNS to room and receiver columns; "
                f"available columns: {list(data.columns)!r}."
            )
        if room is not None and room not in keys:
            keys.insert(0, room)
    if data[keys].isna().any().any():
        raise ValueError(f"{context}: missing room/receiver grouping identifiers.")
    data = data.copy()
    # Generated task CSVs encode source identity in position_id: 0_S0,
    # 0_S1, 0_S2 describe one receiver. Strip only the trailing source token;
    # room identifiers remain intact, and ordinary receiver IDs are unchanged.
    for key in keys:
        normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
        if normalized_key.startswith(("receiver", "position")) or normalized_key in {"pos", "p"}:
            data[key] = data[key].astype(str).str.replace(r"_S\d+$", "", regex=True,
                                                       flags=re.IGNORECASE)
    values = pd.to_numeric(data[rt_column], errors="coerce")
    data[rt_column] = values.where(np.isfinite(values) & (values >= 0))
    grouped = data.groupby(keys, sort=False, dropna=False)[rt_column]
    sizes = grouped.size()
    valid_counts = grouped.count()
    complete = ((sizes == TRAINING_SOURCES_PER_RECEIVER)
                & (valid_counts == TRAINING_SOURCES_PER_RECEIVER))
    if not complete.all():
        raise ValueError(
            f"{context}: {(~complete).sum()} receiver groups do not have exactly "
            f"{TRAINING_SOURCES_PER_RECEIVER} finite nonnegative source RTs. "
            "Check grouping columns and source rows."
        )
    return grouped.mean().to_numpy(float)


def load_training_rt60_values(model):
    """Combine receiver means from unique task CSVs in the training folders.

    Each file scopes room/receiver identifiers, so reused IDs in other tasks
    remain separate. Every receiver contributes one observation, averaging
    its three sources. Overlapping configured folders do not duplicate files.
    """
    if model not in TRAINING_RT60_PATHS:
        raise ValueError(f"No training dataset paths configured for {model!r}; "
                         "add an entry to TRAINING_RT60_PATHS.")
    files = set()
    for folder in TRAINING_RT60_PATHS[model]:
        folder = Path(folder).expanduser()
        matches = sorted(folder.glob(TRAINING_RT60_TASK_GLOB))
        if not matches:
            raise FileNotFoundError(
                f"No training RT task CSVs for {model!r}: "
                f"{folder / TRAINING_RT60_TASK_GLOB}. "
                "Check TRAINING_DATA_ROOT / TRAINING_RT60_PATHS."
            )
        files.update(path.resolve() for path in matches)
    if not files:
        raise ValueError(f"No training dataset folders configured for {model!r}.")
    chunks = [receiver_mean_rt60_values(pd.read_csv(path), context=str(path))
              for path in sorted(files)]
    values = np.concatenate(chunks)
    if not len(values):
        raise ValueError(f"No receiver RT means for {model!r}.")
    return values


def training_rt60_histograms(models):
    """Use shared fixed-width bins to compare the training datasets."""
    if not np.isfinite(RT60_HISTOGRAM_BIN_WIDTH) or RT60_HISTOGRAM_BIN_WIDTH <= 0:
        raise ValueError("RT60_HISTOGRAM_BIN_WIDTH must be finite and positive.")
    samples = {model: load_training_rt60_values(model)
               for model in models if not is_music_model(model)}
    if not samples:
        return {}
    peak = max(float(values.max()) for values in samples.values())
    n_bins = max(1, int(math.ceil(peak / RT60_HISTOGRAM_BIN_WIDTH)))
    edges = np.arange(n_bins + 1, dtype=float) * RT60_HISTOGRAM_BIN_WIDTH
    if edges[-1] < peak:
        edges = np.append(edges, edges[-1] + RT60_HISTOGRAM_BIN_WIDTH)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    return {model: (centers, widths, np.histogram(values, bins=edges)[0])
            for model, values in samples.items()}


def load_rt60_histogram(frame, distribution_csv=None, model=None):
    """Return bin centers, widths and counts without duplicating metrics rows."""
    data = pd.read_csv(distribution_csv) if distribution_csv is not None else frame.copy()
    if "dataset" in data:
        data = data.loc[data["dataset"].isin(SIMULATED_DATASETS)].copy()
    if distribution_csv is None and model is not None and "model" in data:
        data = data.loc[data["model"] == model].copy()
    if data.empty:
        warnings.warn("No simulated-dataset RT60 distribution rows; histogram omitted.")
        return None
    raw_column = next((name for name in ("RT", "rt60", "RT60", "rt60_seconds")
                       if name in data), None)
    if distribution_csv is not None and raw_column:
        values = receiver_mean_rt60_values(data, rt_column=raw_column,
                                           context=str(distribution_csv))
        if not len(values):
            raise ValueError("RT60 distribution CSV has no finite nonnegative samples.")
        counts, edges = np.histogram(values, bins="auto")
        return (edges[:-1] + edges[1:]) / 2, np.diff(edges), counts
    count_column = next((name for name in ("number_of_receiver_positions", "n_samples", "sample_count", "count", "bin_count", "n")
                         if name in data), None)
    if count_column is None or "rt60_bin_center" not in data:
        warnings.warn("RT60 histogram requires per-bin sample counts in the metrics CSV "
                      "or --rt60-distribution-csv with per-sample rt60 values. Histogram omitted.")
        return None
    # Metrics and models repeat the same dataset bins. Choose a single series,
    # then verify all repeated bin counts agree rather than summing them.
    columns = ["rt60_bin_center", count_column]
    if {"rt60_bin_low", "rt60_bin_high"}.issubset(data.columns):
        columns += ["rt60_bin_low", "rt60_bin_high"]
    elif "rt60_bin_width" in data:
        columns += ["rt60_bin_width"]
    bins = data[columns].apply(pd.to_numeric, errors="coerce").dropna()
    bins = bins.loc[np.isfinite(bins).all(axis=1) & (bins[count_column] >= 0)]
    if bins.groupby("rt60_bin_center")[count_column].nunique().gt(1).any():
        raise ValueError("Simulated RT60 counts differ between repeated model/metric rows; "
                         "supply an authoritative --rt60-distribution-csv.")
    bins = bins.drop_duplicates("rt60_bin_center").sort_values("rt60_bin_center")
    if bins.empty:
        raise ValueError("No valid simulated RT60 bin counts.")
    centers = bins["rt60_bin_center"].to_numpy(float)
    if {"rt60_bin_low", "rt60_bin_high"}.issubset(bins.columns):
        widths = (bins["rt60_bin_high"] - bins["rt60_bin_low"]).to_numpy(float)
    elif "rt60_bin_width" in bins:
        widths = bins["rt60_bin_width"].to_numpy(float)
    elif len(centers) == 1:
        widths = np.array([0.2])
    else:
        edges = np.r_[centers[0] - (centers[1] - centers[0]) / 2,
                      (centers[:-1] + centers[1:]) / 2,
                      centers[-1] + (centers[-1] - centers[-2]) / 2]
        widths = np.diff(edges)
    return centers, widths, bins[count_column].to_numpy(float)


def draw_rt60_histogram(axis, histogram):
    """Draw RT observation counts over the model-colored background."""
    centers, widths, counts = histogram
    axis.bar(centers, counts, width=widths, color=RT60_HISTOGRAM_BAR_COLOR,
             alpha=1.0, edgecolor="white", linewidth=0.5, zorder=2)


def plot(
    csv_path: str | Path = CSV_PATH,
    plotted_models: Sequence[str] | None = PLOTTED_MODELS,
    plotted_metrics: Sequence[str] | None = PLOTTED_METRICS,
    output_path: str | Path = OUTPUT_PATH,
    music_plotted_metrics: Sequence[str] | None = MUSIC_PLOTTED_METRICS,
    show_confidence_intervals: bool = SHOW_CONFIDENCE_INTERVALS,
    value_column: str = VALUE_COLUMN,
    rt60_distribution_csv: str | Path | None = RT60_DISTRIBUTION_CSV,
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
    if rt60_distribution_csv is None:
        histograms = training_rt60_histograms(models)
    else:
        histograms = {model: load_rt60_histogram(frame, rt60_distribution_csv, model)
                      for model in models if not is_music_model(model)}
    histogram_peak = max((float(np.max(hist[2])) for hist in histograms.values()
                          if hist is not None), default=0.0)
    has_histograms = any(hist is not None for hist in histograms.values())
    selected = frame.loc[frame["model"].isin(models)].copy()
    metrics = ordered_selection(selected, "metric", plotted_metrics, "metrics")
    selected = selected.loc[selected["metric"].isin(metrics)].copy()

    if music_plotted_metrics is None:
        selected_music_metrics = set(metrics)
    else:
        selected_music_metrics = set(music_plotted_metrics).intersection(metrics)

    datasets = order_datasets(ordered_selection(selected, "dataset", None, "datasets"))
    # Preserve legend order while drawing ARNI last and above every other series.
    plot_datasets = [dataset for dataset in datasets if dataset != "arni-dataset"]
    if "arni-dataset" in datasets:
        plot_datasets.append("arni-dataset")
    color_map = plt.get_cmap("tab20")
    dataset_colors = {
        dataset: KNOWN_DATASET_COLORS.get(dataset, color_map(index % 20)[:3])
        for index, dataset in enumerate(datasets)
    }

    selected = selected.loc[~(selected["model"].map(is_music_model) &
                               selected["dataset"].isin(SIMULATED_DATASETS))].copy()

    numeric_columns = ["rt60_bin_center", value_column, "ci_low", "ci_high"]
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    selected = selected.dropna(subset=["rt60_bin_center", value_column])
    selected = selected.loc[selected["rt60_bin_center"] <= MAX_RT60_SECONDS].copy()
    if selected.empty:
        raise ValueError(
            f"Selected models and metrics contain no finite RT60 values at or below "
            f"{MAX_RT60_SECONDS:g} seconds."
        )

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
    # Row sizes are inches: increasing a gap grows the figure rather than
    # shrinking the metric panels. Titles have their own rows above the axes.
    header_height = (MODEL_GROUP_HEADING_TEXT_SIZE / 72 * 1.3
                     + SPACING_AFTER_MODEL_GROUP_HEADING
                     + len(MODEL_SETTING_ROWS) * FONT_SIZE_MODEL_PARAMETERS / 72 * 1.3
                     + max(0, len(MODEL_SETTING_ROWS) - 1)
                     * SPACING_BETWEEN_MODEL_PARAMETERS)
    title_height = FONT_SIZE_METRIC_TITLE / 72 * 1.3
    row_height_ratios = [header_height, SPACING_AFTER_MODEL_PARAMETERS]
    metric_rows: list[int] = []
    metric_title_rows: list[int] = []
    for metric_index in range(n_metrics):
        metric_title_rows.append(len(row_height_ratios))
        row_height_ratios.extend([title_height, SPACING_AFTER_METRIC_TITLE])
        metric_rows.append(len(row_height_ratios))
        row_height_ratios.extend([FIGURE_HEIGHT_PER_METRIC, SPACING_AFTER_METRIC])

    histogram_row = None
    histogram_title_row = None
    if has_histograms:
        histogram_title_row = len(row_height_ratios)
        row_height_ratios.extend([title_height, SPACING_AFTER_METRIC_TITLE])
        histogram_row = len(row_height_ratios)
        row_height_ratios.append(FIGURE_HEIGHT_PER_METRIC * RT60_HISTOGRAM_HEIGHT_RATIO)

    # Measure the framed legend so multi-row legends also leave the requested gap.
    fig = plt.figure(figsize=(figure_width, 10), facecolor="white")
    temporary_grid = fig.add_gridspec(1, n_models)
    temporary_header = fig.add_subplot(temporary_grid[0, :])
    legend = add_header(fig, [temporary_header], models, datasets,
                        dataset_colors, show_confidence_intervals)
    fig.canvas.draw()
    legend_height = legend.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
    legend.remove()
    fig.delaxes(temporary_header)
    top_padding = LEGEND_TOP_PADDING_INCHES + legend_height + SPACING_AFTER_LEGEND
    figure_height = max(FIGURE_MIN_HEIGHT,
                        top_padding + sum(row_height_ratios) + GRID_BOTTOM_PADDING_INCHES)
    fig.set_size_inches(figure_width, figure_height)
    grid = fig.add_gridspec(
        nrows=len(row_height_ratios),
        ncols=n_models,
        height_ratios=row_height_ratios,
        left=max(GRID_LEFT_MARGIN_MIN, GRID_LEFT_PADDING_INCHES / figure_width),
        right=GRID_RIGHT_MARGIN,
        bottom=GRID_BOTTOM_PADDING_INCHES / figure_height,
        top=1.0 - top_padding / figure_height,
        wspace=SPACING_BETWEEN_MODELS,
        hspace=0,
    )
    header_axes = [fig.add_subplot(grid[0, :])]
    add_header(fig, header_axes, models, datasets, dataset_colors, show_confidence_intervals)

    x_values = selected["rt60_bin_center"].to_numpy(dtype=float)
    for histogram in histograms.values():
        if histogram is not None:
            centers, widths, _ = histogram
            x_values = np.r_[x_values, centers - widths / 2, centers + widths / 2]
    x_lower = max(0.0, float(np.min(x_values)) - X_AXIS_DATA_PADDING)
    # Leave a small visual margin after 3 s without including later samples.
    # With the 1 s tick spacing, the final labeled tick remains exactly 3 s.
    x_upper = MAX_RT60_SECONDS + X_AXIS_DATA_PADDING
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
            axis.grid(axis="y", color=METRIC_GRID_COLOR,
                      linewidth=METRIC_GRID_LINEWIDTH, alpha=METRIC_GRID_ALPHA, zorder=1)
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
                if model_index in {start for start, _, _ in consecutive_model_groups(models)[1:]}:
                    axis.axvline(x_lower, color=MODEL_GROUP_SEPARATOR_COLOR,
                                 linewidth=MODEL_GROUP_SEPARATOR_WIDTH, zorder=2)
            else:
                axis.set_ylabel(
                    "LE (°)" if metric == "LE" else METRIC_LABELS.get(metric, metric),
                    fontsize=FONT_SIZE_Y_AXIS_LABEL,
                    fontweight="normal" if metric == "LE" else "bold",
                    labelpad=Y_AXIS_LABEL_PADDING,
                )

            if has_histograms or metric_index < n_metrics - 1:
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
            for dataset in plot_datasets:
                if is_music_model(model) and dataset in SIMULATED_DATASETS:
                    continue
                series = model_frame.loc[model_frame["dataset"] == dataset].sort_values("rt60_bin_center")
                if series.empty:
                    continue

                x = series["rt60_bin_center"].to_numpy(dtype=float)
                y = series[value_column].to_numpy(dtype=float)
                color = dataset_colors[dataset]
                foreground_offset = 3 if dataset == "arni-dataset" else 0

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
                        zorder=2 + foreground_offset,
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
                            zorder=2.5 + foreground_offset,
                        )

                axis.plot(
                    x,
                    y,
                    color=color,
                    linewidth=8.0,
                    marker="o",
                    markersize=12,
                    markeredgewidth=0,
                    solid_capstyle="round",
                    zorder=3 + foreground_offset,
                )

        title_axis = fig.add_subplot(grid[metric_title_rows[metric_index], :])
        title_axis.set_axis_off()
        title_axis.text(
            0.5, 0.5, METRIC_TITLES.get(metric, metric),
            transform=title_axis.transAxes,
            ha="center", va="center",
            fontsize=FONT_SIZE_METRIC_TITLE,
            fontweight="bold",
        )
        axes.append(metric_axes)

    if has_histograms:
        histogram_axes = []
        title_axis = fig.add_subplot(grid[histogram_title_row, :])
        title_axis.set_axis_off()
        title_axis.text(0.5, 0.5, RT60_HISTOGRAM_TITLE,
                        transform=title_axis.transAxes, ha="center", va="center",
                        fontsize=FONT_SIZE_METRIC_TITLE, fontweight="bold")
        for model_index, model in enumerate(models):
            axis = fig.add_subplot(grid[histogram_row, model_index],
                                   sharex=axes[0][model_index],
                                   sharey=histogram_axes[0] if histogram_axes else None)
            histogram_axes.append(axis)
            axis.set_facecolor("none")
            axis.axvspan(x_lower, x_upper,
                         color=model_background(model, model_index),
                         alpha=MODEL_BACKGROUND_ALPHA, linewidth=0, zorder=0)
            axis.set_ylim(0, max(1.0, histogram_peak * 1.10))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
            axis.xaxis.set_major_locator(MultipleLocator(x_tick_step))
            axis.tick_params(labelsize=FONT_SIZE_TICK_LABELS, length=TICK_LENGTH,
                             color="#666666", labelleft=model_index == 0,
                             left=model_index == 0)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            for spine in ("bottom", "left"):
                axis.spines[spine].set_color("#777777")
            axis.spines["left"].set_visible(model_index == 0)
            histogram = histograms.get(model)
            if histogram is not None:
                draw_rt60_histogram(axis, histogram)

    fig.supxlabel(
        r"Reverberation time RT$_{60}$ (s)",
        fontsize=FONT_SIZE_X_AXIS_LABEL,
        y=-0.4 / figure_height,
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
    parser.add_argument("--rt60-distribution-csv", type=Path, default=RT60_DISTRIBUTION_CSV,
                        help="Override training folders with one shared CSV (source RTs with receiver IDs, or precomputed receiver-mean bin counts).")
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
        rt60_distribution_csv=args.rt60_distribution_csv,
    )
    print(f"Saved one-page RT60 metrics plot: {destination}")


if __name__ == "__main__":
    main()
