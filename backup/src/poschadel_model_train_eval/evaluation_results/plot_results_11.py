import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import MaxNLocator


# Never draw automatic outlines around scatter markers.
plt.rcParams["scatter.edgecolors"] = "none"
plt.rcParams["patch.force_edgecolor"] = False
plt.rcParams["font.size"] = 15


# =====================================================
# CONFIG: the metrics CSV is the only input used by this script
# =====================================================

CSV_PATH = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/significance_test/model_metrics.csv"

# Use the exact names in the CSV's "model" column. Their order here is their
# left-to-right order in the figure. Set this to None to plot every CSV model.

PLOTTED_MODELS = [
    "MUSIC",
    "shoeBox_SH_Direct",
    "shoeBox_em32_Direct",
    "shoeBox_SH_Direct-Late",
    "shoeBox_em32_Direct-Late",
    "shoeBox_SH_ISMOrder1",
    "shoeBox_em32_ISMOrder1",
    "shoeBox_SH_ISMOrder1-Late",
    "shoeBox_em32_ISMOrder1-Late",
    "shoeBox_SH_ISMOrder2",
    "shoeBox_em32_ISMOrder2",
    "shoeBox_SH_ISMOrder2-Late",
    "shoeBox_em32_ISMOrder2-Late",
    "shoeBox_SH_ISMOrder12",
    "shoeBox_em32_ISMOrder12",
    "shoeBox_SH_ISMOrder12-Late",
    "shoeBox_em32_ISMOrder12-Late",
    #"shoeBox_rand-ism_rand-disp-025_SH_ISMOrder12",
    "shoeBox_rand-ism_rand-disp-1_SH_ISMOrder12",
    "shoeBox_rand-ism_rand-disp-1_em32_ISMOrder12",
    #"shoeBox_rand-ism_rand-disp-5_SH_ISMOrder12",
    "shoeBox_rand-ism_rand-disp-10_SH_ISMOrder12",
    #"shoeBox_rand-ism_rand-disp-025_em32_ISMOrder12",
    #"shoeBox_rand-ism_rand-disp-5_em32_ISMOrder12",
    "shoeBox_rand-ism_rand-disp-10_em32_ISMOrder12",
]
MODEL_COLUMN_COLORS = [
    "#d0d0d0",  # MUSIC

    "#006b6b",
    "#3bc2b9",
    "#62938f",
    "#a1d6d1",

    # Blue ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â dark ÃƒÂ¢Ã¢â‚¬ Ã¢â‚¬â„¢ light
    "#1A1D8A",
    "#4146E8",
    "#8183C4",
    "#D7D8FE",

    "#8a6a00",
    "#e6b710",
    "#a28e4b",
    "#dfcd88",

    "#951010",
    "#ec4e4e",
    "#a96868",
    "#e7a5a5", 

    # Green ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â dark ÃƒÂ¢Ã¢â‚¬ Ã¢â‚¬â„¢ light
    "#01940D",
    "#25DA37",
    "#7BB381",
    "#CCF2D1",

    # Purple ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â dark ÃƒÂ¢Ã¢â‚¬ Ã¢â‚¬â„¢ light
    "#5B0AA3",
    "#9654D8",
    "#C29AE8",
    "#E6D4F6",
]
'''

PLOTTED_MODELS = [
    "shoeBox_SH_ISMOrder12",
    "non-shoeBox_disp-corners-05_SH_ISMOrder12",
    "non-shoeBox_disp-corners-1_SH_ISMOrder12",
    "non-shoeBox_disp-corners-3_SH_ISMOrder12",
    "shoeBox_rand-ism_rand-disp-025_SH_ISMOrder12",
    "shoeBox_rand-ism_rand-disp-1_SH_ISMOrder12",
    "shoeBox_rand-ism_rand-disp-5_SH_ISMOrder12",
    "shoeBox_rand-ism_rand-disp-10_SH_ISMOrder12",
]

MODEL_COLUMN_COLORS = [
    "#951010",  # Red

    # Blue Ã¢â‚¬â€ dark Ã¢â€ â€™ light
    "#0108C8",
    "#5C61E0",
    "#A9ABF6",

    # Green Ã¢â‚¬â€ dark Ã¢â€ â€™ light
    "#00B00F",
    "#54D861",
    "#8FDC97",
    "#CCF2D1",
]
'''


# Preserve the original SELD plots. Add "CM" to include the confusion matrices.
# Set to None to include every CSV metric and the confusion matrices.
PLOTTED_METRICS = ["LE", "count_MAE", "ESDL"]

# Draw a per-model mean across these three real-world datasets.
MEAN_DATASETS = ("arni-dataset", "motus-dataset", "ivan-dataset")

# Choose which of the plotted metrics also show the MUSIC model. Set to None
# to show MUSIC on every metric, just like all other models.
MUSIC_PLOTTED_METRICS = ["LE"]

# Set to True to also show the MUSIC model's confusion matrices.
PLOT_MUSIC_CONFUSION_MATRICES = False

OUTPUT_PATH = Path("plot_results_LE_MAE_ESSLD.png")

# Plot only the overall results, without source-count-specific series.
SOURCES = ["all"]

# Apply this order to metric groups, the dataset legend, and confusion rows.
# Any additional CSV datasets follow afterward in their original CSV order.
DATASET_ORDER = [
    "simulated-dataset",
    "arni-dataset",
    "motus-dataset",
    "ivan-dataset",
]

# Known dataset colors. Unknown labels receive stable tab20 colors.
KNOWN_DATASET_COLORS = {
    "checkpoint_dataset": np.array([100, 100, 100]) / 255.0,
    "simulated-dataset": np.array([150, 150, 150]) / 255.0,
    "arni-dataset": np.array([44, 160, 44]) / 255.0,
    #"arni-dataset": np.array([31, 119, 180]) / 255.0,
    "ivan-dataset": np.array([255, 127, 14]) / 255.0,
    #"motus-dataset": np.array([148, 103, 189]) / 255.0,
    "motus-dataset": np.array([31, 119, 180]) / 255.0,
}

# Define the legend label for each dataset independently of its CSV label.
# Datasets missing from this mapping keep their original CSV label.
DATASET_LEGEND_LABELS = {
    "checkpoint_dataset": "Simulation dataset",
    "simulated-dataset": "Simulation dataset",
    "arni-dataset": "Arni dataset",
    "ivan-dataset": "AGrid dataset",
    "motus-dataset": "MOTUS dataset",
}

# Define the vertical label beside each confusion-matrix dataset row.
# Missing entries use DATASET_LEGEND_LABELS, then the original CSV label.
CONFUSION_DATASET_LABELS = {
    "checkpoint_dataset": "Simulation",
    "simulated-dataset": "Simulation",
    "arni-dataset": "Arni",
    "ivan-dataset": "AGrid",
    "motus-dataset": "MOTUS",
}


# Confusion-matrix layout and text settings.
CONFUSION_CLASSES = ["0", "1", "2", "3"]
CM_TEXT_DECIMALS = 0
CM_MODEL_COLUMN_ALPHA = 0.15
CM_CELL_TEXT_SIZE = 20.0
CM_TICK_SIZE = 22
CM_DATASET_LABEL_SIZE = 22
CM_DATASET_LABEL_PAD = 10.0  # Gap to the left of the "True" label, in points.
CM_MODEL_LABEL_SIZE = 28
METRIC_TITLE_TEXT_SIZE = 48
AXIS_LABEL_TEXT_SIZE = 44
AXIS_TICK_TEXT_SIZE = 44
LEGEND_TEXT_SIZE = 48
CM_HEADING_TEXT_SIZE = 48
MODEL_GROUP_HEADING_TEXT_SIZE = 48
MODEL_SETTING_TEXT_SIZE = 48
# Font size for the row labels "Late Reverb", "Array Sim.", and
# "Randomization", independently of the setting values in the table.
MODEL_SETTING_LABEL_TEXT_SIZE = 44
# Center-to-center spacing between the model heading and each setting row,
# in figure-height units (approximately inches). Increase it to spread the
# heading and rows apart; decrease it to make the settings table more compact.
MODEL_SETTING_VERTICAL_SPACING = 1
MODEL_GROUP_SEPARATOR_COLOR = "#666666"
MODEL_GROUP_SEPARATOR_WIDTH = 1.5

# Vertical section spacing in figure-height units (approximately inches).
# Increase an individual value to add space; set it to 0 to remove that gap.
SPACE_AFTER_LEGEND = 1.1
SPACE_AFTER_MODEL_HEADING = 1.45
SPACE_BETWEEN_SELD_METRIC_PLOTS = 1.45
SPACE_AFTER_SELD_METRIC_PLOTS = 0.2
SPACE_AFTER_CONFUSION_HEADING = 0.15
SPACE_BETWEEN_CONFUSION_MATRIX_ROWS = 0.2

MODEL_SETTING_ROWS = ("Late Reverb", "Array Sim.", "Randomization")
DISPLACEMENT_METERS_BY_TOKEN = {
    "025": "0.25",
    "05": "0.5",
    "1": "1",
    "3": "3",
    "5": "5",
    "10": "10",
}

# Extra vertical extension of confusion-matrix columns.
# The labels and heading are shifted together with this extension.
CM_COLUMN_EXTEND_TOP = 0.005
CM_COLUMN_EXTEND_BOTTOM = 0.017

METRIC_LABELS = {
    "LR":   r"$\mathrm{LR}$",
    "LE":   r"$\mathrm{LE}$",
    "ER15": r"$\mathrm{ER}_{15}$",
    "F15":  r"$\mathrm{F}_{15}$",
    "ESDL": r"$\mathrm{E}_{\mathrm{SSLD}}$",
    "F":   r"$\mathrm{F}_{1}$",
    "count_MAE": r"SC-MAE",
}

# Full names are used for plot titles; METRIC_LABELS remains the axis shorthand.
METRIC_TITLES = {
    "LR": "Localization Recall (LR) ↑",
    "LE": "Localization Error (LE) ↓",
    "ER15": r"Location-Sensitive Error Rate ($\mathbf{ER}_{\mathbf{15}}$) ↓",
    "F15": r"Location-Sensitive F-Score ($\mathbf{F}_{\mathbf{15}}$) ↑",
    "ESDL": r"$\mathbf{E}_{\mathbf{SSLD}}\mathbf{-Score}$ ↓",
    "F": r"$\mathbf{F}_{\mathbf{1}}$-Score ↑",
    "count_MAE": r"Source Count Mean Absolute Error (SC-MAE) ↓",
}

# =====================================================
# HELPERS
# =====================================================
def compact_centered_offsets(n, total_width, min_slots=3):
    """Return n evenly spaced offsets centered around zero."""
    if n <= 0:
        return np.array([])
    if n == 1:
        return np.array([0.0])

    step = total_width / max(n, min_slots)
    return (np.arange(n) - (n - 1) / 2) * step


def lighten_color(color, amount):
    """Mix a matplotlib color with white by the requested amount."""
    rgb = np.asarray(plt.matplotlib.colors.to_rgb(color), dtype=float)
    return rgb + (1.0 - rgb) * amount


def model_pair_color(model_index):
    """Return the configured color for each model column."""
    if model_index < len(MODEL_COLUMN_COLORS):
        return MODEL_COLUMN_COLORS[model_index]

    # Fallback for any additional columns beyond the first eight.
    cmap = plt.get_cmap("tab20")
    return np.asarray(cmap(model_index % 20)[:3])


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
    """Return the three values shown below a model-group heading."""
    stem = str(model_name).strip()
    if is_music_model(stem):
        return ("n/a", "n/a", "n/a")

    if stem.startswith("eval_"):
        stem = stem[5:]
    stem = re.sub(r"\(\d+\)$", "", stem)

    microphone_match = re.search(r"_(SH|em32)_(?:Direct|ISMOrder)", stem)
    if microphone_match is None:
        raise ValueError(
            f"Cannot determine microphone simulation from model {model_name!r}; "
            "expected '_SH_' or '_em32_' before Direct/ISMOrder."
        )
    microphone_simulation = microphone_match.group(1)
    if microphone_simulation == "em32":
        microphone_simulation = "EM"

    late_reverb = "\u2714" if stem.endswith("-Late") else "\u00d7"

    random_match = re.search(r"_rand-disp-([^_]+)_", stem)
    random_displacement = (
        displacement_from_token(random_match.group(1), model_name)
        if random_match is not None
        else "\u00d7"
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


def dataset_colors(tests):
    """Keep predefined dataset colors and assign stable fallback colors."""
    cmap = plt.get_cmap("tab20")
    colors = {}
    for index, test in enumerate(tests):
        colors[test] = KNOWN_DATASET_COLORS.get(
            test, np.asarray(cmap(index % 20)[:3])
        )
    return colors


def normalized_metric(metric, value):
    """Normalize LE to 0..1; other CSV metrics retain their original units."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    value = float(value)
    return value / 45.0 if metric == "LE" else value


def read_metrics_csv(csv_path):
    """
    Read the single CSV supplying values, confidence intervals, and matrices.

    Keep first-seen model, dataset, and metric order for automatic selection.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Model-metrics CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "dataset", "model", "gt_source_count", "metric", "full_sample",
            "ci_low", "ci_high",
        }
        required.update(
            f"confusion_gt_{row}_pred_{column}_percent"
            for row in range(len(CONFUSION_CLASSES))
            for column in range(len(CONFUSION_CLASSES))
        )
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Model-metrics CSV is missing columns: "
                + ", ".join(sorted(missing))
            )

        rows = []
        for row_number, raw_row in enumerate(reader, start=2):
            row = {
                key: value.strip() if isinstance(value, str) else value
                for key, value in raw_row.items()
            }
            row["_row_number"] = row_number
            rows.append(row)

    if not rows:
        raise ValueError(f"The model-metrics CSV has no data rows: {csv_path}")
    return rows


def select_csv_values(rows, column, requested, description):
    """Return requested CSV values in user-defined or first-seen order."""
    available = list(dict.fromkeys(row[column] for row in rows if row[column]))
    if requested is None:
        return available

    selected = list(requested)
    if not selected:
        raise ValueError(f"No {description} were selected.")

    duplicated = [value for value in dict.fromkeys(selected) if selected.count(value) > 1]
    if duplicated:
        raise ValueError(
            f"Duplicate selected {description}: " + ", ".join(duplicated)
        )

    missing = [value for value in selected if value not in available]
    if missing:
        raise ValueError(
            f"Selected {description} not found in the CSV: "
            + ", ".join(missing)
            + f"\nAvailable {description}: "
            + ", ".join(available)
        )
    return selected


def order_datasets(datasets):
    """Place known datasets in the configured display order."""
    priorities = {
        dataset: index
        for index, dataset in enumerate(DATASET_ORDER)
    }
    # Older exports use a different identifier for the simulation dataset.
    if "simulated-dataset" in priorities:
        priorities.setdefault("checkpoint_dataset", priorities["simulated-dataset"])
    return sorted(datasets, key=lambda dataset: priorities.get(dataset, len(priorities)))


def load_csv_plot_data(rows, models, tests, metrics):
    """Build metric, confidence-interval, and confusion-matrix plot data."""
    all_metric_data = {
        model: {
            test: {
                source: {metric: None for metric in metrics}
                for source in SOURCES
            }
            for test in tests
        }
        for model in models
    }
    all_confusion_data = {
        model: {test: None for test in tests}
        for model in models
    }
    intervals = {}
    selected_models = set(models)
    selected_metrics = set(metrics)
    selected_sources = set(SOURCES)

    for row in rows:
        model = row["model"]
        dataset = row["dataset"]
        metric = row["metric"]
        source = row["gt_source_count"]
        if (
            model not in selected_models
            or dataset not in tests
            or metric not in selected_metrics
            or source not in selected_sources
        ):
            continue

        row_number = row["_row_number"]
        try:
            value = normalized_metric(metric, row["full_sample"])
            low = normalized_metric(metric, row["ci_low"])
            high = normalized_metric(metric, row["ci_high"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid metric value or confidence interval in CSV row {row_number}."
            ) from error

        key = (dataset, model, source, metric)
        if key in intervals:
            raise ValueError(
                f"Duplicate dataset/model/source/metric combination in CSV row "
                f"{row_number}: {key!r}."
            )
        all_metric_data[model][dataset][source][metric] = value
        intervals[key] = (low, high) if low is not None and high is not None else None

        if source == "all" and all_confusion_data[model][dataset] is None:
            try:
                matrix = np.asarray([
                    [
                        float(row[f"confusion_gt_{true}_pred_{predicted}_percent"])
                        for predicted in range(len(CONFUSION_CLASSES))
                    ]
                    for true in range(len(CONFUSION_CLASSES))
                ], dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid confusion-matrix percentage in CSV row {row_number}."
                ) from error
            all_confusion_data[model][dataset] = matrix

    return all_metric_data, all_confusion_data, intervals


def has_value(value):
    """Zero is a valid metric value; only None means unavailable."""
    return value is not None


def source_is_plottable(dataset, source):
    """Restrict metric plots to overall results for every dataset."""
    return source == "all"


def metric_value_bounds(metric):
    """Return the permitted metric range in the script's normalized units."""
    if metric == "LE":
        return 0.0, normalized_metric(metric, 180.0)
    return 0.0, 1.0


def metric_value_is_plottable(metric, value):
    """Reject missing, non-finite, and physically out-of-range values."""
    if not has_value(value) or not np.isfinite(value):
        return False

    lower_bound, upper_bound = metric_value_bounds(metric)
    return lower_bound <= value <= upper_bound


def metric_value_is_present(value):
    """Retain finite values even when they lie outside the visible plot range."""
    return has_value(value) and np.isfinite(value)


def clipped_confidence_interval(metric, interval):
    """Keep confidence silhouettes inside the metric's permitted range."""
    if interval is None:
        return None

    low, high = interval
    if not np.isfinite(low) or not np.isfinite(high) or high < low:
        return None

    lower_bound, upper_bound = metric_value_bounds(metric)
    low = max(low, lower_bound)
    high = min(high, upper_bound)
    return (low, high) if low <= high else None


def adaptive_metric_limits(metric, models, tests, all_metric_data, intervals):
    """Fit eligible values and intervals while preserving fixed metric bounds."""
    lower_bound, upper_bound = metric_value_bounds(metric)
    visible_values = []

    for model in models:
        for test in tests:
            for source in SOURCES:
                if not source_is_plottable(test, source):
                    continue

                value = all_metric_data[model][test][source][metric]
                if not metric_value_is_plottable(metric, value):
                    continue

                visible_values.append(value)
                interval = clipped_confidence_interval(
                    metric, intervals.get((test, model, source, metric))
                )
                if interval is not None:
                    visible_values.extend(interval)

    if not visible_values:
        return lower_bound, upper_bound

    if metric in {"LR", "F", "F15"}:
        lower_limit = min(visible_values)
        if lower_limit >= upper_bound:
            lower_limit = max(lower_bound, upper_bound - 0.05)
        return lower_limit, upper_bound

    if metric in {"LE", "ER15", "ESDL"}:
        upper_limit = max(visible_values)
        if upper_limit <= lower_bound:
            upper_limit = min(upper_bound, lower_bound + 0.05)
        return lower_bound, upper_limit

    return lower_bound, upper_bound


def dataset_colormap(dataset_color):
    """Create a white-to-dataset-color map for one confusion matrix."""
    rgb = np.asarray(dataset_color, dtype=float)
    pale = lighten_color(rgb, 0.93)
    return LinearSegmentedColormap.from_list("dataset_cm", [pale, rgb])


def contrasting_text_color(rgba):
    """Choose black or white annotation text from cell luminance."""
    r, g, b = rgba[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.52 else "white"


def add_model_column_background(
    fig, axes_in_column, color, alpha, left_edge=None, right_edge=None,
    top_extend=0.0, bottom_extend=0.0
):
    """Shade a CM/model column, optionally extending beyond the axes."""
    positions = [ax.get_subplotspec().get_position(fig) for ax in axes_in_column]
    left = min(pos.x0 for pos in positions) if left_edge is None else left_edge
    right = max(pos.x1 for pos in positions) if right_edge is None else right_edge
    bottom = min(pos.y0 for pos in positions) - bottom_extend
    top = max(pos.y1 for pos in positions) + top_extend

    patch = Rectangle(
        (left, bottom),
        right - left,
        top - bottom,
        transform=fig.transFigure,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=-10,
        clip_on=False,
    )
    fig.add_artist(patch)


def draw_confidence_bell(ax, x, value, interval, color, half_width, source):
    """Draw an opaque density silhouette spanning the actual confidence interval."""
    facecolor = plt.matplotlib.colors.to_rgba(color, alpha=1.0)
    edgecolor = plt.matplotlib.colors.to_rgba(color, alpha=1.0)
    zorder = 4

    if interval is None or interval[1] <= interval[0]:
        ax.scatter(
            x,
            value,
            marker="D",
            s=180,
            facecolors=[facecolor],
            edgecolors=[edgecolor],
            linewidths=2,
            zorder=zorder,
            clip_on=True,
        )
        return

    ci_low, ci_high = interval
    y_values = np.linspace(ci_low, ci_high, 101)
    lower_scale = max(value - ci_low, np.finfo(float).eps) / 2.5
    upper_scale = max(ci_high - value, np.finfo(float).eps) / 2.5
    scales = np.where(y_values <= value, lower_scale, upper_scale)
    profile = np.exp(-0.5 * ((y_values - value) / scales) ** 2)
    profile[[0, -1]] = 0.0
    widths = half_width * profile

    ax.fill_betweenx(
        y_values,
        x - widths,
        x + widths,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0,
        zorder=zorder,
        clip_on=True,
    )

    ax.hlines(
        [ci_low, ci_high],
        x - half_width * 2.5,
        x + half_width * 2.5,
        colors=[edgecolor],
        linewidth=5,
        zorder=zorder + 0.1,
        clip_on=True,
    )


class SourceLegendMarker:
    """Container describing a colored overall confidence bell."""

    def __init__(self, color="gray"):
        self.color = color


class HandlerSourceLegendMarker(HandlerBase):
    """Draw the same bell silhouette and CI caps used by the metric plots."""

    def create_artists(
        self, legend, original_handle, xdescent, ydescent,
        width, height, fontsize, transform
    ):
        center_x = xdescent + width / 2
        center_y = ydescent + height / 2
        positions = np.linspace(-1.0, 1.0, 41)
        widths = width * 0.15 * np.exp(-3.2 * positions**2)
        widths[[0, -1]] = 0.0
        y_values = center_y + positions * height * 0.7
        vertices = np.vstack([
            np.column_stack([center_x - widths, y_values]),
            np.column_stack([center_x + widths[::-1], y_values[::-1]]),
        ])
        bell = Polygon(
            vertices,
            closed=True,
            facecolor=original_handle.color,
            edgecolor=original_handle.color,
            linewidth=0.0,
            alpha=1.0,
            transform=transform,
        )
        cap_half_width = width * 0.3
        ci_caps = Line2D(
            [
                center_x - cap_half_width,
                center_x + cap_half_width,
                np.nan,
                center_x - cap_half_width,
                center_x + cap_half_width,
            ],
            [y_values[0], y_values[0], np.nan, y_values[-1], y_values[-1]],
            color=original_handle.color,
            linewidth=5,
            alpha=1.0,
            transform=transform,
        )
        return [bell, ci_caps]


# =====================================================
# PLOT
# =====================================================
def plot(
    csv_path=CSV_PATH,
    plotted_models=PLOTTED_MODELS,
    plotted_metrics=PLOTTED_METRICS,
    output_path=OUTPUT_PATH,
    show=False,
    music_plotted_metrics=MUSIC_PLOTTED_METRICS,
    plot_music_confusion_matrices=PLOT_MUSIC_CONFUSION_MATRICES,
):
    rows = read_metrics_csv(csv_path)
    models = select_csv_values(rows, "model", plotted_models, "models")
    selected_model_names = set(models)
    selected_rows = [row for row in rows if row["model"] in selected_model_names]
    if plotted_metrics is None:
        csv_metrics = None
        plot_confusion_matrices = True
    else:
        requested_plot_items = list(plotted_metrics)
        plot_confusion_matrices = "CM" in requested_plot_items
        csv_metrics = [item for item in requested_plot_items if item != "CM"]
    metrics = select_csv_values(selected_rows, "metric", csv_metrics, "metrics")
    music_rows = [row for row in selected_rows if is_music_model(row["model"])]
    if music_rows and music_plotted_metrics is not None:
        selected_music_metrics = set(
            select_csv_values(
                music_rows, "metric", music_plotted_metrics, "MUSIC metrics"
            )
        )
        unavailable_music_metrics = selected_music_metrics.difference(metrics)
        if unavailable_music_metrics:
            raise ValueError(
                "Selected MUSIC metrics are not among the plotted metrics: "
                + ", ".join(sorted(unavailable_music_metrics))
            )
    else:
        selected_music_metrics = set(metrics)

    tests = order_datasets(
        select_csv_values(selected_rows, "dataset", None, "datasets")
    )
    colors = dataset_colors(tests)
    all_metric_data, all_confusion_data, confidence_intervals = load_csv_plot_data(
        selected_rows, models, tests, metrics
    )
    if not plot_confusion_matrices:
        for model in models:
            all_confusion_data[model] = {test: None for test in tests}
    if not plot_music_confusion_matrices:
        for model in models:
            if is_music_model(model):
                all_confusion_data[model] = {test: None for test in tests}

    n_models = len(models)
    n_metrics = len(metrics)
    n_tests = len(tests)

    if n_models == 0:
        raise ValueError("No models were supplied.")
    if n_tests == 0:
        raise ValueError("No datasets were found for the selected CSV models.")

    # Give each requested section its own row so their vertical order stays
    # stable regardless of the number of models, datasets, or metrics.
    legend_columns = max(1, math.ceil((n_models + n_tests) / 4))
    legend_rows = math.ceil((n_models + n_tests) / legend_columns)
    legend_height = 1
    if MODEL_SETTING_VERTICAL_SPACING <= 0:
        raise ValueError("MODEL_SETTING_VERTICAL_SPACING must be greater than 0.")
    parameter_table_height = (
        MODEL_SETTING_VERTICAL_SPACING * (len(MODEL_SETTING_ROWS) + 1)
    )
    metric_height = 4.8
    cm_heading_height = 0.85 if plot_confusion_matrices else 0.0
    cm_height = 2.3 if plot_confusion_matrices else 0.0
    fig_width = max(26.0, 2.35 * n_models + 5.0)

    # Add dedicated spacer rows so each section's gap can be adjusted without
    # changing the spacing around the other sections.
    height_ratios = []

    def add_grid_row(height):
        row_index = len(height_ratios)
        height_ratios.append(height)
        return row_index

    def add_vertical_space(height):
        if height < 0:
            raise ValueError("Vertical spacing values must be non-negative.")
        if height > 0:
            add_grid_row(height)

    legend_row = add_grid_row(legend_height)
    add_vertical_space(SPACE_AFTER_LEGEND)

    parameter_table_row = add_grid_row(parameter_table_height)
    add_vertical_space(SPACE_AFTER_MODEL_HEADING)

    metric_rows = []
    for metric_index in range(n_metrics):
        if metric_index > 0:
            add_vertical_space(SPACE_BETWEEN_SELD_METRIC_PLOTS)
        metric_rows.append(add_grid_row(metric_height))
    if metric_rows:
        add_vertical_space(SPACE_AFTER_SELD_METRIC_PLOTS)

    cm_heading_row = add_grid_row(cm_heading_height)
    if plot_confusion_matrices:
        add_vertical_space(SPACE_AFTER_CONFUSION_HEADING)

    cm_rows = []
    for test_index in range(n_tests):
        if plot_confusion_matrices and test_index > 0:
            add_vertical_space(SPACE_BETWEEN_CONFUSION_MATRIX_ROWS)
        cm_rows.append(add_grid_row(cm_height))

    fig_height = sum(height_ratios) + 1.0
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=len(height_ratios),
        ncols=n_models,
        height_ratios=height_ratios,
        left=0.075,
        right=0.99,
        bottom=0.035,
        top=0.995,
        hspace=0.0,
        wspace=-0.04,
    )

    legend_ax = fig.add_subplot(grid[legend_row, :])
    legend_ax.set_axis_off()
    parameter_table_ax = fig.add_subplot(grid[parameter_table_row, :])
    parameter_table_ax.set_axis_off()

    # Metric axes span all model columns.
    metric_axes = [fig.add_subplot(grid[row, :]) for row in metric_rows]
    cm_heading_ax = fig.add_subplot(grid[cm_heading_row, :])
    cm_heading_ax.set_axis_off()

    # One compact confusion matrix for every dataset-model combination.
    cm_axes = np.empty((n_tests, n_models), dtype=object)
    for test_index in range(n_tests):
        for model_index in range(n_models):
            cm_axes[test_index, model_index] = fig.add_subplot(
                grid[cm_rows[test_index], model_index]
            )

    # Align the grouped headings and setting values with the model columns used
    # by the metric plots and confusion matrices.
    parameter_table_ax.set_xlim(-0.5, n_models - 0.5)
    parameter_table_ax.set_ylim(parameter_table_height, 0.0)

    heading_y = 0.5 * MODEL_SETTING_VERTICAL_SPACING
    setting_row_y = [
        (field_index + 1.5) * MODEL_SETTING_VERTICAL_SPACING
        for field_index in range(len(MODEL_SETTING_ROWS))
    ]

    model_groups = consecutive_model_groups(models)
    for group_start, group_end, heading in model_groups:
        parameter_table_ax.text(
            0.5 * (group_start + group_end),
            heading_y,
            heading,
            ha="center",
            va="center",
            fontsize=MODEL_GROUP_HEADING_TEXT_SIZE,
            #fontweight="bold",
        )

    # Separate MUSIC, Direct Sound, and every consecutive ISM-order group.
    for group_start, _, _ in model_groups[1:]:
        parameter_table_ax.axvline(
            group_start - 0.5,
            color=MODEL_GROUP_SEPARATOR_COLOR,
            linewidth=MODEL_GROUP_SEPARATOR_WIDTH,
            zorder=2,
        )

    for field_index, field_name in enumerate(MODEL_SETTING_ROWS):
        parameter_table_ax.text(
            -0.58,
            setting_row_y[field_index],
            field_name,
            ha="right",
            va="center",
            fontsize=MODEL_SETTING_LABEL_TEXT_SIZE,
            clip_on=False,
        )

    for model_index, model in enumerate(models):
        parameter_table_ax.axvspan(
            model_index - 0.5,
            model_index + 0.5,
            ymin=0.0,
            ymax=len(MODEL_SETTING_ROWS) / (len(MODEL_SETTING_ROWS) + 1),
            color=model_pair_color(model_index),
            alpha=0.15,
            zorder=0,
        )
        for field_index, value in enumerate(model_heading_values(model)):
            parameter_table_ax.text(
                model_index,
                setting_row_y[field_index],
                value,
                ha="center",
                va="center",
                fontsize=MODEL_SETTING_TEXT_SIZE,
            )

    if plot_confusion_matrices:
        cm_heading_ax.text(
            0.5,
            0.38,
            "Source-count confusion matrices (%)",
            transform=cm_heading_ax.transAxes,
            ha="center",
            va="center",
            fontsize=CM_HEADING_TEXT_SIZE,
            fontweight="bold",
        )

    model_centers = np.arange(n_models)
    column_width = 1.0
    inner_width = column_width * 0.88

    # ---------------- Metric plots ----------------
    for metric_index, metric in enumerate(metrics):
        ax = metric_axes[metric_index]
        display_metric = METRIC_LABELS.get(metric, metric)
        metric_title = METRIC_TITLES.get(metric, metric)
        metric_models = [
            model
            for model in models
            if not is_music_model(model) or metric in selected_music_metrics
        ]
        metric_limits = adaptive_metric_limits(
            metric, metric_models, tests, all_metric_data, confidence_intervals
        )

        for model_index, model in enumerate(models):
            model_center = model_centers[model_index]
            model_color = model_pair_color(model_index)
            mean_points = []

            ax.axvspan(
                model_center - column_width / 2,
                model_center + column_width / 2,
                color=model_color,
                alpha=0.15,
                zorder=0,
            )

            if is_music_model(model) and metric not in selected_music_metrics:
                continue

            active_tests = [
                test
                for test in tests
                if not (is_music_model(model) and test == "simulated-dataset")
                and any(
                    source_is_plottable(test, source)
                    and metric_value_is_present(
                        all_metric_data[model][test][source][metric]
                    )
                    for source in SOURCES
                )
            ]
            test_offsets = compact_centered_offsets(
                len(active_tests), total_width=inner_width, min_slots=2
            )

            for test_index, test in enumerate(active_tests):
                active_sources = [
                    source
                    for source in SOURCES
                    if source_is_plottable(test, source)
                    and metric_value_is_present(
                        all_metric_data[model][test][source][metric]
                    )
                ]
                source_group_width = inner_width / max(len(active_tests), 1) * 0.62
                source_offsets = compact_centered_offsets(
                    len(active_sources),
                    total_width=source_group_width,
                    min_slots=1,
                )

                for source_index, source in enumerate(active_sources):
                    value = all_metric_data[model][test][source][metric]
                    x = (
                        model_center
                        + test_offsets[test_index]
                        + source_offsets[source_index]
                    )

                    interval = confidence_intervals.get(
                        (test, model, source, metric)
                    )
                    bell_half_width = max(
                        source_group_width / max(len(active_sources), 1) * 0.25,
                        0.02,
                    )
                    draw_confidence_bell(
                        ax,
                        x,
                        value,
                        interval,
                        colors[test],
                        bell_half_width,
                        source,
                    )

                    if test in MEAN_DATASETS and source == "all":
                        mean_points.append((x, value))
            mean_extension = 0.12
            if len(mean_points) == len(MEAN_DATASETS):
                mean_x, mean_values = zip(*mean_points)
                ax.hlines(
                    np.mean(mean_values),
                    min(mean_x) - mean_extension,
                    max(mean_x) + mean_extension,
                    colors="black",
                    linestyles="--",
                    linewidth=4.0,
                    zorder=10,
                )

        # Match the model-group separators shown in the settings table.
        for group_start, _, _ in model_groups[1:]:
            ax.axvline(
                group_start - 0.5,
                color=MODEL_GROUP_SEPARATOR_COLOR,
                linewidth=MODEL_GROUP_SEPARATOR_WIDTH,
                zorder=2,
            )

        ax.set_title(
            metric_title,
            fontsize=METRIC_TITLE_TEXT_SIZE,
            fontweight="bold",
            pad=10,
        )
        # Match the exact GridSpec column boundaries used by the confusion matrices.
        ax.set_xlim(-0.5, n_models - 0.5)
        ax.set_ylim(*metric_limits)
        ax.grid(True, axis="y", alpha=0.27, linewidth=0.7)
        ax.set_xticks(model_centers)

        # Model settings are shown in the parameter table above the metrics.
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

        if metric == "LE":
            ax.set_ylabel("LE (°)", fontsize=AXIS_LABEL_TEXT_SIZE)
            max_degrees = metric_limits[1] * 45.0
            degree_ticks = MaxNLocator(
                nbins=7, steps=[1, 2, 2.5, 5, 10]
            ).tick_values(0.0, max_degrees)
            degree_ticks = degree_ticks[
                (degree_ticks >= 0.0) & (degree_ticks <= max_degrees + 1e-10)
            ]
            ax.set_yticks(degree_ticks / 45)
            ax.set_yticklabels(
                [f"{degree:g}" for degree in degree_ticks],
                fontsize=AXIS_TICK_TEXT_SIZE,
            )
        else:
            ax.set_ylabel(display_metric, fontsize=AXIS_LABEL_TEXT_SIZE)
            ax.tick_params(axis="y", labelsize=AXIS_TICK_TEXT_SIZE)

        ax.spines[["top", "right"]].set_visible(False)

    fig.align_ylabels(metric_axes)

    # ---------------- Confusion matrices ----------------
    if not plot_confusion_matrices:
        for ax in cm_axes.flat:
            ax.set_visible(False)

    first_visible_model_by_test = {
        test: next(
            (
                model_index
                for model_index, model in enumerate(models)
                if all_confusion_data[model][test] is not None
            ),
            None,
        )
        for test in tests
    }
    last_visible_test_by_model = {
        model: next(
            (
                test_index
                for test_index in range(n_tests - 1, -1, -1)
                if all_confusion_data[model][tests[test_index]] is not None
            ),
            None,
        )
        for model in models
    }

    for model_index, model in enumerate(models):
        for test_index, test in enumerate(tests):
            ax = cm_axes[test_index, model_index]
            matrix = all_confusion_data[model][test]

            if matrix is None:
                ax.set_visible(False)
                continue

            matrix_size = matrix.shape[0]
            cmap = dataset_colormap(colors[test])
            ax.imshow(
                matrix,
                cmap=cmap,
                vmin=0,
                vmax=100,
                interpolation="nearest",
                aspect="equal",
                zorder=1,
            )

            for row in range(matrix_size):
                for col in range(matrix_size):
                    value = matrix[row, col]
                    label = f"{value:.{CM_TEXT_DECIMALS}f}"
                    ax.text(
                        col,
                        row,
                        label,
                        ha="center",
                        va="center",
                        fontsize=CM_CELL_TEXT_SIZE,
                        color="black",
                        fontweight="normal",
                        bbox=dict(
                            boxstyle="round,pad=0.1",
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.8,
                        ),
                        zorder=3,
                    )

            class_labels = (
                CONFUSION_CLASSES
                if matrix_size == len(CONFUSION_CLASSES)
                else [str(i) for i in range(matrix_size)]
            )
            ticks = np.arange(matrix_size)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            # X labels belong to the bottom visible matrix in each model column.
            if test_index == last_visible_test_by_model[model]:
                ax.set_xticklabels(class_labels, fontsize=CM_TICK_SIZE)
                ax.set_xlabel("Predicted", fontsize=CM_TICK_SIZE + 1.5, labelpad=3)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)

            # Y labels belong to the first visible matrix in each dataset row.
            if model_index == first_visible_model_by_test[test]:
                ax.set_yticklabels(class_labels, fontsize=CM_TICK_SIZE)
                ax.set_ylabel("True", fontsize=CM_TICK_SIZE + 1.5, labelpad=3)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)

            ax.set_xlim(-0.5, matrix_size - 0.5)
            ax.set_ylim(matrix_size - 0.5, -0.5)
            ax.tick_params(axis="both", which="both", pad=2.0, length=2.0)

            # Thin white cell separators improve readability without enlarging cells.
            ax.set_xticks(np.arange(-0.5, matrix_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, matrix_size, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.85)
            ax.tick_params(which="minor", bottom=False, left=False)

            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("black")

    # Position-based labels/backgrounds must be added after the axes exist.
    fig.canvas.draw()

    # Place each configurable dataset name vertically beside its "True" label.
    renderer = fig.canvas.get_renderer()
    label_padding = CM_DATASET_LABEL_PAD / (72.0 * fig.get_figwidth())
    for test_index, test in enumerate(tests):
        first_visible_model = first_visible_model_by_test[test]
        if first_visible_model is None:
            continue
        ax = cm_axes[test_index, first_visible_model]
        true_label_bounds = ax.yaxis.label.get_window_extent(renderer=renderer)
        true_label_bounds = true_label_bounds.transformed(fig.transFigure.inverted())
        dataset_label = CONFUSION_DATASET_LABELS.get(
            test, DATASET_LEGEND_LABELS.get(test, test)
        )
        fig.text(
            true_label_bounds.x0 - label_padding,
            (true_label_bounds.y0 + true_label_bounds.y1) / 2,
            dataset_label,
            rotation=90,
            ha="right",
            va="center",
            fontsize=CM_DATASET_LABEL_SIZE,
        )

    # Extend each shaded CM column to the midpoint of the neighboring gaps.
    # This creates soft colored separators that match the metric background bands.
    cm_column_positions = [
        cm_axes[0, model_index].get_subplotspec().get_position(fig)
        for model_index in range(n_models)
    ]
    cm_column_edges = [cm_column_positions[0].x0]
    cm_column_edges.extend(
        0.5 * (cm_column_positions[i].x1 + cm_column_positions[i + 1].x0)
        for i in range(n_models - 1)
    )
    cm_column_edges.append(cm_column_positions[-1].x1)

    for model_index, model in enumerate(models):
        # Shade only consecutive runs of real matrices; missing cells stay blank.
        visible_run = []
        for test_index, test in enumerate(tests):
            if all_confusion_data[model][test] is not None:
                visible_run.append(cm_axes[test_index, model_index])
                continue

            if visible_run:
                add_model_column_background(
                    fig,
                    visible_run,
                    model_pair_color(model_index),
                    CM_MODEL_COLUMN_ALPHA,
                    left_edge=cm_column_edges[model_index],
                    right_edge=cm_column_edges[model_index + 1],
                    top_extend=CM_COLUMN_EXTEND_TOP,
                    bottom_extend=CM_COLUMN_EXTEND_BOTTOM,
                )
                visible_run = []

        if visible_run:
            add_model_column_background(
                fig,
                visible_run,
                model_pair_color(model_index),
                CM_MODEL_COLUMN_ALPHA,
                left_edge=cm_column_edges[model_index],
                right_edge=cm_column_edges[model_index + 1],
                top_extend=CM_COLUMN_EXTEND_TOP,
                bottom_extend=CM_COLUMN_EXTEND_BOTTOM,
            )

    # CM column boundaries are formed by the adjacent shaded backgrounds above.

    # ---------------- Dataset legend only ----------------
    legend_handles = [
        SourceLegendMarker(color=colors[test])
        for test in tests
    ]

    legend_labels = [DATASET_LEGEND_LABELS.get(test, test) for test in tests]
    legend_handles.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=6.0)
    )
    legend_labels.append("Mean (Arni, MOTUS, AGrid)")

    legend_ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map={SourceLegendMarker: HandlerSourceLegendMarker()},
        ncol=len(legend_handles),  # all datasets and the mean side by side
        fontsize=LEGEND_TEXT_SIZE,
        loc="center",
        frameon=True,
        columnspacing=1.5,
        handletextpad=0.45,
        borderpad=0.55,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot()
