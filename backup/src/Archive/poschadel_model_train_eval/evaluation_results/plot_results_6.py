import csv
import json
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
# CONFIG: replace these placeholders with your JSON files
# =====================================================

FILE_PATHS = [
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_SH_Direct.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_Direct.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_SH_Direct-Late.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_Direct-Late.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_SH_ISMOrder2.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder2.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_SH_ISMOrder2-Late.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder2-Late.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_SH_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_SH_ISMOrder12-Late.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder12-Late.json"),
    # Add the remaining evaluation JSON files here.
]

'''
FILE_PATHS = [
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_SH_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_non-shoeBox_disp-corners-05_SH_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_non-shoeBox_disp-corners-1_SH_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_non-shoeBox_disp-corners-3_SH_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_rand-ism_rand-disp-025_SH_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_rand-ism_rand-disp-1_SH_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_rand-ism_rand-disp-5_SH_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_rand-ism_rand-disp-10_SH_ISMOrder12.json"),

    # Add the remaining evaluation JSON files here.
]
'''


OUTPUT_PATH = Path("plot_results_with_confusion_matrices.png")
CONFIDENCE_INTERVALS_CSV = Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/significance_test/model_metrics.csv")

# Plot only the overall results, without source-count-specific series.
SOURCES = ["all"]

# Known dataset colors. Unknown labels receive stable tab20 colors.
KNOWN_DATASET_COLORS = {
    "checkpoint_dataset": np.array([44, 160, 44]) / 255.0,
    "arni-dataset": np.array([31, 119, 180]) / 255.0,
    "ivan-dataset": np.array([255, 127, 14]) / 255.0,
    "motus-dataset": np.array([148, 103, 189]) / 255.0,
}

# Define the legend label for each dataset independently of its JSON label.
# Datasets missing from this mapping keep their original JSON label.
DATASET_LEGEND_LABELS = {
    "checkpoint_dataset": "Simulation dataset",
    "arni-dataset": "Arni dataset",
    "ivan-dataset": "Ivan dataset",
    "motus-dataset": "Motus dataset",
}

# Define the vertical label beside each confusion-matrix dataset row.
# Missing entries use DATASET_LEGEND_LABELS, then the original JSON label.
CONFUSION_DATASET_LABELS = {
    "checkpoint_dataset": "Simulation",
    "arni-dataset": "Arni",
    "ivan-dataset": "Ivan",
    "motus-dataset": "Motus",
}

# Model-column colors:
#   1     -> red
#   2-4   -> shades of blue
#   5-8   -> shades of green
MODEL_COLUMN_COLORS = [
    "#c00000",  # 1: red
    "#003f8c",  # 2: dark blue
    "#1f77b4",  # 3: medium blue
    "#8ec7f0",  # 4: light blue
    "#006400",  # 5: dark green
    "#008f3d",  # 6
    "#34a853",  # 7
    "#8fd19e",  # 8: light green
]

# Confusion-matrix layout and text settings.
CONFUSION_CLASSES = ["0", "1", "2", "3"]
CM_TEXT_DECIMALS = 0
CM_MODEL_COLUMN_ALPHA = 0.15
CM_CELL_TEXT_SIZE = 20.0
CM_TICK_SIZE = 20
CM_DATASET_LABEL_SIZE = 20
CM_DATASET_LABEL_PAD = 10.0  # Gap to the left of the "True" label, in points.
CM_MODEL_LABEL_SIZE = 24
METRIC_TITLE_TEXT_SIZE = 24
AXIS_LABEL_TEXT_SIZE = 20
AXIS_TICK_TEXT_SIZE = 20
LEGEND_TEXT_SIZE = 22
CM_HEADING_TEXT_SIZE = 24
MODEL_PARAMETER_DEFINITION_TEXT_SIZE = 22

# Vertical section spacing in figure-height units (approximately inches).
# Increase an individual value to add space; set it to 0 to remove that gap.
SPACE_AFTER_LEGEND = 0
SPACE_AFTER_MIC_DEFINITIONS = 0
SPACE_AFTER_MIC_ORDER_TABLE = 0.8
SPACE_BETWEEN_SELD_METRIC_PLOTS = 0.8
SPACE_AFTER_SELD_METRIC_PLOTS = 0.0
SPACE_AFTER_CONFUSION_HEADING = 0.1
SPACE_BETWEEN_CONFUSION_MATRIX_ROWS = 0.2

MODEL_PARAMETER_HEADERS = ("Mic", "Order", "L", "Corn", "Rand")
MODEL_PARAMETER_DEFINITIONS = (
    r"$\mathbf{Mic}$ = Microphone Simulation | "
    r"$\mathbf{Order}$ = ISM Order | "
    r"$\mathbf{L}$ = Diffuse Late Reverb | "
    r"$\mathbf{Corn}$ = Room Corner Displacement Range (m) | "
    r"$\mathbf{Rand}$ = Random Image-Source Displacement Range (m)"
)
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
CM_COLUMN_EXTEND_TOP = 0.0
CM_COLUMN_EXTEND_BOTTOM = 0.015

METRIC_LABELS = {
    "LR":   r"$\mathbf{LR}$",
    "LE":   r"$\mathbf{LE}$",
    "ER15": r"$\mathbf{ER}_{\mathbf{15}}$",
    "F15":  r"$\mathbf{F}_{\mathbf{15}}$",
    "ESDL": r"$\mathbf{SELD}$",
}

# Full names are used for plot titles; METRIC_LABELS remains the axis shorthand.
METRIC_TITLES = {
    "LR": "Localization Recall (LR)",
    "LE": "Localization Error (LE)",
    "ER15": r"Location-Sensitive Error Rate ($\mathbf{ER}_{\mathbf{15}}$)",
    "F15": r"Location-Sensitive F-Score ($\mathbf{F}_{\mathbf{15}}$)",
    "ESDL": "SELD-Score",
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


def model_name_from_path(file_path):
    """Use the JSON filename as model name and remove a leading 'eval_'."""
    name = file_path.stem
    return name[5:] if name.startswith("eval_") else name


def build_file_map(file_paths):
    """Use JSON filename stems as model labels and suffix duplicates."""
    files = {}
    name_counts = {}

    for file_path in map(Path, file_paths):
        base_name = model_name_from_path(file_path)
        name_counts[base_name] = name_counts.get(base_name, 0) + 1
        occurrence = name_counts[base_name]
        name = base_name if occurrence == 1 else f"{base_name}_{occurrence}"
        files[name] = file_path

    return files


def validate_files(files):
    """Fail early with a readable list of missing JSON paths."""
    missing = [f"{name}: {path}" for name, path in files.items() if not path.is_file()]
    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Replace the path placeholders in FILE_PATHS. Missing files:\n  - " + details
        )


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


def displacement_from_token(token, file_path):
    """Translate a filename displacement token to its value in meters."""
    try:
        return DISPLACEMENT_METERS_BY_TOKEN[token]
    except KeyError as error:
        raise ValueError(
            f"Unknown displacement token {token!r} in {file_path.name!r}. "
            "Add its value to DISPLACEMENT_METERS_BY_TOKEN."
        ) from error


def model_parameter_values(file_path):
    """Extract the same five model-label fields as the significance script."""
    file_path = Path(file_path)
    stem = file_path.stem
    if stem.startswith("eval_"):
        stem = stem[5:]
    stem = re.sub(r"\(\d+\)$", "", stem)

    microphone_match = re.search(r"_(SH|em32)_(?:Direct|ISMOrder)", stem)
    if microphone_match is None:
        raise ValueError(
            f"Cannot determine microphone simulation from {file_path.name!r}; "
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
            f"Cannot determine ISM order from {file_path.name!r}; expected "
            "'_Direct', '_ISMOrder2', or '_ISMOrder12', optionally '-Late'."
        )

    diffuse_late_reverb = "y" if stem.endswith("-Late") else "n"

    corner_match = re.search(r"non-shoeBox_disp-corners-([^_]+)_", stem)
    corner_displacement = (
        displacement_from_token(corner_match.group(1), file_path)
        if corner_match is not None
        else "0"
    )

    random_match = re.search(r"_rand-disp-([^_]+)_", stem)
    random_displacement = (
        displacement_from_token(random_match.group(1), file_path)
        if random_match is not None
        else "0"
    )

    return (
        microphone_simulation,
        ism_order,
        diffuse_late_reverb,
        corner_displacement,
        random_displacement,
    )


def read_json(file_path):
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_metrics(raw_models):
    """Take metric names and order directly from the first JSON result block."""
    for model_json in raw_models.values():
        for dataset in model_json.get("datasets", {}).values():
            overall = dataset.get("results", {}).get("overall", {})
            if overall:
                return list(overall.keys())
    raise ValueError("No metrics found under datasets -> results -> overall.")


def discover_tests(raw_models):
    """Collect dataset labels in first-seen JSON order."""
    tests = []
    for model_json in raw_models.values():
        for dataset in model_json.get("datasets", {}).values():
            label = dataset.get("dataset_label")
            if label and label not in tests:
                tests.append(label)
    return tests


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
    """Normalize LE to 0..1; other JSON metrics are expected in 0..1."""
    if value is None:
        return None
    value = float(value)
    return value / 45.0 if metric == "LE" else value


def load_confidence_intervals(csv_path):
    """
    Load confidence intervals keyed by (dataset, model, source, metric).

    Only overall CSV rows are used. LE intervals are normalized to 0..1 in
    exactly the same way as the plotted LE values.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Confidence-interval CSV not found: {csv_path}")

    source_keys = {"all": "all"}
    intervals = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "dataset", "model", "gt_source_count", "metric", "ci_low", "ci_high"
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Confidence-interval CSV is missing columns: "
                + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            source = source_keys.get(row["gt_source_count"].strip())
            if source is None:
                continue

            metric = row["metric"].strip()
            try:
                low = normalized_metric(metric, row["ci_low"])
                high = normalized_metric(metric, row["ci_high"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid confidence interval in CSV row {row_number}."
                ) from exc

            key = (
                row["dataset"].strip(),
                row["model"].strip(),
                source,
                metric,
            )
            intervals[key] = (low, high)

    return intervals


def load_model_data(model_json, tests, metrics):
    """
    Convert one evaluation JSON into:
        metric_data[test][source][metric] = scalar or None
        confusion_data[test] = 4x4 percentage array or None
    """
    metric_data = {
        test: {source: {metric: None for metric in metrics} for source in SOURCES}
        for test in tests
    }
    confusion_data = {test: None for test in tests}

    for dataset in model_json.get("datasets", {}).values():
        test = dataset.get("dataset_label")
        if test not in metric_data:
            continue

        results = dataset.get("results", {})
        overall = results.get("overall", {})
        for metric in metrics:
            metric_data[test]["all"][metric] = normalized_metric(
                metric, overall.get(metric)
            )

        matrix = dataset.get("source_count_confusion_matrix_percent")
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(
                    f"Confusion matrix for dataset '{test}' must be square; "
                    f"received shape {matrix.shape}."
                )
            confusion_data[test] = matrix

    return metric_data, confusion_data


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

    if metric in {"LR", "F15"}:
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
    file_paths=FILE_PATHS,
    output_path=OUTPUT_PATH,
    confidence_intervals_csv=CONFIDENCE_INTERVALS_CSV,
    show=True,
):
    files = build_file_map(file_paths)
    validate_files(files)

    raw_models = {name: read_json(path) for name, path in files.items()}
    models = list(files.keys())
    metrics = discover_metrics(raw_models)
    tests = discover_tests(raw_models)
    colors = dataset_colors(tests)
    confidence_intervals = load_confidence_intervals(confidence_intervals_csv)

    all_metric_data = {}
    all_confusion_data = {}
    for model in models:
        metric_data, confusion_data = load_model_data(
            raw_models[model], tests, metrics
        )
        all_metric_data[model] = metric_data
        all_confusion_data[model] = confusion_data

    n_models = len(models)
    n_metrics = len(metrics)
    n_tests = len(tests)

    if n_models == 0:
        raise ValueError("No models were supplied.")
    if n_tests == 0:
        raise ValueError("No datasets were found in the supplied JSON files.")

    # Give each requested section its own row so their vertical order stays
    # stable regardless of the number of models, datasets, or metrics.
    legend_columns = max(1, math.ceil((n_models + n_tests) / 4))
    legend_rows = math.ceil((n_models + n_tests) / legend_columns)
    legend_height = 1
    definitions_height = 0.85
    parameter_table_height = 0.47 * len(MODEL_PARAMETER_HEADERS) + 0.35
    metric_height = 4.8
    cm_heading_height = 0.85
    cm_height = 2.3
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

    definitions_row = add_grid_row(definitions_height)
    add_vertical_space(SPACE_AFTER_MIC_DEFINITIONS)

    parameter_table_row = add_grid_row(parameter_table_height)
    add_vertical_space(SPACE_AFTER_MIC_ORDER_TABLE)

    metric_rows = []
    for metric_index in range(n_metrics):
        if metric_index > 0:
            add_vertical_space(SPACE_BETWEEN_SELD_METRIC_PLOTS)
        metric_rows.append(add_grid_row(metric_height))
    if metric_rows:
        add_vertical_space(SPACE_AFTER_SELD_METRIC_PLOTS)

    cm_heading_row = add_grid_row(cm_heading_height)
    add_vertical_space(SPACE_AFTER_CONFUSION_HEADING)

    cm_rows = []
    for test_index in range(n_tests):
        if test_index > 0:
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
    definitions_ax = fig.add_subplot(grid[definitions_row, :])
    definitions_ax.set_axis_off()
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

    # Keep the long parameter descriptions readable on two centered lines.
    parameter_definitions = MODEL_PARAMETER_DEFINITIONS.split(" | ")
    definitions_ax.text(
        0.5,
        0.5,
        MODEL_PARAMETER_DEFINITIONS,
        transform=definitions_ax.transAxes,
        ha="center",
        va="center",
        fontsize=MODEL_PARAMETER_DEFINITION_TEXT_SIZE,
    )

    # Align the parameter values with the model columns used by the metric
    # plots and confusion matrices; draw field names only in the left margin.
    parameter_table_ax.set_xlim(-0.5, n_models - 0.5)
    parameter_table_ax.set_ylim(len(MODEL_PARAMETER_HEADERS) - 0.5, -0.5)
    for field_index, field_name in enumerate(MODEL_PARAMETER_HEADERS):
        parameter_table_ax.text(
            -0.58,
            field_index,
            field_name,
            ha="right",
            va="center",
            fontsize=CM_MODEL_LABEL_SIZE,
            fontfamily="monospace",
            fontweight="bold",
            clip_on=False,
        )

    for model_index, model in enumerate(models):
        parameter_table_ax.axvspan(
            model_index - 0.5,
            model_index + 0.5,
            color=model_pair_color(model_index),
            alpha=0.15,
            zorder=0,
        )
        for field_index, value in enumerate(model_parameter_values(files[model])):
            parameter_table_ax.text(
                model_index,
                field_index,
                value,
                ha="center",
                va="center",
                fontsize=CM_MODEL_LABEL_SIZE,
                fontfamily="monospace",
            )

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
        metric_limits = adaptive_metric_limits(
            metric, models, tests, all_metric_data, confidence_intervals
        )

        for model_index, model in enumerate(models):
            model_center = model_centers[model_index]
            model_color = model_pair_color(model_index)

            ax.axvspan(
                model_center - column_width / 2,
                model_center + column_width / 2,
                color=model_color,
                alpha=0.15,
                zorder=0,
            )

            active_tests = [
                test
                for test in tests
                if any(
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
            ax.set_ylabel("DOA error (Â°)", fontsize=AXIS_LABEL_TEXT_SIZE, fontweight="bold")
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

    # ---------------- Confusion matrices ----------------
    for model_index, model in enumerate(models):
        model_color = model_pair_color(model_index)

        for test_index, test in enumerate(tests):
            ax = cm_axes[test_index, model_index]
            matrix = all_confusion_data[model][test]

            if matrix is None:
                ax.set_facecolor((0.96, 0.96, 0.96))
                ax.text(
                    0.5, 0.5, "n/a",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=CM_CELL_TEXT_SIZE,
                    color="0.45",
                )
                matrix_size = len(CONFUSION_CLASSES)
            else:
                matrix_size = matrix.shape[0]
                cmap = dataset_colormap(colors[test])
                image = ax.imshow(
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

            # X labels only on the bottom dataset row.
            if test_index == n_tests - 1:
                ax.set_xticklabels(class_labels, fontsize=CM_TICK_SIZE)
                ax.set_xlabel("Predicted", fontsize=CM_TICK_SIZE + 1.5, labelpad=3)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)

            # Y labels only in the leftmost model column.
            if model_index == 0:
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
        ax = cm_axes[test_index, 0]
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

    for model_index in range(n_models):
        add_model_column_background(
            fig,
            [cm_axes[test_index, model_index] for test_index in range(n_tests)],
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

    legend_ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map={SourceLegendMarker: HandlerSourceLegendMarker()},
        ncol=len(tests),          # all datasets side by side
        fontsize=LEGEND_TEXT_SIZE,
        loc="center",
        frameon=True,
        columnspacing=1.5,
        handletextpad=0.45,
        borderpad=0.55,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot()
