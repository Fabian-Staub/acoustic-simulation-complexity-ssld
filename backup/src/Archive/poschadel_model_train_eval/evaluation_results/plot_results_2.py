import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Circle, Rectangle
from matplotlib.text import Text


# Never draw automatic outlines around scatter markers.
plt.rcParams["scatter.edgecolors"] = "none"
plt.rcParams["patch.force_edgecolor"] = False


# =====================================================
# CONFIG: replace these placeholders with your JSON files
# =====================================================
'''
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

OUTPUT_PATH = Path("plot_results_with_confusion_matrices.png")

# JSON sections compared in every metric plot.
SOURCES = ["all", "1_source", "2_sources", "3_sources"]
SOURCE_COUNT_KEYS = {
    "1_source": "1",
    "2_sources": "2",
    "3_sources": "3",
}
SOURCE_LABELS = {
    "all": "Overall",
    "1_source": "1 source",
    "2_sources": "2 sources",
    "3_sources": "3 sources",
}

# Known dataset colors. Unknown labels receive stable tab20 colors.
KNOWN_DATASET_COLORS = {
    "checkpoint_dataset": np.array([44, 160, 44]) / 255.0,
    "arni-dataset": np.array([31, 119, 180]) / 255.0,
    "ivan-dataset": np.array([255, 127, 14]) / 255.0,
    "motus-dataset": np.array([148, 103, 189]) / 255.0,
}

# Adjacent model pairs share a base color; each member uses a dark/light shade.
PAIR_BASE_COLORS = [
    "#c00000", "#0050ff", "#00a000",
    "#ff8000", "#7f3c8d", "#11a579",
]
PAIR_SHADE_FACTORS = [0.0, 0.55]

SOURCE_ALPHA = {
    "all": 1.0,
    "1_source": 0.25,
    "2_sources": 0.25,
    "3_sources": 0.25,
}

# Confusion-matrix layout and text settings.
CONFUSION_CLASSES = ["0", "1", "2", "3"]
CM_TEXT_DECIMALS = 0
CM_MODEL_COLUMN_ALPHA = 0.065
CM_CELL_TEXT_SIZE = 8.0
CM_TICK_SIZE = 6.0
CM_DATASET_LABEL_SIZE = 8
CM_MODEL_LABEL_SIZE = 7

# Extra vertical extension of confusion-matrix columns.
# The labels and heading are shifted together with this extension.
CM_COLUMN_EXTEND_TOP = 0.01
CM_COLUMN_EXTEND_BOTTOM = 0.015


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
    """Return a dark/light shade shared by each adjacent model pair."""
    pair_index = model_index // 2
    base = PAIR_BASE_COLORS[pair_index % len(PAIR_BASE_COLORS)]
    shade = PAIR_SHADE_FACTORS[model_index % 2]
    return lighten_color(base, shade)


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
        by_count = results.get("by_gt_source_count", {})

        for metric in metrics:
            metric_data[test]["all"][metric] = normalized_metric(
                metric, overall.get(metric)
            )
            for source, count_key in SOURCE_COUNT_KEYS.items():
                metric_data[test][source][metric] = normalized_metric(
                    metric, by_count.get(count_key, {}).get(metric)
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


class SourceLegendMarker:
    """Container describing a numbered source-count legend marker."""

    def __init__(self, number, alpha):
        self.number = number
        self.alpha = alpha


class HandlerSourceLegendMarker(HandlerBase):
    """Draw a translucent circle with a centered number."""

    def create_artists(
        self, legend, original_handle, xdescent, ydescent,
        width, height, fontsize, transform
    ):
        center_x = xdescent + width / 2
        center_y = ydescent + height / 2
        radius = min(width, height) * 0.7

        circle = Circle(
            (center_x, center_y),
            radius=radius,
            facecolor="gray",
            edgecolor="none",
            alpha=original_handle.alpha,
            transform=transform,
        )
        number = Text(
            x=center_x,
            y=center_y,
            text=original_handle.number,
            ha="center",
            va="center",
            fontsize=fontsize * 0.72,
            fontweight="bold",
            color="black",
            alpha=0.7,
            transform=transform,
        )
        return [circle, number]


# =====================================================
# PLOT
# =====================================================
def plot(file_paths=FILE_PATHS, output_path=OUTPUT_PATH, show=True):
    files = build_file_map(file_paths)
    validate_files(files)

    raw_models = {name: read_json(path) for name, path in files.items()}
    models = list(files.keys())
    metrics = discover_metrics(raw_models)
    tests = discover_tests(raw_models)
    colors = dataset_colors(tests)

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

    # A wide canvas keeps model names and compact matrices readable.
    fig_width = max(18.0, 1.48 * n_models + 3.0)
    metric_height = 2.65
    cm_height = 1.38
    metric_cm_gap = -0.2
    fig_height = (
        1.8
        + n_metrics * metric_height
        + metric_cm_gap
        + n_tests * cm_height
        + 1.0
    )

    # A dedicated spacer row creates extra separation only between the metric
    # plots and the confusion-matrix block.
    height_ratios = (
        [metric_height] * n_metrics
        + [metric_cm_gap]
        + [cm_height] * n_tests
    )
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=n_metrics + 1 + n_tests,
        ncols=n_models,
        height_ratios=height_ratios,
        left=0.065,
        right=0.992,
        bottom=0.070,
        top=0.94,
        hspace=0.25,
        wspace=-0.04,
    )

    # Metric axes span all model columns.
    metric_axes = [fig.add_subplot(grid[row, :]) for row in range(n_metrics)]

    # One compact confusion matrix for every dataset-model combination.
    cm_axes = np.empty((n_tests, n_models), dtype=object)
    for test_index in range(n_tests):
        for model_index in range(n_models):
            cm_axes[test_index, model_index] = fig.add_subplot(
                grid[n_metrics + 1 + test_index, model_index]
            )

    model_centers = np.arange(n_models)
    column_width = 1.0
    inner_width = column_width * 0.88

    # ---------------- Metric plots ----------------
    for metric_index, metric in enumerate(metrics):
        ax = metric_axes[metric_index]

        for model_index, model in enumerate(models):
            model_center = model_centers[model_index]
            model_color = model_pair_color(model_index)

            ax.axvspan(
                model_center - column_width / 2,
                model_center + column_width / 2,
                color=model_color,
                alpha=0.08,
                zorder=0,
            )

            active_tests = [
                test
                for test in tests
                if any(
                    has_value(all_metric_data[model][test][source][metric])
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
                    if has_value(all_metric_data[model][test][source][metric])
                ]
                source_group_width = inner_width / max(len(active_tests), 1) * 0.62
                source_offsets = compact_centered_offsets(
                    len(active_sources),
                    total_width=source_group_width,
                    min_slots=4,
                )

                for source_index, source in enumerate(active_sources):
                    value = all_metric_data[model][test][source][metric]
                    x = (
                        model_center
                        + test_offsets[test_index]
                        + source_offsets[source_index]
                    )

                    ax.scatter(
                        x,
                        value,
                        marker="o",
                        s=105 if source == "all" else 73,
                        facecolors=colors[test],
                        alpha=SOURCE_ALPHA[source],
                        edgecolors="none",
                        linewidths=0.0,
                        zorder=3,
                    )

                    if source != "all":
                        ax.text(
                            x,
                            value,
                            SOURCE_COUNT_KEYS[source],
                            ha="center",
                            va="center",
                            fontsize=6.6,
                            fontweight="bold",
                            color="black",
                            alpha=0.72,
                            zorder=4,
                        )

        ax.set_title(metric, fontsize=10, fontweight="bold", pad=7)
        # Match the exact GridSpec column boundaries used by the confusion matrices.
        ax.set_xlim(-0.5, n_models - 0.5)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.27, linewidth=0.7)
        ax.set_xticks(model_centers)

        # Model names are shown only below the confusion-matrix columns.
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

        if metric == "LE":
            ax.set_ylabel("DOA error (°)", fontsize=8)
            degree_ticks = np.arange(0, 46, 5)
            ax.set_yticks(degree_ticks / 45)
            ax.set_yticklabels([str(degree) for degree in degree_ticks], fontsize=7)
        else:
            ax.set_ylabel(metric, fontsize=8)
            ax.tick_params(axis="y", labelsize=7)

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
                ax.set_xlabel("Predicted", fontsize=CM_TICK_SIZE + 0.5, labelpad=1)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)

            # Y labels only in the leftmost model column.
            if model_index == 0:
                ax.set_yticklabels(class_labels, fontsize=CM_TICK_SIZE)
                ax.set_ylabel("True", fontsize=CM_TICK_SIZE + 0.5, labelpad=1)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)

            # No dataset row labels are drawn beside the confusion matrices.

            ax.set_xlim(-0.5, matrix_size - 0.5)
            ax.set_ylim(matrix_size - 0.5, -0.5)
            ax.tick_params(axis="both", which="both", pad=0.5, length=1.5)

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

    # Model names appear only below the bottom confusion matrix in each column.
    for model_index, model in enumerate(models):
        bottom_ax = cm_axes[-1, model_index]
        pos = bottom_ax.get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y0 - CM_COLUMN_EXTEND_BOTTOM - 0.005,
            model,
            ha="center",
            va="top",
            fontsize=CM_MODEL_LABEL_SIZE,
            fontweight="bold",
            color="black", #model_pair_color(model_index),
            rotation=18 if n_models > 8 else 0,
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

    # Heading between metric plots and confusion matrices.
    # Keep it close to the first CM row and do not draw a divider here.
    cm_top = max(cm_axes[0, i].get_position().y1 for i in range(n_models)) + CM_COLUMN_EXTEND_TOP
    last_metric_bottom = metric_axes[-1].get_position().y0
    gap_bottom = cm_top
    gap_top = last_metric_bottom
    heading_y = gap_bottom + 0.12 * (gap_top - gap_bottom)
    fig.text(
        (0.065 + 0.992) / 2,
        heading_y,
        "Source-count confusion matrices (%)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    # CM column boundaries are formed by the adjacent shaded backgrounds above.

    # ---------------- Combined legend ----------------
    legend_handles = []
    legend_labels = []

    for model_index, model in enumerate(models):
        model_color = model_pair_color(model_index)
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="s",
                linestyle="None",
                markersize=8,
                markerfacecolor=model_color,
                markeredgecolor="none",
            )
        )
        legend_labels.append(model)

    for test in tests:
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markersize=8,
                markerfacecolor=colors[test],
                markeredgecolor="none",
            )
        )
        legend_labels.append(test)

    legend_handles.append(
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor="gray",
            markeredgecolor="none",
            alpha=SOURCE_ALPHA["all"],
        )
    )
    legend_labels.append(SOURCE_LABELS["all"])

    for source in ("1_source", "2_sources", "3_sources"):
        legend_handles.append(
            SourceLegendMarker(
                number=SOURCE_COUNT_KEYS[source],
                alpha=SOURCE_ALPHA[source],
            )
        )
        legend_labels.append(SOURCE_LABELS[source])

    # Matplotlib fills legends column-wise; this gives 4 entries per column
    # (except the final column when the item count is not divisible by 4).
    legend_columns = math.ceil(len(legend_handles) / 4)
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map={SourceLegendMarker: HandlerSourceLegendMarker()},
        ncol=legend_columns,
        fontsize=7.3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        frameon=True,
        columnspacing=1.1,
        handletextpad=0.45,
        borderpad=0.55,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot()
