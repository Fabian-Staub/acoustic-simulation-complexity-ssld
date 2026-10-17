import json
from pathlib import Path

import matplotlib.pyplot as plt

# Never draw automatic outlines around scatter markers.
plt.rcParams["scatter.edgecolors"] = "none"
plt.rcParams["patch.force_edgecolor"] = False
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Circle
from matplotlib.text import Text


# =====================================================
# CONFIG: replace these placeholders with your 12 JSON files
# =====================================================
FILE_PATHS = [
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder12.json"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/eval_shoeBox_em32_ISMOrder12.json"),
]

OUTPUT_PATH = Path("plot_results_json.png")

# Metric names are taken directly from the JSON result dictionaries.
# Their order is discovered from the first available dataset.
METRICS = []

# JSON sections to compare for every dataset.
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

# Known dataset colors. Unknown labels receive colors from matplotlib's tab20 map.
KNOWN_DATASET_COLORS = {
    "checkpoint_dataset": np.array([44, 160, 44]) / 255.0,
    "arni-dataset": np.array([31, 119, 180]) / 255.0,
    "ivan-dataset": np.array([255, 127, 14]) / 255.0,
    "motus-dataset": np.array([148, 103, 189]) / 255.0,
}

# One base color per adjacent model pair: models 1–2 share the first
# base color, models 3–4 the second, and so on. The two models in
# each pair use a dark and a light shade of that base color.
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

        # Keep the first occurrence unchanged; suffix later duplicates.
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
    """Collect dataset labels in their first-seen JSON order."""
    tests = []
    for model_json in raw_models.values():
        for dataset in model_json.get("datasets", {}).values():
            label = dataset.get("dataset_label")
            if label and label not in tests:
                tests.append(label)
    return tests


def dataset_colors(tests):
    """Keep predefined colors and assign stable fallback colors."""
    cmap = plt.get_cmap("tab20")
    colors = {}
    for index, test in enumerate(tests):
        colors[test] = KNOWN_DATASET_COLORS.get(test, np.asarray(cmap(index % 20)[:3]))
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
        data[test][source][metric] = scalar or None
    """
    data = {
        test: {source: {metric: None for metric in metrics} for source in SOURCES}
        for test in tests
    }

    for dataset in model_json.get("datasets", {}).values():
        test = dataset.get("dataset_label")
        if test not in data:
            continue

        results = dataset.get("results", {})
        overall = results.get("overall", {})
        by_count = results.get("by_gt_source_count", {})

        for metric in metrics:
            data[test]["all"][metric] = normalized_metric(
                metric, overall.get(metric)
            )

            for source, count_key in SOURCE_COUNT_KEYS.items():
                source_results = by_count.get(count_key, {})
                data[test][source][metric] = normalized_metric(
                    metric, source_results.get(metric)
                )

    return data


def has_value(value):
    # Zero is a valid value in JSON, but source-count groups with all-zero
    # metrics are usually absent test cases. The metric itself is retained here.
    return value is not None



class SourceLegendMarker:
    """Container describing a numbered source-count legend marker."""

    def __init__(self, number, alpha):
        self.number = number
        self.alpha = alpha


class HandlerSourceLegendMarker(HandlerBase):
    """Draw a translucent circle with a centered number."""

    def create_artists(
        self,
        legend,
        original_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        transform,
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
    all_data = {
        model: load_model_data(raw_models[model], tests, metrics)
        for model in models
    }

    n_models = len(models)
    n_metrics = len(metrics)

    fig_width = max(18, n_models * 1.45)
    fig_height = 3.8 * n_metrics
    fig, axes = plt.subplots(
        n_metrics, 1, figsize=(fig_width, fig_height), sharex=False
    )
    if n_metrics == 1:
        axes = [axes]

    model_centers = np.arange(n_models)
    column_width = 1.0
    inner_width = column_width * 0.88

    for metric_index, metric in enumerate(metrics):
        ax = axes[metric_index]

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
                    has_value(all_data[model][test][source][metric])
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
                    if has_value(all_data[model][test][source][metric])
                ]
                source_group_width = (
                    inner_width / max(len(active_tests), 1) * 0.62
                )
                source_offsets = compact_centered_offsets(
                    len(active_sources),
                    total_width=source_group_width,
                    min_slots=4,
                )

                for source_index, source in enumerate(active_sources):
                    value = all_data[model][test][source][metric]
                    x = (
                        model_center
                        + test_offsets[test_index]
                        + source_offsets[source_index]
                    )

                    ax.scatter(
                        x,
                        value,
                        marker="o",
                        s=115 if source == "all" else 80,
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
                            fontsize=7,
                            fontweight="bold",
                            color="black",
                            alpha=0.7,
                            zorder=4,
                        )

        ax.set_title(metric)
        ax.set_xlim(-0.6, n_models - 0.4)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(model_centers)
        ax.set_xticklabels(models, fontsize=8, rotation=25, ha="right")

        if metric == "LE":
            ax.set_ylabel("DOA Error (°)")
            degree_ticks = np.arange(0, 46, 5)
            ax.set_yticks(degree_ticks / 45)
            ax.set_yticklabels([str(degree) for degree in degree_ticks])
        else:
            ax.set_ylabel(metric)

    legend_handles = []
    for model_index, model in enumerate(models):
        model_color = model_pair_color(model_index)
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="s",
                linestyle="None",
                markersize=9,
                markerfacecolor=model_color,
                markeredgecolor="black",
                label=model,
            )
        )

    for test in tests:
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markersize=9,
                markerfacecolor=colors[test],
                markeredgecolor="none",
                markeredgewidth=0.0,
                label=test,
            )
        )

    # Overall source marker: opaque circle without a number.
    legend_handles.append(
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=9,
            markerfacecolor="gray",
            markeredgecolor="none",
            markeredgewidth=0.0,
            alpha=SOURCE_ALPHA["all"],
            label=SOURCE_LABELS["all"],
        )
    )

    # Source-count markers: same light shade with the count inside the circle.
    for source in ("1_source", "2_sources", "3_sources"):
        legend_handles.append(
            SourceLegendMarker(
                number=SOURCE_COUNT_KEYS[source],
                alpha=SOURCE_ALPHA[source],
            )
        )

    legend_labels = (
        models
        + tests
        + [
            SOURCE_LABELS["all"],
            SOURCE_LABELS["1_source"],
            SOURCE_LABELS["2_sources"],
            SOURCE_LABELS["3_sources"],
        ]
    )

    axes[0].legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map={SourceLegendMarker: HandlerSourceLegendMarker()},
        ncol=4,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.62),
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot()
