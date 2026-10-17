import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =====================================================
# CONFIG
# =====================================================
'''
FILES = {
    "0.5 - 8 kHz": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM_harp_dataset_05-8kHz_10-SCM_randRot_10-repeats.csv",
    "1 - 8 kHz": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM_harp_dataset_1-8kHz_10-SCM_randRot_10-repeats.csv",
    "2 - 8 kHz": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM_harp_dataset_2-8kHz_10-SCM_randRot_10-repeats.csv",

    "0.5 - 8 kHz highpass": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM_harp_dataset_05-8kHz_highpass_10-SCM_randRot_10-repeats.csv",
    "1 - 8 kHz highpass": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM_harp_dataset_1-8kHz_highpass_10-SCM_randRot_10-repeats.csv",
    "2 - 8 kHz highpass": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM_harp_dataset_2-8kHz_highpass_10-SCM_randRot_10-repeats.csv",

    "0.5 - 8 kHz best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM_harp_dataset_05-8kHz_10-SCM_randRot_10-repeats_best-thresh.csv",
    "1 - 8 kHz best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM_harp_dataset_1-8kHz_10-SCM_randRot_10-repeats_best-thresh.csv",
    "2 - 8 kHz best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM_harp_dataset_2-8kHz_10-SCM_randRot_10-repeats_best-thresh.csv",

    "0.5 - 8 kHz highpass best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM_harp_dataset_05-8kHz_highpass_10-SCM_randRot_10-repeats_best-thresh.csv",
    "1 - 8 kHz highpass best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM_harp_dataset_1-8kHz_highpass_10-SCM_randRot_10-repeats_best-thresh.csv",
    "2 - 8 kHz highpass best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM_harp_dataset_2-8kHz_highpass_10-SCM_randRot_10-repeats_best-thresh.csv",
}'''

FILES = {
    "ISM Order2": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISMOrder2_harp_dataset_2-8kHz_highpass_10-SCM_2-src_randRot_10-repeats.csv",
    "ISM Order2 + Late": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISMOrder2+Late_harp_dataset_2-8kHz_highpass_10-SCM_2-src_randRot_10-repeats.csv",
    "ISM Order20": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM_harp_dataset_2-8kHz_highpass_10-SCM_2-src_randRot_10-repeats.csv",
    "ISM Order20 + Late": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM+Late_harp_dataset_2-8kHz_highpass_10-SCM_2-src_randRot_10-repeats.csv",

    "ISM Order2 best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISMOrder2_harp_dataset_2-8kHz_highpass_10-SCM_2-src_randRot_10-repeats_best-thresh.csv",
    "ISM Order2 + Late best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISMOrder2+Late_harp_dataset_2-8kHz_highpass_10-SCM_2-src_randRot_10-repeats_best-thresh.csv",
    "ISM Order20 best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM_harp_dataset_2-8kHz_highpass_10-SCM_2-src_randRot_10-repeats_best-thresh.csv",
    "ISM Order20 + Late best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM+Late_harp_dataset_2-8kHz_highpass_10-SCM_2-src_randRot_10-repeats_best-thresh.csv",
}

metrics = ["ER", "F", "LE", "LR", "SELD"]

metric_names = {
    "ER": "Error Rate ↓",
    "F": "F-Score ↑",
    "LE": "DOA Error ↓",
    "LR": "Localization Recall ↑",
    "SELD": "SELD ↓"
}

SOURCES = ["all", "1_source", "2_sources", "3_sources"]
MODELS = list(FILES.keys())

BASE_COLORS = {
    "VAL": np.array([44, 160, 44]) / 255.0,
    "ARNI": np.array([31, 119, 180]) / 255.0,
    "IVAN": np.array([255, 127, 14]) / 255.0,
}

DEFAULT_TEST_COLOR = np.array([0.45, 0.45, 0.45])

SHADE = {
    "all": 1.0,
    "1_source": 1.0,
    "2_sources": 1.0,
    "3_sources": 1.0,
}

ALPHA_MAP = {
    "all": 1.0,
    "1_source": 0.2,
    "2_sources": 0.2,
    "3_sources": 0.2,
}

SHOW_X = {
    "all": True,
    "1_source": False,
    "2_sources": False,
    "3_sources": False,
}

MARKER_SCALE = {
    "all": 1.5,
    "1_source": 1.0,
    "2_sources": 1.0,
    "3_sources": 1.0,
}
'''
MODEL_COLORS = [
    "#c00000",
    "#e67c7c",
    "#f5d0d0",

    "#0050ff",
    "#7fa8ff",
    "#d6e2ff",

    "#00a000",
    "#7fd67f",
    "#d8f0d8",

    "#ff8000",
    "#ffbf66",
    "#ffe5bf",
]'''

MODEL_COLORS = [
    "#c00000",
    "#0050ff",
    "#00a000",
    "#ff8000",

    "#e67c7c",
    "#7fa8ff",
    "#7fd67f",
    "#ffbf66",
]


# =====================================================
# HELPERS
# =====================================================

def mean(v):
    return float(np.mean(v)) if len(v) > 0 else 0.0


def shade(color, factor):
    color = np.asarray(color)
    return color + (1 - color) * (1 - factor)


def discover_tests(files):
    """
    Use the CSV with the most rows as the reference order for tests.
    The test order is the order in which datasets first appear in that CSV.
    """

    best_file = None
    best_row_count = -1

    # Find CSV with the most entries / rows
    for file_path in files.values():
        with open(file_path, newline="") as f:
            reader = csv.DictReader(f)
            row_count = sum(1 for _ in reader)

        if row_count > best_row_count:
            best_row_count = row_count
            best_file = file_path

    # Take dataset order from that CSV
    tests = []

    with open(best_file, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            dataset = row["dataset"].strip()

            if dataset and dataset not in tests:
                tests.append(dataset)

    return tests


def compact_centered_offsets(n, total_width, min_slots=3):
    """
    Returns n offsets centered around 0.

    The min_slots argument prevents 2 items from being placed
    at the far edges of the available width.
    """
    if n <= 0:
        return np.array([])

    if n == 1:
        return np.array([0.0])

    step = total_width / max(n, min_slots)

    return (
        np.arange(n)
        - (n - 1) / 2
    ) * step


def jitter_offsets(n, width=0.018):
    """
    Small deterministic horizontal jitter for repeated individual runs.
    """
    if n <= 1:
        return np.array([0.0])

    return np.linspace(-width / 2, width / 2, n)


# =====================================================
# LOAD DATA
# =====================================================

TESTS = discover_tests(FILES)


def load_data(file_path):

    data = {
        test: {
            src: {m: [] for m in metrics}
            for src in SOURCES
        }
        for test in TESTS
    }

    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:

            test = row["dataset"].strip()
            if test not in TESTS:
                continue

            split = row["split"].strip()
            if split not in SOURCES:
                continue

            for m in metrics:
                val = float(row[m])

                # Normalize LE from degrees to plotting range.
                if m == "LE":
                    val = val / 45

                data[test][split][m].append(val)

    return data


all_data = {
    Model: load_data(path)
    for Model, path in FILES.items()
}


# =====================================================
# MEAN RESULTS
# =====================================================

results = {
    Model: {
        test: {
            src: {
                m: mean(all_data[Model][test][src][m])
                for m in metrics
            }
            for src in SOURCES
        }
        for test in TESTS
    }
    for Model in MODELS
}


# =====================================================
# PLOT
# =====================================================

def plot():

    n_models = len(MODELS)
    n_metrics = len(metrics)

    fig_width = max(14, n_models * 1.25)
    fig_height = 4 * n_metrics

    fig, axes = plt.subplots(
        n_metrics,
        1,
        figsize=(fig_width, fig_height),
        sharex=False
    )

    if n_metrics == 1:
        axes = [axes]

    # Each model owns exactly one vertical column.
    model_centers = np.arange(n_models)

    column_width = 1
    inner_width = column_width * 0.9

    for k, metric in enumerate(metrics):

        ax = axes[k]

        for model_idx, Model in enumerate(MODELS):

            model_center = model_centers[model_idx]

            # Vertical shaded model column.
            ax.axvspan(
                model_center - column_width / 2,
                model_center + column_width / 2,
                color=MODEL_COLORS[model_idx % len(MODEL_COLORS)],
                alpha=0.08,
                zorder=0
            )

            # Only datasets that actually exist for this model/metric.
            active_tests = []

            for test in TESTS:
                has_values = any(
                    len(all_data[Model][test][src][metric]) > 0
                    for src in SOURCES
                )

                if has_values:
                    active_tests.append(test)

            if len(active_tests) == 0:
                continue

            # Compact, centered dataset positions.
            # min_slots=3 makes two datasets closer together.
            test_offsets = compact_centered_offsets(
                n=len(active_tests),
                total_width=inner_width,
                min_slots=2
            )

            for test_idx, test in enumerate(active_tests):

                active_sources = [
                    src for src in SOURCES
                    if len(all_data[Model][test][src][metric]) > 0
                ]

                if len(active_sources) == 0:
                    continue

                # Source positions are centered around each dataset position.
                source_group_width = (
                    inner_width
                    / max(len(active_tests), 1)
                    * 0.55
                )

                source_offsets = compact_centered_offsets(
                    n=len(active_sources),
                    total_width=source_group_width,
                    min_slots=4
                )

                base = BASE_COLORS.get(test, DEFAULT_TEST_COLOR)

                for src_idx, src in enumerate(active_sources):

                    col = shade(base, SHADE.get(src, 1.0))

                    x = (
                        model_center
                        + test_offsets[test_idx]
                        + source_offsets[src_idx]
                    )

                    values = [
                        v for v in all_data[Model][test][src][metric]
                        if v != 0
                    ]

                    if len(values) == 0:
                        continue

                    # Individual repeated runs.
                    if SHOW_X.get(src, True):

                        jitters = jitter_offsets(len(values), width=0.018)

                        for v, dx in zip(values, jitters):
                            ax.scatter(
                                x + dx,
                                v,
                                marker="x",
                                color=col,
                                alpha=ALPHA_MAP.get(src, 0.6) * 0.8,
                                s=40 * MARKER_SCALE.get(src, 1.0),
                                zorder=2
                            )

                    # Mean marker.
                    mean_val = results[Model][test][src][metric]

                    ax.scatter(
                        x,
                        mean_val,
                        marker="o",
                        s=110 * MARKER_SCALE.get(src, 1.0),
                        color=col,
                        alpha=ALPHA_MAP.get(src, 1.0),
                        edgecolor="black",
                        linewidths=0.3,
                        zorder=3
                    )

                    src_label_map = {
                        "1_source": "1",
                        "2_sources": "2",
                        "3_sources": "3",
                        "all": ""
                    }

                    label = src_label_map.get(src, "")

                    if label != "":
                        ax.text(
                            x,
                            mean_val,
                            label,
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                            color="black",
                            alpha=0.5,
                            zorder=4
                        )

        ax.set_title(metric_names[metric])
        ax.set_xlim(-0.6, n_models - 0.4)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)

        short_model_labels = [
            Model
            .replace(" kHz", "")
            .replace(" highpass", "\nHP")
            .replace(" best-thresh", "\nBT")
            for Model in MODELS
        ]

        ax.set_xticks(model_centers)
        ax.set_xticklabels(short_model_labels, fontsize=8)

        if metric == "LE":
            ax.set_ylabel("DOA Error (°)")

            degree_ticks = np.arange(0, 46, 5)
            ax.set_yticks(degree_ticks / 45)
            ax.set_yticklabels([f"{d}" for d in degree_ticks])
        else:
            ax.set_ylabel(metric_names[metric])

    # =====================================================
    # LEGEND
    # =====================================================

    legend_handles = []

    for model_idx, Model in enumerate(MODELS):
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="s",
                linestyle="None",
                markersize=10,
                markerfacecolor=MODEL_COLORS[model_idx % len(MODEL_COLORS)],
                markeredgecolor="black",
                alpha=0.5,
                label=Model
            )
        )

    for test in TESTS:
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markersize=10,
                markerfacecolor=BASE_COLORS.get(test, DEFAULT_TEST_COLOR),
                markeredgecolor="black",
                label=test
            )
        )

    legend_cols = min(4, max(2, n_models // 3))

    axes[0].legend(
        handles=legend_handles,
        ncol=legend_cols,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        frameon=True
    )

    fig.suptitle(
        "Metrics for each Model on the Test Sets",
        fontsize=16,
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("plot_results.png", dpi=300, bbox_inches="tight")
    plt.show()


plot()