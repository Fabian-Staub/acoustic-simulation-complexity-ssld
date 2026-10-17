import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =====================================================
# CONFIG
# =====================================================

FILES = {
    "0.5 - 8 kHz": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_harp_dataset_05-8kHz_10-SCM_randRot_10-repeats.csv",
    "1 - 8 kHz": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_harp_dataset_1-8kHz_10-SCM_randRot_10-repeats.csv",
    "2 - 8 kHz": "eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_harp_dataset_2-8kHz_10-SCM_randRot_10-repeats.csv",

    "0.5 - 8 kHz highpass": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_harp_dataset_05-8kHz_highpass_10-SCM_randRot_10-repeats.csv",
    "1 - 8 kHz highpass": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_harp_dataset_1-8kHz_highpass_10-SCM_randRot_10-repeats.csv",
    "2 - 8 kHz highpass": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_harp_dataset_2-8kHz_highpass_10-SCM_randRot_10-repeats.csv",

    "0.5 - 8 kHz best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_harp_dataset_05-8kHz_10-SCM_randRot_10-repeats_best-thresh.csv",
    "1 - 8 kHz best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_harp_dataset_1-8kHz_10-SCM_randRot_10-repeats_best-thresh.csv",
    "2 - 8 kHz best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_harp_dataset_2-8kHz_10-SCM_randRot_10-repeats_best-thresh.csv",

    "0.5 - 8 kHz highpass best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_harp_dataset_05-8kHz_highpass_10-SCM_randRot_10-repeats_best-thresh.csv",
    "1 - 8 kHz highpass best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_harp_dataset_1-8kHz_highpass_10-SCM_randRot_10-repeats_best-thresh.csv",
    "2 - 8 kHz highpass best-thresh": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_harp_dataset_2-8kHz_highpass_10-SCM_randRot_10-repeats_best-thresh.csv",
    # add more Models here
}

metrics = ["ER", "F", "LE", "LR", "SELD"]

metric_names = {
    "ER": "Error Rate ↓",
    "F": "F-Score ↑",
    "LE": "DOA Error ↓",
    "LR": "Localization Recall ↑",
    "SELD": "SELD ↓"
}

TESTS = ["VAL", "ARNI", "IVAN"]
SOURCES = ["all", "1_source", "2_sources", "3_sources"]
MODELS = list(FILES.keys())

BASE_COLORS = {
    "VAL": np.array([44, 160, 44]) / 255.0,
    "ARNI": np.array([31, 119, 180]) / 255.0,
    "IVAN": np.array([255, 127, 14]) / 255.0
}

SHADE = {
    "all": 1.0,
    "1_source": 1,
    "2_sources": 1,
    "3_sources": 1
}

ALPHA_MAP = {
    "all": 1,
    "1_source": 0.2,
    "2_sources": 0.2,
    "3_sources": 0.2
}

SHOW_X = {
    "all": True,
    "1_source": False,
    "2_sources": False,
    "3_sources": False
}

MARKER_SCALE = {
    "all": 1.5,
    "1_source": 1,
    "2_sources": 1,
    "3_sources": 1
}

MODEL_COLORS = [
    # Red
    "#c00000",
    "#e67c7c",
    "#f5d0d0",

    # Blue
    "#0050ff",
    "#7fa8ff",
    "#d6e2ff",

    # Green
    "#00a000",
    "#7fd67f",
    "#d8f0d8",

    # Orange
    "#ff8000",
    "#ffbf66",
    "#ffe5bf",
]

# =====================================================
# LOAD DATA (multi-Model)
# =====================================================

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

                # normalize LE (0–180 → 0–1)
                if m == "LE":
                    val = val / 30

                data[test][split][m].append(val)

    return data


all_data = {
    Model: load_data(path)
    for Model, path in FILES.items()
}

# =====================================================
# MEAN
# =====================================================

def mean(v):
    return float(np.mean(v)) if len(v) > 0 else 0.0


results = {
    Model: {
        test: {
            src: {m: mean(all_data[Model][test][src][m]) for m in metrics}
            for src in SOURCES
        }
        for test in TESTS
    }
    for Model in MODELS
}

# =====================================================
# COLOR SHADING
# =====================================================

def shade(color, factor):
    return color + (1 - color) * (1 - factor)

# =====================================================
# PLOT
# =====================================================

def plot():

    fig, axes = plt.subplots(
        len(metrics), 1,
        figsize=(20, 4 * len(metrics)),
        sharex=False
    )

    if len(metrics) == 1:
        axes = [axes]

    Model_spacing = 0.65
    test_spacing = 0.3
    source_spacing = 0.05

    for k, m in enumerate(metrics):

        ax = axes[k]
        x0 = 0

        for m_idx, Model in enumerate(MODELS):

            Model_offset_base = (m_idx - (len(MODELS) - 1) / 2) * Model_spacing

            for i, test in enumerate(TESTS):

                base = BASE_COLORS[test]

                for j, src in enumerate(SOURCES):

                    col = shade(base, SHADE.get(src, 1.0))

                    test_offset = (
                        (i - (len(TESTS) - 1) / 2) * test_spacing
                        + (j - (len(SOURCES) - 1) / 2) * source_spacing
                    )

                    offset = Model_offset_base + test_offset

                    values = [
                        v for v in all_data[Model][test][src][m]
                        if v != 0
                    ]

                    if len(values) == 0:
                        continue

                    if SHOW_X.get(src, True):
                        for v in values:
                            ax.scatter(
                                x0 + offset,
                                v,
                                marker="x",
                                color=col,
                                alpha=ALPHA_MAP.get(src, 0.6) * 0.8,
                                s=40 * MARKER_SCALE.get(src, 1.0)
                            )

                    ax.scatter(
                        x0 + offset,
                        results[Model][test][src][m],
                        marker="o",
                        s=110 * MARKER_SCALE.get(src, 1.0),
                        color=col,
                        alpha=ALPHA_MAP.get(src, 1),
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
                            x0 + offset,
                            results[Model][test][src][m],
                            label,
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                            color="black",
                            alpha=0.5,
                            zorder=4
                        )

        # Model shading
        Model_centers = [
            (m_idx - (len(MODELS) - 1) / 2) * Model_spacing
            for m_idx in range(len(MODELS))
        ]

        for m_idx, Model in enumerate(MODES if False else MODELS):
            Model_center = x0 + Model_centers[m_idx]

            ax.axvspan(
                Model_center - Model_spacing / 2,
                Model_center + Model_spacing / 2,
                color=MODEL_COLORS[m_idx],
                alpha=0.1,
                zorder=0
            )

        ax.set_title(metric_names[m])
        ax.set_ylim(0, 1)
        ax.set_xlim(
            -len(MODELS) * Model_spacing / 2 - 0.4,
            len(MODELS) * Model_spacing / 2 + 0.4
        )
        ax.set_xticks([])
        ax.grid(True, axis="y", alpha=0.3)

        if m == "LE":
            ax.set_ylabel("DOA Error (°)")
            ax.set_ylim(0, 1)

            degree_ticks = np.arange(0, 31, 5)  # 0,5,...,45
            ax.set_yticks(degree_ticks / 30)
            ax.set_yticklabels([f"{d}" for d in degree_ticks])
        else:
            ax.set_ylabel(metric_names[m])

    legend_handles = []

    for m_idx, Model in enumerate(MODELS):
        legend_handles.append(
            Line2D(
                [0], [0],
                marker='s',
                linestyle='None',
                markersize=10,
                markerfacecolor=MODEL_COLORS[m_idx],
                markeredgecolor='black',
                alpha=0.5,
                label=Model
            )
        )

    for test in TESTS:
        legend_handles.append(
            Line2D(
                [0], [0],
                marker='o',
                linestyle='None',
                markersize=10,
                markerfacecolor=BASE_COLORS[test],
                markeredgecolor='black',
                label=test
            )
        )

    axes[0].legend(handles=legend_handles, ncol=2)

    fig.suptitle("Metrics for each Model on the Test Sets", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("plot_results.png", dpi=300, bbox_inches="tight")


plot()