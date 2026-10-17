import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =====================================================
# CONFIG
# =====================================================

FILES = {
    "Model1": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_harp_dataset_05-8kHz_highpass_10-SCM_randRot_10-repeats_best-thresh.csv",
    "Model2": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_harp_dataset_1-8kHz_highpass_10-SCM_randRot_10-repeats_best-thresh.csv",
    "Model3": "eval_models/eval_model_R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_harp_dataset_2-8kHz_highpass_10-SCM_randRot_10-repeats_best-thresh.csv",
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

TESTS = ["ARNI", "IVAN"]
SOURCES = ["all", "1_source", "2_sources", "3_sources"]
MODELS = list(FILES.keys())

BASE_COLORS = {
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

MODEL_COLORS = {
    "Model1": "#b41f1f",
    "Model2": "#0e26ff",
    "Model3": "#2ca02c",
    "Model4": "#d62728",
}

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
                    val = val / 45

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

    fig, ax = plt.subplots(figsize=(20, 6))

    METRIC_SPACING = 2.1
    x = np.arange(len(metrics)) * METRIC_SPACING

    Model_spacing = 0.65
    test_spacing = 0.3
    source_spacing = 0.05

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

                for k, m in enumerate(metrics):

                    values = [
                        v for v in all_data[Model][test][src][m]
                        if v != 0
                    ]

                    if len(values) == 0:
                        continue

                    if SHOW_X.get(src, True):
                        for v in values:
                            ax.scatter(
                                x[k] + offset,
                                v,
                                marker="x",
                                color=col,
                                alpha=(ALPHA_MAP.get(src, 0.6)*0.8),
                                s=40 * MARKER_SCALE.get(src, 1.0)
                            )

                    ax.scatter(
                        x[k] + offset,
                        results[Model][test][src][m],
                        marker="o",
                        s=110 * MARKER_SCALE.get(src, 1.0),   # <-- NEW
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
                            x[k] + offset,
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

    # =================================================
    # AXES
    # =================================================

    ax.set_xticks(x)
    ax.set_xticklabels([metric_names[m] for m in metrics])
    ax.set_ylim(0, 1)

    ax2 = ax.twinx()
    ax2.set_ylim(0, 45)
    ax2.set_ylabel("DOA Error (°)")
    ax.margins(x=0.01)
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Error Rate, F-Score, Localization Recall, SELD")
    ax.set_title("Metrics for each Model on the Test Sets", pad=20)
    ax.grid(True, axis="y", alpha=0.3)

    # =================================================
    # LEGEND
    # =================================================

    legend_handles = []

    for Model in MODELS:
        legend_handles.append(
            Line2D(
                [0], [0],
                marker='s',
                linestyle='None',
                markersize=10,
                markerfacecolor=MODEL_COLORS[Model],
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

    ax.legend(handles=legend_handles, ncol=2)

    # =================================================
    # Model SHADING
    # =================================================

    Model_centers = [
        (m_idx - (len(MODELS) - 1) / 2) * Model_spacing
        for m_idx in range(len(MODELS))
    ]

    for k in range(len(metrics)):
        metric_center = x[k]

        for m_idx, Model in enumerate(MODELS):
            Model_center = metric_center + Model_centers[m_idx]

            ax.axvspan(
                Model_center - Model_spacing / 2,
                Model_center + Model_spacing / 2,
                color=MODEL_COLORS[Model],
                alpha=0.1,
                zorder=0
            )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
    plt.savefig("plot_results.png", dpi=300)


plot()