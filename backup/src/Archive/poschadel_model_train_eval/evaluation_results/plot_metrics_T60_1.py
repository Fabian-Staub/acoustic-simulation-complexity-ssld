#!/usr/bin/env python3
"""Plot SDL metrics versus mean T60 for multiple models in one large PNG.

Layout:
    - one row per model
    - one column per metric
    - all datasets overlaid in every subplot
    - one shared y-axis scale per metric across all model rows
    - checkpoint_dataset is drawn first (behind the other dataset curves)

Dataset colors:
    checkpoint_dataset -> green
    arni dataset       -> blue
    motus dataset      -> purple
    ivan dataset       -> orange

Define the CSV file locations in MODEL_FILES below. Model names are derived
automatically from the CSV filenames.

Requirements:
    pip install numpy matplotlib
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({
    "font.size": 16,          # default text
    "axes.titlesize": 18,     # subplot titles
    "axes.labelsize": 16,     # x/y labels
    "xtick.labelsize": 14,    # x tick labels
    "ytick.labelsize": 14,    # y tick labels
    "legend.fontsize": 16,    # legend
    "figure.titlesize": 22,   # suptitle
})


DEFAULT_METRICS = ("LE", "LR", "F15", "ER15", "ESDL")
METRIC_LABELS = {
    "LE": "LE / °",
    "LR": "LR",
    "F15": "F15",
    "ER15": "ER15",
    "ESDL": "ESDL",
}

# ---------------------------------------------------------------------------
# DEFINE YOUR MODELS / CSV FILES HERE
# Every path becomes one row in the final PNG. The displayed model name is
# derived from the filename without the ``metrics_receiverPos_`` prefix.
# ---------------------------------------------------------------------------
MODEL_FILES = [

    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/metrics_receiverPos_shoeBox_em32_Direct-Late.csv"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/metrics_receiverPos_shoeBox_em32_Direct.csv"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/metrics_receiverPos_shoeBox_em32_ISMOrder2-Late.csv"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/metrics_receiverPos_shoeBox_em32_ISMOrder2.csv"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/metrics_receiverPos_shoeBox_em32_ISMOrder12-Late.csv"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/metrics_receiverPos_shoeBox_em32_ISMOrder12.csv"),

    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/metrics_receiverPos_shoeBox_SH_Direct-Late.csv"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/metrics_receiverPos_shoeBox_SH_Direct.csv"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/metrics_receiverPos_shoeBox_SH_ISMOrder2-Late.csv"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/metrics_receiverPos_shoeBox_SH_ISMOrder2.csv"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/metrics_receiverPos_shoeBox_SH_ISMOrder12-Late.csv"),
    Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/metrics_receiverPos_shoeBox_SH_ISMOrder12.csv"),
]

# Fixed colors requested for the four datasets.
# Keys are normalized, so capitalization and spaces/underscores are handled.
DATASET_COLORS = {
    "checkpoint_dataset": "green",
    "checkpoint dataset": "green",
    "arni_dataset": "blue",
    "arni dataset": "blue",
    "motus_dataset": "purple",
    "motus dataset": "purple",
    "ivan_dataset": "orange",
    "ivan dataset": "orange",
}


@dataclass(frozen=True)
class Observation:
    dataset: str
    t60: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class BinnedSeries:
    centers: np.ndarray
    means: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    counts: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics_vs_t60_all_models.png"),
        help="Output PNG (default: metrics_vs_t60_all_models.png).",
    )
    parser.add_argument(
        "--dataset-column",
        default="prediction_dataset",
        help="CSV column containing the dataset name.",
    )
    parser.add_argument(
        "--t60-column",
        default="mean_rt60",
        help="CSV column containing receiver-position reverberation time.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        metavar="NAME",
        help="Metric to plot; repeat to choose metrics. Defaults to all five.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="NAME",
        help="Dataset to plot; repeat to select several. Defaults to all.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=15,
        help="Number of common, equal-width T60 bins (default: 20).",
    )
    parser.add_argument(
        "--t60-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional fixed T60 range in seconds.",
    )
    parser.add_argument(
        "--uncertainty",
        choices=("ci95", "sem", "std", "none"),
        default="ci95",
        help="Uncertainty band around each bin mean (default: ci95).",
    )
    parser.add_argument(
        "--min-bin-count",
        type=int,
        default=1,
        help="Minimum finite metric observations required to draw a bin.",
    )
    parser.add_argument(
        "--histogram-scope",
        choices=("all", "selected"),
        default="selected",
        help="Use T60 values from selected datasets or all CSV datasets.",
    )
    parser.add_argument(
        "--histogram-alpha",
        type=float,
        default=0,
        help="Background histogram opacity (default: 0.16).",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--title",
        default="SDL metrics versus reverberation time",
        help="Figure title.",
    )
    return parser.parse_args()


def normalized_name(name: str) -> str:
    return name.replace("\\", "").strip().strip("*").lower()


def normalized_dataset_key(name: str) -> str:
    return normalized_name(name).replace("-", "_")


def dataset_color(dataset: str, fallback_index: int) -> str:
    key = normalized_dataset_key(dataset)

    # Direct lookup.
    if key in DATASET_COLORS:
        return DATASET_COLORS[key]

    # Also allow spaces/underscores to vary.
    key_spaces = key.replace("_", " ")
    if key_spaces in DATASET_COLORS:
        return DATASET_COLORS[key_spaces]

    key_underscores = key.replace(" ", "_")
    if key_underscores in DATASET_COLORS:
        return DATASET_COLORS[key_underscores]

    # Fallback for any additional datasets.
    return plt.get_cmap("tab10").colors[fallback_index % 10]


def resolve_column(fieldnames: Sequence[str], wanted: str) -> str:
    wanted_normalized = normalized_name(wanted)
    for name in fieldnames:
        if normalized_name(name) == wanted_normalized:
            return name
    raise ValueError(f"Missing column {wanted!r}; found {list(fieldnames)}")


def finite_float(raw: str | None) -> float:
    try:
        value = float((raw or "").strip())
    except ValueError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def load_observations(
    path: Path,
    dataset_column_wanted: str,
    t60_column_wanted: str,
    metric_names: Sequence[str],
) -> tuple[list[Observation], OrderedDict[str, None], int]:
    observations: list[Observation] = []
    dataset_order: OrderedDict[str, None] = OrderedDict()
    skipped_t60 = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        dataset_column = resolve_column(reader.fieldnames, dataset_column_wanted)
        t60_column = resolve_column(reader.fieldnames, t60_column_wanted)
        metric_columns = {
            metric: resolve_column(reader.fieldnames, metric)
            for metric in metric_names
        }

        for row_number, row in enumerate(reader, start=2):
            dataset = (row.get(dataset_column) or "").strip()
            if not dataset:
                raise ValueError(f"Empty dataset name in {path} row {row_number}")

            dataset_order.setdefault(dataset, None)

            t60 = finite_float(row.get(t60_column))
            if not math.isfinite(t60):
                skipped_t60 += 1
                continue

            values = {
                metric: finite_float(row.get(column))
                for metric, column in metric_columns.items()
            }
            observations.append(
                Observation(dataset=dataset, t60=t60, metrics=values)
            )

    if not observations:
        raise ValueError(f"No rows with finite T60 values were found in {path}")

    return observations, dataset_order, skipped_t60


def select_datasets(
    available: Sequence[str], requested: Sequence[str] | None
) -> list[str]:
    if not requested:
        return list(available)

    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError("Dataset(s) not found: " + ", ".join(missing))

    return list(dict.fromkeys(requested))


def dataset_plot_order(datasets: Sequence[str]) -> list[str]:
    """Return a stable plot order with checkpoint_dataset first."""
    return sorted(
        datasets,
        key=lambda name: (
            normalized_dataset_key(name).replace(" ", "_")
            != "checkpoint_dataset"
        ),
    )


def make_bin_edges(
    t60_values: np.ndarray,
    number_of_bins: int,
    requested_range: Sequence[float] | None,
) -> np.ndarray:
    if number_of_bins < 1:
        raise ValueError("--bins must be at least 1")

    if requested_range is not None:
        lower, upper = map(float, requested_range)
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError("--t60-range requires finite MIN < MAX")
    else:
        lower = float(np.min(t60_values))
        upper = float(np.max(t60_values))
        if lower == upper:
            padding = max(abs(lower) * 0.05, 0.01)
            lower -= padding
            upper += padding

    return np.linspace(lower, upper, number_of_bins + 1)


def bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign values to bins, including the rightmost edge in the last bin."""
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[values == edges[-1]] = len(edges) - 2
    return indices


def uncertainty_half_width(values: np.ndarray, method: str) -> float:
    if method == "none" or len(values) < 2:
        return 0.0

    standard_deviation = float(np.std(values, ddof=1))
    if method == "std":
        return standard_deviation

    sem = standard_deviation / math.sqrt(len(values))
    return 1.96 * sem if method == "ci95" else sem


def binned_metric_series(
    observations: Sequence[Observation],
    dataset: str,
    metric: str,
    edges: np.ndarray,
    uncertainty: str,
    min_bin_count: int,
) -> BinnedSeries:
    centers = (edges[:-1] + edges[1:]) / 2.0
    means = np.full(len(centers), np.nan)
    lower = np.full(len(centers), np.nan)
    upper = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=np.int64)

    rows = [row for row in observations if row.dataset == dataset]
    t60 = np.asarray([row.t60 for row in rows], dtype=np.float64)
    values = np.asarray([row.metrics[metric] for row in rows], dtype=np.float64)
    indices = bin_indices(t60, edges)

    for index in range(len(centers)):
        mask = (indices == index) & np.isfinite(values)
        bin_values = values[mask]
        counts[index] = len(bin_values)

        if len(bin_values) < min_bin_count:
            continue

        mean = float(np.mean(bin_values))
        half_width = uncertainty_half_width(bin_values, uncertainty)

        means[index] = mean
        lower[index] = mean - half_width
        upper[index] = mean + half_width

    return BinnedSeries(centers, means, lower, upper, counts)


def metric_axis_limits(
    metric: str,
    series_by_model_and_dataset: Sequence[BinnedSeries],
) -> tuple[float, float]:
    """Find shared metric limits, including every plotted uncertainty band."""
    lower_bounds = [
        float(np.min(series.lower[np.isfinite(series.lower)]))
        for series in series_by_model_and_dataset
        if np.any(np.isfinite(series.lower))
    ]
    upper_bounds = [
        float(np.max(series.upper[np.isfinite(series.upper)]))
        for series in series_by_model_and_dataset
        if np.any(np.isfinite(series.upper))
    ]

    if not lower_bounds or not upper_bounds:
        return 0.0, 1.0

    global_minimum = min(lower_bounds)
    global_maximum = max(upper_bounds)

    if metric in {"LE", "ER15", "ESDL"}:
        return 0.0, global_maximum if global_maximum > 0.0 else 1.0

    if metric in {"LR", "F15"}:
        return global_minimum if global_minimum < 1.0 else 0.0, 1.0

    if global_minimum == global_maximum:
        padding = max(abs(global_minimum) * 0.05, 0.01)
        return global_minimum - padding, global_maximum + padding

    return global_minimum, global_maximum


def add_background_histogram(
    axis: plt.Axes,
    t60_values: np.ndarray,
    edges: np.ndarray,
    alpha: float,
) -> None:
    histogram_axis = axis.twinx()
    histogram_axis.set_zorder(0)
    axis.set_zorder(2)
    axis.patch.set_alpha(0.0)

    histogram_axis.hist(
        t60_values,
        bins=edges,
        color="#9a9a9a",
        alpha=alpha,
        edgecolor="none",
        rwidth=0.92,
    )
    histogram_axis.set_ylim(bottom=0)
    histogram_axis.set_yticks([])
    histogram_axis.set_ylabel("")

    for spine in histogram_axis.spines.values():
        spine.set_visible(False)


def plot_all_models(
    model_data: OrderedDict[str, dict],
    datasets: Sequence[str],
    metric_names: Sequence[str],
    edges: np.ndarray,
    uncertainty: str,
    min_bin_count: int,
    histogram_alpha: float,
    histogram_scope: str,
    output_path: Path,
    dpi: int,
    title: str,
) -> None:
    model_names = list(model_data)
    nrows = len(model_names)
    ncols = len(metric_names)

    if nrows == 0:
        raise ValueError("MODEL_FILES is empty. Define at least one model CSV.")
    if ncols == 0:
        raise ValueError("At least one metric is required.")

    # Compute every visible series first so each metric column can use the
    # same limits across all models, including uncertainty-band endpoints.
    series_by_key: dict[tuple[str, str, str], BinnedSeries] = {}
    for model_name, data in model_data.items():
        available_datasets = data["datasets"]
        for metric in metric_names:
            for dataset in datasets:
                if dataset not in available_datasets:
                    continue

                series_by_key[(model_name, metric, dataset)] = (
                    binned_metric_series(
                        observations=data["observations"],
                        dataset=dataset,
                        metric=metric,
                        edges=edges,
                        uncertainty=uncertainty,
                        min_bin_count=min_bin_count,
                    )
                )

    y_limits_by_metric = {
        metric: metric_axis_limits(
            metric,
            [
                series
                for (_, series_metric, _), series in series_by_key.items()
                if series_metric == metric
            ],
        )
        for metric in metric_names
    }

    # Wide figure: all metrics in the same row; every row is a model.
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 2.5 * nrows),
        sharex=True,
        sharey="col",
        squeeze=False,
    )

    for row_index, model_name in enumerate(model_names):
        observations = model_data[model_name]["observations"]
        available_datasets = model_data[model_name]["datasets"]

        if histogram_scope == "all":
            histogram_datasets = available_datasets
        else:
            histogram_datasets = [d for d in datasets if d in available_datasets]

        histogram_t60 = np.asarray(
            [
                row.t60
                for row in observations
                if row.dataset in histogram_datasets
            ],
            dtype=np.float64,
        )

        if len(histogram_t60) == 0:
            raise ValueError(
                f"No T60 observations remain for histogram in model {model_name!r}"
            )

        for col_index, metric in enumerate(metric_names):
            axis = axes[row_index, col_index]

            add_background_histogram(
                axis, histogram_t60, edges, histogram_alpha
            )

            for dataset_index, dataset in enumerate(datasets):
                if dataset not in available_datasets:
                    continue

                series = series_by_key[(model_name, metric, dataset)]

                if not np.any(np.isfinite(series.means)):
                    continue

                color = dataset_color(dataset, dataset_index)

                axis.plot(
                    series.centers,
                    series.means,
                    color=color,
                    linewidth=2.0,
                    marker="o",
                    markersize=4.0,
                    label=dataset,
                    zorder=4,
                )

                if uncertainty != "none":
                    axis.fill_between(
                        series.centers,
                        series.lower,
                        series.upper,
                        color=color,
                        alpha=0.16,
                        linewidth=0,
                        zorder=3,
                    )

            if row_index == 0:
                axis.set_title(
                    metric,
                    fontweight="semibold",
                )

            # Put model name on the left side of each row.
            if col_index == 0:
                axis.set_ylabel(
                    f"{model_name}\n{METRIC_LABELS.get(metric, metric)}"
                )
            else:
                axis.set_ylabel(METRIC_LABELS.get(metric, metric))

            if row_index == nrows - 1:
                axis.set_xlabel("Mean T60 / s")

            axis.grid(axis="y", linestyle=(0, (1, 4)), alpha=0.4)
            axis.set_xlim(edges[0], edges[-1])
            axis.set_ylim(*y_limits_by_metric[metric])

    # One global legend for the entire figure.
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=dataset_color(dataset, index),
            linewidth=2.0,
            marker="o",
            markersize=4.0,
        )
        for index, dataset in enumerate(datasets)
    ]

    if legend_handles:
        fig.legend(
            legend_handles,
            list(datasets),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=min(4, len(datasets)),
            frameon=True,
        )

    fig.suptitle(
        title,
        fontweight="semibold",
        y=0.995,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not MODEL_FILES:
        raise ValueError(
            "MODEL_FILES is empty. Add your model names and CSV paths "
            "near the top of the script."
        )

    metric_names = list(dict.fromkeys(args.metric or DEFAULT_METRICS))

    if args.min_bin_count < 1:
        raise ValueError("--min-bin-count must be at least 1")
    if not 0.0 <= args.histogram_alpha <= 1.0:
        raise ValueError("--histogram-alpha must be between 0 and 1")

    # Load every model.
    model_data: OrderedDict[str, dict] = OrderedDict()
    global_dataset_order: OrderedDict[str, None] = OrderedDict()
    all_selected_t60: list[float] = []

    for raw_path in MODEL_FILES:
        path = Path(raw_path).expanduser().resolve()
        model_name = path.stem.removeprefix("metrics_receiverPos_")

        if not path.is_file():
            raise FileNotFoundError(
                f"Metrics CSV for model {model_name!r} not found: {path}"
            )

        observations, dataset_order, skipped_t60 = load_observations(
            path=path,
            dataset_column_wanted=args.dataset_column,
            t60_column_wanted=args.t60_column,
            metric_names=metric_names,
        )

        available_datasets = list(dataset_order)
        for dataset in available_datasets:
            global_dataset_order.setdefault(dataset, None)

        model_data[model_name] = {
            "path": path,
            "observations": observations,
            "datasets": available_datasets,
            "skipped_t60": skipped_t60,
        }

    datasets = dataset_plot_order(
        select_datasets(list(global_dataset_order), args.dataset)
    )

    # Build one common T60 range/binning for all models so columns are comparable.
    for model_name, data in model_data.items():
        selected = [
            row
            for row in data["observations"]
            if row.dataset in datasets
        ]
        all_selected_t60.extend(row.t60 for row in selected)

    if not all_selected_t60:
        raise ValueError("No observations remain for the selected datasets")

    edges = make_bin_edges(
        np.asarray(all_selected_t60, dtype=np.float64),
        args.bins,
        args.t60_range,
    )

    output_path = args.output.expanduser().resolve()

    plot_all_models(
        model_data=model_data,
        datasets=datasets,
        metric_names=metric_names,
        edges=edges,
        uncertainty=args.uncertainty,
        min_bin_count=args.min_bin_count,
        histogram_alpha=args.histogram_alpha,
        histogram_scope=args.histogram_scope,
        output_path=output_path,
        dpi=args.dpi,
        title=args.title,
    )

    print("Models:")
    for model_name, data in model_data.items():
        counts = OrderedDict((dataset, 0) for dataset in datasets)

        for row in data["observations"]:
            if row.dataset in counts:
                counts[row.dataset] += 1

        print(
            f"  {model_name}: "
            + ", ".join(f"{name}: {count:,}" for name, count in counts.items())
        )

        if data["skipped_t60"]:
            print(
                f"    skipped rows with non-finite T60: "
                f"{data['skipped_t60']:,}"
            )

    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
