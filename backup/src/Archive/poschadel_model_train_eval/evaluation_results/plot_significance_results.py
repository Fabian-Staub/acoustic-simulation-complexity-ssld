#!/usr/bin/env python3
"""Plot the NEW significance-analysis output in the OLD script's matrix style.

Install: python -m pip install numpy matplotlib
Run:     python plot_significance_results.py
Or:      python plot_significance_results.py significance_results/significance_results.json
Select:  python plot_significance_results.py significance_results --metrics LE LR F15 ER15 ESDL

Without an input argument, reads significance_results.json in the working directory.

Writes one A4 portrait PDF per available metric, including the count metrics.
Use --formats pdf png for both formats; --no-annotations hides cell values.
Dataset panels retain the old order: Arni, Motus, Agrid. There is no Summary panel.
Dataset cells show the saved bias-corrected difference (row minus column).
Colours use the saved significant/better_model fields; no tests or corrections
are recomputed. No pooled significance test is performed.
Simulated data and RT60/source-count strata have no pairwise tests to plot.
MUSIC is first on both axes for LE only. It is omitted entirely from all other
metrics. The remaining models retain their original input order. Model-name
components are separated by ordinary hyphens, not en dashes or minus signs.
Saved p-values and significance decisions remain unchanged: excluding MUSIC
from plots does not redo the upstream multiple-comparison correction.

JSON is recommended: it contains prediction_file paths even when model labels
were aliases. The raw prediction files do NOT need to exist. CSV input accepts
pairwise_significance.csv and automatically reads an adjacent results JSON for
metadata, if present. significance_matrix.csv alone has no winner information.

For aliases without filename metadata, use --model-map labels.json, containing
an object such as {"Direct": "SH-O0-L0-C0-R0", "ISM12": "SH-O12-L0-C0-R0"}.
Values can also be original prediction filenames. Unknown names cause an error
instead of silently inventing experiment parameters.

The displacement-token mapping is copied from the old script: 025 -> 0.25 m,
05 -> 0.5 m, 1 -> 1 m, etc. Explicit decimal tokens also work. EM32 is displayed
as EM, following the thesis naming table. No external LaTeX install is needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.monospace"] = ["DejaVu Sans Mono"]

# Math: DejaVu first, Computer Modern only as fallback
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "DejaVu Sans"
matplotlib.rcParams["mathtext.it"] = "DejaVu Sans:italic"
matplotlib.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
matplotlib.rcParams["mathtext.sf"] = "DejaVu Sans"
matplotlib.rcParams["mathtext.tt"] = "DejaVu Sans Mono"
matplotlib.rcParams["mathtext.fallback"] = "cm"

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.patches import Patch, Rectangle
import numpy as np


DISPLACEMENTS = {"025": "0.25", "05": "0.5", "1": "1", "3": "3", "5": "5", "10": "10"}
DATASET_NAMES = {
    "arni-dataset": "Arni", "arni": "Arni",
    "motus-dataset": "Motus", "motus": "Motus",
    "ivan-dataset": "Agrid", "ivan": "Agrid",
    "agrid-dataset": "Agrid", "agrid": "Agrid",
}
EXCLUDED_DATASETS = {"checkpoint_dataset", "checkpoint-dataset", "simulated-dataset"}
REQUIRED_COLUMNS = {
    "dataset", "metric", "model_a", "model_b", "difference_a_minus_b",
    "p_value_display", "significant", "better_model",
}

# Keep only the darker test-dataset name boxes; no light panel backgrounds.
DATASET_BOX_COLORS = {
    "Arni": "#bcd6e8",
    "Motus": "#dfd1eb",
    "Agrid": "#ffd9b7",
}
DEFAULT_DATASET_BOX_COLOR = "#d8dde6"

# Training-dataset palettes from the supplied script (both experiment groups).
MODEL_COLORS = {
    'MUSIC': '#d0d0d0',
    'shoeBox_SH_Direct': '#006b6b',
    'shoeBox_em32_Direct': '#3bc2b9',
    'shoeBox_SH_Direct-Late': '#62938f',
    'shoeBox_em32_Direct-Late': '#a1d6d1',
    'shoeBox_SH_ISMOrder2': '#8a6a00',
    'shoeBox_em32_ISMOrder2': '#e6b710',
    'shoeBox_SH_ISMOrder2-Late': '#a28e4b',
    'shoeBox_em32_ISMOrder2-Late': '#dfcd88',
    'shoeBox_SH_ISMOrder12': '#951010',
    'shoeBox_em32_ISMOrder12': '#ec4e4e',
    'shoeBox_SH_ISMOrder12-Late': '#a96868',
    'shoeBox_em32_ISMOrder12-Late': '#e7a5a5',
    'non-shoeBox_disp-corners-05_SH_ISMOrder12': '#0108C8',
    'non-shoeBox_disp-corners-1_SH_ISMOrder12': '#5C61E0',
    'non-shoeBox_disp-corners-3_SH_ISMOrder12': '#A9ABF6',
    'shoeBox_rand-ism_rand-disp-025_SH_ISMOrder12': '#00B00F',
    'shoeBox_rand-ism_rand-disp-1_SH_ISMOrder12': '#54D861',
    'shoeBox_rand-ism_rand-disp-5_SH_ISMOrder12': '#8FDC97',
    'shoeBox_rand-ism_rand-disp-10_SH_ISMOrder12': '#CCF2D1',
}
MODEL_BACKGROUND_ALPHA = 0.15

# Vertical layout controls.
# Increase/decrease these values to adjust spacing without editing plot code below.
TITLE_Y = 0.992              # Absolute title position; larger = higher on the page.
PLOT_TOP = 0.95              # Top edge of the first matrix; larger = plots move upward.
LEGEND_GAP = 0.028            # Gap between first matrix and legend; larger = more space.
BOTTOM_PAGE_MARGIN = 0.0    # Empty page margin below the final x-axis labels.
BOTTOM_LABEL_PAD_PT = 5       # Extra padding below final x-axis labels, in points.
PANEL_HSPACE = 0.03          # Vertical spacing between dataset matrix panels.
TRAIN_HEADING_GAP = 0.007     # Gap between first matrix and "Train Dataset" heading.



def dataset_name(name: str) -> str:
    return DATASET_NAMES.get(name.lower(), name)


def displacement(token: str) -> str:
    if token in DISPLACEMENTS:
        return DISPLACEMENTS[token]
    if token == "0" or re.fullmatch(r"\d+\.\d+", token):
        return token.rstrip("0").rstrip(".") if "." in token else token
    raise ValueError(f"Unknown displacement token {token!r}; extend DISPLACEMENTS.")


def model_stem(source: str) -> str:
    """Normalize a model ID or filename without losing decimal parameters."""
    # Do not use Path.stem on extensionless labels containing decimal numbers.
    stem = re.split(r"[/\\]", str(source))[-1]
    stem = re.sub(r"\.csv$", "", stem, flags=re.I)
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    stem = re.sub(r"^(?:predictions_|metrics_receiverPos_)", "", stem)
    return stem


def is_music_model(source: str) -> bool:
    return model_stem(source).strip().upper() == "MUSIC"


def thesis_label(source: str) -> str:
    """Use MUSIC for the baseline; use thesis parameter labels for other models."""
    stem = model_stem(source)
    if is_music_model(source):
        return "MUSIC"
    number = r"(?:\d+(?:\.\d+)?)"
    named = re.fullmatch(
        rf"(SH|EM|EM32)-O_?(\d+)-L_?([01])-C_?({number})-R_?({number})",
        stem, flags=re.I,
    )
    if named:
        mic, order, late, corner, random = named.groups()
    else:
        match = re.search(r"(?:^|_)(SH|EM32|EM)_(Direct|ISMOrder(\d+))(-Late)?$", stem, re.I)
        if match is None:
            raise ValueError(f"Cannot derive thesis parameters from {source!r}.")
        mic = match.group(1)
        order = match.group(3) or "0"
        late = "1" if match.group(4) else "0"
        corner_match = re.search(r"non-shoeBox_disp-corners-([^_]+)_", stem, re.I)
        if "non-shoebox" in stem.lower() and corner_match is None:
            raise ValueError(f"Missing corner displacement in {source!r}.")
        random_match = re.search(r"(?:^|_)rand-disp-([^_]+)_", stem, re.I)
        corner = displacement(corner_match.group(1)) if corner_match else "0"
        random = displacement(random_match.group(1)) if random_match else "0"
    mic = "SH" if mic.upper() == "SH" else "EM"
    return f"{mic}-O$_{{{order}}}$-L$_{{{late}}}$-C$_{{{corner}}}$-R$_{{{random}}}$"


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"true", "1"}:
        return True
    if str(value).strip().lower() in {"false", "0"}:
        return False
    raise ValueError(f"Invalid significance flag: {value!r}")


def as_float(value: object) -> float:
    return math.nan if value is None or value == "" else float(value)


def load_results(path: Path) -> tuple[list[dict], dict]:
    if path.is_dir():
        json_path = path / "significance_results.json"
        path = json_path if json_path.exists() else path / "pairwise_significance.csv"
    payload = {}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload.get("pairwise_comparisons", [])
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        sibling = path.with_name("significance_results.json")
        if sibling.exists():
            payload = json.loads(sibling.read_text(encoding="utf-8-sig"))
    else:
        raise ValueError("Input must be a results directory, results JSON, or pairwise CSV.")
    cleaned = []
    for row in rows:
        missing = REQUIRED_COLUMNS - row.keys()
        if missing:
            raise ValueError(f"Missing fields {sorted(missing)}. Use pairwise_significance.csv or significance_results.json.")
        if row["dataset"].lower() in EXCLUDED_DATASETS:
            continue
        if row["metric"] != "LE" and any(
            is_music_model(row[key]) for key in ("model_a", "model_b")
        ):
            continue
        row = dict(row)
        row["significant"] = as_bool(row["significant"])
        for key in ("p_value_display", "difference_a_minus_b"):
            row[key] = as_float(row[key])
        cleaned.append(row)
    if not cleaned:
        raise ValueError("No pairwise significance comparisons found.")
    return cleaned, payload


def model_labels(models: list[str], payload: dict, overrides: dict) -> list[str]:
    files = {}
    for dataset in payload.get("datasets", {}).values():
        for model, entry in dataset.get("models", {}).items():
            if entry.get("prediction_file"):
                files.setdefault(model, entry["prediction_file"])
    labels = []
    for model in models:
        candidates = [overrides[model]] if model in overrides else [files.get(model, model), model]
        for candidate in candidates:
            try:
                label = thesis_label(candidate)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Cannot label model {model!r}; supply its filename or thesis name through --model-map.")
        labels.append(label)
    if len(set(labels)) != len(labels):
        raise ValueError("Multiple models map to the same thesis label; check model filenames/overrides.")
    return labels


def ordered_models(rows: list[dict]) -> list[str]:
    """MUSIC first, followed by the other models in their original input order."""
    models = list(dict.fromkeys(
        row[key] for row in rows for key in ("model_a", "model_b")
    ))
    return sorted(models, key=lambda model: not is_music_model(model))


def models_for_metric(rows: list[dict], metric: str, model_order: list[str]) -> list[str]:
    """Use only models present in the applicable comparisons for this metric."""
    participating = {
        row[key] for row in rows if row["metric"] == metric
        for key in ("model_a", "model_b")
    }
    return [model for model in model_order if model in participating
            and (metric == "LE" or not is_music_model(model))]


def training_shades(labels: list[str]) -> list[np.ndarray]:
    """Match model colours by thesis label, including custom model aliases.

    Mix with white explicitly to match the reference's 15% training shades
    consistently on both row and column labels.
    """
    by_label = {thesis_label(model): color for model, color in MODEL_COLORS.items()}
    shades = []
    for i, label in enumerate(labels):
        color = by_label.get(label, plt.get_cmap("tab20")(i % 20))
        rgb = np.asarray(to_rgb(color))
        shades.append(MODEL_BACKGROUND_ALPHA * rgb + (1 - MODEL_BACKGROUND_ALPHA))
    return shades


def square_matrices(rows: list[dict], dataset: str, metric: str,
                    models: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return paired difference and direction matrices for the supplied models."""
    indices = {model: i for i, model in enumerate(models)}
    direction = np.full((len(models), len(models)), np.nan)
    difference = np.full_like(direction, np.nan)
    seen = set()
    for row in rows:
        if (row["dataset"], row["metric"]) != (dataset, metric):
            continue
        a, b = row["model_a"], row["model_b"]
        if a not in indices or b not in indices or a == b:
            raise ValueError(f"Invalid model pair: {a!r}, {b!r}")
        if metric != "LE" and (is_music_model(a) or is_music_model(b)):
            continue
        pair = frozenset((a, b))
        if pair in seen:
            raise ValueError(f"Duplicate comparison: {dataset}, {metric}, {a} vs {b}")
        seen.add(pair)
        i, j = indices[a], indices[b]
        delta, p = row["difference_a_minus_b"], row["p_value_display"]
        difference[i, j], difference[j, i] = delta, -delta
        if not math.isfinite(p):
            if row["significant"]:
                raise ValueError("A significant comparison has a missing/nonfinite p-value.")
            continue
        if not 0 <= p <= 1:
            raise ValueError(f"Invalid p-value: {p}")
        if row["significant"]:
            if row["better_model"] not in (a, b):
                raise ValueError(f"Significant comparison has no valid winner: {a} vs {b}")
            direction[i, j] = 1 if row["better_model"] == a else -1
            direction[j, i] = -direction[i, j]
        else:
            direction[i, j] = direction[j, i] = 0
    active_count = sum(metric == "LE" or not is_music_model(model) for model in models)
    expected = active_count * (active_count - 1) // 2
    if len(seen) != expected:
        raise ValueError(f"{dataset}, {metric}: expected {expected} model pairs, found {len(seen)}.")
    return direction, difference


def metric_title(metric: str) -> str:
    # Match the reference titles. Its ESDL upward arrow is corrected here:
    # the supplied analysis defines ESDL as a lower-is-better error score.
    return {"LE": "Localization Error (LE) ↓",
            "LR": "Localization Recall (LR) ↑",
            "ER15": r"Location-Sensitive Error Rate ($\mathbf{ER}_{\mathbf{15}}$) ↓",
            "F15": r"Location-Sensitive F-Score ($\mathbf{F}_{\mathbf{15}}$) ↑",
            "ESDL": r"$\mathbf{E}_{\mathbf{SDL}}\mathbf{-Score}$ ↓",
            "count_MAE": "Source-count MAE",
            "count_accuracy": "Source-count accuracy"}.get(metric, metric)


def plot_metric(rows: list[dict], datasets: list[str], metric: str,
                models: list[str], labels: list[str], output_dir: Path,
                formats: list[str], dpi: int, annotations: bool) -> list[Path]:
    """Plot dataset panels only, with label widths measured from rendered text."""
    matrices = [square_matrices(rows, dataset, metric, models) for dataset in datasets]
    n = len(models)
    label_shades = training_shades(labels)
    # Preserve the previous dataset-cell colours, without the unused Summary scale.
    colors = [plt.get_cmap("Oranges")(.30), "#e5e7eb", plt.get_cmap("Greens")(.30)]
    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    norm = BoundaryNorm([ -1.5, -.5, .5, 1.5 ], cmap.N)
    fig, axes = plt.subplots(len(matrices), 1, figsize=(8.27, 11.69), squeeze=False)
    tick_size = max(4.5, min(7.6, 10.4 - .18*n))
    cell_size = max(4.2, min(7.2, 104 / n))
    fig.subplots_adjust(left=.270, right=.977, top=PLOT_TOP, bottom=.160, hspace=PANEL_HSPACE)
    dataset_texts = []
    for k, (axis, (direction, difference)) in enumerate(zip(axes.flat, matrices)):
        axis.imshow(direction, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        axis.set_yticks(range(n), labels, fontsize=tick_size, fontfamily="monospace")
        axis.set_xticks(range(n))
        if k == len(datasets) - 1:
            axis.set_xticklabels(labels, rotation=90, ha="center", va="top",
                                 fontsize=tick_size, fontfamily="monospace")
        else:
            axis.set_xticklabels([])
        axis.set_xticks(np.arange(-.5, n), minor=True)
        axis.set_yticks(np.arange(-.5, n), minor=True)
        axis.grid(which="minor", axis="x", color="white", linewidth=.45)
        axis.grid(which="minor", axis="y", color="white", linewidth=.45)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.tick_params(which="major", length=2, pad=3)
        if annotations:
            for i, j in np.ndindex(difference.shape):
                value = difference[i, j]
                if math.isfinite(value):
                    axis.text(j, i, f"{value:+.2g}", ha="center", va="center",
                              fontsize=cell_size, color="#111827")
        dataset_texts.append(fig.text(
            .06, .5, dataset_name(datasets[k]), rotation=90,
            ha="center", va="center", fontsize=8.0, weight="bold",
        ))

    train_heading = fig.text(.17, .925, "Train Dataset", ha="center", va="bottom",
                             fontsize=8, weight="bold")
    test_heading = fig.text(.023, .55, "Test Dataset", rotation=90,
                            ha="center", va="center", fontsize=9, weight="bold")
    fig.suptitle(metric_title(metric), y=TITLE_Y, fontsize=14, weight="bold")
    legend = fig.legend(handles=[Patch(facecolor=colors[2], label="Row better (sig.)"),
                        Patch(facecolor=colors[0], label="Row worse (sig.)"),
                        Patch(facecolor=colors[1], label="Not significant"),
                        Patch(facecolor="white", edgecolor="#aaa", label="Diagonal / unavailable"),
                        Patch(facecolor="none", edgecolor="none",
                              label=r"Cell value = $\widehat{\Delta}_{\mathrm{row,col,jack}}$")],
               loc="upper center", bbox_to_anchor=(.5, .962), ncol=5,
               frameon=False, fontsize=7, handlelength=1.1,
               handletextpad=.4, columnspacing=1.0)

    # Render once to measure actual label extents. Both the test-dataset strip
    # and training-label column follow their widest rendered name, not a fixed
    # page fraction. Text length determines height for vertically rotated labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    xpoint, ypoint = fig.dpi / 72 / width, fig.dpi / 72 / height
    test_width = test_heading.get_window_extent(renderer).width / width
    strip_width = max(t.get_window_extent(renderer).width for t in dataset_texts) / width + 6*xpoint
    train_width = max(
        max(t.get_window_extent(renderer).width for t in axes[0, 0].get_yticklabels()),
        train_heading.get_window_extent(renderer).width,
    ) / width
    outer_left = .005
    strip_left = outer_left + test_width + 6*xpoint
    strip_right = strip_left + strip_width
    label_left = strip_right + 5*xpoint
    matrix_left = label_left + train_width + 6*xpoint
    if matrix_left > .55:
        plt.close(fig)
        raise ValueError("Dataset labels are too wide for an A4 plot; shorten the display labels.")

    bottom_axis = axes[-1, 0]
    old_bottom = bottom_axis.get_position().y0
    xlabel_extent = old_bottom - min(
        label.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y0
        for label in bottom_axis.get_xticklabels()
    )
    # Use the area formerly occupied by Summary and its footer for taller rows.
    bottom_margin = BOTTOM_PAGE_MARGIN + xlabel_extent + BOTTOM_LABEL_PAD_PT*ypoint
    fig.subplots_adjust(left=matrix_left, right=0.995, top=PLOT_TOP,
                        bottom=bottom_margin, hspace=PANEL_HSPACE)
    top_box, bottom_box = axes[0, 0].get_position(), bottom_axis.get_position()
    # Centre on the final matrix area, excluding the left dataset-label columns.
    legend.set_bbox_to_anchor(((top_box.x0 + top_box.x1)/2, top_box.y1 + LEGEND_GAP),
                              transform=fig.transFigure)
    train_heading.set_position(((label_left + matrix_left)/2, top_box.y1 + TRAIN_HEADING_GAP))
    test_heading.set_position((outer_left + test_width/2, (top_box.y1 + bottom_box.y0)/2))

    for axis, dataset, text in zip(axes.flat, datasets, dataset_texts):
        box = axis.get_position()
        strip_color = DATASET_BOX_COLORS.get(dataset_name(dataset), DEFAULT_DATASET_BOX_COLOR)
        fig.add_artist(Rectangle(
            (strip_left, box.y0), strip_width, box.height,
            transform=fig.transFigure, facecolor=strip_color,
            edgecolor="none", zorder=-9,
        ))
        text.set_position(((strip_left + strip_right)/2, (box.y0 + box.y1)/2))
        # One training shade per row; keep the test-dataset strip unchanged.
        for i, shade in enumerate(label_shades):
            fig.add_artist(Rectangle(
                (label_left, box.y1 - (i + 1) * box.height / n),
                box.x0 - label_left, box.height / n,
                transform=fig.transFigure, facecolor=shade,
                edgecolor="none", zorder=-8,
            ))

    # The same training shades follow each model's column below the last matrix.
    for i, shade in enumerate(label_shades):
        fig.add_artist(Rectangle(
            (bottom_box.x0 + i * bottom_box.width / n,
             bottom_box.y0 - xlabel_extent - 4*ypoint),
            bottom_box.width / n, xlabel_extent + 4*ypoint,
            transform=fig.transFigure, facecolor=shade,
            edgecolor="none", zorder=-8,
        ))
    # Keep long small-decimal annotations inside their cells. Most values use
    # the larger cell_size; only labels that would touch neighbours are reduced.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for axis in axes.flat:
        cell_width, cell_height = axis.bbox.width / n, axis.bbox.height / n
        for text in axis.texts:
            bounds = text.get_window_extent(renderer)
            scale = min(1.0, .90 * cell_width / bounds.width,
                        .85 * cell_height / bounds.height)
            if scale < 1.0:
                text.set_fontsize(text.get_fontsize() * scale)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", metric).strip("_") or "metric"
    paths = []
    try:
        for extension in formats:
            path = output_dir / f"significance_matrices_a4_{safe}.{extension}"
            # No tight bounding box: preserve the A4 page dimensions.
            fig.savefig(path, dpi=dpi, facecolor="white")
            paths.append(path)
    finally:
        plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("significance_results.json"),
        help="Results directory, results JSON, or pairwise CSV",
    )
    parser.add_argument("--output-dir", type=Path, help="Default: <input directory>/plots")
    parser.add_argument("--metrics", nargs="+", help="Default: every metric with pairwise results")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png", "svg"), default=["pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-annotations", action="store_true")
    parser.add_argument("--model-map", type=Path, help="JSON object mapping model IDs to filenames/thesis names")
    args = parser.parse_args()
    try:
        if args.dpi <= 0:
            raise ValueError("--dpi must be positive.")
        rows, payload = load_results(args.input)
        models = ordered_models(rows)
        overrides = json.loads(args.model_map.read_text(encoding="utf-8-sig")) if args.model_map else {}
        if not isinstance(overrides, dict) or any(not isinstance(v, str) for v in overrides.values()):
            raise ValueError("--model-map must contain a JSON object with string values.")
        datasets = list(dict.fromkeys(row["dataset"] for row in rows))
        priority = {"Arni": 0, "Motus": 1, "Agrid": 2}
        datasets.sort(key=lambda dataset: priority.get(dataset_name(dataset), 3))
        available = list(dict.fromkeys(row["metric"] for row in rows))
        metrics = list(dict.fromkeys(args.metrics or available))
        if any(metric not in available for metric in metrics):
            raise ValueError(f"Unknown metric selection; available metrics: {', '.join(available)}")
        # Only applicable models appear on each metric; MUSIC stays first in LE.
        metric_models = {}
        metric_labels = {}
        for metric in metrics:
            metric_models[metric] = models_for_metric(rows, metric, models)
            metric_labels[metric] = model_labels(metric_models[metric], payload, overrides)
            for dataset in datasets:
                square_matrices(rows, dataset, metric, metric_models[metric])
        output = args.output_dir or ((args.input if args.input.is_dir() else args.input.parent) / "plots")
        output.mkdir(parents=True, exist_ok=True)
        for metric in metrics:
            for path in plot_metric(rows, datasets, metric, metric_models[metric], metric_labels[metric], output,
                                    list(dict.fromkeys(args.formats)), args.dpi, not args.no_annotations):
                print(f"Saved: {path.resolve()}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
