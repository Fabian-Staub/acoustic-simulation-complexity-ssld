#!/usr/bin/env python3
r"""Generate the colored Overleaf table directly from model_metrics.csv.

Python 3.9+; standard library only. No template file or third-party packages.

Usage:
    python generate_evaluation_table.py "model_metrics(3).csv" -o evaluation_results_overleaf.txt
    python generate_evaluation_table.py model_metrics.csv -o results.tex --decimals 3
    python generate_evaluation_table.py model_metrics.csv --value-column estimate

The output is a LaTeX fragment, not a standalone document. Add these packages
to your existing Overleaf preamble, then paste the generated fragment:
    \usepackage{array,booktabs,longtable,graphicx,xcolor}
Alternatively, upload the output and use \input{evaluation_results_overleaf.txt}.
Replace the old fragment; do not include both sets of macro definitions.

Numeric convention: full_sample +/- standard_error (not standard deviation or
confidence-interval half-width). --value-column estimate selects the alternative
CSV estimate column. LE stays in degrees; other metrics retain their CSV units.
Totals come from gt_source_count=all; they are NOT averages of sources 1/2/3.

Bold results are best across displayed models for each dataset and metric,
separately for Total and source counts 1, 2, and 3. LE, ER15 and ESDL minimize;
LR and F15 maximize. Ranking uses unrounded values from --value-column, not
standard errors. All exact ties are bold; missing and hidden cells are excluded.

Display rules below match the supplied template: ivan-dataset is labeled Agrid,
Agrid source 3 is hidden, and MUSIC shows LE only. These are presentation choices,
even when the CSV contains values in the hidden cells. Change DATASETS or
MUSIC_METRICS below to display them. All twenty configured models are included.

Missing/non-finite values raise an error by default. --allow-missing substitutes
-- and reports the missing cells. Genuine zero values remain numeric zeros.
Duplicate keys, malformed numbers and unknown model/dataset names always fail.
"""

import argparse
import csv
import hashlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


# ------------------------- Editable configuration -------------------------

METRICS = ("LE", "LR", "ER15", "F15", "ESDL")
METRIC_DIRECTIONS = {"LE": "min", "LR": "max", "ER15": "min", "F15": "max", "ESDL": "min"}
MUSIC_METRICS = {"LE"}  # Set to set(METRICS) to show all MUSIC metrics.
MODEL_SHADE_PERCENT = 15  # Equivalent to alpha=0.15 composited over white.
DATASET_HIGHLIGHT_PERCENT = 25


@dataclass(frozen=True)
class Model:
    name: str
    color: str
    hex_color: str


MODELS = (
    Model("MUSIC", "ModelMUSIC", "D0D0D0"),
    Model("shoeBox_SH_Direct", "ModelSHZero", "006B6B"),
    Model("shoeBox_em32_Direct", "ModelEMZero", "3BC2B9"),
    Model("shoeBox_SH_Direct-Late", "ModelSHZeroLate", "62938F"),
    Model("shoeBox_em32_Direct-Late", "ModelEMZeroLate", "A1D6D1"),
    Model("shoeBox_SH_ISMOrder2", "ModelSHTwo", "8A6A00"),
    Model("shoeBox_em32_ISMOrder2", "ModelEMTwo", "E6B710"),
    Model("shoeBox_SH_ISMOrder2-Late", "ModelSHTwoLate", "A28E4B"),
    Model("shoeBox_em32_ISMOrder2-Late", "ModelEMTwoLate", "DFCD88"),
    Model("shoeBox_SH_ISMOrder12", "ModelSHTwelve", "951010"),
    Model("shoeBox_em32_ISMOrder12", "ModelEMTwelve", "EC4E4E"),
    Model("shoeBox_SH_ISMOrder12-Late", "ModelSHTwelveLate", "A96868"),
    Model("shoeBox_em32_ISMOrder12-Late", "ModelEMTwelveLate", "E7A5A5"),
    Model("non-shoeBox_disp-corners-05_SH_ISMOrder12", "ModelCornerHalf", "0108C8"),
    Model("non-shoeBox_disp-corners-1_SH_ISMOrder12", "ModelCornerOne", "5C61E0"),
    Model("non-shoeBox_disp-corners-3_SH_ISMOrder12", "ModelCornerThree", "A9ABF6"),
    Model("shoeBox_rand-ism_rand-disp-025_SH_ISMOrder12", "ModelRandQuarter", "00B00F"),
    Model("shoeBox_rand-ism_rand-disp-1_SH_ISMOrder12", "ModelRandOne", "54D861"),
    Model("shoeBox_rand-ism_rand-disp-5_SH_ISMOrder12", "ModelRandFive", "8FDC97"),
    Model("shoeBox_rand-ism_rand-disp-10_SH_ISMOrder12", "ModelRandTen", "CCF2D1"),
)


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str
    color: str
    hex_color: str
    sources: Tuple[str, ...]


DATASETS = (
    Dataset("simulated-dataset", "Simulated", "DataSimulated", "2CA02C", ("1", "2", "3")),
    Dataset("arni-dataset", "Arni", "DataArni", "1F77B4", ("1", "2", "3")),
    Dataset("motus-dataset", "Motus", "DataMotus", "9467BD", ("1", "2", "3")),
    # The template calls ivan-dataset Agrid. Add "3" here to show source 3.
    Dataset("ivan-dataset", "Agrid", "DataAgrid", "FF7F0E", ("1", "2")),
)
DATASET_ALIASES = {
    "checkpoint_dataset": "simulated-dataset",
    "agrid-dataset": "ivan-dataset",
}
DISPLACEMENTS = {"025": "0.25", "05": "0.5", "1": "1", "3": "3", "5": "5", "10": "10"}


# ---------------------------- LaTeX template -----------------------------

LATEX_DEFINITIONS = r"""
\newcommand{\EvalResult}[2]{\(#1\)}
\newcommand{\EvalBestResult}[2]{{\boldmath\EvalResult{#1}{#2}}}
\newcommand{\EvalSources}[3]{%
  \begin{tabular}[c]{@{}c@{}}
    #1\\#2\\#3
  \end{tabular}%
}

\newlength{\EvalTrainWidth}
\newlength{\EvalTestWidth}
\newlength{\EvalMetricWidth}
\newcolumntype{E}{w{c}{\EvalMetricWidth}}

\newenvironment{EvalGrid}{%
  \begin{tabular}[c]{@{}w{l}{\EvalTestWidth}*{10}{E}@{}}%
}{\end{tabular}}

\newcommand{\DatasetName}[2]{%
  \begingroup
  \setlength{\fboxsep}{1.1pt}%
  \colorbox{#1!\DatasetHighlightPercent!white}{%
    \textcolor{black}{\strut #2}%
  }%
  \endgroup
}

\newcommand{\SourceHeading}{%
  \begin{tabular}[c]{@{}c@{}}
    \textbf{Sources}\\1 / 2 / 3
  \end{tabular}%
}

% Shared baseline for Train and Test.
\newcommand{\EvalHeader}{%
  \begin{tabular}[c]{@{}w{c}{\EvalTrainWidth}@{}w{l}{\EvalTestWidth}c@{}}
    \textbf{Train-/} & \textbf{Test-Dataset} &
    \begin{tabular}[c]{@{}*{10}{E}@{}}
      \multicolumn{2}{@{}c}{\(\mathrm{LE}\,[^\circ]\downarrow\)}
      & \multicolumn{2}{c}{\(\mathrm{LR}\uparrow\)}
      & \multicolumn{2}{c}{\(\mathrm{ER}_{15^\circ}\downarrow\)}
      & \multicolumn{2}{c}{\(\mathrm{F}_{15^\circ}\uparrow\)}
      & \multicolumn{2}{c@{}}{\(\mathcal{E}_{\mathrm{SDL}}\downarrow\)}
      \\
      \cmidrule(lr){1-2}
      \cmidrule(lr){3-4}
      \cmidrule(lr){5-6}
      \cmidrule(lr){7-8}
      \cmidrule(lr){9-10}
      \textbf{Total} & \SourceHeading
      & \textbf{Total} & \SourceHeading
      & \textbf{Total} & \SourceHeading
      & \textbf{Total} & \SourceHeading
      & \textbf{Total} & \SourceHeading
      \\
    \end{tabular}%
  \end{tabular}%
  \\
}

% Arguments: model color, model label, four populated dataset rows.
% One continuous background and one indivisible longtable row per model.
\newcommand{\ModelBlock}[3]{%
  \begingroup
  \setlength{\fboxsep}{0pt}%
  \colorbox{#1!\ModelShadePercent!white}{%
    \begin{tabular}[c]{@{}w{c}{\EvalTrainWidth}@{}c@{}}
      \begin{tabular}[c]{@{}c@{}}
        \rotatebox[origin=c]{90}{\fontsize{8}{9}\selectfont #2}%
      \end{tabular}%
      &
      \begin{EvalGrid}
        #3%
      \end{EvalGrid}%
    \end{tabular}%
  }%
  \endgroup
  \\
}
"""

TABLE_START = r"""
\begingroup
\fontsize{5.5}{6.5}\selectfont
\setlength{\tabcolsep}{1.3pt}
\renewcommand{\arraystretch}{1.05}
\setlength{\EvalTrainWidth}{0.05\textwidth}
\setlength{\EvalTestWidth}{0.085\textwidth}
\setlength{\EvalMetricWidth}{%
  \dimexpr(\textwidth-\EvalTrainWidth-\EvalTestWidth-20\tabcolsep)/10\relax}
\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}
\setlength{\LTcapwidth}{\textwidth}

\begin{longtable}{@{}c@{}}
\caption{}
\label{tab:evaluation_results}\\
\toprule
\EvalHeader
\midrule
\endfirsthead

\multicolumn{1}{c}{\tablename~\thetable{} -- continued}\\[1ex]
\toprule
\EvalHeader
\midrule
\endhead

\midrule
\multicolumn{1}{r@{}}{Continued on next page}\\
\endfoot

\bottomrule
\endlastfoot
"""

TABLE_END = "\\end{longtable}\n\\endgroup\n"


# ---------------------------- CSV processing -----------------------------

Key = Tuple[str, str, str, str]  # dataset, model, source, metric


@dataclass(frozen=True)
class Measurement:
    value: Optional[float]
    standard_error: Optional[float]
    line: int


def canonical_model(name: str) -> str:
    """Accept the same eval_ prefix / (run-number) suffix as the plot script."""
    if name.startswith("eval_"):
        name = name[5:]
    return re.sub(r"\(\d+\)$", "", name)


def parse_number(text: str, column: str, line: int) -> Optional[float]:
    if text.lower() in {"", "nan", "na", "n/a", "none", "null"}:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"CSV line {line}: invalid {column}={text!r}") from exc
    if not math.isfinite(value):
        return None
    if column == "standard_error" and value < 0:
        raise ValueError(f"CSV line {line}: standard_error must not be negative")
    return value


def read_metrics(path: Path, value_column: str = "full_sample") -> Dict[Key, Measurement]:
    required = {"dataset", "model", "gt_source_count", "metric", value_column, "standard_error"}
    known_models = {m.name for m in MODELS}
    known_datasets = {d.key for d in DATASETS}
    index = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("The CSV is empty")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("The CSV has duplicate column headings")
        missing_columns = required.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing_columns)))
        for raw in reader:
            line = reader.line_num
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"CSV line {line}: wrong number of fields")
            row = {key: value.strip() for key, value in raw.items()}
            if not row["metric"]:
                raise ValueError(f"CSV line {line}: empty metric")
            if row["metric"] not in METRICS:
                continue  # The CSV also contains count_MAE and count_accuracy.
            model = canonical_model(row["model"])
            dataset = DATASET_ALIASES.get(row["dataset"], row["dataset"])
            source = row["gt_source_count"]
            if model not in known_models:
                raise ValueError(f"CSV line {line}: unknown model {model!r}; add it to MODELS")
            if dataset not in known_datasets:
                raise ValueError(f"CSV line {line}: unknown dataset {dataset!r}; update DATASETS")
            if source not in {"all", "1", "2", "3"}:
                raise ValueError(f"CSV line {line}: unexpected gt_source_count={source!r}")
            key = (dataset, model, source, row["metric"])
            if key in index:
                raise ValueError(
                    f"Duplicate key {key!r} at CSV lines {index[key].line} and {line}. "
                    "Filter to one run before generating the table."
                )
            index[key] = Measurement(
                parse_number(row[value_column], value_column, line),
                parse_number(row["standard_error"], "standard_error", line),
                line,
            )
    if not index:
        raise ValueError("No rows were found for the five table metrics")
    return index


def model_label(name: str) -> str:
    if name == "MUSIC":
        return r"\texttt{MUSIC}"
    match = re.search(r"_(SH|em32)_(Direct|ISMOrder(\d+))(-Late)?$", name)
    if match is None:
        raise ValueError(f"Cannot construct the compact label for {name!r}")
    mic = "SH" if match.group(1) == "SH" else "EM"
    order = match.group(3) or "0"
    late = "1" if match.group(4) else "0"
    corner_match = re.search(r"disp-corners-([^_]+)_", name)
    random_match = re.search(r"rand-disp-([^_]+)_", name)
    try:
        corner = DISPLACEMENTS[corner_match.group(1)] if corner_match else "0"
        random = DISPLACEMENTS[random_match.group(1)] if random_match else "0"
    except KeyError as exc:
        raise ValueError(f"Unknown displacement token in {name!r}; update DISPLACEMENTS") from exc
    return (
        rf"\(\texttt{{{mic}-O}}_{{{order}}}\texttt{{-L}}_{{{late}}}"
        rf"\texttt{{-C}}_{{{corner}}}\texttt{{-R}}_{{{random}}}\)"
    )


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(c, c) for c in text)


def format_number(value: float, decimals: int) -> str:
    formatted = f"{value:.{decimals}f}"
    # Avoid displaying -0.00 for negative values rounded to zero.
    return formatted[1:] if formatted.startswith("-") and float(formatted) == 0 else formatted


def cell_is_displayed(model: Model, dataset: Dataset, source: str, metric: str) -> bool:
    return ((model.name != "MUSIC" or metric in MUSIC_METRICS)
            and (source == "all" or source in dataset.sources))


def best_result_keys(index: Dict[Key, Measurement]) -> Set[Key]:
    """Rank complete, visible results across models, within each table group."""
    winners: Set[Key] = set()
    for dataset in DATASETS:
        for metric in METRICS:
            direction = METRIC_DIRECTIONS.get(metric)
            if direction not in {"min", "max"}:
                raise ValueError(f"Set METRIC_DIRECTIONS[{metric!r}] to 'min' or 'max'")
            for source in ("all", "1", "2", "3"):
                candidates = {}
                for model in MODELS:
                    if not cell_is_displayed(model, dataset, source, metric):
                        continue
                    key = (dataset.key, model.name, source, metric)
                    record = index.get(key)
                    if record is not None and record.value is not None and record.standard_error is not None:
                        candidates[key] = record.value
                if candidates:
                    best = (min if direction == "min" else max)(candidates.values())
                    winners.update(key for key, value in candidates.items() if value == best)
    return winners


def generate_table(index: Dict[Key, Measurement], *, decimals: int = 2,
                   allow_missing: bool = False, value_column: str = "full_sample",
                   caption: str = "") -> Tuple[str, dict]:
    if not 0 <= decimals <= 6:
        raise ValueError("decimals must be between 0 and 6")
    missing = []
    winners = best_result_keys(index)
    report = {"numeric_cells": 0, "omitted_cells": 0, "missing_cells": missing,
              "highlighted_cells": 0}

    def cell(model: Model, dataset: Dataset, source: str, metric: str) -> str:
        if not cell_is_displayed(model, dataset, source, metric):
            report["omitted_cells"] += 1
            return "--"
        key = (dataset.key, model.name, source, metric)
        record = index.get(key)
        if record is None or record.value is None or record.standard_error is None:
            reason = "row absent" if record is None else f"missing/non-finite value or SE (line {record.line})"
            missing.append(f"{key!r}: {reason}")
            return "--"
        report["numeric_cells"] += 1
        command = r"\EvalResult"
        if key in winners:
            command = r"\EvalBestResult"
            report["highlighted_cells"] += 1
        return (command + "{" + format_number(record.value, decimals) + "}{"
                + format_number(record.standard_error, decimals) + "}")

    blocks = []
    for model in MODELS:
        rows = []
        for dataset in DATASETS:
            row = [rf"  \DatasetName{{{dataset.color}}}{{{latex_escape(dataset.label)}}}"]
            for metric in METRICS:
                total = cell(model, dataset, "all", metric)
                sources = [cell(model, dataset, source, metric) for source in ("1", "2", "3")]
                stack = r"\EvalSources" + "".join("{" + value + "}" for value in sources)
                row.append(f"  & {total} & {stack} % {metric}")
            row.append(r"  \\")
            rows.append("\n".join(row))
        body = "\n  \\cmidrule(l){1-11}\n".join(rows)
        blocks.append(
            f"% {model.name}\n"
            rf"\ModelBlock{{{model.color}}}{{{model_label(model.name)}}}{{%" + "\n"
            + body + "\n}\n"
        )

    if missing and not allow_missing:
        examples = "\n".join(missing[:8])
        raise ValueError(
            f"{len(missing)} requested numeric cells are missing.\n{examples}\n"
            "Use --allow-missing to emit -- for those cells. No output was written."
        )

    preamble = [
        "% Generated by generate_evaluation_table.py; replace the old table fragment.",
        "% Required preamble: \\usepackage{array,booktabs,longtable,graphicx,xcolor}",
        f"% Numeric convention: {value_column}, {decimals} decimal places.",
        "% LE remains in degrees. Totals use gt_source_count=all directly; no averaging or clipping.",
        "% Bold: best across displayed models per dataset/metric, separately for Total and sources 1/2/3.",
        "% Rank unrounded values; all exact ties are bold. Missing and hidden cells are excluded.",
        "% Metric directions: " + ", ".join(f"{m}={METRIC_DIRECTIONS[m]}" for m in METRICS) + ".",
        "% Dataset keys, display names and displayed sources:",
    ]
    for dataset in DATASETS:
        preamble.append(f"% {dataset.key} -> {dataset.label}; sources: {', '.join(dataset.sources)}")
    preamble.append("% MUSIC metrics displayed: " + ", ".join(m for m in METRICS if m in MUSIC_METRICS))
    preamble.append(f"% Numeric cells: {report['numeric_cells']}; intentionally omitted: {report['omitted_cells']}; missing: {len(missing)}")
    preamble.append(f"% Highlighted best results: {report['highlighted_cells']}")
    if missing:
        preamble.append("% WARNING: -- also denotes unavailable input values (see console report).")
    preamble.extend(rf"\definecolor{{{m.color}}}{{HTML}}{{{m.hex_color}}}" for m in MODELS)
    preamble.extend(rf"\definecolor{{{d.color}}}{{HTML}}{{{d.hex_color}}}" for d in DATASETS)
    preamble.extend([
        rf"\newcommand{{\ModelShadePercent}}{{{MODEL_SHADE_PERCENT}}}",
        rf"\newcommand{{\DatasetHighlightPercent}}{{{DATASET_HIGHLIGHT_PERCENT}}}",
    ])
    start = TABLE_START.replace(r"\caption{}", r"\caption{" + latex_escape(caption) + "}")
    latex = ("\n".join(preamble) + "\n" + LATEX_DEFINITIONS + start + "\n"
             + "\n\\midrule\n\n".join(blocks) + "\n" + TABLE_END)
    return latex, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=Path("model_metrics.csv"),
        help="Input model_metrics CSV (default: model_metrics.csv)",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("evaluation_results_overleaf.txt"))
    parser.add_argument("--decimals", type=int, default=2, choices=range(7))
    parser.add_argument("--value-column", choices=("full_sample", "estimate"), default="full_sample")
    parser.add_argument("--allow-missing", action="store_true", help="Use -- instead of failing on missing numeric cells")
    parser.add_argument("--caption", default="", help="Optional plain-text table caption; blank preserves the supplied template")
    args = parser.parse_args(argv)
    try:
        if args.csv_path.resolve() == args.output.resolve():
            raise ValueError("Output must not overwrite the input CSV")
        if args.output.resolve() == Path(__file__).resolve():
            raise ValueError("Output must not overwrite this Python script")
        index = read_metrics(args.csv_path, args.value_column)
        text, report = generate_table(index, decimals=args.decimals, allow_missing=args.allow_missing,
                                      value_column=args.value_column, caption=args.caption)
        # Hash the input so the generated fragment can be tied to its source CSV.
        digest = hashlib.sha256(args.csv_path.read_bytes()).hexdigest()
        text = f"% Input CSV SHA-256: {digest}\n" + text
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Wrote {args.output}: {len(MODELS)} models, {len(DATASETS)} datasets, {len(METRICS)} metrics.")
    print(f"{report['numeric_cells']} numeric entries; {report['omitted_cells']} intentionally omitted; "
          f"{len(report['missing_cells'])} missing.")
    print(f"{report['highlighted_cells']} best results highlighted in bold (including exact ties).")
    for issue in report["missing_cells"][:10]:
        print("Warning: " + issue, file=sys.stderr)
    if len(report["missing_cells"]) > 10:
        print(f"Warning: {len(report['missing_cells']) - 10} additional missing entries.", file=sys.stderr)
    if args.decimals > 2:
        print("Note: extra decimal places may require wider table columns.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
