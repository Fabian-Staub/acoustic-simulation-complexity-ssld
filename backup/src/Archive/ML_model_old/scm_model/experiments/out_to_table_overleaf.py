import csv
import numpy as np

FILE = "eval_model_test.csv"
OUT_FILE = "filled_table.txt"

# -----------------------------
# Metrics
# -----------------------------
metrics = [
    "Error Rate",
    "F-Score",
    "Localization Error",
    "Localization Recall",
    "SELD"
]

metric_map = {
    "ER": "Error Rate",
    "F": "F-Score",
    "LE": "Localization Error",
    "LR": "Localization Recall",
    "SELD": "SELD"
}

split_map = {
    "all": "total",
    "1_source": "1",
    "2_sources": "2",
    "3_sources": "3"
}

# -----------------------------
# Direction
# -----------------------------
higher_is_better = {
    "Error Rate": False,
    "F-Score": True,
    "Localization Error": False,
    "Localization Recall": True,
    "SELD": False
}

# -----------------------------
# Storage
# -----------------------------
def init_storage():
    return {
        "ARNI": {m: {"total": [], "1": [], "2": [], "3": []} for m in metrics},
        "IVAN": {m: {"total": [], "1": [], "2": [], "3": []} for m in metrics},
    }

data = init_storage()

# -----------------------------
# Read CSV
# -----------------------------
with open(FILE, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        model = row["dataset"].strip()
        split = split_map[row["split"].strip()]

        for col, metric_name in metric_map.items():
            value = float(row[col])

            # ignore missing models if any
            if model in data:
                data[model][metric_name][split].append(value)

# -----------------------------
# Aggregation
# -----------------------------
def agg(metric, values):
    if len(values) == 0:
        return (0.0, 0.0, 0.0)

    mean = float(np.mean(values))

    if higher_is_better[metric]:
        best = float(np.max(values))
        worst = float(np.min(values))
    else:
        best = float(np.min(values))
        worst = float(np.max(values))

    return (mean, best, worst)


results = {}

for model in ["ARNI", "IVAN"]:
    results[model] = {}

    for metric in metrics:
        results[model][metric] = {}

        for group in ["total", "1", "2", "3"]:
            results[model][metric][group] = agg(
                metric,
                data[model][metric][group]
            )

# -----------------------------
# Formatting
# -----------------------------
def fmt(x):
    return f"{x:.2f}"


def build_block(model):
    vals = []

    for metric in metrics:
        for group in ["total", "1", "2", "3"]:
            vals.append(results[model][metric][group])

    stats = ["mean", "best", "worst"]
    model_name = model.capitalize()

    lines = []

    for i, stat in enumerate(stats):
        if i == 0:
            line = f"& \\multirow{{3}}{{*}}{{{model_name}}} & {stat} & "
        else:
            line = f"&  & {stat} & "

        row = []

        for mean, best, worst in vals:
            if stat == "mean":
                v = mean
            elif stat == "best":
                v = best
            else:
                v = worst

            row.append(fmt(v))

        line += " & ".join(row) + " \\\\"
        lines.append(line)

        if i < 2:
            lines.append(r"\cline{3-23}")
        else:
            lines.append(r"\cline{2-23}")

    return "\n".join(lines)


# -----------------------------
# Write output
# -----------------------------
output = [
    build_block("ARNI"),
    build_block("IVAN"),
    r"\hline"
]

with open(OUT_FILE, "w") as f:
    f.write("\n".join(output))

print(f"Saved → {OUT_FILE}")