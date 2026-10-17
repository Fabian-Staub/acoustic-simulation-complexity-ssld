import re
import matplotlib.pyplot as plt

# Path to your .out file
out_file = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/generate_dataset/logs/generate_dataset_R2000_P8_S3_Order3_24kHz_shoeBox_SH_ISMOrder12.out"

# Matches lines like:
# 0 active source(s): 1683642 frames (19.97%)
pattern = re.compile(
    r"^\s*(\d+)\s+active source\(s\):\s+(\d+)\s+frames\s+\(([\d.]+)%\)"
)

data = []

with open(out_file, "r") as f:
    for line in f:
        match = pattern.match(line)
        if match:
            n_sources = int(match.group(1))
            n_frames = int(match.group(2))
            percentage = float(match.group(3))

            data.append((n_sources, n_frames, percentage))

# Sort by number of active sources
data.sort(key=lambda x: x[0])

if not data:
    raise RuntimeError("No source-overlap distribution found in the .out file.")

# Extract values
num_sources = [x[0] for x in data]
percentages = [x[2] for x in data]

labels = [f"{n}" for n in num_sources]

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(labels, percentages)

ax.set_title(
    "Frame Distribution by Number of Active Sources",
    fontsize=20,
    pad=15,
)
ax.set_xlabel(
    "Number of active sources",
    fontsize=20
)
ax.set_ylabel(
    "Percent (%)",
    fontsize=20
)

# Leave some space above the tallest bar
ax.set_ylim(0, max(percentages) * 1.18)

# Larger tick labels
ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=20)

# Add percentage labels above bars
for bar, pct in zip(bars, percentages):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{pct:.2f}%",
        ha="center",
        va="bottom",
        fontsize=20
    )

plt.tight_layout()
plt.show()

fig.savefig(
    "active_source_distribution.pdf",
    dpi=300,
    bbox_inches="tight"
)