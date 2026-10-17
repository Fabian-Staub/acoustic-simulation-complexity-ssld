import numpy as np
import matplotlib.pyplot as plt

# Load data
rtProfilesMit = np.load("rtProfilesMit.npy")

# Center frequencies
center_frequencies = [
    125, 250, 500, 1000,
    2000, 4000, 8000, 16000
]

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})

fig, ax = plt.subplots(figsize=(11, 8))

# ------------------------------------------------------------------
# Use the same bins for every frequency band
# ------------------------------------------------------------------

all_values = rtProfilesMit[np.isfinite(rtProfilesMit)]
x_max = np.percentile(all_values, 99.5)

bins = np.linspace(0, x_max, 100)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = np.diff(bins)

colors = plt.cm.tab10(
    np.linspace(0, 1, rtProfilesMit.shape[1])
)

# Vertical distance between frequency bands
spacing = 1.2

for band_idx in range(rtProfilesMit.shape[1]):

    values = rtProfilesMit[:, band_idx]
    values = values[np.isfinite(values)]

    # Histogram as probability density
    hist, _ = np.histogram(
        values,
        bins=bins,
        density=True
    )

    # Normalize each histogram to the same height
    if hist.max() > 0:
        hist = hist / hist.max()

    baseline = band_idx * spacing

    # Histogram bars
    ax.bar(
        bin_centers,
        hist,
        width=bin_widths,
        bottom=baseline,
        align="center",
        color=colors[band_idx],
        alpha=0.65,
        edgecolor=colors[band_idx],
        linewidth=0.5
    )

    # Baseline
    ax.axhline(
        baseline,
        linewidth=0.7,
        alpha=0.25,
        color="black"
    )

# Frequency labels on y-axis
positions = np.arange(len(center_frequencies)) * spacing

ax.set_yticks(positions)
ax.set_yticklabels([
    f"{freq} Hz" for freq in center_frequencies
])

ax.set_xlabel("RT (s)")
ax.set_ylabel("Frequency Band")
ax.set_title("RT Distribution by Frequency Band")

ax.set_xlim(0, x_max)

ax.grid(
    axis="x",
    alpha=0.25,
    linestyle="--"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    "rtProfilesMit_histogram.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.close()