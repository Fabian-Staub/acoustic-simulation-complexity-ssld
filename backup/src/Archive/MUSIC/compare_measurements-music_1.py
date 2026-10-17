import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1. Load data
# -----------------------
df_a = pd.read_csv("../../../data/arni_em32/sh_wavs/arni_directions.csv")
df_b = pd.read_csv("../../../data/arni_em32/sh_wavs/arni_directions_music.csv")

# -----------------------
# 2. Align by filename
# -----------------------
df = pd.merge(df_a, df_b, on="Name", suffixes=("_a", "_b"))

# -----------------------
# 3. Compute az/el errors
# -----------------------
df["d_az"] = df["az_b"] - df["az_a"]
df["d_el"] = df["el_b"] - df["el_a"]

# Normalize azimuth error to [-180, 180]
df["d_az"] = (df["d_az"] + 180) % 360 - 180

# -----------------------
# 4. Systematic offset (bias)
# -----------------------
az_bias = df["d_az"].mean()
el_bias = df["d_el"].mean()

az_std = df["d_az"].std()
el_std = df["d_el"].std()

print("=== Systematic Offset (B - A) ===")
print(f"Azimuth bias:   {az_bias:.3f} deg ± {az_std:.3f}")
print(f"Elevation bias: {el_bias:.3f} deg ± {el_std:.3f}")

# -----------------------
# 5–6. Combined plots (ONE FIGURE)
# -----------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes = axes.flatten()

# Azimuth comparison
axes[0].scatter(df["az_a"], df["az_b"])
axes[0].plot([-180, 180], [-180, 180], "r--")
axes[0].set_title("Azimuth Comparison")
axes[0].set_xlabel("Azimuth Meas. (deg)")
axes[0].set_ylabel("Azimuth MUSIC (deg)")

# Elevation comparison
axes[1].scatter(df["el_a"], df["el_b"])
axes[1].plot([-90, 90], [-90, 90], "r--")
axes[1].set_title("Elevation Comparison")
axes[1].set_xlabel("Elevation Meas. (deg)")
axes[1].set_ylabel("Elevation MUSIC. (deg)")

# Azimuth error histogram
axes[2].hist(df["d_az"], bins=20)
axes[2].set_title("Azimuth Error (MUSIC - Meas.)")
axes[2].set_xlabel("deg")
axes[2].set_ylabel("count")

# Elevation error histogram
axes[3].hist(df["d_el"], bins=20)
axes[3].set_title("Elevation Error (MUSIC - Meas.)")
axes[3].set_xlabel("deg")
axes[3].set_ylabel("count")

for ax in axes:
    ax.grid(True, alpha=0.25)

plt.tight_layout()

# -----------------------
# 7. Save figure
# -----------------------
plt.savefig("arni_direction_comparison_music_summary.pdf", dpi=300, bbox_inches="tight")
plt.show()