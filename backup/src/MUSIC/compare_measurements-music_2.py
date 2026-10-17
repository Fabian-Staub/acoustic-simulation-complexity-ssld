import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1. Load data
# -----------------------
df_a = pd.read_csv(
    "../../../data/arni_em32/sh_wavs/arni_directions.csv"
)

df_b = pd.read_csv(
    "../../../data/arni_em32/sh_wavs/arni_directions_music.csv"
)

# -----------------------
# 2. Align by filename
# -----------------------
df = pd.merge(
    df_a,
    df_b,
    on="Name",
    suffixes=("_a", "_b")
)

# -----------------------
# 3. Compute azimuth/elevation errors
# -----------------------
df["d_az"] = df["az_b"] - df["az_a"]
df["d_el"] = df["el_b"] - df["el_a"]

# Normalize azimuth error to [-180, 180]
df["d_az"] = (df["d_az"] + 180) % 360 - 180

# -----------------------
# 4. Compute 3D angular distance
# -----------------------
# Convert directions to radians
az_a_rad = np.deg2rad(df["az_a"])
az_b_rad = np.deg2rad(df["az_b"])

el_a_rad = np.deg2rad(df["el_a"])
el_b_rad = np.deg2rad(df["el_b"])

# Great-circle angular distance:
#
# cos(theta) =
# sin(el_a) sin(el_b)
# + cos(el_a) cos(el_b) cos(az_b - az_a)
#
cos_angle = (
    np.sin(el_a_rad) * np.sin(el_b_rad)
    + np.cos(el_a_rad)
    * np.cos(el_b_rad)
    * np.cos(az_b_rad - az_a_rad)
)

# Prevent floating-point errors outside [-1, 1]
cos_angle = np.clip(cos_angle, -1.0, 1.0)

# Angular distance in degrees
df["angular_distance"] = np.rad2deg(
    np.arccos(cos_angle)
)

# -----------------------
# 5. Error statistics
# -----------------------

# Number of matched samples
n = len(df)

# ---- Azimuth ----

# Bias / mean signed error
az_bias = df["d_az"].mean()

# Standard deviation
az_std = df["d_az"].std()

# Mean absolute error
az_mae = np.mean(np.abs(df["d_az"]))

# Median absolute error
az_median_abs = np.median(np.abs(df["d_az"]))

# Root mean squared error
az_rmse = np.sqrt(
    np.mean(df["d_az"] ** 2)
)

# Minimum / maximum signed error
az_min = df["d_az"].min()
az_max = df["d_az"].max()

# Maximum absolute error
az_max_abs = np.max(
    np.abs(df["d_az"])
)

# ---- Elevation ----

# Bias / mean signed error
el_bias = df["d_el"].mean()

# Standard deviation
el_std = df["d_el"].std()

# Mean absolute error
el_mae = np.mean(np.abs(df["d_el"]))

# Median absolute error
el_median_abs = np.median(
    np.abs(df["d_el"])
)

# Root mean squared error
el_rmse = np.sqrt(
    np.mean(df["d_el"] ** 2)
)

# Minimum / maximum signed error
el_min = df["d_el"].min()
el_max = df["d_el"].max()

# Maximum absolute error
el_max_abs = np.max(
    np.abs(df["d_el"])
)

# ---- 3D angular distance ----

angular_mean = df["angular_distance"].mean()

angular_median = df["angular_distance"].median()

angular_std = df["angular_distance"].std()

angular_rms = np.sqrt(
    np.mean(df["angular_distance"] ** 2)
)

angular_min = df["angular_distance"].min()

angular_max = df["angular_distance"].max()

# -----------------------
# 6. Print statistics
# -----------------------
print("=" * 60)
print("MUSIC - Measurement Direction Error Statistics")
print("=" * 60)

print(f"Number of matched samples: {n}")

print("\nAzimuth:")
print(
    f"  Bias (mean error):          "
    f"{az_bias:8.3f} deg"
)
print(
    f"  Std. deviation:             "
    f"{az_std:8.3f} deg"
)
print(
    f"  Mean absolute error (MAE):  "
    f"{az_mae:8.3f} deg"
)
print(
    f"  Median absolute error:      "
    f"{az_median_abs:8.3f} deg"
)
print(
    f"  RMSE:                       "
    f"{az_rmse:8.3f} deg"
)
print(
    f"  Minimum signed error:       "
    f"{az_min:8.3f} deg"
)
print(
    f"  Maximum signed error:       "
    f"{az_max:8.3f} deg"
)
print(
    f"  Maximum absolute error:     "
    f"{az_max_abs:8.3f} deg"
)

print("\nElevation:")
print(
    f"  Bias (mean error):          "
    f"{el_bias:8.3f} deg"
)
print(
    f"  Std. deviation:             "
    f"{el_std:8.3f} deg"
)
print(
    f"  Mean absolute error (MAE):  "
    f"{el_mae:8.3f} deg"
)
print(
    f"  Median absolute error:      "
    f"{el_median_abs:8.3f} deg"
)
print(
    f"  RMSE:                       "
    f"{el_rmse:8.3f} deg"
)
print(
    f"  Minimum signed error:       "
    f"{el_min:8.3f} deg"
)
print(
    f"  Maximum signed error:       "
    f"{el_max:8.3f} deg"
)
print(
    f"  Maximum absolute error:     "
    f"{el_max_abs:8.3f} deg"
)

print("\n3D Angular Distance:")
print(
    f"  Mean angular distance:      "
    f"{angular_mean:8.3f} deg"
)
print(
    f"  Median angular distance:    "
    f"{angular_median:8.3f} deg"
)
print(
    f"  Std. deviation:             "
    f"{angular_std:8.3f} deg"
)
print(
    f"  RMS angular distance:       "
    f"{angular_rms:8.3f} deg"
)
print(
    f"  Minimum angular distance:   "
    f"{angular_min:8.3f} deg"
)
print(
    f"  Maximum angular distance:   "
    f"{angular_max:8.3f} deg"
)

print("=" * 60)

# -----------------------
# 7. Plot settings
# -----------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# -----------------------
# 8. Combined plots
# -----------------------
fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5.8)
)

# -----------------------
# Azimuth comparison
# -----------------------
axes[0].scatter(
    df["az_a"],
    df["az_b"]
)

axes[0].plot(
    [-180, 180],
    [-180, 180],
    "r--",
    label="Ideal"
)

axes[0].set_title(
    "Azimuth Comparison"
)

axes[0].set_xlabel(
    "Azimuth Meas. (deg)"
)

axes[0].set_ylabel(
    "Azimuth MUSIC (deg)"
)

axes[0].legend()

# -----------------------
# Elevation comparison
# -----------------------
axes[1].scatter(
    df["el_a"],
    df["el_b"]
)

axes[1].plot(
    [-90, 90],
    [-90, 90],
    "r--",
    label="Ideal"
)

axes[1].set_title(
    "Elevation Comparison"
)

axes[1].set_xlabel(
    "Elevation Meas. (deg)"
)

axes[1].set_ylabel(
    "Elevation MUSIC (deg)"
)

axes[1].legend()

# -----------------------
# Azimuth error histogram
# -----------------------
axes[2].hist(
    df["d_az"],
    bins=20
)

# Bias
axes[2].axvline(
    az_bias,
    linestyle="--",
    linewidth=2,
    label=f"Bias = {az_bias:.2f}°"
)

# Bias - 1 standard deviation
axes[2].axvline(
    az_bias - az_std,
    linestyle=":",
    linewidth=2,
    label=f"Bias ± Std = {az_std:.2f}°"
)

# Bias + 1 standard deviation
axes[2].axvline(
    az_bias + az_std,
    linestyle=":",
    linewidth=2
)

axes[2].set_title(
    "Azimuth Error (MUSIC - Meas.)"
)

axes[2].set_xlabel(
    "Error (deg)"
)

axes[2].set_ylabel(
    "Count"
)

axes[2].legend()

# -----------------------
# Elevation error histogram
# -----------------------
axes[3].hist(
    df["d_el"],
    bins=20
)

# Bias
axes[3].axvline(
    el_bias,
    linestyle="--",
    linewidth=2,
    label=f"Bias = {el_bias:.2f}°"
)

# Bias - 1 standard deviation
axes[3].axvline(
    el_bias - el_std,
    linestyle=":",
    linewidth=2,
    label=f"Bias ± Std = {el_std:.2f}°"
)

# Bias + 1 standard deviation
axes[3].axvline(
    el_bias + el_std,
    linestyle=":",
    linewidth=2
)

axes[3].set_title(
    "Elevation Error (MUSIC - Meas.)"
)

axes[3].set_xlabel(
    "Error (deg)"
)

axes[3].set_ylabel(
    "Count"
)

axes[3].legend()

# -----------------------
# Grid
# -----------------------
for ax in axes:
    ax.grid(
        True,
        alpha=0.25
    )

# -----------------------
# 9. Figure layout
# -----------------------
plt.tight_layout()

# -----------------------
# 10. Save figure
# -----------------------
plt.savefig(
    "arni_direction_comparison_music_summary.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()