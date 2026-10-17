import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_Direct+Late"
    "/harp_dataset_2-8kHz_highpass_10-SCM_randRot_10-repeats.csv"
)


#################################################
# Keep only valid active sources
#################################################

df = df[df["voice_active"] == 1].copy()

#################################################
# Extract IDs from filename
#################################################

df["Name"] = df["Name"].astype(str)

# Room ID, e.g. R1
df["room_id"] = df["Name"].str.extract(r"(R\d+)")

# Unique source-position ID, e.g. R1_P1_S1
df["source_position_id"] = df["Name"].str.extract(r"(R\d+_P\d+_S\d+)")

#################################################
# Per-active-row distributions
#################################################

rt = df["RT"].dropna()
c50 = df["c50"].dropna()

azimuth = df["az"].dropna()
elevation = df["el"].dropna()

#################################################
# Per-room data
# Use each room only once for room dimension / volume histograms
#################################################

room_df = (
    df.dropna(subset=["room_id"])
      .drop_duplicates(subset="room_id")
      .copy()
)

room_df["room_volume"] = (
    room_df["x_room"]
    * room_df["y_room"]
    * room_df["z_room"]
)

room_x = room_df["x_room"].dropna()
room_y = room_df["y_room"].dropna()
room_z = room_df["z_room"].dropna()

room_volumes = room_df["room_volume"].dropna()

#################################################
# RT vs Room Volume
# One point per unique filename/source-position ID, e.g. R1_P1_S1
#################################################

df["room_volume"] = (
    df["x_room"]
    * df["y_room"]
    * df["z_room"]
)

valid_rt_vol = df.dropna(
    subset=["source_position_id", "room_volume", "RT"]
)

source_position_df = (
    valid_rt_vol
    .drop_duplicates(subset="source_position_id")
    .copy()
)

room_volumes_scatter = source_position_df["room_volume"]
rt_scatter = source_position_df["RT"]

#################################################
# Plot
#################################################

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

########################
# 1. DOA distribution
########################

axes[0, 0].hist(
    azimuth,
    bins=180,
    alpha=0.7,
    label="Azimuth"
)

axes[0, 0].hist(
    elevation,
    bins=90,
    alpha=0.7,
    label="Elevation"
)

axes[0, 0].set_title("DOA Distribution")
axes[0, 0].set_xlabel("Angle [deg]")
axes[0, 0].set_ylabel("Count")
axes[0, 0].legend()

########################
# 2. RT distribution
########################

axes[0, 1].hist(rt, bins=100)
axes[0, 1].set_title("RT Distribution")
axes[0, 1].set_xlabel("RT [s]")
axes[0, 1].set_ylabel("Count")

########################
# 3. C50 distribution
########################

axes[0, 2].hist(c50, bins=80)
axes[0, 2].set_title("C50 Distribution")
axes[0, 2].set_xlabel("C50 [dB]")
axes[0, 2].set_ylabel("Count")

########################
# 4. Room dimensions
# One entry per unique room
########################

axes[1, 0].hist(room_x, bins=37, alpha=0.6, label="x")
axes[1, 0].hist(room_y, bins=37, alpha=0.6, label="y")
axes[1, 0].hist(room_z, bins=11, alpha=0.6, label="z")

axes[1, 0].set_title(
    f"Room Dimension Distribution"
)
axes[1, 0].set_xlabel("Length [m]")
axes[1, 0].set_ylabel("Count")
axes[1, 0].legend()

########################
# 5. Room volume
# One entry per unique room
########################

axes[1, 1].hist(
    room_volumes,
    bins=min(100, len(room_volumes))
)

axes[1, 1].set_title(
    f"Room Volume Distribution"
)
axes[1, 1].set_xlabel("Volume [m³]")
axes[1, 1].set_ylabel("Count")

########################
# 6. RT vs Room Volume
# One point per unique R*_P*_S*
########################

axes[1, 2].scatter(
    room_volumes_scatter,
    rt_scatter,
    s=5,
    alpha=0.35
)

axes[1, 2].set_title(
    f"RT vs Room Volume"
)
axes[1, 2].set_xlabel("Volume [m³]")
axes[1, 2].set_ylabel("RT [s]")

plt.tight_layout()

plt.savefig(
    "dataset_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: dataset_distributions.png")