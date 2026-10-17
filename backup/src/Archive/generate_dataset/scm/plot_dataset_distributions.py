import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_anisotropic_source-constraints_smallGrid_adj-Room-Distr"
                 "/harp_dataset_2-7kHz_10-SCM_randRotDegrees_multiSource.csv")

#################################################
# Keep only valid active sources
#################################################

df = df[df["voice_active"] == 1].copy()

#################################################
# RT histogram
#################################################

rt = df["RT"]

#################################################
# DOA (already stored in CSV)
#################################################

azimuth = df["az"]
elevation = df["el"]

# Alternatively, recompute from dx,dy,dz:
#
# azimuth = np.degrees(np.arctan2(df["dy"], df["dx"]))
# elevation = np.degrees(
#     np.arctan2(
#         df["dz"],
#         np.sqrt(df["dx"]**2 + df["dy"]**2)
#     )
# )

#################################################
# Room volumes (one entry per room)
#################################################

# Extract room IDs from filenames
valid_names = df["Name"].astype(str)

df["room_id"] = valid_names.str.extract(r"(R\d+)")

room_df = (
    df.dropna(subset=["room_id"])
      .drop_duplicates(subset="room_id")
)

room_volumes = (
    room_df["x_room"]
    * room_df["y_room"]
    * room_df["z_room"]
)

#################################################
# Plot
#################################################

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# RT
axes[0].hist(rt, bins=100)
axes[0].set_title("RT Distribution")
axes[0].set_xlabel("RT [s]")
axes[0].set_ylabel("Count")

# DOA
axes[1].hist(
    azimuth,
    bins=180,
    alpha=0.7,
    label="Azimuth"
)

axes[1].hist(
    elevation,
    bins=90,
    alpha=0.7,
    label="Elevation"
)

axes[1].set_title("DOA Distribution")
axes[1].set_xlabel("Angle [deg]")
axes[1].set_ylabel("Count")
axes[1].legend()

# Room volume
axes[2].hist(
    room_volumes,
    bins=min(100, len(room_volumes))
)

axes[2].set_title(
    f"Room Volume Distribution\n({len(room_volumes)} unique rooms)"
)
axes[2].set_xlabel("Volume [m³]")
axes[2].set_ylabel("Count")

plt.tight_layout()

plt.savefig(
    "dataset_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: dataset_distributions.png")