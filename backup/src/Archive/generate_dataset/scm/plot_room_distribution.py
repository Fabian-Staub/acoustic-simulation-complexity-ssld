import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R4000_P8_S3_Order3_24kHz_shoeBox_rtProfilesMIT_ISMOrder30/Generated_HOA_SRIR_data.csv"
)

#################################################
# RT distribution
#################################################

rt = df["RT"].dropna()

#################################################
# Room volumes
#################################################

room_volumes = df["x_room"] * df["y_room"] * df["z_room"]

#################################################
# Room dimensions (x, y, z)
#################################################

room_x = df["x_room"]
room_y = df["y_room"]
room_z = df["z_room"]

#################################################
# C50 distribution
#################################################

c50 = df["c50"].dropna()

#################################################
# Plot
#################################################

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

########################
# RT histogram
########################
axes[0, 0].hist(rt, bins=80, color="steelblue")
axes[0, 0].set_title("RT Distribution")
axes[0, 0].set_xlabel("RT [s]")
axes[0, 0].set_ylabel("Count")

########################
# Room volume
########################
axes[0, 1].hist(room_volumes, bins=200, color="darkorange")
axes[0, 1].set_title("Room Volume Distribution")
axes[0, 1].set_xlabel("Volume [m³]")
axes[0, 1].set_ylabel("Count")

########################
# C50 distribution
########################
axes[0, 2].hist(c50, bins=80, color="green")
axes[0, 2].set_title("C50 Distribution")
axes[0, 2].set_xlabel("C50 [dB]")
axes[0, 2].set_ylabel("Count")

########################
# Room dimensions (x, y, z)
########################
axes[1, 0].hist(room_x, bins=37, alpha=0.6, label="x")
axes[1, 0].hist(room_y, bins=37, alpha=0.6, label="y")
axes[1, 0].hist(room_z, bins=11, alpha=0.6, label="z")
axes[1, 0].set_title("Room Dimension Distribution")
axes[1, 0].set_xlabel("Length [m]")
axes[1, 0].set_ylabel("Count")
axes[1, 0].legend()

########################
# RT vs Room Volume
########################
axes[1, 1].scatter(room_volumes, rt, s=5, alpha=0.4)
axes[1, 1].set_title("RT vs Room Volume")
axes[1, 1].set_xlabel("Volume [m³]")
axes[1, 1].set_ylabel("RT [s]")


plt.tight_layout()

plt.savefig(
    "room_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: room_distribution.png")