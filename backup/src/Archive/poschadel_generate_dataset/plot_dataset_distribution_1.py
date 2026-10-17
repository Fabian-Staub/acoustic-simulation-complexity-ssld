from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#################################################
# Configuration
#################################################

DATA_FOLDER = Path(
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P8_S3_Order3_24kHz_shoeBox_SH_ISMOrder12/RIRs"
)

OUTPUT_FILE = DATA_FOLDER / "dataset_distribution.pdf"

# The new CSV format already contains one row per impulse-response/source.
# Duplicate source rows across multiple CSV files are still collapsed below.



#################################################
# Helper functions
#################################################

def first_valid(series: pd.Series):
    """
    Return the first non-null value from a pandas Series.

    This is used when duplicate rows belonging to the same physical source
    are combined.
    """
    valid = series.dropna()

    if valid.empty:
        return np.nan

    return valid.iloc[0]


def load_all_csv_files(folder: Path) -> pd.DataFrame:
    """
    Read and concatenate all CSV files in a folder.

    A source-file column is added so that the origin of every row
    remains available for debugging.
    """
    csv_files = sorted(folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files were found in:\n{folder}"
        )

    dataframes = []

    for csv_file in csv_files:
        print(f"Reading: {csv_file.name}")

        try:
            current_df = pd.read_csv(csv_file, low_memory=False)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read CSV file:\n{csv_file}"
            ) from exc

        current_df["csv_file"] = csv_file.name
        dataframes.append(current_df)

    combined_df = pd.concat(
        dataframes,
        ignore_index=True,
        sort=False,
    )

    print()
    print(f"CSV files read: {len(csv_files)}")
    print(f"Combined rows: {len(combined_df):,}")

    return combined_df


#################################################
# Load all CSV files
#################################################

df = load_all_csv_files(DATA_FOLDER)


#################################################
# Check required columns
#################################################

required_columns = {
    "Name",
    "room_idx",
    "position_id",
    "az_deg",
    "el_deg",
    "RT",
    "c50",
    "x_room",
    "y_room",
    "z_room",
    "status",
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise KeyError(
        "The following required columns are missing:\n"
        + "\n".join(sorted(missing_columns))
    )


#################################################
# Convert relevant columns to numeric
#################################################

numeric_columns = [
    "room_idx",
    "sample_rate",
    "az_deg",
    "el_deg",
    "RT",
    "c50",
    "x_room",
    "y_room",
    "z_room",
    "x_source",
    "y_source",
    "z_source",
    "x_mic",
    "y_mic",
    "z_mic",
    "mic_orientation_az_deg",
    "accdoa_x",
    "accdoa_y",
    "accdoa_z",
]

# Convert only columns that are actually present. This keeps the script
# compatible with files that omit optional metadata columns.
numeric_columns = [column for column in numeric_columns if column in df.columns]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


#################################################
# Build room and source IDs
#################################################

df["Name"] = df["Name"].astype("string")
df["position_id"] = df["position_id"].astype("string")

# Prefer the explicit columns from the new CSV format.
# Examples:
#   room_idx    = 0       -> room_id   = R0000
#   position_id = 3_S2    -> source_id = R0000_P3_S2
df["room_id"] = df["room_idx"].map(
    lambda value: f"R{int(value):04d}" if pd.notna(value) else pd.NA
).astype("string")

position_parts = df["position_id"].str.extract(
    r"^(?P<position>\d+)_S(?P<source>\d+)$"
)

df["source_id"] = (
    df["room_id"]
    + "_P"
    + position_parts["position"].astype("string")
    + "_S"
    + position_parts["source"].astype("string")
)

# Fallback to the Name field if a position_id does not follow the expected
# '<position>_S<source>' pattern.
name_room_id = df["Name"].str.extract(r"(R\d+)", expand=False)
name_source_id = df["Name"].str.extract(
    r"(R\d+_P\d+_S\d+)",
    expand=False,
)

df["room_id"] = df["room_id"].fillna(name_room_id)
df["source_id"] = df["source_id"].fillna(name_source_id)


#################################################
# Remove rows without usable IDs
#################################################

rows_before_id_filter = len(df)

df = df.dropna(
    subset=["room_id", "source_id"]
).copy()

print(
    "Rows removed because room/source ID could not be built: "
    f"{rows_before_id_filter - len(df):,}"
)


#################################################
# Optional status filtering
#################################################

# If a status column exists, remove explicitly failed rows.
# Missing status values are retained.
if "status" in df.columns:
    status_text = df["status"].astype("string").str.strip().str.lower()

    successful_mask = (
        df["status"].isna()
        | status_text.eq("success")
    )

    failed_rows = (~successful_mask).sum()
    df = df.loc[successful_mask].copy()

    print(f"Rows removed because status was not Success: {failed_rows:,}")


#################################################
# Create one row per unique source
#################################################

# The new CSV format contains one row per impulse response. Grouping is still
# useful when several CSV files contain the same source or when duplicates are
# present.
source_df = (
    df.groupby(
        ["source_id", "room_id"],
        as_index=False,
        observed=True,
    )
    .agg(
        azimuth=("az_deg", first_valid),
        elevation=("el_deg", first_valid),
        RT=("RT", first_valid),
        c50=("c50", first_valid),
        x_room=("x_room", first_valid),
        y_room=("y_room", first_valid),
        z_room=("z_room", first_valid),
        source_name=("Name", first_valid),
        position_id=("position_id", first_valid),
        csv_file=("csv_file", first_valid),
    )
)


#################################################
# Calculate source-level room volume
#################################################

source_df["room_volume"] = (
    source_df["x_room"]
    * source_df["y_room"]
    * source_df["z_room"]
)


#################################################
# Create one row per unique room
#################################################

# Each room contributes exactly one data point to the room-dimension
# and room-volume distributions.
room_df = (
    source_df.groupby(
        "room_id",
        as_index=False,
        observed=True,
    )
    .agg(
        x_room=("x_room", first_valid),
        y_room=("y_room", first_valid),
        z_room=("z_room", first_valid),
    )
)

room_df["room_volume"] = (
    room_df["x_room"]
    * room_df["y_room"]
    * room_df["z_room"]
)


#################################################
# Prepare plotting data
#################################################

# One value per unique source
azimuth = source_df["azimuth"].dropna()
elevation = source_df["elevation"].dropna()
rt = source_df["RT"].dropna()
c50 = source_df["c50"].dropna()

# One value per unique room
room_x = room_df["x_room"].dropna()
room_y = room_df["y_room"].dropna()
room_z = room_df["z_room"].dropna()
room_volumes = room_df["room_volume"].dropna()

# One point per unique source
rt_volume_df = source_df.dropna(
    subset=["room_volume", "RT"]
).copy()

room_volumes_scatter = rt_volume_df["room_volume"]
rt_scatter = rt_volume_df["RT"]


#################################################
# Print dataset summary
#################################################

print()
print("Dataset summary")
print("------------------------------")
print(f"Unique rooms: {room_df['room_id'].nunique():,}")
print(f"Unique sources: {source_df['source_id'].nunique():,}")
print(f"Azimuth values: {len(azimuth):,}")
print(f"Elevation values: {len(elevation):,}")
print(f"RT values: {len(rt):,}")
print(f"C50 values: {len(c50):,}")
print(f"RT-volume points: {len(rt_volume_df):,}")


#################################################
# Plot
#################################################

# Larger, consistent fonts for publication/readability
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 10),
)


########################
# 1. DOA distribution
# One value per unique source
########################

axes[0, 0].hist(
    azimuth,
    bins=180,
    alpha=0.7,
    label="Azimuth",
)

axes[0, 0].hist(
    elevation,
    bins=90,
    alpha=0.7,
    label="Elevation",
)

axes[0, 0].set_title(
    f"DOA Distribution\n"
)
axes[0, 0].set_xlabel("Angle [deg]")
axes[0, 0].set_ylabel("Count")
axes[0, 0].legend()


########################
# 2. RT distribution
# One value per unique source
########################

axes[0, 1].hist(
    rt,
    bins=min(100, max(1, len(rt))),
)

axes[0, 1].set_title(
    "$T_{60}$ Distribution\n"
)
axes[0, 1].set_xlabel("$T_{60}$ [s]")
axes[0, 1].set_ylabel("Count")


########################
# 3. C50 distribution
# One value per unique source
########################

axes[0, 2].hist(
    c50,
    bins=min(80, max(1, len(c50))),
)

axes[0, 2].set_title(
    "$C_{50}$ Distribution\n"
)
axes[0, 2].set_xlabel("$C_{50}$ [dB]")
axes[0, 2].set_ylabel("Count")


########################
# 4. Room dimensions
# One value per unique room
########################

axes[1, 0].hist(
    room_x,
    bins=min(36, max(1, len(room_x))),
    alpha=0.6,
    label="x",
)

axes[1, 0].hist(
    room_y,
    bins=min(36, max(1, len(room_y))),
    alpha=0.6,
    label="y",
)

axes[1, 0].hist(
    room_z,
    bins=min(6, max(1, len(room_z))),
    alpha=0.6,
    label="z",
)

axes[1, 0].set_title(
    "Room Dimension Distribution\n"
)
axes[1, 0].set_xlabel("Length [m]")
axes[1, 0].set_ylabel("Count")
axes[1, 0].legend()


########################
# 5. Room volume
# One value per unique room
########################

axes[1, 1].hist(
    room_volumes,
    bins=min(100, max(1, len(room_volumes))),
)

axes[1, 1].set_title(
    "Room Volume Distribution\n"
)
axes[1, 1].set_xlabel("Volume [m³]")
axes[1, 1].set_ylabel("Count")


########################
# 6. RT vs room volume
# One point per unique source
########################

axes[1, 2].scatter(
    room_volumes_scatter,
    rt_scatter,
    s=8,
    alpha=0.35,
)

axes[1, 2].set_title(
    "$T_{60}$ vs. Room Volume\n"
)
axes[1, 2].set_xlabel("Volume [m³]")
axes[1, 2].set_ylabel("$T_{60}$ [s]")


#################################################
# Save figure
#################################################

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print()
print(f"Saved: {OUTPUT_FILE}")