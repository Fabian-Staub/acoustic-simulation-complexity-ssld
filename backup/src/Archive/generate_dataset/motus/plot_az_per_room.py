from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_azimuth_directions(
    csv_path: str | Path,
    output_path: str | Path = "azimuth_directions.png",
) -> None:
    """
    Read room/source direction data from CSV and save an azimuth plot as PNG.

    Expected columns:
        Name, dx, dy, dz, az, el, rt60

    Example Name:
        R0001_P1_S1.wav
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    df = pd.read_csv(csv_path)

    required_columns = {"Name", "az"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    # Extract room and source from names such as R0001_P1_S1.wav.
    extracted = df["Name"].str.extract(
        r"^(?P<room>R\d+)_P\d+_(?P<source>S\d+)\.wav$"
    )

    if extracted.isna().any().any():
        invalid_names = df.loc[
            extracted.isna().any(axis=1),
            "Name",
        ].tolist()

        raise ValueError(
            "Some filenames do not match the expected pattern "
            f"'R0001_P1_S1.wav': {invalid_names[:5]}"
        )

    df = df.assign(
        room=extracted["room"],
        source=extracted["source"],
    )

    # Sort numerically rather than alphabetically.
    df["room_number"] = (
        df["room"]
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )

    df["source_number"] = (
        df["source"]
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )

    df["az"] = pd.to_numeric(df["az"], errors="raise")

    df = df.sort_values(
        ["room_number", "source_number"]
    ).reset_index(drop=True)

    # Create one x-position per room.
    room_order = (
        df[["room", "room_number"]]
        .drop_duplicates()
        .sort_values("room_number")["room"]
        .tolist()
    )

    room_positions = {
        room: index
        for index, room in enumerate(room_order)
    }

    df["x"] = df["room"].map(room_positions)

    fig, ax = plt.subplots(figsize=(10, 5))

    for source, source_df in df.groupby(
        "source",
        sort=False,
    ):
        source_df = source_df.sort_values("room_number")

        line, = ax.plot(
            source_df["x"],
            source_df["az"],
            marker=".",
            markersize=2.5,
            linewidth=1,
        )

        # Label the curve beside its final point.
        last_row = source_df.iloc[-1]

        ax.annotate(
            source,
            xy=(last_row["x"], last_row["az"]),
            xytext=(7, 0),
            textcoords="offset points",
            color=line.get_color(),
            fontsize=9,
            fontweight="bold",
            va="center",
            ha="left",
            clip_on=False,
        )

    # Show a manageable number of room labels.
    max_labels = 5
    tick_step = max(
        1,
        round((len(room_order) - 1) / max_labels),
    )

    tick_positions = list(
        range(0, len(room_order), tick_step)
    )

    # Include the final room only when it is not too close
    # to the previous tick.
    final_position = len(room_order) - 1

    if (
        tick_positions[-1] != final_position
        and final_position - tick_positions[-1] >= tick_step / 2
    ):
        tick_positions.append(final_position)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [room_order[position] for position in tick_positions],
        rotation=45,
        ha="right",
    )

    ax.set_title("Azimuth directions by room configuration")
    ax.set_xlabel("Room configuration")
    ax.set_ylabel("Azimuth direction (degrees)")
    ax.grid(axis="y", alpha=0.3)

    # Add space on the right for direct curve labels.
    ax.set_xlim(
        -0.5,
        len(room_order) - 1 + max(3, len(room_order) * 0.04),
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Plot saved to: {output_path.resolve()}")


if __name__ == "__main__":
    plot_azimuth_directions(
        csv_path=(
            "/scratch/elec/t412-techpsychoacoustics/"
            "msc-thesis/2026_fstaub/staubf1/Projects/"
            "Master_Thesis/data/motus/"
            "motus_directions_music_rt60.csv"
        ),
        output_path="az_directions_motus.png",
    )