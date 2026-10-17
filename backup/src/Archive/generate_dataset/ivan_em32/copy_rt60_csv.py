import pandas as pd


def copy_rt60(
    target_csv: str,
    source_csv: str,
    output_csv: str,
) -> None:
    """
    Copy the rt60 column from source_csv into target_csv,
    matching rows by the Name column.
    """

    target = pd.read_csv(target_csv)
    source = pd.read_csv(source_csv)

    required_target_columns = {"Name"}
    required_source_columns = {"Name", "rt60"}

    missing_target = required_target_columns - set(target.columns)
    missing_source = required_source_columns - set(source.columns)

    if missing_target:
        raise ValueError(
            f"Target CSV is missing columns: {sorted(missing_target)}"
        )

    if missing_source:
        raise ValueError(
            f"Source CSV is missing columns: {sorted(missing_source)}"
        )

    # Ensure each filename occurs only once in the source CSV.
    duplicate_names = source.loc[source["Name"].duplicated(), "Name"].tolist()

    if duplicate_names:
        raise ValueError(
            f"Duplicate names found in source CSV: {duplicate_names}"
        )

    rt60_by_name = source.set_index("Name")["rt60"]

    # Match rt60 values using the filename.
    target["rt60"] = target["Name"].map(rt60_by_name)

    missing_matches = target.loc[target["rt60"].isna(), "Name"].tolist()

    if missing_matches:
        print("Warning: no rt60 value found for:")
        for name in missing_matches:
            print(f"  {name}")

    target.to_csv(output_csv, index=False)

    print(f"Saved merged CSV to: {output_csv}")


if __name__ == "__main__":
    copy_rt60(
        target_csv="/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/motus_directions_music.csv",
        source_csv="/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/motus_rt60.csv",
        output_csv="motus_directions_music_rt60.csv",
    )