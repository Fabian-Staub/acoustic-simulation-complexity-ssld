#!/usr/bin/env python3
"""Add dataset-specific DRR values to a combined predictions CSV.

The prediction rows are joined using the canonical key

    (prediction_dataset, room_idx, position_idx, source_idx)

``room_idx`` and the position/source indices come from the DRR metadata.  In
the predictions CSV, room and position are parsed from ``sample_id`` and the
source index is parsed from ``source_id_raw``.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
import math
from pathlib import Path
from typing import Iterable


DEFAULT_PREDICTIONS_CSV = Path(
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/"
    "staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/"
    "evaluation_results/predictions/"
    "predictions_non-shoeBox_disp-corners-1_SH_ISMOrder12.csv"
)

DATASET_ARGUMENTS = {
    "checkpoint_dataset": "checkpoint_metadata",
    "motus-dataset": "motus_metadata",
    "ivan-dataset": "ivan_metadata",
    "arni-dataset": "arni_metadata",
}

# Canonicalize both underscore and hyphen spellings used by different CSV
# producers. The canonical names match DATASET_ARGUMENTS and therefore the
# keys of the DRR lookup dictionaries.
DATASET_ALIASES = {
    "checkpoint_dataset": "checkpoint_dataset",
    "checkpoint-dataset": "checkpoint_dataset",
    "motus_dataset": "motus-dataset",
    "motus-dataset": "motus-dataset",
    "ivan_dataset": "ivan-dataset",
    "ivan-dataset": "ivan-dataset",
    "arni_dataset": "arni-dataset",
    "arni-dataset": "arni-dataset",
}

JoinKey = tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add drr_db to a combined multi-dataset predictions CSV."
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=DEFAULT_PREDICTIONS_CSV,
        help=(
            "Combined predictions CSV. Defaults to the "
            "predictions_non-shoeBox_disp-corners-1_SH_ISMOrder12.csv file."
        ),
    )
    parser.add_argument(
        "--checkpoint-metadata",
        type=Path,
        required=True,
        metavar="PATH",
        help="02_metadata_T60_DRR.csv for checkpoint_dataset.",
    )
    parser.add_argument(
        "--motus-metadata",
        type=Path,
        required=True,
        metavar="PATH",
        help="02_metadata_T60_DRR.csv for motus-dataset.",
    )
    parser.add_argument(
        "--ivan-metadata",
        type=Path,
        required=True,
        metavar="PATH",
        help="02_metadata_T60_DRR.csv for ivan-dataset.",
    )
    parser.add_argument(
        "--arni-metadata",
        type=Path,
        required=True,
        metavar="PATH",
        help="02_metadata_T60_DRR.csv for arni-dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV. By default, the filename is derived from the "
            "predictions CSV by inserting '_drr' after 'predictions' and is "
            "written beside the predictions CSV."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output CSV.",
    )
    return parser.parse_args()


def default_output_path(
    predictions_csv: Path,
) -> Path:
    """Insert ``_drr`` between ``predictions`` and the model name."""
    predictions_csv = predictions_csv.expanduser()
    stem = predictions_csv.stem
    if stem == "predictions":
        output_stem = "predictions_drr"
    elif stem.startswith("predictions_"):
        output_stem = f"predictions_drr_{stem.removeprefix('predictions_')}"
    else:
        raise ValueError(
            "Could not derive the default output name from --predictions-csv: "
            f"{predictions_csv.name!r} must start with 'predictions_'. "
            "Use --output to choose a different name."
        )
    return predictions_csv.with_name(f"{output_stem}{predictions_csv.suffix}")


def require_columns(
    fieldnames: Iterable[str] | None,
    required: set[str],
    csv_path: Path,
) -> None:
    available = set(fieldnames or ())
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required column(s): {', '.join(missing)}"
        )


def resolve_column(fieldnames: Iterable[str] | None, wanted: str) -> str:
    """Resolve a metadata column while tolerating escaped/Markdown headers."""
    for fieldname in fieldnames or ():
        normalized = fieldname.replace("\\", "").strip().strip("*").lower()
        if normalized == wanted:
            return fieldname
    raise ValueError(f"Predictions CSV is missing required column {wanted!r}.")


def validate_inactive_reference_row(
    row: dict[str, str],
    accdoa_columns: tuple[str, str, str],
    predictions_csv: Path,
    row_number: int,
    tolerance: float = 1e-6,
) -> None:
    """Ensure a blank source ID belongs to an inactive/padded GT slot."""
    values: list[float] = []
    for column in accdoa_columns:
        raw_value = (row.get(column) or "").strip()
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"Invalid reference ACCDOA value {raw_value!r} in column "
                f"{column!r} at {predictions_csv} row {row_number}."
            ) from error
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite reference ACCDOA value {raw_value!r} in column "
                f"{column!r} at {predictions_csv} row {row_number}."
            )
        values.append(value)

    if any(abs(value) > tolerance for value in values):
        raise ValueError(
            f"Blank source_id_raw with nonzero ground-truth ACCDOA "
            f"{tuple(values)!r} in {predictions_csv} row {row_number}; "
            "only inactive/padded source slots may have a blank source ID."
        )


def parse_nonnegative_int(value: str, label: str, context: str) -> int:
    try:
        result = int(value.strip())
    except (AttributeError, ValueError) as error:
        raise ValueError(f"Invalid {label} in {context}: {value!r}") from error
    if result < 0:
        raise ValueError(f"Invalid negative {label} in {context}: {value!r}")
    return result


def metadata_join_key(
    row: dict[str, str],
    dataset: str,
    csv_path: Path,
    row_number: int,
) -> JoinKey:
    """Parse (room, position, source) from one DRR metadata row."""
    context = f"{csv_path} row {row_number}"
    try:
        room_value = row["room_idx"]
    except KeyError as error:
        raise ValueError(f"Missing room_idx in {context}") from error
    room_idx = parse_nonnegative_int(room_value, "room_idx", context)

    position_id = (row.get("position_id") or "").strip()
    if dataset == "checkpoint_dataset":
        match = re.fullmatch(r"(?P<position>\d+)_S(?P<source>\d+)", position_id)
        expected = "<position>_S<source>, for example 5_S0"
    else:
        match = re.fullmatch(r"P(?P<position>\d+)_S(?P<source>\d+)", position_id)
        expected = "P<position>_S<source>, for example P1_S1"
    if match is None:
        raise ValueError(
            f"Invalid position_id {position_id!r} for {dataset!r} in {context}; "
            f"expected {expected}."
        )

    return room_idx, int(match.group("position")), int(match.group("source"))


def read_drr_lookup(metadata_csv: Path, dataset: str) -> dict[JoinKey, str]:
    """Read one DRR table and return (room, position, source) -> drr_db."""
    metadata_csv = metadata_csv.expanduser()
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"DRR metadata CSV not found: {metadata_csv}")

    lookup: dict[JoinKey, str] = {}
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(
            reader.fieldnames,
            {"room_idx", "position_id", "drr_db"},
            metadata_csv,
        )
        for row_number, row in enumerate(reader, start=2):
            key = metadata_join_key(row, dataset, metadata_csv, row_number)
            drr_db = (row.get("drr_db") or "").strip()
            if not drr_db:
                raise ValueError(
                    f"Empty drr_db for join key {key!r} in "
                    f"{metadata_csv} row {row_number}"
                )
            try:
                float(drr_db)
            except ValueError as error:
                raise ValueError(
                    f"Invalid drr_db for join key {key!r} in "
                    f"{metadata_csv} row {row_number}: {drr_db!r}"
                ) from error

            previous = lookup.get(key)
            if previous is not None and previous != drr_db:
                raise ValueError(
                    f"Conflicting drr_db values for join key {key!r} in "
                    f"{metadata_csv}: {previous!r} and {drr_db!r}"
                )
            lookup[key] = drr_db

    if not lookup:
        raise ValueError(f"DRR metadata CSV has no data rows: {metadata_csv}")
    return lookup


def prediction_join_key(
    sample_id: str,
    source_id_raw: str,
    predictions_csv: Path,
    row_number: int,
) -> JoinKey:
    """Parse (room, position, source) from one predictions row."""
    context = f"{predictions_csv} row {row_number}"
    sample_id = sample_id.strip()

    # Supported examples include R12_P5, R12_5, 12_P5, 12_5,
    # room_12_position_5, and the same forms with a trailing segment suffix.
    labeled = re.search(
        r"(?:room(?:_idx)?|r)[_=-]?(?P<room>\d+).*?"
        r"(?:position(?:_idx)?|pos|p)[_=-]?(?P<position>\d+)",
        sample_id,
        flags=re.IGNORECASE,
    )
    compact = re.match(
        r"R?(?P<room>\d+)[_-]P?(?P<position>\d+)(?:$|[_-])",
        sample_id,
        flags=re.IGNORECASE,
    )
    match = labeled or compact
    if match is None:
        raise ValueError(
            f"Could not parse room and position indices from sample_id "
            f"{sample_id!r} in {context}. Expected a form such as R12_P5, "
            "12_5, or room_12_position_5."
        )

    # Some CSV exporters serialize integer source IDs as floats (for example
    # 1.0 instead of 1). Accept only a zero-valued decimal suffix so values
    # such as 1.5 are still rejected.
    source_match = re.fullmatch(
        r"S?(?P<source>\d+)(?:\.0+)?",
        source_id_raw.strip(),
        re.I,
    )
    if source_match is None:
        raise ValueError(
            f"Could not parse source index from source_id_raw "
            f"{source_id_raw!r} in {context}; expected an integer, an "
            "integer-valued decimal such as 1.0, or the same prefixed with S."
        )

    return (
        int(match.group("room")),
        int(match.group("position")),
        int(source_match.group("source")),
    )


def add_drr_column(
    predictions_csv: Path,
    output_csv: Path,
    metadata_paths: dict[str, Path],
    overwrite: bool = False,
) -> tuple[int, dict[str, int], int]:
    """Join DRR values into predictions and atomically write the result."""
    predictions_csv = predictions_csv.expanduser()
    output_csv = output_csv.expanduser()
    if not predictions_csv.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")
    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_csv}. Use --overwrite to replace it."
        )
    if output_csv.resolve() == predictions_csv.resolve():
        raise ValueError("Output path must differ from the predictions input path.")

    lookups = {
        dataset: read_drr_lookup(path, dataset)
        for dataset, path in metadata_paths.items()
    }
    counts = {dataset: 0 for dataset in lookups}
    inactive_rows = 0
    unmatched: list[tuple[int, str, str]] = []
    total_rows = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with predictions_csv.open(
            "r", encoding="utf-8-sig", newline=""
        ) as source, tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=output_csv.parent,
            prefix=f".{output_csv.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temp_path = Path(destination.name)
            reader = csv.DictReader(source)
            require_columns(
                reader.fieldnames,
                {"sample_id", "source_id_raw", "prediction_dataset"},
                predictions_csv,
            )
            fieldnames = list(reader.fieldnames or ())
            accdoa_columns = tuple(
                resolve_column(reader.fieldnames, name)
                for name in ("accdoa_x", "accdoa_y", "accdoa_z")
            )
            if "drr_db" not in fieldnames:
                fieldnames.append("drr_db")
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()

            for row_number, row in enumerate(reader, start=2):
                total_rows += 1
                dataset_raw = (row.get("prediction_dataset") or "").strip()
                dataset = DATASET_ALIASES.get(dataset_raw, dataset_raw)
                sample_id = (row.get("sample_id") or "").strip()
                source_id_raw = (row.get("source_id_raw") or "").strip()

                # The evaluator writes every model slot. Datasets with fewer
                # real sources therefore contain zero-ACCDOA padded rows whose
                # source_id_raw is blank. They have no physical source and no
                # meaningful DRR, so preserve the row with an empty drr_db.
                if not source_id_raw:
                    validate_inactive_reference_row(
                        row,
                        accdoa_columns,
                        predictions_csv,
                        row_number,
                    )
                    row["drr_db"] = ""
                    writer.writerow(row)
                    inactive_rows += 1
                    continue

                lookup = lookups.get(dataset)
                if lookup is None:
                    drr_db = None
                    key = None
                else:
                    key = prediction_join_key(
                        sample_id,
                        source_id_raw,
                        predictions_csv,
                        row_number,
                    )
                    drr_db = lookup.get(key)
                if drr_db is None:
                    if len(unmatched) < 20:
                        unmatched.append(
                            (
                                row_number,
                                dataset_raw,
                                f"{sample_id} / {source_id_raw}",
                            )
                        )
                    continue

                # CSV values are text, but write the source ID in canonical
                # integer form: 1.0/S1/S1.0 -> 1. Padded rows handled above
                # intentionally keep an empty source_id_raw.
                assert key is not None
                row["source_id_raw"] = str(key[2])
                row["drr_db"] = drr_db
                writer.writerow(row)
                counts[dataset] += 1

        if unmatched:
            preview = "; ".join(
                f"row {row_number}: dataset={dataset!r}, sample_id={sample_id!r}"
                for row_number, dataset, sample_id in unmatched
            )
            raise ValueError(
                "Could not find a DRR match for prediction row(s). "
                f"First {len(unmatched)} unmatched: {preview}"
            )
        if total_rows == 0:
            raise ValueError(f"Predictions CSV has no data rows: {predictions_csv}")

        os.replace(temp_path, output_csv)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return total_rows, counts, inactive_rows


def main() -> None:
    args = parse_args()
    output = args.output or default_output_path(args.predictions_csv)
    metadata_paths = {
        dataset: getattr(args, argument_name)
        for dataset, argument_name in DATASET_ARGUMENTS.items()
    }
    total_rows, counts, inactive_rows = add_drr_column(
        predictions_csv=args.predictions_csv,
        output_csv=output,
        metadata_paths=metadata_paths,
        overwrite=args.overwrite,
    )
    print(f"Saved {total_rows} prediction rows: {output}")
    for dataset, count in counts.items():
        print(f"  {dataset}: {count} rows with drr_db")
    print(f"  inactive/padded rows with empty drr_db: {inactive_rows}")


if __name__ == "__main__":
    main()
