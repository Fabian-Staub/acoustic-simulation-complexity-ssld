#!/usr/bin/env python3
"""Update RT and frequency-dependent RT60 metadata in room pickle files.

Expected pickle structure (matching the attached generator script):

    {
        "room_idx": int,
        "sample_rate": int,
        "rirs": {
            "P1_S1": {
                "rir": numpy.ndarray,
                "metadata": {"Name": "room_name_P1_S1", "RT": float, ...},
            },
            ...
        },
    }

Only load pickle files you trust. Python pickle files can execute code while
being loaded.
"""

from __future__ import annotations

import csv
import math
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RT_COLUMN = "RT"
RT60_COLUMNS = (
    "rt60_125Hz",
    "rt60_250Hz",
    "rt60_500Hz",
    "rt60_1000Hz",
    "rt60_2000Hz",
    "rt60_4000Hz",
    "rt60_8000Hz",
    "rt60_16000Hz",
)
VALUE_COLUMNS = (RT_COLUMN, *RT60_COLUMNS)


@dataclass(frozen=True)
class CsvRow:
    name: str
    room_idx: int | None
    position_id: str | None
    values: dict[str, float]


def parse_float(value: str | None, *, column: str, row_number: int) -> float:
    if value is None or not value.strip():
        raise ValueError(f"CSV row {row_number}: {column!r} is empty")
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"CSV row {row_number}: {column!r} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(
            f"CSV row {row_number}: {column!r} must be finite, got {value!r}"
        )
    return number


def normalize_position(value: object) -> str:
    """Normalize CSV values such as '1_S1' to pickle keys such as 'P1_S1'."""
    text = str(value).strip()
    if text and not text.upper().startswith("P"):
        text = "P" + text
    return text.upper()


def load_csv(csv_path: Path) -> tuple[dict[str, CsvRow], dict[tuple[int, str], CsvRow]]:
    by_name: dict[str, CsvRow] = {}
    by_room_position: dict[tuple[int, str], CsvRow] = {}

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        required = {"Name", *VALUE_COLUMNS}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

        for row_number, raw in enumerate(reader, start=2):
            name = (raw.get("Name") or "").strip()
            if not name:
                raise ValueError(f"CSV row {row_number}: 'Name' is empty")
            if name in by_name:
                raise ValueError(f"CSV contains duplicate Name value {name!r}")

            room_idx_text = (raw.get("room_idx") or "").strip()
            position_text = (raw.get("position_id") or "").strip()
            room_idx = int(room_idx_text) if room_idx_text else None
            position_id = normalize_position(position_text) if position_text else None
            values = {
                column: parse_float(raw.get(column), column=column, row_number=row_number)
                for column in VALUE_COLUMNS
            }
            row = CsvRow(name, room_idx, position_id, values)
            by_name[name] = row

            if room_idx is not None and position_id is not None:
                key = (room_idx, position_id)
                if key in by_room_position:
                    raise ValueError(
                        "CSV contains duplicate room_idx/position_id pair "
                        f"{key!r}"
                    )
                by_room_position[key] = row

    if not by_name:
        raise ValueError("CSV contains no data rows")
    return by_name, by_room_position


def find_csv_row(
    *,
    metadata: dict[str, Any],
    pickle_path: Path,
    room_idx: object,
    rir_key: object,
    by_name: dict[str, CsvRow],
    by_room_position: dict[tuple[int, str], CsvRow],
) -> CsvRow | None:
    # Best match: the exact Name already stored by the generator.
    metadata_name = metadata.get("Name")
    if metadata_name is not None:
        row = by_name.get(str(metadata_name).strip())
        if row is not None:
            return row

    # Useful for older files whose metadata did not yet contain Name.
    constructed_name = f"{pickle_path.stem}_{rir_key}"
    row = by_name.get(constructed_name)
    if row is not None:
        return row

    # This supports generator-style R0000.pkl names and CSVs with room indices.
    try:
        numeric_room_idx = int(room_idx)
    except (TypeError, ValueError):
        return None
    return by_room_position.get((numeric_room_idx, normalize_position(rir_key)))


def iter_pickle_paths(folder: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*.pkl" if recursive else "*.pkl"
    yield from sorted(path for path in folder.glob(pattern) if path.is_file())


def validate_pickle_structure(data: Any, path: Path) -> dict[Any, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level object is not a dictionary")
    rirs = data.get("rirs")
    if not isinstance(rirs, dict):
        raise ValueError(f"{path}: missing dictionary at top-level key 'rirs'")
    for rir_key, entry in rirs.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: rirs[{rir_key!r}] is not a dictionary")
        if "rir" not in entry:
            raise ValueError(f"{path}: rirs[{rir_key!r}] has no 'rir' value")
        if not isinstance(entry.get("metadata"), dict):
            raise ValueError(
                f"{path}: rirs[{rir_key!r}] has no metadata dictionary"
            )
    return rirs


def atomic_pickle_dump(data: Any, destination: Path) -> None:
    """Write beside the original, flush to disk, then atomically replace it."""
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def update_one_pickle(
    path: Path,
    *,
    by_name: dict[str, CsvRow],
    by_room_position: dict[tuple[int, str], CsvRow],
    skip_missing: bool,
    dry_run: bool,
    backup: bool,
) -> tuple[int, list[str]]:
    with path.open("rb") as handle:
        data = pickle.load(handle)

    rirs = validate_pickle_structure(data, path)
    missing: list[str] = []
    pending: list[tuple[dict[str, Any], CsvRow]] = []
    for rir_key, entry in rirs.items():
        metadata = entry["metadata"]
        row = find_csv_row(
            metadata=metadata,
            pickle_path=path,
            room_idx=data.get("room_idx"),
            rir_key=rir_key,
            by_name=by_name,
            by_room_position=by_room_position,
        )
        if row is None:
            label = str(metadata.get("Name") or f"{path.stem}_{rir_key}")
            missing.append(label)
        else:
            pending.append((metadata, row))

    if missing and not skip_missing:
        raise ValueError(
            f"{path}: no matching CSV row for: {', '.join(missing)}"
        )

    for metadata, row in pending:
        # dict.update overwrites the old RT and adds/overwrites every RT60 field.
        metadata.update(row.values)

    if pending and not dry_run:
        if backup:
            backup_path = path.with_name(path.name + ".bak")
            if backup_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing backup: {backup_path}"
                )
            shutil.copy2(path, backup_path)
        atomic_pickle_dump(data, path)

    return len(pending), missing


# -----------------------------------------------------------------------------
# Configuration: paste your values here
# -----------------------------------------------------------------------------
PICKLE_FOLDER = Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/poschadel/RIRs")
METADATA_CSV = Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/poschadel/RIRs/01_metadata.csv")

RECURSIVE = False
SKIP_MISSING = False
DRY_RUN = False
BACKUP = False

def main() -> int:
    folder = PICKLE_FOLDER.expanduser().resolve()
    csv_path = METADATA_CSV.expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    by_name, by_room_position = load_csv(csv_path)
    paths = list(iter_pickle_paths(folder, RECURSIVE))
    if not paths:
        raise FileNotFoundError(f"No .pkl files found in {folder}")

    total_updated = 0
    total_missing = 0
    for path in paths:
        updated, missing = update_one_pickle(
            path,
            by_name=by_name,
            by_room_position=by_room_position,
            skip_missing=SKIP_MISSING,
            dry_run=DRY_RUN,
            backup=BACKUP,
        )
        total_updated += updated
        total_missing += len(missing)
        action = "would update" if DRY_RUN else "updated"
        print(f"{path}: {action} {updated} RIR metadata record(s)")
        if missing:
            print(f"  unmatched and unchanged: {', '.join(missing)}")

    action = "Would update" if DRY_RUN else "Updated"
    print(
        f"{action} {total_updated} RIR metadata record(s) in {len(paths)} pickle file(s); "
        f"{total_missing} unmatched."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
