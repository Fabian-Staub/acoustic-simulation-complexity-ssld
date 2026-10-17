#!/usr/bin/env python3
"""
Convert ARNI RIR WAV files into the same room-level pickle structure used by
the synthetic RIR generator.

Expected input CSV columns:
    Name, dx, dy, dz, az, el, rt60

Expected WAV filename pattern:
    <room>_P<position>_S<source>.wav

One pickle is written for every unique room, for example:
    anisotropic.pkl
    dry.pkl
    reverberant.pkl

A single combined metadata CSV is also written:
    arni_metadata.csv

Only metadata columns available from the ARNI dataset are included. Their
order follows the corresponding order in the synthetic metadata:

    Name
    room_idx
    position_id
    sample_rate
    RT
    accdoa_x
    accdoa_y
    accdoa_z
    az_deg
    el_deg
    status
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import wavfile


FILENAME_RE = re.compile(
    r"^(?P<room>.+)_P(?P<position>\d+)_S(?P<source>\d+)\.wav$",
    flags=re.IGNORECASE,
)


def natural_sort_key(value: str) -> list[Any]:
    """Sort strings naturally, so 'arni50' comes before 'arni100'."""
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]

REQUIRED_COLUMNS = {
    "Name",
    "dx",
    "dy",
    "dz",
    "az",
    "el",
    "rt60",
}

METADATA_COLUMNS = [
    "Name",
    "room_idx",
    "position_id",
    "sample_rate",
    "RT",
    "accdoa_x",
    "accdoa_y",
    "accdoa_z",
    "az_deg",
    "el_deg",
    "status",
]

DEFAULT_CSV_PATH = Path(
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/motus_directions_music_rt60.csv"
)

DEFAULT_WAV_DIR = Path(
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/RIRs_truncated"
)

DEFAULT_OUTPUT_DIR = Path(
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/poschadel/RIRs"
)


def pcm_to_float32(audio: np.ndarray) -> np.ndarray:
    """
    Convert WAV audio data to float32.

    Integer PCM data is normalized approximately to the range [-1, 1].
    Floating-point WAV data is converted directly to float32.
    """
    if np.issubdtype(audio.dtype, np.floating):
        return np.asarray(audio, dtype=np.float32)

    if np.issubdtype(audio.dtype, np.signedinteger):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))

        return audio.astype(np.float32) / scale

    if np.issubdtype(audio.dtype, np.unsignedinteger):
        info = np.iinfo(audio.dtype)
        midpoint = (info.max + 1) / 2.0

        return (audio.astype(np.float32) - midpoint) / midpoint

    raise TypeError(f"Unsupported WAV dtype: {audio.dtype}")


def load_rir_channels_first(wav_path: Path) -> tuple[int, np.ndarray]:
    """
    Read a WAV file and return its sample rate and RIR data.

    The returned RIR has shape:
        channels x samples

    scipy.io.wavfile.read returns:
        mono:         samples
        multichannel: samples x channels
    """
    sample_rate, audio = wavfile.read(wav_path)
    audio = pcm_to_float32(audio)

    if audio.ndim == 1:
        rir = audio[np.newaxis, :]
    elif audio.ndim == 2:
        rir = audio.T
    else:
        raise ValueError(
            f"{wav_path}: expected mono or multichannel WAV, "
            f"got shape {audio.shape}"
        )

    rir = np.ascontiguousarray(rir, dtype=np.float32)

    return int(sample_rate), rir


def parse_name(name: str) -> tuple[str, int, int, str]:
    """
    Parse an ARNI WAV filename.

    For:
        dry_P4_S2.wav

    return:
        room_name = "dry"
        position = 4
        source = 2
        rir_key = "P4_S2"
    """
    filename = Path(name).name
    match = FILENAME_RE.match(filename)

    if match is None:
        raise ValueError(
            f"Unexpected filename {name!r}; expected "
            "'<room>_P<number>_S<number>.wav'"
        )

    room_name = match.group("room")
    position = int(match.group("position"))
    source = int(match.group("source"))
    rir_key = f"P{position}_S{source}"

    return room_name, position, source, rir_key


def validate_numeric_value(
    value: Any,
    *,
    column_name: str,
    filename: str,
) -> float:
    """Convert a metadata value to float and reject missing values."""
    if pd.isna(value):
        raise ValueError(
            f"{filename}: missing value in required column "
            f"{column_name!r}"
        )

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{filename}: invalid numeric value {value!r} "
            f"in column {column_name!r}"
        ) from exc


def make_metadata(
    *,
    original_name: str,
    room_idx: int,
    position: int,
    source: int,
    sample_rate: int,
    rt60: float,
    dx: float,
    dy: float,
    dz: float,
    az: float,
    el: float,
) -> dict[str, Any]:
    """
    Create one metadata entry.

    Only columns available from the ARNI data are included, and their order
    follows the matching columns in the synthetic metadata.
    """
    return {
        "Name": Path(original_name).stem,
        "room_idx": room_idx,
        "position_id": f"{position}_S{source}",
        "sample_rate": sample_rate,
        "RT": rt60,
        "accdoa_x": dx,
        "accdoa_y": dy,
        "accdoa_z": dz,
        "az_deg": az,
        "el_deg": el,
        "status": "Success",
    }


def make_pickle_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Create the per-RIR metadata dictionary stored inside the pickle.

    Room-level fields such as room_idx and sample_rate are already available
    elsewhere in the pickle structure, so they are not duplicated here.
    """
    return {
        "Name": metadata["Name"],
        "RT": metadata["RT"],
        "accdoa_x": metadata["accdoa_x"],
        "accdoa_y": metadata["accdoa_y"],
        "accdoa_z": metadata["accdoa_z"],
        "az_deg": metadata["az_deg"],
        "el_deg": metadata["el_deg"],
        "status": metadata["status"],
    }


def convert_arni(
    csv_path: Path,
    wav_dir: Path,
    output_dir: Path,
    *,
    strict: bool = True,
) -> None:
    """
    Convert ARNI WAV files and metadata into room-level pickle files.

    Args:
        csv_path:
            Input ARNI metadata CSV.

        wav_dir:
            Directory containing the WAV files listed in the CSV.

        output_dir:
            Directory where pickle files and arni_metadata.csv are written.

        strict:
            If True, stop when a listed WAV file is missing.
            If False, print a warning and skip missing WAV files.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    if not wav_dir.is_dir():
        raise NotADirectoryError(f"WAV directory not found: {wav_dir}")

    metadata_df = pd.read_csv(csv_path)

    missing_columns = REQUIRED_COLUMNS.difference(metadata_df.columns)

    if missing_columns:
        raise ValueError(
            "CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    duplicate_mask = metadata_df["Name"].duplicated(keep=False)

    if duplicate_mask.any():
        duplicates = (
            metadata_df.loc[duplicate_mask, "Name"]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Duplicate CSV filenames found: "
            f"{duplicates[:10]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    rooms: dict[str, dict[str, Any]] = {}
    combined_metadata: list[dict[str, Any]] = []
    skipped: list[str] = []

    parsed_names = metadata_df["Name"].astype(str).map(parse_name)
    metadata_df = metadata_df.assign(
        _room_name=parsed_names.map(lambda item: item[0]),
        _position_num=parsed_names.map(lambda item: item[1]),
        _source_num=parsed_names.map(lambda item: item[2]),
    )

    parsed_room_names = set(metadata_df["_room_name"])
    room_name_to_idx = {
        room_name: room_idx
        for room_idx, room_name in enumerate(
            sorted(parsed_room_names, key=natural_sort_key)
        )
    }

    # Process rows in natural numeric order: P1_S1, P1_S2, ..., P10_S1.
    # This also preserves that order in each room's ``rirs`` dictionary.
    metadata_df = (
        metadata_df
        .assign(
            _room_idx=metadata_df["_room_name"].map(room_name_to_idx)
        )
        .sort_values(
            by=["_room_idx", "_position_num", "_source_num"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    for row in metadata_df.itertuples(index=False):
        name = str(row.Name)

        room_name, position, source, rir_key = parse_name(name)
        room_idx = room_name_to_idx[room_name]
        wav_path = wav_dir / name

        if not wav_path.is_file():
            message = f"Missing WAV: {wav_path}"

            if strict:
                raise FileNotFoundError(message)

            print(f"WARNING: {message}")
            skipped.append(name)
            continue

        rt60 = validate_numeric_value(
            row.rt60,
            column_name="rt60",
            filename=name,
        )
        dx = validate_numeric_value(
            row.dx,
            column_name="dx",
            filename=name,
        )
        dy = validate_numeric_value(
            row.dy,
            column_name="dy",
            filename=name,
        )
        dz = validate_numeric_value(
            row.dz,
            column_name="dz",
            filename=name,
        )
        az = validate_numeric_value(
            row.az,
            column_name="az",
            filename=name,
        )
        el = validate_numeric_value(
            row.el,
            column_name="el",
            filename=name,
        )

        sample_rate, rir = load_rir_channels_first(wav_path)

        if room_name not in rooms:
            rooms[room_name] = {
                "room_idx": room_idx,
                "sample_rate": sample_rate,
                "rirs": {},
            }

        room_entry = rooms[room_name]

        if sample_rate != room_entry["sample_rate"]:
            raise ValueError(
                f"Room {room_name!r} contains different sample rates: "
                f"{room_entry['sample_rate']} Hz and {sample_rate} Hz "
                f"({name})"
            )

        if rir_key in room_entry["rirs"]:
            raise ValueError(
                f"Duplicate RIR key {rir_key!r} "
                f"in room {room_name!r}"
            )

        metadata = make_metadata(
            original_name=name,
            room_idx=room_idx,
            position=position,
            source=source,
            sample_rate=sample_rate,
            rt60=rt60,
            dx=dx,
            dy=dy,
            dz=dz,
            az=az,
            el=el,
        )

        room_entry["rirs"][rir_key] = {
            "rir": rir,
            "metadata": make_pickle_metadata(metadata),
        }

        combined_metadata.append(metadata)

    if not rooms:
        raise RuntimeError(
            "No room pickle files were created. "
            "Check the input CSV, WAV directory, and filename pattern."
        )

    for room_name, room_entry in sorted(rooms.items()):
        output_path = output_dir / f"{room_name}.pkl"

        with output_path.open("wb") as file:
            pickle.dump(
                room_entry,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(
            f"Saved {len(room_entry['rirs']):4d} RIRs "
            f"for room {room_name!r} -> {output_path}"
        )

    combined_metadata_df = pd.DataFrame(
        combined_metadata,
        columns=METADATA_COLUMNS,
    )

    # ``position_id`` is text, so lexical sorting would place ``10_S1``
    # before ``1_S1``. Extract the numeric parts and sort by them instead.
    position_parts = combined_metadata_df["position_id"].str.extract(
        r"^(?P<_position_num>\d+)_S(?P<_source_num>\d+)$"
    )

    if position_parts.isna().any().any():
        invalid_ids = combined_metadata_df.loc[
            position_parts.isna().any(axis=1), "position_id"
        ].tolist()
        raise ValueError(
            "Unexpected position_id format while sorting: "
            f"{invalid_ids[:10]}"
        )

    combined_metadata_df = (
        combined_metadata_df
        .assign(
            _position_num=position_parts["_position_num"].astype(int),
            _source_num=position_parts["_source_num"].astype(int),
        )
        .sort_values(
            by=["room_idx", "_position_num", "_source_num"],
            kind="stable",
        )
        .drop(columns=["_position_num", "_source_num"])
        .reset_index(drop=True)
    )

    combined_csv_path = output_dir / "01_metadata.csv"

    combined_metadata_df.to_csv(
        combined_csv_path,
        index=False,
    )

    print(
        f"Saved {len(combined_metadata_df):4d} metadata rows "
        f"-> {combined_csv_path}"
    )

    print(
        f"Done: wrote {len(rooms)} room pickle file(s), "
        f"one combined metadata CSV, and skipped "
        f"{len(skipped)} WAV file(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ARNI WAV RIRs to the synthetic room-level "
            "pickle format."
        )
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=(
            "Path to arni_directions_rt60.csv. "
            f"Default: {DEFAULT_CSV_PATH}"
        ),
    )

    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=DEFAULT_WAV_DIR,
        help=(
            "Directory containing the ARNI WAV files. "
            f"Default: {DEFAULT_WAV_DIR}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where the pickle files and metadata CSV "
            f"will be written. Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )

    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help=(
            "Warn and continue instead of failing when a WAV file "
            "listed in the CSV is missing."
        ),
    )

    args = parser.parse_args()

    convert_arni(
        csv_path=args.csv,
        wav_dir=args.wav_dir,
        output_dir=args.output_dir,
        strict=not args.skip_missing,
    )


if __name__ == "__main__":
    main()