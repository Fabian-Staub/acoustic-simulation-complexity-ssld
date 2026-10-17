import os
import pickle
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def find_pickle_files(input_folder: Path, recursive: bool) -> list[Path]:
    """
    Find all .pkl and .pickle files in the input folder.

    Parameters
    ----------
    input_folder : pathlib.Path
        Folder containing the pickle files.
    recursive : bool
        If True, search through subfolders as well.

    Returns
    -------
    list[pathlib.Path]
        Sorted list of pickle files.
    """
    patterns = ("*.pkl", "*.pickle")

    pickle_files = []

    for pattern in patterns:
        if recursive:
            pickle_files.extend(input_folder.rglob(pattern))
        else:
            pickle_files.extend(input_folder.glob(pattern))

    return sorted(set(pickle_files))


def make_wav_filename(
    rir_key: str,
    metadata: dict,
    pickle_path: Path,
) -> str:
    """
    Determine the WAV filename.

    The original simulation name stored in metadata["Name"] is preferred.
    If it is unavailable, a filename is generated from the pickle name and
    the RIR dictionary key.
    """
    original_name = metadata.get("Name")

    if original_name:
        original_name = str(original_name)

        # Remove an existing extension before adding .wav.
        wav_name = Path(original_name).stem + ".wav"
    else:
        safe_key = str(rir_key).replace("/", "_").replace("\\", "_")
        wav_name = f"{pickle_path.stem}_{safe_key}.wav"

    return wav_name


def prepare_rir_for_wav(rir: np.ndarray) -> np.ndarray:
    """
    Convert an RIR from channels x samples to samples x channels.

    scipy.io.wavfile.write expects:
        mono: samples
        multichannel: samples x channels
    """
    rir = np.asarray(rir)

    if rir.ndim == 1:
        # Mono RIR: already samples.
        wav_data = rir

    elif rir.ndim == 2:
        # Pickles store RIRs as channels x samples.
        wav_data = rir.T

    else:
        raise ValueError(
            f"RIR must be one- or two-dimensional, got shape {rir.shape}"
        )

    if not np.all(np.isfinite(wav_data)):
        raise ValueError("RIR contains NaN or infinite values")

    return np.asarray(wav_data, dtype=np.float32)


def convert_pickle_file(
    pickle_path: Path,
    output_folder: Path,
    overwrite: bool,
) -> tuple[int, int]:
    """
    Convert every RIR inside one room pickle to a WAV file.

    Expected pickle structure
    -------------------------
    {
        "room_idx": 0,
        "sample_rate": 16000,
        "rirs": {
            "0_S0": {
                "rir": np.ndarray,       # channels x samples
                "metadata": {
                    "Name": "R0000_P0_S0_IR",
                    ...
                }
            },
            ...
        }
    }

    Returns
    -------
    tuple[int, int]
        Number of WAV files written and skipped.
    """
    with pickle_path.open("rb") as file:
        room_data = pickle.load(file)

    if not isinstance(room_data, dict):
        raise TypeError(
            f"{pickle_path.name}: pickle root must be a dictionary"
        )

    sample_rate = room_data.get("sample_rate")
    if sample_rate is None:
        raise KeyError(
            f"{pickle_path.name}: missing top-level 'sample_rate'"
        )

    sample_rate = int(sample_rate)

    if sample_rate <= 0:
        raise ValueError(
            f"{pickle_path.name}: invalid sample rate {sample_rate}"
        )

    rirs = room_data.get("rirs")
    if not isinstance(rirs, dict):
        raise KeyError(
            f"{pickle_path.name}: missing or invalid top-level 'rirs'"
        )

    written = 0
    skipped = 0

    for rir_key, rir_entry in rirs.items():
        if not isinstance(rir_entry, dict):
            raise TypeError(
                f"{pickle_path.name}, entry {rir_key!r}: "
                "RIR entry must be a dictionary"
            )

        if "rir" not in rir_entry:
            raise KeyError(
                f"{pickle_path.name}, entry {rir_key!r}: "
                "missing 'rir' array"
            )

        rir = rir_entry["rir"]

        metadata = rir_entry.get("metadata", {})
        if metadata is None:
            metadata = {}

        if not isinstance(metadata, dict):
            raise TypeError(
                f"{pickle_path.name}, entry {rir_key!r}: "
                "'metadata' must be a dictionary"
            )

        wav_filename = make_wav_filename(
            rir_key=rir_key,
            metadata=metadata,
            pickle_path=pickle_path,
        )

        wav_path = output_folder / wav_filename

        if wav_path.exists() and not overwrite:
            print(f"Skipping existing file: {wav_path}")
            skipped += 1
            continue

        wav_data = prepare_rir_for_wav(rir)

        wavfile.write(
            filename=str(wav_path),
            rate=sample_rate,
            data=wav_data,
        )

        print(
            f"Written: {wav_path} "
            f"(fs={sample_rate}, shape={wav_data.shape}, "
            f"dtype={wav_data.dtype})"
        )

        written += 1

    return written, skipped


def main() -> None:
    # ------------------------------------------------------------------
    # Edit these settings.
    # ------------------------------------------------------------------

    input_folder = Path(
        r"/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/test/RIRs"
    )

    output_folder = Path(
        r"/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/test/WAVs"
    )

    # Search for pickle files inside subfolders as well.
    recursive = False

    # Replace WAV files that already exist.
    overwrite = True

    # Stop immediately when one pickle or RIR cannot be converted.
    # When False, errors are printed and conversion continues.
    fail_fast = False

    # ------------------------------------------------------------------
    # Validation and conversion.
    # ------------------------------------------------------------------

    input_folder = input_folder.expanduser().resolve()
    output_folder = output_folder.expanduser().resolve()

    if not input_folder.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {input_folder}"
        )

    if not input_folder.is_dir():
        raise NotADirectoryError(
            f"Input path is not a folder: {input_folder}"
        )

    output_folder.mkdir(parents=True, exist_ok=True)

    pickle_files = find_pickle_files(
        input_folder=input_folder,
        recursive=recursive,
    )

    if not pickle_files:
        print(f"No .pkl or .pickle files found in: {input_folder}")
        return

    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Pickle files:  {len(pickle_files)}")
    print()

    total_written = 0
    total_skipped = 0
    failed_pickles = 0

    for index, pickle_path in enumerate(pickle_files, start=1):
        print(
            f"[{index}/{len(pickle_files)}] "
            f"Processing {pickle_path}"
        )

        try:
            written, skipped = convert_pickle_file(
                pickle_path=pickle_path,
                output_folder=output_folder,
                overwrite=overwrite,
            )

            total_written += written
            total_skipped += skipped

        except Exception as error:
            failed_pickles += 1

            print(
                f"Error while processing {pickle_path}: "
                f"{type(error).__name__}: {error}"
            )

            if fail_fast:
                raise

        print()

    print("Conversion finished.")
    print(f"WAV files written: {total_written}")
    print(f"WAV files skipped: {total_skipped}")
    print(f"Pickles failed:     {failed_pickles}")


if __name__ == "__main__":
    main()