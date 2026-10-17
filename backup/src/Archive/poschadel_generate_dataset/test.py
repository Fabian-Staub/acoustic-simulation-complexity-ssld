import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.io import wavfile


def sanitize_filename(name: str) -> str:
    """Make a string safe to use as part of a filename."""
    safe = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in name
    )
    return safe.strip("._") or "audio"


def detect_sample_rate(obj: Any, default_sample_rate: int) -> int:
    """
    Try to detect a sample rate stored inside a dictionary.

    Falls back to default_sample_rate if none is found.
    """
    if not isinstance(obj, dict):
        return default_sample_rate

    possible_keys = (
        "sample_rate",
        "samplerate",
        "sampling_rate",
        "sampling_frequency",
        "fs",
        "sr",
    )

    for key in possible_keys:
        if key in obj:
            try:
                sample_rate = int(obj[key])
                if sample_rate > 0:
                    return sample_rate
            except (TypeError, ValueError):
                pass

    return default_sample_rate


def prepare_audio(
    array: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert an array into WAV-compatible float32 audio.

    Expected output:
        mono:   (samples,)
        stereo: (samples, channels)
    """
    audio = np.asarray(array)

    # Remove dimensions whose size is 1.
    audio = np.squeeze(audio)

    if audio.ndim == 0:
        raise ValueError("Scalar values cannot be saved as audio.")

    if audio.ndim > 2:
        raise ValueError(
            f"Expected a one- or two-dimensional audio array, got shape {audio.shape}."
        )

    # Convert complex impulse responses to their real component.
    if np.iscomplexobj(audio):
        print("  Complex-valued array detected; using the real component.")
        audio = np.real(audio)

    audio = audio.astype(np.float64, copy=False)

    # Replace invalid values before normalization.
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if audio.ndim == 2:
        rows, columns = audio.shape

        # scipy expects shape (samples, channels).
        # If the first dimension appears to be channels, transpose it.
        if rows <= 32 and columns > rows:
            audio = audio.T

    if normalize:
        peak = np.max(np.abs(audio))

        if peak > 0:
            # Leave a small amount of headroom.
            audio = 0.99 * audio / peak

    return audio.astype(np.float32)


def save_array_as_wav(
    array: np.ndarray,
    output_path: Path,
    sample_rate: int,
    normalize: bool,
) -> bool:
    """Save an array as a WAV file if it has a plausible audio shape."""
    try:
        audio = prepare_audio(array, normalize=normalize)
    except ValueError as exc:
        print(f"Skipping {output_path.stem}: {exc}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, sample_rate, audio)

    print(
        f"Saved: {output_path} "
        f"| shape={audio.shape} "
        f"| sample_rate={sample_rate} Hz"
    )
    return True


def convert_object_to_wavs(
    obj: Any,
    output_directory: Path,
    sample_rate: int,
    normalize: bool = True,
    name: str = "rir",
) -> int:
    """
    Recursively search an object and save array-like values as WAV files.

    Returns the number of WAV files written.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    name = sanitize_filename(name)

    # NumPy array
    if isinstance(obj, np.ndarray):
        output_path = output_directory / f"{name}.wav"

        return int(
            save_array_as_wav(
                array=obj,
                output_path=output_path,
                sample_rate=sample_rate,
                normalize=normalize,
            )
        )

    # pandas DataFrame or Series
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            return convert_object_to_wavs(
                obj=obj.to_numpy(),
                output_directory=output_directory,
                sample_rate=sample_rate,
                normalize=normalize,
                name=name,
            )

        if isinstance(obj, pd.Series):
            return convert_object_to_wavs(
                obj=obj.to_numpy(),
                output_directory=output_directory,
                sample_rate=sample_rate,
                normalize=normalize,
                name=name,
            )

    except ImportError:
        pass

    # Dictionary
    if isinstance(obj, dict):
        files_written = 0
        local_sample_rate = detect_sample_rate(obj, sample_rate)

        # Keys that usually contain metadata rather than audio.
        metadata_keys = {
            "sample_rate",
            "samplerate",
            "sampling_rate",
            "sampling_frequency",
            "fs",
            "sr",
        }

        for key, value in obj.items():
            if str(key).lower() in metadata_keys:
                continue

            child_name = f"{name}_{sanitize_filename(str(key))}"

            files_written += convert_object_to_wavs(
                obj=value,
                output_directory=output_directory,
                sample_rate=local_sample_rate,
                normalize=normalize,
                name=child_name,
            )

        return files_written

    # List or tuple
    if isinstance(obj, (list, tuple)):
        # First try to treat a plain numerical list as one audio signal.
        try:
            array = np.asarray(obj)

            if array.dtype != object and array.ndim in (1, 2):
                return convert_object_to_wavs(
                    obj=array,
                    output_directory=output_directory,
                    sample_rate=sample_rate,
                    normalize=normalize,
                    name=name,
                )
        except (TypeError, ValueError):
            pass

        files_written = 0

        for index, value in enumerate(obj):
            files_written += convert_object_to_wavs(
                obj=value,
                output_directory=output_directory,
                sample_rate=sample_rate,
                normalize=normalize,
                name=f"{name}_{index:04d}",
            )

        return files_written

    # Ignore strings, numbers, metadata, model objects, etc.
    return 0


def convert_pickle_to_wavs(
    pickle_path: str,
    output_directory: Optional[str] = None,
    sample_rate: int = 48_000,
    normalize: bool = True,
) -> None:
    """
    Load a pickle and convert all suitable arrays inside it to WAV files.

    Only open pickle files from trusted sources because pickle loading can
    execute arbitrary Python code.
    """
    input_path = Path(pickle_path)

    if not input_path.is_file():
        print(f"File not found: {input_path}")
        return

    if output_directory is None:
        output_path = input_path.parent / f"{input_path.stem}_wavs"
    else:
        output_path = Path(output_directory)

    print(f"Input: {input_path.resolve()}")
    print(f"Output directory: {output_path.resolve()}")
    print(f"Default sample rate: {sample_rate} Hz")
    print(f"Normalize: {normalize}\n")

    try:
        with input_path.open("rb") as file:
            obj = pickle.load(file)
    except Exception as exc:
        print(f"Could not load pickle: {type(exc).__name__}: {exc}")
        return

    files_written = convert_object_to_wavs(
        obj=obj,
        output_directory=output_path,
        sample_rate=sample_rate,
        normalize=normalize,
        name=input_path.stem,
    )

    print(f"\nFinished. Created {files_written} WAV file(s).")


pickle_path = (
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/"
    "2026_fstaub/staubf1/Projects/Master_Thesis/data/"
    "harp/test/RIRs/R0000.pkl"
)

convert_pickle_to_wavs(
    pickle_path=pickle_path,
    sample_rate=24_000,  # Change this to the actual RIR sampling rate.
    normalize=False,
)