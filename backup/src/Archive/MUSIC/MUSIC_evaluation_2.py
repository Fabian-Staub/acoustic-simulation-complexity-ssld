#!/usr/bin/env python3
"""Broadband spherical-harmonic MUSIC DOA estimation.

The program reads SH-domain multichannel WAV files and frame-level ground-truth
metadata. Named dataset arguments are resolved to their ``test/wav/`` and
``test/csv/`` subdirectories; otherwise ``--wav-dir`` and ``--csv-dir`` select
the input directories directly. Audio is processed in one-second segments,
with exactly 50 STFT frames per segment. The active-source count for every
frame comes from the matching room CSV; GT directions are never used by MUSIC.
FFT bins are pooled with overlapping triangular Slaney mel filters, matching
the dataset's log-mel frequency axis.  ``--num-bands`` sets the number of mel
bins used for broadband MUSIC.

All columns from the room CSVs are retained. MUSIC predictions use the
standard ``prediction_*`` output schema; additional ``prediction_music_*``
columns contain MUSIC-specific diagnostics. The ``prediction_dataset`` column
identifies the named dataset that produced each prediction.

Frames are evaluated by a persistent process pool. The worker count defaults
to SLURM_CPUS_PER_TASK when available; override it with --workers. BLAS is
limited to one thread per process unless MUSIC_BLAS_THREADS is explicitly set.
Pass --test-mode to analyze only the first 10 WAV files in filename order.

Expected naming example
-----------------------
WAV ``R0000_anisotropic_P1.wav`` matches metadata sample_id ``R0000_P01``.

SH convention
-------------
Channels must use ACN ordering and contain a complete SH order, i.e. 1, 4, 9,
16, 25, ... channels.  N3D is the default; pass ``--normalization sn3d`` for
SN3D files.  The search directions use spaudiopy's spherical n-design with
degree 124 by default.  Elevation is positive upward and azimuth is positive
counter-clockwise from +x toward +y.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

# Set these before importing NumPy/SciPy. Cluster environments often export
# OPENBLAS_NUM_THREADS=32, which otherwise turns 16 workers into 512 threads.
# MUSIC_BLAS_THREADS is an explicit escape hatch for benchmarking.
_BLAS_THREADS = os.environ.get("MUSIC_BLAS_THREADS", "1")
for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = _BLAS_THREADS

import numpy as np
import pandas as pd
from scipy.io import wavfile

try:
    import spaudiopy as spa
except ImportError:  # Report a focused error only when the grid is requested.
    spa = None

try:
    # SciPy >= 1.15 (theta=colatitude, phi=azimuth).
    from scipy.special import sph_harm_y as _sph_harm_y

    def complex_sph_harm(m: int, n: int, az: np.ndarray, colatitude: np.ndarray) -> np.ndarray:
        return _sph_harm_y(n, m, colatitude, az)

except ImportError:  # pragma: no cover - for older SciPy installations
    from scipy.special import sph_harm as _legacy_sph_harm

    def complex_sph_harm(m: int, n: int, az: np.ndarray, colatitude: np.ndarray) -> np.ndarray:
        return _legacy_sph_harm(m, n, az, colatitude)


NUM_FRAMES = 50
HOP_LENGTH = 320
FRAME_LENGTH = 640
N_FFT = 1024
EPS = 1.0e-12


def default_worker_count() -> int:
    """Honor Slurm's allocation before falling back to CPU affinity."""
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocation:
        try:
            return max(1, int(allocation))
        except ValueError:
            warnings.warn(f"invalid SLURM_CPUS_PER_TASK={allocation!r}; ignoring it")
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class Config:
    num_frames: int = NUM_FRAMES
    hop_length: int = HOP_LENGTH
    frame_length: int = FRAME_LENGTH
    n_fft: int = N_FFT
    num_bands: int = 10
    f_min: float = 0.0
    f_max: float = 8000.0
    grid_degree: int = 124
    min_peak_separation_deg: float = 10.0
    normalization: str = "n3d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-dir", type=Path, default=Path("wav"))
    parser.add_argument("--csv-dir", type=Path, default=Path("csv"))
    parser.add_argument(
        "--checkpoint_dataset",
        "--checkpoint-dataset",
        action="store_true",
        help="accept the checkpoint-dataset evaluation flag for CLI compatibility",
    )
    for dataset_name in ("arni", "motus", "ivan"):
        parser.add_argument(
            f"--{dataset_name}-dataset",
            type=Path,
            metavar="DATASET_ROOT",
            help=(
                f"evaluate {dataset_name}-dataset using DATASET_ROOT/test/wav "
                "and DATASET_ROOT/test/csv"
            ),
        )
    parser.add_argument(
        "--output", type=Path, default=Path("predictions_MUSIC.csv")
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=10,
        help="number of triangular Slaney mel-frequency bins (default: 10)",
    )
    parser.add_argument("--f-min", type=float, default=0.0)
    parser.add_argument("--f-max", type=float, default=math.inf)
    parser.add_argument(
        "--grid-degree",
        type=int,
        default=124,
        help="spaudiopy spherical n-design degree (default: 124)",
    )
    parser.add_argument("--min-peak-separation-deg", type=float, default=15.0)
    parser.add_argument(
        "--normalization", choices=("n3d", "sn3d"), default="n3d"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="worker processes; defaults to SLURM_CPUS_PER_TASK or available CPUs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="frames per worker task; 0 chooses an adaptive size (default)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="analyze only the first 10 WAV files in filename order",
    )
    return parser.parse_args()


def canonical_sample_id(value: object) -> str:
    """Normalize IDs such as R0000_P1 and R0000_P01 to R0000_P01."""
    text = str(value).strip()
    match = re.search(r"(?i)(R\d+).*?_P(\d+)(?:\D|$)", text)
    if not match:
        return text.casefold()
    room = match.group(1).upper()
    position = int(match.group(2))
    return f"{room}_P{position:02d}"


def wav_sample_id(path: Path) -> str:
    return canonical_sample_id(path.stem)


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    fs, audio = wavfile.read(path)
    audio = np.asarray(audio)
    if audio.ndim != 2:
        raise ValueError(f"{path}: expected multichannel WAV, got {audio.shape}")
    # scipy.io.wavfile returns (samples, channels).
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        audio = audio.astype(np.float64) / scale
    else:
        audio = audio.astype(np.float64, copy=False)
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{path}: audio contains NaN or infinity")
    return int(fs), np.ascontiguousarray(audio.T)


def infer_sh_order(n_channels: int) -> int:
    order = math.isqrt(n_channels) - 1
    if (order + 1) ** 2 != n_channels:
        raise ValueError(
            f"SH audio has {n_channels} channels; a complete ACN set must have "
            "(order + 1)^2 channels"
        )
    return order


def n_design_grid(degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load spaudiopy's paired spherical n-design directions.

    Returns unit Cartesian vectors, azimuths, and elevations.  Each row is one
    direction; no meshgrid is formed.
    """
    if degree <= 0:
        raise ValueError("grid_degree must be positive")
    if spa is None:
        raise ImportError(
            "spaudiopy is required for the n-design grid; install it with "
            "`pip install spaudiopy`"
        )
    xyz = np.asarray(spa.grids.load_n_design(degree=degree), dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"spaudiopy returned an invalid n-design shape {xyz.shape}; "
            "expected (directions, 3)"
        )
    norms = np.linalg.norm(xyz, axis=1)
    if np.any(norms <= EPS):
        raise ValueError("spaudiopy n-design contains a zero-length direction")
    xyz = xyz / norms[:, None]
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    zen = np.arccos(np.clip(xyz[:, 2], -1.0, 1.0))
    el = np.pi / 2.0 - zen
    return np.ascontiguousarray(xyz), az, el


def real_sh_matrix(
    order: int, az: np.ndarray, el: np.ndarray, normalization: str
) -> np.ndarray:
    """Real ACN SH matrix, shape ((order+1)^2, directions)."""
    az = np.asarray(az, dtype=np.float64)
    colatitude = np.pi / 2.0 - np.asarray(el, dtype=np.float64)
    matrix = np.empty(((order + 1) ** 2, az.size), dtype=np.float64)
    for n in range(order + 1):
        degree_scale = np.sqrt(4.0 * np.pi)
        if normalization == "sn3d":
            degree_scale /= np.sqrt(2 * n + 1)
        for m in range(-n, n + 1):
            # scipy's complex SH includes the Condon-Shortley phase.  The
            # (-1)^m factor converts it to the common ambisonic real basis.
            y = complex_sph_harm(abs(m), n, az, colatitude)
            if m < 0:
                real_y = np.sqrt(2.0) * ((-1.0) ** m) * y.imag
            elif m == 0:
                real_y = y.real
            else:
                real_y = np.sqrt(2.0) * ((-1.0) ** m) * y.real
            matrix[n * n + n + m] = degree_scale * real_y
    # Unit-norm columns remove arbitrary global and degree-convention scale
    # from the pseudospectrum denominator.
    matrix /= np.maximum(np.linalg.norm(matrix, axis=0, keepdims=True), EPS)
    return matrix


MelBand = tuple[np.ndarray, np.ndarray]


def _hz_to_slaney_mel(frequencies: np.ndarray) -> np.ndarray:
    """Match librosa's default mel scale: linear below 1 kHz, then logarithmic."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    mels = frequencies / (200.0 / 3.0)
    logarithmic = frequencies >= 1000.0
    if np.any(logarithmic):
        mels[logarithmic] = 15.0 + np.log(
            frequencies[logarithmic] / 1000.0
        ) / (np.log(6.4) / 27.0)
    return mels


def _slaney_mel_to_hz(mels: np.ndarray) -> np.ndarray:
    """Convert librosa-compatible Slaney mel coordinates back to Hertz."""
    mels = np.asarray(mels, dtype=np.float64)
    frequencies = (200.0 / 3.0) * mels
    logarithmic = mels >= 15.0
    if np.any(logarithmic):
        frequencies[logarithmic] = 1000.0 * np.exp(
            (np.log(6.4) / 27.0) * (mels[logarithmic] - 15.0)
        )
    return frequencies


def mel_frequency_groups(
    fs: int, config: Config
) -> tuple[np.ndarray, list[MelBand]]:
    """Return sparse triangular mel filters identical to librosa's defaults."""
    if config.num_bands < 1:
        raise ValueError("--num-bands must be at least 1")
    freqs = np.fft.rfftfreq(config.n_fft, d=1.0 / fs)
    f_max = min(config.f_max, fs / 2.0)
    if not np.isfinite(config.f_min) or not np.isfinite(f_max):
        raise ValueError("frequency limits must be finite")
    if not 0.0 <= config.f_min < f_max:
        raise ValueError(
            f"frequency range [{config.f_min}, {config.f_max}] is invalid "
            f"for sample rate {fs}"
        )
    mel_limits = _hz_to_slaney_mel(
        np.asarray([config.f_min, f_max], dtype=np.float64)
    )
    edges = _slaney_mel_to_hz(
        np.linspace(mel_limits[0], mel_limits[1], config.num_bands + 2)
    )

    groups: list[MelBand] = []
    empty_bands: list[int] = []
    for band_idx in range(config.num_bands):
        low, center, high = edges[band_idx : band_idx + 3]
        ascending = (freqs - low) / (center - low)
        descending = (high - freqs) / (high - center)
        weights = np.maximum(0.0, np.minimum(ascending, descending))
        bins = np.flatnonzero(weights > 0.0)
        if bins.size == 0:
            empty_bands.append(band_idx)
            continue
        groups.append((bins, np.ascontiguousarray(weights[bins])))

    if empty_bands:
        raise ValueError(
            f"mel basis contains {len(empty_bands)} empty filters; reduce "
            f"--num-bands or increase FFT resolution "
            f"(fs={fs}, n_fft={config.n_fft}, num_bands={config.num_bands}, "
            f"f_min={config.f_min}, f_max={f_max})"
        )
    return freqs, groups


def stft_frame(
    segment: np.ndarray, frame_idx: int, config: Config
) -> np.ndarray:
    """One zero-padded STFT frame, returned as (channels, frequencies)."""
    start = frame_idx * config.hop_length
    stop = start + config.frame_length
    frame = np.zeros((segment.shape[0], config.frame_length), dtype=np.float64)
    available = segment[:, start:min(stop, segment.shape[1])]
    frame[:, : available.shape[1]] = available
    frame -= np.mean(frame, axis=1, keepdims=True)
    frame *= np.hanning(config.frame_length)[None, :]
    return np.fft.rfft(frame, n=config.n_fft, axis=1)


def stft_frames(
    segment: np.ndarray,
    frame_indices: np.ndarray,
    config: Config,
    window: np.ndarray,
) -> np.ndarray:
    """Vectorized zero-padded STFT, shaped (frames, channels, frequencies)."""
    indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.size == 0:
        return np.empty(
            (0, segment.shape[0], config.n_fft // 2 + 1), dtype=np.complex128
        )
    last_sample = int(indices.max()) * config.hop_length + config.frame_length
    padded_length = max(segment.shape[1], last_sample)
    if padded_length == segment.shape[1]:
        padded = segment
    else:
        padded = np.zeros((segment.shape[0], padded_length), dtype=np.float64)
        padded[:, : segment.shape[1]] = segment
    sample_indices = (
        indices[:, None] * config.hop_length
        + np.arange(config.frame_length, dtype=np.int64)[None, :]
    )
    frames = np.moveaxis(padded[:, sample_indices], 0, 1)
    frames -= np.mean(frames, axis=2, keepdims=True)
    frames *= window[None, None, :]
    return np.fft.rfft(frames, n=config.n_fft, axis=2)


def band_covariances(
    spectrum: np.ndarray, groups: list[MelBand]
) -> np.ndarray:
    """Mel-filter-weighted complex Hermitian SCM for each frequency band."""
    covariances = []
    n_channels = spectrum.shape[0]
    identity = np.eye(n_channels, dtype=np.complex128)
    for bins, weights in groups:
        x = spectrum[:, bins]
        covariance = ((x * weights[None, :]) @ x.conj().T) / np.sum(weights)
        # Trace normalization is scale invariant and diagonal loading makes
        # the eigendecomposition stable for silent or nearly silent frames.
        trace = float(np.trace(covariance).real)
        if trace > EPS:
            covariance /= trace
        covariance += EPS * identity
        covariances.append(covariance)
    return np.stack(covariances)


def broadband_music(
    covariances: np.ndarray, steering: np.ndarray, n_sources: int
) -> np.ndarray:
    """Incoherently average per-band SH-MUSIC pseudospectra."""
    n_channels = covariances.shape[-1]
    if not 0 < n_sources < n_channels:
        raise ValueError(
            f"source count must be between 1 and {n_channels - 1}, got {n_sources}"
        )
    # np.linalg.eigh accepts a stack of Hermitian matrices, avoiding a Python
    # call for every band. The signal and noise eigenspaces are complementary.
    # For the common low-source-count case, projecting onto the much smaller
    # signal space changes the cost from O(channelsÂ² * grid) to
    # O(channels * sources * grid).
    _, eigenvectors = np.linalg.eigh(covariances)
    if n_sources < n_channels - n_sources:
        signal = eigenvectors[:, :, -n_sources:]
        projection = np.swapaxes(signal.conj(), 1, 2) @ steering
        steering_energy = np.einsum("cg,cg->g", steering, steering)
        denominator = steering_energy[None, :] - np.sum(
            projection.real**2 + projection.imag**2, axis=1
        )
    else:
        noise = eigenvectors[:, :, : n_channels - n_sources]
        projection = np.swapaxes(noise.conj(), 1, 2) @ steering
        denominator = np.sum(projection.real**2 + projection.imag**2, axis=1)
    band_spectra = 1.0 / np.maximum(denominator, EPS)
    band_spectra /= np.maximum(np.max(band_spectra, axis=1, keepdims=True), EPS)
    return np.mean(band_spectra, axis=0)


def select_spatial_peaks(
    pseudospectrum: np.ndarray,
    grid_xyz: np.ndarray,
    count: int,
    min_separation_deg: float,
) -> list[tuple[int, float]]:
    """Greedy spherical non-maximum suppression in descending MUSIC score."""
    selected: list[tuple[int, float]] = []
    threshold = np.cos(np.deg2rad(min_separation_deg))
    for idx in np.argsort(pseudospectrum)[::-1]:
        if not np.isfinite(pseudospectrum[idx]):
            continue
        if all(float(grid_xyz[idx] @ grid_xyz[j]) < threshold for j, _ in selected):
            selected.append((int(idx), float(pseudospectrum[idx])))
            if len(selected) == count:
                break
    return selected


def active_mask(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    mask = numeric.fillna(0).to_numpy() > 0
    missing = numeric.isna().to_numpy()
    if np.any(missing):
        text_true = (
            values.astype(str).str.strip().str.casefold().isin({"true", "yes", "y"})
        ).to_numpy()
        mask[missing] = text_true[missing]
    return mask


def load_metadata(csv_dir: Path) -> pd.DataFrame:
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no room CSV files found in {csv_dir}")
    frames = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        frame["prediction_source_csv"] = path.name
        frames.append(frame)
    metadata = pd.concat(frames, ignore_index=True, sort=False)
    required = {"sample_id", "segment_idx", "frame_idx", "active"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"room CSVs are missing required columns: {missing}")
    metadata["_music_key"] = metadata["sample_id"].map(canonical_sample_id)
    return metadata


REQUIRED_OUTPUT_COLUMNS = (
    "sample_id",
    "segment_idx",
    "base_name",
    "segment_start_sample",
    "segment_start_s",
    "frame_idx",
    "frame_start_sample",
    "frame_start_s",
    "slot",
    "speech_path",
    "active",
    "accdoa_x",
    "accdoa_y",
    "accdoa_z",
    "source_accdoa_x",
    "source_accdoa_y",
    "source_accdoa_z",
    "az_deg",
    "el_deg",
    "mic_orientation_az_deg",
    "pause_start_s",
    "pause_middle_s",
    "pause_end_s",
    "rir_delay_samples",
    "rir_delay_s",
    "rt60_s",
    "vad_extension_samples",
    "vad_extension_s",
    "Name",
    "source_id_raw",
    "source_index_offset",
    "RT",
    "c50",
    "export_mode",
    "ism_order",
    "rt60_125Hz",
    "rt60_250Hz",
    "rt60_500Hz",
    "rt60_1000Hz",
    "rt60_2000Hz",
    "rt60_4000Hz",
    "rt60_8000Hz",
    "rt60_16000Hz",
    "x_room",
    "y_room",
    "z_room",
    "x_source",
    "y_source",
    "z_source",
    "x_mic",
    "y_mic",
    "z_mic",
    "status",
    "max_order",
    "prediction_dataset",
    "prediction_source_csv",
    "prediction_accdoa_x",
    "prediction_accdoa_y",
    "prediction_accdoa_z",
    "prediction_activity",
    "prediction_active",
    "prediction_az_deg",
    "prediction_el_deg",
)

CANONICAL_PREDICTION_COLUMNS = (
    "prediction_accdoa_x",
    "prediction_accdoa_y",
    "prediction_accdoa_z",
    "prediction_activity",
    "prediction_active",
    "prediction_az_deg",
    "prediction_el_deg",
)

MUSIC_DIAGNOSTIC_COLUMNS = (
    "prediction_music_wav",
    "prediction_music_num_sources",
    "prediction_music_rank",
    "prediction_music_score",
    "prediction_music_status",
)


def initialize_prediction_columns(
    metadata: pd.DataFrame, dataset_name: str = "music"
) -> None:
    for column in REQUIRED_OUTPUT_COLUMNS:
        if column not in metadata:
            metadata[column] = np.nan

    # Do not leak predictions already present in an input metadata CSV.
    for column in CANONICAL_PREDICTION_COLUMNS:
        metadata[column] = (
            pd.Series(pd.NA, index=metadata.index, dtype="Int64")
            if column == "prediction_active"
            else np.nan
        )
    metadata["prediction_dataset"] = dataset_name

    text_columns = {"prediction_music_wav", "prediction_music_status"}
    for column in MUSIC_DIAGNOSTIC_COLUMNS:
        if column in text_columns:
            metadata[column] = pd.Series(None, index=metadata.index, dtype="object")
        else:
            metadata[column] = np.nan
    metadata["prediction_music_status"] = "no_matching_wav"


def initialize_frame_predictions(metadata: pd.DataFrame, indices: pd.Index) -> None:
    """Silent and inactive slots have a valid zero-valued ACCDOA prediction."""
    metadata.loc[
        indices,
        [
            "prediction_accdoa_x",
            "prediction_accdoa_y",
            "prediction_accdoa_z",
            "prediction_activity",
        ],
    ] = 0.0
    metadata.loc[indices, "prediction_active"] = 0


def segment_start_from_rows(rows: pd.DataFrame, fs: int, segment_idx: int) -> int:
    if "segment_start_sample" in rows:
        values = pd.to_numeric(rows["segment_start_sample"], errors="coerce").dropna()
        if not values.empty:
            return int(values.iloc[0])
    return int(segment_idx * fs)


@dataclass(frozen=True)
class FramePlan:
    local_frame_idx: int
    active_indices: tuple[object, ...]
    n_sources: int


@dataclass(frozen=True)
class FrameBatch:
    spectra: np.ndarray
    source_counts: tuple[int, ...]
    sample_rate: int
    sh_order: int


_WORKER_CONFIG: Config | None = None
_WORKER_GRID: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
_WORKER_STEERING: dict[int, np.ndarray] = {}
_WORKER_FREQUENCIES: dict[int, list[MelBand]] = {}


def initialize_worker(
    config: Config, grid: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Cache grid-dependent state once for the lifetime of a worker process."""
    global _WORKER_CONFIG, _WORKER_GRID, _WORKER_STEERING, _WORKER_FREQUENCIES
    _WORKER_CONFIG = config
    _WORKER_GRID = grid
    _WORKER_STEERING = {}
    _WORKER_FREQUENCIES = {}
    # This additionally handles embedding/import scenarios where NumPy was
    # loaded before this module and the early environment settings were late.
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=max(1, int(_BLAS_THREADS)))
    except (ImportError, ValueError):
        pass


def process_frame_batch(batch: FrameBatch) -> list[list[tuple[int, float]]]:
    """Run independent MUSIC frames in a process; no DataFrames are copied."""
    if _WORKER_CONFIG is None or _WORKER_GRID is None:
        raise RuntimeError("MUSIC worker was not initialized")
    config = _WORKER_CONFIG
    grid_xyz, grid_az, grid_el = _WORKER_GRID
    steering = _WORKER_STEERING.get(batch.sh_order)
    if steering is None:
        steering = real_sh_matrix(
            batch.sh_order, grid_az, grid_el, config.normalization
        )
        _WORKER_STEERING[batch.sh_order] = steering
    groups = _WORKER_FREQUENCIES.get(batch.sample_rate)
    if groups is None:
        _, groups = mel_frequency_groups(batch.sample_rate, config)
        _WORKER_FREQUENCIES[batch.sample_rate] = groups

    results: list[list[tuple[int, float]]] = []
    for spectrum, n_sources in zip(batch.spectra, batch.source_counts):
        covariance = band_covariances(spectrum, groups)
        music = broadband_music(covariance, steering, n_sources)
        results.append(
            select_spatial_peaks(
                music, grid_xyz, n_sources, config.min_peak_separation_deg
            )
        )
    return results


def effective_batch_size(frame_count: int, workers: int, requested: int) -> int:
    if requested > 0:
        return requested
    # Several tasks per process keeps uneven frames balanced while the cap
    # prevents large spectra from becoming expensive IPC payloads.
    return min(8, max(1, math.ceil(frame_count / max(1, workers * 4))))


def process_wav(
    path: Path,
    rows: pd.DataFrame,
    metadata: pd.DataFrame,
    config: Config,
    executor: Executor | None = None,
    workers: int = 1,
    batch_size: int = 0,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    fs, audio = read_audio(path)
    order = infer_sh_order(audio.shape[0])
    if grid is None:
        grid = n_design_grid(config.grid_degree)
    grid_xyz, grid_az, grid_el = grid
    # Validate the frequency configuration in the parent, even when a file
    # has no active frames and never sends work to a child process.
    mel_frequency_groups(fs, config)
    window = np.hanning(config.frame_length)
    if executor is None:
        initialize_worker(config, grid)

    segment_values = pd.to_numeric(rows["segment_idx"], errors="raise").astype(int)
    for segment_idx, segment_rows in rows.groupby(segment_values, sort=True):
        start = segment_start_from_rows(segment_rows, fs, segment_idx)
        if start >= audio.shape[1]:
            metadata.loc[segment_rows.index, "prediction_music_status"] = "segment_out_of_audio"
            continue
        segment = np.zeros((audio.shape[0], fs), dtype=np.float64)
        chunk = audio[:, start : min(start + fs, audio.shape[1])]
        segment[:, : chunk.shape[1]] = chunk

        frame_values = pd.to_numeric(
            segment_rows["frame_idx"], errors="raise"
        ).astype(int)
        unique_frames = sorted(frame_values.unique())
        if len(unique_frames) > config.num_frames:
            warnings.warn(
                f"{path.name}, segment {segment_idx}: {len(unique_frames)} metadata "
                f"frames found; only the first {config.num_frames} are processed"
            )
        plans: list[FramePlan] = []
        frame_groups = {
            frame_value: group
            for frame_value, group in segment_rows.groupby(frame_values, sort=True)
        }
        for local_frame_idx, frame_value in enumerate(unique_frames[: config.num_frames]):
            frame_rows = frame_groups[frame_value]
            is_active = active_mask(frame_rows["active"])
            n_sources = int(np.count_nonzero(is_active))
            metadata.loc[frame_rows.index, "prediction_music_wav"] = path.name
            metadata.loc[
                frame_rows.index, "prediction_music_num_sources"
            ] = n_sources
            if n_sources == 0:
                initialize_frame_predictions(metadata, frame_rows.index)
                metadata.loc[frame_rows.index, "prediction_music_status"] = "no_active_sources"
                continue
            if n_sources >= audio.shape[0]:
                metadata.loc[frame_rows.index, "prediction_music_status"] = "too_many_sources"
                continue

            initialize_frame_predictions(metadata, frame_rows.index)
            active_indices = frame_rows.index[is_active].tolist()
            # MUSIC has no source identity.  Assign peak rank deterministically
            # to active metadata rows in slot order, without using GT angles.
            if "slot" in frame_rows:
                slots = pd.to_numeric(
                    frame_rows.loc[active_indices, "slot"], errors="coerce"
                ).fillna(np.inf)
                active_indices.sort(key=lambda idx: slots.at[idx])
            plans.append(
                FramePlan(local_frame_idx, tuple(active_indices), n_sources)
            )
            inactive_indices = frame_rows.index[~is_active]
            metadata.loc[inactive_indices, "prediction_music_status"] = "inactive_gt_slot"

        if not plans:
            continue
        spectra = stft_frames(
            segment,
            np.asarray([plan.local_frame_idx for plan in plans], dtype=np.int64),
            config,
            window,
        )
        task_size = effective_batch_size(len(plans), workers, batch_size)
        batches = [
            FrameBatch(
                spectra=spectra[offset : offset + task_size],
                source_counts=tuple(
                    plan.n_sources for plan in plans[offset : offset + task_size]
                ),
                sample_rate=fs,
                sh_order=order,
            )
            for offset in range(0, len(plans), task_size)
        ]
        results = (
            executor.map(process_frame_batch, batches, chunksize=1)
            if executor is not None
            else map(process_frame_batch, batches)
        )
        plans_iter = iter(plans)
        for batch_results in results:
            for peaks in batch_results:
                plan = next(plans_iter)
                for rank, (row_idx, (peak_idx, score)) in enumerate(
                    zip(plan.active_indices, peaks), start=1
                ):
                    xyz = grid_xyz[peak_idx]
                    metadata.at[row_idx, "prediction_music_rank"] = rank
                    metadata.at[row_idx, "prediction_accdoa_x"] = xyz[0]
                    metadata.at[row_idx, "prediction_accdoa_y"] = xyz[1]
                    metadata.at[row_idx, "prediction_accdoa_z"] = xyz[2]
                    # MUSIC has no calibrated activity probability: successful
                    # directions are active unit-length ACCDOA vectors.
                    metadata.at[row_idx, "prediction_activity"] = 1.0
                    metadata.at[row_idx, "prediction_active"] = 1
                    metadata.at[row_idx, "prediction_az_deg"] = np.rad2deg(
                        grid_az[peak_idx]
                    )
                    metadata.at[row_idx, "prediction_el_deg"] = np.rad2deg(
                        grid_el[peak_idx]
                    )
                    metadata.at[row_idx, "prediction_music_score"] = score
                    metadata.at[row_idx, "prediction_music_status"] = "ok"


def run(
    wav_dir: Path,
    csv_dir: Path,
    output: Path | None,
    config: Config,
    workers: int | None = None,
    batch_size: int = 0,
    test_mode: bool = False,
    dataset_name: str = "music",
) -> pd.DataFrame:
    workers = default_worker_count() if workers is None else workers
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if batch_size < 0:
        raise ValueError("--batch-size must be 0 or greater")
    metadata = load_metadata(csv_dir)
    initialize_prediction_columns(metadata, dataset_name)
    wav_files = sorted(wav_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"no WAV files found in {wav_dir}")
    if test_mode:
        wav_files = wav_files[:10]

    wavs_by_key: dict[str, Path] = {}
    for path in wav_files:
        key = wav_sample_id(path)
        if key in wavs_by_key:
            raise ValueError(
                f"multiple WAVs resolve to sample_id {key}: "
                f"{wavs_by_key[key].name}, {path.name}"
            )
        wavs_by_key[key] = path

    if test_mode:
        metadata = metadata.loc[metadata["_music_key"].isin(wavs_by_key)].copy()

    grid = n_design_grid(config.grid_degree)
    print(
        f"MUSIC: {workers} worker(s), {grid[0].shape[0]} grid directions, "
        f"{_BLAS_THREADS} BLAS thread(s) per worker",
        flush=True,
    )
    pool_context = (
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_worker,
            initargs=(config, grid),
        )
        if workers > 1
        else nullcontext(None)
    )
    grouped_rows = {
        str(key): group
        for key, group in metadata.groupby("_music_key", sort=False, dropna=False)
    }
    with pool_context as executor:
        for number, (key, path) in enumerate(wavs_by_key.items(), start=1):
            rows = grouped_rows.get(key)
            if rows is None or rows.empty:
                warnings.warn(f"{path.name}: no matching sample_id in room CSVs")
                continue
            print(f"[{number}/{len(wavs_by_key)}] {path.name} -> {key}", flush=True)
            try:
                process_wav(
                    path,
                    rows,
                    metadata,
                    config,
                    executor=executor,
                    workers=workers,
                    batch_size=batch_size,
                    grid=grid,
                )
            except Exception as exc:
                metadata.loc[rows.index, "prediction_music_status"] = f"error: {exc}"
                warnings.warn(f"{path.name}: {exc}")

    metadata.drop(columns=["_music_key"], inplace=True)
    additional_columns = [
        column for column in metadata.columns if column not in REQUIRED_OUTPUT_COLUMNS
    ]
    metadata = metadata.loc[:, [*REQUIRED_OUTPUT_COLUMNS, *additional_columns]]
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(output, index=False)
    return metadata


def main() -> int:
    args = parse_args()
    config = Config(
        num_bands=args.num_bands,
        f_min=args.f_min,
        f_max=args.f_max,
        grid_degree=args.grid_degree,
        min_peak_separation_deg=args.min_peak_separation_deg,
        normalization=args.normalization,
    )
    try:
        datasets = [
            (f"{dataset_name}-dataset", dataset_root)
            for dataset_name in ("arni", "motus", "ivan")
            if (dataset_root := getattr(args, f"{dataset_name}_dataset")) is not None
        ]
        if datasets:
            results: list[pd.DataFrame] = []
            for dataset_name, dataset_root in datasets:
                wav_dir = dataset_root / "test" / "wav"
                csv_dir = dataset_root / "test" / "csv"
                for directory in (wav_dir, csv_dir):
                    if not directory.is_dir():
                        raise FileNotFoundError(
                            f"{dataset_name}: required directory does not exist: "
                            f"{directory}"
                        )
                print(f"Evaluating {dataset_name}: {dataset_root}", flush=True)
                results.append(
                    run(
                        wav_dir,
                        csv_dir,
                        None,
                        config,
                        workers=args.workers,
                        batch_size=args.batch_size,
                        test_mode=args.test_mode,
                        dataset_name=dataset_name,
                    )
                )
            result = pd.concat(results, ignore_index=True, sort=False)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output, index=False)
        else:
            result = run(
                args.wav_dir,
                args.csv_dir,
                args.output,
                config,
                workers=args.workers,
                batch_size=args.batch_size,
                test_mode=args.test_mode,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    counts = result["prediction_music_status"].value_counts(dropna=False).to_dict()
    print(f"saved {len(result)} rows to {args.output}")
    print(f"status counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
