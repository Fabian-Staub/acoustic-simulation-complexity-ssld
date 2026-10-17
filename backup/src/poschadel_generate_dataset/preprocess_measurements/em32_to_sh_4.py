#!/usr/bin/env python3
"""
Convert 32-channel Eigenmike WAV files to the spherical-harmonic domain.

Dependencies
------------
pip install numpy scipy soundfile

Output convention
-----------------
Channel ordering:
    ACN: channel_index = n**2 + n + m

Normalization:
    Real Ambisonics N3D spherical harmonics in ACN channel ordering.

SciPy's orthonormal spherical harmonics are multiplied by sqrt(4*pi),
so the zeroth-order spatial basis function is 1. This preserves the
level convention used by the original MATLAB-compatible implementation.

Notes
-----
The exact sign convention can depend on the implementation of MATLAB getSH().
If your downstream processing expects a different real-SH sign convention,
adjust real_spherical_harmonics().
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Each worker processes a whole WAV file. Keep BLAS/FFT libraries single-threaded
# inside workers to avoid workers × library-threads oversubscription.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from scipy.special import spherical_jn, spherical_yn

try:
    from scipy.special import sph_harm as _scipy_sph_harm

    def _evaluate_sph_harm(
        n: int,
        m: int,
        azimuth: np.ndarray,
        inclination: np.ndarray,
    ) -> np.ndarray:
        return _scipy_sph_harm(m, n, azimuth, inclination)

except ImportError:
    # SciPy >= 1.15 replacement; note the changed angle argument order.
    from scipy.special import sph_harm_y as _scipy_sph_harm_y

    def _evaluate_sph_harm(
        n: int,
        m: int,
        azimuth: np.ndarray,
        inclination: np.ndarray,
    ) -> np.ndarray:
        return _scipy_sph_harm_y(n, m, inclination, azimuth)


def complex_spherical_harmonic(
    n: int,
    m: int,
    azimuth: np.ndarray,
    inclination: np.ndarray,
) -> np.ndarray:
    """
    Complex N3D spherical harmonic Y_n^m.

    sph_harm arguments:
        theta = azimuth
        phi   = polar/inclination angle
    """
    return _evaluate_sph_harm(n, m, azimuth, inclination)


# Eigenmike capsule directions: [azimuth, elevation] in degrees.
MIC_DIRS_DEG = np.array(
    [
        [0, 21],
        [32, 0],
        [0, -21],
        [328, 0],
        [0, 58],
        [45, 35],
        [69, 0],
        [45, -35],
        [0, -58],
        [315, -35],
        [291, 0],
        [315, 35],
        [91, 69],
        [90, 32],
        [90, -31],
        [89, -69],
        [180, 21],
        [212, 0],
        [180, -21],
        [148, 0],
        [180, 58],
        [225, 35],
        [249, 0],
        [225, -35],
        [180, -58],
        [135, -35],
        [111, 0],
        [135, 35],
        [269, 69],
        [270, 32],
        [270, -32],
        [271, -69],
    ],
    dtype=np.float64,
)


def acn_index(n: int, m: int) -> int:
    """Return the ACN channel index for degree n and order m."""
    return n * n + n + m


def real_spherical_harmonics(
    max_order: int,
    azimuth: np.ndarray,
    inclination: np.ndarray,
) -> np.ndarray:
    """
    Evaluate real spherical harmonics in ACN ordering.

    Parameters
    ----------
    max_order:
        Maximum SH degree.
    azimuth:
        Azimuth angles in radians.
    inclination:
        Inclination/polar angles in radians, measured down from +z.

    Returns
    -------
    y:
        Array shaped (n_positions, (max_order + 1)**2).

    Convention
    ----------
    For m < 0:
        sqrt(2) * (-1)^m * imag(Y_n^|m|)

    For m = 0:
        real(Y_n^0)

    For m > 0:
        sqrt(2) * (-1)^m * real(Y_n^m)

    SciPy's complex harmonics are orthonormal, with Y_0^0 = 1/sqrt(4*pi).
    Multiplication by sqrt(4*pi) converts them to the Ambisonics N3D
    convention, for which the zeroth-order basis function is 1. This also
    preserves the level of the original implementation.
    """
    azimuth = np.asarray(azimuth, dtype=np.float64)
    inclination = np.asarray(inclination, dtype=np.float64)

    if azimuth.shape != inclination.shape:
        raise ValueError("azimuth and inclination must have identical shapes")

    n_positions = azimuth.size
    n_sh = (max_order + 1) ** 2
    y = np.zeros((n_positions, n_sh), dtype=np.float64)

    for n in range(max_order + 1):
        for m in range(-n, n + 1):
            channel = acn_index(n, m)

            if m < 0:
                y_complex = complex_spherical_harmonic(
                    n, -m, azimuth, inclination
                )
                y[:, channel] = (
                    np.sqrt(2.0)
                    * ((-1.0) ** m)
                    * np.imag(y_complex)
                )

            elif m == 0:
                y_complex = complex_spherical_harmonic(
                    n, 0, azimuth, inclination
                )
                y[:, channel] = np.real(y_complex)

            else:
                y_complex = complex_spherical_harmonic(
                    n, m, azimuth, inclination
                )
                y[:, channel] = (
                    np.sqrt(2.0)
                    * ((-1.0) ** m)
                    * np.real(y_complex)
                )

    # Convert SciPy's orthonormal basis to Ambisonics N3D.
    # Keeping this scaling inside the array model preserves the encoder
    # filter levels used by the original MATLAB-compatible implementation.
    return np.sqrt(4.0 * np.pi) * y


def rigid_sphere_modal_coefficients(
    max_order: int,
    kr: np.ndarray,
) -> np.ndarray:
    """
    Calculate rigid-sphere modal coefficients.

    The returned coefficients correspond to the MATLAB operation:

        sphModalCoeffs(order_array, kR, 'rigid') / (4*pi)

    using

        b_n(kr) = i^n [
            j_n(kr) -
            j_n'(kr) / h_n^(2)'(kr) * h_n^(2)(kr)
        ]

    where h_n^(2) = j_n - i*y_n.

    Parameters
    ----------
    max_order:
        Maximum array model order.
    kr:
        One-dimensional array of dimensionless wave numbers.

    Returns
    -------
    coefficients:
        Complex array shaped (n_frequencies, max_order + 1).
    """
    kr = np.asarray(kr, dtype=np.float64)
    coefficients = np.zeros(
        (kr.size, max_order + 1),
        dtype=np.complex128,
    )

    nonzero = kr > 1e-12
    z = kr[nonzero]

    # Explicit DC limits.
    coefficients[0, 0] = 1.0

    for n in range(max_order + 1):
        if z.size == 0:
            continue

        jn = spherical_jn(n, z)
        yn = spherical_yn(n, z)

        jn_derivative = spherical_jn(n, z, derivative=True)
        yn_derivative = spherical_yn(n, z, derivative=True)

        # Outgoing-wave convention used here: Hankel function of second kind.
        hn = jn - 1j * yn
        hn_derivative = jn_derivative - 1j * yn_derivative

        rigid_response = jn - (
            jn_derivative / hn_derivative
        ) * hn

        coefficients[nonzero, n] = (1j**n) * rigid_response

    return coefficients


def repeat_per_order(values: np.ndarray) -> np.ndarray:
    """
    Repeat order-dependent values for all m channels.

    Input:
        [b_0, b_1, ..., b_N]

    Output:
        [b_0,
         b_1, b_1, b_1,
         b_2, b_2, b_2, b_2, b_2,
         ...]
    """
    values = np.asarray(values)
    pieces = [
        np.repeat(values[n], 2 * n + 1)
        for n in range(values.shape[-1])
    ]
    return np.concatenate(pieces)


def generate_sht_filters_regls(
    radius: float,
    mic_dirs_az_el_rad: np.ndarray,
    sht_order: int,
    filter_length: int,
    sample_rate: int,
    max_gain_db: float,
    speed_of_sound: float = 343.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate regularized rigid-sphere microphone-to-SH filters.

    This follows the structure of arraySHTfiltersTheory_regLS() in the
    supplied MATLAB code.

    Returns
    -------
    h_frequency:
        Half-spectrum filters shaped:
            (n_sh, n_mics, filter_length // 2 + 1)

    h_time:
        Centered time-domain filters shaped:
            (n_sh, n_mics, filter_length)
    """
    mic_dirs_az_el_rad = np.asarray(
        mic_dirs_az_el_rad,
        dtype=np.float64,
    )

    if mic_dirs_az_el_rad.ndim != 2:
        raise ValueError("Microphone directions must be a 2-D array")

    if mic_dirs_az_el_rad.shape[1] != 2:
        raise ValueError(
            "Microphone directions must contain [azimuth, elevation]"
        )

    if filter_length % 2 != 0:
        raise ValueError("filter_length must be even")

    n_mics = mic_dirs_az_el_rad.shape[0]
    maximum_supported_order = int(np.floor(np.sqrt(n_mics) - 1))

    if sht_order > maximum_supported_order:
        print(
            f"Warning: requested SH order {sht_order} is too high for "
            f"{n_mics} microphones. Using order "
            f"{maximum_supported_order}."
        )
        sht_order = maximum_supported_order

    frequencies = (
        np.arange(filter_length // 2 + 1, dtype=np.float64)
        * sample_rate
        / filter_length
    )

    kr = 2.0 * np.pi * frequencies * radius / speed_of_sound
    array_order = min(30, int(np.floor(2.0 * kr[-1])))

    print(f"Array model order: {array_order}")
    print(f"Output SH order:    {sht_order}")

    azimuth = mic_dirs_az_el_rad[:, 0]
    elevation = mic_dirs_az_el_rad[:, 1]
    inclination = (np.pi / 2.0) - elevation

    y_array = real_spherical_harmonics(
        array_order,
        azimuth,
        inclination,
    )

    modal_coefficients = rigid_sphere_modal_coefficients(
        array_order,
        kr,
    )

    n_frequencies = frequencies.size
    n_array_sh = (array_order + 1) ** 2
    n_output_sh = (sht_order + 1) ** 2

    h_frequency = np.zeros(
        (n_output_sh, n_mics, n_frequencies),
        dtype=np.complex128,
    )

    # MATLAB:
    # alpha = 10^(a_dB/20)
    # beta  = 1/(2*alpha)
    alpha = 10.0 ** (max_gain_db / 20.0)
    beta = 1.0 / (2.0 * alpha)
    regularization = (beta**2) * np.eye(n_mics)

    for frequency_index in range(n_frequencies):
        repeated_modal = repeat_per_order(
            modal_coefficients[frequency_index]
        )

        if repeated_modal.size != n_array_sh:
            raise RuntimeError("Unexpected modal-coefficient length")

        # Equivalent to:
        # H_array = Y_array * diag(repeated_modal)
        h_array = y_array * repeated_modal[np.newaxis, :]

        h_array_truncated = h_array[:, :n_output_sh]

        # MATLAB expression:
        #
        # H_filt = H_trunc' * inv(H*H' + beta^2 I)
        #
        # Use solve instead of an explicit matrix inverse.
        system_matrix = (
            h_array @ h_array.conj().T
            + regularization
        )

        # We want:
        # h_array_truncated^H @ inv(system_matrix)
        #
        # Because system_matrix is Hermitian:
        # solve(system_matrix, h_array_truncated) gives
        # inv(system_matrix) @ h_array_truncated.
        solution = np.linalg.solve(
            system_matrix,
            h_array_truncated,
        )

        h_frequency[:, :, frequency_index] = solution.conj().T

    # Build the full conjugate-symmetric FFT spectrum.
    full_spectrum = np.concatenate(
        [
            h_frequency,
            np.conj(h_frequency[:, :, -2:0:-1]),
        ],
        axis=2,
    )

    # Nyquist must be real for a real-valued impulse response.
    full_spectrum[:, :, filter_length // 2] = np.abs(
        full_spectrum[:, :, filter_length // 2]
    )

    h_time = np.real(np.fft.ifft(full_spectrum, axis=2))

    # Match MATLAB:
    # h_filt = fftshift(h_filt, 3)
    h_time = np.fft.fftshift(h_time, axes=2)

    return h_frequency, h_time


def audio_to_float64(audio: np.ndarray) -> np.ndarray:
    """Convert integer or floating-point WAV data to float64."""
    audio = np.asarray(audio)

    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float64, copy=False)

    if audio.dtype == np.uint8:
        # Unsigned 8-bit PCM is centered at 128.
        return (audio.astype(np.float64) - 128.0) / 128.0

    if np.issubdtype(audio.dtype, np.signedinteger):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        return audio.astype(np.float64) / scale

    raise TypeError(f"Unsupported WAV data type: {audio.dtype}")


def encode_em32_to_sh(
    microphone_signals: np.ndarray,
    filters_time: np.ndarray,
    preserve_length: bool = True,
) -> np.ndarray:
    """
    Encode 32 microphone signals into SH-domain signals.

    Parameters
    ----------
    microphone_signals:
        Array shaped (n_samples, n_mics).

    filters_time:
        FIR filters shaped (n_sh, n_mics, filter_length).

    preserve_length:
        When True, retain only the first n_samples, matching the MATLAB
        script. When False, retain the complete linear convolution.

    Returns
    -------
    sh_signals:
        Array shaped (n_output_samples, n_sh).
    """
    microphone_signals = np.asarray(
        microphone_signals,
        dtype=np.float64,
    )
    filters_time = np.asarray(filters_time, dtype=np.float64)

    if microphone_signals.ndim != 2:
        raise ValueError(
            "microphone_signals must have shape (samples, microphones)"
        )

    if filters_time.ndim != 3:
        raise ValueError(
            "filters_time must have shape (SH, microphones, filter_length)"
        )

    n_samples, n_input_mics = microphone_signals.shape
    n_sh, n_filter_mics, filter_length = filters_time.shape

    if n_input_mics != n_filter_mics:
        raise ValueError(
            f"Input WAV has {n_input_mics} channels, but the filter bank "
            f"expects {n_filter_mics}"
        )

    if preserve_length:
        output_length = n_samples
    else:
        output_length = n_samples + filter_length - 1

    sh_signals = np.zeros(
        (output_length, n_sh),
        dtype=np.float64,
    )

    for sh_index in range(n_sh):
        print(f"Encoding SH channel {sh_index + 1}/{n_sh}")

        accumulated = np.zeros(
            n_samples + filter_length - 1,
            dtype=np.float64,
        )

        for mic_index in range(n_input_mics):
            microphone_channel = microphone_signals[:, mic_index]
            impulse_response = filters_time[
                sh_index,
                mic_index,
                :,
            ]

            accumulated += fftconvolve(
                microphone_channel,
                impulse_response,
                mode="full",
            )

        sh_signals[:, sh_index] = accumulated[:output_length]

    return sh_signals


def normalize_audio(
    audio: np.ndarray,
    peak_limit: float = 0.99,
) -> np.ndarray:
    """Apply the same conditional peak normalization as the MATLAB script."""
    peak = float(np.max(np.abs(audio)))

    if peak > peak_limit:
        print(
            f"Normalizing output: original peak={peak:.6f}, "
            f"target peak={peak_limit:.2f}"
        )
        audio = audio * (peak_limit / peak)

    return audio


def convert_file(
    input_wav: Path,
    output_wav: Path,
    filters_time: np.ndarray,
    expected_sample_rate: int = 48000,
) -> None:
    """Load one EM32 WAV, encode it with a shared filter bank, and save it."""
    audio, sample_rate = sf.read(
        input_wav,
        always_2d=True,
    )
    audio = audio_to_float64(audio)

    print(f"Input:       {input_wav}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples:     {audio.shape[0]}")
    print(f"Channels:    {audio.shape[1]}")

    if audio.shape[1] != 32:
        raise ValueError(
            f"Expected a 32-channel EM32 WAV, received "
            f"{audio.shape[1]} channels"
        )

    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"Expected {expected_sample_rate} Hz input, received "
            f"{sample_rate} Hz. Resample the file or generate the filters "
            f"using its actual sample rate."
        )

    sh_audio = encode_em32_to_sh(
        microphone_signals=audio,
        filters_time=filters_time,
        preserve_length=True,
    )

    sh_audio = normalize_audio(sh_audio, peak_limit=0.99)

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    # FLOAT subtype preserves the generated floating-point SH channels.
    sf.write(
        output_wav,
        sh_audio.astype(np.float32),
        sample_rate,
        subtype="FLOAT",
    )

    print(f"Output shape: {sh_audio.shape}")
    print(f"Saved:        {output_wav}")



_WORKER_FILTERS_TIME: np.ndarray | None = None
_WORKER_SAMPLE_RATE: int = 48000


def detect_worker_count(n_jobs: int) -> int:
    """Choose a sensible process count from CPUs available to this process."""
    if n_jobs < 1:
        return 1

    try:
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpus = os.cpu_count() or 1

    return max(1, min(n_jobs, available_cpus))


def initialize_worker(filters_time: np.ndarray, sample_rate: int) -> None:
    """Install read-only shared job configuration in each worker process."""
    global _WORKER_FILTERS_TIME, _WORKER_SAMPLE_RATE
    _WORKER_FILTERS_TIME = filters_time
    _WORKER_SAMPLE_RATE = sample_rate


def convert_file_worker(task: tuple[Path, Path]) -> tuple[Path, bool, str]:
    """Process one file and return a compact result for the parent process."""
    input_wav, output_wav = task
    if _WORKER_FILTERS_TIME is None:
        return input_wav, False, "worker filter bank was not initialized"

    try:
        convert_file(
            input_wav=input_wav,
            output_wav=output_wav,
            filters_time=_WORKER_FILTERS_TIME,
            expected_sample_rate=_WORKER_SAMPLE_RATE,
        )
        return input_wav, True, ""
    except Exception as exc:
        return input_wav, False, f"{type(exc).__name__}: {exc}"

def parse_arguments() -> argparse.Namespace:
    default_input_dir = Path(
        "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/raw_rirs"
    )
    default_output_dir = default_input_dir.parent / "sh_rirs"

    parser = argparse.ArgumentParser(
        description=(
            "Convert all 32-channel Eigenmike WAV files in a folder to the "
            "spherical-harmonic domain."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help=f"Folder containing input WAV files; default: {default_input_dir}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Folder for converted WAV files; default: {default_output_dir}",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix added before .wav",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also process WAV files in subfolders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files that already exist.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=4,
        help="Output SH order; default: 4",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.042,
        help="Array radius in metres; default: 0.042",
    )
    parser.add_argument(
        "--filter-length",
        type=int,
        default=1024,
        help="SHT FIR/FFT length; default: 1024",
    )
    parser.add_argument(
        "--max-gain-db",
        type=float,
        default=15.0,
        help="Maximum regularized filter gain; default: 15 dB",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Expected input sample rate in Hz; default: 48000",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of parallel file-conversion processes. "
            "0 detects available CPUs automatically; default: 0"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_dir}")

    pattern = "**/*.wav" if args.recursive else "*.wav"
    input_files = sorted(
        path for path in input_dir.glob(pattern) if path.is_file()
    )

    if not input_files:
        raise FileNotFoundError(
            f"No WAV files found in {input_dir}"
            + (" or its subfolders" if args.recursive else "")
        )

    print(f"Found {len(input_files)} WAV file(s).")

    # The array geometry and encoder settings are identical for every file,
    # so generate this expensive filter bank once and reuse it.
    print("Generating shared spherical-harmonic filter bank...")
    mic_dirs_rad = np.deg2rad(MIC_DIRS_DEG)
    _, filters_time = generate_sht_filters_regls(
        radius=args.radius,
        mic_dirs_az_el_rad=mic_dirs_rad,
        sht_order=args.order,
        filter_length=args.filter_length,
        sample_rate=args.sample_rate,
        max_gain_db=args.max_gain_db,
    )
    print(f"Filter-bank shape: {filters_time.shape}")
    print("Filter bank ready. Starting file conversion.")

    converted = 0
    skipped = 0
    failed = 0
    tasks: list[tuple[Path, Path]] = []

    for input_wav in input_files:
        relative_parent = input_wav.parent.relative_to(input_dir)
        output_name = f"{input_wav.stem}{args.output_suffix}.wav"
        output_wav = output_dir / relative_parent / output_name

        if output_wav.exists() and not args.overwrite:
            print(f"Skipping existing output: {output_wav}")
            skipped += 1
            continue

        tasks.append((input_wav, output_wav))

    if args.workers < 0:
        raise ValueError("--workers must be 0 or a positive integer")

    worker_count = (
        detect_worker_count(len(tasks)) if args.workers == 0
        else min(max(1, args.workers), max(1, len(tasks)))
    )
    print(f"Using {worker_count} worker process(es) for {len(tasks)} file(s).")

    if tasks and worker_count == 1:
        initialize_worker(filters_time, args.sample_rate)
        for index, task in enumerate(tasks, start=1):
            input_wav, ok, error = convert_file_worker(task)
            print(f"Completed {index}/{len(tasks)}: {input_wav}")
            if ok:
                converted += 1
            else:
                failed += 1
                print(f"ERROR converting {input_wav}: {error}")

    elif tasks:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=initialize_worker,
            initargs=(filters_time, args.sample_rate),
        ) as executor:
            future_to_input = {
                executor.submit(convert_file_worker, task): task[0]
                for task in tasks
            }

            for completed_index, future in enumerate(
                as_completed(future_to_input), start=1
            ):
                input_wav = future_to_input[future]
                try:
                    _, ok, error = future.result()
                except Exception as exc:
                    ok = False
                    error = f"worker crashed: {type(exc).__name__}: {exc}"

                print(
                    f"Completed {completed_index}/{len(tasks)}: {input_wav}"
                )
                if ok:
                    converted += 1
                else:
                    failed += 1
                    print(f"ERROR converting {input_wav}: {error}")

    print("\n" + "=" * 72)
    print(
        f"Finished: {converted} converted, {skipped} skipped, "
        f"{failed} failed."
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
