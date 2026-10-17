#!/usr/bin/env python3
"""Measure broadband T60 and DRR from channel 0 of grouped RIR pickle files.

Expected pickle structure (as produced by the supplied generator)::

    {
        "room_idx": 0,
        "sample_rate": 32000,
        "rirs": {
            "0_S0": {"rir": ndarray(channels, samples), "metadata": {...}},
            ...
        },
    }

Only load pickle files that you trust: Python pickle is not a safe format for
untrusted input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def json_value(value: Any) -> str:
    """Convert a non-scalar metadata value to compact, valid JSON."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def flatten_metadata(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts; store lists/arrays as JSON strings for the CSV."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten_metadata(value, name))
        elif value is None or isinstance(value, (str, int, float, bool)):
            result[name] = value
        elif isinstance(value, np.generic):
            result[name] = value.item()
        else:
            result[name] = json_value(value)
    return result


def first_direct_peak(
    rir: np.ndarray, fs: float, threshold_db: float = -20.0
) -> int:
    """Find the first plausible direct arrival, then its nearby local maximum.

    The onset is the first sample above ``threshold_db`` relative to the global
    absolute peak. The reported arrival is the strongest sample in the next
    2.5 ms. This is less likely than a plain argmax to select a strong reflection.
    """
    amplitude = np.abs(rir)
    peak = float(np.max(amplitude))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("omni channel is empty or non-finite")
    threshold = peak * 10.0 ** (threshold_db / 20.0)
    candidates = np.flatnonzero(amplitude >= threshold)
    if candidates.size == 0:
        raise ValueError("could not detect direct arrival")
    onset = int(candidates[0])
    search_end = min(rir.size, onset + max(1, round(0.0025 * fs)) + 1)
    return onset + int(np.argmax(amplitude[onset:search_end]))


def measure_drr(
    rir: np.ndarray,
    fs: float,
    direct_index: int,
    window_ms: float = 2.5,
) -> tuple[float, float, float]:
    """Return DRR [dB], direct energy, and reverberant energy.

    Direct energy uses a symmetric +/- ``window_ms`` interval around the
    detected arrival. Reverberant energy starts immediately after that window;
    samples before the direct window are deliberately excluded.
    """
    half_window = max(0, round(window_ms * 1e-3 * fs))
    start = max(0, direct_index - half_window)
    stop = min(rir.size, direct_index + half_window + 1)
    direct_energy = float(np.sum(np.square(rir[start:stop], dtype=np.float64)))
    reverb_energy = float(np.sum(np.square(rir[stop:], dtype=np.float64)))
    if direct_energy <= 0.0:
        raise ValueError("direct-window energy is zero")
    if reverb_energy == 0.0:
        return math.inf, direct_energy, reverb_energy
    return 10.0 * math.log10(direct_energy / reverb_energy), direct_energy, reverb_energy


def _decay_fit(
    time_s: np.ndarray,
    edc_db: np.ndarray,
    upper_db: float,
    lower_db: float,
    min_samples: int = 3,
) -> tuple[float, float, int]:
    indices = np.flatnonzero((edc_db <= upper_db) & (edc_db >= lower_db))
    if indices.size < min_samples:
        raise ValueError(
            f"fewer than {min_samples} EDC samples between "
            f"{upper_db:g} and {lower_db:g} dB"
        )
    # Use only the first contiguous crossing interval.
    indices = np.arange(indices[0], indices[-1] + 1)
    x = time_s[indices]
    y = edc_db[indices]
    if np.ptp(x) <= 0.0 or np.ptp(y) <= 0.0:
        raise ValueError("EDC fit interval has no usable variation")
    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope) or slope >= 0.0:
        raise ValueError("energy decay slope is not negative")
    prediction = slope * x + intercept
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else math.nan
    return -60.0 / float(slope), r_squared, int(indices.size)


def measure_t60(
    rir: np.ndarray, fs: float, direct_index: int, min_fit_samples: int = 3
) -> tuple[float, str, float, int]:
    """Measure broadband T60 with Schroeder integration and linear regression.

    T30 (-5..-35 dB) is preferred. If the RIR lacks that decay range, T20
    (-5..-25 dB), then T10 (-5..-15 dB) is used. In every case the fitted slope
    is extrapolated to 60 dB.
    """
    signal = np.asarray(rir[direct_index:], dtype=np.float64)
    if signal.size < 10:
        raise ValueError("RIR is too short after the direct arrival")
    energy = np.square(signal)
    edc = np.cumsum(energy[::-1], dtype=np.float64)[::-1]
    if not np.isfinite(edc[0]) or edc[0] <= 0.0:
        raise ValueError("omni-channel energy is zero or non-finite")
    positive = edc > 0.0
    edc_db = np.full(edc.shape, -np.inf, dtype=np.float64)
    edc_db[positive] = 10.0 * np.log10(edc[positive] / edc[0])
    time_s = np.arange(signal.size, dtype=np.float64) / fs

    errors: list[str] = []
    for method, lower_db in (("T30", -35.0), ("T20", -25.0), ("T10", -15.0)):
        try:
            t60, r_squared, count = _decay_fit(
                time_s, edc_db, -5.0, lower_db, min_fit_samples
            )
            return t60, method, r_squared, count
        except ValueError as exc:
            errors.append(f"{method}: {exc}")
    raise ValueError("; ".join(errors))


def omni_channel(rir: Any) -> np.ndarray:
    array = np.asarray(rir)
    if array.ndim != 2:
        raise ValueError(f"expected RIR shape (channels, samples), got {array.shape}")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"empty RIR array with shape {array.shape}")
    channel = np.asarray(array[0], dtype=np.float64)
    if not np.all(np.isfinite(channel)):
        raise ValueError("omni channel contains NaN or infinity")
    return channel


def prepare_group(
    group: Any, fallback_fs: float | None
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Validate a grouped pickle and resolve its metadata and sample rate."""
    if not isinstance(group, dict) or not isinstance(group.get("rirs"), dict):
        raise ValueError("top-level object must be a dict containing a 'rirs' dict")

    group_metadata = {
        key: value for key, value in group.items() if key not in {"rirs", "sample_rate"}
    }
    fs_value = group.get("sample_rate", fallback_fs)
    if fs_value is None:
        raise ValueError("sample rate missing (use --sample-rate to provide it)")
    fs = float(fs_value)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"invalid sample rate: {fs_value!r}")
    return group, group_metadata, fs


def measure_entry(
    path: Path,
    position_id: Any,
    item: Any,
    group_metadata: dict[str, Any],
    fs: float,
    window_ms: float,
    threshold_db: float,
    min_fit_samples: int,
) -> dict[str, Any]:
    """Measure one RIR and return the same row used by normal CSV mode."""
    base: dict[str, Any] = {
        "source_file": path.name,
        "position_id": str(position_id),
        "sample_rate": fs,
        **flatten_metadata(group_metadata),
    }
    # Use explicit text for unavailable/error fields so pandas does not turn
    # harmless empty CSV cells into NaN. Numeric NaNs are reserved for metrics
    # that genuinely could not be computed.
    base.update(
        {
            "t60_s": math.nan,
            "t60_method": "unavailable",
            "t60_fit_r2": math.nan,
            "t60_fit_samples": 0,
            "t60_status": "not_run",
            "t60_error": "none",
            "drr_db": math.nan,
            "drr_direct_energy": math.nan,
            "drr_reverberant_energy": math.nan,
            "drr_status": "not_run",
            "drr_error": "none",
            "direct_arrival_sample": math.nan,
            "direct_arrival_s": math.nan,
            "measurement_status": "error",
            "measurement_error": "none",
        }
    )

    try:
        if not isinstance(item, dict) or "rir" not in item:
            raise ValueError("RIR entry must be a dict containing 'rir'")
        metadata = item.get("metadata", {})
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a dict")
            # Entry metadata wins if a name collides with group metadata.
            base.update(flatten_metadata(metadata))
        omni = omni_channel(item["rir"])
        direct_index = first_direct_peak(omni, fs, threshold_db)
        base.update(
            {
                "direct_arrival_sample": direct_index,
                "direct_arrival_s": direct_index / fs,
            }
        )

    except Exception as exc:  # Entry/onset failure prevents both measurements.
        base.update(
            {
                "measurement_status": "error",
                "measurement_error": str(exc),
                "t60_status": "error",
                "t60_error": f"prerequisite failed: {exc}",
                "drr_status": "error",
                "drr_error": f"prerequisite failed: {exc}",
            }
        )
        return base

    errors: list[str] = []
    try:
        t60, method, r_squared, count = measure_t60(
            omni, fs, direct_index, min_fit_samples
        )
        base.update(
            {
                "t60_s": t60,
                "t60_method": method,
                "t60_fit_r2": r_squared,
                "t60_fit_samples": count,
                "t60_status": "ok",
            }
        )
    except Exception as exc:
        base.update({"t60_status": "error", "t60_error": str(exc)})
        errors.append(f"T60: {exc}")

    try:
        drr, direct_energy, reverb_energy = measure_drr(
            omni, fs, direct_index, window_ms
        )
        if not np.isfinite(direct_energy) or not np.isfinite(reverb_energy):
            raise ValueError("DRR energy overflowed")
        base.update(
            {
                "drr_db": drr,
                "drr_direct_energy": direct_energy,
                "drr_reverberant_energy": reverb_energy,
                "drr_status": "ok",
            }
        )
    except Exception as exc:
        base.update({"drr_status": "error", "drr_error": str(exc)})
        errors.append(f"DRR: {exc}")

    succeeded = sum(base[name] == "ok" for name in ("t60_status", "drr_status"))
    base["measurement_status"] = ("ok", "partial", "error")[2 - succeeded]
    base["measurement_error"] = "; ".join(errors) if errors else "none"
    return base


def process_pickle(
    path: Path,
    fallback_fs: float | None,
    window_ms: float,
    threshold_db: float,
    min_fit_samples: int,
) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        group, group_metadata, fs = prepare_group(pickle.load(handle), fallback_fs)

    for position_id, item in group["rirs"].items():
        yield measure_entry(
            path, position_id, item, group_metadata, fs, window_ms, threshold_db,
            min_fit_samples,
        )


def save_test_plot(
    rir: np.ndarray,
    fs: float,
    row: dict[str, Any],
    window_ms: float,
    output: Path,
) -> None:
    """Save waveform, direct-window detail, and Schroeder-decay diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    direct_index = int(row["direct_arrival_sample"])
    half_window = max(0, round(window_ms * 1e-3 * fs))
    start = max(0, direct_index - half_window)
    stop = min(rir.size, direct_index + half_window + 1)
    time_ms = np.arange(rir.size, dtype=np.float64) / fs * 1000.0
    arrival_ms = direct_index / fs * 1000.0
    window_start_ms = start / fs * 1000.0
    window_stop_ms = stop / fs * 1000.0

    figure, axes = plt.subplots(3, 1, figsize=(11.7, 10.0), layout="constrained")
    figure.suptitle(
        f"RIR analysis: {row['source_file']} / {row['position_id']}",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].plot(time_ms, rir, color="#176B87", linewidth=0.65)
    axes[0].axvspan(
        window_start_ms, window_stop_ms, color="#F4A261", alpha=0.45,
        label=f"Direct-sound window (±{window_ms:g} ms)",
    )
    axes[0].axvline(arrival_ms, color="#D62828", linestyle="--", label="Direct arrival")
    axes[0].set(title="Full impulse response", ylabel="Amplitude")
    axes[0].legend(loc="upper right")

    context_ms = max(10.0, 4.0 * window_ms)
    zoom_start_ms = max(0.0, window_start_ms - context_ms)
    zoom_stop_ms = min(time_ms[-1], window_stop_ms + context_ms)
    zoom_mask = (time_ms >= zoom_start_ms) & (time_ms <= zoom_stop_ms)
    axes[1].plot(time_ms[zoom_mask], rir[zoom_mask], color="#176B87", linewidth=1.0)
    axes[1].axvspan(
        window_start_ms, window_stop_ms, color="#F4A261", alpha=0.45,
        label="Direct energy",
    )
    if window_stop_ms < zoom_stop_ms:
        axes[1].axvspan(
            window_stop_ms, zoom_stop_ms, color="#84A98C", alpha=0.23,
            label="Reverberant energy",
        )
    axes[1].axvline(arrival_ms, color="#D62828", linestyle="--", label="Direct arrival")
    axes[1].set(
        title=f"Direct arrival and DRR window: DRR = {row['drr_db']:.2f} dB",
        ylabel="Amplitude",
    )
    axes[1].legend(loc="upper right")

    signal = np.asarray(rir[direct_index:], dtype=np.float64)
    edc = np.cumsum(np.square(signal)[::-1], dtype=np.float64)[::-1]
    edc_db = np.full(edc.shape, -np.inf, dtype=np.float64)
    positive = edc > 0.0
    edc_db[positive] = 10.0 * np.log10(edc[positive] / edc[0])
    decay_time_ms = np.arange(signal.size, dtype=np.float64) / fs * 1000.0
    axes[2].plot(decay_time_ms[positive], edc_db[positive], color="#176B87", label="Schroeder EDC")

    lower_db = {"T30": -35.0, "T20": -25.0, "T10": -15.0}[row["t60_method"]]
    fit_indices = np.flatnonzero((edc_db <= -5.0) & (edc_db >= lower_db))
    fit_indices = np.arange(fit_indices[0], fit_indices[-1] + 1)
    fit_time_ms = decay_time_ms[fit_indices]
    slope, intercept = np.polyfit(fit_time_ms, edc_db[fit_indices], 1)
    axes[2].plot(
        fit_time_ms, slope * fit_time_ms + intercept, color="#D62828",
        linewidth=2.0, label=f"{row['t60_method']} fit ({lower_db:g} to -5 dB)",
    )
    for level_db in (-5.0, lower_db):
        axes[2].axhline(level_db, color="#777777", linestyle=":", linewidth=0.9)
    axes[2].set(
        title=(
            f"Energy decay: T60 = {row['t60_s']:.3f} s "
            f"({row['t60_method']}, R² = {row['t60_fit_r2']:.4f})"
        ),
        xlabel="Time after direct arrival [ms]",
        ylabel="Relative energy [dB]",
    )
    axes[2].legend(loc="upper right")

    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].set_xlabel("Time [ms]")
    axes[1].set_xlabel("Time [ms]")
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)


def run_test_mode(paths: list[Path], args: argparse.Namespace) -> int:
    """Measure one randomly selected RIR and write only its diagnostic PDF."""
    rng = np.random.default_rng()
    shuffled_paths = list(paths)
    rng.shuffle(shuffled_paths)

    for path in shuffled_paths:
        try:
            with path.open("rb") as handle:
                group, group_metadata, fs = prepare_group(
                    pickle.load(handle), args.sample_rate
                )
            if not group["rirs"]:
                raise ValueError("pickle contains no RIR entries")
            entries = list(group["rirs"].items())
            position_id, item = entries[int(rng.integers(len(entries)))]
        except Exception as exc:
            print(f"warning: skipped {path}: {exc}", file=sys.stderr)
            continue

        row = measure_entry(
            path, position_id, item, group_metadata, fs,
            args.direct_window_ms, args.arrival_threshold_db, args.min_fit_samples,
        )
        if row["measurement_status"] != "ok":
            print(
                f"error: measurement failed for {path.name} / {position_id}: "
                f"{row['measurement_error']}",
                file=sys.stderr,
            )
            return 1

        safe_position_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(position_id)).strip("._")
        pdf_path = args.input_folder / (
            f"analysis_{path.stem}_{safe_position_id or 'position'}.pdf"
        )
        try:
            save_test_plot(
                omni_channel(item["rir"]), fs, row, args.direct_window_ms, pdf_path
            )
        except ImportError as exc:
            print(f"error: test mode requires matplotlib: {exc}", file=sys.stderr)
            return 1

        print("Test mode: analyzed one randomly selected RIR (no CSV written).")
        for name, value in row.items():
            if name != "measurement_error":
                print(f"  {name}: {value}")
        print(f"  direct_window_ms: ±{args.direct_window_ms:g}")
        print(f"  analysis_pdf: {pdf_path}")
        return 0

    print("error: no usable RIR entries could be read", file=sys.stderr)
    return 1


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    priority = [
        "source_file", "room_idx", "position_id", "Name", "sample_rate",
        "t60_s", "t60_method", "t60_fit_r2", "t60_fit_samples", "drr_db",
        "direct_arrival_sample", "direct_arrival_s", "drr_direct_energy",
        "drr_reverberant_energy", "t60_status", "t60_error", "drr_status",
        "drr_error", "measurement_status", "measurement_error",
    ]
    all_fields = {key for row in rows for key in row}
    fields = [key for key in priority if key in all_fields]
    fields.extend(sorted(all_fields.difference(fields)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_folder",
        type=Path,
        nargs="?",
        default=Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P8_S3_Order3_24kHz_non-shoeBox_disp-corners-1_SH_ISMOrder12/RIRs"),
        help="Folder containing grouped .pkl files (default: current folder)",
    )

    parser.add_argument(
        "output_csv",
        type=Path,
        nargs="?",
        default=Path("02_metadata_T60_DRR.csv"),
        help="Destination CSV file (default: rir_t60_drr.csv)",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Also search subfolders for .pkl files"
    )
    parser.add_argument(
        "--test-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Analyze one random RIR, print its metrics, and save a PDF instead of CSV",
    )
    parser.add_argument(
        "--sample-rate", type=float, default=None,
        help="Fallback sample rate in Hz if absent from a pickle",
    )
    parser.add_argument(
        "--direct-window-ms", type=float, default=1,
        help="DRR half-window around direct arrival in ms (default: 1)",
    )
    parser.add_argument(
        "--arrival-threshold-db", type=float, default=-20.0,
        help="Direct-onset threshold relative to global peak (default: -20)",
    )
    parser.add_argument(
        "--min-fit-samples", type=int, default=3,
        help="Minimum EDC samples required for a T60 regression (default: 3)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input_folder.is_dir():
        print(f"error: not a folder: {args.input_folder}", file=sys.stderr)
        return 2

    # Always save the CSV inside the input folder
    output_csv = args.input_folder / args.output_csv.name

    if args.direct_window_ms < 0.0:
        print("error: --direct-window-ms must be non-negative", file=sys.stderr)
        return 2
    if args.min_fit_samples < 3:
        print("error: --min-fit-samples must be at least 3", file=sys.stderr)
        return 2

    pattern = "**/*.pkl" if args.recursive else "*.pkl"
    paths = sorted(args.input_folder.glob(pattern))

    if not paths:
        print(f"error: no .pkl files found in {args.input_folder}", file=sys.stderr)
        return 1

    if args.test_mode:
        return run_test_mode(paths, args)

    rows: list[dict[str, Any]] = []
    file_errors = 0

    for path in paths:
        try:
            rows.extend(
                process_pickle(
                    path,
                    args.sample_rate,
                    args.direct_window_ms,
                    args.arrival_threshold_db,
                    args.min_fit_samples,
                )
            )
        except Exception as exc:
            file_errors += 1
            print(f"warning: skipped {path}: {exc}", file=sys.stderr)

    if not rows:
        print("error: no RIR entries could be read", file=sys.stderr)
        return 1

    write_csv(rows, output_csv)

    failed = sum(row["measurement_status"] != "ok" for row in rows)
    print(
        f"Wrote {len(rows)} RIR rows to {output_csv} "
        f"({failed} measurement errors, {file_errors} skipped files)."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
