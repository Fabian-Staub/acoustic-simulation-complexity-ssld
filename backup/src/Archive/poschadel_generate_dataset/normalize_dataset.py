#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers for loading and traversing PT files
# ---------------------------------------------------------------------------

def find_pt_files(folder: str) -> list[str]:
    folder = os.path.abspath(folder)
    normalized_root = os.path.join(folder, "normalized")
    paths = sorted(
        path
        for path in glob.glob(os.path.join(folder, "**", "*.pt"), recursive=True)
        if os.path.isfile(path)
        and os.path.commonpath([os.path.abspath(path), normalized_root]) != normalized_root
    )
    if not paths:
        raise RuntimeError(f"No .pt files found in: {folder}")
    return paths


def load_pt(path: str) -> Any:
    # mmap avoids eagerly materializing every storage page where supported.
    # Accessing/normalizing X will still fault the required pages into memory.
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def init_worker() -> None:
    # Prevent each process from starting its own large OpenMP/MKL thread pool.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def unpack_pt_tensors(data: Any, path: str = "<memory>") -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and return a ``(X, accdoa)`` PT payload."""
    if not isinstance(data, (tuple, list)) or len(data) != 2:
        raise ValueError(
            f"Expected {path} to contain exactly (X, accdoa), "
            f"got {type(data).__name__}."
        )

    x, accdoa = data
    if not torch.is_tensor(x) or not torch.is_tensor(accdoa):
        raise TypeError(
            f"Expected X and accdoa in {path} to be torch tensors, "
            f"got {type(x).__name__} and {type(accdoa).__name__}."
        )
    if x.ndim != 4:
        raise ValueError(
            f"Expected X with shape (N, T, M, C) in {path}, got {tuple(x.shape)}."
        )
    if accdoa.ndim != 4:
        raise ValueError(
            f"Expected accdoa with shape (N, T, S, 3) in {path}, "
            f"got {tuple(accdoa.shape)}."
        )
    if x.shape[0] != accdoa.shape[0] or x.shape[1] != accdoa.shape[1]:
        raise ValueError(
            f"X and accdoa batch/time dimensions differ in {path}: "
            f"{tuple(x.shape)} vs {tuple(accdoa.shape)}."
        )
    return x, accdoa


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class PartialStats:
    hopiv_sum: float = 0.0
    hopiv_sumsq: float = 0.0
    hopiv_count: int = 0
    logmel_sum: float = 0.0
    logmel_sumsq: float = 0.0
    logmel_count: int = 0
    hopiv_channels: Optional[int] = None
    logmel_channels: Optional[int] = None
    n_mels: Optional[int] = None

    def merge(self, other: "PartialStats") -> None:
        if other.hopiv_count == 0 and other.logmel_count == 0:
            return

        if self.hopiv_channels is None:
            self.hopiv_channels = other.hopiv_channels
            self.logmel_channels = other.logmel_channels
            self.n_mels = other.n_mels
        else:
            expected = (
                self.hopiv_channels,
                self.logmel_channels,
                self.n_mels,
            )
            received = (
                other.hopiv_channels,
                other.logmel_channels,
                other.n_mels,
            )
            if expected != received:
                raise ValueError(
                    f"Incompatible feature shapes. Expected {expected}, "
                    f"received {received}."
                )

        self.hopiv_sum += other.hopiv_sum
        self.hopiv_sumsq += other.hopiv_sumsq
        self.hopiv_count += other.hopiv_count
        self.logmel_sum += other.logmel_sum
        self.logmel_sumsq += other.logmel_sumsq
        self.logmel_count += other.logmel_count


def infer_channel_split(num_feature_channels: int) -> tuple[int, int]:
    """
    For C = 2*(Q-1) + Q = 3Q - 2:
      Q = (C + 2) / 3
      HO-PIV channels = 2*(Q-1)
      log-mel channels = Q
    """
    if (num_feature_channels + 2) % 3 != 0:
        raise ValueError(
            f"Cannot infer HOA channel count from C={num_feature_channels}. "
            "Expected C = 3Q - 2. Pass correctly generated features."
        )

    q = (num_feature_channels + 2) // 3
    hopiv_channels = 2 * (q - 1)
    logmel_channels = q

    if hopiv_channels + logmel_channels != num_feature_channels:
        raise ValueError("Internal channel split error.")

    return hopiv_channels, logmel_channels


def update_partial_stats(partial: PartialStats, x: Any) -> None:
    if torch.is_tensor(x):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)

    if x_np.ndim != 4:
        raise ValueError(f"Expected X with shape (N, T, M, C), got {x_np.shape}")

    x_np = np.asarray(x_np, dtype=np.float32)
    n_mels = int(x_np.shape[2])
    num_channels = int(x_np.shape[3])
    hopiv_channels, logmel_channels = infer_channel_split(num_channels)

    if partial.hopiv_channels is None:
        partial.hopiv_channels = hopiv_channels
        partial.logmel_channels = logmel_channels
        partial.n_mels = n_mels
    else:
        current = (
            partial.hopiv_channels,
            partial.logmel_channels,
            partial.n_mels,
        )
        incoming = (hopiv_channels, logmel_channels, n_mels)
        if current != incoming:
            raise ValueError(
                f"Inconsistent feature shape. Expected {current}, got {incoming}."
            )

    hopiv = x_np[..., :hopiv_channels].astype(np.float64, copy=False)
    logmel = x_np[..., hopiv_channels:].astype(np.float64, copy=False)

    partial.hopiv_sum += float(hopiv.sum(dtype=np.float64))
    partial.hopiv_sumsq += float(np.square(hopiv).sum(dtype=np.float64))
    partial.hopiv_count += int(hopiv.size)

    partial.logmel_sum += float(logmel.sum(dtype=np.float64))
    partial.logmel_sumsq += float(np.square(logmel).sum(dtype=np.float64))
    partial.logmel_count += int(logmel.size)


def stats_worker(path: str) -> PartialStats:
    data = load_pt(path)
    partial = PartialStats()

    try:
        x, _ = unpack_pt_tensors(data, path)
        update_partial_stats(partial, x)
    finally:
        del data

    return partial


def finalize_stats(total: PartialStats, eps: float) -> dict:
    if total.hopiv_count <= 0 or total.logmel_count <= 0:
        raise RuntimeError(
            "No feature values were included in the statistics. "
            "Check room selection and PT structure."
        )

    hopiv_mean = total.hopiv_sum / total.hopiv_count
    hopiv_var = total.hopiv_sumsq / total.hopiv_count - hopiv_mean ** 2

    logmel_mean = total.logmel_sum / total.logmel_count
    logmel_var = total.logmel_sumsq / total.logmel_count - logmel_mean ** 2

    # Small negative values can appear from floating-point cancellation.
    hopiv_var = max(float(hopiv_var), 0.0)
    logmel_var = max(float(logmel_var), 0.0)

    return {
        "hopiv_channels": int(total.hopiv_channels),
        "logmel_channels": int(total.logmel_channels),
        "n_mels": int(total.n_mels),
        "eps": float(eps),
        "hopiv_mean": float(hopiv_mean),
        "hopiv_std": float(math.sqrt(max(hopiv_var, eps))),
        "logmel_mean": float(logmel_mean),
        "logmel_std": float(math.sqrt(max(logmel_var, eps))),
        "hopiv_count": int(total.hopiv_count),
        "logmel_count": int(total.logmel_count),
    }


def calculate_stats(
    pt_paths: list[str],
    stats_json: str,
    workers: int,
    eps: float,
    total_input_files: int,
    requested_stats_file_percent: float,
) -> dict:
    total = PartialStats()
    jobs = pt_paths

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        results = executor.map(stats_worker, jobs)

        for partial in tqdm(
            results,
            total=len(jobs),
            desc="Calculating statistics",
        ):
            total.merge(partial)

    stats = finalize_stats(total, eps=eps)
    stats["stats_files_used"] = len(pt_paths)
    stats["total_input_files"] = int(total_input_files)
    stats["requested_stats_file_percent"] = float(requested_stats_file_percent)
    stats["actual_stats_file_percent"] = (
        100.0 * len(pt_paths) / float(total_input_files)
    )
    stats["stats_file_selection"] = "ascending_path_order"
    stats["first_stats_file"] = pt_paths[0] if pt_paths else None
    stats["last_stats_file"] = pt_paths[-1] if pt_paths else None

    stats_dir = os.path.dirname(os.path.abspath(stats_json))
    os.makedirs(stats_dir, exist_ok=True)

    temp_fd, temp_path = tempfile.mkstemp(
        prefix=".normalization_",
        suffix=".json.tmp",
        dir=stats_dir,
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as file:
            json.dump(stats, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, stats_json)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise

    print(f"Saved normalization statistics to: {stats_json}")
    return stats


# ---------------------------------------------------------------------------
# Normalization to output subfolder
# ---------------------------------------------------------------------------

def validate_stats(stats: dict) -> None:
    required = {
        "hopiv_channels",
        "logmel_channels",
        "n_mels",
        "hopiv_mean",
        "hopiv_std",
        "logmel_mean",
        "logmel_std",
    }
    missing = sorted(required.difference(stats))
    if missing:
        raise ValueError(f"Normalization JSON is missing keys: {missing}")

    if float(stats["hopiv_std"]) <= 0 or float(stats["logmel_std"]) <= 0:
        raise ValueError("Normalization standard deviations must be positive.")


def normalize_x(x: Any, stats: dict) -> Any:
    split = int(stats["hopiv_channels"])
    expected_channels = split + int(stats["logmel_channels"])

    if torch.is_tensor(x):
        if x.ndim != 4:
            raise ValueError(f"Expected X with shape (N, T, M, C), got {tuple(x.shape)}")
        if int(x.shape[2]) != int(stats["n_mels"]):
            raise ValueError(f"Expected {stats['n_mels']} mel bins, got {x.shape[2]}")
        if int(x.shape[3]) != expected_channels:
            raise ValueError(f"Expected {expected_channels} feature channels, got {x.shape[3]}")
        if not x.is_floating_point():
            x = x.to(dtype=torch.float32)
        elif x.device.type != "cpu":
            x = x.cpu()
        # Normalize the loaded tensor directly; avoid a full clone/copy.
        x[..., :split].sub_(float(stats["hopiv_mean"])).div_(float(stats["hopiv_std"]))
        x[..., split:].sub_(float(stats["logmel_mean"])).div_(float(stats["logmel_std"]))
        return x

    array = np.asarray(x)
    if array.ndim != 4:
        raise ValueError(f"Expected X with shape (N, T, M, C), got {array.shape}")
    if int(array.shape[2]) != int(stats["n_mels"]):
        raise ValueError(f"Expected {stats['n_mels']} mel bins, got {array.shape[2]}")
    if int(array.shape[3]) != expected_channels:
        raise ValueError(f"Expected {expected_channels} feature channels, got {array.shape[3]}")

    if not np.issubdtype(array.dtype, np.floating) or not array.flags.writeable:
        array = array.astype(np.float32, copy=True)
    # In-place NumPy operations avoid converting through Torch and copying back.
    array[..., :split] -= float(stats["hopiv_mean"])
    array[..., :split] /= float(stats["hopiv_std"])
    array[..., split:] -= float(stats["logmel_mean"])
    array[..., split:] /= float(stats["logmel_std"])
    return array


def atomic_torch_save(data: Any, destination: str, fsync: bool = False) -> None:
    destination = os.path.abspath(destination)
    folder = os.path.dirname(destination)
    os.makedirs(folder, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp",
        dir=folder,
    )
    os.close(fd)

    try:
        torch.save(data, temp_path)

        if fsync:
            with open(temp_path, "rb") as file:
                os.fsync(file.fileno())
        os.replace(temp_path, destination)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def normalized_destination(path: str, input_folder: str) -> str:
    input_folder = os.path.abspath(input_folder)
    path = os.path.abspath(path)
    relative_path = os.path.relpath(path, input_folder)
    return os.path.join(input_folder, "normalized", relative_path)


def normalize_file_worker(
    args: tuple[str, str, dict],
) -> tuple[str, str, int]:
    path, destination, stats = args
    data = load_pt(path)

    try:
        x, accdoa = unpack_pt_tensors(data, path)
        normalized_x = normalize_x(x, stats)
        atomic_torch_save((normalized_x, accdoa), destination)
        sample_count = int(normalized_x.shape[0])
    finally:
        del data

    return path, destination, sample_count


def apply_normalization(
    pt_paths: list[str],
    input_folder: str,
    stats_json: str,
    workers: int,
    max_in_flight: int,
    skip_existing: bool,
) -> None:
    with open(stats_json, "r", encoding="utf-8") as file:
        stats = json.load(file)

    validate_stats(stats)
    total_samples = 0
    jobs = []
    skipped = 0
    for path in pt_paths:
        destination = normalized_destination(path, input_folder)
        if skip_existing and os.path.isfile(destination) and os.path.getsize(destination) > 0:
            skipped += 1
            continue
        jobs.append((path, destination, stats))

    # Keep only a small number of 2-GB jobs resident at once. Submitting every
    # file up front can create large queues and memory pressure.
    max_in_flight = max(workers, max_in_flight)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
    ) as executor:
        job_iter = iter(jobs)
        pending = {}

        def submit_next() -> bool:
            try:
                job = next(job_iter)
            except StopIteration:
                return False
            future = executor.submit(normalize_file_worker, job)
            pending[future] = job
            return True

        for _ in range(min(max_in_flight, len(jobs))):
            submit_next()

        with tqdm(total=len(jobs), desc="Writing normalized PT files") as bar:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    path, destination, _ = pending.pop(future)
                    try:
                        _, _, sample_count = future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"Normalization failed for {path}: {exc}"
                        ) from exc
                    total_samples += sample_count
                    bar.set_postfix_str(os.path.basename(destination), refresh=False)
                    bar.update(1)
                    submit_next()

    output_folder = os.path.join(os.path.abspath(input_folder), "normalized")
    print(
        f"Normalized {total_samples} samples in {len(jobs)} PT files; "
        f"skipped {skipped} existing files. Saved outputs under: {output_folder}"
    )



# ---------------------------------------------------------------------------
# Worker detection
# ---------------------------------------------------------------------------

def detect_workers() -> tuple[int, str]:
    """
    Determine worker count from the Slurm allocation without a CLI argument.

    Priority:
      1. SLURM_CPUS_PER_TASK
      2. SLURM_CPUS_ON_NODE
      3. SLURM_JOB_CPUS_PER_NODE
      4. CPU affinity
      5. os.cpu_count(), capped at 4 outside Slurm
    """
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        raw = os.environ.get(name)
        if raw:
            try:
                value = int(raw)
            except ValueError:
                continue
            if value > 0:
                return value, name

    raw = os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    if raw:
        # Examples: "16", "16(x2)", "8,16". For a single task, use the first node entry.
        match = re.match(r"\s*(\d+)", raw)
        if match:
            value = int(match.group(1))
            if value > 0:
                return value, "SLURM_JOB_CPUS_PER_NODE"

    try:
        affinity_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_count = 0

    if affinity_count > 0:
        # CPU affinity usually reflects cgroup/Slurm restrictions correctly.
        return affinity_count, "CPU affinity"

    cpu_count = os.cpu_count() or 1
    return min(4, max(1, cpu_count - 1)), "local fallback"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate z-score statistics for PT files containing (X, accdoa) "
            "and save normalized copies under input-folder/normalized/."
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=("calculate-and-apply", "apply-only"),
        help=(
            "'calculate-and-apply': calculate statistics, save them, then "
            "write normalized copies. 'apply-only': load an existing JSON "
            "and write normalized copies."
        ),
    )
    parser.add_argument(
        "--input-folder",
        required=True,
        help="Folder containing PT files. Files are searched recursively.",
    )
    parser.add_argument(
        "--stats-json",
        required=True,
        help=(
            "Output JSON in calculate-and-apply mode, or existing input JSON "
            "in apply-only mode."
        ),
    )
    parser.add_argument(
        "--stats-file-percent",
        type=float,
        default=100.0,
        help=(
            "Percentage of PT files used to calculate normalization statistics. "
            "Files are selected from the beginning of the ascending sorted path "
            "list. All PT files are still normalized. Default: 100."
        ),
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help=(
            "Maximum submitted normalization jobs. Default: automatically detected "
            "worker count. For ~2 GB files, keep this close to workers."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip non-empty normalized output files that already exist.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Minimum variance used when calculating standard deviations.",
    )

    args = parser.parse_args()

    if not 0.0 < args.stats_file_percent <= 100.0:
        parser.error("--stats-file-percent must be in the range (0, 100].")
    if args.eps <= 0:
        parser.error("--eps must be positive.")
    if args.max_in_flight < 0:
        parser.error("--max-in-flight must be non-negative.")
    if args.mode == "apply-only" and not os.path.isfile(args.stats_json):
        parser.error(
            f"--stats-json does not exist for apply-only mode: {args.stats_json}"
        )

    return args


def main() -> None:
    args = parse_args()
    pt_paths = find_pt_files(args.input_folder)
    workers, worker_source = detect_workers()

    print(f"Found {len(pt_paths)} PT files.")
    print(f"Using {workers} worker processes from {worker_source}.")

    if args.mode == "calculate-and-apply":
        num_stats_files = max(
            1,
            math.ceil(len(pt_paths) * args.stats_file_percent / 100.0),
        )
        stats_pt_paths = pt_paths[:num_stats_files]
        actual_percent = 100.0 * num_stats_files / len(pt_paths)

        print(
            f"Using the first {num_stats_files} of {len(pt_paths)} PT files "
            f"({actual_percent:.2f}%) in ascending path order for statistics."
        )
        print(f"First statistics file: {stats_pt_paths[0]}")
        print(f"Last statistics file:  {stats_pt_paths[-1]}")

        calculate_stats(
            pt_paths=stats_pt_paths,
            stats_json=args.stats_json,
            workers=workers,
            eps=args.eps,
            total_input_files=len(pt_paths),
            requested_stats_file_percent=args.stats_file_percent,
        )

    max_in_flight = args.max_in_flight or workers
    apply_normalization(
        pt_paths=pt_paths,
        input_folder=args.input_folder,
        stats_json=args.stats_json,
        workers=workers,
        max_in_flight=max_in_flight,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    # Required for safe multiprocessing on Windows and harmless elsewhere.
    main()
