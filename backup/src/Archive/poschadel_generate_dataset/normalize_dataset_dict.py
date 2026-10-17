#!/usr/bin/env python3
import argparse
import gc
import glob
import json
import math
import os
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers for loading and traversing PT files
# ---------------------------------------------------------------------------

def find_pt_files(folder: str) -> list[str]:
    paths = sorted(
        path
        for path in glob.glob(os.path.join(folder, "**", "*.pt"), recursive=True)
        if os.path.isfile(path)
    )
    if not paths:
        raise RuntimeError(f"No .pt files found in: {folder}")
    return paths


def load_pt(path: str) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def init_worker() -> None:
    # Prevent each process from starting its own large OpenMP/MKL thread pool.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def iter_feature_samples(obj: Any) -> Iterable[dict]:
    """
    Recursively yield dictionaries containing an 'X' feature tensor/array.

    Supported examples:
      {"X": ...}
      {"samples": [{"X": ...}, ...]}
      {"samples": [{"room": 1, "samples": [{"X": ...}, ...]}, ...]}
      arbitrary nested dict/list/tuple combinations
    """
    if isinstance(obj, dict):
        if "X" in obj:
            yield obj
            return

        for value in obj.values():
            if isinstance(value, (dict, list, tuple)):
                yield from iter_feature_samples(value)

    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from iter_feature_samples(value)


def inherited_room_id(sample: dict, default_room: Any = None) -> Any:
    for key in ("room", "room_id", "room_idx", "r"):
        if key in sample:
            return sample[key]
    return default_room


def iter_feature_samples_with_room(
    obj: Any,
    inherited_room: Any = None,
) -> Iterable[tuple[dict, Any]]:
    """
    Recursively yield (sample, room_id), propagating room metadata from parent
    dictionaries to nested feature samples.
    """
    if isinstance(obj, dict):
        current_room = inherited_room
        for key in ("room", "room_id", "room_idx", "r"):
            if key in obj:
                current_room = obj[key]
                break

        if "X" in obj:
            yield obj, current_room
            return

        for value in obj.values():
            if isinstance(value, (dict, list, tuple)):
                yield from iter_feature_samples_with_room(value, current_room)

    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from iter_feature_samples_with_room(value, inherited_room)


def normalize_room_id(room_id: Any) -> Optional[str]:
    if room_id is None:
        return None
    if torch.is_tensor(room_id):
        if room_id.numel() != 1:
            return str(room_id.tolist())
        room_id = room_id.item()
    if isinstance(room_id, np.generic):
        room_id = room_id.item()
    return str(room_id)


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

    if x_np.ndim != 3:
        raise ValueError(f"Expected X with shape (T, M, C), got {x_np.shape}")

    x_np = np.asarray(x_np, dtype=np.float32)
    n_mels = int(x_np.shape[1])
    num_channels = int(x_np.shape[2])
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

    hopiv = x_np[:, :, :hopiv_channels].astype(np.float64, copy=False)
    logmel = x_np[:, :, hopiv_channels:].astype(np.float64, copy=False)

    partial.hopiv_sum += float(hopiv.sum(dtype=np.float64))
    partial.hopiv_sumsq += float(np.square(hopiv).sum(dtype=np.float64))
    partial.hopiv_count += int(hopiv.size)

    partial.logmel_sum += float(logmel.sum(dtype=np.float64))
    partial.logmel_sumsq += float(np.square(logmel).sum(dtype=np.float64))
    partial.logmel_count += int(logmel.size)


def discover_rooms_in_file(path: str) -> set[str]:
    data = load_pt(path)
    rooms: set[str] = set()

    try:
        for _, room_id in iter_feature_samples_with_room(data):
            normalized = normalize_room_id(room_id)
            if normalized is not None:
                rooms.add(normalized)
    finally:
        del data

    return rooms


def choose_stats_rooms(
    pt_paths: list[str],
    num_stats_rooms: Optional[int],
    room_selection: str,
    seed: int,
    workers: int,
) -> Optional[set[str]]:
    if num_stats_rooms is None:
        return None

    if num_stats_rooms <= 0:
        raise ValueError("--num-stats-rooms must be positive.")

    all_rooms: set[str] = set()

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        results = executor.map(discover_rooms_in_file, pt_paths)
        for rooms in tqdm(
            results,
            total=len(pt_paths),
            desc="Discovering rooms",
        ):
            all_rooms.update(rooms)

    if not all_rooms:
        raise RuntimeError(
            "No room metadata was found. Expected one of: "
            "'room', 'room_id', 'room_idx', or 'r'."
        )

    ordered = sorted(all_rooms, key=int)
    if num_stats_rooms > len(ordered):
        raise ValueError(
            f"Requested {num_stats_rooms} rooms, but only "
            f"{len(ordered)} unique rooms were found."
        )

    if room_selection == "random":
        rng = random.Random(seed)
        selected = set(rng.sample(ordered, num_stats_rooms))
    else:
        selected = set(ordered[:num_stats_rooms])

    print(
        f"Using {len(selected)} of {len(ordered)} rooms "
        f"for normalization statistics."
    )
    return selected


def stats_worker(args: tuple[str, Optional[set[str]]]) -> PartialStats:
    path, selected_rooms = args
    data = load_pt(path)
    partial = PartialStats()

    try:
        found_feature = False

        for sample, room_id in iter_feature_samples_with_room(data):
            found_feature = True
            normalized_room = normalize_room_id(room_id)

            if selected_rooms is not None and normalized_room not in selected_rooms:
                continue

            update_partial_stats(partial, sample["X"])

        if not found_feature:
            raise ValueError(f"No feature samples containing 'X' found in {path}")

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
    num_stats_rooms: Optional[int],
    room_selection: str,
    seed: int,
    workers: int,
    eps: float,
) -> dict:
    selected_rooms = choose_stats_rooms(
        pt_paths=pt_paths,
        num_stats_rooms=num_stats_rooms,
        room_selection=room_selection,
        seed=seed,
        workers=workers,
    )

    total = PartialStats()
    jobs = [(path, selected_rooms) for path in pt_paths]

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        results = executor.map(stats_worker, jobs)

        for partial in tqdm(
            results,
            total=len(jobs),
            desc="Calculating statistics",
        ):
            total.merge(partial)

    stats = finalize_stats(total, eps=eps)
    stats["num_stats_rooms"] = (
        None if selected_rooms is None else len(selected_rooms)
    )
    stats["room_selection"] = room_selection
    stats["selected_rooms"] = (
        None if selected_rooms is None else sorted(selected_rooms, key=int)
    )

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
# In-place normalization
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
        if x.ndim != 3:
            raise ValueError(f"Expected X with shape (T, M, C), got {tuple(x.shape)}")
        if int(x.shape[1]) != int(stats["n_mels"]):
            raise ValueError(f"Expected {stats['n_mels']} mel bins, got {x.shape[1]}")
        if int(x.shape[2]) != expected_channels:
            raise ValueError(f"Expected {expected_channels} feature channels, got {x.shape[2]}")
        if not x.is_floating_point():
            x = x.to(dtype=torch.float32)
        elif x.device.type != "cpu":
            x = x.cpu()
        # Normalize the loaded tensor directly; avoid a full clone/copy.
        x[..., :split].sub_(float(stats["hopiv_mean"])).div_(float(stats["hopiv_std"]))
        x[..., split:].sub_(float(stats["logmel_mean"])).div_(float(stats["logmel_std"]))
        return x

    array = np.asarray(x)
    if array.ndim != 3:
        raise ValueError(f"Expected X with shape (T, M, C), got {array.shape}")
    if int(array.shape[1]) != int(stats["n_mels"]):
        raise ValueError(f"Expected {stats['n_mels']} mel bins, got {array.shape[1]}")
    if int(array.shape[2]) != expected_channels:
        raise ValueError(f"Expected {expected_channels} feature channels, got {array.shape[2]}")

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


def normalize_file_worker(args: tuple[str, dict, str]) -> tuple[str, int]:
    path, stats, stats_json = args
    data = load_pt(path)
    normalized_samples = 0

    try:
        for sample in iter_feature_samples(data):
            sample["X"] = normalize_x(sample["X"], stats)
            sample["feature"] = "mel-HO-PIV+logmel_zscore"
            sample["normalization_stats"] = os.path.basename(stats_json)
            normalized_samples += 1

        if normalized_samples == 0:
            raise ValueError(f"No feature samples containing 'X' found in {path}")

        # Atomic replacement: the original file remains intact if saving fails.
        atomic_torch_save(data, path)

    finally:
        del data

    return path, normalized_samples


def apply_normalization(
    pt_paths: list[str],
    stats_json: str,
    workers: int,
) -> None:
    with open(stats_json, "r", encoding="utf-8") as file:
        stats = json.load(file)

    validate_stats(stats)
    total_samples = 0

    # Submit individually and consume whichever file finishes first. This keeps
    # progress moving and identifies the exact file that failed or stalled.
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        max_tasks_per_child=25,
    ) as executor:
        future_to_path = {
            executor.submit(normalize_file_worker, (path, stats, stats_json)): path
            for path in pt_paths
        }
        with tqdm(total=len(pt_paths), desc="Applying normalization in place") as bar:
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    _, sample_count = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Normalization failed for {path}: {exc}") from exc
                total_samples += sample_count
                bar.set_postfix_str(os.path.basename(path), refresh=False)
                bar.update(1)

    print(
        f"Normalized {total_samples} samples in {len(pt_paths)} PT files. "
        "The original PT files were replaced atomically."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate z-score statistics and normalize PT feature files "
            "in place, or apply an existing normalization file."
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=("calculate-and-apply", "apply-only"),
        help=(
            "'calculate-and-apply': calculate statistics, save them, then "
            "normalize every PT file in place. "
            "'apply-only': load an existing JSON and normalize in place."
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
        "--num-stats-rooms",
        type=int,
        default=None,
        help=(
            "Number of unique rooms used to calculate statistics. "
            "Default: all rooms. Only used in calculate-and-apply mode."
        ),
    )
    parser.add_argument(
        "--room-selection",
        choices=("first", "random"),
        default="first",
        help="How to choose rooms when --num-stats-rooms is set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for --room-selection random.",
    )
    default_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", min(4, max(1, (os.cpu_count() or 1) - 1))))
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=(
            "Number of worker processes. Defaults to SLURM_CPUS_PER_TASK "
            "when running under Slurm, otherwise to CPU count minus one."
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Minimum variance used when calculating standard deviations.",
    )

    args = parser.parse_args()

    if args.workers <= 0:
        parser.error("--workers must be positive.")
    if args.eps <= 0:
        parser.error("--eps must be positive.")
    if args.mode == "apply-only" and not os.path.isfile(args.stats_json):
        parser.error(
            f"--stats-json does not exist for apply-only mode: {args.stats_json}"
        )

    return args


def main() -> None:
    args = parse_args()
    pt_paths = find_pt_files(args.input_folder)

    print(f"Found {len(pt_paths)} PT files.")
    print(f"Using {args.workers} worker processes.")

    if args.mode == "calculate-and-apply":
        calculate_stats(
            pt_paths=pt_paths,
            stats_json=args.stats_json,
            num_stats_rooms=args.num_stats_rooms,
            room_selection=args.room_selection,
            seed=args.seed,
            workers=args.workers,
            eps=args.eps,
        )

    apply_normalization(
        pt_paths=pt_paths,
        stats_json=args.stats_json,
        workers=args.workers,
    )


if __name__ == "__main__":
    # Required for safe multiprocessing on Windows and harmless elsewhere.
    main()
