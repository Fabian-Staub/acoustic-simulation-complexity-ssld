import itertools
import os
import random
import resource
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler


# ============================================================
# Reproducibility / CPU resources
# ============================================================

def available_cpu_count() -> int:
    """Return CPUs actually available to this Slurm task/process."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass

    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass

    return max(1, os.cpu_count() or 1)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


# ============================================================
# Dataset index loading
# ============================================================

def load_dataset_index(index_path: str) -> dict:
    """Load and validate dataset_index.pth."""
    index_path = os.path.abspath(index_path)

    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Dataset index not found: {index_path}")

    index_data = torch.load(
        index_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(index_data, dict):
        raise ValueError(
            f"Expected index data to be a dictionary, "
            f"got {type(index_data).__name__}"
        )

    pt_files = index_data.get("pt_files")
    entries = index_data.get("entries")

    if not isinstance(pt_files, list):
        raise ValueError("Dataset index is missing a valid 'pt_files' list")

    if not isinstance(entries, list):
        raise ValueError("Dataset index is missing a valid 'entries' list")

    normalized_pt_files = []

    for file_idx, filename in enumerate(pt_files):
        if not isinstance(filename, (str, os.PathLike)):
            raise ValueError(
                f"Invalid filename at pt_files[{file_idx}]: {filename!r}"
            )

        normalized_pt_files.append(
            os.path.basename(os.fspath(filename))
        )

    if len(normalized_pt_files) != len(set(normalized_pt_files)):
        raise ValueError("Dataset index contains duplicate shard filenames")

    index_data["pt_files"] = normalized_pt_files

    declared_num_files = index_data.get("num_files")

    if (
        declared_num_files is not None
        and int(declared_num_files) != len(normalized_pt_files)
    ):
        raise ValueError(
            "Dataset index file-count mismatch: "
            f"num_files={declared_num_files}, "
            f"len(pt_files)={len(normalized_pt_files)}"
        )

    declared_num_samples = index_data.get("num_samples")

    if (
        declared_num_samples is not None
        and int(declared_num_samples) != len(entries)
    ):
        raise ValueError(
            "Dataset index sample-count mismatch: "
            f"num_samples={declared_num_samples}, "
            f"len(entries)={len(entries)}"
        )

    return index_data



# ============================================================
# Lightweight performance diagnostics
# ============================================================

def format_bytes(num_bytes: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)

    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{value:.2f} TiB"


def cuda_sync(device: torch.device) -> None:
    """Synchronize only when CUDA timing must be made accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def print_cuda_memory(prefix: str, device: torch.device) -> None:
    if device.type != "cuda":
        return

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)

    print(
        f"{prefix} CUDA memory: "
        f"allocated={format_bytes(allocated)}, "
        f"reserved={format_bytes(reserved)}, "
        f"peak_allocated={format_bytes(peak)}",
        flush=True,
    )


def get_process_rss_bytes() -> int:
    """Return resident memory of the current process."""
    status_path = f"/proc/{os.getpid()}/status"

    try:
        with open(status_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass

    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def get_system_memory() -> tuple[int | None, int | None]:
    """Return (total_bytes, available_bytes) from /proc/meminfo."""
    values: dict[str, int] = {}

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, raw_value = line.split(":", maxsplit=1)
                fields = raw_value.strip().split()

                if fields:
                    values[key] = int(fields[0]) * 1024
    except (OSError, ValueError):
        return None, None

    return values.get("MemTotal"), values.get("MemAvailable")


def print_ram_status(prefix: str) -> None:
    total, available = get_system_memory()
    rss = get_process_rss_bytes()

    parts = [f"process_RSS={format_bytes(rss)}"]

    if available is not None:
        parts.append(f"system_available={format_bytes(available)}")

    if total is not None:
        parts.append(f"system_total={format_bytes(total)}")

    print(f"{prefix} RAM: " + ", ".join(parts), flush=True)


# ============================================================
# Dataset
# ============================================================

class HOARoomDataset(Dataset):
    """
    Dataset backed by tensor-only .pt shards and dataset_index.pth.

    Each shard must contain:
        (X, accdoa)

    Shapes:
        X:      (N, T, M, C)
        accdoa: (N, T, 3, 3)

    Each item returns:
        X:      (T, M, C)
        target: (T, 3, 3)
    """

    def __init__(
        self,
        pt_folder: str,
        allowed_files,
        index_data: dict | None = None,
        index_path: str | None = None,
        cache_size: int = 1,
        dtype: torch.dtype = torch.float32,
        preload_to_ram: bool = False,
        preload_label: str = "dataset",
    ) -> None:
        super().__init__()

        self.pt_folder = os.path.abspath(pt_folder)
        self.cache_size = max(1, int(cache_size))
        self.dtype = dtype
        self.preload_to_ram = bool(preload_to_ram)
        self.preload_label = str(preload_label)

        self.allowed_files = {
            os.path.basename(os.fspath(path))
            for path in allowed_files
        }

        if not self.allowed_files:
            raise ValueError("allowed_files must not be empty")

        if index_data is None:
            if index_path is None:
                index_path = os.path.join(
                    self.pt_folder,
                    "dataset_index.pth",
                )

            index_data = load_dataset_index(index_path)

        if not isinstance(index_data, dict):
            raise TypeError("index_data must be a dictionary")

        index_files = index_data.get("pt_files")
        index_entries = index_data.get("entries")

        if not isinstance(index_files, list):
            raise ValueError("index_data is missing the 'pt_files' list")

        if not isinstance(index_entries, list):
            raise ValueError("index_data is missing the 'entries' list")

        selected_filenames = [
            os.path.basename(os.fspath(filename))
            for filename in index_files
            if os.path.basename(os.fspath(filename)) in self.allowed_files
        ]

        selected_filename_set = set(selected_filenames)
        missing_from_index = self.allowed_files - selected_filename_set

        if missing_from_index:
            missing_text = ", ".join(sorted(missing_from_index))
            raise ValueError(
                "The following requested shard files are missing "
                f"from the dataset index: {missing_text}"
            )

        if not selected_filenames:
            raise RuntimeError("No indexed shards matched allowed_files")

        self.pt_files = [
            os.path.join(self.pt_folder, filename)
            for filename in selected_filenames
        ]

        for path in self.pt_files:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Indexed shard does not exist: {path}"
                )

        filename_to_local_idx = {
            filename: local_file_idx
            for local_file_idx, filename in enumerate(selected_filenames)
        }

        self.sample_index: list[tuple[int, int]] = []

        for entry_position, entry in enumerate(index_entries):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Index entry {entry_position} is not a dictionary"
                )

            raw_filename = entry.get("filename")

            if not isinstance(raw_filename, (str, os.PathLike)):
                raise ValueError(
                    f"Index entry {entry_position} has an invalid filename: "
                    f"{raw_filename!r}"
                )

            filename = os.path.basename(os.fspath(raw_filename))
            local_file_idx = filename_to_local_idx.get(filename)

            if local_file_idx is None:
                continue

            sample_idx = entry.get("sample_idx")

            if not isinstance(sample_idx, int) or isinstance(sample_idx, bool):
                raise ValueError(
                    f"Index entry {entry_position} has an invalid sample_idx: "
                    f"{sample_idx!r}"
                )

            if sample_idx < 0:
                raise ValueError(
                    f"Index entry {entry_position} has a negative sample_idx: "
                    f"{sample_idx}"
                )

            self.sample_index.append((local_file_idx, sample_idx))

        if not self.sample_index:
            raise RuntimeError(
                "No samples from allowed_files were found in the dataset index"
            )

        self._cache: OrderedDict[
            int,
            tuple[torch.Tensor, torch.Tensor],
        ] = OrderedDict()

        self._ram_shards: dict[
            int,
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

        if self.preload_to_ram:
            self.preload_all_shards()

    def _validate_shard(
        self,
        data,
        path: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not isinstance(data, (tuple, list))
            or len(data) != 2
            or not all(torch.is_tensor(item) for item in data)
        ):
            raise ValueError(
                f"Expected {path} to contain the tensor tuple (X, accdoa)"
            )

        X_all, accdoa_all = data

        if X_all.ndim != 4:
            raise ValueError(
                f"Expected shard X shape (N, T, M, C), "
                f"got {tuple(X_all.shape)} in {path}"
            )

        if accdoa_all.ndim != 4 or accdoa_all.shape[-2:] != (3, 3):
            raise ValueError(
                f"Expected shard accdoa shape (N, T, 3, 3), "
                f"got {tuple(accdoa_all.shape)} in {path}"
            )

        if X_all.shape[0] != accdoa_all.shape[0]:
            raise ValueError(
                f"Shard sample-count mismatch in {path}: "
                f"X has {X_all.shape[0]}, "
                f"accdoa has {accdoa_all.shape[0]}"
            )

        if X_all.shape[1] != accdoa_all.shape[1]:
            raise ValueError(
                f"Shard frame-count mismatch in {path}: "
                f"X has {X_all.shape[1]}, "
                f"accdoa has {accdoa_all.shape[1]}"
            )

        return X_all, accdoa_all

    def preload_all_shards(self) -> None:
        """Load every selected shard into this process's RAM once."""
        if self._ram_shards:
            return

        total_disk_bytes = sum(
            os.path.getsize(path)
            for path in self.pt_files
        )
        preload_start = time.perf_counter()
        loaded_disk_bytes = 0

        print(
            f"[RAM PRELOAD {self.preload_label}] "
            f"starting {len(self.pt_files)} shards, "
            f"disk_size={format_bytes(total_disk_bytes)}",
            flush=True,
        )
        print_ram_status(f"[RAM PRELOAD {self.preload_label} BEFORE]")

        for position, path in enumerate(self.pt_files, start=1):
            file_idx = position - 1
            file_size = os.path.getsize(path)
            shard_start = time.perf_counter()

            data = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            data = self._validate_shard(data, path)
            self._ram_shards[file_idx] = data

            shard_seconds = time.perf_counter() - shard_start
            loaded_disk_bytes += file_size
            elapsed = time.perf_counter() - preload_start

            print(
                f"[RAM PRELOAD {self.preload_label} "
                f"{position}/{len(self.pt_files)}] "
                f"{os.path.basename(path)} | "
                f"size={format_bytes(file_size)} | "
                f"load={shard_seconds:.3f}s | "
                f"shard_rate={format_bytes(file_size / max(shard_seconds, 1e-12))}/s | "
                f"overall_rate={format_bytes(loaded_disk_bytes / max(elapsed, 1e-12))}/s",
                flush=True,
            )

            if (
                position == 1
                or position == len(self.pt_files)
                or position % 5 == 0
            ):
                print_ram_status(
                    f"[RAM PRELOAD {self.preload_label} {position}/{len(self.pt_files)}]"
                )

        total_seconds = time.perf_counter() - preload_start

        print(
            f"[RAM PRELOAD {self.preload_label} DONE] "
            f"loaded={format_bytes(loaded_disk_bytes)} "
            f"in {total_seconds:.2f}s "
            f"({format_bytes(loaded_disk_bytes / max(total_seconds, 1e-12))}/s)",
            flush=True,
        )
        print_ram_status(f"[RAM PRELOAD {self.preload_label} AFTER]")

    def _load_file(
        self,
        file_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one tensor shard using RAM preload or a per-worker LRU cache."""
        if file_idx in self._ram_shards:
            return self._ram_shards[file_idx]

        if file_idx in self._cache:
            data = self._cache.pop(file_idx)
            self._cache[file_idx] = data
            return data

        if not 0 <= file_idx < len(self.pt_files):
            raise IndexError(f"Invalid local shard index: {file_idx}")

        path = self.pt_files[file_idx]
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else "main"
        file_size = os.path.getsize(path)

        print(
            f"[DATA worker={worker_id} pid={os.getpid()}] "
            f"loading shard {os.path.basename(path)} "
            f"({format_bytes(file_size)})",
            flush=True,
        )

        load_start = time.perf_counter()

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        load_seconds = time.perf_counter() - load_start
        throughput = file_size / max(load_seconds, 1e-12)

        print(
            f"[DATA worker={worker_id} pid={os.getpid()}] "
            f"loaded shard {os.path.basename(path)} in "
            f"{load_seconds:.3f}s "
            f"({format_bytes(throughput)}/s)",
            flush=True,
        )

        data = self._validate_shard(data, path)
        self._cache[file_idx] = data

        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return data

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        file_idx, sample_idx = self.sample_index[index]

        X_all, accdoa_all = self._load_file(file_idx)

        if sample_idx >= X_all.shape[0]:
            filename = os.path.basename(self.pt_files[file_idx])
            raise IndexError(
                f"Dataset index references sample {sample_idx} in {filename}, "
                f"but the shard contains only {X_all.shape[0]} samples"
            )

        X = X_all[sample_idx]
        target = accdoa_all[sample_idx]

        if X.dtype != self.dtype:
            X = X.to(dtype=self.dtype)

        if target.dtype != self.dtype:
            target = target.to(dtype=self.dtype)

        if X.ndim != 3:
            raise ValueError(
                f"Expected X item shape (T, M, C), got {tuple(X.shape)}"
            )

        if target.ndim != 3 or target.shape[-2:] != (3, 3):
            raise ValueError(
                f"Expected target item shape (T, 3, 3), "
                f"got {tuple(target.shape)}"
            )

        return X, target


# ============================================================
# File-local batch sampler
# ============================================================

class TwoLevelWorkerBatchSampler(Sampler):
    """Two-level shuffling optimized for large on-disk shards."""

    def __init__(
        self,
        dataset: HOARoomDataset,
        batch_size: int,
        worker_slots: int,
        drop_last: bool = True,
        shuffle_files: bool = True,
        shuffle_within_file: bool = True,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.worker_slots = max(1, int(worker_slots))
        self.drop_last = bool(drop_last)
        self.shuffle_files = bool(shuffle_files)
        self.shuffle_within_file = bool(shuffle_within_file)
        self.seed = int(seed)
        self.epoch = 0

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.indices_by_file: dict[int, list[int]] = {}

        for dataset_idx, (file_idx, _sample_idx) in enumerate(
            dataset.sample_index
        ):
            self.indices_by_file.setdefault(int(file_idx), []).append(
                dataset_idx
            )

        if not self.indices_by_file:
            raise RuntimeError("Dataset contains no indexed samples")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _make_batches(
        self,
        indices: list[int],
    ) -> list[list[int]]:
        batches = []

        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]

            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        return batches

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        file_indices = list(self.indices_by_file)

        if self.shuffle_files:
            order = torch.randperm(
                len(file_indices),
                generator=generator,
            ).tolist()

            file_indices = [file_indices[i] for i in order]

        batches_by_file: dict[int, list[list[int]]] = {}

        for file_idx in file_indices:
            indices = list(self.indices_by_file[file_idx])

            if self.shuffle_within_file:
                order = torch.randperm(
                    len(indices),
                    generator=generator,
                ).tolist()

                indices = [indices[i] for i in order]

            batches_by_file[file_idx] = self._make_batches(indices)

        pending_files = iter(file_indices)
        active: list[dict | None] = [None] * self.worker_slots

        def assign_next(slot: int) -> None:
            for file_idx in pending_files:
                batches = batches_by_file[file_idx]

                if batches:
                    active[slot] = {
                        "file_idx": file_idx,
                        "batches": batches,
                        "position": 0,
                    }
                    return

            active[slot] = None

        for slot in range(self.worker_slots):
            assign_next(slot)

        while any(state is not None for state in active):
            for slot in range(self.worker_slots):
                state = active[slot]

                if state is None:
                    continue

                position = state["position"]
                batches = state["batches"]

                yield batches[position]

                state["position"] = position + 1

                if state["position"] >= len(batches):
                    assign_next(slot)

    def __len__(self) -> int:
        total = 0

        for indices in self.indices_by_file.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (
                    len(indices) + self.batch_size - 1
                ) // self.batch_size

        return total


# ============================================================
# Model
# ============================================================

class SDLCRNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 46,
        num_speakers: int = 3,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.num_speakers = num_speakers
        self.output_size = num_speakers * 3

        self.conv1 = nn.Conv2d(
            input_channels,
            128,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(128)
        self.elu1 = nn.ELU()
        self.pool1 = nn.MaxPool2d((1, 2))
        self.drop1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.elu2 = nn.ELU()
        self.pool2 = nn.MaxPool2d((1, 8))
        self.drop2 = nn.Dropout2d(dropout_rate)

        self.conv3 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.elu3 = nn.ELU()
        self.pool3 = nn.MaxPool2d((1, 4))
        self.drop3 = nn.Dropout2d(dropout_rate)

        self.bilstm1 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.bilstm2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(256, 256)
        self.fc_elu = nn.ELU()
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected model input shape (B, T, M, C), "
                f"got {tuple(x.shape)}"
            )

        if x.shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, "
                f"got {x.shape[-1]}"
            )

        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.drop1(self.pool1(self.elu1(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.elu2(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.elu3(self.bn3(self.conv3(x)))))

        if x.shape[-1] != 2:
            raise ValueError(
                "Unexpected pooled mel dimension. "
                f"Expected 2, got {x.shape[-1]}. "
                "The current model expects 128 input mel bins."
            )

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.flatten(start_dim=2)

        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)

        x = self.fc1(x)
        x = self.fc_elu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x


# ============================================================
# Framewise PIT loss
# ============================================================

class FramewisePITLoss(nn.Module):
    def __init__(self, nb_tracks: int = 3) -> None:
        super().__init__()

        self.nb_tracks = int(nb_tracks)

        permutations = list(
            itertools.permutations(range(self.nb_tracks))
        )

        self.register_buffer(
            "permutations",
            torch.tensor(permutations, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if pred.ndim == 3:
            expected_output_size = self.nb_tracks * 3

            if pred.shape[-1] != expected_output_size:
                raise ValueError(
                    f"Expected prediction output size "
                    f"{expected_output_size}, got {pred.shape[-1]}"
                )

            pred = pred.reshape(
                pred.shape[0],
                pred.shape[1],
                self.nb_tracks,
                3,
            )

        if pred.ndim != 4:
            raise ValueError(
                f"Expected pred shape (B, T, tracks, 3), "
                f"got {tuple(pred.shape)}"
            )

        if target.ndim != 4:
            raise ValueError(
                f"Expected target shape (B, T, tracks, 3), "
                f"got {tuple(target.shape)}"
            )

        if pred.shape != target.shape:
            raise ValueError(
                f"pred shape {tuple(pred.shape)} does not match "
                f"target shape {tuple(target.shape)}"
            )

        permutation_losses = []

        for permutation in self.permutations:
            permuted_target = target[:, :, permutation, :]

            loss = (
                pred - permuted_target
            ).pow(2).mean(dim=(-1, -2))

            permutation_losses.append(loss)

        permutation_losses = torch.stack(
            permutation_losses,
            dim=0,
        )

        min_loss = permutation_losses.min(dim=0).values

        return min_loss.mean()


# ============================================================
# Train / validation / test loops
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp: bool,
    grad_clip=None,
    epoch: int | None = None,
    log_every: int = 20,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    data_wait_total = 0.0
    transfer_total = 0.0
    forward_loss_total = 0.0
    backward_step_total = 0.0
    epoch_start = time.perf_counter()
    previous_step_end = epoch_start

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    num_batches = len(loader)

    for batch_idx, (X, target) in enumerate(loader, start=1):
        batch_received = time.perf_counter()
        data_wait = batch_received - previous_step_end
        data_wait_total += data_wait

        transfer_start = time.perf_counter()
        X = X.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        cuda_sync(device)
        transfer_time = time.perf_counter() - transfer_start
        transfer_total += transfer_time

        optimizer.zero_grad(set_to_none=True)

        forward_start = time.perf_counter()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            pred = model(X)
            loss = criterion(pred, target)

        cuda_sync(device)
        forward_loss_time = time.perf_counter() - forward_start
        forward_loss_total += forward_loss_time

        backward_start = time.perf_counter()
        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        scaler.step(optimizer)
        scaler.update()
        cuda_sync(device)
        backward_step_time = time.perf_counter() - backward_start
        backward_step_total += backward_step_time

        batch_size = X.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

        step_time = (
            data_wait
            + transfer_time
            + forward_loss_time
            + backward_step_time
        )
        compute_time = forward_loss_time + backward_step_time
        data_fraction = data_wait / max(step_time, 1e-12)

        should_log = (
            batch_idx == 1
            or batch_idx == num_batches
            or batch_idx % log_every == 0
            or data_wait > max(2.0, compute_time)
        )

        if should_log:
            epoch_label = f" epoch={epoch}" if epoch is not None else ""

            print(
                f"[TRAIN{epoch_label} batch={batch_idx}/{num_batches}] "
                f"wait={data_wait:.3f}s | "
                f"H2D={transfer_time:.3f}s | "
                f"fwd+loss={forward_loss_time:.3f}s | "
                f"bwd+step={backward_step_time:.3f}s | "
                f"total={step_time:.3f}s | "
                f"data_wait={100.0 * data_fraction:.1f}% | "
                f"loss={loss.detach().item():.6f}",
                flush=True,
            )

            print_cuda_memory(
                f"[TRAIN batch={batch_idx}]",
                device,
            )

        previous_step_end = time.perf_counter()

    if total_samples == 0:
        raise RuntimeError("Training loader produced no samples")

    epoch_seconds = time.perf_counter() - epoch_start
    measured_total = (
        data_wait_total
        + transfer_total
        + forward_loss_total
        + backward_step_total
    )

    print(
        f"[TRAIN SUMMARY epoch={epoch}] "
        f"batches={num_batches} | "
        f"samples={total_samples} | "
        f"wall={epoch_seconds:.2f}s | "
        f"wait={data_wait_total:.2f}s "
        f"({100.0 * data_wait_total / max(measured_total, 1e-12):.1f}%) | "
        f"H2D={transfer_total:.2f}s "
        f"({100.0 * transfer_total / max(measured_total, 1e-12):.1f}%) | "
        f"fwd+loss={forward_loss_total:.2f}s "
        f"({100.0 * forward_loss_total / max(measured_total, 1e-12):.1f}%) | "
        f"bwd+step={backward_step_total:.2f}s "
        f"({100.0 * backward_step_total / max(measured_total, 1e-12):.1f}%) | "
        f"throughput={total_samples / max(epoch_seconds, 1e-12):.1f} samples/s",
        flush=True,
    )

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp: bool,
    phase: str = "eval",
    log_every: int = 20,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    data_wait_total = 0.0
    transfer_total = 0.0
    compute_total = 0.0
    phase_start = time.perf_counter()
    previous_step_end = phase_start

    num_batches = len(loader)

    for batch_idx, (X, target) in enumerate(loader, start=1):
        batch_received = time.perf_counter()
        data_wait = batch_received - previous_step_end
        data_wait_total += data_wait

        transfer_start = time.perf_counter()
        X = X.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        cuda_sync(device)
        transfer_time = time.perf_counter() - transfer_start
        transfer_total += transfer_time

        compute_start = time.perf_counter()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            pred = model(X)
            loss = criterion(pred, target)

        cuda_sync(device)
        compute_time = time.perf_counter() - compute_start
        compute_total += compute_time

        batch_size = X.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

        if (
            batch_idx == 1
            or batch_idx == num_batches
            or batch_idx % log_every == 0
            or data_wait > max(2.0, compute_time)
        ):
            print(
                f"[{phase.upper()} batch={batch_idx}/{num_batches}] "
                f"wait={data_wait:.3f}s | "
                f"H2D={transfer_time:.3f}s | "
                f"forward+loss={compute_time:.3f}s | "
                f"loss={loss.detach().item():.6f}",
                flush=True,
            )

        previous_step_end = time.perf_counter()

    if total_samples == 0:
        raise RuntimeError("Evaluation loader produced no samples")

    wall_seconds = time.perf_counter() - phase_start
    measured_total = data_wait_total + transfer_total + compute_total

    print(
        f"[{phase.upper()} SUMMARY] "
        f"batches={num_batches} | "
        f"samples={total_samples} | "
        f"wall={wall_seconds:.2f}s | "
        f"wait={data_wait_total:.2f}s "
        f"({100.0 * data_wait_total / max(measured_total, 1e-12):.1f}%) | "
        f"H2D={transfer_total:.2f}s "
        f"({100.0 * transfer_total / max(measured_total, 1e-12):.1f}%) | "
        f"compute={compute_total:.2f}s "
        f"({100.0 * compute_total / max(measured_total, 1e-12):.1f}%) | "
        f"throughput={total_samples / max(wall_seconds, 1e-12):.1f} samples/s",
        flush=True,
    )

    return total_loss / total_samples


# ============================================================
# Main training function
# ============================================================

def main() -> None:
    seed = 42
    set_seed(seed)

    pt_folder = (
        "../../../data/harp/"
        "R4000_P8_S3_Order3_24kHz_shoeBox_rtProfilesMIT_ISMOrder30/"
        "hopiv_accdoa_mel128_TIMIT_babble_full/pt/normalized"
    )

    index_path = os.path.join(
        pt_folder,
        "dataset_index.pth",
    )

    checkpoint_path = (
        "best_model_TIMIT_babble_full_512-batch.pth"
    )

    index_data = load_dataset_index(index_path)
    all_pt_files = list(index_data["pt_files"])

    # Optional dataset-size constraint. Set to None to use all indexed files.
    # The selected files are taken before the 70/20/10 split.
    max_dataset_files = 40

    if max_dataset_files is not None:
        max_dataset_files = int(max_dataset_files)

        if max_dataset_files <= 0:
            raise ValueError("max_dataset_files must be positive or None")

        all_pt_files = all_pt_files[:max_dataset_files]

    num_files = len(all_pt_files)

    if num_files < 3:
        raise RuntimeError(
            "At least 3 indexed .pt files are required "
            "for a 70/20/10 split"
        )

    train_end = int(num_files * 0.70)
    val_end = int(num_files * 0.90)

    if train_end <= 0:
        raise RuntimeError("Training split contains no files")

    if val_end <= train_end:
        raise RuntimeError("Validation split contains no files")

    if val_end >= num_files:
        raise RuntimeError("Test split contains no files")

    train_files = all_pt_files[:train_end]
    val_files = all_pt_files[train_end:val_end]
    test_files = all_pt_files[val_end:]

    print(f"Index:       {index_path}")
    print(f"File limit:  {max_dataset_files if max_dataset_files is not None else 'all'}")
    print(f"Total files: {num_files}")
    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")
    print(f"Test files:  {len(test_files)}")

    shard_sizes = [
        os.path.getsize(os.path.join(pt_folder, filename))
        for filename in all_pt_files
    ]
    print(
        "Shard sizes: "
        f"min={format_bytes(min(shard_sizes))}, "
        f"mean={format_bytes(sum(shard_sizes) / len(shard_sizes))}, "
        f"max={format_bytes(max(shard_sizes))}, "
        f"total={format_bytes(sum(shard_sizes))}"
    )

    batch_size = 512
    available_cpus = available_cpu_count()

    # RAM preload configuration:
    # - Training preload is enabled by default.
    # - Validation preload is optional and disabled by default.
    # Override from Slurm with PRELOAD_TRAIN_TO_RAM=0/1 and
    # PRELOAD_VAL_TO_RAM=0/1.
    preload_train_to_ram = 1
    preload_val_to_ram = 1

    # Keep preloaded tensors in the main process. num_workers=0 avoids
    # creating worker-process copies or relying on fork semantics.
    train_num_workers = 0 if preload_train_to_ram else 1
    eval_num_workers = 0 if preload_val_to_ram else 1

    print(f"Available CPUs: {available_cpus}")
    print(f"Preload train to RAM: {preload_train_to_ram}")
    print(f"Preload validation to RAM: {preload_val_to_ram}")
    print(f"Training DataLoader workers: {train_num_workers}")
    print(f"Evaluation DataLoader workers: {eval_num_workers}")
    print_ram_status("[STARTUP]")

    max_epochs = 10000

    initial_lr = 1e-3
    weight_decay = 3e-4

    early_stopping_patience = 20
    gradient_clip = None

    input_channels = 46
    num_speakers = 3

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Process PID: {os.getpid()}")
    print(f"CPU affinity count: {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 'unknown'}")
    print(f"torch.get_num_threads(): {torch.get_num_threads()}")
    print(f"torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}")
        print(f"GPU memory: {format_bytes(props.total_memory)}")

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    print(f"Mixed precision: {use_amp}")

    train_dataset = HOARoomDataset(
        pt_folder=pt_folder,
        allowed_files=train_files,
        index_data=index_data,
        cache_size=1,
        preload_to_ram=preload_train_to_ram,
        preload_label="train",
    )

    val_dataset = HOARoomDataset(
        pt_folder=pt_folder,
        allowed_files=val_files,
        index_data=index_data,
        cache_size=1,
        preload_to_ram=preload_val_to_ram,
        preload_label="validation",
    )

    test_dataset = HOARoomDataset(
        pt_folder=pt_folder,
        allowed_files=test_files,
        index_data=index_data,
        cache_size=1,
        preload_to_ram=False,
        preload_label="test",
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    train_worker_slots = max(1, train_num_workers)

    train_batch_sampler = TwoLevelWorkerBatchSampler(
        dataset=train_dataset,
        batch_size=batch_size,
        worker_slots=train_worker_slots,
        drop_last=True,
        shuffle_files=True,
        shuffle_within_file=True,
        seed=seed,
    )

    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_sampler": train_batch_sampler,
        "num_workers": train_num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": train_num_workers > 0,
    }

    if train_num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(**train_loader_kwargs)

    val_loader_kwargs = {
        "dataset": val_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": eval_num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": eval_num_workers > 0,
        "drop_last": False,
    }

    if eval_num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = 2

    val_loader = DataLoader(**val_loader_kwargs)

    test_loader_kwargs = {
        "dataset": test_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": eval_num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": eval_num_workers > 0,
        "drop_last": False,
    }

    if eval_num_workers > 0:
        test_loader_kwargs["prefetch_factor"] = 2

    test_loader = DataLoader(**test_loader_kwargs)

    model = SDLCRNN(
        input_channels=input_channels,
        num_speakers=num_speakers,
        dropout_rate=0.2,
    ).to(device)

    criterion = FramewisePITLoss(
        nb_tracks=num_speakers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.9,
        patience=5,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_batch_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=gradient_clip,
            epoch=epoch,
            log_every=20,
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            phase=f"val epoch={epoch}",
            log_every=20,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f} | "
            f"lr={current_lr:.7f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "input_channels": input_channels,
                    "num_speakers": num_speakers,
                    "train_files": train_files,
                    "val_files": val_files,
                    "test_files": test_files,
                    "dataset_index_path": os.path.abspath(index_path),
                    "dataset_index_version": index_data.get("version"),
                    "dataset_index_format": index_data.get("format"),
                },
                checkpoint_path,
            )

            print(f"Saved new best model to {checkpoint_path}")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(
                "Early stopping: validation loss did not improve "
                f"for {early_stopping_patience} epochs."
            )
            break

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"No best-model checkpoint was created: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"Loaded best model from epoch {checkpoint['epoch']} "
        f"with validation loss "
        f"{checkpoint['best_val_loss']:.6f}"
    )

    test_loss = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        phase="test",
        log_every=20,
    )

    print(f"Final test PIT loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()
