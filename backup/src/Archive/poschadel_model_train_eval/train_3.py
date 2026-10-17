import os
import glob
import json
import itertools
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler, get_worker_info



# ============================================================
# Reproducibility
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




def scan_shard_sample_count(path: str) -> tuple[str, int]:
    """Load one shard in a worker process and return its sample count."""
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

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
            f"Shard sample count mismatch in {path}: "
            f"X has {X_all.shape[0]}, accdoa has {accdoa_all.shape[0]}"
        )

    return os.path.basename(path), int(X_all.shape[0])

# ============================================================
# Dataset
# ============================================================

class HOARoomDataset(Dataset):
    """
    Dataset backed by tensor-only .pt shards.

    Every shard must contain exactly:
        (X, accdoa)

    with:
        X:      (N, 50, 128, C)
        accdoa: (N, 50, 3, 3)

    No external dataset index is required. During initialization, each
    selected shard is opened once to determine its sample count.

    Each item returns:
        X:      (50, 128, C)
        target: (50, 3, 3)
    """

    def __init__(
        self,
        pt_folder: str,
        allowed_files,
        cache_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.pt_folder = os.path.abspath(pt_folder)
        self.allowed_files = set(
            os.path.basename(path)
            for path in allowed_files
        )
        self.cache_size = max(1, int(cache_size))
        self.dtype = dtype

        all_pt_files = sorted(
            path
            for path in glob.glob(os.path.join(self.pt_folder, "*.pt"))
            if os.path.basename(path) != "dataset_index.pth"
            and not path.endswith(".tmp")
        )

        self.pt_files = [
            path
            for path in all_pt_files
            if os.path.basename(path) in self.allowed_files
        ]

        if not self.pt_files:
            raise RuntimeError("No tensor .pt shards matched allowed_files")

        self.sample_index = []

        count_cache_path = os.path.join(
            self.pt_folder,
            "shard_sample_counts.json",
        )

        try:
            with open(count_cache_path, "r", encoding="utf-8") as file:
                count_cache = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            count_cache = {}

        sample_counts: dict[str, int] = {}
        files_to_scan: list[str] = []

        for path in self.pt_files:
            filename = os.path.basename(path)
            stat = os.stat(path)
            cached = count_cache.get(filename)

            if (
                isinstance(cached, dict)
                and cached.get("size") == stat.st_size
                and cached.get("mtime_ns") == stat.st_mtime_ns
                and isinstance(cached.get("samples"), int)
            ):
                sample_counts[filename] = cached["samples"]
            else:
                files_to_scan.append(path)

        if files_to_scan:
            scan_workers = min(
                available_cpu_count(),
                len(files_to_scan),
            )

            with ProcessPoolExecutor(max_workers=scan_workers) as executor:
                futures = {
                    executor.submit(scan_shard_sample_count, path): path
                    for path in files_to_scan
                }

                for future in as_completed(futures):
                    path = futures[future]
                    filename, num_samples = future.result()
                    stat = os.stat(path)

                    sample_counts[filename] = num_samples
                    count_cache[filename] = {
                        "samples": num_samples,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }

            temporary_cache_path = count_cache_path + ".tmp"
            with open(
                temporary_cache_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(count_cache, file, indent=2, sort_keys=True)

            os.replace(temporary_cache_path, count_cache_path)
        else:
            pass

        for local_file_idx, path in enumerate(self.pt_files):
            filename = os.path.basename(path)
            num_samples = sample_counts[filename]

            self.sample_index.extend(
                (local_file_idx, sample_idx)
                for sample_idx in range(num_samples)
            )

        if not self.sample_index:
            raise RuntimeError("No samples found in selected tensor shards")

        # Per-worker LRU cache. Each value is the tuple (X, accdoa).
        self._cache: OrderedDict[
            int,
            tuple[torch.Tensor, torch.Tensor],
        ] = OrderedDict()

    def _load_file(self, file_idx: int):
        if file_idx in self._cache:
            data = self._cache.pop(file_idx)
            self._cache[file_idx] = data

            return data

        path = self.pt_files[file_idx]

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if (
            not isinstance(data, (tuple, list))
            or len(data) != 2
            or not all(torch.is_tensor(item) for item in data)
        ):
            raise ValueError(
                f"Expected {path} to contain the tensor tuple "
                "(X, accdoa)"
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
                f"Shard sample count mismatch in {path}: "
                f"X has {X_all.shape[0]}, accdoa has {accdoa_all.shape[0]}"
            )

        data = (X_all, accdoa_all)

        self._cache[file_idx] = data

        while len(self._cache) > self.cache_size:
            evicted_idx, _ = self._cache.popitem(last=False)

        return data

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int):
        file_idx, sample_idx = self.sample_index[index]

        X_all, accdoa_all = self._load_file(file_idx)

        if sample_idx >= X_all.shape[0]:
            raise IndexError(
                f"Sample {sample_idx} is out of range for "
                f"{os.path.basename(self.pt_files[file_idx])}; "
                f"the shard contains only {X_all.shape[0]} samples."
            )

        X = X_all[sample_idx]
        target = accdoa_all[sample_idx]

        # Avoid unnecessary copies when tensors are already float32.
        if X.dtype != self.dtype:
            X = X.to(dtype=self.dtype)
        if target.dtype != self.dtype:
            target = target.to(dtype=self.dtype)

        if X.ndim != 3:
            raise ValueError(
                f"Expected X shape (T, M, C), got {tuple(X.shape)}"
            )

        if target.ndim != 3 or target.shape[-2:] != (3, 3):
            raise ValueError(
                f"Expected target shape (T, 3, 3), "
                f"got {tuple(target.shape)}"
            )

        return X, target


# ============================================================
# File-local batch sampler
# ============================================================

class TwoLevelWorkerBatchSampler(Sampler):
    """
    Two-level shuffling optimized for large on-disk shards.

    Per epoch:
      1. Shuffle shard order.
      2. Shuffle samples independently inside every shard.
      3. Assign one active shard to each DataLoader worker slot.
      4. Emit one batch per active worker slot in round-robin order.

    DataLoader dispatches batch-sampler outputs to workers round-robin.
    Therefore, when ``worker_slots == num_workers``, worker N repeatedly
    receives batches from the same shard until that shard is exhausted.
    This prevents multiple workers from repeatedly loading the same 800 MB
    shard and preserves good sample-level randomness inside each shard.
    """

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
            self.indices_by_file.setdefault(
                int(file_idx),
                [],
            ).append(dataset_idx)

        if not self.indices_by_file:
            raise RuntimeError("Dataset contains no indexed samples")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _make_batches(self, indices: list[int]) -> list[list[int]]:
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
                len(file_indices), generator=generator
            ).tolist()
            file_indices = [file_indices[i] for i in order]

        batches_by_file: dict[int, list[list[int]]] = {}
        for file_idx in file_indices:
            indices = list(self.indices_by_file[file_idx])
            if self.shuffle_within_file:
                order = torch.randperm(
                    len(indices), generator=generator
                ).tolist()
                indices = [indices[i] for i in order]
            batches_by_file[file_idx] = self._make_batches(indices)

        # One active shard per worker slot. Since DataLoader assigns batch
        # requests round-robin, slot i maps to worker i.
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
                    # Keep the round-robin worker alignment. An empty list is
                    # not valid, so no new worker tasks are emitted once fewer
                    # than worker_slots shards remain. Instead, refill slots
                    # immediately whenever their prior shard is exhausted.
                    continue

                pos = state["position"]
                batches = state["batches"]
                yield batches[pos]
                state["position"] = pos + 1

                if state["position"] >= len(batches):
                    assign_next(slot)

    def __len__(self) -> int:
        total = 0
        for indices in self.indices_by_file.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
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

    def forward(self, x):
        # Input:
        # (B, T, M, C)
        #
        # Conv2D expects:
        # (B, C, T, M)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Expected shape:
        # (B, 128, 50, 2)
        x = x.permute(0, 2, 1, 3).contiguous()

        # (B, 50, 128, 2) -> (B, 50, 256)
        x = x.flatten(start_dim=2)

        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)

        x = self.fc1(x)
        x = self.fc_elu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x


# ============================================================
# Framewise PIT Loss
# ============================================================

class FramewisePITLoss(nn.Module):
    def __init__(self, nb_tracks: int = 3):
        super().__init__()

        self.nb_tracks = nb_tracks

        permutations = list(
            itertools.permutations(range(nb_tracks))
        )

        self.register_buffer(
            "permutations",
            torch.tensor(
                permutations,
                dtype=torch.long,
            ),
            persistent=False,
        )

    def forward(self, pred, target):
        """
        pred:
            (B, T, 9) or (B, T, 3, 3)

        target:
            (B, T, 3, 3)
        """

        if pred.ndim == 3:
            pred = pred.reshape(
                pred.shape[0],
                pred.shape[1],
                self.nb_tracks,
                3,
            )

        if pred.shape != target.shape:
            raise ValueError(
                f"pred shape {tuple(pred.shape)} does not match "
                f"target shape {tuple(target.shape)}"
            )

        permutation_losses = []

        for perm in self.permutations:
            permuted_target = target[:, :, perm, :]

            loss = (
                pred - permuted_target
            ).pow(2).mean(dim=(-1, -2))

            permutation_losses.append(loss)

        # (num_perms, B, T)
        permutation_losses = torch.stack(
            permutation_losses,
            dim=0,
        )

        # Framewise minimum over permutations
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
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for X, target in loader:
        X = X.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            pred = model(X)
            loss = criterion(pred, target)

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        scaler.step(optimizer)
        scaler.update()

        batch_size = X.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Training loader produced no samples")

    return total_loss / total_samples




@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp: bool,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for X, target in loader:
        X = X.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            pred = model(X)
            loss = criterion(pred, target)

        batch_size = X.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Evaluation loader produced no samples")

    return total_loss / total_samples



# ============================================================
# Main training function
# ============================================================

def main():
    set_seed(42)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    pt_folder = (
        "../../../data/harp/"
        "R4000_P8_S3_Order3_24kHz_shoeBox_rtProfilesMIT_ISMOrder30/"
        "hopiv_accdoa_mel128_TIMIT_babble_full/pt/normalized"
    )

    checkpoint_path = "best_model_TIMIT_babble_full_512-batch.pth"

    # --------------------------------------------------------
    # Discover tensor shards and create a file-based split
    # --------------------------------------------------------

    all_pt_files = sorted(
        os.path.basename(path)
        for path in glob.glob(os.path.join(pt_folder, "*.pt"))
        if os.path.basename(path) != "dataset_index.pth"
        and not path.endswith(".tmp")
    )

    num_files = len(all_pt_files)

    if num_files < 3:
        raise RuntimeError(
            "At least 3 .pt files are required for a 70/20/10 split"
        )

    train_end = int(num_files * 0.70)
    val_end = int(num_files * 0.90)

    train_files = all_pt_files[:train_end]
    val_files = all_pt_files[train_end:val_end]
    test_files = all_pt_files[val_end:]

    print(f"Total files: {num_files}")
    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")
    print(f"Test files:  {len(test_files)}")

    # --------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------

    batch_size = 512
    available_cpus = available_cpu_count()
    train_num_workers = 1
    eval_num_workers = 1

    print(f"Available CPUs: {available_cpus}")
    print(f"Training DataLoader workers: {train_num_workers}")
    print(f"Evaluation DataLoader workers: {eval_num_workers}")

    max_epochs = 10000

    initial_lr = 1e-3
    weight_decay = 3e-4

    early_stopping_patience = 20
    gradient_clip = None

    input_channels = 46
    num_speakers = 3

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Mixed precision: {use_amp}")

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = HOARoomDataset(
        pt_folder=pt_folder,
        allowed_files=train_files,
        cache_size=1,
    )

    val_dataset = HOARoomDataset(
        pt_folder=pt_folder,
        allowed_files=val_files,
        cache_size=1,
    )

    test_dataset = HOARoomDataset(
        pt_folder=pt_folder,
        allowed_files=test_files,
        cache_size=1,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_batch_sampler = TwoLevelWorkerBatchSampler(
        dataset=train_dataset,
        batch_size=batch_size,
        worker_slots=train_num_workers,
        drop_last=True,
        shuffle_files=True,
        shuffle_within_file=True,
        seed=42,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=train_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(train_num_workers > 0),
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(eval_num_workers > 0),
        prefetch_factor=2,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(eval_num_workers > 0),
        prefetch_factor=2,
        drop_last=False,
    )

    # --------------------------------------------------------
    # Model, loss, optimizer
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

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
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
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
                },
                checkpoint_path,
            )

            print(
                f"Saved new best model to {checkpoint_path}"
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(
                "Early stopping: validation loss did not improve "
                f"for {early_stopping_patience} epochs."
            )
            break

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Loaded best model from epoch {checkpoint['epoch']} "
        f"with validation loss "
        f"{checkpoint['best_val_loss']:.6f}"
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    test_loss = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
    )

    print(f"Final test PIT loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()