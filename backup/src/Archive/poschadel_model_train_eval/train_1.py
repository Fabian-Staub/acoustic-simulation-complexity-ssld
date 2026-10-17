import os
import glob
import itertools
import random
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class HOARoomDataset(Dataset):
    """
    Dataset backed by a lightweight dataset_index.pth file.

    The large .pt shards are not scanned during initialization.
    Only shards listed in allowed_files are included.

    Each item returns:
        X:      (50, 128, C)
        target: (50, 3, 3)
    """

    def __init__(
        self,
        pt_folder: str,
        index_path: str,
        allowed_files,
        cache_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.pt_folder = os.path.abspath(pt_folder)
        self.index_path = os.path.abspath(index_path)
        self.allowed_files = set(
            os.path.basename(path)
            for path in allowed_files
        )
        self.cache_size = max(1, int(cache_size))
        self.dtype = dtype

        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(
                f"Dataset index not found: {self.index_path}\n"
                "Run initialize_dataset_index.py before training."
            )

        index_data = torch.load(
            self.index_path,
            map_location="cpu",
            weights_only=False,
        )

        indexed_files = index_data["pt_files"]

        self.pt_files = [
            os.path.join(self.pt_folder, filename)
            for filename in indexed_files
            if filename in self.allowed_files
        ]

        if not self.pt_files:
            raise RuntimeError(
                "No indexed .pt files matched allowed_files"
            )

        filename_to_local_idx = {
            os.path.basename(path): file_idx
            for file_idx, path in enumerate(self.pt_files)
        }

        # Each entry:
        # (local_file_idx, room_idx, sample_idx)
        self.sample_index = []

        for entry in index_data["entries"]:
            filename = entry["filename"]

            if filename not in filename_to_local_idx:
                continue

            self.sample_index.append(
                (
                    filename_to_local_idx[filename],
                    int(entry["room_idx"]),
                    int(entry["sample_idx"]),
                )
            )

        if not self.sample_index:
            raise RuntimeError(
                "No samples found for selected files"
            )

        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def _load_file(self, file_idx: int):
        if file_idx in self._cache:
            data = self._cache.pop(file_idx)
            self._cache[file_idx] = data
            return data

        path = self.pt_files[file_idx]

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        self._cache[file_idx] = data

        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return data

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int):
        file_idx, room_idx, sample_idx = self.sample_index[index]

        data = self._load_file(file_idx)

        room_record = data["samples"][room_idx]
        sample = room_record["samples"][sample_idx]

        X = sample["X"].to(dtype=self.dtype)
        target = sample["accdoa"].to(dtype=self.dtype)

        if X.ndim != 3:
            raise ValueError(
                f"Expected X shape (T, M, C), got {tuple(X.shape)}"
            )

        if target.shape[-2:] != (3, 3):
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
        for dataset_idx, entry in enumerate(dataset.sample_index):
            file_idx = int(entry[0])
            self.indices_by_file.setdefault(file_idx, []).append(dataset_idx)

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
    grad_clip=None,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for X, target in loader:
        X = X.to(
            device,
            non_blocking=True,
        )
        target = target.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        pred = model(X)
        loss = criterion(pred, target)

        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        optimizer.step()

        batch_size = X.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for X, target in loader:
        X = X.to(
            device,
            non_blocking=True,
        )
        target = target.to(
            device,
            non_blocking=True,
        )

        pred = model(X)
        loss = criterion(pred, target)

        batch_size = X.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def overfit_single_batch(
    train_loader,
    device,
    num_steps=2000,
):
    X, target = next(iter(train_loader))

    X = X.to(device)
    target = target.to(device)

    model = SDLCRNN(
        input_channels=46,
        num_speakers=3,
        dropout_rate=0.0,
    ).to(device)

    criterion = FramewisePITLoss(
        nb_tracks=3,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )

    for step in range(1, num_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        pred = model(X)
        loss = criterion(pred, target)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}: {loss.item()}"
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float("inf"),
        )

        optimizer.step()

        if step in {1, 10, 50, 100, 250, 500, 1000, 2000}:
            print(
                f"step={step:04d} | "
                f"loss={loss.item():.8f} | "
                f"grad_norm={float(grad_norm):.6f}",
                flush=True,
            )

    return model


# ============================================================
# Main training function
# ============================================================


# ============================================================
# Single-batch overfit debug
# ============================================================

def overfit_single_batch(
    train_loader,
    device,
    num_steps=2000,
):
    print("Loading one fixed training batch...", flush=True)

    X, target = next(iter(train_loader))

    print(
        f"Batch shapes: X={tuple(X.shape)}, "
        f"target={tuple(target.shape)}",
        flush=True,
    )

    if not torch.isfinite(X).all():
        raise RuntimeError("Input batch contains NaN or Inf")

    if not torch.isfinite(target).all():
        raise RuntimeError("Target batch contains NaN or Inf")

    X = X.to(device)
    target = target.to(device)

    model = SDLCRNN(
        input_channels=X.shape[-1],
        num_speakers=3,
        dropout_rate=0.0,
    ).to(device)

    criterion = FramewisePITLoss(
        nb_tracks=3,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )

    report_steps = {
        1,
        10,
        50,
        100,
        250,
        500,
        1000,
        1500,
        2000,
    }

    print("\nSingle-batch overfit test")
    print("-------------------------")

    for step in range(1, num_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        pred = model(X)
        loss = criterion(pred, target)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Loss became NaN or Inf at step {step}: "
                f"{loss.item()}"
            )

        loss.backward()

        grad_sq_sum = 0.0

        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_norm = parameter.grad.detach().norm(2).item()
                grad_sq_sum += grad_norm ** 2

        total_grad_norm = grad_sq_sum ** 0.5

        if not torch.isfinite(
            torch.tensor(total_grad_norm)
        ):
            raise RuntimeError(
                f"Gradient norm became NaN or Inf at step {step}"
            )

        optimizer.step()

        if step in report_steps:
            with torch.no_grad():
                pred_tracks = pred.reshape(
                    pred.shape[0],
                    pred.shape[1],
                    3,
                    3,
                )

                pred_norm = torch.linalg.vector_norm(
                    pred_tracks,
                    dim=-1,
                )

                target_norm = torch.linalg.vector_norm(
                    target,
                    dim=-1,
                )

            print(
                f"step={step:04d} | "
                f"loss={loss.item():.8f} | "
                f"grad_norm={total_grad_norm:.6f} | "
                f"pred_norm_mean={pred_norm.mean().item():.4f} | "
                f"target_norm_mean={target_norm.mean().item():.4f}",
                flush=True,
            )

    print("\nOverfit test finished.", flush=True)

    final_loss = loss.item()

    if final_loss < 0.001:
        print("RESULT: very good — pipeline can overfit one batch.")
    elif final_loss < 0.005:
        print("RESULT: good — pipeline can overfit one batch.")
    elif final_loss < 0.020:
        print(
            "RESULT: suspicious — learning works, but the fixed batch "
            "is not fitted closely."
        )
    else:
        print(
            "RESULT: failed — likely a problem in model, loss, "
            "features, or target alignment."
        )

    return model


# ============================================================
# Main debug function
# ============================================================

def main():
    set_seed(42)

    pt_folder = (
        "../../../data/harp/"
        "R4500_P8_S3_Order3_24kHz_shoeBox_rtProfilesMIT_ISMOrder30/"
        "hopiv_accdoa_mel128/pt"
    )

    index_path = os.path.join(
        pt_folder,
        "dataset_index.pth",
    )

    index_data = torch.load(
        index_path,
        map_location="cpu",
        weights_only=False,
    )

    all_pt_files = sorted(index_data["pt_files"])
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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_dataset = HOARoomDataset(
        pt_folder=pt_folder,
        index_path=index_path,
        allowed_files=train_files,
        cache_size=2,
    )

    print(f"Train samples: {len(train_dataset)}")

    # num_workers=0 is intentional for debugging:
    # errors appear directly and no worker process can stall.
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    overfit_single_batch(
        train_loader=train_loader,
        device=device,
        num_steps=2000,
    )


if __name__ == "__main__":
    main()
