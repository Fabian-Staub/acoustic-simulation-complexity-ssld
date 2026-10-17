import glob
import os
from collections import OrderedDict
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class HOAPTDataset(Dataset):
    """
    Dataset for the generated room_XXXX-XXXX.pt files.

    Each item returns:
        X:      (50, 128, C)
        target: (50, 3, 3)

    The PT files are indexed once during initialization. Samples are loaded
    lazily, with a small file cache to avoid repeatedly reading the same file.
    """

    def __init__(
        self,
        pt_folder: str,
        cache_size: int = 2,
        return_metadata: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.pt_folder = os.path.abspath(pt_folder)
        self.cache_size = max(1, int(cache_size))
        self.return_metadata = return_metadata
        self.dtype = dtype

        self.pt_files = sorted(
            glob.glob(os.path.join(self.pt_folder, "*.pt"))
        )

        if not self.pt_files:
            raise FileNotFoundError(
                f"No .pt files found in: {self.pt_folder}"
            )

        # Each entry is:
        # (file_index, room_index, sample_index)
        self.sample_index: list[tuple[int, int, int]] = []

        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

        self._build_index()

    def _build_index(self) -> None:
        """
        Read each PT file once and record where every sample is located.
        """

        for file_idx, pt_path in enumerate(self.pt_files):
            batch_data = torch.load(
                pt_path,
                map_location="cpu",
                weights_only=False,
            )

            room_records = batch_data.get("samples")

            if room_records is None:
                raise KeyError(
                    f"Missing key 'samples' in {pt_path}"
                )

            for room_idx, room_record in enumerate(room_records):
                room_samples = room_record.get("samples")

                if room_samples is None:
                    raise KeyError(
                        f"Missing room-level 'samples' in {pt_path}, "
                        f"room index {room_idx}."
                    )

                for sample_idx in range(len(room_samples)):
                    self.sample_index.append(
                        (file_idx, room_idx, sample_idx)
                    )

        if not self.sample_index:
            raise RuntimeError(
                f"No samples found in PT files under {self.pt_folder}"
            )

    def _load_file(self, file_idx: int) -> dict[str, Any]:
        """
        Load one PT file and keep recently accessed files in memory.
        """

        if file_idx in self._cache:
            data = self._cache.pop(file_idx)
            self._cache[file_idx] = data
            return data

        pt_path = self.pt_files[file_idx]

        data = torch.load(
            pt_path,
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

        batch_data = self._load_file(file_idx)

        room_record = batch_data["samples"][room_idx]
        sample = room_record["samples"][sample_idx]

        X = sample["X"].to(dtype=self.dtype)
        target = sample["accdoa"].to(dtype=self.dtype)

        if X.ndim != 3:
            raise ValueError(
                f"Expected X shape (T, M, C), got {tuple(X.shape)}."
            )

        if target.ndim != 3 or target.shape[-1] != 3:
            raise ValueError(
                "Expected ACCDOA shape (T, tracks, 3), "
                f"got {tuple(target.shape)}."
            )

        if X.shape[0] != target.shape[0]:
            raise ValueError(
                f"Time mismatch: X has {X.shape[0]} frames, "
                f"target has {target.shape[0]} frames."
            )

        if not self.return_metadata:
            return X, target

        metadata = {
            "sample_id": sample.get("sample_id"),
            "segment_idx": sample.get("segment_idx"),
            "base_name": sample.get("base_name"),
            "room": sample.get("room", room_record.get("room")),
            "position": sample.get("position"),
            "segment_start_s": sample.get("segment_start_s"),
            "padded": sample.get("padded"),
            "pt_file": os.path.basename(self.pt_files[file_idx]),
        }

        return X, target, metadata