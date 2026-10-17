#!/usr/bin/env python3
"""
Framewise test-set evaluation for the SDLCRNN model.

This script:
1. Loads best_model.pth first.
2. Imports SDLCRNN and HOARoomDataset from train.py.
3. Reconstructs the exact test split stored in the checkpoint.
4. Evaluates every time frame independently.
5. Reports the paper's framewise SDL metrics only:
   - localization error (LE), without an angular threshold
   - localization recall (LR), without an angular threshold
   - location-sensitive F-score at 15 degrees (F15)
   - location-sensitive error rate at 15 degrees (ER15)
   - aggregate SDL score (ESDL)

Requirements:
    pip install scipy

Example:
    python evaluate_test.py \
        --checkpoint best_model.pth \
        --train-module train \
        --pt-folder ../../../data/harp/.../hopiv_accdoa_mel128/pt
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset


EPS = 1e-10
SPATIAL_THRESHOLD_DEG = 15.0


@dataclass
class MetricAccumulator:
    num_frames: int = 0
    num_reference_events: int = 0
    num_prediction_events: int = 0

    tp: int = 0
    fp: int = 0
    fn: int = 0

    tp_spatial: int = 0
    fp_spatial: int = 0

    localization_error_sum: float = 0.0
    localization_matches: int = 0

    exact_count_frames: int = 0
    count_abs_error_sum: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SDLCRNN on every test-set frame."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("best_model.pth"),
        help="Path to the saved best-model checkpoint.",
    )
    parser.add_argument(
        "--train-module",
        default="train",
        help=(
            "Python module containing SDLCRNN and HOARoomDataset. "
            "For train.py, use: train"
        ),
    )
    parser.add_argument(
        "--pt-folder",
        type=Path,
        required=True,
        help="Folder containing the dataset .pt shards and dataset_index.pth.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to dataset_index.pth. "
            "Defaults to <pt-folder>/dataset_index.pth. If it does not exist, "
            "all .pt shards are indexed in RAM for evaluation."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=0.5,
        help="Predicted ACCDOA norm required for an active source.",
    )
    parser.add_argument(
        "--reference-activity-threshold",
        type=float,
        default=1e-6,
        help="Reference ACCDOA norm required for an active source.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("test_framewise_seld_metrics.json"),
    )
    parser.add_argument(
        "--confusion-matrix-png",
        type=Path,
        default=Path("source_count_confusion_matrix.png"),
        help=(
            "Output path for the source-count confusion matrix PNG. "
            "Counts are based only on activity thresholds; no spatial threshold is used."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")

    return torch.device(name)


def import_training_objects(module_name: str):
    """
    Import the model and dataset definitions from train.py.

    The supplied training script uses the class name SDLCRNN. If your local
    file instead uses SDL_CRNN, this function accepts that name as a fallback.
    """
    module = importlib.import_module(module_name)

    model_class = getattr(module, "SDLCRNN", None)
    if model_class is None:
        model_class = getattr(module, "SDL_CRNN", None)

    if model_class is None:
        raise ImportError(
            f"{module_name}.py must define SDLCRNN or SDL_CRNN."
        )

    dataset_class = getattr(module, "HOARoomDataset", None)
    if dataset_class is None:
        raise ImportError(
            f"{module_name}.py must define HOARoomDataset."
        )

    return model_class, dataset_class


def load_checkpoint_first(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load best_model.pth before constructing the model or test dataset.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected best_model.pth to contain a checkpoint dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    if "test_files" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'test_files'. "
            "The training script should save the file-based test split."
        )

    return checkpoint


def normalize_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Remove a DataParallel/DistributedDataParallel 'module.' prefix if present.
    """
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }




class InMemoryPTShardDataset(Dataset):
    """
    RAM-only fallback dataset used when dataset_index.pth is unavailable.

    It scans all .pt files in ``pt_folder``, reads the generated shard format,
    and stores lightweight (shard_path, nested sample position) references in
    memory. Shard contents are loaded lazily and cached per worker. No index
    file is written to disk.
    """

    def __init__(self, pt_folder: str | Path, cache_size: int = 2):
        self.pt_folder = Path(pt_folder)
        self.cache_size = max(1, int(cache_size))
        self._cache: dict[str, Any] = {}
        self._cache_order: list[str] = []
        self.entries: list[tuple[str, int, int | None]] = []

        shard_paths = sorted(
            path for path in self.pt_folder.glob("*.pt")
            if path.is_file() and path.name != "dataset_index.pth"
        )
        if not shard_paths:
            raise FileNotFoundError(
                f"No .pt dataset shards found in: {self.pt_folder}"
            )

        for shard_path in shard_paths:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            samples = payload.get("samples") if isinstance(payload, dict) else None
            if not isinstance(samples, list):
                raise ValueError(
                    f"Unsupported shard format in {shard_path}: expected a "
                    "dictionary containing a list under 'samples'."
                )

            # Current generator format: top-level samples are room records,
            # and each room record contains its own list under 'samples'.
            if samples and isinstance(samples[0], dict) and isinstance(samples[0].get("samples"), list):
                for room_index, room_record in enumerate(samples):
                    for sample_index in range(len(room_record["samples"])):
                        self.entries.append(
                            (str(shard_path), room_index, sample_index)
                        )
            else:
                # Also accept a flat list of sample dictionaries.
                for sample_index in range(len(samples)):
                    self.entries.append(
                        (str(shard_path), sample_index, None)
                    )

        if not self.entries:
            raise RuntimeError(
                f"The .pt shards in {self.pt_folder} contain no samples."
            )

    def __len__(self) -> int:
        return len(self.entries)

    def _load_shard(self, shard_path: str) -> Any:
        if shard_path in self._cache:
            self._cache_order.remove(shard_path)
            self._cache_order.append(shard_path)
            return self._cache[shard_path]

        payload = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )
        self._cache[shard_path] = payload
        self._cache_order.append(shard_path)

        while len(self._cache_order) > self.cache_size:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)

        return payload

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_path, outer_index, inner_index = self.entries[index]
        payload = self._load_shard(shard_path)
        samples = payload["samples"]

        if inner_index is None:
            sample = samples[outer_index]
        else:
            sample = samples[outer_index]["samples"][inner_index]

        if not isinstance(sample, dict) or "X" not in sample or "accdoa" not in sample:
            raise ValueError(
                f"Invalid sample in {shard_path}: expected keys 'X' and 'accdoa'."
            )

        features = sample["X"]
        targets = sample["accdoa"]

        if not torch.is_tensor(features):
            features = torch.as_tensor(features)
        if not torch.is_tensor(targets):
            targets = torch.as_tensor(targets)

        return features.to(torch.float32), targets.to(torch.float32)


def make_test_loader(
    dataset_class,
    pt_folder: Path,
    index_path: Path,
    test_files: list[str],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    if index_path.is_file():
        dataset = dataset_class(
            pt_folder=str(pt_folder),
            index_path=str(index_path),
            allowed_files=test_files,
            cache_size=max(1, num_workers),
        )
        dataset_source = f"saved index: {index_path}"
    else:
        print(
            f"Index not found: {index_path}\n"
            f"Building a RAM-only index from all .pt files in: {pt_folder}"
        )
        dataset = InMemoryPTShardDataset(
            pt_folder=pt_folder,
            cache_size=max(1, num_workers),
        )
        dataset_source = "RAM-only index over all .pt shards"

    print(f"Dataset source: {dataset_source}")

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "drop_last": False,
    }

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(**loader_kwargs)


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def pairwise_angular_distance_deg(
    reference_vectors: torch.Tensor,
    prediction_vectors: torch.Tensor,
) -> torch.Tensor:
    reference_unit = reference_vectors / reference_vectors.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(EPS)

    prediction_unit = prediction_vectors / prediction_vectors.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(EPS)

    cosine = reference_unit @ prediction_unit.transpose(0, 1)
    cosine = cosine.clamp(-1.0, 1.0)

    return torch.rad2deg(torch.acos(cosine))


def update_one_frame(
    accumulator: MetricAccumulator,
    reference_frame: torch.Tensor,
    prediction_frame: torch.Tensor,
    activity_threshold: float,
    reference_activity_threshold: float,
) -> None:
    reference_frame = reference_frame.to(dtype=torch.float64, device="cpu")
    prediction_frame = prediction_frame.to(dtype=torch.float64, device="cpu")

    reference_norm = reference_frame.norm(dim=-1)
    prediction_norm = prediction_frame.norm(dim=-1)

    active_reference = reference_frame[
        reference_norm > reference_activity_threshold
    ]
    active_prediction = prediction_frame[
        prediction_norm >= activity_threshold
    ]

    num_reference = int(active_reference.shape[0])
    num_prediction = int(active_prediction.shape[0])

    accumulator.num_frames += 1
    accumulator.num_reference_events += num_reference
    accumulator.num_prediction_events += num_prediction
    accumulator.count_abs_error_sum += abs(num_reference - num_prediction)

    if num_reference == num_prediction:
        accumulator.exact_count_frames += 1

    if num_reference == 0 and num_prediction == 0:
        return

    if num_reference == 0:
        accumulator.fp += num_prediction
        return

    if num_prediction == 0:
        accumulator.fn += num_reference
        return

    cost_matrix = pairwise_angular_distance_deg(
        active_reference,
        active_prediction,
    )

    reference_indices, prediction_indices = linear_sum_assignment(
        cost_matrix.numpy()
    )

    matched_angles = cost_matrix[
        reference_indices,
        prediction_indices,
    ]

    num_matches = int(len(reference_indices))
    unmatched_reference = num_reference - num_matches
    unmatched_prediction = num_prediction - num_matches

    accumulator.tp += num_matches
    accumulator.fn += unmatched_reference
    accumulator.fp += unmatched_prediction

    accumulator.localization_error_sum += float(matched_angles.sum().item())
    accumulator.localization_matches += num_matches

    num_spatial_tp = int(
        (matched_angles <= SPATIAL_THRESHOLD_DEG).sum().item()
    )
    accumulator.tp_spatial += num_spatial_tp
    accumulator.fp_spatial += num_matches - num_spatial_tp


def finalize_metrics(accumulator: MetricAccumulator) -> dict[str, float]:
    # F15 and ER15 use the paper's fixed 15-degree location threshold.
    precision_15 = safe_divide(
        accumulator.tp_spatial,
        accumulator.tp_spatial
        + accumulator.fp_spatial
        + accumulator.fp,
    )

    recall_15 = safe_divide(
        accumulator.tp_spatial,
        accumulator.tp_spatial + accumulator.fn,
    )

    f15 = safe_divide(
        2.0 * precision_15 * recall_15,
        precision_15 + recall_15,
    )

    er15 = safe_divide(
        accumulator.fp
        + accumulator.fp_spatial
        + accumulator.fn,
        accumulator.num_reference_events,
    )

    # LE and LR are threshold-free localization metrics. Every Hungarian
    # assignment contributes to LE, irrespective of its angular error.
    le = safe_divide(
        accumulator.localization_error_sum,
        accumulator.localization_matches,
    )

    lr = safe_divide(
        accumulator.tp,
        accumulator.tp + accumulator.fn,
    )

    esdl = 0.25 * (
        er15
        + (1.0 - f15)
        + le / 180.0
        + (1.0 - lr)
    )

    return {
        "LE": le,
        "LR": lr,
        "F15": f15,
        "ER15": er15,
        "ESDL": esdl,
    }


@torch.inference_mode()
def evaluate_every_frame(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    activity_threshold: float,
    reference_activity_threshold: float,
) -> dict[str, Any]:
    """
    Evaluate all frames and also evaluate frames grouped by the number of
    active ground-truth sources.

    The grouped metrics are conditioned on the reference source count:
        by_gt_source_count["0"] -> frames with 0 active reference sources
        by_gt_source_count["1"] -> frames with 1 active reference source
        by_gt_source_count["2"] -> frames with 2 active reference sources
        by_gt_source_count["3"] -> frames with 3 active reference sources
    """
    model.eval()

    overall_accumulator = MetricAccumulator()
    count_accumulators = {
        count: MetricAccumulator()
        for count in range(4)
    }
    source_count_confusion = np.zeros((4, 4), dtype=np.int64)

    for batch_index, (features, targets) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=True)

        predictions = model(features)

        if predictions.ndim == 3:
            if predictions.shape[-1] % 3 != 0:
                raise ValueError(
                    "The model output's final dimension must be divisible by 3."
                )

            predictions = predictions.reshape(
                predictions.shape[0],
                predictions.shape[1],
                predictions.shape[-1] // 3,
                3,
            )

        if targets.ndim != 4 or targets.shape[-1] != 3:
            raise ValueError(
                "Expected targets with shape (B, T, tracks, 3), "
                f"got {tuple(targets.shape)}."
            )

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Prediction shape {tuple(predictions.shape)} does not match "
                f"target shape {tuple(targets.shape)}."
            )

        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()

        batch_size, num_frames, _, _ = targets.shape

        for sample_index in range(batch_size):
            for frame_index in range(num_frames):
                reference_frame = targets[sample_index, frame_index]
                prediction_frame = predictions[sample_index, frame_index]

                reference_count = int(
                    (
                        reference_frame.norm(dim=-1)
                        > reference_activity_threshold
                    ).sum().item()
                )
                prediction_count = int(
                    (
                        prediction_frame.norm(dim=-1)
                        >= activity_threshold
                    ).sum().item()
                )

                # Source-count estimation uses activity only. No angular or
                # spatial threshold is involved in this confusion matrix.
                max_count = source_count_confusion.shape[0] - 1
                source_count_confusion[
                    min(reference_count, max_count),
                    min(prediction_count, max_count),
                ] += 1

                update_one_frame(
                    accumulator=overall_accumulator,
                    reference_frame=reference_frame,
                    prediction_frame=prediction_frame,
                    activity_threshold=activity_threshold,
                    reference_activity_threshold=reference_activity_threshold,
                )

                if reference_count in count_accumulators:
                    update_one_frame(
                        accumulator=count_accumulators[reference_count],
                        reference_frame=reference_frame,
                        prediction_frame=prediction_frame,
                        activity_threshold=activity_threshold,
                        reference_activity_threshold=reference_activity_threshold,
                    )

        if batch_index % 100 == 0:
            print(
                f"Processed {batch_index} batches / "
                f"{overall_accumulator.num_frames} frames"
            )

    return {
        "overall": finalize_metrics(overall_accumulator),
        "by_gt_source_count": {
            str(count): finalize_metrics(accumulator)
            for count, accumulator in count_accumulators.items()
        },
        "source_count_confusion_matrix": source_count_confusion.tolist(),
    }



def plot_source_count_confusion_matrix(
    confusion_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save a row-normalized true-vs-estimated source-count confusion matrix.

    Each row corresponds to one ground-truth source count and sums to 100%
    when that row contains at least one frame. Values are displayed with
    zero decimal places.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    confusion_matrix = np.asarray(confusion_matrix, dtype=np.float64)
    row_totals = confusion_matrix.sum(axis=1, keepdims=True)

    confusion_percent = np.divide(
        confusion_matrix,
        row_totals,
        out=np.zeros_like(confusion_matrix),
        where=row_totals != 0,
    ) * 100.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(
        confusion_percent,
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=100.0,
    )
    fig.colorbar(
        image,
        ax=ax,
        label="Percentage of ground-truth frames (%)",
    )

    labels = [str(index) for index in range(confusion_percent.shape[0])]
    ax.set(
        title="Source number estimation confusion matrix",
        xlabel="Estimated number of sources",
        ylabel="Ground-truth number of sources",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )

    for row in range(confusion_percent.shape[0]):
        for column in range(confusion_percent.shape[1]):
            value = confusion_percent[row, column]
            ax.text(
                column,
                row,
                f"{value:.0f}%",
                ha="center",
                va="center",
                color="white" if value > 50.0 else "black",
            )

    ax.set_ylim(confusion_percent.shape[0] - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_metric_block(
    results: dict[str, float],
    title: str,
) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)
    print(f"LE                                    {results['LE']:.2f} deg")
    print(f"LR                                    {results['LR']:.4f}")
    print(f"F15                                   {results['F15']:.4f}")
    print(f"ER15                                  {results['ER15']:.4f}")
    print(f"ESDL                                  {results['ESDL']:.4f}")


def print_results(results: dict[str, Any]) -> None:
    print_metric_block(
        results["overall"],
        "Framewise test-set SDL evaluation: all frames",
    )

    for source_count in range(4):
        print_metric_block(
            results["by_gt_source_count"][str(source_count)],
            (
                "Framewise SDL evaluation: "
                f"{source_count} active ground-truth source(s)"
            ),
        )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    # Load best_model.pth first, as requested.
    checkpoint = load_checkpoint_first(args.checkpoint, device)

    # Import reusable definitions from train.py after loading the checkpoint.
    model_class, dataset_class = import_training_objects(args.train_module)

    input_channels = int(checkpoint.get("input_channels", 46))
    num_speakers = int(checkpoint.get("num_speakers", 3))
    test_files = list(checkpoint["test_files"])

    index_path = (
        args.index_path
        if args.index_path is not None
        else args.pt_folder / "dataset_index.pth"
    )

    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Device:      {device}")
    print(f"Test shards: {len(test_files)}")
    print(f"Index:       {index_path}")

    model = model_class(
        input_channels=input_channels,
        num_speakers=num_speakers,
        dropout_rate=0.2,
    ).to(device)

    model.load_state_dict(
        normalize_state_dict(checkpoint["model_state_dict"])
    )

    if "epoch" in checkpoint:
        print(f"Loaded epoch: {checkpoint['epoch']}")
    if "best_val_loss" in checkpoint:
        print(f"Best val loss: {checkpoint['best_val_loss']:.6f}")

    test_loader = make_test_loader(
        dataset_class=dataset_class,
        pt_folder=args.pt_folder,
        index_path=index_path,
        test_files=test_files,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Test samples: {len(test_loader.dataset)}")

    results = evaluate_every_frame(
        model=model,
        loader=test_loader,
        device=device,
        activity_threshold=args.activity_threshold,
        reference_activity_threshold=args.reference_activity_threshold,
    )

    print_results(results)

    source_count_confusion = np.asarray(
        results["source_count_confusion_matrix"],
        dtype=np.int64,
    )
    plot_source_count_confusion_matrix(
        confusion_matrix=source_count_confusion,
        output_path=args.confusion_matrix_png,
    )
    print(f"Saved source-count confusion matrix to: {args.confusion_matrix_png}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nSaved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
