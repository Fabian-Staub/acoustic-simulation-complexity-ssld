#!/usr/bin/env python3
"""
Framewise test-set evaluation for the SDLCRNN model.

This script:
1. Loads best_model.pth first.
2. Imports SDLCRNN and HOARoomDataset from train.py.
3. Reads dataset_root from the checkpoint and evaluates its test/pt split.
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
    python evaluate_multi_dataset.py \
        --checkpoint best_model.pth \
        --train-module train \
        --arni-dataset /path/to/arni-dataset \
        --motus-dataset /path/to/motus-dataset \
        --ivan-dataset /path/to/ivan-dataset
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader


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
        description=(
            "Evaluate one checkpoint on the checkpoint dataset and up to three "
            "additional dataset test splits."
        )
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
        "--arni-dataset",
        "--eval-dataset-root-1",
        dest="arni_dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Root of the arni-dataset evaluation set; evaluates <PATH>/test/pt. "
            "The legacy alias --eval-dataset-root-1 is also accepted."
        ),
    )
    parser.add_argument(
        "--motus-dataset",
        "--eval-dataset-root-2",
        dest="motus_dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Root of the motus_dataset evaluation set; evaluates <PATH>/test/pt. "
            "The legacy alias --eval-dataset-root-2 is also accepted."
        ),
    )
    parser.add_argument(
        "--ivan-dataset",
        "--eval-dataset-root-3",
        dest="ivan_dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Root of the ivan_dataset evaluation set; evaluates <PATH>/test/pt. "
            "The legacy alias --eval-dataset-root-3 is also accepted."
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
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Directory for the single combined JSON results file.",
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


def resolve_checkpoint_path(checkpoint_argument: Path) -> Path:
    """Accept either a checkpoint file or a directory containing best_model.pth."""
    checkpoint_argument = checkpoint_argument.expanduser().resolve()
    if checkpoint_argument.is_dir():
        return checkpoint_argument / "best_model.pth"
    return checkpoint_argument


def checkpoint_title(checkpoint_argument: Path, checkpoint_path: Path) -> str:
    """Create a filesystem-safe title for the combined JSON output."""
    if checkpoint_argument.expanduser().resolve().is_dir():
        raw_title = checkpoint_argument.expanduser().resolve().name
    elif checkpoint_path.name == "best_model.pth":
        raw_title = checkpoint_path.parent.name
    else:
        raw_title = checkpoint_path.stem

    safe_title = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in raw_title
    )
    return safe_title or "checkpoint"


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

    if "dataset_root" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'dataset_root'. "
            "The training script must save the dataset root."
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




def load_training_distribution(checkpoint: dict[str, Any]) -> tuple[Path, dict[str, tuple[float, float]], int]:
    """Load normalization statistics only from the checkpoint dataset root."""
    checkpoint_dataset_root = Path(checkpoint["dataset_root"]).expanduser().resolve()
    distribution_path = checkpoint_dataset_root / "distribution.json"

    if not distribution_path.is_file():
        raise FileNotFoundError(
            "distribution.json from the checkpoint dataset root was not found: "
            f"{distribution_path}"
        )

    with distribution_path.open("r", encoding="utf-8") as handle:
        distribution = json.load(handle)

    statistics = distribution.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError(f"Missing 'statistics' in {distribution_path}")

    normalization_statistics: dict[str, tuple[float, float]] = {}
    for feature_name in ("hopiv", "logmel"):
        feature_stats = statistics.get(feature_name)
        if not isinstance(feature_stats, dict):
            raise ValueError(
                f"Missing statistics for {feature_name!r} in {distribution_path}"
            )
        normalization_statistics[feature_name] = (
            float(feature_stats["mean"]),
            float(feature_stats["stdev"]),
        )

    if distribution.get("scope") != "train":
        raise ValueError(
            f"Expected scope='train' in {distribution_path} to avoid leakage."
        )

    hoa_order = int(distribution["hoa_order"])
    return distribution_path, normalization_statistics, hoa_order


def discover_test_shards(dataset_root: Path) -> tuple[Path, list[str]]:
    """Discover all .pt shards in <dataset_root>/test/pt."""
    dataset_root = dataset_root.expanduser().resolve()
    pt_folder = dataset_root / "test" / "pt"

    if not pt_folder.is_dir():
        raise FileNotFoundError(f"Test PT folder not found: {pt_folder}")

    test_files = sorted(
        path.name for path in pt_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".pt"
    )
    if not test_files:
        raise RuntimeError(f"No .pt test shards found in {pt_folder}")

    return pt_folder, test_files


def make_test_loader(
    dataset_class,
    dataset_root: Path,
    normalization_statistics: dict[str, tuple[float, float]],
    hoa_order: int,
    feature_layout: str,
    batch_size: int,
    num_workers: int,
    dataset_label: str,
) -> tuple[DataLoader, Path, list[str]]:
    """Construct a test loader from <dataset_root>/test/pt."""
    pt_folder, test_files = discover_test_shards(dataset_root)

    dataset = dataset_class(
        pt_folder=str(pt_folder),
        pt_files=test_files,
        normalization_statistics=normalization_statistics,
        hoa_order=hoa_order,
        preload_label=f"test-eval-{dataset_label}",
        feature_layout=feature_layout,
    )

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
        loader_kwargs["multiprocessing_context"] = "fork"

    return DataLoader(**loader_kwargs), pt_folder, test_files


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


def print_results(results: dict[str, Any], dataset_name: str) -> None:
    print_metric_block(
        results["overall"],
        f"{dataset_name}: framewise test-set SDL evaluation: all frames",
    )

    for source_count in range(4):
        print_metric_block(
            results["by_gt_source_count"][str(source_count)],
            (
                f"{dataset_name}: framewise SDL evaluation: "
                f"{source_count} active ground-truth source(s)"
            ),
        )


def safe_dataset_name(dataset_root: Path, index: int) -> str:
    name = dataset_root.resolve().name or f"dataset_{index}"
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in name)
    return f"{index}_{safe}"


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = load_checkpoint_first(checkpoint_path, device)
    model_class, dataset_class = import_training_objects(args.train_module)

    checkpoint_dataset_root = Path(checkpoint["dataset_root"]).expanduser().resolve()
    distribution_path, normalization_statistics, hoa_order = load_training_distribution(
        checkpoint
    )

    dataset_specs: list[tuple[str, Path]] = [
        ("checkpoint_dataset", checkpoint_dataset_root),
    ]
    for dataset_name, dataset_root in (
        ("arni-dataset", args.arni_dataset),
        ("motus-dataset", args.motus_dataset),
        ("ivan-dataset", args.ivan_dataset),
    ):
        if dataset_root is not None:
            dataset_specs.append((dataset_name, dataset_root))

    input_channels = int(checkpoint.get("input_channels", 46))
    num_speakers = int(checkpoint.get("num_speakers", 3))
    feature_layout = str(checkpoint.get("feature_layout", "hopiv_then_logmel"))

    print(f"Checkpoint:                    {checkpoint_path}")
    print(f"Device:                        {device}")
    print(f"Checkpoint dataset root:       {checkpoint_dataset_root}")
    print(f"Normalization distribution:    {distribution_path}")
    print(f"Number of evaluation datasets: {len(dataset_specs)}")

    model = model_class(
        input_channels=input_channels,
        num_speakers=num_speakers,
        dropout_rate=0.1,
    ).to(device)
    model.load_state_dict(normalize_state_dict(checkpoint["model_state_dict"]))

    if "epoch" in checkpoint:
        print(f"Loaded epoch:                  {checkpoint['epoch']}")
    if "best_val_loss" in checkpoint:
        print(f"Best val loss:                 {checkpoint['best_val_loss']:.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}

    for dataset_index, (dataset_label, dataset_root) in enumerate(dataset_specs, start=1):
        dataset_root = dataset_root.expanduser().resolve()
        dataset_name = safe_dataset_name(Path(dataset_label), dataset_index)

        print("\n" + "#" * 76)
        print(f"Evaluating dataset {dataset_index}/{len(dataset_specs)}: {dataset_label} ({dataset_root})")
        print("#" * 76)

        test_loader, test_folder, test_files = make_test_loader(
            dataset_class=dataset_class,
            dataset_root=dataset_root,
            normalization_statistics=normalization_statistics,
            hoa_order=hoa_order,
            feature_layout=feature_layout,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dataset_label=dataset_name,
        )

        print(f"Test folder: {test_folder}")
        print(f"Test shards: {len(test_files)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"Normalization source: {distribution_path}")

        results = evaluate_every_frame(
            model=model,
            loader=test_loader,
            device=device,
            activity_threshold=args.activity_threshold,
            reference_activity_threshold=args.reference_activity_threshold,
        )
        print_results(results, dataset_name=dataset_label)

        source_count_confusion = np.asarray(
            results["source_count_confusion_matrix"],
            dtype=np.int64,
        )
        row_totals = source_count_confusion.sum(axis=1, keepdims=True)
        source_count_confusion_percent = np.divide(
            source_count_confusion.astype(np.float64),
            row_totals,
            out=np.zeros_like(source_count_confusion, dtype=np.float64),
            where=row_totals != 0,
        ) * 100.0

        print("\nSource-count confusion matrix (%) (rows=true, columns=predicted):")
        print(np.round(source_count_confusion_percent, 2))
        result_payload = {
            "dataset_label": dataset_label,
            "dataset_root": str(dataset_root),
            "test_folder": str(test_folder),
            #"test_files": test_files,
            "normalization_distribution": str(distribution_path),
            "results": results,
            "source_count_confusion_matrix_counts": source_count_confusion.tolist(),
            "source_count_confusion_matrix_percent": source_count_confusion_percent.tolist(),
        }
        all_results[dataset_name] = result_payload

        del test_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_path = args.output_dir / f"eval_{checkpoint_title(args.checkpoint, checkpoint_path)}.json"
    combined_payload = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "checkpoint_dataset_root": str(checkpoint_dataset_root),
        "normalization_distribution": str(distribution_path),
        "activity_threshold": args.activity_threshold,
        "reference_activity_threshold": args.reference_activity_threshold,
        "datasets": all_results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(combined_payload, handle, indent=2)
    print(f"\nSaved combined results: {output_path}")


if __name__ == "__main__":
    main()
