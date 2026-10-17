#!/usr/bin/env python3

import gc
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import torch


def scan_shard(
    file_idx: int,
    path: str,
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    """
    Load one tensor-format .pt shard and extract lightweight sample metadata.

    Expected shard format:
        (X, accdoa)

    Expected shapes:
        X:      (N, 50, n_mels, feature_channels)
        accdoa: (N, 50, 3, 3)

    This function runs inside a worker process.
    """
    filename = os.path.basename(path)

    # Avoid each worker spawning additional PyTorch CPU threads.
    torch.set_num_threads(1)

    loaded = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(loaded, (tuple, list)) or len(loaded) != 2:
        raise ValueError(
            f"Expected {filename} to contain a tuple (X, accdoa), "
            f"but found {type(loaded).__name__}"
        )

    X, accdoa = loaded

    if not isinstance(X, torch.Tensor):
        raise TypeError(
            f"Expected X in {filename} to be a torch.Tensor, "
            f"but found {type(X).__name__}"
        )

    if not isinstance(accdoa, torch.Tensor):
        raise TypeError(
            f"Expected accdoa in {filename} to be a torch.Tensor, "
            f"but found {type(accdoa).__name__}"
        )

    if X.ndim != 4:
        raise ValueError(
            f"Expected X in {filename} to have shape "
            f"(N, frames, mel_bins, feature_channels), "
            f"but got {tuple(X.shape)}"
        )

    if accdoa.ndim != 4:
        raise ValueError(
            f"Expected accdoa in {filename} to have shape "
            f"(N, frames, speakers, coordinates), "
            f"but got {tuple(accdoa.shape)}"
        )

    if X.shape[0] != accdoa.shape[0]:
        raise ValueError(
            f"Sample-count mismatch in {filename}: "
            f"X contains {X.shape[0]} samples, while accdoa contains "
            f"{accdoa.shape[0]} samples"
        )

    if X.shape[1] != accdoa.shape[1]:
        raise ValueError(
            f"Frame-count mismatch in {filename}: "
            f"X has {X.shape[1]} frames per sample, while accdoa has "
            f"{accdoa.shape[1]}"
        )

    num_samples = int(X.shape[0])

    entries = [
        {
            "filename": filename,
            "file_idx": file_idx,
            "sample_idx": sample_idx,
        }
        for sample_idx in range(num_samples)
    ]

    file_record = {
        "filename": filename,
        "file_idx": file_idx,
        "num_samples": num_samples,
        "x_shape": tuple(int(value) for value in X.shape),
        "accdoa_shape": tuple(int(value) for value in accdoa.shape),
        "x_dtype": str(X.dtype),
        "accdoa_dtype": str(accdoa.dtype),
    }

    del X
    del accdoa
    del loaded

    gc.collect()

    return file_idx, file_record, entries


def build_dataset_index(
    pt_folder: str,
    output_path: str,
    max_workers: int | None = None,
) -> None:
    """
    Scan tensor-format .pt shards in parallel and save a sample index.

    Each input shard must contain:

        (X, accdoa)

    where the first tensor dimension is the sample dimension.

    The per-sample index contains only:
        - filename
        - file_idx
        - sample_idx

    Room indices and room IDs are not included because the new .pt files do
    not contain room metadata. That information remains in the companion CSV
    files created during dataset generation.
    """
    pt_folder = os.path.abspath(pt_folder)
    output_path = os.path.abspath(output_path)

    pt_files = sorted(
        glob.glob(os.path.join(pt_folder, "*.pt"))
    )

    # Prevent a previously generated index from being treated as an input
    # shard if it uses the .pt extension.
    pt_files = [
        path
        for path in pt_files
        if os.path.abspath(path) != output_path
    ]

    if not pt_files:
        raise FileNotFoundError(
            f"No .pt files found in: {pt_folder}"
        )

    if max_workers is None:
        # Loading shards is generally constrained by RAM and disk bandwidth.
        max_workers = min(
            4,
            len(pt_files),
            os.cpu_count() or 1,
        )
    else:
        max_workers = max(
            1,
            min(max_workers, len(pt_files)),
        )

    print(f"Found {len(pt_files)} .pt files")
    print(f"Using {max_workers} worker processes")
    print(f"Writing index to: {output_path}")

    # Store results by file index so output ordering remains deterministic,
    # regardless of worker completion order.
    file_results: list[
        tuple[dict[str, Any], list[dict[str, Any]]] | None
    ] = [None] * len(pt_files)

    with ProcessPoolExecutor(
        max_workers=max_workers,
    ) as executor:
        future_to_file = {
            executor.submit(
                scan_shard,
                file_idx,
                path,
            ): (file_idx, path)
            for file_idx, path in enumerate(pt_files)
        }

        completed = 0

        for future in as_completed(future_to_file):
            expected_file_idx, path = future_to_file[future]
            filename = os.path.basename(path)

            try:
                file_idx, file_record, shard_entries = future.result()
            except Exception as exc:
                # Cancel work that has not started yet.
                for pending_future in future_to_file:
                    pending_future.cancel()

                raise RuntimeError(
                    f"Failed to scan shard: {path}"
                ) from exc

            if file_idx != expected_file_idx:
                raise RuntimeError(
                    f"Worker returned unexpected file index for {filename}: "
                    f"expected {expected_file_idx}, got {file_idx}"
                )

            file_results[file_idx] = (
                file_record,
                shard_entries,
            )

            completed += 1

            print(
                f"[{completed}/{len(pt_files)}] "
                f"Scanned and released {filename} "
                f"({file_record['num_samples']} samples, "
                f"X={file_record['x_shape']}, "
                f"accdoa={file_record['accdoa_shape']})",
                flush=True,
            )

    file_records: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []

    for file_idx, result in enumerate(file_results):
        if result is None:
            raise RuntimeError(
                f"Missing scan result for file index {file_idx}"
            )

        file_record, shard_entries = result
        file_records.append(file_record)
        entries.extend(shard_entries)

    index_data = {
        "version": 2,
        "format": "x_accdoa_tensor_tuple",
        "pt_folder": pt_folder,
        "pt_files": [
            os.path.basename(path)
            for path in pt_files
        ],
        "files": file_records,
        "entries": entries,
        "num_files": len(pt_files),
        "num_samples": len(entries),
    }

    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    # Write to a temporary file first to reduce the chance of leaving a
    # partially written index after an interruption.
    temporary_output_path = f"{output_path}.tmp"

    try:
        torch.save(
            index_data,
            temporary_output_path,
        )

        os.replace(
            temporary_output_path,
            output_path,
        )
    finally:
        if os.path.exists(temporary_output_path):
            os.remove(temporary_output_path)

    print()
    print("Index created successfully")
    print(f"Files:   {len(pt_files)}")
    print(f"Samples: {len(entries)}")
    print(f"Output:  {output_path}")


def main() -> None:
    pt_folder = (
        "../../../data/harp/"
        "R4000_P8_S3_Order3_24kHz_shoeBox_rtProfilesMIT_ISMOrder30/"
        "hopiv_accdoa_mel128_TIMIT_babble_full/pt/normalized"
    )

    output_path = os.path.join(
        pt_folder,
        "dataset_index.pth",
    )

    build_dataset_index(
        pt_folder=pt_folder,
        output_path=output_path,
        max_workers=10,
    )


if __name__ == "__main__":
    main()