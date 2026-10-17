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
    Load one .pt shard and extract lightweight sample metadata.

    This function runs inside a worker process.
    """
    filename = os.path.basename(path)

    # Avoid each worker spawning additional PyTorch CPU threads.
    torch.set_num_threads(1)

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    room_records = data["samples"]
    entries: list[dict[str, Any]] = []

    for room_idx, room_record in enumerate(room_records):
        room_id = int(room_record["room"])
        samples = room_record["samples"]

        entries.extend(
            {
                "filename": filename,
                "file_idx": file_idx,
                "room_idx": room_idx,
                "room_id": room_id,
                "sample_idx": sample_idx,
            }
            for sample_idx in range(len(samples))
        )

    file_record = {
        "filename": filename,
        "file_idx": file_idx,
        "num_rooms": len(room_records),
        "num_samples": len(entries),
    }

    del room_records
    del data

    gc.collect()

    return file_idx, file_record, entries


def build_dataset_index(
    pt_folder: str,
    output_path: str,
    max_workers: int | None = None,
) -> None:
    """
    Scan .pt shards in parallel and save a lightweight sample index.

    Parallelism is applied across shard files. Each worker loads one shard,
    extracts metadata, and releases the loaded tensors before processing
    another shard.

    The index contains only:
      - filename
      - file_idx
      - room_idx
      - room_id
      - sample_idx
    """
    pt_folder = os.path.abspath(pt_folder)
    output_path = os.path.abspath(output_path)

    pt_files = sorted(
        glob.glob(os.path.join(pt_folder, "*.pt"))
    )

    # Prevent the generated index itself from being treated as an input shard
    # if it happens to use the .pt extension.
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
        # Loading shards is usually constrained by RAM and disk bandwidth.
        # A conservative default helps avoid loading too many large shards.
        max_workers = min(4, len(pt_files), os.cpu_count() or 1)
    else:
        max_workers = max(1, min(max_workers, len(pt_files)))

    print(f"Found {len(pt_files)} .pt files")
    print(f"Using {max_workers} worker processes")
    print(f"Writing index to: {output_path}")

    # Results are stored by file index so the final index remains deterministic,
    # regardless of the order in which workers finish.
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
                f"({file_record['num_samples']} samples)",
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
        "version": 1,
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

    # Write to a temporary file first to reduce the risk of leaving a partial
    # index if the process is interrupted during torch.save.
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
        "hopiv_accdoa_mel128_TIMIT_babble_test/pt"
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