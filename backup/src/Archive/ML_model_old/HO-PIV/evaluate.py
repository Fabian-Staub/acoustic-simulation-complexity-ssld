import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from torchinfo import summary
import os
import re
import csv


# ---------------------------------------------------------
# Angular distance
# ---------------------------------------------------------
def angular_distance(a, b):
    """
    Angular distance between two xyz vectors in degrees.
    """
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)

    dot = np.clip(np.dot(a, b), -1.0, 1.0)

    return np.degrees(np.arccos(dot))


# ---------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------
def match_tracks(gt, pred):
    """
    gt:   (N_gt, 3)
    pred: (N_pred, 3)

    Returns:
        matched angular distances.
    """
    if len(gt) == 0 or len(pred) == 0:
        return np.array([])

    cost = np.zeros((len(gt), len(pred)), dtype=np.float32)

    for i in range(len(gt)):
        for j in range(len(pred)):
            cost[i, j] = angular_distance(gt[i], pred[j])

    row_ind, col_ind = linear_sum_assignment(cost)

    return cost[row_ind, col_ind]


# ---------------------------------------------------------
# Metric container
# ---------------------------------------------------------
def init_metrics(max_sources=3):
    return {
        # SED metrics
        "TP": 0,
        "FP": 0,
        "FN": 0,

        # DOA metrics
        "total_LE": 0.0,
        "total_DOA_matches": 0,

        # Frame-level localization recall proxy
        "frame_TP": 0,
        "frame_FN": 0,

        # ER
        "S": 0,
        "D": 0,
        "I": 0,
        "Nref": 0,

        # Predicted source count histogram:
        # index 0, 1, ..., max_sources
        "pred_count_hist": np.zeros(max_sources + 1, dtype=int),

        # Number of evaluated active GT frames
        "active_frames": 0,
    }


# ---------------------------------------------------------
# Update metrics for one frame
# ---------------------------------------------------------
def update_metrics(
    metrics,
    gt_active,
    pred_active,
    dists,
    doa_threshold,
    max_sources=3,
):
    n_gt = len(gt_active)
    n_pred = len(pred_active)

    # Ignore frames with no active ground-truth source.
    # This matches your old behavior.
    if n_gt == 0:
        return

    metrics["active_frames"] += 1

    # -----------------------------------------
    # Predicted source-count distribution
    # -----------------------------------------
    metrics["pred_count_hist"][min(n_pred, max_sources)] += 1

    # -------------------------------------------------
    # Frame recall/count correctness
    # -------------------------------------------------
    if n_gt == n_pred:
        metrics["frame_TP"] += 1
    else:
        metrics["frame_FN"] += 1

    # -------------------------------------------------
    # Error Rate statistics
    # -------------------------------------------------
    FP = max(0, n_pred - n_gt)
    FN = max(0, n_gt - n_pred)

    S = min(FN, FP)
    D = max(0, FN - FP)
    I = max(0, FP - FN)

    metrics["S"] += S
    metrics["D"] += D
    metrics["I"] += I
    metrics["Nref"] += n_gt

    # -------------------------------------------------
    # F-score statistics
    # -------------------------------------------------
    matched_correct = np.sum(dists <= doa_threshold)

    metrics["TP"] += matched_correct
    metrics["FP"] += max(0, n_pred - matched_correct)
    metrics["FN"] += max(0, n_gt - matched_correct)

    # -------------------------------------------------
    # Localization Error
    # Use all matched DOA distances, as your old code did.
    # -------------------------------------------------
    for d in dists:
        metrics["total_LE"] += d
        metrics["total_DOA_matches"] += 1


# ---------------------------------------------------------
# Compute SELD metrics
# ---------------------------------------------------------
def compute_seld(metrics):
    TP = metrics["TP"]
    FP = metrics["FP"]
    FN = metrics["FN"]

    S = metrics["S"]
    D = metrics["D"]
    I = metrics["I"]
    Nref = metrics["Nref"]

    total_LE = metrics["total_LE"]
    total_DOA_matches = metrics["total_DOA_matches"]

    frame_TP = metrics["frame_TP"]
    frame_FN = metrics["frame_FN"]

    if Nref == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # -------------------------------------------------
    # SED metrics
    # -------------------------------------------------
    ER = (S + D + I) / Nref
    F = (2 * TP) / (2 * TP + FP + FN + 1e-10)

    # -------------------------------------------------
    # DOA metrics
    # -------------------------------------------------
    LE = total_LE / (total_DOA_matches + 1e-10)
    LR = frame_TP / (frame_TP + frame_FN + 1e-10)

    # -------------------------------------------------
    # SELD score
    # -------------------------------------------------
    SED_score = (ER + (1 - F)) / 2
    DOA_score = ((LE / 180.0) + (1 - LR)) / 2
    SELD = (SED_score + DOA_score) / 2

    return ER, F, LE, LR, SELD


# ---------------------------------------------------------
# Print metrics
# ---------------------------------------------------------
def print_metrics(title, metrics):
    ER, F, LE, LR, SELD = compute_seld(metrics)

    print(f"\n===== {title} =====")
    print(f"Error Rate          (low) : {ER:.4f}")
    print(f"F-Score             (high): {F:.4f}")
    print(f"Localization Error  (low) : {LE:.2f} deg")
    print(f"Localization Recall (high): {LR:.4f}")
    print(f"SELD                (low) : {SELD:.4f}")
    print(f"Active GT frames          : {metrics['active_frames']}")

    hist = metrics["pred_count_hist"]
    total = np.sum(hist)

    print("\nPredicted Source Count Distribution:")

    if total == 0:
        print("No predictions recorded.")
    else:
        for k in range(len(hist)):
            label = "0 sources" if k == 0 else f"{k} source(s)"
            pct = 100.0 * hist[k] / total
            print(f"  Predicted {label}: {pct:.2f}%")


# ---------------------------------------------------------
# Convert tensors to framewise shape
# ---------------------------------------------------------
def ensure_framewise(arr, nb_tracks=None):
    """
    Accepts either:
        (tracks, 3)      old models
        (T, tracks, 3)   new CRNN

    Returns:
        (T, tracks, 3)
    """
    if arr.ndim == 2:
        # Old non-framewise output: one pseudo-frame
        arr = arr[None, :, :]

    if arr.ndim != 3:
        raise ValueError(
            f"Expected array shape (tracks,3) or (T,tracks,3), got {arr.shape}"
        )

    if arr.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3, got {arr.shape}")

    if nb_tracks is not None and arr.shape[1] != nb_tracks:
        raise ValueError(
            f"Expected {nb_tracks} tracks, got shape {arr.shape}"
        )

    return arr


# ---------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------
def evaluate_model(
    model,
    val_loader,
    accdoa_thresh=0.5,
    doa_threshold=20,
    device="cuda",
    verbose=False,
    print_predictions=False,
    max_sources=3,
):
    """
    Supports both old and new output shapes:

    Old:
        pred:   (B, tracks, 3)
        target: (B, tracks, 3)

    New CRNN:
        pred:   (B, T, tracks, 3)
        target: (B, T, tracks, 3)
    """

    model.to(device)
    model.eval()

    metrics_all = init_metrics(max_sources=max_sources)
    metrics_1 = init_metrics(max_sources=max_sources)
    metrics_2 = init_metrics(max_sources=max_sources)
    metrics_3 = init_metrics(max_sources=max_sources)

    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            pred = model(X_batch)

            if pred.shape[0] != Y_batch.shape[0]:
                raise ValueError(
                    f"Batch mismatch: pred batch={pred.shape[0]}, "
                    f"target batch={Y_batch.shape[0]}"
                )

            for i in range(pred.shape[0]):
                gt_sample = Y_batch[i].detach().cpu().numpy()
                pred_sample = pred[i].detach().cpu().numpy()

                gt_sample = ensure_framewise(gt_sample)
                pred_sample = ensure_framewise(pred_sample)

                if gt_sample.shape[0] != pred_sample.shape[0]:
                    raise ValueError(
                        f"Time-frame mismatch for sample {i}: "
                        f"gt T={gt_sample.shape[0]}, pred T={pred_sample.shape[0]}"
                    )

                T = gt_sample.shape[0]

                for t in range(T):
                    gt_frame = gt_sample[t]          # (tracks, 3)
                    pred_frame = pred_sample[t]      # (tracks, 3)

                    # -----------------------------------------
                    # Active GT tracks for this frame
                    # -----------------------------------------
                    gt_active = np.array([
                        v for v in gt_frame
                        if np.linalg.norm(v) > 0
                    ])

                    # -----------------------------------------
                    # Active predicted tracks for this frame
                    # -----------------------------------------
                    pred_active = np.array([
                        v for v in pred_frame
                        if np.linalg.norm(v) >= accdoa_thresh
                    ])

                    if print_predictions:
                        print("\n==============================")
                        print(f"Sample index in batch: {i}")
                        print(f"Frame index: {t}")

                        print("\nRaw Ground Truth Frame:")
                        print(gt_frame)

                        print("\nRaw Prediction Frame:")
                        print(pred_frame)

                        print("\nActive Ground Truth:")
                        print(gt_active)

                        print("\nActive Predictions:")
                        print(pred_active)

                    # -----------------------------------------
                    # Hungarian matching
                    # -----------------------------------------
                    dists = match_tracks(gt_active, pred_active)

                    if print_predictions:
                        print("\nMatched Angular Distances:")
                        print(dists)

                    # -----------------------------------------
                    # Update overall metrics
                    # -----------------------------------------
                    update_metrics(
                        metrics_all,
                        gt_active,
                        pred_active,
                        dists,
                        doa_threshold,
                        max_sources=max_sources,
                    )

                    # -----------------------------------------
                    # Metrics by number of GT sources
                    # -----------------------------------------
                    n_sources = len(gt_active)

                    if n_sources == 1:
                        update_metrics(
                            metrics_1,
                            gt_active,
                            pred_active,
                            dists,
                            doa_threshold,
                            max_sources=max_sources,
                        )

                    elif n_sources == 2:
                        update_metrics(
                            metrics_2,
                            gt_active,
                            pred_active,
                            dists,
                            doa_threshold,
                            max_sources=max_sources,
                        )

                    elif n_sources == 3:
                        update_metrics(
                            metrics_3,
                            gt_active,
                            pred_active,
                            dists,
                            doa_threshold,
                            max_sources=max_sources,
                        )

    if verbose:
        print_metrics("TOTAL", metrics_all)
        print_metrics("1 Active Source", metrics_1)
        print_metrics("2 Active Sources", metrics_2)
        print_metrics("3 Active Sources", metrics_3)

    return (
        compute_seld(metrics_all),
        compute_seld(metrics_1),
        compute_seld(metrics_2),
        compute_seld(metrics_3),
    )


# ---------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------
def find_best_threshold(
    model,
    loader,
    device,
    doa_threshold=20,
    thresholds=None,
    max_sources=3,
    name="dataset",
):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    best_thresh = None
    best_seld = float("inf")

    for thresh in thresholds:
        results_total, _, _, _ = evaluate_model(
            model,
            loader,
            accdoa_thresh=thresh,
            doa_threshold=doa_threshold,
            device=device,
            verbose=False,
            max_sources=max_sources,
        )

        _, _, _, _, SELD = results_total

        print(f"Threshold {thresh:.2f} -> SELD: {SELD:.4f}")

        if SELD < best_seld:
            best_seld = SELD
            best_thresh = thresh

    print(f"\nBest threshold ({name}): {best_thresh:.2f} | SELD: {best_seld:.4f}")

    return best_thresh


# ---------------------------------------------------------
# Model summary saving, optional
# ---------------------------------------------------------
def save_model_summary(
    model,
    seld_score,
    model_type,
    filename="model_results.txt",
    csv_filename="model_results.csv",
    input_size=(1, 50, 513, 73),
):
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "N/A")

    model_id = 1

    if os.path.exists(filename):
        with open(filename, "r") as f:
            content = f.read()

        ids = re.findall(r"Model ID:\s*(\d+)", content)
        if ids:
            model_id = max(map(int, ids)) + 1

    num_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    with open(filename, "a") as f:
        f.write("\n")
        f.write("=" * 150 + "\n")
        f.write(f"Model ID: {model_id:02d}\n")
        f.write("=" * 150 + "\n\n")

        f.write(f"SLURM Job ID: {slurm_job_id}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Trainable Parameters: {num_params:,}\n")
        f.write(f"SELD Score: {seld_score:.6f}\n\n")

        model_summary = summary(
            model,
            input_size=input_size,
            verbose=0,
        )

        f.write(str(model_summary))
        f.write("\n\n")

        f.write("Model Architecture:\n")
        f.write(str(model))
        f.write("\n\n")

    csv_exists = os.path.exists(csv_filename)

    with open(csv_filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not csv_exists:
            writer.writerow([
                "Model ID",
                "Model Type",
                "Num Parameters",
                "SELD Score",
            ])

        writer.writerow([
            f"{model_id:02d}",
            model_type,
            num_params,
            f"{seld_score:.6f}",
        ])