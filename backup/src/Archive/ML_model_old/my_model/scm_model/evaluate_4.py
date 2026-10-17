import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.optimize import linear_sum_assignment

from dataset import HarpDataset
from mlp_model import SimpleSeldMLP
from cnn_model import SimpleSeldCNN, ImprovedSeldCNN, SimpleSeldCNN2D, ImprovedSeldCNN2D


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

    if len(gt) == 0 or len(pred) == 0:
        return np.array([])

    cost = np.zeros((len(gt), len(pred)))

    for i in range(len(gt)):
        for j in range(len(pred)):
            cost[i, j] = angular_distance(gt[i], pred[j])

    row_ind, col_ind = linear_sum_assignment(cost)

    return cost[row_ind, col_ind]


# ---------------------------------------------------------
# Metric container
# ---------------------------------------------------------
def init_metrics():

    return {
        # SED metrics
        "TP": 0,
        "FP": 0,
        "FN": 0,

        # DOA metrics
        "total_LE": 0.0,
        "total_DOA_matches": 0,

        # Frame recall
        "frame_TP": 0,
        "frame_FN": 0,

        # ER
        "S": 0,
        "D": 0,
        "I": 0,
        "Nref": 0,
    }


# ---------------------------------------------------------
# Update metrics
# ---------------------------------------------------------
def update_metrics(metrics,
                   gt_active,
                   pred_active,
                   dists,
                   doa_threshold):

    n_gt = len(gt_active)
    n_pred = len(pred_active)

    # -------------------------------------------------
    # Frame recall
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

    # -------------------------------------------------
    # SED metrics
    # -------------------------------------------------

    ER = (S + D + I) / Nref

    F = (2 * TP) / (2 * TP + FP + FN + 1e-10)

    # -------------------------------------------------
    # DOA metrics
    # -------------------------------------------------

    LE = total_LE / (total_DOA_matches + 1e-10)

    LR = frame_TP / (frame_TP + frame_FN)

    # -------------------------------------------------
    # SELD score (paper definition)
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

    print(f"\n=== {title} ===")
    print(f"Error Rate          (low) : {ER:.4f}")
    print(f"F-Score             (high): {F:.4f}")
    print(f"Localization Error  (low) : {LE:.2f} deg")
    print(f"Localization Recall (high): {LR:.4f}")
    print(f"SELD                (low) : {SELD:.4f}")


# ---------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------
def evaluate_model(model,
                   val_loader,
                   accdoa_thresh=0.5,
                   doa_threshold=20,
                   device='cuda',
                   verbose=False):

    model.to(device)
    model.eval()

    metrics_all = init_metrics()
    metrics_1 = init_metrics()
    metrics_2 = init_metrics()
    metrics_3 = init_metrics()

    with torch.no_grad():

        for X_batch, Y_batch in val_loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            pred = model(X_batch)

            for i in range(pred.shape[0]):

                gt = Y_batch[i].cpu().numpy()
                pred_vec = pred[i].cpu().numpy()

                # -----------------------------------------
                # Active GT
                # -----------------------------------------
                gt_active = np.array([
                    v for v in gt
                    if np.linalg.norm(v) > 0
                ])

                # -----------------------------------------
                # Active predictions
                # -----------------------------------------
                pred_active = np.array([
                    v for v in pred_vec
                    if np.linalg.norm(v) >= accdoa_thresh
                ])

                # -----------------------------------------
                # Debug prints
                # -----------------------------------------
                if verbose:
                    print("\n==============================")
                    print(f"Sample index in batch: {i}")

                    print("Raw Ground Truth:")
                    print(gt)

                    print("\nRaw Predictions:")
                    print(pred_vec)

                    print("\nActive Ground Truth:")
                    print(gt_active)

                    print("\nActive Predictions:")
                    print(pred_active)

                # -----------------------------------------
                # Hungarian matching
                # -----------------------------------------
                dists = match_tracks(
                    gt_active,
                    pred_active
                )

                if verbose:
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
                    doa_threshold
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
                        doa_threshold
                    )

                elif n_sources == 2:

                    update_metrics(
                        metrics_2,
                        gt_active,
                        pred_active,
                        dists,
                        doa_threshold
                    )

                elif n_sources == 3:

                    update_metrics(
                        metrics_3,
                        gt_active,
                        pred_active,
                        dists,
                        doa_threshold
                    )

    if verbose:
        print_metrics(
            "Evaluation Results TOTAL",
            metrics_all
        )

        print_metrics(
            "Evaluation Results 1 Active Source",
            metrics_1
        )

        print_metrics(
            "Evaluation Results 2 Active Sources",
            metrics_2
        )

        print_metrics(
            "Evaluation Results 3 Active Sources",
            metrics_3
        )

    return (
        compute_seld(metrics_all),
        compute_seld(metrics_1),
        compute_seld(metrics_2),
        compute_seld(metrics_3)
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":

    
    data_path = "../../../data/arni_em32/sh_wavs/arni_dataset_2-6kHz_8-SCM_multiSource.pt"
    test_dataset = HarpDataset(data_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False
    )
    '''
    data_path = "../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_anisotropic_ISM/harp_dataset_2-6kHz_8-SCM_randRot_multiSource.pt"
    dataset = HarpDataset(data_path)
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train
    train_indices = list(range(n_train))
    val_indices   = list(range(n_train, n_total))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_loader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )'''

    #model = SimpleSeldMLP(nb_tracks=3)
    model = ImprovedSeldCNN()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.load_state_dict(
        torch.load(
            "cnn_trained.pth",
            map_location=device
        )
    )


    # ---------------------------------------------------------
    # Sweep ACCDOA thresholds
    # ---------------------------------------------------------
    best_thresh = None
    best_seld = float("inf")
    best_results = None

    thresholds = np.arange(0.05, 0.96, 0.05)

    for thresh in thresholds:

        results_total, results_1, results_2, results_3 = evaluate_model(
            model,
            test_loader,
            accdoa_thresh=thresh,
            doa_threshold=20,
            device=device,
            verbose=False
        )

        ER, F, LE, LR, SELD = results_total

        print(f"Threshold {thresh:.2f} -> SELD: {SELD:.4f}")

        if SELD < best_seld:
            best_seld = SELD
            best_thresh = thresh

    # ---------------------------------------------------------
    # Print best results only
    # ---------------------------------------------------------
    print("\n======================================")
    print(f"BEST ACCDOA THRESHOLD: {best_thresh:.2f}")
    print(f"BEST TOTAL SELD SCORE: {best_seld:.4f}")
    print("======================================")

    evaluate_model(
        model,
        test_loader,
        accdoa_thresh=best_thresh,
        doa_threshold=20,
        device=device,
        verbose=True
    )