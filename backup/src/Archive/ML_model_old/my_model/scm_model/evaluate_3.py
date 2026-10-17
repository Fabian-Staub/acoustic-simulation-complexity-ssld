import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataset import HarpDataset
from mlp_model import SimpleSeldMLP


def angular_distance(a, b):
    """Angular distance between two xyz vectors (degrees)."""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)

    dot = np.clip(np.dot(a, b), -1, 1)
    return np.degrees(np.arccos(dot))


def match_tracks(gt, pred):
    """Hungarian matching between GT and predictions."""
    if len(gt) == 0 or len(pred) == 0:
        return []

    cost = np.zeros((len(gt), len(pred)))

    for i in range(len(gt)):
        for j in range(len(pred)):
            cost[i, j] = angular_distance(gt[i], pred[j])

    row, col = linear_sum_assignment(cost)
    return cost[row, col]


def init_metrics():
    return {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "total_LE": 0,
        "total_matches": 0,
        "total_ref": 0,
        "correct_n_sources": 0,
        "num_samples": 0
    }


def update_metrics(metrics, gt_active, pred_active, dists, doa_threshold):

    metrics["total_ref"] += len(gt_active)
    metrics["num_samples"] += 1

    if len(pred_active) == len(gt_active):
        metrics["correct_n_sources"] += 1

    for d in dists:
        metrics["total_LE"] += d
        metrics["total_matches"] += 1

        if d <= doa_threshold:
            metrics["TP"] += 1
        else:
            metrics["FP"] += 1

    if len(pred_active) > len(gt_active):
        metrics["FP"] += len(pred_active) - len(gt_active)

    if len(gt_active) > len(pred_active):
        metrics["FN"] += len(gt_active) - len(pred_active)


def compute_seld(metrics):

    TP = metrics["TP"]
    FP = metrics["FP"]
    FN = metrics["FN"]
    total_LE = metrics["total_LE"]
    total_matches = metrics["total_matches"]
    total_ref = metrics["total_ref"]
    correct_n_sources = metrics["correct_n_sources"]
    num_samples = metrics["num_samples"]

    ER = (FP + FN) / (total_ref + 1e-10)
    F = TP / (TP + 0.5 * (FP + FN) + 1e-10)
    LE = total_LE / (total_matches + 1e-10)
    LR = correct_n_sources / (num_samples + 1e-10)

    seld = np.mean([ER, 1 - F, LE / 180, 1 - LR])

    return ER, F, LE, LR, seld


def print_metrics(title, metrics):

    ER, F, LE, LR, seld = compute_seld(metrics)

    print(f"\n=== {title} ===")
    print(f"Error Rate          (low) : {ER:.4f}")
    print(f"F-Score             (high): {F:.4f}")
    print(f"Localization Error  (low) : {LE:.2f} deg")
    print(f"Localization Recall (high): {LR:.4f}")
    print(f"SELD                (low) : {seld:.4f}")


def evaluate_model(model, val_loader,
                   accdoa_thresh=0.5,
                   doa_threshold=20,
                   device='cuda'):

    model.to(device)
    model.eval()

    # overall metrics
    metrics_all = init_metrics()

    # metrics by number of active GT sources
    metrics_1 = init_metrics()
    metrics_2 = init_metrics()
    metrics_3 = init_metrics()

    with torch.no_grad():

        for X_batch, Y_batch in val_loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            pred = model(X_batch)   # (batch, tracks, 3)

            for i in range(pred.shape[0]):

                gt = Y_batch[i].cpu().numpy()
                pred_vec = pred[i].cpu().numpy()

                print("\n==============================")
                print(f"Sample index in batch: {i}")

                print("Raw Ground Truth:")
                print(gt)

                print("\nRaw Predictions:")
                print(pred_vec)

                # Active predictions
                pred_active = []

                for v in pred_vec:
                    if np.linalg.norm(v) >= accdoa_thresh:
                        pred_active.append(v)

                # Active GT
                gt_active = []

                for v in gt:
                    if np.linalg.norm(v) > 0:
                        gt_active.append(v)

                gt_active = np.array(gt_active)
                pred_active = np.array(pred_active)

                print("\nActive Ground Truth:")
                print(gt_active)

                print("\nActive Predictions:")
                print(pred_active)

                # Hungarian matching
                dists = match_tracks(gt_active, pred_active)

                print("\nMatched Angular Distances:")
                print(dists)

                # Update overall metrics
                update_metrics(
                    metrics_all,
                    gt_active,
                    pred_active,
                    dists,
                    doa_threshold
                )

                # Update metrics based on number of GT sources
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

    # Print results
    print_metrics("Evaluation Results TOTAL", metrics_all)
    print_metrics("Evaluation Results 1 active Source", metrics_1)
    print_metrics("Evaluation Results 2 active Sources", metrics_2)
    print_metrics("Evaluation Results 3 active Sources", metrics_3)

    return (
        compute_seld(metrics_all),
        compute_seld(metrics_1),
        compute_seld(metrics_2),
        compute_seld(metrics_3)
    )


#############################################
# Example Usage
#############################################
if __name__ == "__main__":
    
    data_path = "../../../data/arni_em32/sh_wavs/arni_dataset_2-6kHz_multiSCM_multiSource.pt"
    test_dataset = HarpDataset(data_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False
    )
    '''
    data_path = "../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_anisotropic_ISM/harp_dataset_2-6kHz_multiSCM_randRot_multiSource_merged.pt"
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

    model = SimpleSeldMLP(nb_tracks=3)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.load_state_dict(
        torch.load("mlp_trained.pth", map_location=device)
    )

    evaluate_model(
        model,
        test_loader,
        accdoa_thresh=0.5,
        doa_threshold=20,
        device=device
    )