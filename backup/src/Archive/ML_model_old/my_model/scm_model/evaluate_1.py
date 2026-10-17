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


def evaluate_model(model, val_loader, accdoa_thresh=0.5, doa_threshold=20, device='cuda'):

    model.to(device)
    model.eval()

    TP = 0
    FP = 0
    FN = 0
    total_LE = 0
    total_matches = 0
    total_ref = 0

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

                # remove inactive predictions
                pred_active = []
                for v in pred_vec:
                    if np.linalg.norm(v) >= accdoa_thresh:
                        pred_active.append(v)

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

                total_ref += len(gt_active)

                # match
                dists = match_tracks(gt_active, pred_active)

                print("\nMatched Angular Distances:")
                print(dists)

                for d in dists:
                    total_LE += d
                    total_matches += 1

                    if d <= doa_threshold:
                        TP += 1
                    else:
                        FP += 1

                if len(pred_active) > len(gt_active):
                    FP += len(pred_active) - len(gt_active)

                if len(gt_active) > len(pred_active):
                    FN += len(gt_active) - len(pred_active)

    ER = (FP + FN) / (total_ref + 1e-10)
    F = TP / (TP + 0.5*(FP + FN) + 1e-10)
    LE = total_LE / (total_matches + 1e-10)
    LR = total_matches / (total_ref + 1e-10)

    seld = np.mean([ER, 1-F, LE/180, 1-LR])

    print("\n=== Evaluation Results ===")
    print(f"Error Rate          (low) : {ER:.4f}")
    print(f"F-Score             (high): {F:.4f}")
    print(f"Localization Error  (low) : {LE:.2f} deg")
    print(f"Localization Recall (high): {LR:.4f}")
    print(f"SELD                (low) : {seld:.4f}")

    return ER, F, LE, LR, seld


    #############################################
# Example Usage
#############################################
if __name__ == "__main__":
    data_path = "../../../data/arni_em32/sh_wavs/arni_dataset_scm.pt"  # .pt Datei mit {'X':..., 'Y':...}

    test_dataset = HarpDataset(data_path)
    test_loader   = DataLoader(test_dataset, batch_size=64, shuffle=False)   # Val nicht shuffeln

    model = SimpleSeldMLP(nb_tracks=3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load("mlp_trained.pth", map_location=device))

    # Evaluate sample by sample
    evaluate_model(model, test_loader, accdoa_thresh=0.5, doa_threshold=20, device=device)