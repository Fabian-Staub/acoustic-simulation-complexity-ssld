import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import argparse

from dataset import HarpDataset
from cnn_model import CNN
from rnn_model import CNN_RNN
from evaluate_6 import evaluate_model


def find_best_threshold(model, loader, device, doa_threshold=20, name="dataset"):
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
            verbose=False
        )

        _, _, _, _, SELD = results_total

        if SELD < best_seld:
            best_seld = SELD
            best_thresh = thresh
    return round(best_thresh, 2)

#############################################
# EVALUATION HELPERS (same style as training script)
#############################################
def evaluate_and_store(
    model,
    loader,
    device,
    storage_dict,
    key_prefix,
    find_best_accdoa_thresh=False,
    doa_threshold=20,
    verbose=True,
    name="dataset"):

    if find_best_accdoa_thresh:
        accdoa_threshold = find_best_threshold(model=model, loader=loader, device=device, doa_threshold=doa_threshold, name=name)
    else: 
        accdoa_threshold = 0.5

    print(f"\n\n===== EVALUATION: {name} =========================")
    print(f"AccDOA Threshold: {accdoa_threshold}, DOA Threshold: {doa_threshold} deg")
    results_all, results_1, results_2, results_3 = evaluate_model(
        model,
        loader,
        accdoa_thresh=accdoa_threshold,
        doa_threshold=doa_threshold,
        device=device,
        verbose=verbose
    )

    storage_dict[f"{key_prefix}_all"].append(results_all)
    storage_dict[f"{key_prefix}_1"].append(results_1)
    storage_dict[f"{key_prefix}_2"].append(results_2)
    storage_dict[f"{key_prefix}_3"].append(results_3)

    return results_all, results_1, results_2, results_3


def print_cv_report(all_folds, one_folds, two_folds, three_folds, title="CROSS VALIDATION RESULTS"):
    def mean(x):
        return np.mean(np.stack(x, axis=0), axis=0)

    mean_all = mean(all_folds)
    mean_1   = mean(one_folds)
    mean_2   = mean(two_folds)
    mean_3   = mean(three_folds)

    print("\n" + "*" * 100)
    print(title)
    print("*" * 100)

    def print_block(name, vals):
        ER, F, LE, LR, SELD = vals
        print(f"\n--- {name} ---")
        print(f"Error Rate          (low) : {ER:.4f}")
        print(f"F-Score             (high): {F:.4f}")
        print(f"Localization Error  (low) : {LE:.2f} deg")
        print(f"Localization Recall (high): {LR:.4f}")
        print(f"SELD                (low) : {SELD:.4f}")

    print_block("TOTAL", mean_all)
    print_block("1 SOURCE", mean_1)
    print_block("2 SOURCES", mean_2)
    print_block("3 SOURCES", mean_3)


def save_storage_to_csv(storage, filename="cv_results.csv"):
    rows = []

    def add_rows(dataset_name, key_prefix):
        for fold_idx in range(len(storage[f"{key_prefix}_all"])):
            all_vals   = storage[f"{key_prefix}_all"][fold_idx]
            one_vals   = storage[f"{key_prefix}_1"][fold_idx]
            two_vals   = storage[f"{key_prefix}_2"][fold_idx]
            three_vals = storage[f"{key_prefix}_3"][fold_idx]

            def pack(split_name, vals):
                ER, F, LE, LR, SELD = vals
                rows.append({
                    "dataset": dataset_name,
                    "fold": fold_idx + 1,
                    "split": split_name,
                    "ER": ER,
                    "F": F,
                    "LE": LE,
                    "LR": LR,
                    "SELD": SELD
                })

            pack("all", all_vals)
            pack("1_source", one_vals)
            pack("2_sources", two_vals)
            pack("3_sources", three_vals)

    add_rows("VAL", "val")
    add_rows("ARNI", "arni")
    add_rows("IVAN", "ivan")

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"\nSaved CSV to: {filename}")


#############################################
# MAIN
#############################################
if __name__ == "__main__":

    #############################################
    # ARGUMENTS
    #############################################
    parser = argparse.ArgumentParser(
        description="Evaluate trained CNN fold models on VAL, ARNI, and IVAN datasets."
    )

    parser.add_argument(
        "--val_path",
        type=str,
        required=True,
        help="Path to validation HarpDataset .pt file."
    )

    parser.add_argument(
        "--arni_path",
        type=str,
        required=True,
        help="Path to ARNI HarpDataset .pt file."
    )

    parser.add_argument(
        "--ivan_path",
        type=str,
        required=True,
        help="Path to IVAN HarpDataset .pt file."
    )

    parser.add_argument(
        "--find_best_accdoa_thresh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to find the best AccDOA threshold separately for each evaluation dataset."
    )

    args = parser.parse_args()

    #############################################
    # CONFIG
    #############################################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nb_tracks = 3
    in_channels = 10
    num_folds = 5

    model_dir = "trained_models"
    job_name = os.environ.get("SLURM_JOB_NAME", "local")

    # Remove optional evaluation suffix from job name when loading trained models
    model_job_name = job_name.removesuffix("_best-thresh")

    val_path  = args.val_path
    arni_path = args.arni_path
    ivan_path = args.ivan_path

    #############################################
    # DATA
    #############################################
    arni_dataset = HarpDataset(arni_path)
    ivan_dataset = HarpDataset(ivan_path)
    val_dataset  = HarpDataset(val_path)

    # Split validation dataset into 5 folds
    kfold = KFold(n_splits=num_folds, shuffle=False)
    val_fold_indices = [val_idx for _, val_idx in kfold.split(val_dataset)]

    arni_loader = DataLoader(
        arni_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    ivan_loader = DataLoader(
        ivan_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    #############################################
    # STORAGE
    #############################################
    storage = {
        "val_all": [],  "val_1": [],  "val_2": [],  "val_3": [],
        "arni_all": [], "arni_1": [], "arni_2": [], "arni_3": [],
        "ivan_all": [], "ivan_1": [], "ivan_2": [], "ivan_3": []
    }

    #############################################
    # MAIN LOOP (LOAD EACH FOLD MODEL)
    #############################################
    for fold in range(1, 2):

        print("\n\n" + "=" * 100)
        print(f"FOLD {fold}/{num_folds}")
        print("=" * 100)

        model_path = os.path.join(model_dir, f"model_{model_job_name}_fold_{fold}.pth")

        #model = CNN(in_channels=in_channels, nb_tracks=nb_tracks)
        model = CNN_RNN(nb_tracks=3, in_channels=10, rnn_hidden_size=256, rnn_layers=2, bidirectional=True, rnn_type="gru")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Corresponding validation fold
        val_fold_dataset = Subset(val_dataset, val_fold_indices[fold - 1])

        val_loader = DataLoader(
            val_fold_dataset,
            batch_size=1024,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        # VAL corresponding fold
        evaluate_and_store(
            model=model,
            find_best_accdoa_thresh=args.find_best_accdoa_thresh,
            loader=val_loader,
            device=device,
            storage_dict=storage,
            key_prefix="val",
            name=f"VAL fold {fold}"
        )

        # ARNI
        evaluate_and_store(
            model=model,
            find_best_accdoa_thresh=args.find_best_accdoa_thresh,
            loader=arni_loader,
            device=device,
            storage_dict=storage,
            key_prefix="arni",
            name="ARNI"
        )

        # IVAN
        evaluate_and_store(
            model=model,
            find_best_accdoa_thresh=args.find_best_accdoa_thresh,
            loader=ivan_loader,
            device=device,
            storage_dict=storage,
            key_prefix="ivan",
            name="IVAN"
        )

    #############################################
    # FINAL REPORTS
    #############################################
    print_cv_report(
        storage["val_all"],
        storage["val_1"],
        storage["val_2"],
        storage["val_3"],
        "VALIDATION RESULTS"
    )

    print_cv_report(
        storage["arni_all"],
        storage["arni_1"],
        storage["arni_2"],
        storage["arni_3"],
        "ARNI RESULTS"
    )

    print_cv_report(
        storage["ivan_all"],
        storage["ivan_1"],
        storage["ivan_2"],
        storage["ivan_3"],
        "IVAN RESULTS"
    )

    save_storage_to_csv(storage, filename=f"eval_models/eval_model_{job_name}.csv")