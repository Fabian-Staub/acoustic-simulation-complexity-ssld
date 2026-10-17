import torch
from torchinfo import summary
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import HarpDataset
from evaluate_6 import evaluate_model, save_model_summary
import time
import resource
import sys
import numpy as np
import copy
import os
import re
import csv
from sklearn.model_selection import KFold

# Import the MLP and PITLoss
from mlp_model import MLP, PITLoss
from cnn_model import CNN, CNN2D

#############################################
# Training Function
#############################################
def train_model(train_loader, val_loader=None, epochs=10, lr=1e-3, 
                device='cuda', patience=10, min_delta=1e-4, nb_tracks=3):
    #model = MLP(nb_tracks=nb_tracks)  # Ensure model is initialized with correct nb_tracks
    model = CNN(in_channels=10, nb_tracks=nb_tracks)
    #model = SeldGRU(input_size=25, nb_tracks=3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = PITLoss(nb_tracks=nb_tracks)  # Use PITLoss for multi-track regression
    model.to(device)

    best_val_loss = float('inf')
    best_seld = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}", end='')

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, Y_val in val_loader:
                    X_val = X_val.to(device)
                    Y_val = Y_val.to(device)

                    pred_val = model(X_val)
                    loss_val = criterion(pred_val, Y_val)
                    val_loss += loss_val.item() * X_val.size(0)

            val_loss /= len(val_loader.dataset)

            # -----------------------------------------
            # Compute SELD metrics on validation set
            # -----------------------------------------
            total_metrics, _, _, _ = evaluate_model(
                model,
                val_loader,
                accdoa_thresh=0.5,
                doa_threshold=20,
                device=device,
                verbose=False
            )

            ER, F, LE, LR, SELD = total_metrics

            print(
                f", Val Loss: {val_loss:.6f}, "
                f"SELD: {SELD:.4f}, "
            )

            # -----------------------------------------
            # Early stopping on SELD score
            # lower is better
            # -----------------------------------------
            
            if SELD < best_seld - min_delta:
                best_seld = SELD
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
                #torch.save(best_model_state, "best_model.pth")
                print(f"New best SELD: {best_seld:.4f}")
            else:
                epochs_no_improve += 1
                print(f"EarlyStopping counter: {epochs_no_improve}/{patience}")
                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break
            '''
            # Early stopping on validation loss (lower is better)
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"New best Val Loss: {best_val_loss:.6f}")
            else:
                epochs_no_improve += 1
                print(f"EarlyStopping counter: {epochs_no_improve}/{patience}")
                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break
            '''
        else:
            print()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model



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

    print(f"Best threshold ({name}): {best_thresh:.2f} | SELD: {best_seld:.4f}")
    return best_thresh


def evaluate_and_store(
    model,
    loader,
    device,
    storage_dict,
    key_prefix,
    doa_threshold=20,
    verbose=True,
    name="dataset"):

    print(f"\n\n===== EVALUATION: {name} =========================")
    thresh = find_best_threshold(model, loader, device, doa_threshold, name)

    results_all, results_1, results_2, results_3 = evaluate_model(
        model,
        loader,
        accdoa_thresh=thresh,
        doa_threshold=doa_threshold,
        device=device,
        verbose=verbose
    )

    # store
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

#############################################
# Usage
#############################################
if __name__ == "__main__":

    start = time.time()

    data_path = "../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_source-constraints_smallGrid_adj-Room-Distr/harp_dataset_2-7kHz_10-SCM_randRotDegrees_multiSource.pt"
    test_datapath_arni = "../../../data/arni_em32/sh_wavs/arni_dataset_2-7kHz_10-SCM_multiSource.pt"
    test_datapath_summer = "../../../data/SummerAaltoMeasurements/processed_data/sh_wavs/SummerAaltoMeasurements_dataset_2-7kHz_10-SCM_multiSource.pt"

    full_dataset = HarpDataset(data_path)
    test_dataset_arni = HarpDataset(test_datapath_arni)
    test_dataset_summer = HarpDataset(test_datapath_summer)

    test_loader_arni = DataLoader(
        test_dataset_arni,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader_summer = DataLoader(
        test_dataset_summer,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # -------------------------
    # K-FOLD CONFIG
    # -------------------------
    k = 5
    kfold = KFold(n_splits=k, shuffle=False)

    test_metrics_all_folds = []
    test_metrics_1_folds   = []
    test_metrics_2_folds   = []
    test_metrics_3_folds   = []

    storage = {
        "arni_all": [], "arni_1": [], "arni_2": [], "arni_3": [],
        "summer_all": [], "summer_1": [], "summer_2": [], "summer_3": []
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):

        print("\n\n\n" + "=" * 100)
        print(f"FOLD {fold+1}/{k}")
        print("=" * 100)
        print("\n===== TRAINING =========================")

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=1024,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=1024,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        # -------------------------
        # Train model from scratch per fold
        # -------------------------
        model = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=200,
            lr=0.001,
            device=device,
            patience=10,
            min_delta=1e-3,
            nb_tracks=3
        )


        # ARNI
        evaluate_and_store(
            model=model,
            loader=test_loader_arni,
            device=device,
            storage_dict=storage,
            key_prefix="arni",
            name="ARNI"
        )

        # Summer
        evaluate_and_store(
            model=model,
            loader=test_loader_summer,
            device=device,
            storage_dict=storage,
            key_prefix="summer",
            name="SUMMER"
        )
    print()
    print_cv_report(
        storage["arni_all"],
        storage["arni_1"],
        storage["arni_2"],
        storage["arni_3"],
        "ARNI RESULTS"
    )

    print_cv_report(
        storage["summer_all"],
        storage["summer_1"],
        storage["summer_2"],
        storage["summer_3"],
        "SUMMER RESULTS"
    )



    # Performance summary
    print(file=sys.stderr)
    rusage_s = resource.getrusage(resource.RUSAGE_SELF)
    rusage_c = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(f'Walltime {time.time() - start:.2f} s', file=sys.stderr)
    print(f'User time: {rusage_s.ru_utime + rusage_c.ru_utime:.2f} s', file=sys.stderr)
    print(f'System time: {rusage_s.ru_stime + rusage_c.ru_stime:.2f} s', file=sys.stderr)
    print(f'MaxRSS: {(rusage_s.ru_maxrss + rusage_c.ru_maxrss)/2**20:.3f} GiB', file=sys.stderr)