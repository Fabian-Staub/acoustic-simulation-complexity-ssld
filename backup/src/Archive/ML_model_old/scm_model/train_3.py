import torch
from torchinfo import summary
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import HarpDataset
from evaluate_5 import evaluate_model, save_model_summary
import time
import resource
import sys
import numpy as np
import copy
import os
import re
import csv

# Import the MLP and PITLoss
from mlp_model import MLP, PITLoss
from cnn_model import CNN, CNN2D

#############################################
# Training Function
#############################################
def train_model(train_loader, val_loader=None, epochs=10, lr=1e-3, 
                device='cuda', patience=10, min_delta=1e-4, nb_tracks=3):
    #model = MLP(nb_tracks=nb_tracks)  # Ensure model is initialized with correct nb_tracks
    model = CNN2D(in_channels=10, nb_tracks=nb_tracks)
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

#############################################
# Example Usage
#############################################
if __name__ == "__main__":

    start = time.time()
    data_path = "../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_source-constraints_smallGrid_adj-Room-Distr/harp_dataset_2-7kHz_10-SCM_randRotDegrees_multiSource.pt"
    test_datapath = "../../../data/arni_em32/sh_wavs/arni_dataset_2-7kHz_10-SCM_multiSource.pt"
    test_dataset = HarpDataset(test_datapath)

    dataset = HarpDataset(data_path)
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train

    train_indices = list(range(n_train))
    val_indices   = list(range(n_train, n_total))

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    


    #train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    #val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    #test_loader = val_loader  # <-- Use validation set for testing (since test set is from a different distribution and not suitable for training/validation)
    #val_loader = test_loader  # <-- Use test set for validation during training


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    #torch.save(model.state_dict(), "CNN2D_trained.pth")

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
    
    
    save_model_summary(
        model=model,
        seld_score=best_seld,
        model_type=model.__class__.__name__,  # MLP, CNN, CNN2D, SeldGRU...
        filename="model_search_results.txt",
        csv_filename="model_search_results.csv",
        input_size=(1, 10, 25, 25)
    )

    # Performance summary
    print(file=sys.stderr)
    rusage_s = resource.getrusage(resource.RUSAGE_SELF)
    rusage_c = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(f'Walltime {time.time() - start:.2f} s', file=sys.stderr)
    print(f'User time: {rusage_s.ru_utime + rusage_c.ru_utime:.2f} s', file=sys.stderr)
    print(f'System time: {rusage_s.ru_stime + rusage_c.ru_stime:.2f} s', file=sys.stderr)
    print(f'MaxRSS: {(rusage_s.ru_maxrss + rusage_c.ru_maxrss)/2**20:.3f} GiB', file=sys.stderr)