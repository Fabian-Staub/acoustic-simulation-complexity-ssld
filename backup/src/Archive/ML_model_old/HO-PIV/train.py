import os
import sys
import time
import copy
import resource
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from crnn import PoschadelCRNN, PITLoss

try:
    from evaluate import evaluate_model
except Exception:
    evaluate_model = None


class HopivPTDataset(Dataset):
    """
    Loads new per-sample .pt files:

        {
            "X": Tensor, shape (T, F, C), usually (50, 513, C)
            "Y": Tensor, shape (T, S, 3)
        }

    Returns:
        X: (T, F, C)
        Y: (T, nb_tracks, 3)
    """
    def __init__(self, data_path, nb_tracks=3):
        self.data_path = Path(data_path)
        self.nb_tracks = nb_tracks

        if self.data_path.is_file():
            obj = torch.load(self.data_path, map_location="cpu")

            if isinstance(obj, (list, tuple)):
                self.samples = obj
                self.files = None
            elif isinstance(obj, dict) and "X" in obj and "Y" in obj:
                self.samples = [obj]
                self.files = None
            else:
                raise TypeError(
                    "Single .pt file must be either a sample dict with keys "
                    "'X' and 'Y', or a list/tuple of such sample dicts."
                )

        else:
            self.files = sorted(self.data_path.rglob("*.pt"))
            self.samples = None

            if not self.files:
                raise FileNotFoundError(f"No .pt files found in {self.data_path}")

    def __len__(self):
        if self.samples is not None:
            return len(self.samples)
        return len(self.files)

    def _fix_y_tracks(self, y):
        if y.ndim != 3:
            raise ValueError(f"Expected Y shape (T,S,3), got {tuple(y.shape)}")

        if y.shape[-1] != 3:
            raise ValueError(f"Expected Y last dimension to be 3, got {tuple(y.shape)}")

        current_tracks = y.shape[1]

        if current_tracks == self.nb_tracks:
            return y

        if current_tracks > self.nb_tracks:
            return y[:, :self.nb_tracks, :]

        pad = torch.zeros(
            y.shape[0],
            self.nb_tracks - current_tracks,
            3,
            dtype=y.dtype,
        )

        return torch.cat([y, pad], dim=1)

    def __getitem__(self, idx):
        if self.samples is not None:
            sample = self.samples[idx]
        else:
            sample = torch.load(self.files[idx], map_location="cpu")

        if not isinstance(sample, dict):
            raise TypeError("Expected each sample to be a dict with keys 'X' and 'Y'.")

        x = sample["X"].float()
        y = sample["Y"].float()

        if x.ndim != 3:
            raise ValueError(f"Expected X shape (T,F,C), got {tuple(x.shape)}")

        y = self._fix_y_tracks(y)

        return x, y


def train_model(
    train_loader,
    val_loader=None,
    epochs=200,
    lr=1e-3,
    device="cuda",
    patience=20,
    min_delta=1e-4,
    nb_tracks=3,
    weight_decay=3e-4,
):
    model = PoschadelCRNN(nb_tracks=nb_tracks, dropout=0.2).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.9,
        patience=5,
    )

    criterion = PITLoss(nb_tracks=nb_tracks)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred = model(X_batch)
            loss = criterion(pred, Y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        if val_loader is None:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.6f}"
            )
            continue

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val = X_val.to(device, non_blocking=True)
                Y_val = Y_val.to(device, non_blocking=True)

                pred_val = model(X_val)
                loss_val = criterion(pred_val, Y_val)

                val_loss += loss_val.item() * X_val.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6e}"
        )

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

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


if __name__ == "__main__":
    # =========================================================
    # CONFIG: edit these directly
    # =========================================================
    data_path = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM/hopiv_timit_1s_random_window_pt_debug"

    nb_tracks = 3
    epochs = 200
    batch_size = 32
    lr = 1e-3
    weight_decay = 3e-4
    patience = 20
    min_delta = 1e-4
    val_fraction = 0.2
    num_workers = 4
    seed = 1337

    run_evaluation = True
    accdoa_thresh = 0.5
    doa_threshold = 20
    # =========================================================

    start = time.time()

    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    full_dataset = HopivPTDataset(
        data_path=data_path,
        nb_tracks=nb_tracks,
    )

    n_total = len(full_dataset)
    n_val = int(round(val_fraction * n_total))
    n_train = n_total - n_val

    if n_train <= 0 or n_val <= 0:
        raise ValueError(
            f"Bad split: n_total={n_total}, n_train={n_train}, n_val={n_val}"
        )

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    print(f"Dataset size: {n_total}")
    print(f"Train size:   {n_train}")
    print(f"Val size:     {n_val}")

    model = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        patience=patience,
        min_delta=min_delta,
        nb_tracks=nb_tracks,
        weight_decay=weight_decay,
    )

    if run_evaluation and evaluate_model is not None:
        try:
            print("\n===== VALIDATION EVALUATION =====")
            total_metrics, results_1, results_2, results_3 = evaluate_model(
                model,
                val_loader,
                accdoa_thresh=accdoa_thresh,
                doa_threshold=doa_threshold,
                device=device,
                verbose=True,
            )
            print(f"Total metrics: {total_metrics}")
            print(f"Results for track 1: {results_1}")
            print(f"Results for track 2: {results_2}")
            print(f"Results for track 3: {results_3}")
        except Exception as exc:
            print(
                "\nSkipped evaluate_model because it failed with the new "
                f"time-distributed output shape: {exc}"
            )

    rusage_s = resource.getrusage(resource.RUSAGE_SELF)
    rusage_c = resource.getrusage(resource.RUSAGE_CHILDREN)

    print(file=sys.stderr)
    print(f"Walltime {time.time() - start:.2f} s", file=sys.stderr)
    print(
        f"User time: {rusage_s.ru_utime + rusage_c.ru_utime:.2f} s",
        file=sys.stderr,
    )
    print(
        f"System time: {rusage_s.ru_stime + rusage_c.ru_stime:.2f} s",
        file=sys.stderr,
    )
    print(
        f"MaxRSS: {(rusage_s.ru_maxrss + rusage_c.ru_maxrss) / 2**20:.3f} GiB",
        file=sys.stderr,
    )