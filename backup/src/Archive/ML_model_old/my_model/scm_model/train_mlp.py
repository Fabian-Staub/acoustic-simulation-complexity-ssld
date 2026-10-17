import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import HarpDataset
from eval_mlp import evaluate_model
import time
import resource
import sys

# Import the MLP and PITLoss
from mlp_model import SimpleSeldMLP

#############################################
# Training Function
#############################################
def train_model(model, train_loader, val_loader=None, epochs=10, lr=1e-3, 
                device='cuda', patience=10, min_delta=1e-4, nb_tracks=1):
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')  # MSE for regression
    model.to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

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
            print(f", Val Loss: {val_loss:.6f}")

            # Early Stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
                torch.save(best_model_state, "best_model.pth")
            else:
                epochs_no_improve += 1
                print(f"EarlyStopping counter: {epochs_no_improve}/{patience}")

                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break
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
    data_path = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM/harp_dataset_scm.pt"

    dataset = HarpDataset(data_path)
    n_total = len(dataset)
    n_train = int(n_total * 0.16)
    n_val = n_total - n_train

    train_indices = list(range(n_train))
    val_indices   = list(range(n_train, n_total))

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset   = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)


    model = SimpleSeldMLP(in_feat_shape=(7,25,25))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        lr=0.001,
        device=device,
        patience=20,
        nb_tracks=1
    )

    torch.save(model.state_dict(), "mlp_trained.pth")

    evaluate_model(model, val_loader, accdoa_thresh=0.5, doa_threshold=20, device=device)

    # Performance summary
    print(file=sys.stderr)
    rusage_s = resource.getrusage(resource.RUSAGE_SELF)
    rusage_c = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(f'Walltime {time.time() - start:.2f} s', file=sys.stderr)
    print(f'User time: {rusage_s.ru_utime + rusage_c.ru_utime:.2f} s', file=sys.stderr)
    print(f'System time: {rusage_s.ru_stime + rusage_c.ru_stime:.2f} s', file=sys.stderr)
    print(f'MaxRSS: {(rusage_s.ru_maxrss + rusage_c.ru_maxrss)/2**20:.3f} GiB', file=sys.stderr)