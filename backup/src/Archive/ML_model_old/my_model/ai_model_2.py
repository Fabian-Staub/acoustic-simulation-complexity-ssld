import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import itertools


# -----------------------------
# 1️⃣ Model definition
# -----------------------------
class SELD_CNN(nn.Module):
    def __init__(self, input_channels=7, max_sources=3):
        super(SELD_CNN, self).__init__()
        self.max_sources = max_sources
        
        # -----------------------
        # CNN feature extractor
        # -----------------------
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2,2))
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2,2))
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2,2))
        self.drop3 = nn.Dropout(0.2)

        # Fully connected layer for ACCDOA regression
        self.fc = nn.Linear(128*3*3, max_sources * 3)
        
    def forward(self, x):
        # x: (batch, 7, 25, 25)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        b, c, h, w = x.shape
        x = x.view(b, c*h*w)  # flatten
        out = self.fc(x)      # (batch, max_sources*3)
        out = out.view(b, self.max_sources, 3)
        return out

# -----------------------------
# Load dataset and extract numeric targets
# -----------------------------
dataset_path = "../data/harp/Generated_HOA_IRs_ISM/harp_dataset.pt"
data = torch.load(dataset_path, weights_only=False)

X = data["X"].float()  # (batch, 7, 25, 25)
Y = data["Y"]          # object array (batch, max_sources, 17)

# Convert only first 4 columns (ACCDOA xyz + mask) to float
Y_accdoa_list = []
Y_mask_list = []

for y in Y:
    # y: (max_sources, 17) as object
    y_float = np.zeros((y.shape[0], 4), dtype=np.float32)
    y_float[:, :4] = y[:, :4].astype(np.float32)  # only first 4 columns
    Y_accdoa_list.append(y_float[:, :3])  # xyz
    Y_mask_list.append(y_float[:, 3])     # mask

# Stack to create tensors
Y_accdoa = torch.tensor(np.stack(Y_accdoa_list, axis=0), dtype=torch.float32)
Y_mask   = torch.tensor(np.stack(Y_mask_list, axis=0), dtype=torch.float32)

print("X:", X.shape)
print("Y_accdoa:", Y_accdoa.shape)
print("Y_mask:", Y_mask.shape)


# -----------------------------
# 3️⃣ Masked MSE loss
# -----------------------------

def pit_mse_loss(pred, target, mask=None, eps=1e-8, alpha=0.01):
    """
    Stable single-class ACCDOA PIT loss with optional external mask.
    
    pred:   (B, M, 3)       -> predicted ACCDOA xyz
    target: (B, M, 3)       -> ground truth ACCDOA xyz
    mask:   (B, M) or None  -> optional activity mask (1=active, 0=inactive)
    eps:    small number to avoid divide by zero
    alpha:  weight for activity regularization
    """
    B, M, _ = pred.shape
    perms = list(itertools.permutations(range(M)))  # all permutations
    total_loss = 0.0

    for b in range(B):
        pred_b = pred[b]      # (M,3)
        target_b = target[b]  # (M,3)
        
        # Use external mask if provided, else compute from target
        if mask is not None:
            mask_b = mask[b]
        else:
            mask_b = (target_b.norm(dim=-1) > 0).float()  # (M,)

        perm_losses = []

        for perm in perms:
            perm = torch.tensor(perm, device=pred.device)
            target_perm = target_b[perm]
            mask_perm = mask_b[perm]

            # ✅ Masked, stable MSE for active sources
            active_idx = mask_perm > 0
            if active_idx.sum() > 0:
                se = ((pred_b[active_idx] - target_perm[active_idx])**2).sum(dim=-1)
                target_norm_sq = (target_perm[active_idx]**2).sum(dim=-1)
                scaled_se = se / (target_norm_sq + eps)
                mse_loss = scaled_se.mean()
            else:
                mse_loss = torch.tensor(0.0, device=pred.device)

            # ✅ Activity regularization (avoid divide by zero)
            pred_norm = pred_b.norm(dim=-1)
            activity_loss = ((mask_perm * 1.0) / (pred_norm + 1e-3)).mean()

            perm_loss = mse_loss + alpha * activity_loss
            perm_losses.append(perm_loss)

        perm_losses = torch.stack(perm_losses)  # (num_permutations,)
        total_loss += perm_losses.min()          # pick minimum over permutations

    return total_loss / B



# -----------------------------
# 4️⃣ DataLoader
# -----------------------------
batch_size = 1
dataset = TensorDataset(X, Y_accdoa, Y_mask)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -----------------------------
# 5️⃣ Initialize model and optimizer
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SELD_CNN(input_channels=X.shape[1], max_sources=Y_accdoa.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)


# -----------------------------
# 6️⃣ Training loop
# -----------------------------
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_Y, batch_mask in loader:
        batch_X, batch_Y, batch_mask = batch_X.to(device), batch_Y.to(device), batch_mask.to(device)

        optimizer.zero_grad()
        pred_Y = model(batch_X)
        loss = pit_mse_loss(pred_Y, batch_Y, batch_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(loader):.6f}")

# -----------------------------
# 7️⃣ Save trained model
# -----------------------------
torch.save(model.state_dict(), "seld_accdoa_cnn.pth")
print("Model saved as seld_accdoa_cnn.pth")

# -----------------------------
# 8️⃣ Inference example
# -----------------------------
model.eval()
with torch.no_grad():
    pred_Y = model(X.to(device))
    activity = pred_Y.norm(dim=-1) > 0.8  # threshold predicted activity
    print("Predicted ACCDOA shape:", pred_Y.shape)
    print("Predicted ACCDOA shape:", pred_Y)
    print("Predicted activity mask:", activity)

import torch

def angular_error_pit(pred, target, mask):
    """
    pred:   (batch, K, 3)
    target: (batch, K, 3)
    mask:   (batch, K)
    """

    B, K, _ = pred.shape
    perms = list(itertools.permutations(range(K)))

    total_angle = 0.0
    total_active = 0.0

    for b in range(B):
        pred_b = pred[b]       # (K,3)
        target_b = target[b]
        mask_b = mask[b]

        # normalize once
        pred_unit = pred_b / (pred_b.norm(dim=-1, keepdim=True) + 1e-8)
        target_unit = target_b / (target_b.norm(dim=-1, keepdim=True) + 1e-8)

        min_angle_sum = None
        best_active = None

        for perm in perms:
            perm = torch.tensor(perm, device=pred.device)

            target_perm = target_unit[perm]
            mask_perm = mask_b[perm]

            cos = (pred_unit * target_perm).sum(dim=-1)
            cos = cos.clamp(-1.0, 1.0)

            angles = torch.acos(cos)  # radians
            angles_deg = angles * 180.0 / torch.pi

            angle_sum = (angles_deg * mask_perm).sum()
            active_count = mask_perm.sum()

            if (min_angle_sum is None) or (angle_sum < min_angle_sum):
                min_angle_sum = angle_sum
                best_active = active_count

        total_angle += min_angle_sum
        total_active += best_active

    return total_angle / (total_active + 1e-8)

batch_angular_error = angular_error_pit(pred_Y, Y_accdoa.to(device), Y_mask.to(device))
print(f"Mean angular error: {batch_angular_error:.2f}°")

