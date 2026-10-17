import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If dimensions change, adjust shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)
        return out

# -----------------------------
# 1️⃣ Model definition
# -----------------------------
class SELD_ResNet(nn.Module):
    def __init__(self, input_channels=7, max_sources=3):
        super(SELD_ResNet, self).__init__()
        self.max_sources = max_sources

        # Initial conv
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual stages
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)

        # Adaptive pooling (removes need to hardcode 3×3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128, max_sources * 3)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        out = self.fc(x)
        out = out.view(x.size(0), self.max_sources, 3)
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

# Select only the first sample
X = X[0:1]          # shape: (1, 7, 25, 25)
Y_accdoa = Y_accdoa[0:1]  # shape: (1, 3, 3)
Y_mask = Y_mask[0:1]      # shape: (1, 3)

print("X:", X.shape)
print("Y_accdoa:", Y_accdoa.shape)
print("Y_mask:", Y_mask.shape)
#print("Y_accdoa:", Y_accdoa)
#print("Y_mask:", Y_mask)
#print("Y:", Y)


# -----------------------------
# 3️⃣ Masked MSE loss
# -----------------------------

def pit_mse_loss(pred, target):
    """
    Permutation Invariant MSE Loss for:
    - 1 class
    - 3 tracks

    Args:
        pred:   (batch_size, 3)   predicted values
        target: (batch_size, 3)   ground truth values

    Returns:
        loss: scalar (mean PIT-MSE over batch)
    """

    batch_size, num_tracks, _ = pred.shape

    # All 3! permutations of indices [0,1,2]
    perms = list(itertools.permutations(range(num_tracks)))

    losses = []

    for perm in perms:
        # Permute target tracks
        permuted_target = target[:, perm]

        # Compute MSE per sample (mean over tracks)
        mse = F.mse_loss(pred, permuted_target, reduction='none')
        mse = mse.mean(dim=1)  # mean over 3 tracks

        losses.append(mse)

    # Stack shape: (num_perms, batch_size)
    losses = torch.stack(losses, dim=0)

    # Take minimum over permutations
    min_loss, _ = torch.min(losses, dim=0)

    # Mean over batch
    return min_loss.mean()


def pit_mse_loss_fast(pred, target):
    perms = torch.tensor(list(itertools.permutations(range(3))),
                         device=pred.device)

    # Expand for broadcasting
    pred_exp = pred.unsqueeze(0)                # (1, B, 3)
    target_exp = target.unsqueeze(0)[:, :, perms]  # (6, B, 3)

    mse = (pred_exp - target_exp) ** 2
    mse = mse.mean(dim=2)  # mean over tracks

    min_mse = mse.min(dim=0)[0]

    return min_mse.mean()



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
model = SELD_ResNet(input_channels=X.shape[1], max_sources=Y_accdoa.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)


# -----------------------------
# 6️⃣ Training loop
# -----------------------------
num_epochs = 10

# -----------------------------
# 6️⃣ Training loop (direction-only)
# -----------------------------
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_Y, batch_mask in loader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)
        batch_mask = batch_mask.to(device)

        optimizer.zero_grad()
        pred_Y = model(batch_X)  # (B, max_sources, 3)

        # -----------------------------
        # Normalize GT and pred, mask inactive tracks
        # -----------------------------
        active_mask = batch_mask.unsqueeze(-1)  # (B, max_sources, 1)
        batch_Y_unit = batch_Y / (batch_Y.norm(dim=-1, keepdim=True) + 1e-8)
        batch_Y_unit = batch_Y_unit * active_mask

        pred_Y_unit = pred_Y / (pred_Y.norm(dim=-1, keepdim=True) + 1e-8)
        pred_Y_unit = pred_Y_unit * active_mask

        # Compute PIT-MSE on unit vectors
        loss = pit_mse_loss(pred_Y_unit, batch_Y_unit)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(loader):.6f}")
    
'''
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_Y, batch_mask in loader:
        batch_X, batch_Y, batch_mask = batch_X.to(device), batch_Y.to(device), batch_mask.to(device)

        optimizer.zero_grad()
        pred_Y = model(batch_X)
        loss = pit_mse_loss(pred_Y, batch_Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(loader):.6f}")
'''

# -----------------------------
# 7️⃣ Save trained model
# -----------------------------
torch.save(model.state_dict(), "seld_accdoa_resnet.pth")
print("Model saved as seld_accdoa_resnet.pth")


def pit_angular_error(pred, gt, eps=1e-8):
    """
    PIT Angular Error für ACCDOA-Vektoren
    pred, gt: (batch, 3, 3) -> batch_size, tracks, xyz
    """
    batch_size, num_tracks, _ = pred.shape
    perms = list(itertools.permutations(range(num_tracks)))
    
    all_loss = []

    for perm in perms:
        # Permutiere GT Tracks
        gt_perm = gt[:, perm, :]
        
        # Maske für aktive Events
        active_mask = gt_perm.norm(dim=2) > eps  # shape: (batch, tracks)
        
        # Angular Error für aktive Tracks
        pred_norm = pred / (pred.norm(dim=2, keepdim=True) + eps)
        gt_norm = gt_perm / (gt_perm.norm(dim=2, keepdim=True) + eps)
        
        cos_theta = (pred_norm * gt_norm).sum(dim=2).clamp(-1, 1)
        theta = torch.acos(cos_theta)  # in rad
        
        # Nur aktive Tracks zählen
        theta = theta * active_mask.float()
        # Mittlerer Fehler über aktive Tracks (avoid div 0)
        mean_theta = theta.sum(dim=1) / (active_mask.sum(dim=1) + eps)
        
        all_loss.append(mean_theta)
    
    # Alle Permutationen vergleichen, Minimum nehmen
    all_loss = torch.stack(all_loss, dim=0)  # (num_perms, batch)
    min_loss, _ = torch.min(all_loss, dim=0)
    return min_loss.mean()



def evaluate_seld(pred, gt, eps=1e-8, thresholds=None):
    """
    Joint SELD evaluation:
    - Uses permutation with lowest angular error
    - Returns mean angular error
    - Computes optimal SED threshold
    """

    batch_size, num_tracks, _ = pred.shape
    perms = list(itertools.permutations(range(num_tracks)))

    all_ang_errors = []
    permuted_gts = []

    for perm in perms:
        gt_perm = gt[:, perm, :]
        
        active_mask = gt_perm.norm(dim=2) > eps
        
        pred_norm = pred / (pred.norm(dim=2, keepdim=True) + eps)
        gt_norm = gt_perm / (gt_perm.norm(dim=2, keepdim=True) + eps)

        cos_theta = (pred_norm * gt_norm).sum(dim=2).clamp(-1, 1)
        theta = torch.acos(cos_theta)

        theta = theta * active_mask.float()
        mean_theta = theta.sum(dim=1) / (active_mask.sum(dim=1) + eps)

        all_ang_errors.append(mean_theta)
        permuted_gts.append(gt_perm)

    all_ang_errors = torch.stack(all_ang_errors, dim=0)  # (num_perms, batch)

    # Best permutation per sample
    min_ang_error, best_perm_idx = torch.min(all_ang_errors, dim=0)

    # Apply best permutation
    optimal_gt = torch.zeros_like(gt)
    for i in range(batch_size):
        optimal_gt[i] = permuted_gts[best_perm_idx[i]][i]

    mean_angular_error = min_ang_error.mean().item()

    # -------------------------
    # SED evaluation
    # -------------------------
    if thresholds is None:
        thresholds = torch.linspace(0.01, 0.99, 99)

    gt_active = (optimal_gt.norm(dim=2) > eps).cpu().numpy().astype(int)

    best_f1 = 0
    best_thresh = 0.5
    best_metrics = {}

    for t in thresholds:
        t = float(t)
        pred_active = (pred.norm(dim=2) > t).cpu().numpy().astype(int)

        precision = precision_score(gt_active.flatten(), pred_active.flatten(), zero_division=0)
        recall = recall_score(gt_active.flatten(), pred_active.flatten(), zero_division=0)
        f1 = f1_score(gt_active.flatten(), pred_active.flatten(), zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

    return {
        "mean_angular_error_rad": mean_angular_error,
        "mean_angular_error_deg": mean_angular_error * 180 / torch.pi,
        "best_threshold": best_thresh,
        "sed_metrics": best_metrics
    }



# -----------------------------
# Example evaluation
# -----------------------------
model.eval()
with torch.no_grad():
    pred_Y = model(X.to(device))
    results = evaluate_seld(pred_Y, Y_accdoa.to(device))

print(pred_Y, Y_accdoa)

batch_angular_error = pit_angular_error(pred_Y, Y_accdoa.to(device)) * 180 / torch.pi
print(f"Mean angular error: {batch_angular_error:.2f}°")

print(f"Mean Angular Error: {results['mean_angular_error_deg']:.2f}°")
print(f"Optimal Threshold: {results['best_threshold']:.2f}")
print(f"SED F1: {results['sed_metrics']['f1']:.3f}")

