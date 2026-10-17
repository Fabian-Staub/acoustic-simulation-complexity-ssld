import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score


# -----------------------------
# 1️⃣ ResNet model definition
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class SELD_ResNet(nn.Module):
    def __init__(self, input_channels=7, max_sources=3):
        super(SELD_ResNet, self).__init__()
        self.max_sources = max_sources

        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, max_sources * 3)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out.view(x.size(0), self.max_sources, 3)


# -----------------------------
# 3️⃣ Load dataset
# -----------------------------
dataset_path = "../data/harp/Generated_HOA_IRs_ISM/harp_dataset.pt"
data = torch.load(dataset_path, weights_only=False)

X = data["X"].float()
Y = data["Y"]

# Extract numeric targets
Y_accdoa_list = []
Y_mask_list = []

for y in Y:
    y_float = np.zeros((y.shape[0], 4), dtype=np.float32)
    y_float[:, :4] = y[:, :4].astype(np.float32)
    Y_accdoa_list.append(y_float[:, :3])
    Y_mask_list.append(y_float[:, 3])

Y_accdoa = torch.tensor(np.stack(Y_accdoa_list, axis=0), dtype=torch.float32)
Y_mask   = torch.tensor(np.stack(Y_mask_list, axis=0), dtype=torch.float32)


# -----------------------------
# 4️⃣ Load trained model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SELD_ResNet(
    input_channels=X.shape[1],
    max_sources=Y_accdoa.shape[1]
).to(device)

model.load_state_dict(torch.load("seld_accdoa_resnet.pth", map_location=device))
model.eval()

print("Model loaded successfully.")






def evaluate_seld_doa(pred, gt, tdoa_deg=20.0, eps=1e-8):
    """
    SELD evaluation simplified for single class, 3 tracks, DOA only.
    - pred: (batch, num_tracks, 3) predicted DOA vectors
    - gt: (batch, num_tracks, 3) ground truth DOA vectors
    """
    batch_size, num_tracks, _ = pred.shape
    tdoa_rad = tdoa_deg * np.pi / 180

    all_tp = []
    all_fp = []
    all_fn = []
    all_doa_errors = []

    for b in range(batch_size):
        # compute cost matrix for Hungarian (angular distance)
        cost_matrix = torch.zeros((num_tracks, num_tracks))
        for i in range(num_tracks):
            for j in range(num_tracks):
                pred_vec = pred[b, i] / (pred[b, i].norm() + eps)
                gt_vec = gt[b, j] / (gt[b, j].norm() + eps)
                cos_theta = torch.clamp(torch.dot(pred_vec, gt_vec), -1, 1)
                cost_matrix[i, j] = torch.acos(cos_theta)  # angular error in radians

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
        matched_angles = cost_matrix[row_ind, col_ind]

        # Apply angular threshold
        tp_mask = matched_angles <= tdoa_rad
        tp = tp_mask.sum().item()
        fp = num_tracks - tp  # unmatched predictions
        fn = num_tracks - tp  # unmatched GTs

        # Save metrics
        all_tp.append(tp)
        all_fp.append(fp)
        all_fn.append(fn)

        if tp > 0:
            doa_error = matched_angles[tp_mask].mean().item()
            all_doa_errors.append(doa_error)

    # Location-dependent F1
    total_tp = sum(all_tp)
    total_fp = sum(all_fp)
    total_fn = sum(all_fn)
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # DOA error: mean over all frames with TP
    mean_doa_error_rad = np.mean(all_doa_errors) if len(all_doa_errors) > 0 else 0.0
    mean_doa_error_deg = mean_doa_error_rad * 180 / np.pi

    return {
        "FLD": f1,
        "DOAE_rad": mean_doa_error_rad,
        "DOAE_deg": mean_doa_error_deg,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn
    }



# -----------------------------
# Example evaluation
# -----------------------------
model.eval()
with torch.no_grad():
    pred_Y = model(X.to(device))
    results = evaluate_seld(pred_Y, Y_accdoa.to(device))

print(f"SELD F1: {results['FLD']:.3f}")
print(f"Mean Angular Error: {results['DOAE_deg']:.2f}°")