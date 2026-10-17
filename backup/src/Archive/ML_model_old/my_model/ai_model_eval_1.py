import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools


# -----------------------------
# 1️⃣ Model definition (same as training)
# -----------------------------
class SELD_CNN(nn.Module):
    def __init__(self, input_channels=7, max_sources=3):
        super(SELD_CNN, self).__init__()
        self.max_sources = max_sources
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2,2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2,2))

        self.fc = nn.Linear(128*3*3, max_sources * 3)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        b, c, h, w = x.shape
        x = x.view(b, c*h*w)
        out = self.fc(x)
        out = out.view(b, self.max_sources, 3)
        return out


# -----------------------------
# 2️⃣ PIT Angular Error
# -----------------------------
def angular_error_pit(pred, target, gt_mask):
    """
    Angular error computed ONLY where GT is active (gt_mask == 1).
    
    pred:    (B, K, 3)
    target:  (B, K, 3)
    gt_mask: (B, K)
    """

    B, K, _ = pred.shape
    perms = list(itertools.permutations(range(K)))

    total_angle = 0.0
    total_count = 0.0

    for b in range(B):
        pred_b = pred[b]
        target_b = target[b]
        gt_mask_b = gt_mask[b]

        # normalize
        pred_unit = pred_b / (pred_b.norm(dim=-1, keepdim=True) + 1e-8)
        target_unit = target_b / (target_b.norm(dim=-1, keepdim=True) + 1e-8)

        min_angle_sum = None
        min_count = None

        for perm in perms:
            perm = torch.tensor(perm, device=pred.device)

            target_perm = target_unit[perm]
            gt_mask_perm = gt_mask_b[perm]

            # Only evaluate where GT is active
            valid_mask = gt_mask_perm

            cos = (pred_unit * target_perm).sum(dim=-1).clamp(-1.0, 1.0)
            angles = torch.acos(cos) * 180.0 / torch.pi

            angle_sum = (angles * valid_mask).sum()
            count = valid_mask.sum()

            if (min_angle_sum is None) or (angle_sum < min_angle_sum):
                min_angle_sum = angle_sum
                min_count = count

        total_angle += min_angle_sum
        total_count += min_count

    return total_angle / (total_count + 1e-8)


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

model = SELD_CNN(
    input_channels=X.shape[1],
    max_sources=Y_accdoa.shape[1]
).to(device)

model.load_state_dict(torch.load("seld_accdoa_cnn.pth", map_location=device))
model.eval()

print("Model loaded successfully.")


# -----------------------------
# 5️⃣ Inference
# -----------------------------
with torch.no_grad():
    X = X.to(device)
    Y_accdoa = Y_accdoa.to(device)
    Y_mask = Y_mask.to(device)

    pred_Y = model(X)

    pred_activity = (pred_Y.norm(dim=-1) > 1).float()

    print("\nSample-wise Activity Comparison:\n")
    for i in range(len(pred_activity)):
        print(f"Sample {i}:")
        print("Predicted:", pred_activity[i].cpu().numpy())
        print("GroundTruth:", Y_mask[i].cpu().numpy())
        print("-" * 40)


# -----------------------------
# 6️⃣ Evaluation
# -----------------------------
mean_angular_error = angular_error_pit(pred_Y, Y_accdoa, Y_mask)
print(f"\nMean angular error: {mean_angular_error:.2f}°")