import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

#########################################
# Residual Block
#########################################
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

#########################################
# SCMNet using params dictionary
#########################################
class SimpleSeldCNN(nn.Module):
    def __init__(self, in_feat_shape):
        super().__init__()

        in_channels = in_feat_shape[0]  # e.g., 7 for SCM
        base = 256
        fnn_size = 512
        dropout = 0.1

        #################################
        # Convolutions
        #################################
        self.conv1 = nn.Conv2d(in_channels, base, kernel_size=1)  # channel mixing
        self.res1 = ResidualBlock(base)

        self.conv2 = nn.Conv2d(base, base, kernel_size=3, padding=1)
        self.res2 = ResidualBlock(base)

        #################################
        # Global Pooling
        #################################
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        #################################
        # Fully Connected Head
        #################################
        self.fc = nn.Sequential(
            nn.Linear(base, fnn_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fnn_size, fnn_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fnn_size // 2, 3)  # predict x, y, z
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)

        x = F.relu(self.conv2(x))
        x = self.res2(x)

        x = self.pool(x)             # (B, base, 1, 1)
        x = x.view(x.size(0), -1)    # (B, base)

        out = self.fc(x)             # (B, 3)
        out = out.unsqueeze(1)       # (B, 1, 3)
        return out

#########################################
# PIT Loss (optional)
#########################################
class PITLoss(nn.Module):
    def __init__(self, nb_tracks=1):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.perms = list(itertools.permutations(range(nb_tracks)))

    def forward(self, pred, target):
        losses = []
        for perm in self.perms:
            permuted_target = target[:, perm, :]
            loss = self.mse(pred, permuted_target).mean(dim=(1, 2))
            losses.append(loss)

        losses = torch.stack(losses, dim=1)
        min_loss, _ = torch.min(losses, dim=1)
        return min_loss.mean()