import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

#########################################
# Simple Residual Block
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
# Flexible Simple CNN SELD Model
#########################################
class SimpleSeldCNN(nn.Module):
    def __init__(self, in_feat_shape, params):
        super().__init__()

        in_channels = in_feat_shape[0]   # z.B. 7
        base = params.get('nb_cnn2d_filt', 128)
        dropout = params.get('dropout_rate', 0.05)
        fnn_size = params.get('fnn_size', 512)
        self.nb_tracks = params.get('nb_tracks', 3)

        # Pooling-Faktoren: tuple (scm_dim1, scm_dim2)
        pool_size = params.get('pool_size', (2, 2))

        #################################
        # 3 Conv Layers + Skip Blocks
        #################################
        self.conv1 = nn.Conv2d(in_channels, base, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(base)
        self.pool1 = nn.MaxPool2d(pool_size)

        self.conv2 = nn.Conv2d(base, base, kernel_size=3, padding=1)
        self.res2 = ResidualBlock(base)
        self.pool2 = nn.MaxPool2d(pool_size)

        self.conv3 = nn.Conv2d(base, base, kernel_size=3, padding=1)
        self.res3 = ResidualBlock(base)
        self.pool3 = nn.MaxPool2d(pool_size)

        #################################
        # Head
        #################################
        self.fc = nn.Sequential(
            nn.Linear(base, fnn_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fnn_size, 3)   # only x, y, z
        )

    def forward(self, x):
        # (B, C, T, F)
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.res2(x)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.res3(x)
        x = self.pool3(x)

        # collapse spatial dimensions
        x = x.mean(dim=[2, 3])   # (B, base)

        out = self.fc(x)         # (B, 3)
        out = out.unsqueeze(1)   # (B, 1, 3)


#########################################
# PIT Loss bleibt gleich
#########################################
class PITLoss(nn.Module):
    def __init__(self, nb_tracks=3):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.perms = list(itertools.permutations(range(nb_tracks)))

    def forward(self, pred, target):
        losses = []
        for perm in self.perms:
            permuted_target = target[:, perm, :]
            loss = self.mse(pred, permuted_target).mean(dim=(1,2))
            losses.append(loss)

        losses = torch.stack(losses, dim=1)
        min_loss,_ = torch.min(losses, dim=1)
        return min_loss.mean()