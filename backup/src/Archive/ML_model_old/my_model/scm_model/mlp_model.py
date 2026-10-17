import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

#########################################
# Utility: Flatten only unique SCM elements
#########################################
def flatten_unique_scm(x):
    """
    x: (B, C, H, W) SCM tensor
    Returns only unique features (upper triangle of each HxW matrix per channel)
    """
    B, C, H, W = x.shape
    assert H == W, "SCM matrices must be square"
    mask = torch.triu(torch.ones(H, W, dtype=torch.bool, device=x.device))
    x_unique = x[:, :, mask]  # shape -> (B, C, H*(H+1)//2)
    x_unique = x_unique.view(B, -1)  # flatten to (B, C * H*(H+1)//2)
    return x_unique

#########################################
# Multi-Layer Perceptron for SCM
#########################################
class SimpleSeldMLP(nn.Module):
    def __init__(self, in_feat_shape):
        super().__init__()
        # Flattened unique input dimension
        B, C, H, W = 1, *in_feat_shape  # dummy B
        in_features = C * (H * (H + 1) // 2)

        # MLP hyperparameters
        hidden_sizes = [512, 256, 128]
        dropout = 0.05
        output_dim = 3  # predict x, y, z

        # Build layers
        layers = []
        prev_size = in_features
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = size

        # Final output layer
        layers.append(nn.Linear(prev_size, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # Remove redundancy
        x = flatten_unique_scm(x)  # (B, unique_features)
        out = self.mlp(x)
        out = out.unsqueeze(1)      # (B, 1, 3) to match original interface
        return out

#########################################
# PIT Loss (works with MLP output)
#########################################
class PITLoss(nn.Module):
    def __init__(self, nb_tracks=1):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.perms = list(itertools.permutations(range(nb_tracks)))

    def forward(self, pred, target):
        # pred: (B, 1, 3) or (B, nb_tracks, 3)
        # target: (B, nb_tracks, 3)
        losses = []
        for perm in self.perms:
            permuted_target = target[:, perm, :]
            loss = self.mse(pred, permuted_target).mean(dim=(1, 2))
            losses.append(loss)

        losses = torch.stack(losses, dim=1)
        min_loss, _ = torch.min(losses, dim=1)
        return min_loss.mean()