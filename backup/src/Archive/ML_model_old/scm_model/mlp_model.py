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


class MLP(nn.Module):
    def __init__(self, nb_tracks=3, hidden_sizes=[256,256,256], dropout=0.05):
        super().__init__()

        self.nb_tracks = nb_tracks
        layers = []
        prev_size = None

        for size in hidden_sizes:
            if prev_size is None:
                layers.append(nn.LazyLinear(size))
            else:
                layers.append(nn.Linear(prev_size, size))

            layers.append(nn.LayerNorm(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev_size = size

        layers.append(nn.Linear(prev_size, nb_tracks * 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = flatten_unique_scm(x)
        out = self.mlp(x)
        out = out.view(x.shape[0], self.nb_tracks, 3)
        return out

#########################################
# PIT Loss (works with MLP output)
#########################################
class PITLoss(nn.Module):
    def __init__(self, nb_tracks=3):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.perms = list(itertools.permutations(range(nb_tracks)))

    def forward(self, pred, target):
        # pred: (B, nb_tracks, 3)
        # target: (B, nb_tracks, 3)
        losses = []
        for perm in self.perms:
            permuted_target = target[:, perm, :]
            loss = self.mse(pred, permuted_target).mean(dim=(1, 2))
            losses.append(loss)

        losses = torch.stack(losses, dim=1)
        min_loss, _ = torch.min(losses, dim=1)
        return min_loss.mean()