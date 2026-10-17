import itertools
import torch
import torch.nn as nn


class PoschadelCRNN(nn.Module):
    """
    CRNN from Poschadel et al. Table 2, adapted to PyTorch.

    Expected input:
        x: (B, T, F, C), e.g. (B, 50, 513, C)

    Output:
        out: (B, T, nb_tracks, 3), e.g. (B, 50, 3, 3)

    For nb_tracks=3, the final FC has 9 outputs per frame:
        3 tracks * 3 ACCDOA Cartesian coordinates.
    """
    def __init__(self, nb_tracks=3, dropout=0.2):
        super().__init__()
        self.nb_tracks = nb_tracks

        # LazyConv2d lets the model infer the feature-channel count C
        # on the first forward pass.
        self.conv = nn.Sequential(
            nn.LazyConv2d(128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # After pooling: F = 513 -> 64 -> 8 -> 2
        # Channels = 128, so per time frame: 128 * 2 = 256.
        self.bilstm1 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, nb_tracks * 3),
        )

    def forward(self, x):
        # x from your dataset should be (B, T, F, C).
        # Conv2d expects (B, C, T, F).
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B,T,F,C), got {tuple(x.shape)}")

        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, T, F)

        x = self.conv(x)                        # (B, 128, T, 2)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, 128, 2)
        x = x.flatten(start_dim=2)              # (B, T, 256)

        x, _ = self.bilstm1(x)                  # (B, T, 256)
        x, _ = self.bilstm2(x)                  # (B, T, 256)

        out = self.fc(x)                        # (B, T, nb_tracks * 3)
        out = out.view(x.shape[0], x.shape[1], self.nb_tracks, 3)
        return out
    


class PITLoss(nn.Module):
    def __init__(self, nb_tracks=3):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.perms = list(itertools.permutations(range(nb_tracks)))

    def forward(self, pred, target):
        """
        pred:   (B, T, nb_tracks, 3)
        target: (B, T, nb_tracks, 3)
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape, "
                f"got pred={tuple(pred.shape)}, target={tuple(target.shape)}"
            )

        losses = []
        for perm in self.perms:
            permuted_target = target[:, :, perm, :]
            loss = self.mse(pred, permuted_target).mean(dim=(1, 2, 3))
            losses.append(loss)

        losses = torch.stack(losses, dim=1)
        min_loss, _ = torch.min(losses, dim=1)
        return min_loss.mean()