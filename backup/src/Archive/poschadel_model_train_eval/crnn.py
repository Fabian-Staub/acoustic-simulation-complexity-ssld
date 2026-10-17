import torch
import torch.nn as nn
import itertools


class SDLCRNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 46,
        num_speakers: int = 3,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.num_speakers = num_speakers
        self.output_size = num_speakers * 3

        self.conv1 = nn.Conv2d(
            input_channels,
            128,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(128)
        self.elu1 = nn.ELU()
        self.pool1 = nn.MaxPool2d((1, 2))
        self.drop1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.elu2 = nn.ELU()
        self.pool2 = nn.MaxPool2d((1, 8))
        self.drop2 = nn.Dropout2d(dropout_rate)

        self.conv3 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.elu3 = nn.ELU()
        self.pool3 = nn.MaxPool2d((1, 4))
        self.drop3 = nn.Dropout2d(dropout_rate)

        self.bilstm1 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.bilstm2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(256, 256)
        self.fc_elu = nn.ELU()
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected model input shape (B, T, M, C), "
                f"got {tuple(x.shape)}"
            )

        if x.shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, "
                f"got {x.shape[-1]}"
            )

        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.drop1(self.pool1(self.elu1(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.elu2(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.elu3(self.bn3(self.conv3(x)))))

        if x.shape[-1] != 2:
            raise ValueError(
                "Unexpected pooled mel dimension. "
                f"Expected 2, got {x.shape[-1]}. "
                "The current model expects 128 input mel bins."
            )

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.flatten(start_dim=2)

        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)

        x = self.fc1(x)
        x = self.fc_elu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x




class FramewisePITLoss(nn.Module):
    def __init__(self, nb_tracks: int = 3) -> None:
        super().__init__()

        self.nb_tracks = int(nb_tracks)

        permutations = list(
            itertools.permutations(range(self.nb_tracks))
        )

        self.register_buffer(
            "permutations",
            torch.tensor(permutations, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if pred.ndim == 3:
            expected_output_size = self.nb_tracks * 3

            if pred.shape[-1] != expected_output_size:
                raise ValueError(
                    f"Expected prediction output size "
                    f"{expected_output_size}, got {pred.shape[-1]}"
                )

            pred = pred.reshape(
                pred.shape[0],
                pred.shape[1],
                self.nb_tracks,
                3,
            )

        if pred.ndim != 4:
            raise ValueError(
                f"Expected pred shape (B, T, tracks, 3), "
                f"got {tuple(pred.shape)}"
            )

        if target.ndim != 4:
            raise ValueError(
                f"Expected target shape (B, T, tracks, 3), "
                f"got {tuple(target.shape)}"
            )

        if pred.shape != target.shape:
            raise ValueError(
                f"pred shape {tuple(pred.shape)} does not match "
                f"target shape {tuple(target.shape)}"
            )

        permutation_losses = []

        for permutation in self.permutations:
            permuted_target = target[:, :, permutation, :]

            loss = (
                pred - permuted_target
            ).pow(2).mean(dim=(-1, -2))

            permutation_losses.append(loss)

        permutation_losses = torch.stack(
            permutation_losses,
            dim=0,
        )

        min_loss = permutation_losses.min(dim=0).values

        return min_loss.mean()