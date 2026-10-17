import torch
import torch.nn as nn
import itertools

#########################################
# Utility:
# Flatten only unique SCM entries
#########################################

def flatten_unique_scm(x):
    """
    Input:
        x -> (B, C, H, W)

    Output:
        (B, C, T)
    """

    B, C, H, W = x.shape

    mask = torch.triu(
        torch.ones(H, W, dtype=torch.bool, device=x.device)
    )

    x = x[:, :, mask]

    return x


#########################################
# CNN on flattened triangular SCMs
#########################################

class CNNxx(nn.Module):

    def __init__(
        self,
        nb_tracks=3,
        in_channels=8,
        conv_channels=32,
        hidden_sizes=[1024, 512, 512, 256, 128],
        dropout=0.05
    ):
        super().__init__()

        self.nb_tracks = nb_tracks

        #################################
        # 1D CNN frontend
        #################################

        self.conv = nn.Sequential(

            nn.Conv1d(
                in_channels=in_channels,
                out_channels=conv_channels,
                kernel_size=3
            ),

            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        #################################
        # MLP backend
        #################################

        layers = []

        prev_size = None

        for size in hidden_sizes:

            if prev_size is None:
                layers.append(nn.LazyLinear(size))
            else:
                layers.append(nn.Linear(prev_size, size))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev_size = size

        layers.append(
            nn.Linear(prev_size, nb_tracks * 3)
        )

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):

        #################################
        # Extract triangular SCM entries
        #################################

        x = flatten_unique_scm(x)
        # shape:
        # (B, 8, T)

        #################################
        # Conv1D across SCM channels
        #################################

        x = self.conv(x)
        # (B, conv_channels, T)

        #################################
        # Flatten
        #################################

        x = x.flatten(start_dim=1)

        #################################
        # MLP
        #################################

        x = self.mlp(x)

        x = x.view(
            x.shape[0],
            self.nb_tracks,
            3
        )

        return x


class CNNx(nn.Module):

    def __init__(
        self,
        nb_tracks=3,
        in_channels=10,
        conv_channels=[32, 64, 128, 256],
        hidden_sizes=[1024, 512, 256],
        dropout=0.05
    ):
        super().__init__()

        self.nb_tracks = nb_tracks

        #################################
        # Deeper CNN frontend
        #################################

        conv_layers = []

        prev_channels = in_channels

        for out_channels in conv_channels:

            conv_layers.extend([
                nn.Conv1d(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ),

                nn.BatchNorm1d(out_channels),
                nn.ReLU(),

                # second conv block
                nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ),

                nn.BatchNorm1d(out_channels),
                nn.ReLU(),

                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout)
            ])

            prev_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        #################################
        # MLP backend
        #################################

        layers = []

        prev_size = None

        for size in hidden_sizes:

            if prev_size is None:
                layers.append(nn.LazyLinear(size))
            else:
                layers.append(nn.Linear(prev_size, size))

            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

            prev_size = size

        layers.append(
            nn.Linear(prev_size, nb_tracks * 3)
        )

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):

        #################################
        # Extract triangular SCM entries
        #################################

        x = flatten_unique_scm(x)
        # (B, 8, T)

        #################################
        # CNN frontend
        #################################

        x = self.conv(x)
        # (B, C, T')

        #################################
        # Flatten
        #################################

        x = x.flatten(start_dim=1)

        #################################
        # MLP backend
        #################################

        x = self.mlp(x)

        #################################
        # Output
        #################################

        x = x.view(
            x.shape[0],
            self.nb_tracks,
            3
        )

        return x

# Final Archtecture!!!!
class CNN(nn.Module):

    def __init__(
        self,
        nb_tracks=3,
        in_channels=10,
        conv_channels=[32, 64, 128, 256],
        hidden_sizes=[1024,512,256],
        dropout=0.15
    ):
        super().__init__()

        self.nb_tracks = nb_tracks

        #################################
        # Deeper CNN frontend
        #################################

        conv_layers = []

        prev_channels = in_channels

        for out_channels in conv_channels:

            conv_layers.extend([
                nn.Conv1d(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ),

                nn.BatchNorm1d(out_channels),
                nn.ReLU(),

                nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ),

                nn.BatchNorm1d(out_channels),
                nn.ReLU(),

                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout)
            ])

            prev_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        #################################
        # MLP backend
        #################################

        layers = []

        prev_size = None

        for size in hidden_sizes:

            if prev_size is None:
                layers.append(nn.LazyLinear(size))
            else:
                layers.append(nn.Linear(prev_size, size))

            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

            prev_size = size

        layers.append(
            nn.Linear(prev_size, nb_tracks * 3)
        )

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):

        #################################
        # Extract triangular SCM entries
        #################################

        x = flatten_unique_scm(x)
        # (B, 8, T)

        #################################
        # CNN frontend
        #################################

        x = self.conv(x)
        # (B, C, T')

        #################################
        # Flatten
        #################################

        x = x.flatten(start_dim=1)

        #################################
        # MLP backend
        #################################

        x = self.mlp(x)

        #################################
        # Output
        #################################

        x = x.view(
            x.shape[0],
            self.nb_tracks,
            3
        )

        return x







class SimpleSeldCNN2D(nn.Module):

    def __init__(
        self,
        nb_tracks=3,
        in_channels=8,
        conv_channels=[32, 64, 128],
        hidden_sizes=[512, 256],
        dropout=0.05
    ):
        super().__init__()

        self.nb_tracks = nb_tracks

        #################################
        # 2D CNN frontend
        #################################

        layers = []

        prev_channels = in_channels

        for out_channels in conv_channels:

            layers.extend([

                #################################
                # Conv block
                #################################

                nn.Conv2d(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ),

                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ),

                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                #################################
                # Downsampling
                #################################

                nn.MaxPool2d(kernel_size=2),

                nn.Dropout2d(dropout)
            ])

            prev_channels = out_channels

        self.conv = nn.Sequential(*layers)

        #################################
        # Global pooling
        #################################

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        #################################
        # MLP backend
        #################################

        mlp_layers = []

        prev_size = conv_channels[-1]

        for size in hidden_sizes:

            mlp_layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

            prev_size = size

        mlp_layers.append(
            nn.Linear(prev_size, nb_tracks * 3)
        )

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):

        #################################
        # Input shape
        #################################
        # (B, 8, 25, 25)

        #################################
        # CNN frontend
        #################################

        x = self.conv(x)

        # Example progression:
        # (B, 8, 25, 25)
        # -> (B, 32, 12, 12)
        # -> (B, 64, 6, 6)
        # -> (B, 128, 3, 3)

        #################################
        # Global average pooling
        #################################

        x = self.global_pool(x)

        # (B, 128, 1, 1)

        #################################
        # Flatten
        #################################

        x = x.flatten(start_dim=1)

        # (B, 128)

        #################################
        # MLP backend
        #################################

        x = self.mlp(x)

        #################################
        # Output reshape
        #################################

        x = x.view(
            x.shape[0],
            self.nb_tracks,
            3
        )

        return x


class CNN2Dxx(nn.Module):

    def __init__(
        self,
        nb_tracks=3,
        in_channels=8,
        hidden_sizes=[512, 256],
        dropout=0.05
    ):
        super().__init__()

        self.nb_tracks = nb_tracks

        #################################
        # 2D CNN frontend
        #################################

        self.conv = nn.Sequential(

            #################################
            # Block 1
            # Input: (B, 8, 25, 25)
            #################################

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            # 25 -> 12

            nn.Dropout2d(dropout),

            #################################
            # Block 2
            # (B, 32, 12, 12)
            #################################

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            # 12 -> 6

            nn.Dropout2d(dropout),

            #################################
            # Block 3
            # (B, 64, 6, 6)
            #################################

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Dropout2d(dropout)
        )

        #################################
        # MLP backend
        #################################

        mlp_layers = []

        # Final CNN shape:
        # (B, 128, 6, 6)

        prev_size = 128 * 6 * 6

        for size in hidden_sizes:

            mlp_layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

            prev_size = size

        mlp_layers.append(
            nn.Linear(prev_size, nb_tracks * 3)
        )

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):

        #################################
        # Input
        #################################
        # Expected:
        # (B, 8, 25, 25)

        #################################
        # CNN frontend
        #################################

        x = self.conv(x)

        # Output:
        # (B, 128, 6, 6)

        #################################
        # Flatten
        #################################

        x = x.flatten(start_dim=1)

        # (B, 4608)

        #################################
        # MLP backend
        #################################

        x = self.mlp(x)

        #################################
        # Reshape output
        #################################

        x = x.view(
            x.shape[0],
            self.nb_tracks,
            3
        )

        return x

class CNN2D(nn.Module):
    def __init__(
        self,
        nb_tracks=3,
        in_channels=10,
        hidden_sizes=[512, 256, 128],
        dropout=0.1,
    ):
        super().__init__()

        self.nb_tracks = nb_tracks

        #################################
        # CNN frontend
        #################################

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout(dropout),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )

        #################################
        # MLP backend
        #################################

        self.mlp = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, nb_tracks * 3),
        )

    def forward(self, x):
        x = self.conv(x)          # (B, 64, H', W')
        x = x.flatten(1)          # (B, 2304)
        x = self.mlp(x)
        x = x.view(x.size(0), self.nb_tracks, 3)

        return x
