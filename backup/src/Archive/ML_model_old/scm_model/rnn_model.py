import torch
import torch.nn as nn


def flatten_unique_scm_temporal(x, include_diagonal=True):
    """
    Input:
        x: (B, W, F, C, C)

    Returns:
        x_flat: (B, W, F, N)
        where N is the number of unique SCM entries.
    """

    if x.ndim != 5:
        raise ValueError(
            "Expected x with shape (B, n_windows, n_scms, n_channels, n_channels)"
        )

    B, W, F, C1, C2 = x.shape

    if C1 != C2:
        raise ValueError("SCM matrices must be square")

    offset = 0 if include_diagonal else 1

    idx = torch.triu_indices(
        C1,
        C2,
        offset=offset,
        device=x.device
    )

    x_flat = x[:, :, :, idx[0], idx[1]]
    # (B, W, F, N)

    return x_flat


class CNN_RNN(nn.Module):

    def __init__(
        self,
        nb_tracks=3,
        in_channels=10,          # number of SCM frequency bands
        conv_channels=[32, 64, 128, 256],
        rnn_hidden_size=256,
        rnn_layers=2,
        hidden_sizes=[512, 256],
        dropout=0.05,
        bidirectional=True,
        rnn_type="gru"
    ):
        super().__init__()

        self.nb_tracks = nb_tracks
        self.bidirectional = bidirectional

        #################################
        # CNN frontend per STFT window
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

        # Makes CNN output independent of exact SCM matrix size
        self.pool = nn.AdaptiveAvgPool1d(1)

        cnn_feature_size = conv_channels[-1]

        #################################
        # RNN backend over STFT windows
        #################################

        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=cnn_feature_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=dropout if rnn_layers > 1 else 0.0,
                bidirectional=bidirectional
            )
        elif rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=cnn_feature_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=dropout if rnn_layers > 1 else 0.0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        rnn_output_size = rnn_hidden_size * (2 if bidirectional else 1)

        #################################
        # MLP prediction head
        #################################

        layers = []

        prev_size = rnn_output_size

        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = size

        layers.append(
            nn.Linear(prev_size, nb_tracks * 3)
        )

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Expected input:
            x: (B, W, F, C, C)

        Example:
            B = batch size
            W = number of STFT windows
            F = number of SCM bands, e.g. 10
            C = number of HOA channels, e.g. 25

        Output:
            y: (B, nb_tracks, 3)
        """

        #################################
        # Flatten SCM matrices
        #################################

        x = flatten_unique_scm_temporal(x)
        # (B, W, F, N)

        B, W, F, N = x.shape

        #################################
        # Apply CNN independently per window
        #################################

        x = x.reshape(B * W, F, N)
        # (B * W, F, N)

        x = self.conv(x)
        # (B * W, conv_channels[-1], N')

        x = self.pool(x)
        # (B * W, conv_channels[-1], 1)

        x = x.squeeze(-1)
        # (B * W, conv_channels[-1])

        #################################
        # Restore temporal sequence
        #################################

        x = x.reshape(B, W, -1)
        # (B, W, cnn_feature_size)

        #################################
        # RNN over STFT windows
        #################################

        rnn_out, _ = self.rnn(x)
        # (B, W, rnn_output_size)

        # Use last time step as temporal summary
        x = rnn_out[:, -1, :]
        # (B, rnn_output_size)

        #################################
        # MLP prediction head
        #################################

        x = self.mlp(x)
        # (B, nb_tracks * 3)

        x = x.view(
            x.shape[0],
            self.nb_tracks,
            3
        )

        return x