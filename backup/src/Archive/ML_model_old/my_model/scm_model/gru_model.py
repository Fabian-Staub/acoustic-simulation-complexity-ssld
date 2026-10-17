import torch
import torch.nn as nn
import itertools

#########################################
# GRU-based SELD model
#########################################
class SeldGRU(nn.Module):
    def __init__(
        self,
        input_size,        # = num_scm_channels
        hidden_size=128,
        num_layers=1,
        nb_tracks=3,
        dropout=0.1,
        bidirectional=True
    ):
        super().__init__()

        self.nb_tracks = nb_tracks
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        gru_out_size = hidden_size * (2 if bidirectional else 1)

        self.attention = nn.Sequential(
            nn.Linear(gru_out_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.norm = nn.LayerNorm(gru_out_size)

        # Final regression head
        self.fc = nn.Sequential(
            nn.Linear(gru_out_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, nb_tracks * 3)
        )

    def forward(self, x):
        """
        x: (B, num_scm_channels, time_steps)
        """

        # → (B, time_steps, features)
        x = x.transpose(1, 2)

        # GRU
        '''
        out, _ = self.gru(x)

        out = out[:, -1, :]

        out = self.norm(out)

        out = self.fc(out)
        out = out.view(x.shape[0], self.nb_tracks, 3)
        '''

        out, _ = self.gru(x)   # (B, T, H)

        attn_scores = self.attention(out)              
        attn_weights = torch.softmax(attn_scores, dim=1)

        out = torch.sum(out * attn_weights, dim=1)    # (B, H)

        out = self.fc(out)
        out = out.view(x.shape[0], self.nb_tracks, 3)

        return out