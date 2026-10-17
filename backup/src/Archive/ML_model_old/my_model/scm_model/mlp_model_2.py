import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################
# Multi-Layer Perceptron for SCM
#########################################
class SimpleSeldMLP(nn.Module):
    def __init__(self, in_feat_shape, params):
        super().__init__()
        # Flattened input dimension
        in_features = in_feat_shape[0] * in_feat_shape[1] * in_feat_shape[2]

        # MLP hyperparameters
        hidden_sizes = params.get('hidden_sizes', [512, 256, 128])
        dropout = params.get('dropout_rate', 0.1)
        output_dim = params.get('output_dim', 3)  # predict x, y, z

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
        # Flatten the input: (B, C, H, W) -> (B, C*H*W)
        x = x.view(x.size(0), -1)
        out = self.mlp(x)
        out = out.unsqueeze(1)  # (B, 1, 3) to match previous interface
        return out