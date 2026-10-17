import torch
import torch.nn as nn
import torch.nn.functional as F

class SELD_CRNN(nn.Module):
    def __init__(self, input_channels=7, max_sources=3):
        super(SELD_CRNN, self).__init__()
        self.max_sources = max_sources
        
        # -----------------------
        # CNN feature extractor
        # -----------------------
        # Conv block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2,2))
        self.drop1 = nn.Dropout(0.2)
        
        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2,2))
        self.drop2 = nn.Dropout(0.2)
        
        # Conv block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2,2))
        self.drop3 = nn.Dropout(0.2)
        
        # FC Output: max_sources * 3 (ACCDOA x,y,z)
        self.fc = nn.Linear(128*3*3, max_sources * 3)
        
    def forward(self, x):
        """
        x: tensor (batch, 7, 25, 25)
        """
        # -----------------------
        # CNN feature extractor
        # -----------------------
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        # -----------------------
        # Flatten and fully connected
        # -----------------------
        b, c, h, w = x.shape
        x = x.view(b, c * h * w)  # flatten: (batch, channels*H*W)
        
        # FC output
        out = self.fc(x)  # (batch, max_sources*3)
        out = out.view(b, self.max_sources, 3)  # reshape: (batch, sources, xyz)
          
        return out


# ==============================
# Example usage
# ==============================

#cov_mat = torch.rand(batch_size, 7, 25, 25)  # random covariance matrices
data = torch.load("../data/harp/Generated_HOA_IRs_ISM/harp_dataset.pt", weights_only=False)
cov_mat = data["X"] # shape: (batch, 7, 25, 25)
print("Covariance matrix shape:", cov_mat.shape)  # (batch, 7, 25, 25)

model = SELD_CRNN(input_channels=7, max_sources=3)

accdoa_out = model(cov_mat)
print("Output shape:", accdoa_out.shape)  # (batch, 3, 3)
print("Sample output (first batch):\n", accdoa_out[0])
