import torch
from torch.utils.data import Dataset

class HarpDataset(Dataset):
    def __init__(self, data_path, indices=None):
        """
        data_path: Pfad zur .pt Datei, die ein dict {'X':..., 'Y':...} enthält
                   X: [num_samples, 7, 25, 25]  -> SCMs
                   Y: [num_samples, 3]          -> x,y,z positions
        indices: optionale Liste von Indizes, um nur einen Teil der Daten zu laden
        """
        data = torch.load(data_path)
        self.X = data['X'].float()  # [num_samples, 7, 25, 25]
        self.Y = data['Y'].float()  # [num_samples, 3]

        self.Y = self.Y.unsqueeze(1)  # [num_samples, 1, 3] für Konsistenz mit vorherigem Interface

        if indices is not None:
            self.X = self.X[indices]
            self.Y = self.Y[indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]