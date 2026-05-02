import torch
from torch.utils.data import Dataset
from PIL import Image

class SpectrogramDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

    def __getitem__(self, idx):
        item = self.paths[idx]
        img = Image.open(item).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.paths)