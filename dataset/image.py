import torch
import torchvision
from pathlib import Path
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, path, image_size, size=None):
        super().__init__()
        self.paths = [p for p in Path(path).iterdir() if p.name.endswith('.png')]
        self.size = min(size if size is not None else len(self.paths), len(self.paths))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return {
            'image': self.transform(img) * 2 - 1
        }
