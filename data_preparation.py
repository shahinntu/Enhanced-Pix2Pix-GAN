import os
import glob

from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self._image_paths = glob.glob(os.path.join(root, "*.jpg"))
        self._transform = transform

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image_path = self._image_paths[idx]
        image = Image.open(image_path)

        width, height = image.size
        condition = image.crop((0, 0, width // 2, height))
        real = image.crop((width // 2, 0, width, height))

        if self._transform:
            condition = self._transform(condition)
            real = self._transform(real)

        return condition, real
