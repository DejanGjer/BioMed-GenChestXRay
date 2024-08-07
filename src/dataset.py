import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class XRayDataset(Dataset):
    def __init__(self, img_size, image_dir: str, crop_size=128, total_images=None):
        assert img_size <= crop_size <= 1024
        self.img_size = img_size
        self.crop_size = crop_size
        self.data_root = os.path.abspath(image_dir)
        self.image_paths = glob.glob(os.path.join(self.data_root, "*.png"))
        self.image_paths = self.image_paths[:total_images]
        self.transforms = transforms.Compose(
            [
                # transforms.CenterCrop((self.crop_size, self.crop_size)),
                transforms.Grayscale(),  # TODO: make sure we set the channels based on the training config, currently hardcoding grayscale
                transforms.Resize((self.crop_size, self.crop_size)),
                transforms.ToTensor(),  # this puts the image in the CxHxW format and normalizes it to [0,1). See if we want this!
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        return self.transforms(image)
