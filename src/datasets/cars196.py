from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import time
from PIL import Image
import os
import glob
from scipy.io import loadmat

class TrainDataset(Dataset):
    def __init__(self, data_dir):
        self.image_path = os.path.join(data_dir, "cars196", "cars_train")

        # Get all image paths
        self.images = sorted(glob.glob(os.path.join(self.image_path, "*", "*.jpg")))

        # Use parent folder name as class label
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(self.image_path)))
        }
        self.labels = [
            self.class_to_idx[os.path.basename(os.path.dirname(path))] for path in self.images
        ]

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(227, scale=(0.08, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]).convert("RGB"))
        label = self.labels[idx]
        return image, label


class TestDataset(Dataset):
    def __init__(self, data_dir):
        self.image_path = os.path.join(data_dir, "cars196", "cars_test")

        # Get all image paths recursively
        self.images = sorted(glob.glob(os.path.join(self.image_path, "*", "*.jpg")))

        # Map class folder name → label index
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(self.image_path)))
        }

        # Assign labels by class folder name
        self.labels = [
            self.class_to_idx[os.path.basename(os.path.dirname(path))] for path in self.images
        ]

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]).convert("RGB"))
        label = self.labels[idx]
        return image, label
