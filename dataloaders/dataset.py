import os
import pandas as pd

from PIL import Image
from typing import List

import torch
import timm

from torch.utils.data import Dataset, ConcatDataset
from timm.data import rand_augment_transform, RandomResizedCropAndInterpolation
from torchvision import transforms


class MultiLabelDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = self._load_labels()
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, "images")))

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir, "images")))
    
    def _load_labels(self):
        try:
            labels = pd.read_csv(os.path.join(self.root_dir, "labels.csv"), index_col=0)
            assert list(labels.columns) == ['HG', 'HT', 'TR', 'CTH', 'BD', 'VH', 'CTQ', 'DQT', 'KS', 'CVN'], "The label format is not correct."
            return labels
        except:
            return None
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.img_names[idx]
        image = Image.open(os.path.join(self.root_dir, f"images/{img_name}"))
        # try:
        #     img_name = f'{idx + 1}.jpg'
        #     image = Image.open(os.path.join(self.root_dir, f"images/{img_name}"))
        # except:
        #     try:
        #         img_name = f'{idx + 1}.png'
        #         image = Image.open(os.path.join(self.root_dir, f"images/{img_name}"))
        #     except:
        #         img_name = f'{idx + 1}.jpeg'
        #         image = Image.open(os.path.join(self.root_dir, f"images/{img_name}"))

        if self.transform:
            image = self.transform(image)

        # Load label
        if self.labels is not None:
            label = torch.tensor(self.labels.iloc[idx].values.astype('int')).float()
            return image, label, img_name
        else:
            return image, img_name
        

def create_dataset(root_dir: List[str], is_train: bool = False):
    datasets = []
    
    # Augmentation
    rand_augment = rand_augment_transform("rand-m9-n3-mstd0.5", hparams=dict())
    train_transform = transforms.Compose([
        RandomResizedCropAndInterpolation(224),
        rand_augment,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    for i, dir in enumerate(root_dir):
        if is_train:
            datasets.append(MultiLabelDataset(dir, transform=train_transform))
        else:
            datasets.append(MultiLabelDataset(dir, transform=test_transform))

    return ConcatDataset(datasets)