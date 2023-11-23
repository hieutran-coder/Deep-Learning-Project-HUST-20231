import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class MultiLabelDataset(Dataset):
    def __init__(self, root_dir: str, label_file: str = None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = self._load_labels(label_file)


    def __len__(self):
        return len(self.labels)
    
    def _load_labels(self, label_file: str):
        if label_file is None:
            return None
        
        labels = pd.read_csv(label_file, index_col=0)
        assert list(labels.columns) == ['HG', 'HT', 'TR', 'CTH', 'BD', 'VH', 'CTQ', 'DQT', 'KS', 'CVN'], "The label format is not correct."
        return labels
    
    def __getitem__(self, idx):
        # Load image
        try:
            img_name = f'{idx + 1}.jpg'
            image = Image.open(os.path.join(self.root_dir, img_name))
        except:
            img_name = f'{idx + 1}.png'
            image = Image.open(os.path.join(self.root_dir, img_name))

        if self.transform:
            image = self.transform(image)

        # Load label
        if self.labels is not None:
            label = torch.tensor(self.labels.iloc[idx].values.astype('int'))
            return image, label, img_name
        else:
            return image, img_name