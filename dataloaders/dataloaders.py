import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import transforms
from functools import partial

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
    is_train: bool = True,
):
    if is_train:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=partial(collate_fn, is_train=True),
            drop_last=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=partial(collate_fn, is_train=False),
        )

    return dataloader
    


def collate_fn(batch, is_train: bool = True):
    if is_train:
        images, labels, img_names = zip(*batch)
        images = torch.stack(images, dim=0)
        img_names = list(img_names)
        labels = torch.stack(labels, dim=0)
        return images, labels, img_names
    else:
        images, img_names = zip(*batch)
        images = torch.stack(images, dim=0)
        img_names = list(img_names)
        return images, img_names