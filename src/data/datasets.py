from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Generator, Tensor
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import transforms

from src import CHEXPERT_DIR, CXR14_DIR, MIMIC_CXR_DIR
from src.data.chexpert import (load_chexpert_age_split,
                               load_chexpert_naive_split,
                               load_chexpert_race_split,
                               load_chexpert_sex_split)
from src.data.cxr14 import (load_cxr14_age_split,
                            load_cxr14_naive_split,
                            load_cxr14_sex_split)
from src.data.mimic_cxr import (load_mimic_cxr_age_split,
                                load_mimic_cxr_intersectional_age_sex_race_split,
                                load_mimic_cxr_naive_split,
                                load_mimic_cxr_race_split,
                                load_mimic_cxr_sex_split)


class NormalDataset(Dataset):
    """
    Anomaly detection training dataset.
    Receives a list of filenames
    """

    def __init__(
            self,
            data: List[str],
            labels: List[int],
            meta: List[int],
            transform=None,
            index_mapping: Optional[List[int]] = None,
            load_fn: Callable = lambda x: x):
        """
        :param filenames: Paths to training images
        :param labels: Class labels (0 == normal, other == anomaly)
        :param meta: Metadata (such as age or sex labels)
        :param transform: Transformations to apply to images
        :param index_mapping: Mapping from indices to data
        """
        self.data = data
        self.labels = labels
        self.meta = meta
        self.load_fn = load_fn
        self.transform = transform
        self.index_mapping = index_mapping

        if self.index_mapping is None:
            self.index_mapping = torch.arange(len(self.data))

        for i, d in enumerate(self.data):
            if isinstance(d, str):
                self.data[i] = transform(load_fn(d))
                self.load_fn = lambda x: x
                self.transform = lambda x: x

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tensor:
        img = self.transform(self.load_fn(self.data[self.index_mapping[idx]]))
        label = self.labels[idx]
        meta = self.meta[idx]
        return img, label, meta


class AnomalFairnessDataset(Dataset):
    """
    Anomaly detection test dataset.
    Receives a list of filenames and a list of class labels (0 == normal).
    """
    def __init__(
            self,
            data: Dict[str, List[str]],
            labels: Dict[str, List[int]],
            meta: Dict[str, List[int]],
            transform=None,
            index_mapping: Optional[Dict[str, List[int]]] = None,
            load_fn: Callable = lambda x: x):
        """
        :param filenames: Paths to images for each subgroup
        :param labels: Class labels for each subgroup (0 == normal, other == anomaly)
        """
        super().__init__()
        self.data = data
        self.labels = labels
        self.meta = meta
        self.transform = transform
        self.load_fn = load_fn
        self.index_mapping = index_mapping

        if self.index_mapping is None:
            self.index_mapping = {mode: torch.arange(len(self.data)) for mode in self.data.keys()}

    def __len__(self) -> int:
        return min([len(v) for v in self.labels.values()])

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = {k: self.transform(self.load_fn(v[self.index_mapping[k][idx]])) for k, v in self.data.items()}
        label = {k: v[idx] for k, v in self.labels.items()}
        meta = {k: v[idx] for k, v in self.meta.items()}
        return img, label, meta


# default_collate does not work with Lists of dictionaries
def group_collate_fn(batch: List[Tuple[Any, ...]]):
    assert len(batch[0]) == 3
    keys = batch[0][0].keys()

    imgs = {k: default_collate([sample[0][k] for sample in batch]) for k in keys}
    labels = {k: default_collate([sample[1][k] for sample in batch]) for k in keys}
    meta = {k: default_collate([sample[2][k] for sample in batch]) for k in keys}

    return imgs, labels, meta


def get_dataloaders(dataset: str,
                    batch_size: int,
                    img_size: int,
                    protected_attr: str,
                    num_workers: Optional[int] = 4,
                    male_percent: Optional[float] = 0.5,
                    old_percent: Optional[float] = 0.5,
                    white_percent: Optional[float] = 0.5,
                    max_train_samples: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns dataloaders for the desired dataset.
    """
    print(f'Loading dataset {dataset} with protected attribute {protected_attr}')

    # Load filenames and labels
    if dataset == 'cxr14':
        def load_fn(x):
            return torch.tensor(x)
        if protected_attr == 'none':
            data, labels, meta, idx_map = load_cxr14_naive_split()
        elif protected_attr == 'sex':
            data, labels, meta, idx_map = load_cxr14_sex_split(
                cxr14_dir=CXR14_DIR,
                male_percent=male_percent)
        elif protected_attr == 'age':
            data, labels, meta, idx_map = load_cxr14_age_split(
                cxr14_dir=CXR14_DIR,
                old_percent=old_percent)
        else:
            raise NotImplementedError
    elif dataset == 'mimic-cxr':
        def load_fn(x):
            return torch.tensor(x)
        if protected_attr == 'none':
            data, labels, meta, idx_map = load_mimic_cxr_naive_split(
                max_train_samples=max_train_samples)
        elif protected_attr == 'sex':
            data, labels, meta, idx_map = load_mimic_cxr_sex_split(
                mimic_cxr_dir=MIMIC_CXR_DIR,
                male_percent=male_percent,
                max_train_samples=max_train_samples)
        elif protected_attr == 'age':
            data, labels, meta, idx_map = load_mimic_cxr_age_split(
                mimic_cxr_dir=MIMIC_CXR_DIR,
                old_percent=old_percent,
                max_train_samples=max_train_samples)
        elif protected_attr == 'race':
            data, labels, meta, idx_map = load_mimic_cxr_race_split(
                mimic_cxr_dir=MIMIC_CXR_DIR,
                white_percent=white_percent,
                max_train_samples=max_train_samples)
        elif protected_attr == 'intersectional_age_sex_race':
            data, labels, meta, idx_map = load_mimic_cxr_intersectional_age_sex_race_split(
                mimic_cxr_dir=MIMIC_CXR_DIR)
        else:
            raise ValueError(f'Unknown protected attribute: {protected_attr} for dataset {dataset}')
    elif dataset == 'chexpert':
        def load_fn(x):
            return torch.tensor(x)
        if protected_attr == 'none':
            data, labels, meta, idx_map = load_chexpert_naive_split(
                chexpert_dir=CHEXPERT_DIR,
                max_train_samples=max_train_samples)
        elif protected_attr == 'sex':
            data, labels, meta, idx_map = load_chexpert_sex_split(
                chexpert_dir=CHEXPERT_DIR,
                male_percent=male_percent,
                max_train_samples=max_train_samples)
        elif protected_attr == 'age':
            data, labels, meta, idx_map = load_chexpert_age_split(
                chexpert_dir=CHEXPERT_DIR,
                old_percent=old_percent,
                max_train_samples=max_train_samples)
        elif protected_attr == 'race':
            data, labels, meta, idx_map = load_chexpert_race_split(
                chexpert_dir=CHEXPERT_DIR,
                white_percent=white_percent,
                max_train_samples=max_train_samples)
        else:
            raise ValueError(f'Unknown protected attribute: {protected_attr} for dataset {dataset}')
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    train_data = data['train']
    train_labels = labels['train']
    train_meta = meta['train']
    train_idx_map = idx_map['train']
    val_data = {k: v for k, v in data.items() if 'val' in k}
    val_labels = {k: v for k, v in labels.items() if 'val' in k}
    val_meta = {k: v for k, v in meta.items() if 'val' in k}
    val_idx_map = {k: v for k, v in idx_map.items() if 'val' in k}
    test_data = {k: v for k, v in data.items() if 'test' in k}
    test_labels = {k: v for k, v in labels.items() if 'test' in k}
    test_meta = {k: v for k, v in meta.items() if 'test' in k}
    test_idx_map = {k: v for k, v in idx_map.items() if 'test' in k}

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=False),
        transforms.Normalize([0.5], [0.5])
    ])

    # Create datasets
    train_dataset = NormalDataset(
        train_data,
        train_labels,
        train_meta,
        transform=transform,
        index_mapping=train_idx_map,
        load_fn=load_fn)
    anomal_ds = partial(AnomalFairnessDataset, transform=transform, load_fn=load_fn)
    val_dataset = anomal_ds(val_data, val_labels, val_meta, index_mapping=val_idx_map)
    test_dataset = anomal_ds(test_data, test_labels, test_meta, index_mapping=test_idx_map)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=Generator().manual_seed(2147483647),
        pin_memory=True)
    dl = partial(
        DataLoader,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=group_collate_fn,
        pin_memory=True)
    val_dataloader = dl(
        val_dataset,
        shuffle=False,
        generator=Generator().manual_seed(2147483647))
    test_dataloader = dl(
        test_dataset,
        shuffle=False,
        generator=Generator().manual_seed(2147483647))

    return (train_dataloader,
            val_dataloader,
            test_dataloader)
