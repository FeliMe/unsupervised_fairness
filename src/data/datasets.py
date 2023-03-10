from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch import Tensor
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import transforms

from src import BRATS_DIR, CAMCAN_DIR, RSNA_DIR
from src.data.camcan_brats import (load_camcan_brats_age_split,
                                   load_camcan_only_age_split)
from src.data.data_utils import load_dicom_img
from src.data.rsna_pneumonia_detection import (load_rsna_age_split,
                                               load_rsna_gender_split,
                                               load_rsna_naive_split)


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
            load_fn: Callable = load_dicom_img):
        """
        :param filenames: Paths to training images
        :param gender:
        """
        self.data = data
        self.labels = labels
        self.meta = meta
        self.transform = transform
        self.load_fn = load_fn

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tensor:
        img = self.load_fn(self.data[idx])
        img = self.transform(img)
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
            transform=None,
            load_fn: Callable = load_dicom_img):
        """
        :param filenames: Paths to images for each subgroup
        :param labels: Class labels for each subgroup (0 == normal, other == anomaly)
        """
        super().__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        self.load_fn = load_fn

    def __len__(self) -> int:
        return len(self.data[list(self.data.keys())[0]])

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = {k: self.transform(self.load_fn(v[idx])) for k, v in self.data.items()}
        label = {k: v[idx] for k, v in self.labels.items()}
        return img, label


# default_collate does not work with Lists of dictionaries
def group_collate_fn(batch: List[Tuple[Any, ...]]):
    assert len(batch[0]) == 2
    keys = batch[0][0].keys()

    imgs = {k: default_collate([sample[0][k] for sample in batch]) for k in keys}
    labels = {k: default_collate([sample[1][k] for sample in batch]) for k in keys}

    return imgs, labels


def get_dataloaders(dataset: str,
                    batch_size: int,
                    img_size: int,
                    protected_attr: str,
                    num_workers: Optional[int] = 4,
                    male_percent: Optional[float] = 0.5,
                    train_age: Optional[str] = 'avg',
                    supervised: Optional[bool] = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns dataloaders for the RSNA dataset.
    """
    # Load filenames and labels
    if dataset == 'rsna':
        load_fn = load_dicom_img
        if protected_attr == 'none':
            data, labels, meta = load_rsna_naive_split(RSNA_DIR, for_supervised=supervised)
        elif protected_attr == 'age':
            data, labels, meta = load_rsna_age_split(RSNA_DIR, train_age=train_age, for_supervised=supervised)
        elif protected_attr == 'sex':
            data, labels, meta = load_rsna_gender_split(RSNA_DIR, male_percent=male_percent, for_supervised=supervised)
        else:
            raise ValueError(f'Unknown protected attribute: {protected_attr} for dataset {dataset}')
    elif dataset == 'camcan/brats':
        def load_fn(x):
            return x
        if protected_attr == 'none':
            raise NotImplementedError
            # filenames, labels, meta = load_camcan_naive_split(CAMCAN_DIR)
        elif protected_attr == 'age':
            data, labels, meta = load_camcan_brats_age_split(CAMCAN_DIR, BRATS_DIR,
                                                             sequence='T2',
                                                             train_age=train_age,
                                                             slice_range=(73, 103))
        else:
            raise ValueError(f'Unknown protected attribute: {protected_attr} for dataset {dataset}')
    elif dataset == 'camcan':
        def load_fn(x):
            return x
        if protected_attr == 'none':
            raise NotImplementedError
            # filenames, labels, meta = load_camcan_naive_split(CAMCAN_DIR)
        elif protected_attr == 'age':
            data, labels, meta = load_camcan_only_age_split(CAMCAN_DIR,
                                                            sequence='T2',
                                                            train_age=train_age,
                                                            slice_range=(73, 103))
        else:
            raise ValueError(f'Unknown protected attribute: {protected_attr} for dataset {dataset}')
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    train_data = data['train']
    train_labels = labels['train']
    train_meta = meta['train']
    val_data = {k: v for k, v in data.items() if 'val' in k}
    val_labels = {k: v for k, v in labels.items() if 'val' in k}
    test_data = {k: v for k, v in data.items() if 'test' in k}
    test_labels = {k: v for k, v in labels.items() if 'test' in k}

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Normalize([0.5], [0.5])
    ])

    # Create datasets
    train_dataset = NormalDataset(train_data, train_labels, train_meta, transform=transform, load_fn=load_fn)
    anomal_ds = partial(AnomalFairnessDataset, transform=transform, load_fn=load_fn)
    val_dataset = anomal_ds(val_data, val_labels)
    test_dataset = anomal_ds(test_data, test_labels)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl = partial(DataLoader, batch_size=batch_size, num_workers=num_workers, collate_fn=group_collate_fn)
    val_dataloader = dl(val_dataset, shuffle=False)
    test_dataloader = dl(test_dataset, shuffle=False)

    return (train_dataloader,
            val_dataloader,
            test_dataloader)
