from functools import partial
from typing import Any, Dict, List, Tuple, Optional

from torch import Tensor
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision import transforms

from src.data.data_utils import load_dicom_img
from src.data.rsna_pneumonia_detection import load_rsna_age_split, load_rsna_gender_split


class NormalDataset(Dataset):
    """
    Anomaly detection training dataset.
    Receives a list of filenames
    """

    def __init__(self, filenames: List[str], gender: List[str], transform=None):
        """
        :param filenames: Paths to training images
        :param gender:
        """
        self.filenames = filenames
        self.gender = gender
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tensor:
        img = load_dicom_img(self.filenames[idx])
        img = self.transform(img)
        gender = self.gender[idx]
        return img, gender


class AnomalDataset(Dataset):
    """
    Anomaly detection test dataset.
    Receives a list of filenames and a list of class labels (0 == normal).
    """
    def __init__(self, filenames: List[str], labels: List[int], gender: List[Dict], transform=None):
        """
        :param filenames: Paths to images
        :param labels: Class labels (0 == normal, other == anomaly)
        :param gender: Metadata (age and gender)
        """
        super().__init__()
        self.filenames = filenames
        self.labels = labels
        self.gender = gender
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = load_dicom_img(self.filenames[idx])
        img = self.transform(img)
        label = self.labels[idx]
        gender = self.gender[idx]
        return img, label, gender


class AnomalFairnessDataset(Dataset):
    """
    Anomaly detection test dataset.
    Receives a list of filenames and a list of class labels (0 == normal).
    """
    def __init__(
        self,
        filenames: Dict[str, List[str]],
        labels: Dict[str, List[int]],
        transform=None
    ):
        """
        :param filenames: Paths to images for each subgroup
        :param labels: Class labels for each subgroup (0 == normal, other == anomaly)
        """
        super().__init__()
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames[list(self.filenames.keys())[0]])

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = {k: self.transform(load_dicom_img(v[idx])) for k, v in self.filenames.items()}
        label = {k: v[idx] for k, v in self.labels.items()}
        return img, label


# default_collate does not work with Lists of dictionaries
def group_collate_fn(batch: List[Tuple[Any, ...]]):
    assert len(batch[0]) == 2
    keys = batch[0][0].keys()

    imgs = {k: default_collate([sample[0][k] for sample in batch]) for k in keys}
    labels = {k: default_collate([sample[1][k] for sample in batch]) for k in keys}

    return imgs, labels


def get_rsna_dataloaders(rsna_dir: str,
                         batch_size: int,
                         img_size: int,
                         protected_attr: str,
                         num_workers: Optional[int] = 4,
                         male_percent: Optional[float] = 0.5,
                         train_age: Optional[str] = 'avg'):
    """
    Returns dataloaders for the RSNA dataset.
    """
    # Load filenames and labels
    if protected_attr == 'age':
        filenames, labels, meta = load_rsna_age_split(rsna_dir, train_age=train_age)
    elif protected_attr == 'sex':
        filenames, labels, meta = load_rsna_gender_split(rsna_dir, male_percent=male_percent)
    else:
        raise ValueError(f'Unknown protected attribute: {protected_attr}')

    train_filenames = filenames['train']
    train_meta = meta['train']
    anomaly = 'lungOpacity'  # 'otherAnomaly'
    val_filenames = {k: v for k, v in filenames.items() if f'val/{anomaly}' in k}
    val_labels = {k: v for k, v in labels.items() if f'val/{anomaly}' in k}
    test_filenames = {k: v for k, v in filenames.items() if f'test/{anomaly}' in k}
    test_labels = {k: v for k, v in labels.items() if f'test/{anomaly}' in k}

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Normalize([0.5], [0.5])
    ])

    # Create datasets
    train_dataset = NormalDataset(train_filenames, train_meta, transform=transform)
    # anomal_ds = partial(AnomalDataset, transform=transform)
    anomal_ds = partial(AnomalFairnessDataset, transform=transform)
    val_dataset = anomal_ds(val_filenames, val_labels)
    test_dataset = anomal_ds(test_filenames, test_labels)

    # Create dataloaders
    # dl = partial(DataLoader, batch_size=batch_size, num_workers=num_workers)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # dl = partial(DataLoader, batch_size=batch_size, num_workers=num_workers, collate_fn=group_collate_fn)
    dl = partial(DataLoader, batch_size=batch_size, num_workers=num_workers, collate_fn=group_collate_fn)
    val_dataloader = dl(val_dataset, shuffle=False)
    test_dataloader = dl(test_dataset, shuffle=False)

    return (train_dataloader,
            val_dataloader,
            test_dataloader)
