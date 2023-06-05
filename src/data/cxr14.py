"""
Split CXR14 dataset.
train_ad: 40000 no finding samples from train_val_list.txt
train_cls: 90% of the remaining samples from train_val_list.txt
val_cls: 10% of the remaining samples from train_val_list.txt
test_cls: all samples from test_list.txt
"""
import os
from functools import partial
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

from src import CXR14_DIR
from src.data.data_utils import write_hf5_file

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SEX_MAPPING = {
    'M': 0,
    'F': 1
}

CXR14LABELS = [  # All data
    'No Finding',  # 60361
    'Atelectasis',  # 11559
    'Cardiomegaly',  # 2776
    'Consolidation',  # 4667
    'Edema',  # 2303
    'Effusion',  # 13317
    'Emphysema',  # 2516
    'Fibrosis',  # 1686
    'Hernia',  # 227
    'Infiltration',  # 19894
    'Mass',  # 5782
    'Nodule',  # 6331
    'Pleural_Thickening',  # 3385
    'Pneumonia',  # 1431
    'Pneumothorax',  # 5302
]


def prepare_cxr14(cxr14_dir: str = CXR14_DIR):
    """Loads metadata (filenames, labels, age, gender) for each sample of the
    CXR14 dataset."""
    metadata = pd.read_csv(os.path.join(cxr14_dir, 'Data_Entry_2017.csv'))
    print(f"Total number of images: {len(metadata)}")

    # We only consider frontal view images. (AP)
    metadata = metadata[metadata['View Position'] == 'AP']
    print(f"Number of frontal view images: {len(metadata)}")

    # Prepend the path to the image filename
    metadata["path"] = metadata.apply(
        lambda row: os.path.join(
            cxr14_dir,
            "images",
            row['Image Index']
        ), axis=1
    )

    # Reset index
    metadata = metadata.reset_index(drop=True)

    # Save ordering of files in a new column 'hf5_idx'
    metadata['hf5_idx'] = np.arange(len(metadata))

    hf5_dir = os.path.join(cxr14_dir, 'hf5')
    os.makedirs(hf5_dir, exist_ok=True)

    # Select sets of all pathologies
    pathologies = {}
    for i, pathology in enumerate(CXR14LABELS):
        # Filter all samples where pathology is in metadata['Finding Labels']
        pathologies[pathology] = metadata[metadata['Finding Labels'].str.contains(pathology)]
        print(f"Number of images for '{pathology}': {len(pathologies[pathology])}")

        # Add labels
        pathologies[pathology]['label'] = [i] * len(pathologies[pathology])

        # Save files
        os.makedirs(os.path.join(THIS_DIR, 'csvs/cxr14'), exist_ok=True)
        pathologies[pathology].to_csv(os.path.join(THIS_DIR, 'csvs/cxr14', f'{pathology}.csv'), index=True)

    # Write hf5 files for whole dataset
    hf5_file = os.path.join(hf5_dir, 'cxr14_ap_only.hf5')
    print(f"Writing hf5 file '{hf5_file}'...")
    write_hf5_file(
        metadata['path'].values.tolist(),
        hf5_file,
        load_fn=partial(load_and_resize, target_size=(256, 256)),
        target_size=(256, 256)
    )


def load_and_resize(path: str, target_size: Tuple[int, int]):
    image = Image.open(path).convert('L')
    image = transforms.CenterCrop(min(image.size))(image)
    image = transforms.Resize(target_size)(image)
    image = transforms.ToTensor()(image)
    return image


def load_cxr14_naive_split(cxr14_dir: str = CXR14_DIR):
    normal = pd.read_csv(os.path.join(THIS_DIR, 'csvs/cxr14', 'No Finding.csv'))
    abnormal = pd.concat([
        pd.read_csv(os.path.join(THIS_DIR, 'csvs/cxr14', f'{label}.csv'))
        for label in CXR14LABELS if label != 'No Finding'
    ]).sample(frac=1, random_state=42)

    # Split normal images into train, val, test (use 1000 for val and test)
    val_test_normal = normal.sample(n=2000, random_state=42)
    train = normal[~normal['path'].isin(val_test_normal['path'])]
    val_normal = val_test_normal[:1000]
    test_normal = val_test_normal[1000:]

    # Split abnormal images into val, test (use maximum 1000 for val and test)
    val_test_abnormal = abnormal.sample(n=2000, random_state=42)
    val_abnormal = val_test_abnormal[:1000]
    test_abnormal = val_test_abnormal[1000:]

    # Aggregate validation and test sets and shuffle
    val = pd.concat([val_normal, val_abnormal]).sample(frac=1, random_state=42)
    test = pd.concat([test_normal, test_abnormal]).sample(frac=1, random_state=42)

    hf5_file = h5py.File(
        os.path.join(
            cxr14_dir,
            'hf5',
            'cxr14_ap_only.hf5'),
        'r')['images']

    # Return
    filenames = {}
    labels = {}
    meta = {}
    index_mapping = {}
    sets = {
        'train': train,
        'val': val,
        'test': test,
    }
    for mode, data in sets.items():
        filenames[mode] = hf5_file
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.zeros(len(data), dtype=np.float32)
        index_mapping[mode] = data.hf5_idx.values
    return filenames, labels, meta, index_mapping


def load_cxr14_sex_split(cxr14_dir: str = CXR14_DIR,
                         male_percent: float = 0.5):
    """Load data with sex-balanced val and test sets."""
    assert 0.0 <= male_percent <= 1.0
    female_percent = 1 - male_percent

    normal = pd.read_csv(os.path.join(THIS_DIR, 'csvs/cxr14', 'No Finding.csv'))
    abnormal = pd.concat([
        pd.read_csv(os.path.join(THIS_DIR, 'csvs/cxr14', f'{label}.csv'))
        for label in CXR14LABELS if label != 'No Finding'
    ]).sample(frac=1, random_state=42)

    # Split normal images into train, val, test (use 500 for val and test)
    normal_male = normal[normal['Patient Gender'] == 'M']
    normal_female = normal[normal['Patient Gender'] == 'F']
    val_test_normal_male = normal_male.sample(n=1000, random_state=42)
    val_test_normal_female = normal_female.sample(n=1000, random_state=42)
    val_normal_male = val_test_normal_male[:500]
    val_normal_female = val_test_normal_female[:500]
    test_normal_male = val_test_normal_male[500:]
    test_normal_female = val_test_normal_female[500:]

    # Split abnormal images into val, test (use maximum 500 for val and test)
    abnormal_male = abnormal[abnormal['Patient Gender'] == 'M']
    abnormal_female = abnormal[abnormal['Patient Gender'] == 'F']
    val_test_abnormal_male = abnormal_male.sample(n=1000, random_state=42)
    val_test_abnormal_female = abnormal_female.sample(n=1000, random_state=42)
    val_abnormal_male = val_test_abnormal_male[:500]
    val_abnormal_female = val_test_abnormal_female[:500]
    test_abnormal_male = val_test_abnormal_male[500:]
    test_abnormal_female = val_test_abnormal_female[500:]

    # Aggregate validation and test sets and shuffle
    val_male = pd.concat([val_normal_male, val_abnormal_male]).sample(frac=1, random_state=42)
    val_female = pd.concat([val_normal_female, val_abnormal_female]).sample(frac=1, random_state=42)
    test_male = pd.concat([test_normal_male, test_abnormal_male]).sample(frac=1, random_state=42)
    test_female = pd.concat([test_normal_female, test_abnormal_female]).sample(frac=1, random_state=42)

    # Rest for training
    rest_normal_male = normal_male[~normal_male['Patient ID'].isin(val_test_normal_male['Patient ID'])]
    rest_normal_female = normal_female[~normal_female['Patient ID'].isin(val_test_normal_female['Patient ID'])]
    n_samples = min(len(rest_normal_male), len(rest_normal_female))
    n_male = int(n_samples * male_percent)
    n_female = int(n_samples * female_percent)
    train_male = rest_normal_male.sample(n=n_male, random_state=42)
    train_female = rest_normal_female.sample(n=n_female, random_state=42)

    # Aggregate training set and shuffle
    train = pd.concat([train_male, train_female]).sample(frac=1, random_state=42)
    print(f"Using {n_male} male and {n_female} female samples for training.")

    hf5_file = h5py.File(
        os.path.join(
            cxr14_dir,
            'hf5',
            'cxr14_ap_only.hf5'),
        'r')['images']

    # Return
    filenames = {}
    labels = {}
    meta = {}
    index_mapping = {}
    sets = {
        'train': train,
        'val/male': val_male,
        'val/female': val_female,
        'test/male': test_male,
        'test/female': test_female,
    }
    for mode, data in sets.items():
        filenames[mode] = hf5_file
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.zeros(len(data), dtype=np.float32)
        index_mapping[mode] = data.hf5_idx.values
    return filenames, labels, meta, index_mapping


def load_cxr14_age_split(cxr14_dir: str = CXR14_DIR,
                         old_percent: float = 0.5):
    """Load data with age-balanced val and test sets."""
    assert 0.0 <= old_percent <= 1.0
    young_percent = 1 - old_percent

    normal = pd.read_csv(os.path.join(THIS_DIR, 'csvs/cxr14', 'No Finding.csv'))
    abnormal = pd.concat([
        pd.read_csv(os.path.join(THIS_DIR, 'csvs/cxr14', f'{label}.csv'))
        for label in CXR14LABELS if label != 'No Finding'
    ]).sample(frac=1, random_state=42)

    # Filter ages over 100 years
    normal = normal[normal['Patient Age'] < 100]
    abnormal = abnormal[abnormal['Patient Age'] < 100]

    # Split data into bins by age
    # n_bins = 3
    # t = np.histogram(normal['Patient Age'], bins=n_bins)[1]
    # print(f"Splitting data into {n_bins - 1} bins by age: {t}")

    max_young = 31  # 31  # 41
    min_old = 61  # 61  # 66
    normal_young = normal[normal['Patient Age'] <= max_young]
    normal_old = normal[normal['Patient Age'] >= min_old]
    abnormal_young = abnormal[abnormal['Patient Age'] <= max_young]
    abnormal_old = abnormal[abnormal['Patient Age'] >= min_old]

    # Split normal images into train, val, test (use 500 for val and test)
    val_test_normal_old = normal_old.sample(n=1000, random_state=42)
    val_test_normal_young = normal_young.sample(n=1000, random_state=42)
    val_normal_old = val_test_normal_old[:500]
    val_normal_young = val_test_normal_young[:500]
    test_normal_old = val_test_normal_old[500:]
    test_normal_young = val_test_normal_young[500:]

    # Split abnormal images into val, test (use maximum 500 for val and test)
    val_test_abnormal_old = abnormal_old.sample(n=1000, random_state=42)
    val_test_abnormal_young = abnormal_young.sample(n=1000, random_state=42)
    val_abnormal_old = val_test_abnormal_old[:500]
    val_abnormal_young = val_test_abnormal_young[:500]
    test_abnormal_old = val_test_abnormal_old[500:]
    test_abnormal_young = val_test_abnormal_young[500:]

    # Aggregate validation and test sets and shuffle
    val_old = pd.concat([val_normal_old, val_abnormal_old]).sample(frac=1, random_state=42)
    val_young = pd.concat([val_normal_young, val_abnormal_young]).sample(frac=1, random_state=42)
    test_old = pd.concat([test_normal_old, test_abnormal_old]).sample(frac=1, random_state=42)
    test_young = pd.concat([test_normal_young, test_abnormal_young]).sample(frac=1, random_state=42)

    # Rest for training
    rest_normal_old = normal_old[~normal_old['Patient ID'].isin(val_test_normal_old['Patient ID'])]
    rest_normal_young = normal_young[~normal_young['Patient ID'].isin(val_test_normal_young['Patient ID'])]
    n_samples = min(len(rest_normal_old), len(rest_normal_young))
    n_old = int(n_samples * old_percent)
    n_young = int(n_samples * young_percent)
    train_old = rest_normal_old.sample(n=n_old, random_state=42)
    train_young = rest_normal_young.sample(n=n_young, random_state=42)

    # Aggregate training set and shuffle
    train = pd.concat([train_old, train_young]).sample(frac=1, random_state=42)
    print(f"Using {n_old} old and {n_young} young samples for training.")

    hf5_file = h5py.File(
        os.path.join(
            cxr14_dir,
            'hf5',
            'cxr14_ap_only.hf5'),
        'r')['images']

    # Return
    filenames = {}
    labels = {}
    meta = {}
    index_mapping = {}
    sets = {
        'train': train,
        'val/old': val_old,
        'val/young': val_young,
        'test/old': test_old,
        'test/young': test_young,
    }
    for mode, data in sets.items():
        filenames[mode] = hf5_file
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.zeros(len(data), dtype=np.float32)
        index_mapping[mode] = data.hf5_idx.values
    return filenames, labels, meta, index_mapping


if __name__ == '__main__':
    # prepare_cxr14()
    pass
