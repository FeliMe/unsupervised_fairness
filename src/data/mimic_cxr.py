"""Prepare MIMIC-CXR dataset for anomaly detection.

We only consider frontal view images.
Images with CheXpert label "No Finding" are considered normal, all others anomal.
We exlude images with CheXpert label "Uncertain" or "Support Devices".

We use the mimic-cxr-jpg_2-0-0 version. It has the following structure:
files:
    p10:
        p<subject_id>:
            s<study_id>:
                <dicom_id>.jpg
                ...
            ...
        ...
    ...
    p19:
        ...
mimic-cxr-2.0.0-metadata.csv
mimic-cxr-2.0.0-chexpert.csv
"""
import os
from functools import partial
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

from src import MIMIC_CXR_DIR
from src.data.data_utils import write_hf5_file


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SEX_MAPPING = {
    'M': 0,
    'F': 1
}


CHEXPERT_LABELS = [
    'No Finding',
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Enlarged Cardiomediastinum',
    'Fracture',
    'Lung Lesion',
    'Lung Opacity',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Pneumothorax',
]


def prepare_mimic_cxr(mimic_dir: str = MIMIC_CXR_DIR):
    metadata = pd.read_csv(os.path.join(mimic_dir, 'mimic-cxr-2.0.0-metadata.csv'))
    chexpert = pd.read_csv(os.path.join(mimic_dir, 'mimic-cxr-2.0.0-chexpert.csv'))
    mimic_sex = pd.read_csv(os.path.join(mimic_dir, 'patients.csv'))  # From MIMIC-IV, v2.2
    print(f"Total number of images: {len(metadata)}")

    # We only consider frontal view images. (AP)
    metadata = metadata[metadata['ViewPosition'] == 'AP']
    print(f"Number of frontal view images: {len(metadata)}")

    # Add sex information to metadata
    metadata = metadata.merge(mimic_sex, on='subject_id')
    print(f'Number of images with age and sex metadata: {len(metadata)}')

    # Match metadata and chexpert.
    metadata = metadata.merge(chexpert, on=['subject_id', 'study_id'])
    print(f"Number of images with CheXpert labels: {len(metadata)}")

    # Exclude images with support devices. 'Support Devices' is NaN
    metadata = metadata[metadata['Support Devices'].isna()]
    print(f"Number of images without support devices: {len(metadata)}")

    # Exclude images with uncertain labels. 'Uncertain' means no 1.0 or 0.0 in any label
    metadata = metadata[metadata[CHEXPERT_LABELS].isin([0.0, 1.0]).any(axis=1)]
    metadata[CHEXPERT_LABELS] = metadata[CHEXPERT_LABELS].replace(-1.0, float('nan'))
    print(f"Number of images with certain labels: {len(metadata)}\n")

    # Add absolute path to images
    metadata['path'] = metadata.apply(
        lambda row: os.path.join(
            mimic_dir,
            f'files/p{str(row.subject_id)[:2]}',
            f'p{row.subject_id}',
            f's{row.study_id}',
            f'{row.dicom_id}.jpg'),
        axis=1
    )

    # Reset index
    metadata = metadata.reset_index(drop=True)

    # Save ordering of files in a new column 'hf5_idx'
    metadata['hf5_idx'] = np.arange(len(metadata))

    hf5_dir = os.path.join(mimic_dir, 'hf5')
    os.makedirs(hf5_dir, exist_ok=True)

    # Select sets of all pathologies
    pathologies = {}
    for i, pathology in enumerate(CHEXPERT_LABELS):
        pathologies[pathology] = metadata[metadata[pathology] == 1.0]
        print(f"Number of images for '{pathology}': {len(pathologies[pathology])}")
        print(f"Number of male patients for '{pathology}': "
              f"{len(pathologies[pathology][pathologies[pathology]['gender'] == 'M' ])}")
        print(f"Number of female patients for '{pathology}': "
              f"{len(pathologies[pathology][pathologies[pathology]['gender'] == 'F' ])}")

        # Add labels
        pathologies[pathology]['label'] = [i] * len(pathologies[pathology])

        # Save files
        pathologies[pathology].to_csv(os.path.join(THIS_DIR, 'csvs', f'{pathology}.csv'), index=True)

    # Write hf5 files for whole dataset
    hf5_file = os.path.join(hf5_dir, 'ap_no_support_devices_no_uncertain.hf5')
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


def load_mimic_cxr_naive_split():
    """Load MIMIC-CXR dataset with naive split."""
    normal = pd.read_csv(os.path.join(THIS_DIR, 'csvs', 'No Finding.csv'))
    abnormal = {
        label: pd.read_csv(os.path.join(THIS_DIR, 'csvs', f'{label}.csv'))
        for label in CHEXPERT_LABELS if label != 'No Finding'
    }

    # Split normal images into train, val, test (use 1000 for val and test)
    normal_val_test = normal.sample(n=2000, random_state=42)
    normal_train = normal[~normal['path'].isin(normal_val_test['path'])]
    normal_val = normal_val_test[:1000]
    normal_test = normal_val_test[1000:]

    # Split abnormal images into val, test (use maximum 1000 for val and test)
    val = {}
    test = {}
    val_labels = {}
    test_labels = {}
    for pathology in CHEXPERT_LABELS:
        if pathology == 'No Finding':
            continue
        n_files = len(abnormal[pathology])
        n_use = min(n_files, 2000) // 2
        # Select anomal samples for val and test
        abnormal_val_test_pathology = abnormal[pathology].sample(n=2 * n_use, random_state=42)
        abnormal_val_pathology = abnormal_val_test_pathology[:n_use]
        abnormal_test_pathology = abnormal_val_test_pathology[n_use:]
        # Select normal samples for val and test
        normal_val_pathology = normal_val[:n_use]
        normal_test_pathology = normal_test[:n_use]
        # Merge normal and abnormal samples
        val[pathology] = pd.concat([abnormal_val_pathology, normal_val_pathology])
        test[pathology] = pd.concat([abnormal_test_pathology, normal_test_pathology])
        # Add labels
        val[pathology]['label'] = np.concatenate([np.ones(n_use), np.zeros(n_use)])
        test[pathology]['label'] = np.concatenate([np.ones(n_use), np.zeros(n_use)])
        # Shuffle
        val[pathology] = val[pathology].sample(frac=1, random_state=42)
        test[pathology] = test[pathology].sample(frac=1, random_state=42)
        # Save labels
        val_labels[pathology] = val[pathology]['label'].values
        test_labels[pathology] = test[pathology]['label'].values

    hf5_file = h5py.File(
        os.path.join(
            MIMIC_CXR_DIR,
            'hf5',
            'ap_no_support_devices_no_uncertain.hf5'),
        'r')['images']

    # Return
    filenames = {'train': hf5_file}
    labels = {'train': np.zeros(len(normal_train))}
    meta = {'train': np.zeros(len(normal_train))}
    index_mapping = {'train': normal_train['hf5_idx'].values}
    for pathology in CHEXPERT_LABELS:
        if pathology == 'No Finding':
            continue
        filenames[f'val/{pathology}'] = hf5_file
        labels[f'val/{pathology}'] = val_labels[pathology]
        meta[f'val/{pathology}'] = np.zeros(len(val[pathology]))
        index_mapping[f'val/{pathology}'] = val[pathology]['hf5_idx'].values
        filenames[f'test/{pathology}'] = hf5_file
        labels[f'test/{pathology}'] = test_labels[pathology]
        meta[f'test/{pathology}'] = np.zeros(len(test[pathology]))
        index_mapping[f'test/{pathology}'] = test[pathology]['hf5_idx'].values
    return filenames, labels, meta, index_mapping


def load_mimic_cxr_sex_split(mimic_cxr_dir: str = MIMIC_CXR_DIR,
                             male_percent: float = 0.5):
    """Load data with sex-balanced val and test sets."""
    assert 0.0 <= male_percent <= 1.0
    female_percent = 1 - male_percent

    # Load metadata
    normal = pd.read_csv(os.path.join(THIS_DIR, 'csvs', 'No Finding.csv'))
    abnormal = pd.concat([
        pd.read_csv(os.path.join(THIS_DIR, 'csvs', f'{label}.csv'))
        for label in CHEXPERT_LABELS if label != 'No Finding'
    ]).sample(frac=1, random_state=42)

    # Split normal images into train, val, test (use 1000 for val and test)
    normal_male = normal[normal.gender == 'M']
    normal_female = normal[normal.gender == 'F']
    val_test_normal_male = normal_male.sample(n=2000, random_state=42)
    val_test_normal_female = normal_female.sample(n=2000, random_state=42)
    val_normal_male = val_test_normal_male[:1000]
    val_normal_female = val_test_normal_female[:1000]
    test_normal_male = val_test_normal_male[1000:]
    test_normal_female = val_test_normal_female[1000:]

    # Split abnormal images into val, test (use maximum 1000 for val and test)
    abnormal_male = abnormal[abnormal.gender == 'M']
    abnormal_female = abnormal[abnormal.gender == 'F']
    val_test_abnormal_male = abnormal_male.sample(n=2000, random_state=42)
    val_test_abnormal_female = abnormal_female.sample(n=2000, random_state=42)
    val_abnormal_male = val_test_abnormal_male.iloc[:1000, :]
    val_abnormal_female = val_test_abnormal_female.iloc[:1000, :]
    test_abnormal_male = val_test_abnormal_male.iloc[1000:, :]
    test_abnormal_female = val_test_abnormal_female.iloc[1000:, :]

    # Aggregate validation and test sets and shuffle
    val_male = pd.concat([val_normal_male, val_abnormal_male]).sample(frac=1, random_state=42)
    val_female = pd.concat([val_normal_female, val_abnormal_female]).sample(frac=1, random_state=42)
    test_male = pd.concat([test_normal_male, test_abnormal_male]).sample(frac=1, random_state=42)
    test_female = pd.concat([test_normal_female, test_abnormal_female]).sample(frac=1, random_state=42)

    # Rest for training
    rest_normal_male = normal_male[~normal_male.subject_id.isin(val_test_normal_male.subject_id)]
    rest_normal_female = normal_female[~normal_female.subject_id.isin(val_test_normal_female.subject_id)]
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
            mimic_cxr_dir,
            'hf5',
            'ap_no_support_devices_no_uncertain.hf5'),
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
        meta[mode] = np.array([SEX_MAPPING[v] for v in data.gender.values])
        index_mapping[mode] = data.hf5_idx.values
    return filenames, labels, meta, index_mapping


if __name__ == '__main__':
    # prepare_mimic_cxr()
    # filesnames, labels, meta = load_mimic_cxr_naive_split()
    pass
