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
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src import MIMIC_CXR_DIR


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


CHEXPERT_LABELS = [
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
    'No Finding',
]


def prepare_mimic_cxr(mimic_dir: str = MIMIC_CXR_DIR):
    metadata = pd.read_csv(os.path.join(mimic_dir, 'mimic-cxr-2.0.0-metadata.csv'))
    chexpert = pd.read_csv(os.path.join(mimic_dir, 'mimic-cxr-2.0.0-chexpert.csv'))
    print(f"Total number of images: {len(metadata)}")

    # We only consider frontal view images. (AP)
    metadata = metadata[metadata['ViewPosition'] == 'AP']
    print(f"Number of frontal view images: {len(metadata)}")

    # Match metadata and chexpert.
    metadata = metadata.merge(chexpert, on=['subject_id', 'study_id'])

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

    # Save ordering of files in a new column 'hf5_idx'
    metadata['hf5_idx'] = np.arange(len(metadata))

    hf5_dir = os.path.join(mimic_dir, 'hf5')
    os.makedirs(hf5_dir, exist_ok=True)

    # Select sets of all pathologies
    pathologies = {}
    for pathology in CHEXPERT_LABELS:
        pathologies[pathology] = metadata[metadata[pathology] == 1.0]
        print(f"Number of images for '{pathology}': {len(pathologies[pathology])}")

        # Save files
        pathologies[pathology].to_csv(os.path.join(THIS_DIR, 'csvs', f'{pathology}.csv'), index=True)

    # Write hf5 files for whole dataset
    hf5_file = os.path.join(hf5_dir, 'ap_no_support_devices_no_uncertain.hf5')
    print(f"Writing hf5 file '{hf5_file}'...")
    write_hf5_file(
        metadata['path'].values.tolist(),
        hf5_file,
        target_size=(256, 256)
    )


def write_hf5_file(files: List[str], output_file: str, target_size: Tuple[int, int] = (256, 256)):
    """Write hf5 file with images and labels."""
    # Write images to hf5 file
    hf5_file = h5py.File(output_file, 'w')
    hf5_file.create_dataset('images', (len(files), 1, *target_size), dtype='float32')

    n_failed = 0
    failed = []
    for i, file in tqdm(enumerate(files), total=len(files)):
        try:
            image = load_and_resize(file, target_size)
        except Exception as e:
            n_failed += 1
            failed.append(file)
            print(f"Failed to load image '{file}': {e}")
            continue
        hf5_file['images'][i] = image

    print(f"Failed to load {n_failed} images.")
    for file in failed:
        print(file)
    hf5_file.close()


def load_and_resize(path, target_size):
    image = Image.open(path).convert('L')
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

    # Return
    filenames = {'train': normal_train['path'].values.tolist()}
    labels = {'train': np.zeros(len(normal_train))}
    meta = {'train': np.zeros(len(normal_train))}
    for pathology in CHEXPERT_LABELS:
        if pathology == 'No Finding':
            continue
        filenames[f'val/{pathology}'] = val[pathology]['path'].values.tolist()
        labels[f'val/{pathology}'] = val_labels[pathology]
        meta[f'val/{pathology}'] = np.zeros(len(val[pathology]))
        filenames[f'test/{pathology}'] = test[pathology]['path'].values.tolist()
        labels[f'test/{pathology}'] = test_labels[pathology]
        meta[f'test/{pathology}'] = np.zeros(len(test[pathology]))
    return filenames, labels, meta


if __name__ == '__main__':
    prepare_mimic_cxr()
    # filesnames, labels, meta = load_mimic_cxr_naive_split()
    pass
