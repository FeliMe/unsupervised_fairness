import argparse
import os
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat
from tqdm import tqdm

from src import BRATS_DIR, CAMCAN_DIR


def load_camcan_age_split(
        data_dir: str = CAMCAN_DIR,
        old_percent: float = 0.5,
        sequence: str = 'T2',
        slice_range: Tuple[int, int] = None):
    assert 0.0 <= old_percent <= 1.0
    young_percent = 1 - old_percent

    # Load data
    with h5py.File(os.path.join(data_dir, 'hf5', f'camcan_{sequence}_ordered_as_participants_tsv.hf5'), 'r') as f:
        data = torch.from_numpy(f['mri'][...])
    meta = pd.read_csv(os.path.join(
        data_dir,
        'cc700/mri/pipeline/release004/BIDS_20190411/anat/participants.tsv'
    ), sep='\t')
    ages = meta['age'].values

    # Select slice range
    if slice_range is not None:
        data = data[:, slice_range[0]:slice_range[1]]

    # Split data into bins by age
    n_bins = 3
    age_bins = np.histogram(ages, bins=n_bins)[1]
    data_young = data[ages < age_bins[1]]
    data_old = data[ages >= age_bins[2]]
    ages_young = ages[ages < age_bins[1]]
    ages_old = ages[ages >= age_bins[2]]

    # Select age group
    n_samples = min(len(data_young), len(data_old))
    n_young = int(n_samples * young_percent)
    n_old = int(n_samples * old_percent)
    sampled_inds_young = np.random.default_rng(42).choice(np.arange(n_samples), n_young, replace=False)
    sampled_inds_old = np.random.default_rng(42).choice(np.arange(n_samples), n_old, replace=False)
    sampled_data_young = data_young[sampled_inds_young]
    sampled_ages_young = ages_young[sampled_inds_young]
    sampled_data_old = data_old[sampled_inds_old]
    sampled_ages_old = ages_old[sampled_inds_old]

    # Concatenate and shuffle
    data = torch.cat([sampled_data_young, sampled_data_old], dim=0)
    ages = np.concatenate([sampled_ages_young, sampled_ages_old], axis=0)
    inds = np.random.default_rng(42).permutation(np.arange(len(data)))
    data = data[inds]
    ages = ages[inds]

    ages = repeat(ages, 'n -> (n d)', d=data.shape[1])
    data = rearrange(data, 'n d h w -> (n d) 1 h w')
    labels = np.zeros(ages.shape)
    return data, labels, ages, age_bins


def load_brats_age_split(
        age_bins: np.ndarray,
        data_dir: str = CAMCAN_DIR,
        sequence: str = 'T2',
        slice_range: Tuple[int, int] = None):
    # Load data
    with h5py.File(os.path.join(data_dir, 'hf5', f'brats_{sequence}_ordered_as_survival_info_csv.hf5'), 'r') as f:
        data = torch.from_numpy(f['mri'][...])
    with h5py.File(os.path.join(data_dir, 'hf5', 'brats_segmentation_ordered_as_survival_info_csv.hf5'), 'r') as f:
        seg_data = f['mri'][...]
    meta = pd.read_csv(os.path.join(
        data_dir, 'MICCAI_BraTS2020_TrainingData/survival_info.csv'
    ), sep=',')
    ages = meta['Age'].values

    # Select slice range
    if slice_range is not None:
        data = data[:, slice_range[0]:slice_range[1]]
        seg_data = seg_data[:, slice_range[0]:slice_range[1]]
    seg_data_bin = np.where(seg_data > 0.9, 1, 0)
    labels = np.where(seg_data_bin.sum(axis=(2, 3)) > 16, 1, 0)  # Require at least 5 pixels to be a positive sample

    # Reshape data and labels
    ages = repeat(ages, 'n -> (n d)', d=data.shape[1])
    data = rearrange(data, 'n d h w -> (n d) 1 h w')
    labels = rearrange(labels, 'n d -> (n d)')

    # Split data into bins by age
    data_young = data[ages < age_bins[1]]
    data_old = data[ages >= age_bins[2]]
    labels_young = labels[ages < age_bins[1]]
    labels_old = labels[ages >= age_bins[2]]
    ages_young = ages[ages < age_bins[1]]
    ages_old = ages[ages >= age_bins[2]]

    # Split into val and test
    # young
    young_inds = np.random.default_rng(42).permutation(np.arange(len(data_young)))
    young_inds_val = young_inds[:len(young_inds) // 2]
    young_inds_test = young_inds[len(young_inds) // 2:]
    data_young_val = data_young[young_inds_val]
    data_young_test = data_young[young_inds_test]
    labels_young_val = labels_young[young_inds_val]
    labels_young_test = labels_young[young_inds_test]
    ages_young_val = ages_young[young_inds_val]
    ages_young_test = ages_young[young_inds_test]
    # old
    old_inds = np.random.default_rng(42).permutation(np.arange(len(data_old)))
    old_inds_val = old_inds[:len(old_inds) // 2]
    old_inds_test = old_inds[len(old_inds) // 2:]
    data_old_val = data_old[old_inds_val]
    data_old_test = data_old[old_inds_test]
    labels_old_val = labels_old[old_inds_val]
    labels_old_test = labels_old[old_inds_test]
    ages_old_val = ages_old[old_inds_val]
    ages_old_test = ages_old[old_inds_test]

    # Aggregate
    data = {
        'val/young': data_young_val,
        'val/old': data_old_val,
        'test/young': data_young_test,
        'test/old': data_old_test,
    }
    labels = {
        'val/young': labels_young_val,
        'val/old': labels_old_val,
        'test/young': labels_young_test,
        'test/old': labels_old_test,
    }
    ages = {
        'val/young': ages_young_val,
        'val/old': ages_old_val,
        'test/young': ages_young_test,
        'test/old': ages_old_test,
    }
    return data, labels, ages


def load_camcan_brats_age_split(
        camcan_dir: str = CAMCAN_DIR,
        brats_dir: str = BRATS_DIR,
        sequence: str = 'T2',
        old_percent: float = 0.5,
        slice_range: Tuple[int, int] = None):
    data_train, labels_train, ages_train, age_bins = load_camcan_age_split(
        data_dir=camcan_dir, old_percent=old_percent, sequence=sequence, slice_range=slice_range)
    data, labels, ages = load_brats_age_split(
        age_bins=age_bins, data_dir=brats_dir, sequence=sequence, slice_range=slice_range)
    data['train'] = data_train
    labels['train'] = labels_train
    ages['train'] = ages_train
    return data, labels, ages


def load_camcan_only_age_split(
        data_dir: str = CAMCAN_DIR,
        old_percent: float = 0.5,
        sequence: str = 'T2',
        slice_range: Tuple[int, int] = None):
    assert 0.0 <= old_percent <= 1.0
    young_percent = 1 - old_percent

    # Load data
    with h5py.File(os.path.join(data_dir, 'hf5', f'camcan_{sequence}_ordered_as_participants_tsv.hf5'), 'r') as f:
        data = torch.from_numpy(f['mri'][...])
    meta = pd.read_csv(os.path.join(
        data_dir,
        'cc700/mri/pipeline/release004/BIDS_20190411/anat/participants.tsv'
    ), sep='\t')
    ages = meta['age'].values

    # Select slice range
    if slice_range is not None:
        data = data[:, slice_range[0]:slice_range[1]]

    # Split data into bins by age
    n_bins = 3
    age_bins = np.histogram(ages, bins=n_bins)[1]
    data_young = data[ages < age_bins[1]]
    data_old = data[ages >= age_bins[2]]
    ages_young = ages[ages < age_bins[1]]
    ages_old = ages[ages >= age_bins[2]]

    # Randomly split into train, val, and test
    # young
    inds_young = np.random.default_rng(42).permutation(np.arange(len(data_young)))
    inds_young_train = inds_young[:int(len(inds_young) * 0.8)]
    inds_young_val = inds_young[int(len(inds_young) * 0.8):int(len(inds_young) * 0.9)]
    inds_young_test = inds_young[int(len(inds_young) * 0.9):]
    data_young_train = data_young[inds_young_train]
    data_young_val = data_young[inds_young_val]
    data_young_test = data_young[inds_young_test]
    ages_young_train = ages_young[inds_young_train]
    ages_young_val = ages_young[inds_young_val]
    ages_young_test = ages_young[inds_young_test]
    # old
    inds_old = np.random.default_rng(42).permutation(np.arange(len(data_old)))
    inds_old_train = inds_old[:int(len(inds_old) * 0.8)]
    inds_old_val = inds_old[int(len(inds_old) * 0.8):int(len(inds_old) * 0.9)]
    inds_old_test = inds_old[int(len(inds_old) * 0.9):]
    data_old_train = data_old[inds_old_train]
    data_old_val = data_old[inds_old_val]
    data_old_test = data_old[inds_old_test]
    ages_old_train = ages_old[inds_old_train]
    ages_old_val = ages_old[inds_old_val]
    ages_old_test = ages_old[inds_old_test]

    # Select age group
    n_samples = min(len(data_young_train), len(data_old_train))
    n_young = int(n_samples * young_percent)
    n_old = int(n_samples * old_percent)
    sampled_inds_train_young = np.random.default_rng(42).choice(np.arange(n_samples), n_young, replace=False)
    sampled_inds_train_old = np.random.default_rng(42).choice(np.arange(n_samples), n_old, replace=False)
    sampled_data_train_young = data_young_train[sampled_inds_train_young]
    sampled_ages_train_young = ages_young_train[sampled_inds_train_young]
    sampled_data_train_old = data_old_train[sampled_inds_train_old]
    sampled_ages_train_old = ages_old_train[sampled_inds_train_old]

    # Concatenate and shuffle
    data_train = np.concatenate([sampled_data_train_young, sampled_data_train_old], axis=0)
    ages_train = np.concatenate([sampled_ages_train_young, sampled_ages_train_old], axis=0)
    inds = np.random.default_rng(42).permutation(np.arange(len(data_train)))
    data_train = data_train[inds]
    ages_train = ages_train[inds]

    # Repeat for each slice and reshape
    ages_train = repeat(ages_train, 'n -> (n d)', d=data_train.shape[1])
    ages_young_val = repeat(ages_young_val, 'n -> (n d)', d=data_young_val.shape[1])
    ages_old_val = repeat(ages_old_val, 'n -> (n d)', d=data_old_val.shape[1])
    ages_young_test = repeat(ages_young_test, 'n -> (n d)', d=data_young_test.shape[1])
    ages_old_test = repeat(ages_old_test, 'n -> (n d)', d=data_old_test.shape[1])
    # Reshape data
    data_train = rearrange(data_train, 'n d h w -> (n d) 1 h w')
    data_young_val = rearrange(data_young_val, 'n d h w -> (n d) 1 h w')
    data_old_val = rearrange(data_old_val, 'n d h w -> (n d) 1 h w')
    data_young_test = rearrange(data_young_test, 'n d h w -> (n d) 1 h w')
    data_old_test = rearrange(data_old_test, 'n d h w -> (n d) 1 h w')

    data_train = torch.from_numpy(data_train)

    data = {
        'train': data_train,
        'val/young': data_young_val,
        'val/old': data_old_val,
        'test/young': data_young_test,
        'test/old': data_old_test,
    }
    labels = {
        'train': torch.zeros(ages_train.shape),
        'val/young': torch.zeros(ages_young_val.shape),
        'val/old': torch.zeros(ages_old_val.shape),
        'test/young': torch.zeros(ages_young_test.shape),
        'test/old': torch.zeros(ages_old_test.shape),
    }
    ages = {
        'train': ages_train,
        'val/young': ages_young_val,
        'val/old': ages_old_val,
        'test/young': ages_young_test,
        'test/old': ages_old_test,
    }
    return data, labels, ages


def mri_to_hf5(mri_paths: List[Union[str, Path]], out_filepath: str, size: Tuple[int, int, int]):
    """
    Convert directory of mri scans into a .hf5 file.
    """
    os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
    failed_images = []
    ds_size = len(mri_paths)
    with h5py.File(out_filepath, 'w') as h5f:
        dset = h5f.create_dataset('mri', shape=(ds_size, *size), dtype='float32', chunks=(1, 1, *size[1:],))
        for idx, path in enumerate(tqdm(mri_paths)):
            try:
                mri = load_nii(path)
                dset[idx] = mri
            except Exception as e:
                failed_images.append((path, e))
    print(f"{len(failed_images)} / {len(mri_paths)} mris failed to be added to hf5.", failed_images)


def prepare_hf5_camcan(data_dir: str = CAMCAN_DIR, sequence: str = 'T2'):
    """
    Convert all mri scans into a .h5 file.
    """
    assert sequence in ['T1', 'T2']
    hf5_dir = os.path.join(data_dir, 'hf5')
    meta_path = os.path.join(
        data_dir,
        'cc700/mri/pipeline/release004/BIDS_20190411/anat/participants.tsv')
    meta = pd.read_csv(meta_path, sep='\t')
    mri_paths = [os.path.join(data_dir, 'normal', f, f'{f}_{sequence}w_stripped_registered.nii.gz') for f in meta['participant_id']]
    for path in mri_paths:
        assert os.path.exists(path), f"File {path} does not exist."
    sample = load_nii(mri_paths[0])
    os.makedirs(hf5_dir, exist_ok=True)
    mri_to_hf5(
        mri_paths,
        out_filepath=os.path.join(hf5_dir, f'camcan_{sequence}_ordered_as_participants_tsv.hf5'),
        size=sample.shape)


def prepare_hf5_brats(data_dir: str = BRATS_DIR, sequence: str = 'T2'):
    """
    Convert all mri scans into a .h5 file.
    """
    assert sequence in ['T1', 'T2']
    hf5_dir = os.path.join(data_dir, 'hf5')
    data_dir = os.path.join(data_dir, 'MICCAI_BraTS2020_TrainingData')
    meta = pd.read_csv(os.path.join(data_dir, 'survival_info.csv'))
    mri_paths = [os.path.join(data_dir, f, f'{f}_{sequence.lower()}_registered.nii.gz') for f in meta['Brats20ID']]
    for path in mri_paths:
        assert os.path.exists(path), f"File {path} does not exist."
    sample = load_nii(mri_paths[0])
    os.makedirs(hf5_dir, exist_ok=True)
    mri_to_hf5(
        mri_paths,
        out_filepath=os.path.join(hf5_dir, f'brats_{sequence}_ordered_as_survival_info_csv.hf5'),
        size=sample.shape)


def prepare_hf5_brats_segmentations(data_dir: str = BRATS_DIR):
    """
    Convert all mri scans into a .h5 file.
    """
    hf5_dir = os.path.join(data_dir, 'hf5')
    data_dir = os.path.join(data_dir, 'MICCAI_BraTS2020_TrainingData')
    meta = pd.read_csv(os.path.join(data_dir, 'survival_info.csv'))
    mri_paths = [os.path.join(data_dir, f, 'anomaly_segmentation.nii.gz') for f in meta['Brats20ID']]
    for path in mri_paths:
        assert os.path.exists(path), f"File {path} does not exist."
    sample = load_nii(mri_paths[0])
    os.makedirs(hf5_dir, exist_ok=True)
    mri_to_hf5(
        mri_paths,
        out_filepath=os.path.join(hf5_dir, 'brats_segmentation_ordered_as_survival_info_csv.hf5'),
        size=sample.shape)


def load_nii(path: str, dtype: str = 'float32'):
    return nib.load(path).get_fdata().astype(dtype).transpose(2, 0, 1)  # (d, h, w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camcan_dir', type=str, default=CAMCAN_DIR)
    parser.add_argument('--brats_dir', type=str, default=BRATS_DIR)
    parser.add_argument('--sequence', type=str, default='T2')
    args = parser.parse_args()
    # prepare_hf5_camcan(data_dir=args.camcan_dir, sequence=args.sequence)
    # prepare_hf5_brats(data_dir=args.brats_dir, sequence=args.sequence)
    # prepare_hf5_brats_segmentations(data_dir=args.brats_dir)
