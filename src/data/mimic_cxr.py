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
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

from src import MIMIC_CXR_DIR, SEED
from src.data.data_utils import read_memmap, write_memmap


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SEX_MAPPING = {
    'M': 0,
    'F': 1
}
RACE_MAPPING = {
    'Black': 0,
    'White': 1
}

MAX_YOUNG = 31  # 31  # 41
MIN_OLD = 61  # 61  # 66


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
    # mimic_dir = "/vol/aimspace/projects/mimic_cxr/mimic-cxr-jpg_2-0-0"
    # metadata = pd.read_csv("/vol/aimspace/projects/mimic_cxr/mimic-cxr-jpg_2-0-0/mimic-cxr-2.0.0-metadata.csv.gz")
    # chexpert = pd.read_csv("/vol/aimspace/projects/mimic_cxr/mimic-cxr-jpg_2-0-0/mimic-cxr-2.0.0-chexpert.csv.gz")
    # mimic_sex = pd.read_csv("/vol/aimspace/users/meissen/patients.csv")  # From MIMIC-IV, v2.2
    metadata = pd.read_csv(os.path.join(mimic_dir, 'mimic-cxr-2.0.0-metadata.csv'))
    chexpert = pd.read_csv(os.path.join(mimic_dir, 'mimic-cxr-2.0.0-chexpert.csv'))
    mimic_sex = pd.read_csv(os.path.join(mimic_dir, 'patients.csv'))  # From MIMIC-IV, v2.2
    print(f"Total number of images: {len(metadata)}")

    # Add sex information to metadata
    metadata = metadata.merge(mimic_sex, on='subject_id')
    print(f'Number of images with age and sex metadata: {len(metadata)}')

    # We only consider frontal view images. (AP and PA)
    metadata = metadata[metadata['ViewPosition'].isin(['AP', 'PA'])]
    print(f"Number of frontal view images: {len(metadata)}")

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

    # Save ordering of files in a new column 'memmap_idx'
    metadata['memmap_idx'] = np.arange(len(metadata))

    memmap_dir = os.path.join(mimic_dir, 'memmap')
    # memmap_dir = '/vol/aimspace/users/meissen/datasets/MIMIC-CXR/memmap'
    os.makedirs(memmap_dir, exist_ok=True)

    # csv_dir = os.path.join(THIS_DIR, 'csvs', 'mimic-cxr_ap')
    csv_dir = os.path.join(THIS_DIR, 'csvs', 'mimic-cxr_ap_pa')
    os.makedirs(csv_dir, exist_ok=True)

    # Save csv for normal and abnormal images
    normal = metadata[metadata['No Finding'] == 1.0]
    print(f"Number of normal images: {len(normal)}")
    normal['label'] = 0
    normal.to_csv(os.path.join(csv_dir, 'normal.csv'), index=True)

    abnormal = metadata[metadata['No Finding'].isna()]
    print(f"Number of abnormal images: {len(abnormal)}")
    abnormal['label'] = 1
    abnormal.to_csv(os.path.join(csv_dir, 'abnormal.csv'), index=True)

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
        pathologies[pathology].to_csv(os.path.join(csv_dir, f'{pathology}.csv'), index=True)

    # Write memmap files for whole dataset
    memmap_file = os.path.join(memmap_dir, 'ap_pa_no_support_devices_no_uncertain')
    print(f"Writing memmap file '{memmap_file}'...")
    write_memmap(
        metadata['path'].values.tolist(),
        memmap_file,
        load_fn=partial(load_and_resize, target_size=(256, 256)),
        target_size=(256, 256)
    )


def load_and_resize(path: str, target_size: Tuple[int, int]):
    image = Image.open(path).convert('L')
    image = transforms.CenterCrop(min(image.size))(image)
    image = transforms.Resize(target_size)(image)
    image = transforms.ToTensor()(image)
    return image


def load_mimic_cxr_naive_split(mimic_cxr_dir: str = MIMIC_CXR_DIR,
                               max_train_samples: Optional[int] = None):
    """Load MIMIC-CXR dataset with naive split."""
    csv_dir = os.path.join(THIS_DIR, 'csvs', 'mimic-cxr_ap_pa')
    normal = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
    abnormal = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))

    # Split normal images into train, val, test (use 1000 for val and test)
    normal_val_test = normal.sample(n=2000, random_state=SEED)
    normal_train = normal[~normal.subject_id.isin(normal_val_test.subject_id)]
    normal_train = normal_train.sample(n=max_train_samples, random_state=SEED) if max_train_samples else normal_train
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
        abnormal_val_test_pathology = abnormal[pathology].sample(n=2 * n_use, random_state=SEED)
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
        val[pathology] = val[pathology].sample(frac=1, random_state=SEED)
        test[pathology] = test[pathology].sample(frac=1, random_state=SEED)
        # Save labels
        val_labels[pathology] = val[pathology]['label'].values
        test_labels[pathology] = test[pathology]['label'].values

    img_data = read_memmap(
        os.path.join(
            mimic_cxr_dir,
            'memmap',
            'ap_pa_no_support_devices_no_uncertain'),
    )

    # Return
    filenames = {'train': img_data}
    labels = {'train': np.zeros(len(normal_train))}
    meta = {'train': np.zeros(len(normal_train))}
    index_mapping = {'train': normal_train['memmap_idx'].values}
    for pathology in CHEXPERT_LABELS:
        if pathology == 'No Finding':
            continue
        filenames[f'val/{pathology}'] = img_data
        labels[f'val/{pathology}'] = val_labels[pathology]
        meta[f'val/{pathology}'] = np.zeros(len(val[pathology]))
        index_mapping[f'val/{pathology}'] = val[pathology]['memmap_idx'].values
        filenames[f'test/{pathology}'] = img_data
        labels[f'test/{pathology}'] = test_labels[pathology]
        meta[f'test/{pathology}'] = np.zeros(len(test[pathology]))
        index_mapping[f'test/{pathology}'] = test[pathology]['memmap_idx'].values
    return filenames, labels, meta, index_mapping


def load_mimic_cxr_sex_split(mimic_cxr_dir: str = MIMIC_CXR_DIR,
                             male_percent: float = 0.5,
                             max_train_samples: Optional[int] = None):
    """Load data with sex-balanced val and test sets."""
    assert 0.0 <= male_percent <= 1.0
    female_percent = 1 - male_percent

    # Load metadata
    csv_dir = os.path.join(THIS_DIR, 'csvs', 'mimic-cxr_ap_pa')
    normal = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
    abnormal = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))

    # Split normal images into train, val, test (use 500 for val and test)
    normal_male = normal[normal.gender == 'M']
    normal_female = normal[normal.gender == 'F']
    val_test_normal_male = normal_male.sample(n=1000, random_state=SEED)
    val_test_normal_female = normal_female.sample(n=1000, random_state=SEED)
    val_normal_male = val_test_normal_male[:500]
    val_normal_female = val_test_normal_female[:500]
    test_normal_male = val_test_normal_male[500:]
    test_normal_female = val_test_normal_female[500:]

    # Split abnormal images into val, test (use maximum 500 for val and test)
    abnormal_male = abnormal[abnormal.gender == 'M']
    abnormal_female = abnormal[abnormal.gender == 'F']
    val_test_abnormal_male = abnormal_male.sample(n=1000, random_state=SEED)
    val_test_abnormal_female = abnormal_female.sample(n=1000, random_state=SEED)
    val_abnormal_male = val_test_abnormal_male.iloc[:500, :]
    val_abnormal_female = val_test_abnormal_female.iloc[:500, :]
    test_abnormal_male = val_test_abnormal_male.iloc[500:, :]
    test_abnormal_female = val_test_abnormal_female.iloc[500:, :]

    # Aggregate validation and test sets and shuffle
    val_male = pd.concat([val_normal_male, val_abnormal_male]).sample(frac=1, random_state=SEED)
    val_female = pd.concat([val_normal_female, val_abnormal_female]).sample(frac=1, random_state=SEED)
    test_male = pd.concat([test_normal_male, test_abnormal_male]).sample(frac=1, random_state=SEED)
    test_female = pd.concat([test_normal_female, test_abnormal_female]).sample(frac=1, random_state=SEED)

    # Rest for training
    rest_normal_male = normal_male[~normal_male.subject_id.isin(val_test_normal_male.subject_id)]
    rest_normal_female = normal_female[~normal_female.subject_id.isin(val_test_normal_female.subject_id)]
    if max_train_samples is not None:
        max_available = min(len(rest_normal_male), len(rest_normal_female)) * 2
        n_samples = min(max_available, max_train_samples)
    else:
        n_samples = min(len(rest_normal_male), len(rest_normal_female))
    n_male = int(n_samples * male_percent)
    n_female = int(n_samples * female_percent)
    train_male = rest_normal_male.sample(n=n_male, random_state=SEED)
    train_female = rest_normal_female.sample(n=n_female, random_state=SEED)

    # Aggregate training set and shuffle
    train = pd.concat([train_male, train_female]).sample(frac=1, random_state=SEED)
    print(f"Using {n_male} male and {n_female} female samples for training.")

    img_data = read_memmap(
        os.path.join(
            mimic_cxr_dir,
            'memmap',
            'ap_pa_no_support_devices_no_uncertain'),
    )

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
        filenames[mode] = img_data
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.array([SEX_MAPPING[v] for v in data.gender.values])
        index_mapping[mode] = data.memmap_idx.values
    return filenames, labels, meta, index_mapping


def load_mimic_cxr_age_split(mimic_cxr_dir: str = MIMIC_CXR_DIR,
                             old_percent: float = 0.5,
                             max_train_samples: Optional[int] = None):
    """Load data with age-balanced val and test sets."""
    assert 0.0 <= old_percent <= 1.0
    young_percent = 1 - old_percent

    # Load metadata
    csv_dir = os.path.join(THIS_DIR, 'csvs', 'mimic-cxr_ap_pa')
    normal = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
    abnormal = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))

    # Filter ages over 90 years (outliers in MIMIC-IV)
    normal = normal[normal.anchor_age < 91]
    abnormal = abnormal[abnormal.anchor_age < 91]

    # Split data into bins by age
    # n_bins = 3
    # t = np.histogram(normal.anchor_age, bins=n_bins)[1]
    # print(f"Splitting data into {n_bins - 1} bins by age: {t}")

    # normal_young = normal[normal.anchor_age < t[1]]
    # normal_old = normal[normal.anchor_age >= t[2]]
    # abnormal_young = abnormal[abnormal.anchor_age < t[1]]
    # abnormal_old = abnormal[abnormal.anchor_age >= t[2]]

    normal_young = normal[normal.anchor_age <= MAX_YOUNG]
    normal_old = normal[normal.anchor_age >= MIN_OLD]
    abnormal_young = abnormal[abnormal.anchor_age <= MAX_YOUNG]
    abnormal_old = abnormal[abnormal.anchor_age >= MIN_OLD]

    # Split normal images into train, val, test (use 500 for val and test)
    val_test_normal_old = normal_old.sample(n=1000, random_state=SEED)
    val_test_normal_young = normal_young.sample(n=1000, random_state=SEED)
    val_normal_old = val_test_normal_old[:500]
    val_normal_young = val_test_normal_young[:500]
    test_normal_old = val_test_normal_old[500:]
    test_normal_young = val_test_normal_young[500:]

    # Split abnormal images into val, test (use maximum 500 for val and test)
    val_test_abnormal_old = abnormal_old.sample(n=1000, random_state=SEED)
    val_test_abnormal_young = abnormal_young.sample(n=1000, random_state=SEED)
    val_abnormal_old = val_test_abnormal_old.iloc[:500, :]
    val_abnormal_young = val_test_abnormal_young.iloc[:500, :]
    test_abnormal_old = val_test_abnormal_old.iloc[500:, :]
    test_abnormal_young = val_test_abnormal_young.iloc[500:, :]

    # Aggregate validation and test sets and shuffle
    val_old = pd.concat([val_normal_old, val_abnormal_old]).sample(frac=1, random_state=SEED)
    val_young = pd.concat([val_normal_young, val_abnormal_young]).sample(frac=1, random_state=SEED)
    test_old = pd.concat([test_normal_old, test_abnormal_old]).sample(frac=1, random_state=SEED)
    test_young = pd.concat([test_normal_young, test_abnormal_young]).sample(frac=1, random_state=SEED)

    # Rest for training
    rest_normal_old = normal_old[~normal_old.subject_id.isin(val_test_normal_old.subject_id)]
    rest_normal_young = normal_young[~normal_young.subject_id.isin(val_test_normal_young.subject_id)]
    if max_train_samples is not None:
        max_available = min(len(rest_normal_old), len(rest_normal_young)) * 2
        n_samples = min(max_available, max_train_samples)
    else:
        n_samples = min(len(rest_normal_old), len(rest_normal_young))
    n_old = int(n_samples * old_percent)
    n_young = int(n_samples * young_percent)
    train_old = rest_normal_old.sample(n=n_old, random_state=SEED)
    train_young = rest_normal_young.sample(n=n_young, random_state=SEED)

    # Aggregate training set and shuffle
    train = pd.concat([train_old, train_young]).sample(frac=1, random_state=SEED)
    print(f"Using {n_old} old and {n_young} young samples for training.")

    img_data = read_memmap(
        os.path.join(
            mimic_cxr_dir,
            'memmap',
            'ap_pa_no_support_devices_no_uncertain'),
    )

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
        filenames[mode] = img_data
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.zeros(len(data), dtype=np.float32)  # Unused
        index_mapping[mode] = data.memmap_idx.values
    return filenames, labels, meta, index_mapping


def load_mimic_cxr_race_split(mimic_cxr_dir: str = MIMIC_CXR_DIR,
                              white_percent: float = 0.5,
                              max_train_samples: Optional[int] = None):
    """Load data with race-balanced val and test sets."""
    if white_percent is not None:
        assert 0.0 <= white_percent <= 1.0
        black_percent = 1 - white_percent

    # Load metadata
    csv_dir = os.path.join(THIS_DIR, 'csvs', 'mimic-cxr_ap_pa')
    normal = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
    abnormal = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))

    # remove patients who have inconsistent documented race information
    # credit to github.com/robintibor
    admissions_df = pd.read_csv(os.path.join(mimic_cxr_dir, 'admissions.csv'))
    race_df = admissions_df.loc[:, ['subject_id', 'race']].drop_duplicates()
    v = race_df.subject_id.value_counts()
    subject_id_more_than_once = v.index[v.gt(1)]
    ambiguous_race_df = race_df[race_df.subject_id.isin(subject_id_more_than_once)]
    inconsistent_race = ambiguous_race_df.subject_id.unique()
    normal = pd.merge(normal, race_df, on='subject_id')
    normal = normal[~normal.subject_id.isin(inconsistent_race)]
    abnormal = pd.merge(abnormal, race_df, on='subject_id')
    abnormal = abnormal[~abnormal.subject_id.isin(inconsistent_race)]

    # Only consider white and black patients
    mask = (normal.race.str.contains('BLACK', na=False))
    normal.loc[mask, 'race'] = 'Black'
    mask = (normal.race == 'WHITE')
    normal.loc[mask, 'race'] = 'White'
    mask = (abnormal.race.str.contains('BLACK', na=False))
    abnormal.loc[mask, 'race'] = 'Black'
    mask = (abnormal.race == 'WHITE')
    abnormal.loc[mask, 'race'] = 'White'
    normal = normal[normal.race.isin(['Black', 'White'])]
    abnormal = abnormal[abnormal.race.isin(['Black', 'White'])]

    n_v_t = 500

    # Split normal images into train, val, test
    normal_black = normal[normal.race == 'Black']
    normal_white = normal[normal.race == 'White']
    val_test_normal_black = normal_black.sample(n=2 * n_v_t, random_state=SEED)
    val_test_normal_white = normal_white.sample(n=2 * n_v_t, random_state=SEED)
    val_normal_black = val_test_normal_black[:n_v_t]
    val_normal_white = val_test_normal_white[:n_v_t]
    test_normal_black = val_test_normal_black[n_v_t:]
    test_normal_white = val_test_normal_white[n_v_t:]

    # Split abnormal images into val, test
    abnormal_black = abnormal[abnormal.race == 'Black']
    abnormal_white = abnormal[abnormal.race == 'White']
    val_test_abnormal_black = abnormal_black.sample(n=2 * n_v_t, random_state=SEED)
    val_test_abnormal_white = abnormal_white.sample(n=2 * n_v_t, random_state=SEED)
    val_abnormal_black = val_test_abnormal_black.iloc[:n_v_t, :]
    val_abnormal_white = val_test_abnormal_white.iloc[:n_v_t, :]
    test_abnormal_black = val_test_abnormal_black.iloc[n_v_t:, :]
    test_abnormal_white = val_test_abnormal_white.iloc[n_v_t:, :]

    # Aggregate validation and test sets and shuffle
    val_black = pd.concat([val_normal_black, val_abnormal_black]).sample(frac=1, random_state=SEED)
    val_white = pd.concat([val_normal_white, val_abnormal_white]).sample(frac=1, random_state=SEED)
    test_black = pd.concat([test_normal_black, test_abnormal_black]).sample(frac=1, random_state=SEED)
    test_white = pd.concat([test_normal_white, test_abnormal_white]).sample(frac=1, random_state=SEED)

    # Rest for training
    rest_normal_black = normal_black[~normal_black.subject_id.isin(val_test_normal_black.subject_id)]
    rest_normal_white = normal_white[~normal_white.subject_id.isin(val_test_normal_white.subject_id)]
    if white_percent is None:
        n_white = len(rest_normal_white)
        n_black = len(rest_normal_black)
        if max_train_samples is not None:
            frac_white = n_white / (n_white + n_black)
            n_white = int(max_train_samples * frac_white)
            n_black = max_train_samples - n_white
    else:
        if max_train_samples is not None:
            max_available = min(len(rest_normal_black), len(rest_normal_white)) * 2
            n_samples = min(max_available, max_train_samples)
        else:
            n_samples = min(len(rest_normal_black), len(rest_normal_white))
        n_black = int(n_samples * black_percent)
        n_white = int(n_samples * white_percent)
    train_black = rest_normal_black.sample(n=n_black, random_state=SEED)
    train_white = rest_normal_white.sample(n=n_white, random_state=SEED)

    # Aggregate training set and shuffle
    train = pd.concat([train_black, train_white]).sample(frac=1, random_state=SEED)
    print(f"Using {n_black} black and {n_white} white samples for training.")

    img_data = read_memmap(
        os.path.join(
            mimic_cxr_dir,
            'memmap',
            'ap_pa_no_support_devices_no_uncertain'),
    )

    # Return
    filenames = {}
    labels = {}
    meta = {}
    index_mapping = {}
    sets = {
        'train': train,
        'val/black': val_black,
        'val/white': val_white,
        'test/black': test_black,
        'test/white': test_white,
    }
    for mode, data in sets.items():
        filenames[mode] = img_data
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.array([RACE_MAPPING[v] for v in data.race.values])
        index_mapping[mode] = data.memmap_idx.values
    return filenames, labels, meta, index_mapping


def load_mimic_cxr_intersectional_age_sex_split(mimic_cxr_dir: str = MIMIC_CXR_DIR):
    """Load MIMIC-CXR dataset with intersectional val and test sets."""
    csv_dir = os.path.join(THIS_DIR, 'csvs', 'mimic-cxr_ap_pa')
    normal = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
    abnormal = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))

    # Split normal images into sets
    normal_male_young = normal[(normal.gender == 'M') & (normal.anchor_age <= MAX_YOUNG)]
    normal_female_young = normal[(normal.gender == 'F') & (normal.anchor_age <= MAX_YOUNG)]
    normal_male_old = normal[(normal.gender == 'M') & (normal.anchor_age >= MIN_OLD)]
    normal_female_old = normal[(normal.gender == 'F') & (normal.anchor_age >= MIN_OLD)]

    val_test_normal_male_young = normal_male_young.sample(n=1000, random_state=SEED)
    val_test_normal_female_young = normal_female_young.sample(n=1000, random_state=SEED)
    val_test_normal_male_old = normal_male_old.sample(n=1000, random_state=SEED)
    val_test_normal_female_old = normal_female_old.sample(n=1000, random_state=SEED)

    val_normal_male_young = val_test_normal_male_young[:500]
    val_normal_female_young = val_test_normal_female_young[:500]
    val_normal_male_old = val_test_normal_male_old[:500]
    val_normal_female_old = val_test_normal_female_old[:500]

    test_normal_male_young = val_test_normal_male_young[500:]
    test_normal_female_young = val_test_normal_female_young[500:]
    test_normal_male_old = val_test_normal_male_old[500:]
    test_normal_female_old = val_test_normal_female_old[500:]

    # Split abnormal images into sets
    abnormal_male_young = abnormal[(abnormal.gender == 'M') & (abnormal.anchor_age <= MAX_YOUNG)]
    abnormal_female_young = abnormal[(abnormal.gender == 'F') & (abnormal.anchor_age <= MAX_YOUNG)]
    abnormal_male_old = abnormal[(abnormal.gender == 'M') & (abnormal.anchor_age >= MIN_OLD)]
    abnormal_female_old = abnormal[(abnormal.gender == 'F') & (abnormal.anchor_age >= MIN_OLD)]

    val_test_abnormal_male_young = abnormal_male_young.sample(n=1000, random_state=SEED)
    val_test_abnormal_female_young = abnormal_female_young.sample(n=1000, random_state=SEED)
    val_test_abnormal_male_old = abnormal_male_old.sample(n=1000, random_state=SEED)
    val_test_abnormal_female_old = abnormal_female_old.sample(n=1000, random_state=SEED)

    val_abnormal_male_young = val_test_abnormal_male_young[:500]
    val_abnormal_female_young = val_test_abnormal_female_young[:500]
    val_abnormal_male_old = val_test_abnormal_male_old[:500]
    val_abnormal_female_old = val_test_abnormal_female_old[:500]

    test_abnormal_male_young = val_test_abnormal_male_young[500:]
    test_abnormal_female_young = val_test_abnormal_female_young[500:]
    test_abnormal_male_old = val_test_abnormal_male_old[500:]
    test_abnormal_female_old = val_test_abnormal_female_old[500:]

    # Merge and shuffle normal and abnormal val and test sets
    val_male_young = pd.concat([val_normal_male_young, val_abnormal_male_young]).sample(frac=1, random_state=SEED)
    val_female_young = pd.concat([val_normal_female_young, val_abnormal_female_young]).sample(frac=1, random_state=SEED)
    val_male_old = pd.concat([val_normal_male_old, val_abnormal_male_old]).sample(frac=1, random_state=SEED)
    val_female_old = pd.concat([val_normal_female_old, val_abnormal_female_old]).sample(frac=1, random_state=SEED)

    val_male = pd.concat([val_male_young, val_male_old]).sample(frac=1, random_state=SEED)
    val_female = pd.concat([val_female_young, val_female_old]).sample(frac=1, random_state=SEED)
    val_young = pd.concat([val_male_young, val_female_young]).sample(frac=1, random_state=SEED)
    val_old = pd.concat([val_male_old, val_female_old]).sample(frac=1, random_state=SEED)

    test_male_young = pd.concat([test_normal_male_young, test_abnormal_male_young]).sample(frac=1, random_state=SEED)
    test_female_young = pd.concat([test_normal_female_young, test_abnormal_female_young]).sample(frac=1, random_state=SEED)
    test_male_old = pd.concat([test_normal_male_old, test_abnormal_male_old]).sample(frac=1, random_state=SEED)
    test_female_old = pd.concat([test_normal_female_old, test_abnormal_female_old]).sample(frac=1, random_state=SEED)

    test_male = pd.concat([test_male_young, test_male_old]).sample(frac=1, random_state=SEED)
    test_female = pd.concat([test_female_young, test_female_old]).sample(frac=1, random_state=SEED)
    test_young = pd.concat([test_male_young, test_female_young]).sample(frac=1, random_state=SEED)
    test_old = pd.concat([test_male_old, test_female_old]).sample(frac=1, random_state=SEED)

    # Use rest of normal samples for training
    val_test_normal = pd.concat([
        val_test_normal_male_young,
        val_test_normal_female_young,
        val_test_normal_male_old,
        val_test_normal_female_old
    ])
    train = normal[~normal.subject_id.isin(val_test_normal.subject_id)]
    print(f"Using {len(train)} normal samples for training.")
    print(f"Average age of training samples: {train.anchor_age.mean():.2f}, std: {train.anchor_age.std():.2f}")
    print(f"Fraction of female samples in training: {(train.gender == 'F').mean():.2f}")
    print(f"Fraction of male samples in training: {(train.gender == 'M').mean():.2f}")
    print(f"Fraction of young samples in training: {(train.anchor_age <= MAX_YOUNG).mean():.2f}")
    print(f"Fraction of old samples in training: {(train.anchor_age >= MIN_OLD).mean():.2f}")

    img_data = read_memmap(
        os.path.join(
            mimic_cxr_dir,
            'memmap',
            'ap_pa_no_support_devices_no_uncertain'),
    )

    # Return
    filenames = {}
    labels = {}
    meta = {}
    index_mapping = {}
    sets = {
        'train': train,
        'val/male_young': val_male_young,
        'val/female_young': val_female_young,
        'val/male_old': val_male_old,
        'val/female_old': val_female_old,
        'val/male': val_male,
        'val/female': val_female,
        'val/young': val_young,
        'val/old': val_old,
        'test/male_young': test_male_young,
        'test/female_young': test_female_young,
        'test/male_old': test_male_old,
        'test/female_old': test_female_old,
        'test/male': test_male,
        'test/female': test_female,
        'test/young': test_young,
        'test/old': test_old,
    }
    for mode, data in sets.items():
        filenames[mode] = img_data
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.zeros(len(data), dtype=np.float32)  # Unused
        index_mapping[mode] = data.memmap_idx.values
    return filenames, labels, meta, index_mapping


if __name__ == '__main__':
    # prepare_mimic_cxr()
    pass
