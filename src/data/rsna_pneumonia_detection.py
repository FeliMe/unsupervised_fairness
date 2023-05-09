import os
from glob import glob

import kaggle
import numpy as np
import pandas as pd
import pydicom as dicom
from tqdm import tqdm

from src import RSNA_DIR

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_MAPPING = {
    'Normal': 0,  # 8851, female: 2905, male: 4946, age mean: 44.94, std: 16.39, min: 2, max: 155
    'Lung Opacity': 1,  # 6012, female: 2502, male: 3510, age mean: 45.58, std: 17.46, min: 1, max: 92
    'No Lung Opacity / Not Normal': 2  # 11821, female: 5111, male: 6710, age mean: 49.33, std: 16.49, min: 1, max: 153
}

SEX_MAPPING = {
    'M': 0,
    'F': 1
}


def download_rsna(rsna_dir: str = RSNA_DIR):
    """Downloads the RSNA dataset."""
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        'kmader/rsna-pneumonia-detection-challenge',
        path=rsna_dir,
        unzip=True
    )


def extract_metadata(rsna_dir: str = RSNA_DIR):
    """Extracts metadata (labels, age, gender) from each sample of the RSNA
    dataset."""
    class_info = pd.read_csv(os.path.join(rsna_dir, 'stage_2_detailed_class_info.csv'))
    class_info.drop_duplicates(subset='patientId', inplace=True)

    metadata = []
    files = glob(f"{rsna_dir}/stage_2_train_images/*.dcm")
    for file in tqdm(files):
        ds = dicom.dcmread(file)
        patient_id = ds.PatientID
        label = class_info[class_info.patientId == patient_id]['class'].values[0]
        metadata.append({
            'patientId': patient_id,
            'label': CLASS_MAPPING[label],
            'PatientAge': int(ds.PatientAge),
            'PatientSex': ds.PatientSex
        })

    metadata = pd.DataFrame.from_dict(metadata)
    metadata.to_csv(os.path.join(THIS_DIR, 'rsna_metadata.csv'), index=False)


def load_rsna_naive_split(rsna_dir: str = RSNA_DIR,
                          anomaly: str = 'lungOpacity',
                          for_supervised: bool = False):
    assert anomaly in ['lungOpacity', 'otherAnomaly']
    """Naive train/val/test split."""
    # Load metadata
    metadata = pd.read_csv(os.path.join(THIS_DIR, 'rsna_metadata.csv'))
    normal_data = metadata[metadata.label == 0]
    if anomaly == 'lungOpacity':
        data = metadata[metadata.label == 1]
    else:
        data = metadata[metadata.label == 2]

    # Use 8051 normal samples for training
    train = normal_data.sample(n=8051, random_state=42)

    # Rest for validation and test
    rest_normal = normal_data[~normal_data.patientId.isin(train.patientId)]
    val_normal = rest_normal.sample(n=400, random_state=42)
    test_normal = rest_normal[~rest_normal.patientId.isin(val_normal.patientId)]
    val_test = data.sample(n=800, random_state=42)
    val = val_test.iloc[:400, :]
    test = val_test.iloc[400:, :]
    # Rest of anomal data
    rest_anomal = data[~data.patientId.isin(val_test.patientId)]

    # Mix train with anomal data if supervised
    if for_supervised:
        n_normal = len(train) // 2
        train_anomal = rest_anomal.sample(n_normal, random_state=42)
        train = pd.concat([train[:n_normal], train_anomal])

    # Concatenate and shuffle
    val = pd.concat([val_normal, val]).sample(frac=1, random_state=42).reset_index(drop=True)
    test = pd.concat([test_normal, test]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Return
    filenames = {}
    labels = {}
    meta = {}
    sets = {
        'train': train,
        f'val/{anomaly}': val,
        f'test/{anomaly}': test,
    }
    img_dir = os.path.join(rsna_dir, 'stage_2_train_images')
    for mode, data in sets.items():
        filenames[mode] = [f'{img_dir}/{patient_id}.dcm' for patient_id in data.patientId]
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.zeros_like(data['PatientSex'].values)
    return filenames, labels, meta


def load_rsna_gender_split(rsna_dir: str = RSNA_DIR,
                           male_percent: float = 0.5,
                           anomaly: str = 'lungOpacity',
                           for_supervised: bool = False):
    """Load data with age balanced val and test sets. Training is either young, avg, or old.
    lo = lung opacity
    oa = other anomaly
    """
    assert 0.0 <= male_percent <= 1.0
    assert anomaly in ['lungOpacity', 'otherAnomaly']
    female_percent = 1 - male_percent

    # Load metadata
    metadata = pd.read_csv(os.path.join(THIS_DIR, 'rsna_metadata.csv'))
    normal_data = metadata[metadata.label == 0]
    if anomaly == 'lungOpacity':
        data = metadata[metadata.label == 1]
    else:
        data = metadata[metadata.label == 2]

    normal_male = normal_data[normal_data.PatientSex == 'M']
    normal_female = normal_data[normal_data.PatientSex == 'F']
    male = data[data.PatientSex == 'M']
    female = data[data.PatientSex == 'F']

    # Save 100 male and 100 female samples for every label for validation and test
    # Normal
    val_test_normal_male = normal_male.sample(n=50, random_state=42)
    val_test_normal_female = normal_female.sample(n=50, random_state=42)
    val_test_male = male.sample(n=100, random_state=42)
    val_test_female = female.sample(n=100, random_state=42)
    val_male = val_test_male.iloc[:50, :]
    val_female = val_test_female.iloc[:50, :]
    test_male = val_test_male.iloc[50:, :]
    test_female = val_test_female.iloc[50:, :]
    # Aggregate validation and test sets and shuffle
    val_male = pd.concat([val_test_normal_male, val_male]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_female = pd.concat([val_test_normal_female, val_female]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_male = pd.concat([val_test_normal_male, test_male]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_female = pd.concat([val_test_normal_female, test_female]).sample(frac=1, random_state=42).reset_index(drop=True)
    # Rest of anomal data
    rest_male = male[~male.patientId.isin(val_test_male.patientId)]
    rest_female = female[~female.patientId.isin(val_test_female.patientId)]

    # Rest for training
    rest_normal_male = normal_male[~normal_male.patientId.isin(val_test_normal_male.patientId)]
    rest_normal_female = normal_female[~normal_female.patientId.isin(val_test_normal_female.patientId)]
    n_samples = min(len(rest_normal_male), len(rest_normal_female))
    n_male = int(n_samples * male_percent)
    n_female = int(n_samples * female_percent)
    train_male = rest_normal_male.sample(n=n_male, random_state=42)
    train_female = rest_normal_female.sample(n=n_female, random_state=42)

    # Mix train with anomal data if supervised
    if for_supervised:
        n_male = len(train_male) // 2
        n_female = len(train_female) // 2
        train_anomal_male = rest_male.sample(n_male, random_state=42)
        train_anomal_female = rest_female.sample(n_female, random_state=42)
        train_male = pd.concat([train_male[:n_male], train_anomal_male])
        train_female = pd.concat([train_female[:n_female], train_anomal_female])

    # Aggregate training set and shuffle
    train = pd.concat([train_male, train_female]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Using {n_male} male and {n_female} female samples for training.")

    # Return
    filenames = {}
    labels = {}
    meta = {}
    sets = {
        'train': train,
        f'val/{anomaly}_male': val_male,
        f'val/{anomaly}_female': val_female,
        f'test/{anomaly}_male': test_male,
        f'test/{anomaly}_female': test_female,
    }
    img_dir = os.path.join(rsna_dir, 'stage_2_train_images')
    for mode, data in sets.items():
        filenames[mode] = [f'{img_dir}/{patient_id}.dcm' for patient_id in data.patientId]
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.array([SEX_MAPPING[v] for v in data['PatientSex'].values])
    return filenames, labels, meta


def load_rsna_age_two_split(rsna_dir: str = RSNA_DIR,
                            train_age: float = 'young',
                            anomaly: str = 'lungOpacity',
                            for_supervised: bool = False):
    """Load data with age balanced val and test sets. Training is either young or old.
    lo = lung opacity
    oa = other anomaly
    """
    assert train_age in ['young', 'old']
    assert anomaly in ['lungOpacity', 'otherAnomaly']

    # Load metadata
    metadata = pd.read_csv(os.path.join(THIS_DIR, 'rsna_metadata.csv'))
    normal_data = metadata[metadata.label == 0]
    if anomaly == 'lungOpacity':
        data = metadata[metadata.label == 1]
    else:
        data = metadata[metadata.label == 2]

    # Filter ages over 110 years (outliers)
    normal_data = normal_data[normal_data.PatientAge < 110]
    data = data[data.PatientAge < 110]

    # Split data into bins by age
    n_bins = 3
    t = np.histogram(normal_data.PatientAge, bins=n_bins)[1]
    print(f"Splitting data into {n_bins - 1} bins by age: {t}")

    normal_young = normal_data[normal_data.PatientAge < t[1]]
    normal_old = normal_data[normal_data.PatientAge >= t[2]]
    young = data[data.PatientAge < t[1]]
    old = data[data.PatientAge >= t[2]]

    # Save 100 young and 100 old samples for every label for validation and test
    # Normal
    vc_normal_young = normal_young.sample(n=50, random_state=42)
    vc_normal_old = normal_old.sample(n=50, random_state=42)
    # Anomalous
    vc_young = young.sample(n=100, random_state=42)
    vc_old = old.sample(n=100, random_state=42)
    val_young = vc_young.iloc[:50, :]
    val_old = vc_old.iloc[:50, :]
    test_young = vc_young.iloc[50:, :]
    test_old = vc_old.iloc[50:, :]
    # Rest of anomal data
    rest_young = young[~young.patientId.isin(vc_young.patientId)]
    rest_old = old[~old.patientId.isin(vc_old.patientId)]

    # Aggregate validation and test sets and shuffle
    val_young = pd.concat([vc_normal_young, val_young]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_old = pd.concat([vc_normal_old, val_old]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_young = pd.concat([vc_normal_young, test_young]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_old = pd.concat([vc_normal_old, test_old]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Rest for training
    rest_normal_young = normal_young[~normal_young.patientId.isin(vc_normal_young.patientId)]
    rest_normal_old = normal_old[~normal_old.patientId.isin(vc_normal_old.patientId)]
    n_samples = min(len(rest_normal_young), len(rest_normal_old))
    train_young = rest_normal_young.sample(n=n_samples, random_state=42)
    train_old = rest_normal_old.sample(n=n_samples, random_state=42)

    # Mix train with anomal data if supervised
    if for_supervised:
        n_young = len(train_young)
        train_young = pd.concat([train_young[:n_young // 2], rest_young.sample(n=n_young // 2, random_state=42)])
        n_old = len(train_old)
        train_old = pd.concat([train_old[:n_old // 2], rest_old.sample(n=n_old // 2, random_state=42)])

    if train_age == 'young':
        train = train_young
    elif train_age == 'old':
        train = train_old
    else:
        raise ValueError("train_age must be 'young' or 'old'.")

    print(f"Using {n_samples} {train_age} samples for training.")

    # Return
    filenames = {}
    labels = {}
    meta = {}
    sets = {
        'train': train,
        f'val/{anomaly}_young': val_young,
        f'val/{anomaly}_old': val_old,
        f'test/{anomaly}_young': test_young,
        f'test/{anomaly}_old': test_old,
    }
    img_dir = os.path.join(rsna_dir, 'stage_2_train_images')
    for mode, data in sets.items():
        filenames[mode] = [f'{img_dir}/{patient_id}.dcm' for patient_id in data.patientId]
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = data['PatientAge'].values
    return filenames, labels, meta


def load_rsna_age_three_split(rsna_dir: str = RSNA_DIR,
                              train_age: float = 'avg',
                              anomaly: str = 'lungOpacity',
                              for_supervised: bool = False):
    """Load data with gender balanced val and test sets. The ratio between male
    and female samples in the training set can be controlled.
    lo = lung opacity
    oa = other anomaly
    """
    assert train_age in ['young', 'avg', 'old']
    assert anomaly in ['lungOpacity', 'otherAnomaly']

    # Load metadata
    metadata = pd.read_csv(os.path.join(THIS_DIR, 'rsna_metadata.csv'))
    normal_data = metadata[metadata.label == 0]
    if anomaly == 'lungOpacity':
        data = metadata[metadata.label == 1]
    else:
        data = metadata[metadata.label == 2]

    # Filter ages over 110 years (outliers)
    normal_data = normal_data[normal_data.PatientAge < 110]
    data = data[data.PatientAge < 110]

    # Split data into bins by age
    n_bins = 3
    t = np.histogram(normal_data.PatientAge, bins=n_bins)[1]

    normal_young = normal_data[normal_data.PatientAge < t[1]]
    normal_avg = normal_data[(normal_data.PatientAge >= t[1]) & (normal_data.PatientAge < t[2])]
    normal_old = normal_data[normal_data.PatientAge >= t[2]]
    young = data[data.PatientAge < t[1]]
    avg = data[(data.PatientAge >= t[1]) & (data.PatientAge < t[2])]
    old = data[data.PatientAge >= t[2]]

    # Save 100 male and 100 female samples for every label for validation and test
    # Normal
    vc_normal_young = normal_young.sample(n=50, random_state=42)
    vc_normal_avg = normal_avg.sample(n=50, random_state=42)
    vc_normal_old = normal_old.sample(n=50, random_state=42)
    # Anomalous
    vc_young = young.sample(n=100, random_state=42)
    vc_avg = avg.sample(n=100, random_state=42)
    vc_old = old.sample(n=100, random_state=42)
    val_young = vc_young.iloc[:50, :]
    val_avg = vc_avg.iloc[:50, :]
    val_old = vc_old.iloc[:50, :]
    test_young = vc_young.iloc[50:, :]
    test_avg = vc_avg.iloc[50:, :]
    test_old = vc_old.iloc[50:, :]
    # Rest of anomal data
    rest_young = young[~young.patientId.isin(vc_young.patientId)]
    rest_avg = avg[~avg.patientId.isin(vc_avg.patientId)]
    rest_old = old[~old.patientId.isin(vc_old.patientId)]

    # Aggregate validation and test sets and shuffle
    val_young = pd.concat([vc_normal_young, val_young]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_avg = pd.concat([vc_normal_avg, val_avg]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_old = pd.concat([vc_normal_old, val_old]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_young = pd.concat([vc_normal_young, test_young]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_avg = pd.concat([vc_normal_avg, test_avg]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_old = pd.concat([vc_normal_old, test_old]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Rest for training
    rest_normal_young = normal_young[~normal_young.patientId.isin(vc_normal_young.patientId)]
    rest_normal_avg = normal_avg[~normal_avg.patientId.isin(vc_normal_avg.patientId)]
    rest_normal_old = normal_old[~normal_old.patientId.isin(vc_normal_old.patientId)]
    n_samples = min(len(rest_normal_young), len(rest_normal_avg), len(rest_normal_old))
    train_young = rest_normal_young.sample(n=n_samples, random_state=42)
    train_avg = rest_normal_avg.sample(n=n_samples, random_state=42)
    train_old = rest_normal_old.sample(n=n_samples, random_state=42)

    # Mix train with anomal data if supervised
    if for_supervised:
        n_young = len(train_young)
        train_young = pd.concat([train_young[:n_young // 2], rest_young.sample(n=n_young // 2, random_state=42)])
        n_avg = len(train_avg)
        train_avg = pd.concat([train_avg[:n_avg // 2], rest_avg.sample(n=n_avg // 2, random_state=42)])
        n_old = len(train_old)
        train_old = pd.concat([train_old[:n_old // 2], rest_old.sample(n=n_old // 2, random_state=42)])

    if train_age == 'young':
        train = train_young
    elif train_age == 'avg':
        train = train_avg
    elif train_age == 'old':
        train = train_old
    else:
        raise ValueError("train_age must be 'young', 'avg' or 'old'.")

    print(f"Using {n_samples} {train_age} samples for training.")

    # Return
    filenames = {}
    labels = {}
    meta = {}
    sets = {
        'train': train,
        f'val/{anomaly}_young': val_young,
        f'val/{anomaly}_avg': val_avg,
        f'val/{anomaly}_old': val_old,
        f'test/{anomaly}_young': test_young,
        f'test/{anomaly}_avg': test_avg,
        f'test/{anomaly}_old': test_old,
    }
    img_dir = os.path.join(rsna_dir, 'stage_2_train_images')
    for mode, data in sets.items():
        filenames[mode] = [f'{img_dir}/{patient_id}.dcm' for patient_id in data.patientId]
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = data['PatientAge'].values
    return filenames, labels, meta


if __name__ == '__main__':
    # download_rsna()
    # extract_metadata()
    pass
