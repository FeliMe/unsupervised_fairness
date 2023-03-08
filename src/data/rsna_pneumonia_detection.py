import os
from glob import glob
from tqdm import tqdm

import kaggle
import numpy as np
import pandas as pd
import pydicom as dicom


RSNA_DIR = os.environ.get('RSNA_DIR', '/datasets/RSNA')
CLASS_MAPPING = {
    'Normal': 0,  # 8851, female: 2905, male: 4946, age mean: 44.94, std: 16.39, min: 2, max: 155
    'Lung Opacity': 1,  # 6012, female: 2502, male: 3510, age mean: 45.58, std: 17.46, min: 1, max: 92
    'No Lung Opacity / Not Normal': 2  # 11821, female: 5111, male: 6710, age mean: 49.33, std: 16.49, min: 1, max: 153
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
    metadata.to_csv(os.path.join(rsna_dir, 'train_metadata.csv'), index=False)


def load_rsna_naive_split(rsna_dir: str = RSNA_DIR):
    """Naive train/val/test split."""
    # Load metadata
    metadata = pd.read_csv(os.path.join(rsna_dir, 'train_metadata.csv'))
    normal_data = metadata[metadata.label == 0]
    lo_data = metadata[metadata.label == 1]
    oa_data = metadata[metadata.label == 2]

    # Use 8051 normal samples for training
    train = normal_data.sample(n=8051, random_state=42)

    # Rest for validation and test
    rest_normal = normal_data[~normal_data.patientId.isin(train.patientId)]
    val_normal = rest_normal.sample(n=400, random_state=42)
    test_normal = rest_normal[~rest_normal.patientId.isin(val_normal.patientId)]
    val_test_lo = lo_data.sample(n=800, random_state=42)
    val_lo = val_test_lo.iloc[:400, :]
    test_lo = val_test_lo.iloc[400:, :]
    val_test_oa = oa_data.sample(n=800, random_state=42)
    val_oa = val_test_oa.iloc[:400, :]
    test_oa = val_test_oa.iloc[400:, :]

    # Concatenate and shuffle
    val_lo = pd.concat([val_normal, val_lo]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_oa = pd.concat([val_normal, val_oa]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_lo = pd.concat([test_normal, test_lo]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_oa = pd.concat([test_normal, test_oa]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Return
    filenames = {}
    labels = {}
    meta = {}
    sets = {
        'train': train,
        'val/lungOpacity': val_lo,
        'val/otherAnomaly': val_oa,
        'test/lungOpacity': test_lo,
        'test/otherAnomaly': test_oa,
    }
    img_dir = os.path.join(rsna_dir, 'stage_2_train_images')
    for mode, data in sets.items():
        filenames[mode] = [f'{img_dir}/{patient_id}.dcm' for patient_id in data.patientId]
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.zeros_like(data['PatientSex'].values)
    return filenames, labels, meta


def load_rsna_gender_split(rsna_dir: str = RSNA_DIR, male_percent: float = 0.5):
    """Load data with gender balanced val and test sets. The ratio between male
    and female samples in the training set can be controlled.
    lo = lung opacity
    oa = other anomaly
    """
    assert 0.0 <= male_percent <= 1.0
    female_percent = 1 - male_percent

    # Load metadata
    metadata = pd.read_csv(os.path.join(rsna_dir, 'train_metadata.csv'))
    normal_data = metadata[metadata.label == 0]
    lo_data = metadata[metadata.label == 1]
    oa_data = metadata[metadata.label == 2]

    normal_male = normal_data[normal_data.PatientSex == 'M']
    normal_female = normal_data[normal_data.PatientSex == 'F']
    lo_male = lo_data[lo_data.PatientSex == 'M']
    lo_female = lo_data[lo_data.PatientSex == 'F']
    oa_male = oa_data[oa_data.PatientSex == 'M']
    oa_female = oa_data[oa_data.PatientSex == 'F']

    # Save 100 male and 100 female samples for every label for validation and test
    # Normal
    val_test_normal_male = normal_male.sample(n=50, random_state=42)
    val_test_normal_female = normal_female.sample(n=50, random_state=42)
    # Lung opacity
    val_test_lo_male = lo_male.sample(n=100, random_state=42)
    val_test_lo_female = lo_female.sample(n=100, random_state=42)
    val_lo_male = val_test_lo_male.iloc[:50, :]
    val_lo_female = val_test_lo_female.iloc[:50, :]
    test_lo_male = val_test_lo_male.iloc[50:, :]
    test_lo_female = val_test_lo_female.iloc[50:, :]
    # Other anomaly
    val_test_oa_male = oa_male.sample(n=100, random_state=42)
    val_test_oa_female = oa_female.sample(n=100, random_state=42)
    val_oa_male = val_test_oa_male.iloc[:50, :]
    val_oa_female = val_test_oa_female.iloc[:50, :]
    test_oa_male = val_test_oa_male.iloc[50:, :]
    test_oa_female = val_test_oa_female.iloc[50:, :]
    # Aggregate validation and test sets and shuffle
    val_lo_male = pd.concat([val_test_normal_male, val_lo_male]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_lo_female = pd.concat([val_test_normal_female, val_lo_female]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_lo_male = pd.concat([val_test_normal_male, test_lo_male]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_lo_female = pd.concat([val_test_normal_female, test_lo_female]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_oa_male = pd.concat([val_test_normal_male, val_oa_male]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_oa_female = pd.concat([val_test_normal_female, val_oa_female]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_oa_male = pd.concat([val_test_normal_male, test_oa_male]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_oa_female = pd.concat([val_test_normal_female, test_oa_female]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Rest for training
    rest_normal_male = normal_male[~normal_male.patientId.isin(val_test_normal_male.patientId)]
    rest_normal_female = normal_female[~normal_female.patientId.isin(val_test_normal_female.patientId)]
    n_samples = min(len(rest_normal_male), len(rest_normal_female))
    n_male = int(n_samples * male_percent)
    n_female = int(n_samples * female_percent)
    train_male = rest_normal_male.sample(n=n_male, random_state=42)
    train_female = rest_normal_female.sample(n=n_female, random_state=42)
    # Aggregate training set and shuffle
    train = pd.concat([train_male, train_female]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Using {n_male} male and {n_female} female samples for training.")

    # Return
    filenames = {}
    labels = {}
    meta = {}
    sets = {
        'train': train,
        'val/lungOpacity_male': val_lo_male,
        'val/lungOpacity_female': val_lo_female,
        'val/otherAnomaly_male': val_oa_male,
        'val/otherAnomaly_female': val_oa_female,
        'test/lungOpacity_male': test_lo_male,
        'test/lungOpacity_female': test_lo_female,
        'test/otherAnomaly_male': test_oa_male,
        'test/otherAnomaly_female': test_oa_female
    }
    img_dir = os.path.join(rsna_dir, 'stage_2_train_images')
    for mode, data in sets.items():
        filenames[mode] = [f'{img_dir}/{patient_id}.dcm' for patient_id in data.patientId]
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = data['PatientSex'].values
    return filenames, labels, meta


def load_rsna_age_split(rsna_dir: str = RSNA_DIR, train_age: float = 'avg'):
    """Load data with gender balanced val and test sets. The ratio between male
    and female samples in the training set can be controlled.
    lo = lung opacity
    oa = other anomaly
    """
    assert train_age in ['young', 'avg', 'old']

    # Load metadata
    metadata = pd.read_csv(os.path.join(rsna_dir, 'train_metadata.csv'))
    normal_data = metadata[metadata.label == 0]
    lo_data = metadata[metadata.label == 1]
    oa_data = metadata[metadata.label == 2]

    # Filter ages over 110 years (outliers)
    normal_data = normal_data[normal_data.PatientAge < 110]
    lo_data = lo_data[lo_data.PatientAge < 110]
    oa_data = oa_data[oa_data.PatientAge < 110]

    # Split data into bins by age
    n_bins = 3
    t = np.histogram(normal_data.PatientAge, bins=n_bins)[1]

    normal_young = normal_data[normal_data.PatientAge < t[1]]
    normal_avg = normal_data[(normal_data.PatientAge >= t[1]) & (normal_data.PatientAge < t[2])]
    normal_old = normal_data[normal_data.PatientAge >= t[2]]
    lo_young = lo_data[lo_data.PatientAge < t[1]]
    lo_avg = lo_data[(lo_data.PatientAge >= t[1]) & (lo_data.PatientAge < t[2])]
    lo_old = lo_data[lo_data.PatientAge >= t[2]]
    oa_young = oa_data[oa_data.PatientAge < t[1]]
    oa_avg = oa_data[(oa_data.PatientAge >= t[1]) & (oa_data.PatientAge < t[2])]
    oa_old = oa_data[oa_data.PatientAge >= t[2]]

    # Save 100 male and 100 female samples for every label for validation and test
    # Normal
    vc_normal_young = normal_young.sample(n=50, random_state=42)
    vc_normal_avg = normal_avg.sample(n=50, random_state=42)
    vc_normal_old = normal_old.sample(n=50, random_state=42)
    # Lung opacity
    vc_lo_young = lo_young.sample(n=100, random_state=42)
    vc_lo_avg = lo_avg.sample(n=100, random_state=42)
    vc_lo_old = lo_old.sample(n=100, random_state=42)
    val_lo_young = vc_lo_young.iloc[:50, :]
    val_lo_avg = vc_lo_avg.iloc[:50, :]
    val_lo_old = vc_lo_old.iloc[:50, :]
    test_lo_young = vc_lo_young.iloc[50:, :]
    test_lo_avg = vc_lo_avg.iloc[50:, :]
    test_lo_old = vc_lo_old.iloc[50:, :]
    # Other anomaly
    vc_oa_young = oa_young.sample(n=100, random_state=42)
    vc_oa_avg = oa_avg.sample(n=100, random_state=42)
    vc_oa_old = oa_old.sample(n=100, random_state=42)
    val_oa_young = vc_oa_young.iloc[:50, :]
    val_oa_avg = vc_oa_avg.iloc[:50, :]
    val_oa_old = vc_oa_old.iloc[:50, :]
    test_oa_young = vc_oa_young.iloc[50:, :]
    test_oa_avg = vc_oa_avg.iloc[50:, :]
    test_oa_old = vc_oa_old.iloc[50:, :]

    # Aggregate validation and test sets and shuffle
    val_lo_young = pd.concat([vc_normal_young, val_lo_young]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_lo_avg = pd.concat([vc_normal_avg, val_lo_avg]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_lo_old = pd.concat([vc_normal_old, val_lo_old]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_lo_young = pd.concat([vc_normal_young, test_lo_young]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_lo_avg = pd.concat([vc_normal_avg, test_lo_avg]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_lo_old = pd.concat([vc_normal_old, test_lo_old]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_oa_young = pd.concat([vc_normal_young, val_oa_young]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_oa_avg = pd.concat([vc_normal_avg, val_oa_avg]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_oa_old = pd.concat([vc_normal_old, val_oa_old]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_oa_young = pd.concat([vc_normal_young, test_oa_young]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_oa_avg = pd.concat([vc_normal_avg, test_oa_avg]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_oa_old = pd.concat([vc_normal_old, test_oa_old]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Rest for training
    rest_normal_young = normal_young[~normal_young.patientId.isin(vc_normal_young.patientId)]
    rest_normal_avg = normal_avg[~normal_avg.patientId.isin(vc_normal_avg.patientId)]
    rest_normal_old = normal_old[~normal_old.patientId.isin(vc_normal_old.patientId)]
    n_samples = min(len(rest_normal_young), len(rest_normal_avg), len(rest_normal_old))
    if train_age == 'young':
        train = rest_normal_young.sample(n=n_samples, random_state=42)
    elif train_age == 'avg':
        train = rest_normal_avg.sample(n=n_samples, random_state=42)
    elif train_age == 'old':
        train = rest_normal_old.sample(n=n_samples, random_state=42)
    else:
        raise ValueError("train_age must be 'young', 'avg' or 'old'.")

    print(f"Using {n_samples} {train_age} samples for training.")

    # Return
    filenames = {}
    labels = {}
    meta = {}
    sets = {
        'train': train,
        'val/lungOpacity_young': val_lo_young,
        'val/lungOpacity_avg': val_lo_avg,
        'val/lungOpacity_old': val_lo_old,
        'val/otherAnomaly_young': val_oa_young,
        'val/otherAnomaly_avg': val_oa_avg,
        'val/otherAnomaly_old': val_oa_old,
        'test/lungOpacity_young': test_lo_young,
        'test/lungOpacity_avg': test_lo_avg,
        'test/lungOpacity_old': test_lo_old,
        'test/otherAnomaly_young': test_oa_young,
        'test/otherAnomaly_avg': test_oa_avg,
        'test/otherAnomaly_old': test_oa_old
    }
    img_dir = os.path.join(rsna_dir, 'stage_2_train_images')
    for mode, data in sets.items():
        filenames[mode] = [f'{img_dir}/{patient_id}.dcm' for patient_id in data.patientId]
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = data['PatientAge'].values
    return filenames, labels, meta


if __name__ == '__main__':
    download_rsna()
    extract_metadata()
