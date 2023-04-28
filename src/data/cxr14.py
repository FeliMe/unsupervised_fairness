"""
Split CXR14 dataset.
train_ad: 40000 no finding samples from train_val_list.txt
train_cls: 90% of the remaining samples from train_val_list.txt
val_cls: 10% of the remaining samples from train_val_list.txt
test_cls: all samples from test_list.txt
"""
import os

import pandas as pd

from src import CXR14_DIR

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSNAMES = [  # All data
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
    'No Finding',  # 60361
]
CLASSMAPPING = {c: i for i, c in enumerate(CLASSNAMES)}


def load_metadata(data_dir: str = CXR14_DIR):
    """Loads metadata (filenames, labels, age, gender) for each sample of the
    CXR14 dataset."""
    class_info = pd.read_csv(os.path.join(data_dir, 'Data_Entry_2017.csv'))
    metadata = class_info[[
        "Image Index", "Patient Age", "Patient Gender", "Finding Labels"]]

    train_list = pd.read_csv(os.path.join(data_dir, 'train_val_list.txt'),
                             header=None, names=["Image Index"])
    test_list = pd.read_csv(os.path.join(data_dir, 'test_list.txt'),
                            header=None, names=["Image Index"])

    # Prepend the path to the image filename
    metadata["Image Index"] = f"{data_dir}/images/" + metadata["Image Index"]

    # Convert labels into a one-hot encoding
    def convert_to_list(label):
        return [CLASSMAPPING[c] for c in label.split('|')]

    def convert_to_one_hot(label):
        one_hot = [0] * len(CLASSNAMES)
        for c in label:
            one_hot[c] = 1
        return one_hot

    metadata["Finding Labels"] = metadata["Finding Labels"].apply(convert_to_list)
    metadata["Finding Labels"] = metadata["Finding Labels"].apply(convert_to_one_hot)
    i = 1


if __name__ == '__main__':
    load_metadata()
