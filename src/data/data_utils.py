from tqdm import tqdm
from typing import Callable, List, Tuple

import h5py
import pydicom as dicom
import torch

from PIL import Image
from torch import Tensor
from torchvision import transforms


def load_dicom_img(filename: str) -> Tensor:
    """Loads a DICOM image."""
    ds = dicom.dcmread(filename)
    img = torch.tensor(ds.pixel_array, dtype=torch.float32) / 255.0
    return img[None]  # (1, H, W)


def load_png_img_grayscale(filename: str) -> Tensor:
    """Loads a PNG image."""
    img = Image.open(filename).convert('L')
    img = transforms.ToTensor()(img)
    return img


def write_hf5_file(files: List[str], output_file: str, load_fn: Callable, target_size: Tuple[int, int]):
    """Write hf5 file with images and labels."""
    # Write images to hf5 file
    hf5_file = h5py.File(output_file, 'w')
    hf5_file.create_dataset('images', (len(files), 1, *target_size), dtype='float32')

    n_failed = 0
    failed = []
    for i, file in tqdm(enumerate(files), total=len(files)):
        try:
            image = load_fn(file)
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


if __name__ == '__main__':
    img = load_dicom_img('/datasets/RSNA/stage_2_train_images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm')
    print(img)
