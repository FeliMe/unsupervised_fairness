import json
from tqdm import tqdm
from typing import Callable, List, Tuple

import numpy as np


def write_memmap(files: List[str], filename: str, load_fn: Callable, target_size: Tuple[int, int]):
    """Write NumPy memmap file with separate JSON metadata file."""
    memmap_file = f"{filename}.dat"
    shape = (len(files), 1, *target_size)
    dtype = 'float32'
    fp = np.memmap(memmap_file, dtype=dtype, mode='w+', shape=shape)

    n_failed = 0
    failed = []
    for i, file in tqdm(enumerate(files), total=len(files)):
        try:
            fp[i] = load_fn(file)
        except Exception as e:
            n_failed += 1
            failed.append(file)
            print(f"Failed to load image '{file}': {e}")
            fp[i] = np.zeros(shape[1:], dtype=dtype)
            continue
    fp.flush()

    # Write metadata JSON file
    metadata = {
        'shape': shape,
        'dtype': str(dtype)
    }
    metadata_file = f"{filename}.json"
    with open(metadata_file, 'w') as file:
        json.dump(metadata, file)

    print(f"Failed to load {n_failed} images.")
    for file in failed:
        print(file)


def read_memmap(filename: str) -> np.ndarray:
    """
    Read NumPy memmap file with separate JSON metadata file.

    Args:
        filename (str): File name for the memmap file (without extension).

    Returns:
        numpy.ndarray: Loaded NumPy array.
    """
    # Read metadata JSON file
    metadata_file = f"{filename}.json"
    with open(metadata_file, 'r') as file:
        metadata = json.load(file)
        shape = tuple(metadata['shape'])
        dtype = np.dtype(metadata['dtype'])

    # Read memmap file
    memmap_file = f"{filename}.dat"
    data = np.memmap(memmap_file, dtype=dtype, shape=shape, mode='r')

    return data
