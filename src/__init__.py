import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
RSNA_DIR = os.environ.get('RSNA_DIR', '/datasets/RSNA')
CAMCAN_DIR = os.environ.get('CAMCAN_DIR', '/datasets/CamCAN')
BRATS_DIR = os.environ.get('BRATS_DIR', '/datasets/BraTS2020')
CXR14_DIR = os.environ.get('CXR14_DIR', '/datasets/CXR8')
MIMIC_CXR_DIR = os.environ.get('MIMIC-CXR_DIR', '/datasets/MIMIC-CXR/mimic-cxr-jpg_2-0-0')
