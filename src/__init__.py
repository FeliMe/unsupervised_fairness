import os

SEED = 42

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
CXR14_DIR = os.environ.get('CXR14_DIR', '/datasets/CXR8')
MIMIC_CXR_DIR = os.environ.get('MIMIC-CXR_DIR', '/datasets/MIMIC-CXR/mimic-cxr-jpg_2-0-0')
CHEXPERT_DIR = os.environ.get('CHEXPERT_DIR', '/datasets/CheXpert')
