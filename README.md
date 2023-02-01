# Evaluating Fairness in Unsupervised Anomaly Detection Models


## Usage

Download this repository by running

```bash
git clone https://github.com/FeliMe/unsupervised_fairness
```

## Environment

Create and activate the Anaconda environment:

```bash
conda env create -f environment.yml
conda activate ad_fairness
```

Additionally, you need to install the repository as a package:

```bash
python3 -m pip install --editable .
```

To be able to use [Weights & Biases](https://wandb.ai) for logging follow the
instructions at https://docs.wandb.ai/quickstart.


## Data

To download the RSNA dataset, specify the environment variable ```RSNA_DIR```
to the directory where you want the RSNA dataset to be stored and run
```python src/data/rsna_pneumonia_detection.py```.
