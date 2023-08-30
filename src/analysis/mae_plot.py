import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter

from src import LOG_DIR

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20


def assign_dataset(filename: str):
    if 'chexpert' in filename.lower():
        return 'CheXpert'
    elif 'cxr14' in filename.lower():
        return 'CXR14'
    elif 'mimic' in filename.lower():
        return 'MIMIC-CXR'
    else:
        raise ValueError(f"Unknown dataset: {filename}")


def assign_attribute(filename: str):
    if 'sex' in filename.lower():
        return 'Gender'
    elif 'age' in filename.lower():
        return 'Age'
    elif 'race' in filename.lower():
        return 'Race'
    else:
        raise ValueError(f"Unknown attribute: {filename}")


model = 'FAE'
mae_files = glob(os.path.join(LOG_DIR, 'maes', '*.npy'))
mae_files = [f for f in mae_files if model in f]

df = []
for mae_file in mae_files:
    experiment = mae_file.split('/')[-1].split('_')[0]
    maes = np.load(mae_file)
    for i, mae in enumerate(maes):
        df.append({
            'MAE': mae,
            'Dataset': assign_dataset(mae_file),
            'Attribute': assign_attribute(mae_file),
        })

df = pd.DataFrame(df)

df_gender = df[df['Attribute'] == 'Gender']
print(f"Gender: {df_gender.MAE.mean():.4f} +/- {df_gender.MAE.std():.4f}")
df_age = df[df['Attribute'] == 'Age']
print(f"Age: {df_age.MAE.mean():.4f} +/- {df_age.MAE.std():.4f}")
df_race = df[df['Attribute'] == 'Race']
print(f"Race: {df_race.MAE.mean():.4f} +/- {df_race.MAE.std():.4f}")

fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8))
fig.patch.set_facecolor('#F3F3F3')
ax.patch.set_facecolor('#F3F3F3')
sns.boxplot(data=df, x='Attribute', y='MAE', hue='Dataset', order=['Gender', 'Age', 'Race'])
ylimit = ax.get_ylim()
ax.set_yticks(np.linspace(ylimit[0], ylimit[1], num=5))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.title('Mean absolute errors (MAE) when using the "fairness laws"')
plt.xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(THIS_DIR, f'maes_{model}.pdf'))
