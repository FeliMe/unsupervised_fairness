import os
from typing import List

from src.analysis.utils import gather_data_from_anomaly_scores

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def create_tables(experiment_dirs: List[str],
                  metrics: List[str],
                  subgroup_names: List[str],
                  attr_key: str):
    """Create the tables for the paper"""

    # Gather data
    data = []
    for experiment_dir in experiment_dirs:
        data_, attr_key_values = gather_data_from_anomaly_scores(
            experiment_dir, metrics, subgroup_names, attr_key)
        data.append(data_)

    for i_attr_key in range(len(attr_key_values)):
        line = f"${attr_key_values[i_attr_key]}$ &"
        for data_ in data:
            for metric in metrics:
                d = data_[metric][i_attr_key]
                line += f" ${d.mean():.3f}$ \\sd{{${d.std():.3f}$}} &"
        line = line[:-1] + "\\\\"
        line = line.replace(".", "{\\cdot}")
        print(line)


if __name__ == '__main__':
    """ Sex tables """
    print("Sex tables")
    experiment_dirs = [
        os.path.join(THIS_DIR, '../../logs/RD_mimic-cxr_sex'),
        os.path.join(THIS_DIR, '../../logs/RD_cxr14_sex'),
        os.path.join(THIS_DIR, '../../logs/RD_chexpert_sex'),
    ]
    create_tables(
        experiment_dirs=experiment_dirs,
        metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
        subgroup_names=["test/male", "test/female"],
        attr_key='male_percent',
    )
    """ Age tables """
    print("Age tables")
    experiment_dirs = [
        os.path.join(THIS_DIR, '../../logs/RD_mimic-cxr_age'),
        os.path.join(THIS_DIR, '../../logs/RD_cxr14_age'),
        os.path.join(THIS_DIR, '../../logs/RD_chexpert_age'),
    ]
    create_tables(
        experiment_dirs=experiment_dirs,
        metrics=["test/old_subgroupAUROC", "test/young_subgroupAUROC"],
        subgroup_names=["test/old", "test/young"],
        attr_key='old_percent',
    )
    """ Race tables """
    print("Race tables")
    experiment_dirs = [
        os.path.join(THIS_DIR, '../../logs/RD_mimic-cxr_race'),
    ]
    create_tables(
        experiment_dirs=experiment_dirs,
        metrics=["test/white_subgroupAUROC", "test/black_subgroupAUROC"],
        subgroup_names=["test/white", "test/black"],
        attr_key='white_percent',
    )
