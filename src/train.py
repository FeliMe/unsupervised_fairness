import os
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from time import time

import pandas as pd
import torch
import wandb

from src.data.datasets import get_dataloaders
from src.models.models import init_model
from src.utils.metrics import AvgDictMeter, build_metrics
from src.utils.utils import seed_everything

""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""

parser = ArgumentParser()
# General script settings
parser.add_argument('--initial_seed', type=int, default=1, help='Random seed')
parser.add_argument('--num_seeds', type=int, default=1, help='Number of random seed')
parser.add_argument('--debug', action='store_true', help='Debug mode')
parser.add_argument('--disable_wandb', action='store_true', help='Debug mode')

# Experiment settings
parser.add_argument('--experiment_name', type=str, default='')

# Data settings
parser.add_argument('--dataset', type=str, default='mimic-cxr',
                    choices=['mimic-cxr', 'cxr14', 'chexpert'])
parser.add_argument('--protected_attr', type=str, default='sex',
                    choices=['none', 'age', 'sex', 'race', 'intersectional_age_sex_race'])
parser.add_argument('--male_percent', type=float, default=0.5)
parser.add_argument('--old_percent', type=float, default=0.5)
parser.add_argument('--white_percent', type=float, default=0.5)
parser.add_argument('--img_size', type=int, default=128, help='Image size')
parser.add_argument('--max_train_samples', type=int, default=None, help='Max number of training samples')
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of workers for dataloader')

# Logging settings
parser.add_argument('--val_frequency', type=int, default=1000,
                    help='Validation frequency')
parser.add_argument('--val_steps', type=int, default=50,
                    help='Steps per validation')
parser.add_argument('--log_frequency', type=int, default=100,
                    help='Logging frequency')
parser.add_argument('--log_img_freq', type=int, default=1000)
parser.add_argument('--num_imgs_log', type=int, default=8)
parser.add_argument(
    '--log_dir', type=str, help="Logging directory",
    default=os.path.join(
        'logs',
        datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S")
    )
)

# Hyperparameters
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay')
parser.add_argument('--max_steps', type=int, default=10000,
                    help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

# Model settings
parser.add_argument('--model_type', type=str, default='FAE',
                    choices=['FAE', 'RD'])
# FAE settings
parser.add_argument('--fae_hidden_dims', type=int, nargs='+',
                    default=[100, 150, 200, 250, 300],
                    help='Autoencoder hidden dimensions')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--loss_fn', type=str, default='ssim', help='loss function',
                    choices=['mse', 'ssim'])
parser.add_argument('--extractor_cnn_layers', type=str, nargs='+',
                    default=['layer0', 'layer1', 'layer2'])
parser.add_argument('--keep_feature_prop', type=float, default=1.0,
                    help='Proportion of ResNet features to keep')

config = parser.parse_args()

config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
config.seed = config.initial_seed

if config.debug:
    config.num_workers = 0
    config.max_steps = 1
    config.val_frequency = 1
    config.log_frequency = 1


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""


print("Loading data...")
t_load_data_start = time()
train_loader, val_loader, test_loader = get_dataloaders(
    dataset=config.dataset,
    batch_size=config.batch_size,
    img_size=config.img_size,
    num_workers=config.num_workers,
    protected_attr=config.protected_attr,
    male_percent=config.male_percent,
    old_percent=config.old_percent,
    white_percent=config.white_percent,
    max_train_samples=config.max_train_samples
)
print(f'Loaded datasets in {time() - t_load_data_start:.2f}s')


""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""


def train_step(model, optimizer, x, y, meta, device):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    y = y.to(device)
    meta = meta.to(device)
    loss_dict = model.loss(x, y=y)
    loss = loss_dict['loss']
    loss.backward()
    optimizer.step()
    return loss_dict


def train(train_loader, val_loader, config, log_dir):
    # Reproducibility
    print(f"Setting seed to {config.seed}...")
    seed_everything(config.seed)

    # Init model
    _, model, optimizer = init_model(config)

    # Init logging
    if not config.debug:
        os.makedirs(log_dir, exist_ok=True)
    wandb_tags = [config.model_type, config.dataset, config.protected_attr]
    wandb.init(
        project='unsupervised-fairness',
        entity='felix-meissen',
        dir=log_dir,
        name=log_dir.lstrip('logs/'),
        tags=wandb_tags,
        config=config,
        mode="disabled" if (config.debug or config.disable_wandb) else "online"
    )

    print('Starting training...')
    step = 0
    i_epoch = 0
    train_losses = AvgDictMeter()
    t_start = time()
    while True:
        for x, y, meta in train_loader:
            # Catch batches with only one sample
            if len(x) == 1:
                continue

            step += 1

            loss_dict = train_step(model, optimizer, x, y, meta, config.device)
            train_losses.add(loss_dict)

            if step % config.log_frequency == 0:
                train_results = train_losses.compute()
                # Print training loss
                log_msg = " - ".join([f'{k}: {v:.4f}' for k, v in train_results.items()])
                log_msg = f"Iteration {step} - " + log_msg
                # Elapsed time
                elapsed_time = datetime.utcfromtimestamp(time() - t_start)
                log_msg += f" - time: {elapsed_time.strftime('%H:%M:%S')}s"
                # Estimate remaining time
                time_per_step = (time() - t_start) / step
                remaining_steps = config.max_steps - step
                remaining_time = remaining_steps * time_per_step
                remaining_time = datetime.utcfromtimestamp(remaining_time)
                log_msg += f" - remaining time: {remaining_time.strftime('%H:%M:%S')}"
                print(log_msg)

                # Log to w&b or tensorboard
                wandb.log({f'train/{k}': v for k, v in train_results.items()}, step=step)

                # Reset
                train_losses.reset()

            if step % config.val_frequency == 0:
                log_imgs = step % config.log_img_freq == 0
                val_results = validate(config, model, val_loader, step,
                                       log_imgs)
                # Log to w&b
                wandb.log(val_results, step=step)

            if step >= config.max_steps:
                print(f'Reached {config.max_steps} iterations.',
                      'Finished training.')

                # Final validation
                print("Final validation...")
                validate(config, model, val_loader, step, False)
                return model

        i_epoch += 1


""""""""""""""""""""""""""""""""" Validation """""""""""""""""""""""""""""""""


def val_step(model, x, y, meta, device):
    model.eval()
    x = x.to(device)
    y = y.to(device)
    meta = meta.to(device)
    with torch.no_grad():
        loss_dict = model.loss(x, y=y)
        anomaly_map, anomaly_score = model.predict_anomaly(x)
    x = x.cpu()
    y = y.cpu()
    anomaly_score = anomaly_score.cpu() if anomaly_score is not None else None
    anomaly_map = anomaly_map.cpu() if anomaly_map is not None else None
    return loss_dict, anomaly_map, anomaly_score


def validate(config, model, loader, step, log_imgs=False):
    i_step = 0
    device = next(model.parameters()).device
    x, y, meta = next(iter(loader))
    metrics = build_metrics(subgroup_names=list(x.keys()))
    losses = defaultdict(AvgDictMeter)
    imgs = defaultdict(list)
    anomaly_maps = defaultdict(list)

    for x, y, meta in loader:
        # Catch batches with only one sample
        if len(x) == 1:
            continue

        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        for i, k in enumerate(x.keys()):
            loss_dict, anomaly_map, anomaly_score = val_step(model, x[k], y[k], meta[k], device)

            # Update metrics
            group = torch.tensor([i] * len(anomaly_score))
            metrics.update(group, anomaly_score, y[k])
            losses[k].add(loss_dict)

            imgs[k].append(x[k])
            if anomaly_map is not None:
                anomaly_maps[k].append(anomaly_map)
            else:
                log_imgs = False

        i_step += 1
        if i_step >= config.val_steps:
            break

    # Compute and flatten metrics and losses
    metrics_c = metrics.compute()
    losses_c = {k: v.compute() for k, v in losses.items()}
    losses_c = {f'{k}_{m}': v[m] for k, v in losses_c.items() for m in v.keys()}
    if log_imgs:
        imgs = {f'{k}_imgs': wandb.Image(torch.cat(v)[:config.num_imgs_log]) for k, v in imgs.items()}
        anomaly_maps = {f'{k}_anomaly_maps': wandb.Image(torch.cat(v)[:config.num_imgs_log]) for k, v in anomaly_maps.items()}
        imgs_log = {**imgs, **anomaly_maps}
        wandb.log(imgs_log, step=step)

    # Compute metrics
    results = {**metrics_c, **losses_c}

    # Print validation results
    print("\nval results:")
    log_msg = "\n".join([f'{k}: {v:.4f}' for k, v in results.items()])
    log_msg += "\n"
    print(log_msg)

    return results


""""""""""""""""""""""""""""""""" Testing """""""""""""""""""""""""""""""""


def test(config, model, loader, log_dir):
    print("Testing...")

    device = next(model.parameters()).device
    x, y, meta = next(iter(loader))
    metrics = build_metrics(subgroup_names=list(x.keys()))
    losses = defaultdict(AvgDictMeter)
    anomaly_scores = []
    labels = []
    subgroup_names = []

    for x, y, meta in loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        for i, k in enumerate(x.keys()):
            loss_dict, _, anomaly_score = val_step(model, x[k], y[k], meta[k], device)

            # Update metrics
            group = torch.tensor([i] * len(anomaly_score))
            metrics.update(group, anomaly_score, y[k])
            losses[k].add(loss_dict)

            # Store anomaly scores, labels, and subgroup names
            anomaly_scores.append(anomaly_score)
            labels.append(y[k])
            subgroup_names += [k] * len(anomaly_score)

    # Aggregate anomaly scores, labels, and subgroup names
    anomaly_scores = torch.cat(anomaly_scores)
    labels = torch.cat(labels)

    # Compute and flatten metrics and losses
    metrics_c = metrics.compute()
    losses_c = {k: v.compute() for k, v in losses.items()}
    losses_c = {f'{k}_{m}': v[m] for k, v in losses_c.items() for m in v.keys()}

    results = {**metrics_c, **losses_c}

    # Print validation results
    print("\nTest results:")
    log_msg = "\n".join([f'{k}: {v:.4f}' for k, v in results.items()])
    log_msg += "\n"
    print(log_msg)

    # Write test results to wandb summary
    for k, v in results.items():
        wandb.run.summary[k] = v

    # Save test results to csv
    if not config.debug:
        csv_path = os.path.join(log_dir, 'test_results.csv')
        results_df = pd.DataFrame({k: v.item() for k, v in metrics_c.items()}, index=[0])
        for k, v in vars(config).items():
            results_df[k] = pd.Series([v])
        results_df.to_csv(csv_path, index=False)

    # Save anomaly scores and labels to csv
    if not config.debug:
        # os.makedirs(log_dir, exist_ok=True)
        anomaly_score_df = pd.DataFrame({
            'anomaly_score': anomaly_scores,
            'label': labels,
            'subgroup_name': subgroup_names
        })
        csv_path = os.path.join(log_dir, 'anomaly_scores.csv')
        anomaly_score_df['male_percent'] = config.male_percent
        anomaly_score_df['old_percent'] = config.old_percent
        anomaly_score_df['white_percent'] = config.white_percent
        anomaly_score_df.to_csv(csv_path, index=False)


""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""


if __name__ == '__main__':
    for i in range(config.num_seeds):
        config.seed = config.initial_seed + i
        log_dir = os.path.join(config.log_dir, f'seed_{config.seed}')
        print(f"Starting run {i + 1}/{config.num_seeds} with seed {config.seed}. Logging to {log_dir}...")
        model = train(train_loader, val_loader, config, log_dir)
        test(config, model, test_loader, log_dir)
