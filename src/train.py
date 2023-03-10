import os
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from time import time

import torch
import torch.nn as nn
import wandb
from torch import Tensor

from src.data.datasets import get_dataloaders
from src.models.DeepSVDD.deepsvdd import DeepSVDD
from src.models.FAE.fae import FeatureReconstructor
from src.models.RD.reverse_distillation import ReverseDistillation
from src.utils.metrics import AvgDictMeter, build_metrics
from src.utils.utils import seed_everything

""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""

parser = ArgumentParser()
# General script settings
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--debug', action='store_true', help='Debug mode')

# Data settings
parser.add_argument('--dataset', type=str, default='camcan', choices=['rsna', 'camcan', 'camcan/brats'])
parser.add_argument('--protected_attr', type=str, default='age',
                    choices=['none', 'age', 'sex'])
parser.add_argument('--male_percent', type=float, default=0.5)
parser.add_argument('--train_age', type=str, default='avg',
                    choices=['young', 'avg', 'old'])
parser.add_argument('--img_size', type=int, default=128, help='Image size')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers for dataloader')

# Logging settings
parser.add_argument('--val_frequency', type=int, default=200,
                    help='Validation frequency')
parser.add_argument('--val_steps', type=int, default=50,
                    help='Steps per validation')
parser.add_argument('--log_frequency', type=int, default=50,
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
parser.add_argument('--max_steps', type=int, default=8000,  # 10000
                    help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

# Model settings
parser.add_argument('--model_type', type=str, default='FAE',
                    choices=['FAE', 'RD', 'DeepSVDD'])
# FAE settings
parser.add_argument('--hidden_dims', type=int, nargs='+',
                    default=[100, 150, 200, 300],
                    help='Autoencoder hidden dimensions')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--loss_fn', type=str, default='ssim', help='loss function',
                    choices=['mse', 'ssim'])
parser.add_argument('--extractor_cnn_layers', type=str, nargs='+',
                    default=['layer0', 'layer1', 'layer2'])
parser.add_argument('--keep_feature_prop', type=float, default=1.0,
                    help='Proportion of ResNet features to keep')
# DeepSVDD settings
parser.add_argument('--repr_dim', type=int, default=256,
                    help='Dimensionality of the hypersphere c')

config = parser.parse_args()

# Select training device
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

if config.debug:
    config.num_workers = 0
    config.max_steps = 1
    config.val_frequency = 1
    config.val_steps = 1
    config.log_frequency = 1


""""""""""""""""""""""""""""""" Reproducibility """""""""""""""""""""""""""""""
seed_everything(config.seed)


""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""


print("Initializing model...")
if config.model_type == 'FAE':
    model = FeatureReconstructor(config)
elif config.model_type == 'RD':
    model = ReverseDistillation()
elif config.model_type == 'DeepSVDD':
    model = DeepSVDD(config)
else:
    raise ValueError(f'Unknown model type {config.model_type}')
model = model.to(config.device)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                             weight_decay=config.weight_decay)


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
    train_age=config.train_age,
)
print(f'Loaded datasets in {time() - t_load_data_start:.2f}s')


""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""
if not config.debug:
    os.makedirs(config.log_dir, exist_ok=True)
wandb.init(
    project='unsupervised-fairness',
    entity='felix-meissen',
    dir=config.log_dir,
    name=config.log_dir.lstrip('logs/'),
    tags=[config.model_type, config.dataset, config.protected_attr],
    config=config,
    mode="disabled" if config.debug else "online"
)


def train_step(model, optimizer, x, device):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    loss_dict = model.loss(x)
    loss = loss_dict['loss']
    loss.backward()
    optimizer.step()
    return loss_dict


def train(model, optimizer, train_loader, val_loader, config):
    print('Starting training...')
    step = 0
    i_epoch = 0
    train_losses = AvgDictMeter()
    t_start = time()
    while True:
        for x, _ in train_loader:
            step += 1

            loss_dict = train_step(model, optimizer, x, config.device)
            train_losses.add(loss_dict)

            if step % config.log_frequency == 0:
                train_results = train_losses.compute()
                # Print training loss
                log_msg = " - ".join([f'{k}: {v:.4f}' for k, v in train_results.items()])
                log_msg = f"Iteration {step} - " + log_msg
                log_msg += f" - time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to w&b or tensorboard
                wandb.log({f'train/{k}': v for k, v in train_results.items()}, step=step)

                # Reset
                train_losses.reset()

            if step % config.val_frequency == 0:
                log_imgs = step % config.log_img_freq == 0
                val_results = validate(config, model, val_loader, step, 'val',
                                       log_imgs)
                # Log to w&b
                wandb.log(val_results, step=step)

            if step >= config.max_steps:
                print(f'Reached {config.max_steps} iterations.',
                      'Finished training.')

                # Final validation
                print("Final validation...")
                validate(config, model, val_loader, step, 'val', False)
                return model

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({step} iterations)')


def val_step(model: nn.Module, x: Tensor, device):
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        loss_dict = model.loss(x)
        anomaly_map, anomaly_score = model.predict_anomaly(x)
    x = x.cpu()
    anomaly_score = anomaly_score.cpu() if anomaly_score is not None else None
    anomaly_map = anomaly_map.cpu() if anomaly_map is not None else None
    return loss_dict, anomaly_map, anomaly_score


def validate(config, model, loader, step, mode, log_imgs=False):
    assert mode in ['val', 'test']
    i_step = 0
    device = next(model.parameters()).device
    x, y = next(iter(loader))
    metrics = build_metrics(subgroup_names=list(x.keys()))
    losses = defaultdict(AvgDictMeter)
    imgs = defaultdict(list)
    anomaly_scores = defaultdict(list)
    anomaly_maps = defaultdict(list)

    for x, y in loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        for i, k in enumerate(x.keys()):
            loss_dict, anomaly_map, anomaly_score = val_step(model, x[k], device)

            # Update metrics
            group = torch.tensor([i] * len(anomaly_score))
            scores_and_group = torch.stack([anomaly_score, group], dim=1)
            metrics.update(scores_and_group, y[k])
            losses[k].add(loss_dict)
            anomaly_scores[k].append(anomaly_score)

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
    anomaly_scores_c = {f'{k}_anomaly_scores': torch.cat(v).mean() for k, v in anomaly_scores.items()}
    losses_c = {k: v.compute() for k, v in losses.items()}
    losses_c = {f'{k}_{m}': v[m] for k, v in losses_c.items() for m in v.keys()}
    if log_imgs:
        imgs = {f'{k}_imgs': wandb.Image(torch.cat(v)[:config.num_imgs_log]) for k, v in imgs.items()}
        anomaly_maps = {f'{k}_anomaly_maps': wandb.Image(torch.cat(v)[:config.num_imgs_log]) for k, v in anomaly_maps.items()}
        imgs_log = {**imgs, **anomaly_maps}
        wandb.log(imgs_log, step=step)

    # Compute metrics
    results = {**metrics_c, **losses_c, **anomaly_scores_c}

    # Print validation results
    print(f"\n{mode} results:")
    log_msg = " - ".join([f'{k}: {v:.4f}' for k, v in results.items()])
    log_msg += "\n"
    print(log_msg)

    return results


def test(config, model, loader):
    print("Testing...")
    test_results = validate(config, model, loader, 0, 'test', True)
    for k, v in test_results.items():
        wandb.run.summary[k] = v


if __name__ == '__main__':
    model = train(model, optimizer, train_loader, val_loader, config)
    test(config, model, test_loader)
