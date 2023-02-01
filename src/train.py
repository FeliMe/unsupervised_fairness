import os

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from time import time

import torch

import wandb

from src.models.model import FeatureReconstructor
from src.data.datasets import get_rsna_dataloaders
from src.data.rsna_pneumonia_detection import RSNA_DIR
from src.utils.metrics import build_metrics, AvgDictMeter
from src.utils.utils import seed_everything


""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""

parser = ArgumentParser()
# General script settings
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--debug', action='store_true', help='Debug mode')

# Data settings
parser.add_argument('--data_dir', type=str, default=RSNA_DIR)
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
model = FeatureReconstructor(config).to(config.device)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                             weight_decay=config.weight_decay)
# Print model
# print(model.enc)
# print(model.dec)

# Init metrics
metrics = build_metrics()


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""


print("Loading data...")
t_load_data_start = time()
train_loader, val_loader, _ = get_rsna_dataloaders(
    rsna_dir=config.data_dir,
    batch_size=config.batch_size,
    img_size=config.img_size,
    num_workers=config.num_workers,
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
    config=config,
    mode="disabled" if config.debug else "online"
)


def train_step(model, optimizer, x, device):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    loss_dict = model.loss(x)
    loss = loss_dict['rec_loss']
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
                validate(model, val_loader, config.device, step)

            if step >= config.max_steps:
                print(f'Reached {config.max_steps} iterations.',
                      'Finished training.')

                # Final validation
                print("Final validation...")
                validate(model, val_loader, config.device, step)
                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({step} iterations)')


def val_step(model, x, device):
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        loss_dict = model.loss(x)
        anomaly_map, anomaly_score = model.predict_anomaly(x)
    x = x.cpu()
    return loss_dict, anomaly_map.cpu(), anomaly_score.cpu()


def validate(model, val_loader, device, step):
    i_val_step = 0
    metrics = defaultdict(build_metrics)
    val_losses = defaultdict(AvgDictMeter)

    for x, y in val_loader:
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        for k in x.keys():
            loss_dict, _, anomaly_score = val_step(model, x[k], device)
            val_losses[k].add(loss_dict)

            # Update metrics
            metrics[k].update(anomaly_score, y[k])

        i_val_step += 1
        if i_val_step >= config.val_steps:
            break

    # Compute and flatten metrics and losses
    metrics_c = {k: v.compute() for k, v in metrics.items()}
    metrics_c = {f'{k}_{m}': v[m] for k, v in metrics_c.items() for m in v.keys()}
    val_losses_c = {k: v.compute() for k, v in val_losses.items()}
    val_losses_c = {f'{k}_{m}': v[m] for k, v in val_losses_c.items() for m in v.keys()}

    # Compute metrics
    val_results = {**metrics_c, **val_losses_c}

    # Print validation results
    print("\nValidation results:")
    log_msg = " - ".join([f'{k}: {v:.4f}' for k, v in val_results.items()])
    log_msg += "\n"
    print(log_msg)

    # Log to w&b
    wandb.log(val_results, step=step)


if __name__ == '__main__':
    train(model, optimizer, train_loader, val_loader, config)
