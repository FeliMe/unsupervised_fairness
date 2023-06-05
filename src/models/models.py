import torch

from src.models.DeepSVDD.deepsvdd import DeepSVDD
from src.models.FAE.fae import FeatureReconstructor
from src.models.RD.reverse_distillation import ReverseDistillation
from src.models.supervised.resnet import ResNet18
from src.models.UncertaintyAE.uncertainty_ae import UncertaintyAE


def init_model(config):
    print("Initializing model...")
    if config.model_type == 'FAE':
        model = FeatureReconstructor(config)
    elif config.model_type == 'RD':
        model = ReverseDistillation(config)
    elif config.model_type == 'DeepSVDD':
        model = DeepSVDD(config)
    elif config.model_type == 'ResNet18':
        model = ResNet18(config)
    elif config.model_type == 'UncertaintyAE':
        model = UncertaintyAE(config)
    else:
        raise ValueError(f'Unknown model type {config.model_type}')
    model = model.to(config.device)
    compiled_model = torch.compile(model)

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)

    return model, compiled_model, optimizer
