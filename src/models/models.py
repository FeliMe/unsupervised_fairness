import torch

from src.models.FAE.fae import FeatureReconstructor
from src.models.RD.reverse_distillation import ReverseDistillation


def init_model(config):
    print("Initializing model...")
    if config.model_type == 'FAE':
        model = FeatureReconstructor(config)
    elif config.model_type == 'RD':
        model = ReverseDistillation(config)
    else:
        raise ValueError(f'Unknown model type {config.model_type}')
    model = model.to(config.device)
    compiled_model = torch.compile(model)

    # Log number of trainable parameters
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)

    return model, compiled_model, optimizer
