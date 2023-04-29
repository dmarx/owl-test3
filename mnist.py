"""
The following file implements a simple variational auto-encoder in PyTorch.
The architecture is parameterized by a Hydra config file.
The training boilerplate uses PyTorch Lightning and DeepSpeed for distributed training.

Usage:
    python train.py --config-file /path/to/config.yaml

Requirements:
    - PyTorch
    - Hydra
    - PyTorch Lightning
    - DeepSpeed

This module exports the following classes:

Classes:
    VAE(nn.Module): Variational Auto Encoder model
    VAETrainer(pl.LightningModule): PyTorch Lightning trainer for VAE model

Functions:
    train_from_config(config: DictConfig, rank: Optional[int]=None) -> None
        Starts VAE trainer based on the provided hydra config
"""

import os
from typing import Optional
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.logging import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.optim import Adam
from vae_model import VAE
from vae_data import VAEData

def train_from_config(config: DictConfig, rank: Optional[int] = None) -> None:
    """
    Starts VAE trainer based on the provided Hydra config

    Args:
        config (DictConfig): Hydra config containing VAE and Trainer settings
        rank (Optional[int]): rank of the trainer for distributed training
    """
    # setup data
    data_module = VAEData(config=config.data)
    train_loader = DataLoader(
        data_module.train,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        data_module.val,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        shuffle=False,
        pin_memory=True
    )

    # model
    model = VAE(config.model)

    # optimizer
    optimizer = Adam(model.parameters(), lr=config.train.learning_rate)

    # lightning trainer
    trainer = Trainer(
        logger=TensorBoardLogger(save_dir=os.getcwd()),
        gpus=config.train.gpus,
        max_epochs=config.train.max_epochs,
        accelerator='ddp' if config.train.gpus > 1 else None,
        deterministic=True,
        profiler=SimpleProfiler(profile_memory=True, record_shapes=True),
        precision=config.train.precision,
        plugins=DeepSpeed(config.train.deepspeed) if config.train.deepspeed.enabled else None,
        distributed_backend='deepspeed' if config.train.deepspeed.enabled else None,
        checkpoint_callback=False,
        fast_dev_run=config.debug,
        progress_bar_refresh_rate=20,
        limit_train_batches=config.train.limit_train_batches,
        limit_val_batches=config.train.limit_val_batches,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    # Load configuration
    config = OmegaConf.load('/path/to/config.yaml')

    # Train the model
    train_from_config(config=config)
"""
