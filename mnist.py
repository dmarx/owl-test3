"""
The following file implements a simple variational auto-encoder in PyTorch.
The architecture is parameterized by a Hydra config file.
The training boilerplate uses PyTorch Lightning and DeepSpeed for distributed training.

Usage:
    python train.py --config-file /path/to/config.yaml

Example Config File:
    # model architecture
    model:
        input_dim: 784
        hidden_dim: 256
        latent_dim: 64

    # training parameters
    trainer:
        gpus: 2
        max_epochs: 100

    # optimizer
    optimizer:
        type: Adam
        lr: 0.001

    # dataset
    dataset:
        name: MNIST
        root: /path/to/data

Requirements:
    - Python 3.8 or higher
    - PyTorch 1.9.0 or higher
    - PyTorch Lightning 1.4.9 or higher
    - DeepSpeed 0.5.3 or higher

"""

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate
from deepspeed.ops.adam import FusedAdam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class VAE(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, logvar = self.model(x)
        loss = self.loss_fn(x_hat, x) + 0.5 * torch.sum((mu ** 2 + torch.exp(logvar) - logvar - 1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = FusedAdam(self.parameters(), lr=self.hparams.optimizer.lr)
        return optimizer


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, root: str, batch_size: int):
        super().__init__()
        self.root = root
        self.batch_size = batch_size

    def prepare_data(self):
        MNIST(self.root, train=True, download=True)
        MNIST(self.root, train=False, download=True)

    def setup(self, stage=None):
        transform = ToTensor()
        self.train_dataset = MNIST(self.root, train=True, transform=transform)
        self.val_dataset = MNIST(self.root, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)


def train(cfg: DictConfig):
    model = instantiate(cfg.model)
    vae = VAE(model)
    dm = MNISTDataModule(cfg.dataset.root, batch_size=cfg.trainer.batch_size)
    trainer = pl.Trainer(
        gpus=cfg.trainer.gpus,
        max_epochs=cfg.trainer.max_epochs,
        precision=16 if cfg.trainer.precision == 16 else 32,
        plugins=deepspeed.init_distributed(),
        accelerator="ddp" if cfg.trainer.gpus > 1 else None,
    )
    trainer.fit(vae, dm)


if __name__ == "__main__":
    import hydra
    import deepspeed

    @hydra.main(config_path="configs", config_name="config")
    def run(cfg: DictConfig):
        train(cfg)

    run()
"""
