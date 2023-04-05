the following file implements a simple variational auto-encoder in pytorch.
the architecture is parameterized by a hydra config file.
the training boilerplate uses pytorch-lightning and deepspeed for distributed training.

```python
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from vae import VAE

class VAEDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_data = MNIST(self.data_dir, train=True, transform=ToTensor())
            self.val_data = MNIST(self.data_dir, train=False, transform=ToTensor())

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

class VAEModel(pl.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.hparams = hparams
        self.vae = VAE(hparams)

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.vae.loss(x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.vae.loss(x)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

class VAE:
    def __init__(self, hparams: DictConfig):
        self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Linear(hparams.input_dim, hparams.hidden_dim),
            nn.ReLU(),
            nn.Linear(hparams.hidden_dim, hparams.hidden_dim),
            nn.ReLU(),
            nn.Linear(hparams.hidden_dim, hparams.latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hparams.latent_dim, hparams.hidden_dim),
            nn.ReLU(),
            nn.Linear(hparams.hidden_dim, hparams.hidden_dim),
            nn.ReLU(),
            nn.Linear(hparams.hidden_dim, hparams.input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def loss(self, x):
        z, mu, logvar = self.encode(x.view(-1, self.hparams.input_dim))
        x_hat = self.decode(z)
        recon_loss = nn.functional.binary_cross_entropy(x_hat, x.view(-1, self.hparams.input_dim), reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + kl_div) / x.shape[0]

if __name__ == '__main__':
    from omegaconf import OmegaConf
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from hydra.experimental import compose, initialize

    with initialize(config_path="../config"):
        cfg = compose(config_name="config")

    data_module = VAEDataModule(data_dir=cfg.data_dir, batch_size=cfg.batch_size)
    model = VAEModel(hparams=cfg.model)
    logger = TensorBoardLogger(save_dir=cfg.log_dir, name=cfg.exp_name)
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.checkpoint_dir, filename='vae-{epoch:02d}-{val_loss:.2f}')

    trainer = Trainer(
        gpus=cfg.gpus,
        max_epochs=cfg.max_epochs,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        accelerator='ddp' if cfg.gpus > 1 else None,
        plugins=deepspeed.init_deepspeed(cfg.deepspeed) if cfg.deepspeed else None
    )

    trainer.fit(model, data_module)
```
