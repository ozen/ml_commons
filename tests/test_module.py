import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from ml_commons.pytorch.lightning import AxLightningModule


class TestModule(AxLightningModule):
    def __init__(self, config):
        super().__init__(config)

        self.classifier = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits

    def loss(self, logits, labels):
        loss = F.cross_entropy(logits, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['hparams']['lr'])
        return optimizer

    def _get_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,))
        ])

    def _get_dataloader(self, is_train):
        self.prepare_data()
        transform = self._get_transform()
        dataset = MNIST(root=self.config['data_root'], train=is_train,
                        transform=transform, download=False)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config['batch_size'],
            num_workers=0
        )
        return loader

    def prepare_data(self):
        transform = self._get_transform()
        _ = MNIST(root=self.config['data_root'], train=True,
                  transform=transform, download=True)

    def train_dataloader(self):
        return self._get_dataloader(True)

    def val_dataloader(self):
        return self._get_dataloader(False)
