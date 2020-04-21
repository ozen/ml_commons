import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from ml_commons.pytorch.lightning import AxLightningModule
from ml_commons.pytorch.models import StitchedModel


class TestModule(AxLightningModule):
    def __init__(self, config):
        super().__init__(config)

        resnet = models.resnet50(pretrained=True)

        self.classifier = StitchedModel(
            (resnet, 0, -1),
            nn.Linear(2048, 10)
        )

    def forward(self, batch):
        images, labels = batch
        logits = self.classifier(images)
        loss = F.cross_entropy(logits, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['hparams']['lr'])
        return optimizer

    def _get_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _get_dataloader(self, is_train):
        self.prepare_data()
        transform = self._get_transform()
        dataset = CIFAR10(root=self.config['data_root'], train=is_train,
                          transform=transform, download=False)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config['batch_size'],
            num_workers=0
        )
        return loader

    def prepare_data(self):
        transform = self._get_transform()
        _ = CIFAR10(root=self.config['data_root'], train=True,
                    transform=transform, download=True)

    def train_dataloader(self):
        return self._get_dataloader(True)

    def val_dataloader(self):
        return self._get_dataloader(False)
