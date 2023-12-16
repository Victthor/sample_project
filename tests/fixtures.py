
import numpy as np
from typing import Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytest
from lightning import Trainer, LightningModule, LightningDataModule


NUM_CLASSES = 7
INPUT_SIZE = 27
DATASET_SIZE = 512
BATCH_SIZE = 32
MAX_EPOCHS = 1


class DummyPLLightningModule(LightningModule):
    def __init__(self, input_size, num_classes, lr=0.001):
        super(DummyPLLightningModule, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.lr = lr

        self.class_weights = None

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):

        inputs, targets = batch
        logits = self(inputs)
        targets = targets

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            weight=self.class_weights,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.fc.parameters(), lr=self.lr)


class SampleDataset(Dataset):
    def __init__(self):
        # self.targets = torch.randint(low=0, high=NUM_CLASSES, size=(DATASET_SIZE, ))
        # self.data = torch.randn((len(self.targets), INPUT_SIZE))
        self.targets = np.random.randint(low=0, high=NUM_CLASSES, size=(DATASET_SIZE,))
        self.data = np.random.randn(len(self.targets), INPUT_SIZE).astype(np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]


class DummyPLDataModule(LightningDataModule):
    def __init__(self, batch_size: int):
        super(DummyPLDataModule, self).__init__()
        self.batch_size = batch_size
        self._train_dataset = None

    def setup(self, stage=None):
        self._train_dataset = SampleDataset()

    def train_dataloader(self) -> DataLoader:
        # self._train_dataset = TensorDataset(
        #     torch.randn(len(self.targets), self.input_size),  # [dataset_size, input_size]
        #     torch.LongTensor(self.targets)  # [dataset_size]
        # )
        return DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=True)


@pytest.fixture
def datamodule() -> DummyPLDataModule:
    return DummyPLDataModule(batch_size=BATCH_SIZE)


@pytest.fixture
def trainer(datamodule):
    return Trainer(
        max_epochs=MAX_EPOCHS,
        fast_dev_run=True,
        # auto_lr_find=True,
        # datamodule=datamodule,
    )


@pytest.fixture
def pl_module():
    return DummyPLLightningModule(INPUT_SIZE, NUM_CLASSES)
