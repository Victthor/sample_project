
from typing import Optional, Union

import torch
import numpy
import torch.nn as nn
from lightning.pytorch import LightningModule
import torchmetrics

from lightning_addons.layers.simple_cnn import SimpleCNN
from lightning_addons.losses.focal import multiclass_focal_loss


class ConvModel(LightningModule):

    def __init__(
            self,
            n_classes: int,
            dropout: float = 0.1,
            lr: float = 0.001,
            class_weights: Optional[Union["numpy.array", "torch.Tensor"]] = None
    ):
        """

        Args:
            n_classes:
            dropout:
            lr:
            class_weights:
        """
        super().__init__()
        self.n_classes = n_classes
        self.lr = lr

        self.model = SimpleCNN(dropout=dropout)

        self.class_weights = class_weights
        self.logits = None
        self.targets = None
        self.val_logits = None
        self.val_targets = None

        self.val_acc_micro = torchmetrics.classification.MulticlassAccuracy(
            num_classes=n_classes,
            top_k=1,
            average='micro',
        )

        self.val_acc_macro = torchmetrics.classification.MulticlassAccuracy(
            num_classes=n_classes,
            top_k=1,
            average='macro',
        )

        self.val_auc_roc = torchmetrics.classification.MulticlassAUROC(
            num_classes=n_classes,
            average='none',
            thresholds=300,  # None
        )

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):

        inputs, targets = batch
        self.logits = self(inputs)
        self.targets = targets

        # if batch_idx == 0:
        #     self.class_weights = torch.ones([self.n_classes, ], dtype=torch.float32, device=self.device)
        #     self.class_weights[2] = 2.5
        #     self.class_weights[3] = 40
        #     self.class_weights[4] = 1.8
        #     self.class_weights[5] = 2.0

        # loss = nn.functional.cross_entropy(
        #     self.logits.view(-1, self.logits.size(-1)),
        #     targets.view(-1),
        #     ignore_index=-1,
        #     weight=self.class_weights,
        # )

        loss = multiclass_focal_loss(
            self.logits,
            self.targets,
            weight=self.class_weights,
            ignore_index=-1,
            gamma=2.0,
            temperature=None,
        )

        per_class_loss = torch.zeros([self.n_classes, ], dtype=self.dtype, device=self.device)

        # cur_unique_classes = torch.unique(targets)
        # cur_n_classes = cur_unique_classes.size()[0]
        # cur_per_class_loss = torch.zeros([cur_n_classes, ], dtype=self.dtype, device=self.device)
        #
        # for inx, i_cur_class in enumerate(cur_unique_classes):
        #     i_class_logits = self.logits[targets == i_cur_class]
        #     i_targets = i_cur_class * torch.ones([i_class_logits.size()[0], ], dtype=torch.int64, device=self.device)
        #     i_loss = nn.functional.cross_entropy(i_class_logits, i_targets, ignore_index=-1)
        #     cur_per_class_loss[inx] = i_loss
        #
        # std, mean = torch.std_mean(cur_per_class_loss)

        for i_class in range(self.n_classes):
            i_class_logits = self.logits[targets == i_class]

            if i_class_logits.size()[0] > 0:
                i_targets = i_class * torch.ones([i_class_logits.size()[0], ], dtype=torch.int64, device=self.device)
                i_loss = nn.functional.cross_entropy(i_class_logits, i_targets, ignore_index=-1)
                per_class_loss[i_class] = i_loss
                self.log(f'train_loss_{i_class}', i_loss, on_step=True, on_epoch=True, logger=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        self.val_logits = logits
        self.val_targets = targets

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            # weight=self.class_weights,
        )

        per_class_loss = torch.zeros([self.n_classes, ], dtype=torch.float32, device=self.device)

        for i_class in range(self.n_classes):
            i_class_logits = logits[targets == i_class]

            if i_class_logits.size()[0] > 0:
                i_targets = i_class * torch.ones([i_class_logits.size()[0], ], dtype=torch.int64, device=self.device)
                i_loss = nn.functional.cross_entropy(i_class_logits, i_targets, ignore_index=-1)
                per_class_loss[i_class] = i_loss
                self.log(f'val_loss_{i_class}', i_loss, on_epoch=True, logger=True)

        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True)

        aucs = self.val_auc_roc(logits, targets)
        for i_class, auc in enumerate(aucs):
            self.log(f'val_auc_{i_class}', auc, on_step=False, on_epoch=True)

        self.val_acc_micro.update(logits, targets)
        self.val_acc_macro.update(logits, targets)

        self.log('val_acc_micro', self.val_acc_micro, on_step=False, on_epoch=True)
        self.log('val_acc_macro', self.val_acc_macro, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # return torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


if __name__ == '__main__':
    import sys
    print(sys.path)

    print(dir())

    import os

    print(os.getcwd())

    model = ConvModel(n_classes=10)
