
from typing import Optional, Union
import numpy as np
import torch
import lightning as pl
from lightning.pytorch.callbacks import Callback
from sklearn.utils.class_weight import compute_class_weight


class ClassWeights(Callback):

    def __init__(self, class_weight: Optional[Union[str, dict]] = 'balanced'):
        self.class_weight = class_weight

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        targets = trainer.datamodule._train_dataset.targets
        class_weights = compute_class_weight(
            class_weight=self.class_weight,
            classes=np.unique(targets),
            y=targets
        )

        pl_module.class_weights = torch.tensor(class_weights, dtype=pl_module.dtype, device=pl_module.device).detach()
