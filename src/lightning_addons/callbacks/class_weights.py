
from typing import Optional, Union
import numpy as np
import torch
import lightning as pl
from lightning.pytorch.callbacks import Callback
from sklearn.utils.class_weight import compute_class_weight


class ClassWeights(Callback):
    """
    Callback to calculate and set class weights for imbalanced classification in a PyTorch Lightning training process.

    Args:
        class_weight: Specifies the method for computing class weights.
            It can be 'balanced' (default), a dictionary of class weights, or None.
            If 'balanced', it uses sklearn's `compute_class_weight` with the 'balanced' strategy.
            If a dictionary, it directly uses the provided class weights.
            If None, it defaults to equal class weights.

    Methods:
        on_train_start(trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            Called when the training begins. Computes class weights based on the provided or default strategy
            and sets the calculated weights to the `class_weights` attribute of the PyTorch Lightning module.

    Attributes:
        class_weight (Optional[Union[str, dict]]): Specifies the method for computing class weights.
            It can be 'balanced' (default), a dictionary of class weights, or None.

    Note:
        The computed class weights are used to address class imbalance during training in classification tasks.
    """
    def __init__(self, class_weight: Optional[Union[str, dict]] = 'balanced'):
        self.class_weight = class_weight

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Method called when the training starts. Computes class weights based on the provided or default strategy
        and sets the calculated weights to the `class_weights` attribute of the PyTorch Lightning module.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer.
            pl_module (pl.LightningModule): The PyTorch Lightning module.

        Returns:
            None
        """

        targets = trainer.datamodule._train_dataset.targets
        class_weights = compute_class_weight(
            class_weight=self.class_weight,
            classes=np.unique(targets),
            y=targets
        )

        pl_module.class_weights = torch.tensor(class_weights, dtype=pl_module.dtype, device=pl_module.device).detach()
