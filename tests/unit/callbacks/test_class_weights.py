
import pytest
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

from lightning_addons.callbacks.class_weights import ClassWeights
from fixtures import trainer, pl_module, datamodule, NUM_CLASSES


class_weights_dict = {inx: inx + 1.5 for inx in range(NUM_CLASSES)}


@pytest.mark.parametrize(
    "class_weight", [None, "balanced", class_weights_dict]
)
def test_on_train_start(class_weight, trainer, pl_module, datamodule) -> None:

    trainer.fit(pl_module, datamodule)

    callback = ClassWeights(class_weight=class_weight)

    callback.on_train_start(trainer, pl_module)

    targets = trainer.datamodule._train_dataset.targets

    class_weights = compute_class_weight(
        class_weight=class_weight,
        classes=np.unique(targets),
        y=targets,
    )

    class_weights = torch.tensor(class_weights, dtype=pl_module.dtype, device=pl_module.device).detach()

    assert torch.equal(class_weights, pl_module.class_weights)


# if __name__ == '__main__':
#     trainer_ = trainer()
