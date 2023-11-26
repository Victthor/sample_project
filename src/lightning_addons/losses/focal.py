

from typing import Optional, Literal
import torch
from torch import Tensor
from torch.nn import functional as f


# todo: make loss combining focal per sample and focal per class


def multiclass_focal_loss(
        inputs: Tensor,  # logits
        targets: Tensor,
        weight: Optional[Tensor] = None,  # alpha
        ignore_index: int = -1,
        gamma: float = 2.0,
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        temperature: Optional[float] = None,  # todo: separate temp of focal term and cross entropy term
) -> Tensor:

    if inputs.ndim > 2:
        # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
        c = inputs.shape[1]
        inputs = inputs.permute(0, *range(2, inputs.ndim), 1).reshape(-1, c)
        # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
        targets = targets.view(-1)

    # mask out the positions where labels equal the ignore_index
    if ignore_index >= 0:
        unignored_mask = targets != ignore_index
        targets = targets[unignored_mask]
        if len(targets) == 0:
            return torch.tensor(0.)
        inputs = inputs[unignored_mask]

    if temperature is not None:
        inputs /= temperature

    hot_log_probs = f.log_softmax(inputs, dim=-1)

    # first calculate nll loss without reduction (including alpha == weight)
    nll_loss = f.nll_loss(hot_log_probs, targets, weight=weight, ignore_index=ignore_index, reduction='none')

    # second focal term
    hot_log_probs = hot_log_probs[torch.arange(len(inputs)), targets]  # select probs of the target class
    hot_probs = hot_log_probs.exp()
    focal_term = (1 - hot_probs) ** gamma

    # final product
    loss = focal_term * nll_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


class MulticlassFocalLoss(torch.nn.Module):

    def __init__(
            self,
            weight: Optional[Tensor] = None,  # alpha
            ignore_index: Optional[int] = None,
            gamma: float = 2.0,
            reduction: Literal['mean', 'sum', 'none'] = 'mean',
            temperature: Optional[float] = None,
    ):
        super().__init__()

        self.weight = weight
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return multiclass_focal_loss(
            inputs,
            targets,
            self.weight,
            self.ignore_index,
            self.gamma,
            self.reduction,
            self.temperature,
        )


if __name__ == '__main__':
    # Set device to CPU
    device = torch.device("cpu")

    batch_size = 2048
    n_classes = 23

    in_logits = torch.randn(batch_size, n_classes, device=device)
    labels = torch.randint(0, n_classes, size=(batch_size, ), device=device)

    mfl = MulticlassFocalLoss(
        weight=None,
        # ignore_index,
        gamma=0.0,
        # reduction,
        # temperature=10,
    )

    floss = mfl(in_logits, labels)
