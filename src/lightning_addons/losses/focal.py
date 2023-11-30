

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
    """
    Compute the multiclass focal loss.

    Parameters:
        inputs (Tensor): Logits predicted by the model.
        targets (Tensor): Ground truth class labels.
        weight (Optional[Tensor]): Weight tensor, typically representing class weights (optional).
        ignore_index (int): Index to ignore in the loss calculation.
        gamma (float): Focusing parameter for the focal term.
        reduction (Literal['mean', 'sum', 'none']): Specifies the reduction to apply to the loss.
            Options: 'mean', 'sum', 'none'.
        temperature (Optional[float]): Temperature parameter for adjusting the logits (optional).

    Returns:
        Tensor: Computed focal loss.

    Notes:
        If `inputs` has more than two dimensions, it is reshaped to (N * d1 * ... * dK, C), where N is the
        number of samples, and C is the number of classes. The targets are also reshaped accordingly.

        If `ignore_index` is specified, positions where labels equal `ignore_index` are masked out.

        If `temperature` is provided, the logits are divided by the temperature value.

        The focal loss is calculated as a combination of negative log-likelihood (nll) loss and a focal term.
    """
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
    """
    PyTorch module for computing the multiclass focal loss.

    Methods:
        forward(inputs: Tensor, targets: Tensor) -> Tensor:
            Forward pass of the module. Computes the multiclass focal loss using the `multiclass_focal_loss` function.

    Attributes:
        weight (Optional[Tensor]): Weight tensor, typically representing class weights (optional).
        ignore_index (Optional[int]): Index to ignore in the loss calculation (optional).
        gamma (float): Focusing parameter for the focal term.
        reduction (Literal['mean', 'sum', 'none']): Specifies the reduction to apply to the loss.
        temperature (Optional[float]): Temperature parameter for adjusting the logits (optional).

    Notes:
        The module wraps the `multiclass_focal_loss` function and allows using it as a part of PyTorch model architecture.
    """
    def __init__(
            self,
            weight: Optional[Tensor] = None,  # alpha
            ignore_index: Optional[int] = None,
            gamma: float = 2.0,
            reduction: Literal['mean', 'sum', 'none'] = 'mean',
            temperature: Optional[float] = None,
    ):
        """
        Constructor for the MulticlassFocalLoss module.

        Parameters:
            weight (Optional[Tensor]): Weight tensor, typically representing class weights (optional).
            ignore_index (Optional[int]): Index to ignore in the loss calculation (optional).
            gamma (float): Focusing parameter for the focal term.
            reduction (Literal['mean', 'sum', 'none']): Specifies the reduction to apply to the loss.
                Options: 'mean', 'sum', 'none'.
            temperature (Optional[float]): Temperature parameter for adjusting the logits (optional).
        """
        super().__init__()

        self.weight = weight
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of the MulticlassFocalLoss module.

        Parameters:
            inputs (Tensor): Logits predicted by the model.
            targets (Tensor): Ground truth class labels.

        Returns:
            Tensor: Computed focal loss.

        Notes:
            The forward pass uses the `multiclass_focal_loss` function to compute the loss.
        """
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
