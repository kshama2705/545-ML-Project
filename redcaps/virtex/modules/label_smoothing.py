import torch
from torch import nn
from torch.nn import functional as F


class CrossEntropyLossWithLabelSmoothing(nn.Module):
    r"""
    PyTorch :class:`~torch.nn.CrossEntropyLoss` with label smoothing. Quoting
    documentation from original PyTorch module:

    It is useful when training a classification problem with ``C`` classes.
    The ``inputs`` is expected to contain raw, unnormalized scores for each class.

    ``inputs`` has to be a Tensor of size either ``(N, C)``. This criterion
    expects a class index in the range ``[0, C - 1]`` as the ``targets`` for each
    value of a 1D tensor of size ``minibatch``; if ``ignore_index`` is specified,
    this criterion also accepts this class index (this index may not necessarily be
    in the class range).

    Parameters
    ----------
    smoothing: float, optional (default = 0.1)
        Label smoothing value. It sets target weights as ``(1 - smoothing)``
        and all other weights as ``smoothing / (C - 1)``. Setting this to
        zero will default to vanilla cross entropy loss.
    """

    def __init__(self, smoothing: float = 0.0, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):

        if self.smoothing == 0.0:
            # Use PyTorch cross entropy when smoothing is 0. This is slightly
            # faster than what we are doing manually below.
            return F.cross_entropy(
                inputs, targets, ignore_index=self.ignore_index, reduction="mean"
            )

        # Remove entries matching ``ignore_index``.
        if self.ignore_index >= 0:
            _targets = targets[targets != self.ignore_index]
            _inputs = inputs[targets != self.ignore_index]

        # shape: (batch_size, num_classes)
        logprobs = F.log_softmax(_inputs, dim=-1)

        # shape: (batch_size, num_classes)
        weights = (
            torch.ones_like(_inputs) * self.smoothing / (_inputs.size(-1) - 1.0)
        )
        weights.scatter_(-1, _targets.unsqueeze(-1), (1.0 - self.smoothing))

        # shape: (batch_size, )
        loss = (- weights * logprobs).sum(dim=-1)
        return loss.mean()
