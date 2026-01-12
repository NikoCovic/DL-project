from __future__ import annotations

from itertools import islice
from typing import Tuple

import torch
from pyhessian import hessian


def pyhessian_sharpness(
    model: torch.nn.Module,
    loader,
    num_batches: int = 10,
) -> Tuple[float, float]:
    """Compute top Hessian eigenvalue and relative sharpness.

    Relative sharpness is computed by scaling the top Hessian eigenvalue by the
    squared Frobenius norm of the model parameters.
    """

    model = model._orig_mod if hasattr(model, "_orig_mod") else model
    criterion = torch.nn.CrossEntropyLoss()
    limited_loader = list(islice(loader, num_batches))

    hessian_comp = hessian(model, criterion, dataloader=limited_loader, cuda=True)
    lambda_max = hessian_comp.eigenvalues(top_n=1)[0][0]

    total_norm_sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_norm_sq += torch.norm(p.data) ** 2

    relative_sharpness = lambda_max * total_norm_sq.item()
    return lambda_max, relative_sharpness
