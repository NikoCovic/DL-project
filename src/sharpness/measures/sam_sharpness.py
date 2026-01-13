from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def sam_sharpness(
    model: torch.nn.Module,
    data_batch,
    rho: float = 0.05,
) -> float:
    model.eval()
    
    inputs, targets = data_batch
    # inputs, targets = inputs.to(device), targets.to(device)

    # 1. Compute base loss and gradients
    with torch.enable_grad():
        outputs = model(inputs)
        base_loss = F.cross_entropy(outputs, targets)
        model.zero_grad()
        base_loss.backward()

    # 2. Compute gradient norm (stack of per-tensor norms)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_norm = (
        torch.linalg.norm(torch.stack([torch.linalg.norm(g) for g in grads])) + 1e-12
    )

    # 3. Perturb parameters
    scale = rho / grad_norm
    epsilons = []
    for p in model.parameters():
        if p.grad is not None:
            eps = p.grad * scale
            p.add_(eps)
            epsilons.append(eps)

    # 4. Compute perturbed loss
    adv_loss = F.cross_entropy(model(inputs), targets)

    # 5. Restore parameters
    for p, eps in zip((p for p in model.parameters() if p.grad is not None), epsilons):
        p.sub_(eps)

    sharpness = (adv_loss - base_loss).item()
    model.zero_grad()

    return sharpness