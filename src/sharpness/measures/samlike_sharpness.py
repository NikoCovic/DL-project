from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def samlike_sharpness(
    model: torch.nn.Module,
    loader,
    rho: float = 0.05,
    num_batches: int = 10,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    total_sharpness = 0.0

    loader_iter = iter(loader)
    for batch_idx, (inputs, targets) in enumerate(loader_iter):
        if batch_idx >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        # 1. Compute base loss and gradients
        with torch.enable_grad():
            outputs = model(inputs)
            base_loss = F.cross_entropy(outputs, targets)
            model.zero_grad()
            base_loss.backward()

        # 2. Compute TRUE gradient norm (flatten and concatenate)
        grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        grad_norm = torch.linalg.norm(torch.cat(grads)) + 1e-12

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

        # Normalize by gradient norm squared and rho squared
        sharpness = (adv_loss - base_loss) / (grad_norm**2 * rho**2 + 1e-12)
        total_sharpness += sharpness.item()

        model.zero_grad()

    return total_sharpness / min(num_batches, len(loader))
