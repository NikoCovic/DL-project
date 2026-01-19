import torch
from typing import Callable, Iterable, Tuple, Dict

Tensor = torch.Tensor


def adaptive_sharpness(
    model: torch.nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    metric_batch,
    adaptive_sharpness_config,
) -> float:
    val = adaptive_sharpness_implementation(
        model=model,
        loss_fn=loss_fn,
        batch=metric_batch,
        rho=adaptive_sharpness_config.rho,
        eta=adaptive_sharpness_config.eta,
        ascent_steps=adaptive_sharpness_config.ascent_steps,
        ascent_lr=adaptive_sharpness_config.ascent_lr,
        use_eval_mode=adaptive_sharpness_config.use_eval_mode,
    )
    return float(val.item())

def adaptive_sharpness_implementation(
    model: torch.nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    batch: Tuple[Tensor, Tensor],
    rho: float,
    eta: float = 1e-2,
    ascent_steps: int = 20,
    ascent_lr: float = 1e-1,
    use_eval_mode: bool = False,
) -> Tensor:

    assert rho >= 0.0
    assert ascent_steps >= 1

    x, y = batch

    was_training = model.training
    if use_eval_mode:
        model.eval()

    params: Iterable[torch.nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
    w0: Dict[torch.nn.Parameter, Tensor] = {p: p.detach().clone() for p in params}

    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        base_loss = loss_fn(model(x), y)
        base_loss.backward()

        best_loss = base_loss.detach()

        e_tilde: Dict[torch.nn.Parameter, Tensor] = {
            p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in params
        }

        for i in range(ascent_steps):
            for p in params:
                if p.grad is None:
                    continue
                Tw = w0[p].abs().add(eta)
                grad_tilde = Tw * p.grad
                e_tilde[p].add_(ascent_lr * grad_tilde)

            sq_norm = torch.zeros((), device=base_loss.device)
            for p in params:
                sq_norm = sq_norm + e_tilde[p].pow(2).sum()
            norm = sq_norm.sqrt().clamp_min(1e-12)

            if norm > rho:
                scale = rho / norm
                for p in params:
                    e_tilde[p].mul_(scale)

            for p in params:
                Tw = w0[p].abs().add(eta)
                p.data.copy_(w0[p] + Tw * e_tilde[p])

            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)

            if loss.detach() > best_loss:
                best_loss = loss.detach()

            if i < ascent_steps - 1:
                loss.backward()

        for p in params:
            p.data.copy_(w0[p])
        
        model.zero_grad(set_to_none=True)

    if use_eval_mode and was_training:
        model.train()

    return (best_loss - base_loss.detach())
