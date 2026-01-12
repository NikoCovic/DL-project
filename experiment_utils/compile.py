import torch
import torch.nn as nn

def compile_for_training(model: nn.Module) -> nn.Module:
    """Compile model for training while keeping VRAM headroom for heavy analysis.

    On some systems, `mode="max-autotune"` enables CUDA Graphs and allocates large
    private pools that are great for throughput but can starve Hessian/SAM analysis.
    """
    try:
        return torch.compile(model, mode="max-autotune-no-cudagraphs")
    except Exception:
        pass
    try:
        # Older/newer torch versions may not support the mode string but do support options.
        return torch.compile(model, mode="max-autotune", options={"triton.cudagraphs": False})
    except Exception:
        return torch.compile(model, mode="max-autotune")
