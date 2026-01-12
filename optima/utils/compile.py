# import os
# import warnings

# import torch
# import torch.nn as nn

# def compile_for_training(model: nn.Module) -> nn.Module:
#     """Compile model for training while keeping VRAM headroom for heavy analysis.

#     On some systems, `mode="max-autotune"` enables CUDA Graphs and allocates large
#     private pools that are great for throughput but can starve Hessian/SAM analysis.
#     """
#     if os.environ.get("OPTIMA_DISABLE_TORCH_COMPILE", "0") == "1":
#         return model

#     try:
#         return torch.compile(model, mode="max-autotune-no-cudagraphs")
#     except (ModuleNotFoundError, ImportError) as e:
#         warnings.warn(
#             "torch.compile() unavailable due to missing/broken optional dependency "
#             f"({type(e).__name__}: {e}). Falling back to eager mode. "
#             "Set OPTIMA_DISABLE_TORCH_COMPILE=1 to silence this warning.",
#             RuntimeWarning,
#         )
#         return model
