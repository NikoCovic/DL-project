from torch.optim import Optimizer, Muon, RMSprop, SGD, Adam
from .muonpreconditioner import MuonPreconditioner
from .rmsproppreconditioner import RMSpropPreconditioner
from .sgdlrpreconditioner import SGDLRPreconditioner
from .adampreconditioner import AdamPreconditioner
from torch.nn import Module


def fetch_preconditioner(optim:Optimizer, model:Module):
    if isinstance(optim, Muon):
        return MuonPreconditioner(optim, model)
    elif isinstance(optim, RMSprop):
        return RMSpropPreconditioner(optim, model)
    elif isinstance(optim, SGD):
        return SGDLRPreconditioner(optim, model)
    elif isinstance(optim, Adam):
        return AdamPreconditioner(optim, model)
    else:
        raise Exception(f"No preconditioner exists yet for Optimizer of type: {type(optim)}")