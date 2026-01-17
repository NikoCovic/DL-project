from torch.optim import Optimizer, Muon, RMSprop, SGD, Adam, SGD
from .preconditioner.muonpreconditioner import MuonPreconditioner
from .preconditioner.rmsproppreconditioner import RMSpropPreconditioner
from .preconditioner.sgdlrpreconditioner import SGDLRPreconditioner
from .preconditioner.adampreconditioner import AdamPreconditioner
from torch.nn import Module, Parameter
from src.sharpness.airbench94_muon import VanillaMuon
from src.edge_of_stability.optims import FrozenOptim
from typing import Iterable


def fetch_preconditioner(optim:Optimizer, model:Module, params_old:Iterable[Parameter]=None):
    if isinstance(optim, VanillaMuon):
        return MuonPreconditioner(optim, model, params_old)
    elif isinstance(optim, RMSprop):
        return RMSpropPreconditioner(optim, model)
    elif isinstance(optim, Adam):
        return AdamPreconditioner(optim, model)
    elif isinstance(optim, FrozenOptim):
        return optim.preconditioner
    elif isinstance(optim, SGD):
        return None
    else:
        raise Exception(f"No preconditioner exists yet for Optimizer of type: {type(optim)}")