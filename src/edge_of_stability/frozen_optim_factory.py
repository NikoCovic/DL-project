from torch.optim import Optimizer, Muon, RMSprop, Adam
from torch.nn import Module
from src.edge_of_stability.preconditioner_factory import fetch_preconditioner
from src.edge_of_stability.optims import FrozenMuonOptim, FrozenAdamOptim
from src.sharpness.airbench94_muon import VanillaMuon


def fetch_frozen_optim(optim:Optimizer, model:Module, lr:float, **optim_kwargs):
    if isinstance(optim, VanillaMuon):
        preconditioner = fetch_preconditioner(optim, model)
        return FrozenMuonOptim(model.parameters(), preconditioner, lr, **optim_kwargs)
    elif isinstance(optim, Adam):
        preconditioner = fetch_preconditioner(optim, model)
        return FrozenAdamOptim(model.parameters(), preconditioner, lr, **optim_kwargs)
    else:
        raise Exception(f"There is no frozen optimizer alternative for {optim}.")
