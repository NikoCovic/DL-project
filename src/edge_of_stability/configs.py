from dataclasses import dataclass
from typing import Union, Literal


@dataclass
class MuonConfig:   
    lr:float = 2e-3
    momentum:float = 0.9
    nesterov:bool = False
    weight_decay:float = 0


@dataclass
class RMSpropConfig:
    lr:float = 2e-5
    eps:float = 1e-8
    alpha:float = 0.99


@dataclass
class AdamConfig:
    lr:float = 2e-3
    betas:tuple[float, float] = (0.9, 0.999)


@dataclass
class MLPConfig:
    width:int = 200
    n_hidden:int = 5
    activation:str = "tanh"
    bias:bool = True


@dataclass
class CIFAR10Config:
    n_classes:int = 10
    n_samples_per_class:int = 200
    loss:Union[Literal["mse", "ce"]] = "mse"


@dataclass 
class TrackerConfig:
    freq:int = 1
    n_warmup:int = 5


@dataclass
class FrozenOptimConfig:
    n_epochs:int = 250


@dataclass
class SGDConfig:
    lr:float = 2e-3