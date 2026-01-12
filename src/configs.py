from dataclasses import dataclass
from typing import Union, Literal


@dataclass
class MuonConfig:   
    lr:float = 2e-3
    momentum:float = 0.99
    nesterov:bool = False
    weight_decay:float = 0


@dataclass
class RMSpropConfig:
    lr:float = 2e-5
    eps:float = 1e-8
    alpha:float = 0.99


@dataclass
class AdamConfig:
    lr:float = 2e-5
    betas:tuple[float, float] = (0.0, 0.99)


@dataclass
class MLPConfig:
    width:int = 64
    n_hidden:int = 2
    activation:str = "gelu"
    bias:bool = True


@dataclass
class CIFAR10Config:
    n_classes:int = 4
    n_samples_per_class:int = 250
    loss:Union[Literal["mse", "ce"]] = "mse"


@dataclass 
class TrackerConfig:
    freq:int = 1
    n_warmup:int = 5
