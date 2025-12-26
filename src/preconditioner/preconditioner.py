from typing import Iterable
from torch.nn import Parameter


class Preconditioner:
    def pow(self, p:float, inplace:bool=False) -> "Preconditioner":
        pass

    def prepare(self):
        pass

    def dot(self, v:Iterable[Parameter], inplace:bool=False) -> Iterable[Parameter]:
        pass