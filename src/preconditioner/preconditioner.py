from src.utils import ParameterVector


class Preconditioner:
    def pow(self, p:float, inplace:bool=False) -> "Preconditioner":
        pass

    def prepare(self):
        pass

    def dot(self, v:ParameterVector, inplace:bool=False) -> ParameterVector:
        pass