import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ml_lib.misc import auto_repr

@auto_repr("size")
class ParametricMarkovMatrix(nn.Module):

    size:int
    m: nn.Parameter
    heat: float = 1.

    def __init__(self, size:int, *, heat:float = 1.):
        super().__init__()
        self.size = size
        self.m = nn.Parameter(torch.randn(size, size))
        self.heat = heat

    def forward(self) -> Tensor:
        return F.softmax(self.m / self.heat, dim=-1)

@auto_repr("size1", "size2")
class ParametricMatching(nn.Module):
    
    size1: int
    size2: int
    m: nn.Parameter
    heat: float = 1.

    def __init__(self, size1:int, size2: int, *, heat:float = 1.):
        super().__init__()
        self.size1 = size1
        self.size2 = size2
        self.m = nn.Parameter(torch.randn(size1, size2))
        self.heat = heat

    def forward(self) -> Tensor:
        return F.log_softmax(self.m / self.heat, dim=-1)

    def get(self):
        return F.softmax(self.m / self.heat, dim=-1)


class ParametricMarkovMatrixWithMatchings(nn.Module):
    markov: ParametricMarkovMatrix
    matchings: nn.ModuleList #[ParametricMatching] but no type arguments here

    def __init__(self, size:int, *other_sizes: int, heat:float = 1., matching_heat: float|list[float] = 1.):
        super().__init__()
        self.markov = ParametricMarkovMatrix(size, heat=heat)
        self.matchings = nn.ModuleList()
        
        match matching_heat:
            case [*heats]:
                matching_heats = heats
            case float():
                matching_heats = [matching_heat] * len(other_sizes)

        for other_size, h in zip(other_sizes, matching_heats):
            self.matchings.append(ParametricMatching(size, other_size, heat=h))

    def forward(self) -> tuple[Tensor, ...]:
        return self.markov(), *[m() for m in self.matchings]

    def get(self) -> tuple[Tensor, ...]:
        return self.markov(), *[m.get() for m in self.matchings]


