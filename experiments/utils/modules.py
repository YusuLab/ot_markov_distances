import torch
from torch import nn, Tensor
from torch.nn import functional as F

class ParametricMarkovMatrix(nn.Module):

    m: nn.Parameter
    heat: float = 1.

    def __init__(self, size:int, *, heat:float = 1.):
        super().__init__()
        self.m = nn.Parameter(torch.randn(size, size))
        self.heat = heat

    def forward(self) -> Tensor:
        return F.softmax(self.m / self.heat, dim=-1)


class ParametricMarkovMatrixWithMatchings(nn.Module):
    m: ParametricMarkovMatrix
    matchings: nn.ParameterList
    matching_heat: float | list[float] = 1.

    def __init__(self, size:int, *other_sizes, heat:float = 1., matching_heat: float|list[float] = 1.):
        self.m = ParametricMarkovMatrix(size, heat=heat)
        self.matching_heat = matching_heat

        self.matchings = nn.ParameterList()


        for other_size in other_sizes:
            self.matchings.append(torch.randn((size, other_size)))

    def forward(self) -> tuple[Tensor, ...]:
        matching_heat: list[float]
        if isinstance(self.matching_heat, float):
            matching_heat = [self.matching_heat]*len(self.matchings)
        else:
            matching_heat = self.matching_heat
        assert len(matching_heat) == len(self.matchings)

        return self.m(), *[F.softmax(matching/heat, dim = -1 ) 
                for matching, heat in zip(self.matchings, matching_heat)]

