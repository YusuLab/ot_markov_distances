from typing import overload, Optional, Literal

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np

from ot_markov_distances.misc import auto_repr, all_equal
from ot_markov_distances.utils import draw_markov

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

    def get(self) -> Tensor:
        return F.softmax(self.m / self.heat, dim=-1)


@auto_repr("size", "time_factor")
class WarpedTimeParametricMarkovMatrix(ParametricMarkovMatrix):

    size:int
    m: nn.Parameter
    heat: float = 1.
    time_factor: int= 1

    def __init__(self, size:int, *, heat:float = 1., time_factor: int= 1):
        super().__init__(size, heat=heat)
        self.time_factor = time_factor

    def forward(self) -> Tensor:
        M = super().forward()
        warped_M = torch.matrix_power(M, self.time_factor)
        return warped_M

    def get(self) -> Tensor:
        return super().forward()

@auto_repr("size1", "size2")
class ParametricMatching(nn.Module):
    
    size1: int
    size2: int
    m: nn.Parameter
    heat: float = 1.
    type: Literal["rows", "full"] = "rows"

    def __init__(self, size1:int, size2: int, *, heat:float = 1., type: Literal["rows", "full"] = "rows"):
        super().__init__()
        self.size1 = size1
        self.size2 = size2
        self.m = nn.Parameter(torch.randn(size1, size2))
        self.heat = heat
        self.type = type

    def forward(self) -> Tensor:
        match self.type:
            case "rows":
                return - F.log_softmax(self.m / self.heat, dim=-1)
            case "full":
                m = self.m.view(-1)
                m = - F.log_softmax(m / self.heat, dim=-1)
                return m.view(self.size1, self.size2)

    def get(self):
        match self.type:
            case "rows":
                return F.softmax(self.m / self.heat, dim=-1)
            case "full":
                m = self.m.view(-1)
                m = F.softmax(m / self.heat, dim=-1)
                return m.view(self.size1, self.size2)


class ParametricMarkovMatrixWithMatchings(nn.Module):
    markov: ParametricMarkovMatrix
    matchings: nn.ModuleList #[ParametricMatching] but no type arguments here

    def __init__(self, size:int, *other_sizes: int, heat:float = 1., matching_heat: float|list[float] = 1., matching_type: Literal["rows", "full"] = "rows"):
        super().__init__()
        self.markov = ParametricMarkovMatrix(size, heat=heat)
        self.matchings = nn.ModuleList()
        
        match matching_heat:
            case [*heats]:
                matching_heats = heats
            case float():
                matching_heats = [matching_heat] * len(other_sizes)

        for other_size, h in zip(other_sizes, matching_heats):
            self.matchings.append(ParametricMatching(size, other_size, heat=h, type=matching_type))

    def forward(self) -> tuple[Tensor, ...]:
        return self.markov(), *[m() for m in self.matchings]

    def get(self) -> tuple[Tensor, ...]:
        return self.markov.get(), *[m.get() for m in self.matchings]

    def draw(self, original_positions, ax=None):
        if ax is None:
            import matplotlib.pyplot  as plt #type:ignore
            ax = plt.gca()

        markov, matching1, *_ = self.get()

        positions = np.einsum("mi,id->md",matching1.numpy(force=True), original_positions)
        if self.matchings[0].type == "full":
            positions = positions / matching1.sum(-1, keepdim=True).numpy(force=True)
        pos = {i: positions[i] for i in range(len(positions))}

        draw_markov(markov, pos, ax=ax)

class ParametricMarkovMatrixWithLabels(nn.Module):

    markov: ParametricMarkovMatrix
    label: nn.Parameter

    others: Optional[list[Tensor]]

    @overload
    def __init__(self, size:int, label_size:int, /, *, heat=1., time_factor=None):
        ...

    @overload
    def __init__(self, size:int, *target_labels: Tensor, heat=1., time_factor=None):
        ...

    def __init__(self, size:int, *target_labels, heat=1., time_factor=None):
        """If initialized with target labels, the forward method will return cost matrices.
        Otherwise the label will be returned
        """
        super().__init__()
        if time_factor is not None:
            self.markov = WarpedTimeParametricMarkovMatrix(size, heat=heat, time_factor=time_factor)
        else:
            self.markov = ParametricMarkovMatrix(size, heat=heat)
        match target_labels:
            case [int(label_size)]:
                self.label = nn.Parameter(torch.randn(size, label_size))
                self.others = None
            case [*tensors]:
                tensors = [tensor.clone() for tensor in tensors]
                label_sizes = [t.shape[-1] for t in tensors]
                assert all_equal(label_sizes)
                label_size = label_sizes.pop()
                self.others = tensors
                for i, t in enumerate(tensors):
                    self.register_buffer(f"other{i}", t)
                self.label = nn.Parameter(torch.randn(size, label_size))

    def forward(self):
        if self.others is None:
            return self.markov(), self.label

        return self.markov(), *[
                (self.label[:, None, :] - other[None, :, :]).square().sum(-1) 
                for other in self.others]
    
    def get(self) -> tuple[Tensor, ...]:
        return self.markov.get(), self.label
    
    def update_others(self):
        if self.others is None:
            return
        self.others = [self.get_buffer(f"other{i}") for i in
                      range(len(self.others))]
    
    def to(self, *args, **kwargs):
        r = super().to(*args, **kwargs)
        r.update_others()
        return r
        
    
    def draw(self, positions=None, ax=None):
        #if there are no positions, 
        #assume the first two coordinates of the label are
        #positions
        if ax is None:
            import matplotlib.pyplot  as plt #type:ignore
            ax = plt.gca()
            
        markov, label = self.get()
        if positions is None:
            positions = label[:, :2].numpy(force=True)
        pos = {i: positions[i] for i in range(len(positions))}
        draw_markov(markov, pos, ax=ax)
