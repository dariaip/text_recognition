import torch
from torch import nn


class GlobalMaxPool2d(nn.Module):
    def __init__(self):
        """
        Класс, имплементирующий max pooling на 2-й с конца размерности.
        Например:
        GlobalMaxPool2d()(torch.rand(3, 3, 3, 3)).shape
        вернет
        torch.Size([3, 3, 1, 3]).
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=-2, keepdim=True)[0]
