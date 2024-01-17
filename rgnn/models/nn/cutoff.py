import torch

from rgnn.common.typing import Tensor


class CosineCutoff(torch.nn.Module):
    r"""Cosine cutoff function.
    $$
    f_c(x) = \frac{1}{2} \left[1 + \cos\left(\pi \frac{x}{r_c}\right) \right]
    $$
    Args:
        cutoff (float): Cutoff radius. Defaults to 10.0.
    """

    def __init__(self, cutoff: float = 10.0):
        super().__init__()
        self.register_buffer("cutoff", torch.as_tensor(cutoff, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        out = 0.5 * (1 + torch.cos(torch.pi * x / self.cutoff))
        mask = x <= self.cutoff
        return out * mask


class PolynomialCutoff(torch.nn.Module):
    r"""Polynomial cutoff function.
    $$
    f_c(x)
    $$
    Args:
        cutoff (float): Cutoff radius. Defaults to 10.0.
    """

    def __init__(self, cutoff: float = 10.0):
        super().__init__()
        self.register_buffer("cutoff", torch.as_tensor(cutoff, dtype=torch.float32))

    def forward(self, x: Tensor):
        """Smooth cutoff function."""
        mask = x < self.cutoff
        x = x * mask
        d = self.cutoff * 0.25
        x_r_4 = torch.pow(x - self.cutoff, 4)
        out = x_r_4 / (d**4 + x_r_4)
        return out * mask
