import warnings

import torch

from rgnn.common.typing import Tensor


def gaussian_rbf(inputs: Tensor, offsets: Tensor, widths: Tensor) -> Tensor:
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(torch.nn.Module):
    r"""Gaussian radial basis functions.
    .. math::
        \phi_{\mu, \sigma}(x) = \exp\left(-\frac{1}{2\sigma^2}(x-\mu)^2\right)

    Args:
        n_rbf: total number of Gaussian functions, :math:`N_g`.
        cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
        start: center of first Gaussian function, :math:`\mu_0`.
        trainable: If True, widths and offset of Gaussian functions
            are adjusted during training process.
    """

    def __init__(self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False):
        r""" """
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(torch.abs(offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.widths = torch.nn.Parameter(widths)
            self.offsets = torch.nn.Parameter(offset)
        else:
            self.register_buffer("offsets", offset)
            self.register_buffer("widths", widths)

    def forward(self, x: Tensor) -> Tensor:
        return gaussian_rbf(x, self.offsets, self.widths)


class BesselRBF(torch.nn.Module):
    """
    Sine for radial basis functions with coulomb decay (0th order bessel).
    Args:
        cutoff: radial cutoff
        n_rbf: number of basis functions.

    References:
    .. [#dimenet] Klicpera, Groß, Günnemann:
       Directional message passing for molecular graphs.
       ICLR 2020
    """

    def __init__(self, n_rbf: int, cutoff: float, trainable: bool = False):
        """ """
        super(BesselRBF, self).__init__()
        self.n_rbf = n_rbf
        if n_rbf > 20:
            warnings.warn("n_rbf > 20 for bessel rbf may cause numerical instability.", stacklevel=1)

        freqs = torch.arange(1, n_rbf + 1) * torch.pi / cutoff
        if trainable:
            self.freqs = torch.nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

    def forward(self, inputs):
        ax = inputs[..., None] * self.freqs
        sinax = torch.sin(ax)
        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm[..., None]
        return y
