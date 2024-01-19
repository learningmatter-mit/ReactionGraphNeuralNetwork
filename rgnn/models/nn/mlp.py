from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from rgnn.common.registry import registry
from rgnn.common.typing import Tensor


DEAFAULT_DROPOUT_RATE = 0.15


class MLP(torch.nn.Module):
    """Multi-layer perceptron.

    Args:
        n_input (int): Number of input features.
        n_output (int): Number of output features.
        hidden_layers (Sequence[int]): Number of hidden units in each layer. Defaults to (64, 64).
        activation (str): Activation function to use. Defaults to "silu".
        activation_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the activation function.
            Defaults to None.
        activate_final (bool): Whether to apply the activation function to the final layer. Defaults to False.
        w_init (str): Weight initializer. Defaults to "xavier_uniform".
        b_init (str): Bias initializer. Defaults to "zeros".
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        hidden_layers: Sequence[int] = (64, 64),
        activation: str = "silu",
        activation_kwargs: Optional[Dict[str, Any]] = None,
        activate_final: bool = False,
        w_init: str = "xavier_uniform",
        b_init: str = "zeros",
        dropout_rate: float = DEAFAULT_DROPOUT_RATE,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        activation_kwargs = activation_kwargs or {}
        self.activation = registry.get_activation_class(activation)
        self.activate_final = activate_final
        self.w_init = registry.get_initializer_function(w_init)
        self.b_init = registry.get_initializer_function(b_init)

        # Create layers
        layers = []
        layers.append(torch.nn.Linear(n_input, hidden_layers[0]))
        if use_batch_norm:
            layers.append(torch.nn.BatchNorm1d(hidden_layers[0]))  #
        layers.append(self.activation(**activation_kwargs))
        layers.append(torch.nn.Dropout(dropout_rate))

        for i in range(len(hidden_layers) - 1):
            layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(hidden_layers[i + 1]))  # Add batch normalization layer
            layers.append(self.activation(**activation_kwargs))
            layers.append(torch.nn.Dropout(dropout_rate))

        layers.append(torch.nn.Linear(hidden_layers[-1], n_output))
        if self.activate_final:
            layers.append(self.activation(**activation_kwargs))
            layers.append(torch.nn.Dropout(dropout_rate))

        self.layers = torch.nn.ModuleList(layers)

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                self.w_init(layer.weight.data)
                self.b_init(layer.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class GatedEquivariantMLP(torch.nn.Module):
    """Originally from schnetpack.

    Gated equivariant block as used for the prediction of tensorial properties by PaiNN.
    Transforms scalar and vector representation using gated nonlinearities.

    Args:
        n_scalar_input (int): Number of input scalar features.
        n_vector_input (int): Number of input vector features.
        n_scalar_output (int): Number of output scalar features.
        n_vector_output (int): Number of output vector features.
        hidden_layers (Sequence[int]): Number of hidden units in each layer. Defaults to (64, 64).
        activation (str): Activation function to use. Defaults to "silu".
        scalar_activation (Optional[str]): Activation function to use for scalar outputs. Defaults to None.
        w_init (str): Weight initializer. Defaults to "xavier_uniform".
        b_init (str): Bias initializer. Defaults to "zeros".

    References:
    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021 (to appear)
    """

    def __init__(
        self,
        n_scalar_input: int,
        n_vector_input: int,
        n_scalar_output: int,
        n_vector_output: int,
        hidden_layers: Sequence[int] = (64, 64),
        activation="silu",
        scalar_activation=None,
        w_init: str = "xavier_uniform",
        b_init: str = "zeros",
    ):
        """
        Args:
            n_sin: number of input scalar features
            n_vin: number of input vector features
            n_sout: number of output scalar features
            n_vout: number of output vector features
            n_hidden: number of hidden units
            activation: interal activation function
            sactivation: activation function for scalar outputs
        """
        super().__init__()
        self.n_scalar_input = n_scalar_input
        self.n_vector_input = n_vector_input
        self.n_scalar_output = n_scalar_output
        self.n_vector_output = n_vector_output
        self.hidden_layers = hidden_layers
        self.w_init = registry.get_initializer_function(w_init)
        self.b_init = registry.get_initializer_function(b_init)

        self.mix_vectors = torch.nn.Linear(n_vector_input, 2 * n_vector_output, bias=False)
        self.scalar_net = MLP(
            n_scalar_input + n_vector_output,
            n_scalar_output + n_vector_output,
            hidden_layers,
            activation,
            w_init=w_init,
            b_init=b_init,
        )

        self.scalar_activation = (
            None if scalar_activation is None else registry.get_activation_class(scalar_activation)()
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.w_init(self.mix_vectors.weight.data)

    def forward(self, scalars: Tensor, vectors: Tensor) -> Tuple[Tensor, Tensor]:
        vmix = self.mix_vectors(vectors)
        vectors_V, vectors_W = torch.split(vmix, self.n_vector_output, dim=-1)
        vectors_Vn = torch.norm(vectors_V, dim=-2)

        ctx = torch.cat([scalars, vectors_Vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, x = torch.split(x, [self.n_scalar_output, self.n_vector_output], dim=-1)
        v_out = x.unsqueeze(-2) * vectors_W

        if self.scalar_activation:
            s_out = self.scalar_activation(s_out)

        return s_out, v_out
