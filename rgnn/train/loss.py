import torch
from torch.nn import functional as F

from rgnn.models.registry import registry
from rgnn.common import keys as K


@registry.register_loss("mse_loss")
class MSELoss(torch.nn.Module):
    def __init__(self, key: str, per_atom: bool = False):
        super().__init__()
        self.key = key
        self.per_atom = per_atom

    def forward(self, data, outputs):
        target = data[self.key]
        pred = outputs[self.key]
        # print(f"{self.key}, {target}, {pred}")
        if self.per_atom:
            target = target / data[K.n_atoms]
            pred = pred / data[K.n_atoms]
        return F.mse_loss(pred, target)

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.key}, per_atom={self.per_atom})"

    def __str__(self):
        return self.__repr__()


@registry.register_loss("huber_loss")
class HuberLoss(torch.nn.Module):
    def __init__(self, key: str, per_atom: bool = False, delta=0.01):
        super().__init__()
        self.key = key
        self.per_atom = per_atom

    def forward(self, data, outputs):
        target = data[self.key]
        pred = outputs[self.key]
        if self.per_atom:
            target = target / data[K.n_atoms]
            pred = pred / data[K.n_atoms]
        return F.huber_loss(pred, target)

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.key}, per_atom={self.per_atom})"

    def __str__(self):
        return self.__repr__()


@registry.register_loss("weighted_sum_loss")
class WeightedSumLoss(torch.nn.Module):
    def __init__(
        self,
        keys: tuple[str, ...],
        weights: tuple[float, ...],
        loss_fns: tuple[str, ...] | str | None = None,
        per_atom_keys: tuple[str, ...] = None,
        **kwargs,
    ):
        super().__init__()
        self.keys = keys
        self.weights = weights
        self.loss_fns = loss_fns
        self.per_atom_keys = per_atom_keys or ()

        assert len(self.keys) == len(self.weights), "keys and weights must have the same length"

        if self.loss_fns is None:
            self.loss_fns = ["mse_loss"] * len(self.keys)

        if isinstance(self.loss_fns, str):
            loss_classes = [registry.get_loss_class(self.loss_fns)] * len(self.keys)
        else:
            loss_classes = [registry.get_loss_class(fn) for fn in self.loss_fns]
        self.loss_fns = []
        for key in self.keys:
            per_atom = key in self.per_atom_keys
            loss_fn = loss_classes.pop(0)(key=key, per_atom=per_atom, **kwargs)
            self.loss_fns.append(loss_fn)

    def forward(self, data, outputs):
        loss = 0.0
        for weight, loss_fn in zip(self.weights, self.loss_fns, strict=True):
            loss += weight * loss_fn(data, outputs)
        return loss

    def __repr__(self):
        s = f"{self.__class__.__name__}(\n"
        s += "    keys=(" + ", ".join(self.keys) + "),\n"
        s += "    weights=(" + ", ".join([f"{w:.3f}" for w in self.weights]) + "),\n"
        s += "    loss_fns=(" + ", ".join([fn.__class__.__name__ for fn in self.loss_fns]) + "),\n"
        s += "    per_atom_keys=(" + ", ".join(self.per_atom_keys) + "),\n"
        s += ")"
        return s

    def __str__(self):
        return self.__repr__()
