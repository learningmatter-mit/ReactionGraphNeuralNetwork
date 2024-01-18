# Originally from https://github.com/facebookresearch/mmf
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Registry is central source of truth in MMF. Inspired from Redux's
concept of global store, Registry maintains mappings of various information
to unique keys. Special functions in registry can be used as decorators to
register different kind of classes.

Import the global registry object using

``from mmf.common.registry import registry``

Various decorators for registry different kind of classes with unique keys

- Register a callback function: ``@registry.register_callback``
- Register a loss: ``@registry.register_loss``
- Register a model: ``@registry.register_model``
- Register a optimizer: ``@registry.register_optimizer``
- Register a scheduler: ``@registry.register_scheduler``
"""

import sys
from copy import deepcopy
from typing import Any, Iterable, Optional

import inflection


def camelcase_to_snakecase(name: str) -> str:
    """Convert camel case to snake case

    Args:
        name (str): Name to convert

    Returns:
        str: Converted name
    """
    # Do not add underscore if the last character is single upper cases
    # e.g. "AdamW" -> "adamw"
    if name[-1].isupper() and not name[-2].isupper():
        name = name[:-1] + name[-1].lower()

    return inflection.underscore(name)


def _normalize_str(s: str) -> str:
    return s.lower().replace("_", "").replace("-", "")


def _find_match(query: str, choices: Iterable[str]) -> Optional[str]:
    """Find the best match for query in choices.
    The stringss are normalized by removing all underscores and hyphens.

    Args:
        query (str): Query string
        choices (Iterable[str]): Iterable of choices

    Returns:
        Optional[str]: Best match if found else None
    """
    for choice in choices:
        if _normalize_str(query) == _normalize_str(choice):
            if query != choice:
                print(f"Using {choice} instead of {query} as key", file=sys.stderr)
            return choice
    return None


def _find_all_torch_optimizers():
    import torch
    from torch.optim.optimizer import Optimizer

    def cond(v):
        if not isinstance(v, type):
            return False
        if not issubclass(v, Optimizer):
            return False
        if v.__name__ == "Optimizer":
            return False
        return True

    all_vars = vars(torch.optim)
    optimizers = {camelcase_to_snakecase(k): v for k, v in all_vars.items() if cond(v)}
    # Handle some wierd cases
    nadam = optimizers.pop("n_adam")
    radam = optimizers.pop("r_adam")
    rmsprop = optimizers.pop("rm_sprop")
    optimizers["nadam"] = nadam
    optimizers["radam"] = radam
    optimizers["rms_prop"] = rmsprop
    for name in optimizers:
        optimizers[name].name = name
        optimizers[name].category = "optimizer"
    return optimizers


def _find_all_torch_activations():
    import torch
    from torch.nn import Module

    all_activation_names = torch.nn.modules.activation.__all__
    all_activations = [getattr(torch.nn, name) for name in all_activation_names]

    activations = {
        camelcase_to_snakecase(k).replace("_lu", "lu"): v
        for k, v in zip(all_activation_names, all_activations, strict=False)
        if issubclass(v, Module)
    }
    return activations


def _find_all_torch_lr_schedulers():
    import torch
    from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

    def cond(v):
        if not isinstance(v, type):
            return False
        if not issubclass(v, _LRScheduler):
            return False
        if v.__name__ in ("_LRScheduler", "chainedscheduler"):
            return False
        return True

    all_vars = vars(torch.optim.lr_scheduler)
    schedulers = {camelcase_to_snakecase(k): v for k, v in all_vars.items() if cond(v)}
    schedulers["reduce_lr_on_plateau"] = ReduceLROnPlateau
    return schedulers


def _find_all_torch_initilizers():
    import torch

    all_initializers = {k[:-1]: v for k, v in vars(torch.nn.init).items() if k.endswith("_")}
    return all_initializers


class Registry:
    r"""Class for registry object which acts as central source of truth
    for AML.
    """
    mapping = {
        # Mappings of builder name to their respective classes
        # Use `registry.register_builder` to register a builder class
        # with a specific name
        "model_name_mapping": {},
        "energy_model_name_mapping": {},
        "reaction_model_name_mapping": {},
        "loss_name_mapping": {},
        "optimizer_name_mapping": _find_all_torch_optimizers(),
        "scheduler_name_mapping": _find_all_torch_lr_schedulers(),
        "activation_name_mapping": _find_all_torch_activations(),
        "initializer_name_mapping": _find_all_torch_initilizers(),
        "dataset_name_mapping": {},
        "state": {},
    }

    @classmethod
    def list_models(cls):
        return list(cls.mapping["model_name_mapping"].keys())

    @classmethod
    def list_energy_models(cls):
        return list(cls.mapping["energy_model_name_mapping"].keys())

    @classmethod
    def list_reaction_models(cls):
        return list(cls.mapping["reaction_model_name_mapping"].keys())

    @classmethod
    def list_losses(cls):
        return list(cls.mapping["loss_name_mapping"].keys())

    @classmethod
    def list_optimizers(cls):
        return list(cls.mapping["optimizer_name_mapping"].keys())

    @classmethod
    def list_schedulers(cls):
        return list(cls.mapping["scheduler_name_mapping"].keys())

    @classmethod
    def list_callbacks(cls):
        return list(cls.mapping["callback_name_mapping"].keys())

    @classmethod
    def list_loggers(cls):
        return list(cls.mapping["logger_name_mapping"].keys())

    @classmethod
    def list_activations(cls):
        return list(cls.mapping["activation_name_mapping"].keys())

    @classmethod
    def list_initializers(cls):
        return list(cls.mapping["initializer_name_mapping"].keys())

    @classmethod
    def list_datasets(cls):
        return list(cls.mapping["dataset_name_mapping"].keys())

    @classmethod
    def register_callback(cls, name):
        def wrap(func):
            func.name = name
            func.category = "callback"
            cls.mapping["callback_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_logger(cls, name):
        def wrap(func):
            func.name = name
            func.category = "logger"
            cls.mapping["logger_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_loss(cls, name):
        def wrap(func):
            func.name = name
            func.category = "loss"
            cls.mapping["loss_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_model(cls, name):
        def wrap(func):
            func.name = name
            func.category = "model"
            cls.mapping["model_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_reaction_model(cls, name):
        def wrap(func):
            func.name = name
            func.category = "reaction_model"
            cls.mapping["reaction_model_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_energy_model(cls, name):
        def wrap(func):
            func.name = name
            func.category = "energy_model"
            cls.mapping["energy_model_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_optimizer(cls, name):
        def wrap(func):
            func.name = name
            func.category = "optimizer"
            cls.mapping["optimizer_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_scheduler(cls, name):
        def wrap(func):
            func.name = name
            func.category = "scheduler"
            cls.mapping["scheduler_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_activation(cls, name):
        def wrap(func):
            func.name = name
            cls.mapping["activation_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_initializer(cls, name):
        def wrap(func):
            cls.mapping["initializer_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_dataset(cls, name):
        def wrap(func):
            func.name = name
            cls.mapping["dataset_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register(cls, name, obj):
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def _get(cls, name, map_key):
        name = _find_match(name, cls.mapping[map_key].keys())
        if name is None:
            raise ValueError("No such object: {}. Available items: {}".format(name, list(cls.mapping[map_key].keys())))
        return cls.mapping[map_key].get(name, None)

    # @classmethod
    # def get_callback_class(cls, name):
    #     key = "callback_name_mapping"
    #     return cls._get(name, key)

    @classmethod
    def get_model_class(cls, name):
        key = "model_name_mapping"
        return cls._get(name, key)

    @classmethod
    def get_energy_model_class(cls, name):
        key = "energy_model_name_mapping"
        return cls._get(name, key)

    @classmethod
    def get_reaction_model_class(cls, name):
        key = "reaction_model_name_mapping"
        return cls._get(name, key)

    @classmethod
    def get_loss_class(cls, name):
        key = "loss_name_mapping"
        return cls._get(name, key)

    @classmethod
    def get_optimizer_class(cls, name):
        key = "optimizer_name_mapping"
        return cls._get(name, key)

    @classmethod
    def get_scheduler_class(cls, name):
        key = "scheduler_name_mapping"
        return cls._get(name, key)

    # @classmethod
    # def get_logger_class(cls, name):
    #     key = "logger_name_mapping"
    #     return cls._get(name, key)

    @classmethod
    def get_activation_class(cls, name):
        key = "activation_name_mapping"
        return cls._get(name, key)

    @classmethod
    def get_initializer_function(cls, name):
        key = "initializer_name_mapping"
        return cls._get(name, key)

    @classmethod
    def get_dataset_class(cls, name):
        key = "dataset_name_mapping"
        return cls._get(name, key)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retrieved.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for MMF's
                               internal operations. Default: False
        Usage::

            from mmf.common.registry import registry

            config = registry.get("config")
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if "writer" in cls.mapping["state"] and value == default and no_warning is False:
            cls.mapping["state"]["writer"].warning(
                "Key {} is not present in registry, returning default value " "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from mmf.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)

    def construct_from_config(self, config: dict[str, Any], name: str, category: str, **kwargs):
        config = deepcopy(config)
        mapping_key = f"{category}_name_mapping"
        if mapping_key not in self.mapping:
            raise ValueError(f"No such category {category} exists")

        cls_obj = self.mapping[mapping_key].get(name, None)
        if cls_obj is None:
            raise ValueError(f"{name} not found in {category} registry")

        return self.mapping[mapping_key][name](**config, **kwargs)


registry = Registry()

# Only setup imports in main process, this means registry won't be
# fully available in spawned child processes (such as dataloader processes)
# but instantiated. This is to prevent issues such as
# https://github.com/facebookresearch/mmf/issues/355
# if __name__ == "__main__":
#     setup_imports()
