"""This module contains utility functions and classes.
"""
import functools
import inspect
import json
import warnings
from collections.abc import Mapping, Sequence, Set
from copy import deepcopy
from numbers import Number
from os import PathLike
from pathlib import Path
from typing import Any


def _get_init_args(cls) -> tuple[list[str], list[Any]]:
    """Get the arguments of the __init__ function of a class.

    Args:
        cls: The class to get the arguments from.

    Returns:
        tuple[list[str], list[Any]]: The arguments and their default values.
    """
    params = inspect.signature(cls.__init__).parameters
    args = list(params.keys())[1:]
    defaults = [params[arg].default for arg in args]
    return args, defaults


class Configurable:
    """Mixin class for configurable classes.
    Any class that inherits from this class can be initialized from a config dict (``from_config``),
    and can be converted to a config dict(``get_config``).
    Supports nested Configurable objects, and any object that has an ``as_dict`` method.
    """

    supported_val_types = (str, bool, Number, type(None))

    def get_config(self, param_name_map: dict[str, str] | None = None) -> dict[str, Any]:
        """Get the config of the object.
        The object must contain the attributes that are used in the __init__ function.
        If the name of attribute does not match to the name of the parameter in the __init__ function,
        use ``param_name_map`` to map the attribute name to the parameter name.
        Any attributes that is instance of ``Configurable`` will be converted to config dict recursively.

        Args:
            param_name_map (dict[str, str], optional): A mapping from the parameter names to the config names.

        Returns:
            dict[str, Any]: The config dict.
        """
        param_name_map = param_name_map or {}
        args, _ = _get_init_args(self.__class__)

        def recursive_as_dict(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, str):
                return obj
            if isinstance(obj, (Sequence, Set)):
                return type(obj)([recursive_as_dict(it) for it in obj])
            if isinstance(obj, Mapping):
                return type(obj)({kk: recursive_as_dict(vv) for kk, vv in obj.items()})
            if hasattr(obj, "as_dict"):
                return obj.as_dict()
            if isinstance(obj, Configurable):
                return obj.get_config()
            return obj

        config = {}
        for arg in args:
            argname = param_name_map.get(arg, arg)
            try:
                val = getattr(self, argname)
            except AttributeError as e:
                raise ValueError(
                    f"Argument {arg}(mapped name: {param_name_map.get(arg, arg)}) "
                    "should be present as attribute of the object."
                ) from e
            config[arg] = recursive_as_dict(val)

        return config

    @classmethod
    def from_config(cls, config: dict[str, Any], actual_cls: type | None = None) -> Any:
        """Initialize the object from a config dict.
        In case that the object which calls ``from_config`` is different from the object to be initialized,
        use ``actual_cls`` to specify the class to be initialized.
        This is useful when many classes share one base class, so that the base class can be used to initialize.
        ex) ``BaseDataset.from_config(config, actual_cls=MyDataset)``

        Args:
            config (dict[str, Any]): The config dict.
            actual_cls (type, optional): The class to be initialized. Defaults to None.

        Returns:
            Any: The initialized object.
        """

        if actual_cls is None:
            actual_cls = cls
        config = deepcopy(config)
        args, defaults = _get_init_args(actual_cls)
        # Construct default config
        sanitized_config = {k: v for k, v in zip(args, defaults, strict=True) if v is not inspect.Parameter.empty}
        # Update the default config with the config file
        for k, v in config.items():
            if k in args:
                sanitized_config[k] = v
            else:
                warnings.warn(f"Argument {k} is not used in {actual_cls.__name__}.__init__(). Ignored.", stacklevel=1)

        init_signature = inspect.signature(actual_cls.__init__)
        types = {k: p.annotation for k, p in init_signature.parameters.items() if p.annotation != inspect._empty}
        # Convert the config to the correct type
        for k, v in types.items():
            if hasattr(v, "from_config"):
                c = config[k]
                sanitized_config[k] = v.from_config(c)
        return actual_cls(**sanitized_config)


def warn_unstable(cls_or_fn):
    """Decorator that warns that a class or function is unstable.

    Args:
        cls_or_fn: Class or function to warn about.
    """
    if inspect.isclass(cls_or_fn):
        name = cls_or_fn.__name__
        orig_init = cls_or_fn.__init__

        def __init__(self, *args, **kws):
            warnings.warn(f"{name} is unstable and may change in future versions.", UserWarning, stacklevel=1)
            orig_init(self, *args, **kws)  # Call the original __init__

        cls_or_fn.__init__ = __init__
        return cls_or_fn

    else:
        name = cls_or_fn.__qualname__

        @functools.wraps(cls_or_fn)
        def wrapper(*args, **kwargs):
            warnings.warn(f"{name} is unstable and may change in future versions.", UserWarning, stacklevel=1)

            return cls_or_fn(*args, **kwargs)

    return wrapper


def log_and_print(contents: str, filepath: PathLike = None, end="\n"):
    """Log and print the contents.

    Args:
        contents (str): The contents to log and print.
        filepath (PathLike): The path to the log file.

    Returns:
        None
    """
    if filepath is not None:
        with open(filepath, "a") as f:
            f.write(contents + end)
    print(contents, end=end)


def remove_unused_kwargs(func: callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove unused kwargs from a function call.

    Args:
        func (callable): The function to call.
        kwargs (dict[str, Any]): The kwargs to pass to the function.

    Returns:
        dict[str, Any]: The kwargs that are used in the function.
    """
    valid_args = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in valid_args}


def load_config(filepath: PathLike) -> dict:
    """A convinience function to load a config file using the correct loader.
    supports json, yaml, and toml.

    Args:
        filepath (PathLike): The path to the config file.

    Returns:
        dict: The loaded config.
    """
    filepath = Path(filepath)
    if filepath.suffix == ".json":
        with open(filepath, "r") as f:
            config = json.load(f)
    elif filepath.suffix == ".yaml":
        try:
            import yaml
        except ImportError as e:
            raise ImportError("Please install pyyaml to load yaml config files.") from e
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
    elif filepath.suffix == ".toml":
        try:
            import tomli
        except ImportError as e:
            raise ImportError("Please install tomli to load toml config files.") from e
        with open(filepath, "rb") as f:
            config = tomli.load(f)
    else:
        raise ValueError(f"Invalid config file extension: {filepath.suffix}")
    return config
