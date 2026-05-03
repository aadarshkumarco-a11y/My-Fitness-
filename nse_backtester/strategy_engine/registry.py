"""Strategy plug-in registry. Use ``@register_strategy("name")`` to add one."""
from __future__ import annotations

from typing import Callable, Dict, List, Type

from .base import Strategy

_REGISTRY: Dict[str, Type[Strategy]] = {}


def register_strategy(name: str) -> Callable[[Type[Strategy]], Type[Strategy]]:
    name = name.strip().lower()

    def decorator(cls: Type[Strategy]) -> Type[Strategy]:
        if not issubclass(cls, Strategy):
            raise TypeError(f"{cls!r} must subclass Strategy")
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def get_strategy(name: str) -> Type[Strategy]:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Strategy '{name}' not registered. Available: {list_strategies()}")
    return _REGISTRY[key]


def list_strategies() -> List[str]:
    return sorted(_REGISTRY.keys())
