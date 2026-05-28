from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(f"{self._kind} '{name}' already registered")
            self._items[name] = cls
            return cls

        return decorator

    def add(self, name: str, cls: type[T]) -> None:
        if name in self._items:
            raise ValueError(f"{self._kind} '{name}' already registered")
        self._items[name] = cls

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(f"Unknown {self._kind}: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))
