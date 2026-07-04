"""Small name-to-object registry used by operation decorators."""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            self.add(name, item)
            return item

        return decorator

    def add(self, name: str, item: T) -> None:
        if name in self._items:
            raise ValueError(f"{self._kind} '{name}' already registered")
        self._items[name] = item

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(f"Unknown {self._kind}: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))
