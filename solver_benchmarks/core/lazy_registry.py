"""A name -> class registry that imports each class on first lookup."""

from __future__ import annotations

import importlib
from collections.abc import Iterator, MutableMapping
from typing import Generic, TypeVar

T = TypeVar("T")


class LazyRegistry(MutableMapping[str, T], Generic[T]):
    """Maps names to classes, given as ``(module, attribute)`` until first looked up.

    Names, membership and iteration import nothing; looking a name up
    imports its module once. Assigning a class (as tests do to register
    fakes) and deleting a name work as on a dict.
    """

    def __init__(self, entries: dict[str, tuple[str, str]]) -> None:
        self._entries: dict[str, T | tuple[str, str]] = dict(entries)

    def __getitem__(self, name: str) -> T:
        entry = self._entries[name]
        if isinstance(entry, tuple):
            module, attribute = entry
            entry = getattr(importlib.import_module(module), attribute)
            self._entries[name] = entry
        return entry

    def __setitem__(self, name: str, value: T) -> None:
        self._entries[name] = value

    def __delitem__(self, name: str) -> None:
        del self._entries[name]

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({sorted(self._entries)})"
