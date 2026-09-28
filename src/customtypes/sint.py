from __future__ import annotations

from .uint import UINT


class INT(UINT):
    __slots__ = ()
    signed = True

    # Getters
    @property
    def val(self) -> int:
        if self._val >> (self._bits - 1):
            return self._val - (1 << self._bits)
        return self._val

    @property
    def min(self) -> int:
        return -(1 << (self._bits - 1))

    @property
    def max(self) -> int:
        return (1 << (self._bits - 1)) - 1
