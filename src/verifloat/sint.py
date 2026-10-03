"""Fixed-width two's-complement integers (native type from the C++ core)."""

from __future__ import annotations

from ._core import INT, UINT  # noqa: F401  (UINT as in 0.1)

__all__ = ["INT"]
