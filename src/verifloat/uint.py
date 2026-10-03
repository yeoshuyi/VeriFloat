"""Fixed-width unsigned integers (native type from the C++ core).

``UINT(val, bits=32, saturate=False)`` wraps modulo 2**bits, or clamps to
[min, max] with ``saturate=True``. Bit operations always wrap.
"""

from __future__ import annotations

from ._core import UINT
from ._warnings import IntCastWarning  # noqa: F401  (as in 0.1)

__all__ = ["UINT"]
