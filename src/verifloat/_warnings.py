"""Warning hierarchy shared by all VeriFloat types."""

from __future__ import annotations

import os
import warnings


class VeriFloatWarning(RuntimeWarning):
    """Base class for every warning VeriFloat raises."""


class CastWarning(VeriFloatWarning):
    """A value was implicitly cast or promoted (lossy literals, format mixing)."""


class FPFormatWarning(CastWarning):
    """Operands of different FP formats were implicitly promoted."""


class FPModeWarning(FPFormatWarning):
    """Operands disagree on an FP mode tag: signed, wrap, ftz or rounding."""


class IntCastWarning(CastWarning):
    """UINT/INT operands of different width or signedness were promoted, or an
    int literal did not fit the operand's type."""


class BlockFormatWarning(CastWarning):
    """Block tensors of different block formats were combined."""


class FPOverflowWarning(VeriFloatWarning):
    """A result overflowed an unsigned or wrapping FP format."""


class FPUnderflowWarning(VeriFloatWarning):
    """A result underflowed an unsigned or wrapping FP format."""


_PKG_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def warn(category, msg: str) -> None:
    # Attribute the warning to the first caller outside this package.
    warnings.warn(msg, category, skip_file_prefixes=(_PKG_DIR,))
