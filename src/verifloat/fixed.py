"""Fixed-point numbers: value = raw * 2**-frac_bits.

Arithmetic is exact: + and - grow the result by one integer bit and * adds the
widths, as VHDL ``sfixed``/``ufixed`` and APyTypes do. Rounding and overflow
happen only on an explicit ``cast`` (or when a format is called on a value),
with the rounding modes of ``Rounding`` and wrap or saturate on overflow.

FIXED values and their arithmetic live in the C++ core.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from fractions import Fraction

from . import _core
from ._warnings import CastWarning, IntCastWarning  # noqa: F401  (IntCastWarning, FP, UINT as in 0.1)
from .fp import FP, FPFlags, Rounding, _ROUNDINGS, _pow2  # noqa: F401
from .uint import UINT  # noqa: F401

_F = FPFlags
_MAX_BITS = 1 << 24          # widest format (bits), as mantissa_bits is capped for FP


@dataclass(frozen=True)
class FixedFormat(_core.FixedBase):
    """``int_bits`` includes the sign bit of a signed format (as in APyTypes
    and VHDL fixed_pkg); ``frac_bits`` may be negative or exceed the width."""
    int_bits: int
    frac_bits: int
    signed: bool = True
    _: KW_ONLY
    rounding: Rounding = Rounding.RNE
    overflow: str = "wrap"

    def __post_init__(self):
        for name in ("int_bits", "frac_bits"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool):
                raise TypeError(f"{name} must be an int")
        if self.bits < 1 or (self.signed and self.bits < 1):
            raise ValueError("a fixed-point format needs at least 1 bit")
        if max(abs(self.int_bits), abs(self.frac_bits), self.bits) > _MAX_BITS:
            raise OverflowError("fixed-point widths above 2**24 bits are not supported")
        if self.overflow not in ("wrap", "saturate"):
            raise ValueError("overflow must be 'wrap' or 'saturate'")
        if not isinstance(self.rounding, Rounding) or self.rounding is Rounding.SR:
            raise ValueError("rounding must be RNE, RNA, RTZ, RUP or RDN")
        object.__setattr__(self, "signed", bool(self.signed))
        self._setup(self.int_bits, self.frac_bits, self.signed,
                    _ROUNDINGS.index(self.rounding), self.overflow == "saturate")

    @property
    def bits(self) -> int:
        return self.int_bits + self.frac_bits

    @property
    def size(self) -> int:
        return self.bits

    @property
    def ulp(self) -> Fraction:
        return _pow2(-self.frac_bits)

    @property
    def min_raw(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def max_raw(self) -> int:
        return (1 << (self.bits - (1 if self.signed else 0))) - 1

    @property
    def min(self) -> Fraction:
        return self.min_raw * self.ulp

    @property
    def max(self) -> Fraction:
        return self.max_raw * self.ulp

    # Calling a format rounds a value into it (``fmt(value)``), and
    # ``from_raw(raw)`` builds from a bit pattern: both native.

    def replace(self, **changes) -> "FixedFormat":
        from dataclasses import replace
        return replace(self, **changes)

    def __reduce__(self):
        return (_rebuild_format, (self.int_bits, self.frac_bits, self.signed,
                                  self.rounding, self.overflow))

    def __str__(self) -> str:
        name = f"{'s' if self.signed else 'u'}fix{self.int_bits}.{self.frac_bits}"
        if self.rounding is not Rounding.RNE:
            name += f", {self.rounding.value}"
        if self.overflow != "wrap":
            name += ", saturate"
        return name


def _rebuild_format(int_bits, frac_bits, signed, rounding, overflow):
    return FixedFormat(int_bits, frac_bits, signed, rounding=rounding, overflow=overflow)


_core._init_fixed(FixedFormat, CastWarning)

FIXED = _core.FIXED


def _cast(self, int_bits: int | FixedFormat, frac_bits: int | None = None,
          signed: bool | None = None, **modes) -> FIXED:
    """Convert to another fixed-point format (default: same signedness)."""
    if isinstance(int_bits, FixedFormat):
        fmt = int_bits
    else:
        fmt = FixedFormat(int_bits, frac_bits, self.signed if signed is None else signed, **modes)
    return _core.fixed_cast_value(self, fmt)


def _cast_value(cls, value, fmt: FixedFormat) -> FIXED:
    """Round (per fmt.rounding) and wrap/saturate (per fmt.overflow)."""
    return _core.fixed_cast_value(value, fmt)


def _restore(fmt: FixedFormat, raw: int, flags: int) -> FIXED:
    x = fmt.from_raw(raw)
    return x if not flags else _core.fixed_with_flags(x, flags)


_cast.__name__, _cast.__qualname__ = "cast", "FIXED.cast"
_cast_value.__name__, _cast_value.__qualname__ = "cast_value", "FIXED.cast_value"
FIXED.cast = _cast
FIXED.cast_value = classmethod(_cast_value)
FIXED.__reduce__ = lambda self: (_restore, (self.format, self.raw, int(self.flags)))
