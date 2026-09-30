"""Fixed-point numbers: value = raw * 2**-frac_bits.

Arithmetic is exact: + and - grow the result by one integer bit and * adds the
widths, as VHDL ``sfixed``/``ufixed`` and APyTypes do. Rounding and overflow
happen only on an explicit ``cast`` (or when a format is called on a value),
with the rounding modes of ``Rounding`` and wrap or saturate on overflow.
"""

from __future__ import annotations

import math
from dataclasses import KW_ONLY, dataclass
from fractions import Fraction

from ._warnings import CastWarning, IntCastWarning
from ._warnings import warn as _warn
from .fp import FP, FPFlags, Rounding, _pow2, _round_to_int
from .uint import UINT

_F = FPFlags


@dataclass(frozen=True)
class FixedFormat:
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
        if self.overflow not in ("wrap", "saturate"):
            raise ValueError("overflow must be 'wrap' or 'saturate'")
        if not isinstance(self.rounding, Rounding) or self.rounding is Rounding.SR:
            raise ValueError("rounding must be RNE, RNA, RTZ, RUP or RDN")
        object.__setattr__(self, "signed", bool(self.signed))

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

    def __call__(self, value) -> "FIXED":
        return FIXED.cast_value(value, self)

    def from_raw(self, raw: int) -> "FIXED":
        """Build from a bit pattern (two's complement if signed)."""
        return FIXED._make(_wrap(raw, self), self)

    def replace(self, **changes) -> "FixedFormat":
        from dataclasses import replace
        return replace(self, **changes)

    def __str__(self) -> str:
        name = f"{'s' if self.signed else 'u'}fix{self.int_bits}.{self.frac_bits}"
        if self.rounding is not Rounding.RNE:
            name += f", {self.rounding.value}"
        if self.overflow != "wrap":
            name += ", saturate"
        return name


def _wrap(q: int, fmt: FixedFormat) -> int:
    q &= (1 << fmt.bits) - 1
    if fmt.signed and q >> (fmt.bits - 1):
        q -= 1 << fmt.bits
    return q


def _to_fraction(v) -> Fraction:
    if isinstance(v, FIXED):
        return v.exact
    if isinstance(v, FP):
        return v.exact
    if isinstance(v, UINT):
        return Fraction(v.val)
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError("fixed point has no inf/NaN")
    return Fraction(v)


class FIXED:
    """A fixed-point value: an integer ``val`` scaled by 2**-frac_bits."""
    __slots__ = ("_val", "_format", "_flags")

    @classmethod
    def _make(cls, val: int, fmt: FixedFormat, flags: FPFlags = _F(0)) -> "FIXED":
        x = object.__new__(cls)
        x._val, x._format, x._flags = val, fmt, flags
        return x

    @classmethod
    def cast_value(cls, value, fmt: FixedFormat) -> "FIXED":
        """Round (per fmt.rounding) and wrap/saturate (per fmt.overflow)."""
        x = _to_fraction(value)
        q, inexact = _round_to_int(x / fmt.ulp, fmt.rounding)
        flags = _F.INEXACT if inexact else _F(0)
        if not fmt.min_raw <= q <= fmt.max_raw:
            flags |= _F.OVERFLOW
            q = (min(max(q, fmt.min_raw), fmt.max_raw) if fmt.overflow == "saturate"
                 else _wrap(q, fmt))
        return cls._make(q, fmt, flags)

    def cast(self, int_bits: int | FixedFormat, frac_bits: int | None = None,
             signed: bool | None = None, **modes) -> "FIXED":
        """Convert to another fixed-point format (default: same signedness)."""
        if isinstance(int_bits, FixedFormat):
            fmt = int_bits
        else:
            fmt = FixedFormat(int_bits, frac_bits,
                              self.signed if signed is None else signed, **modes)
        return FIXED.cast_value(self, fmt)

    # Getters
    @property
    def format(self) -> FixedFormat:
        return self._format

    @property
    def val(self) -> int:
        """The scaled integer (signed if the format is)."""
        return self._val

    @property
    def raw(self) -> int:
        """Bit pattern (two's complement)."""
        return self._val & ((1 << self._format.bits) - 1)

    @property
    def exact(self) -> Fraction:
        return self._val * self._format.ulp

    @property
    def flags(self) -> FPFlags:
        """INEXACT and/or OVERFLOW from the cast that produced this value."""
        return self._flags

    @property
    def signed(self) -> bool:
        return self._format.signed

    @property
    def int_bits(self) -> int:
        return self._format.int_bits

    @property
    def frac_bits(self) -> int:
        return self._format.frac_bits

    @property
    def bits(self) -> int:
        return self._format.bits

    size = bits

    # Exact arithmetic with bit growth
    def _other(self, o) -> "FIXED | None":
        if isinstance(o, FIXED):
            return o
        if isinstance(o, (int, Fraction, float)) and not isinstance(o, bool):
            # A literal is cast into this format; warn if that loses anything.
            r = FIXED.cast_value(o, self._format)
            if r.flags:
                _warn(CastWarning, f"implicit cast of {o!r} to {self._format} is lossy: "
                      f"{float(r.exact)!r} ({r.flags.name})")
            return r
        return None

    @staticmethod
    def _grow(a: "FIXED", b: "FIXED", op: str) -> FixedFormat:
        fa, fb = a._format, b._format
        if fa.signed != fb.signed:
            _warn(IntCastWarning, f"mixing signedness: {fa} and {fb}, "
                  f"the unsigned operand gains a sign bit")
        signed = fa.signed or fb.signed or op == "-"
        # An unsigned operand needs one more integer bit once signed.
        ia = fa.int_bits + (signed and not fa.signed)
        ib = fb.int_bits + (signed and not fb.signed)
        if op == "*":
            return FixedFormat(ia + ib, fa.frac_bits + fb.frac_bits, signed,
                               rounding=fa.rounding, overflow=fa.overflow)
        return FixedFormat(max(ia, ib) + 1, max(fa.frac_bits, fb.frac_bits), signed,
                           rounding=fa.rounding, overflow=fa.overflow)

    def _binop(self, o, op, reverse=False):
        b = self._other(o)
        if b is None:
            return NotImplemented
        a = self
        if reverse:
            a, b = b, a
        fmt = FIXED._grow(a, b, op)
        x = {"+": a.exact + b.exact, "-": a.exact - b.exact,
             "*": a.exact * b.exact}[op]
        return FIXED._make(int(x / fmt.ulp), fmt)   # exact by construction

    def __add__(self, o):   return self._binop(o, "+")
    def __radd__(self, o):  return self._binop(o, "+", True)
    def __sub__(self, o):   return self._binop(o, "-")
    def __rsub__(self, o):  return self._binop(o, "-", True)
    def __mul__(self, o):   return self._binop(o, "*")
    def __rmul__(self, o):  return self._binop(o, "*", True)

    def __neg__(self):
        f = self._format
        fmt = FixedFormat(f.int_bits + 1, f.frac_bits, True,
                          rounding=f.rounding, overflow=f.overflow)
        return FIXED._make(-self._val, fmt)

    def div(self, other, fmt: FixedFormat) -> "FIXED":
        """Quotient rounded into an explicit format (division is not exact)."""
        b = self._other(other)
        if b.exact == 0:
            raise ZeroDivisionError("fixed-point division by zero")
        return FIXED.cast_value(self.exact / b.exact, fmt)

    def __lshift__(self, n: int) -> "FIXED":
        """Multiply by 2**n exactly (moves the binary point)."""
        f = self._format
        return FIXED._make(self._val, f.replace(int_bits=f.int_bits + n,
                                                frac_bits=f.frac_bits - n))

    def __rshift__(self, n: int) -> "FIXED":
        return self << -n

    # Comparison (by value)
    def _key(self, o):
        return _to_fraction(o) if not isinstance(o, FIXED) else o.exact

    def __eq__(self, o):
        try:
            return self.exact == self._key(o)
        except (TypeError, ValueError):
            return NotImplemented

    def __lt__(self, o): return self.exact < self._key(o)
    def __le__(self, o): return self.exact <= self._key(o)
    def __gt__(self, o): return self.exact > self._key(o)
    def __ge__(self, o): return self.exact >= self._key(o)

    def __hash__(self):
        return hash(self.exact)

    def __float__(self):
        return float(self.exact)

    def __int__(self):
        return int(self.exact)

    def __bool__(self):
        return self._val != 0

    def to_bin(self) -> str:
        return format(self.raw, f"0{self.bits}b")

    def to_hex(self) -> str:
        return format(self.raw, f"0{(self.bits + 3) // 4}x")

    def __repr__(self):
        extra = f", flags={self._flags.name}" if self._flags else ""
        return f"FIXED({float(self.exact)!r}, {self._format}{extra})"
