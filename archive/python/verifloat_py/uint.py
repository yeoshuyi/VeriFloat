from __future__ import annotations

from functools import total_ordering

from ._warnings import IntCastWarning
from ._warnings import warn as _warn


@total_ordering
class UINT:
    __slots__ = ("_bits", "_val", "_sat")
    signed = False

    def __init__(self, val: int = 0, bits: int = 32, saturate: bool = False):
        """``saturate=True``: out-of-range values (at construction and from
        arithmetic) clamp to [min, max] instead of wrapping."""
        # Zero-width UINTs exist only as empty fields (e.g. an E8M0 mantissa).
        if (not isinstance(bits, int) or isinstance(bits, bool) or bits < 0
                or (bits == 0 and self.signed)):
            raise ValueError("bits must be a positive integer")
        self._bits = bits
        self._sat = bool(saturate)
        val = int(val)
        if self._sat:
            val = min(max(val, self.min), self.max)
        self._val = val & self.mask

    # Getters
    @property
    def bits(self) -> int:
        return self._bits

    @property
    def size(self) -> int:
        """Storage width in bits."""
        return self._bits

    @property
    def val(self) -> int:
        return self._val

    @property
    def raw(self) -> int:
        return self._val

    @property
    def saturate(self) -> bool:
        return self._sat

    @property
    def mask(self) -> int:
        return (1 << self._bits) - 1

    @property
    def min(self) -> int:
        return 0

    @property
    def max(self) -> int:
        return self.mask

    # Helpers
    @property
    def _kind(self) -> str:
        return f"{'i' if self.signed else 'u'}{self._bits}"

    def _coerce(self, other):
        if isinstance(other, UINT):
            if self._sat != other._sat:
                _warn(IntCastWarning, f"mismatched saturate: {self._sat} (left) vs "
                      f"{other._sat} (right), using {self._sat}")
            bits = max(self._bits, other._bits)
            if self.signed and other.signed:
                cls, a, b = type(self), self.val, other.val
            else:
                cls = type(other) if self.signed else type(self)
                a, b = self.raw, other.raw
            if self.signed != other.signed:
                _warn(IntCastWarning, f"mixing signedness: {self._kind} and "
                      f"{other._kind}, signed operand reinterpreted as "
                      f"u{bits} raw bits")
            elif self._bits != other._bits:
                ext = "sign-extended" if self.signed else "zero-extended"
                _warn(IntCastWarning, f"mixing widths: {self._kind} and "
                      f"{other._kind}, {ext} to {cls(0, bits)._kind}")
            return a, b, bits, cls
        if isinstance(other, int):
            # A literal lives in a register of the operand's type: cast it.
            cast = type(self)(other, self._bits, self._sat)
            if not self.min <= other <= self.max:
                _warn(IntCastWarning, f"implicit cast of {other} to "
                      f"{self._kind} is lossy: {cast.val}")
            return self.val, cast.val, self._bits, type(self)
        return None

    def _binop(self, other, fn, reverse=False, arith=True):
        c = self._coerce(other)
        if c is None:
            return NotImplemented
        a, b, bits, cls = c
        if reverse:
            a, b = b, a
        return cls._new(fn(a, b), bits, self._sat, arith)

    @classmethod
    def _new(cls, v: int, bits: int, sat: bool, clamp: bool = True) -> "UINT":
        """Result in saturate mode ``sat``; bit operations wrap even then."""
        x = cls(v, bits, sat and clamp)
        x._sat = sat
        return x

    @staticmethod
    def _div(a, b):
        if b == 0:
            raise ZeroDivisionError("division by zero")
        q = abs(a) // abs(b)
        return q if (a < 0) == (b < 0) else -q

    @staticmethod
    def _mod(a, b):
        return a - UINT._div(a, b) * b

    # Arith
    def __add__(self, o):       return self._binop(o, lambda a, b: a + b)
    def __radd__(self, o):      return self._binop(o, lambda a, b: a + b, True)
    def __sub__(self, o):       return self._binop(o, lambda a, b: a - b)
    def __rsub__(self, o):      return self._binop(o, lambda a, b: a - b, True)
    def __mul__(self, o):       return self._binop(o, lambda a, b: a * b)
    def __rmul__(self, o):      return self._binop(o, lambda a, b: a * b, True)
    def __floordiv__(self, o):  return self._binop(o, UINT._div)
    def __rfloordiv__(self, o): return self._binop(o, UINT._div, True)
    def __mod__(self, o):       return self._binop(o, UINT._mod)
    def __rmod__(self, o):      return self._binop(o, UINT._mod, True)

    def __neg__(self):      return type(self)._new(-self.val, self._bits, self._sat)

    # Logical (bit operations wrap, even in saturate mode)
    def __and__(self, o):   return self._binop(o, lambda a, b: a & b, arith=False)
    def __rand__(self, o):  return self._binop(o, lambda a, b: a & b, True, False)
    def __or__(self, o):    return self._binop(o, lambda a, b: a | b, arith=False)
    def __ror__(self, o):   return self._binop(o, lambda a, b: a | b, True, False)
    def __xor__(self, o):   return self._binop(o, lambda a, b: a ^ b, arith=False)
    def __rxor__(self, o):  return self._binop(o, lambda a, b: a ^ b, True, False)
    def __invert__(self):   return type(self)._new(~self._val, self._bits, self._sat, False)

    def __lshift__(self, n):
        n = int(n)
        if n < 0:
            raise ValueError("negative shift count")
        return type(self)._new(self._val << n, self._bits, self._sat, False)

    def __rshift__(self, n):
        n = int(n)
        if n < 0:
            raise ValueError("negative shift count")
        return type(self)._new(self.val >> n, self._bits, self._sat, False)

    # Comparison
    def _cmp(self, other, fn):
        c = self._coerce(other)
        if c is None:
            return NotImplemented
        a, b, _, _ = c
        return fn(a, b)

    def __eq__(self, o): return self._cmp(o, lambda a, b: a == b)
    def __lt__(self, o): return self._cmp(o, lambda a, b: a < b)

    def __hash__(self):
        return hash(self.val)

    # Casting
    def resize(self, bits: int) -> "UINT":
        """Change width: truncate (or clamp, in saturate mode) or extend."""
        return type(self)(self.val, bits, self._sat)

    def __int__(self):   return self.val
    def __index__(self): return self.val
    def __bool__(self):  return self._val != 0

    def __repr__(self):
        kind = "i" if self.signed else "u"
        sat = ", sat" if self._sat else ""
        return f"{type(self).__name__}({self.val}, {kind}{self._bits}{sat})"

    def to_bin(self) -> str:
        return format(self._val, f"0{self._bits}b") if self._bits else ""

    def to_hex(self) -> str:
        return format(self._val, f"0{(self._bits + 3) // 4}x")
