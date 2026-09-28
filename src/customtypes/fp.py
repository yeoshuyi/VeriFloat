from __future__ import annotations

import math
from fractions import Fraction
from functools import total_ordering

from .uint import UINT


def _pow2(k: int) -> Fraction:
    return Fraction(1 << k) if k >= 0 else Fraction(1, 1 << -k)


@total_ordering
class FP:
    __slots__ = ("_sign", "_exp", "_mantissa")

    def __init__(self, sign: bool, exp: UINT, mantissa: UINT):
        if not isinstance(exp, UINT) or not isinstance(mantissa, UINT):
            raise TypeError("exp and mantissa must be UINT")
        if exp.bits < 2:
            raise ValueError("need at least 2 exponent bits")
        self._sign = bool(sign)
        self._exp = exp
        self._mantissa = mantissa

    # Helpers
    @classmethod
    def from_value(cls, value, exp_bits: int = 2, mantissa_bits: int = 1) -> "FP":
        if isinstance(value, FP):
            return value.convert(exp_bits, mantissa_bits)
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ValueError("NaN and inf are not supported")
            zero_sign = math.copysign(1.0, value) < 0
        else:
            zero_sign = False
        return cls._encode(Fraction(value), exp_bits, mantissa_bits, zero_sign)

    @classmethod
    def zero(cls, exp_bits, mantissa_bits, sign=False) -> "FP":
        return cls(sign, UINT(0, exp_bits), UINT(0, mantissa_bits))

    @classmethod
    def max_value(cls, exp_bits, mantissa_bits, sign=False) -> "FP":
        return cls(sign, UINT(-1, exp_bits), UINT(-1, mantissa_bits))

    @classmethod
    def from_raw(cls, raw: int, exp_bits: int = 2, mantissa_bits: int = 1) -> "FP":
        m = UINT(raw, mantissa_bits)
        e = UINT(raw >> mantissa_bits, exp_bits)
        s = (raw >> (mantissa_bits + exp_bits)) & 1
        return cls(bool(s), e, m)

    # Getters
    @property
    def sign(self) -> bool:
        return self._sign

    @property
    def exp(self) -> UINT:
        return self._exp

    @property
    def mantissa(self) -> UINT:
        return self._mantissa

    @property
    def exp_bits(self) -> int:
        return self._exp.bits

    @property
    def mantissa_bits(self) -> int:
        return self._mantissa.bits

    @property
    def size(self) -> int:
        """Storage width in bits: sign + exponent + mantissa."""
        return 1 + self.exp_bits + self.mantissa_bits

    @property
    def bias(self) -> int:
        return (1 << (self.exp_bits - 1)) - 1

    @property
    def raw(self) -> int:
        return ((int(self._sign) << (self.exp_bits + self.mantissa_bits))
                | (self._exp.val << self.mantissa_bits)
                | self._mantissa.val)

    @property
    def is_zero(self) -> bool:
        return self._exp.val == 0 and self._mantissa.val == 0

    @property
    def is_subnormal(self) -> bool:
        return self._exp.val == 0 and self._mantissa.val != 0

    def _fmt(self):
        return self.exp_bits, self.mantissa_bits

    def _exact(self) -> Fraction:
        M, b, m = self.mantissa_bits, self._exp.val, self._mantissa.val
        if b == 0:
            v = m * _pow2(1 - self.bias - M)
        else:
            v = ((1 << M) | m) * _pow2(b - self.bias - M)
        return -v if self._sign else v

    @classmethod
    def _encode(cls, x: Fraction, E: int, M: int, zero_sign: bool = False) -> "FP":
        if x == 0:
            return cls.zero(E, M, zero_sign)
        bias = (1 << (E - 1)) - 1
        emin, emax = 1 - bias, ((1 << E) - 1) - bias
        sign = x < 0
        x = abs(x)
        n, d = x.numerator, x.denominator
        e = n.bit_length() - d.bit_length()
        if (n << max(0, -e)) < (d << max(0, e)):
            e -= 1
        e = max(e, emin)

        shift = e - M
        num, den = (n, d << shift) if shift >= 0 else (n << -shift, d)
        sig, rem = divmod(num, den)
        if 2 * rem > den or (2 * rem == den and sig & 1):
            sig += 1
        if sig >> (M + 1):
            sig >>= 1
            e += 1
        if e > emax:
            return cls.max_value(E, M, sign)
        if sig >> M:
            return cls(sign, UINT(e + bias, E), UINT(sig, M))
        return cls(sign, UINT(0, E), UINT(sig, M))

    def convert(self, exp_bits: int, mantissa_bits: int) -> "FP":
        return FP._encode(self._exact(), exp_bits, mantissa_bits, self._sign)

    def _coerce(self, other):
        """Return (self, other) as FPs in a common format, or None."""
        if isinstance(other, FP):
            E = max(self.exp_bits, other.exp_bits)
            M = max(self.mantissa_bits, other.mantissa_bits)
            return self.convert(E, M), other.convert(E, M)
        if isinstance(other, (int, float, Fraction)):
            return self, FP.from_value(other, *self._fmt())
        return None

    def _binop(self, other, fn, reverse=False):
        c = self._coerce(other)
        if c is None:
            return NotImplemented
        a, b = c
        return fn(b, a) if reverse else fn(a, b)

    @staticmethod
    def _add(a: "FP", b: "FP") -> "FP":
        E, M = a._fmt()
        return FP._encode(a._exact() + b._exact(), E, M, a.sign and b.sign)

    @staticmethod
    def _sub(a: "FP", b: "FP") -> "FP":
        return FP._add(a, -b)

    @staticmethod
    def _mul(a: "FP", b: "FP") -> "FP":
        E, M = a._fmt()
        return FP._encode(a._exact() * b._exact(), E, M, a.sign != b.sign)

    @staticmethod
    def _div(a: "FP", b: "FP") -> "FP":
        E, M = a._fmt()
        if b.is_zero:
            raise ZeroDivisionError("FP division by zero")
        return FP._encode(a._exact() / b._exact(), E, M, a.sign != b.sign)

    # Arith
    def __add__(self, o):       return self._binop(o, FP._add)
    def __radd__(self, o):      return self._binop(o, FP._add, True)
    def __sub__(self, o):       return self._binop(o, FP._sub)
    def __rsub__(self, o):      return self._binop(o, FP._sub, True)
    def __mul__(self, o):       return self._binop(o, FP._mul)
    def __rmul__(self, o):      return self._binop(o, FP._mul, True)
    def __truediv__(self, o):   return self._binop(o, FP._div)
    def __rtruediv__(self, o):  return self._binop(o, FP._div, True)

    def __neg__(self):  return FP(not self._sign, self._exp, self._mantissa)
    def __pos__(self):  return FP(self._sign, self._exp, self._mantissa)
    def __abs__(self):  return FP(False, self._exp, self._mantissa)

    # Logical
    def _bitop(self, other, fn, reverse=False):
        c = self._coerce(other)
        if c is None:
            return NotImplemented
        a, b = c
        if reverse:
            a, b = b, a
        return FP.from_raw(fn(a.raw, b.raw), *a._fmt())

    def __and__(self, o):   return self._bitop(o, lambda a, b: a & b)
    def __rand__(self, o):  return self._bitop(o, lambda a, b: a & b, True)
    def __or__(self, o):    return self._bitop(o, lambda a, b: a | b)
    def __ror__(self, o):   return self._bitop(o, lambda a, b: a | b, True)
    def __xor__(self, o):   return self._bitop(o, lambda a, b: a ^ b)
    def __rxor__(self, o):  return self._bitop(o, lambda a, b: a ^ b, True)
    def __invert__(self):
        return FP.from_raw(~self.raw & ((1 << self.size) - 1), *self._fmt())

    # Comparison
    def _cmp(self, other, fn):
        c = self._coerce(other)
        if c is None:
            return NotImplemented
        a, b = c
        return fn(a._exact(), b._exact())

    def __eq__(self, o): return self._cmp(o, lambda a, b: a == b)
    def __lt__(self, o): return self._cmp(o, lambda a, b: a < b)

    def __hash__(self):
        return hash(self._exact())

    # Casting
    def __float__(self):
        if self.is_zero:
            return -0.0 if self._sign else 0.0
        return float(self._exact())

    def __int__(self):
        return int(self._exact())

    def __bool__(self):
        return not self.is_zero

    def to_bin(self) -> str:
        return (f"{int(self._sign)} "
                f"{self._exp.to_bin()} "
                f"{self._mantissa.to_bin()}")

    def __repr__(self):
        return f"FP({float(self)!r}, e{self.exp_bits}m{self.mantissa_bits})"

    def __str__(self):
        return str(float(self))
