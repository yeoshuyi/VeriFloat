from __future__ import annotations

from functools import total_ordering


@total_ordering
class UINT:
    __slots__ = ("_bits", "_val")
    signed = False

    def __init__(self, val: int = 0, bits: int = 32):
        if not isinstance(bits, int) or bits <= 0:
            raise ValueError("bits must be a positive integer")
        self._bits = bits
        self._val = int(val) & self.mask

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
    def mask(self) -> int:
        return (1 << self._bits) - 1

    @property
    def min(self) -> int:
        return 0

    @property
    def max(self) -> int:
        return self.mask

    # Helpers
    def _coerce(self, other):
        if isinstance(other, UINT):
            bits = max(self._bits, other._bits)
            if self.signed and other.signed:
                return self.val, other.val, bits, type(self)
            cls = type(other) if self.signed else type(self)
            return self.raw, other.raw, bits, cls
        if isinstance(other, int):
            return self.val, other, self._bits, type(self)
        return None

    def _binop(self, other, fn, reverse=False):
        c = self._coerce(other)
        if c is None:
            return NotImplemented
        a, b, bits, cls = c
        if reverse:
            a, b = b, a
        return cls(fn(a, b), bits)

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

    # Logical
    def __and__(self, o):   return self._binop(o, lambda a, b: a & b)
    def __rand__(self, o):  return self._binop(o, lambda a, b: a & b, True)
    def __or__(self, o):    return self._binop(o, lambda a, b: a | b)
    def __ror__(self, o):   return self._binop(o, lambda a, b: a | b, True)
    def __xor__(self, o):   return self._binop(o, lambda a, b: a ^ b)
    def __rxor__(self, o):  return self._binop(o, lambda a, b: a ^ b, True)
    def __invert__(self):   return type(self)(~self._val, self._bits)

    def __lshift__(self, n):
        n = int(n)
        if n < 0:
            raise ValueError("negative shift count")
        return type(self)(self._val << n, self._bits)

    def __rshift__(self, n):
        n = int(n)
        if n < 0:
            raise ValueError("negative shift count")
        return type(self)(self.val >> n, self._bits)

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
        return type(self)(self.val, bits)

    def __int__(self):   return self.val
    def __index__(self): return self.val
    def __bool__(self):  return self._val != 0

    def __repr__(self):
        kind = "i" if self.signed else "u"
        return f"{type(self).__name__}({self.val}, {kind}{self._bits})"

    def to_bin(self) -> str:
        return format(self._val, f"0{self._bits}b")
