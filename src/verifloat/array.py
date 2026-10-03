"""FPArray: an N-dimensional array of FP values in one format.

The array stores raw codes and runs in the C++ core. Every operation gives,
element by element, exactly what the scalar FP operation gives: the same code,
the same flags and the same warnings. Use it where a testbench handles whole
tensors at once::

    a = FP16.array([[1.5, 2.25], [0.1, -3.0]])   # round floats into the format
    b = FPArray.from_raw([[0x3c00, 0x4000], [0x0001, 0xfc00]], FP16)
    c = a + b                                    # also - * /, FP or number operands
    c.raw, c.flags, c.tolist(), c[0, 1]
    matmul(a, b, acc=Accumulator(FP32, "sequential"))   # an FPArray in FP32

Indexing takes integers, slices, ``...`` and ``None`` as NumPy's basic
indexing does (always a copy), and operands broadcast as in NumPy. The
element-wise methods are the FP methods of the same names (``sqrt``, ``fma``,
``minimum``, ``remainder``, ``eq`` ...), ``sum`` adds elements as an
Accumulator models it, and ``FPArray.full`` / ``zeros`` / ``ones`` build
constant arrays.

NumPy arrays (and other numeric buffers) are accepted wherever nested lists
are, and ``to_numpy("value" | "raw" | "flags")`` gives arrays back; NumPy
itself stays optional.

Formats of up to 64 bits are supported. ``flags`` is the OR over all elements;
each FP from ``tolist()`` or indexing carries its own. Arrays do not keep the
``unrounded`` values.
"""

from __future__ import annotations

import inspect

from . import _core
from .fp import FP

FPArray = _core.FPArray


def _restore(fmt, shape, raw, flags) -> FPArray:
    """Unpickle an FPArray (see FPArray.__reduce__)."""
    return _core.array_restore(fmt, shape, raw, flags)


FPArray._restore = staticmethod(_restore)


def _to_numpy(self, what: str = "value"):
    """The elements as a new NumPy array of the same shape.

    what="value"  float64 values (exact when the format fits binary64, else
                  rounded as ``float(x)`` rounds; NaN and inf are kept)
    what="raw"    the codes, in the smallest unsigned dtype that holds them
    what="flags"  each element's status flags (FPFlags bits), uint8
    """
    try:
        import numpy as np
    except ImportError as e:          # NumPy is optional
        raise ImportError("FPArray.to_numpy needs NumPy") from e
    if what == "value":
        out = np.frombuffer(self._bytes(2), dtype=np.float64)
    elif what == "raw":
        bits = self.format.size
        dtype = np.uint8 if bits <= 8 else np.uint16 if bits <= 16 else np.uint32 if bits <= 32 else np.uint64
        out = np.frombuffer(self._bytes(0), dtype=np.uint64).astype(dtype)
    elif what == "flags":
        out = np.frombuffer(self._bytes(1), dtype=np.uint8)
    else:
        raise ValueError("what must be 'value', 'raw' or 'flags'")
    return out.reshape(self.shape).copy()


_to_numpy.__name__, _to_numpy.__qualname__ = "to_numpy", "FPArray.to_numpy"
FPArray.to_numpy = _to_numpy

# ---------------------------------------------------------------- element-wise methods
# Each is the FP method of the same name applied to every element: the same
# arguments, defaults, results, flags and warnings. Array arguments broadcast
# with the array; an FP or a number is used for every element.

_VALUE_METHODS = ("sqrt", "fma", "minimum", "maximum", "minimum_number", "maximum_number", "remainder", "fmod",
                  "copysign", "fsgnj", "fsgnjn", "fsgnjx", "next_up", "next_down", "scaleb", "logb",
                  "round_to_integral")
_PAIR_METHODS = ("eq", "lt", "le", "compare", "to_int")     # FP gives (value, flags)
_PLAIN_METHODS = ("fclass",)


def _elementwise(name: str, returns: str):
    scalar = getattr(FP, name)
    sig = inspect.signature(scalar)

    def method(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()          # so that every argument is positional
        return _core.array_map(name, bound.args)

    method.__name__, method.__qualname__ = name, f"FPArray.{name}"
    method.__signature__ = sig
    method.__doc__ = (f"FP.{name} of every element: {returns}. FPArray arguments broadcast.\n\n"
                      f"FP.{name}: {inspect.getdoc(scalar) or ''}").rstrip()
    return method


for _name in _VALUE_METHODS:
    setattr(FPArray, _name, _elementwise(_name, "an FPArray"))
for _name in _PAIR_METHODS:
    setattr(FPArray, _name, _elementwise(_name, "(nested lists of the results, the OR of all flags)"))
for _name in _PLAIN_METHODS:
    setattr(FPArray, _name, _elementwise(_name, "nested lists"))
del _name


# ---------------------------------------------------------------- constructors

def _full(shape, value, fmt) -> FPArray:
    """An array of this shape with every element ``fmt(value)`` (an int shape
    is one axis). Each element carries the flags of that rounding."""
    return FPArray([value], fmt).broadcast_to(shape)


def _zeros(shape, fmt) -> FPArray:
    """An array of ``fmt(0)``."""
    return FPArray([0], fmt).broadcast_to(shape)


def _ones(shape, fmt) -> FPArray:
    """An array of ``fmt(1)``."""
    return FPArray([1], fmt).broadcast_to(shape)


for _fn, _name in ((_full, "full"), (_zeros, "zeros"), (_ones, "ones")):
    _fn.__name__, _fn.__qualname__ = _name, f"FPArray.{_name}"
    setattr(FPArray, _name, staticmethod(_fn))
del _fn, _name


# ---------------------------------------------------------------- reductions

def _sum(self, axis: int | None = None, acc=None):
    """The sum of the elements, added as ``acc`` says, in index order.

    acc     None: the exact sum (a Fraction). An FPFormat: the exact sum
            rounded once into it. An Accumulator: its model of a hardware
            adder chain or tree (see Accumulator); the flags are the OR of
            every rounding.
    axis    None: all elements in C order, one result. An axis: the sum
            along it for every position of the other axes, as an FPArray
            (nested lists of Fractions when ``acc`` is None); a 1-D array
            gives one result.

    Exactly ``dot(elements, [1, 1, ...], acc)`` for each sum.
    """
    from .blockscale import dot
    if axis is None:
        return dot(self._flat(), [1] * len(self._flat()), acc)
    nd = self.ndim
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise TypeError("axis must be an int or None")
    if not -nd <= axis < nd:
        raise ValueError(f"axis {axis} is out of range for an array of {nd} dimensions")
    axis %= nd
    k = self.shape[axis]
    rows = self.transpose(*[d for d in range(nd) if d != axis], axis)._flat()
    ones = [1] * k
    sums = [dot(rows[i:i + k], ones, acc) for i in range(0, len(rows), k)]
    if nd == 1:
        return sums[0]
    shape = tuple(d for i, d in enumerate(self.shape) if i != axis)
    if acc is not None:
        return _core.array_from_fps(sums, shape)
    for n in reversed(shape[1:]):       # nested lists of Fractions
        sums = [sums[i:i + n] for i in range(0, len(sums), n)]
    return sums


_sum.__name__, _sum.__qualname__ = "sum", "FPArray.sum"
FPArray.sum = _sum

__all__ = ["FPArray"]
