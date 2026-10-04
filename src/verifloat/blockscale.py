"""Block-scaled tensors of UINT/INT/FP elements (NVFP4, OCP MX and custom).

A tensor (any number of dimensions) is split into blocks of ``block_size``
consecutive elements along one axis (the last, by default). Each block stores low-precision element codes plus a shared scale
(and optionally an additive zero-point); the whole tensor may carry one more
per-tensor scale:

    value = tensor_scale * block_scale * (element - zero_point)

Every quantity is kept exactly (``Fraction``), so the codes a ``BlockTensor``
holds are exactly what a bit-accurate quantizer or tensor core must produce.
"""

from __future__ import annotations

import math
from dataclasses import KW_ONLY, dataclass
from fractions import Fraction

from . import _core
from ._warnings import BlockFormatWarning, CastWarning
from ._warnings import warn as _warn
from .accum import Accumulator
from .array import FPArray
from .fp import E2M1, E4M3, FP, FP32, FPFlags, FPFormat, Rounding, _ROUNDINGS, _pow2, _round_sig
from .sint import INT
from .uint import UINT


def _to_fraction(v) -> Fraction:
    if isinstance(v, FP):
        return v.exact
    if isinstance(v, UINT):
        return Fraction(v.val)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        raise ValueError("NaN and inf are not supported")
    return Fraction(v)


@dataclass(frozen=True)
class IntFormat:
    """Integer element (or zero-point) format: code q means q * 2**-frac_bits.
    Quantizing into it rounds with ``rounding`` and saturates to [min, max],
    as standard quantizers do. OCP MXINT8 is IntFormat(8, frac_bits=6)."""
    bits: int = 8
    signed: bool = True
    rounding: Rounding = Rounding.RNE
    frac_bits: int = 0

    def __post_init__(self):
        if not isinstance(self.bits, int) or self.bits < 1:
            raise ValueError("bits must be a positive integer")
        if not isinstance(self.rounding, Rounding):
            raise TypeError("rounding must be a Rounding member")
        if not isinstance(self.frac_bits, int) or self.frac_bits < 0:
            raise ValueError("frac_bits must be a non-negative integer")
        if self.bits > 1 << 24 or self.frac_bits > 1 << 24:
            raise OverflowError("integer element widths above 2**24 bits are not supported")

    @property
    def unit(self) -> Fraction:
        """Value of one code step."""
        return _pow2(-self.frac_bits)

    @property
    def max_value(self) -> Fraction:
        return self.max * self.unit

    def value(self, q: UINT) -> Fraction:
        return q.val * self.unit

    @property
    def min(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def max(self) -> int:
        return (1 << (self.bits - (1 if self.signed else 0))) - 1

    @property
    def size(self) -> int:
        return self.bits

    def __call__(self, v: int) -> UINT:
        return (INT if self.signed else UINT)(v, self.bits)

    def from_raw(self, raw: int) -> UINT:
        return self(raw)

    def quantize(self, x: Fraction) -> tuple[UINT, FPFlags]:
        """Round a code-domain value x to an integer code and saturate."""
        neg = x < 0
        ax = abs(x)
        q, inexact = _round_sig(ax.numerator, ax.denominator, 0, neg, self.rounding)
        q = -q if neg else q
        flags = FPFlags.INEXACT if inexact else FPFlags(0)
        if q > self.max:
            q, flags = self.max, FPFlags.OVERFLOW | FPFlags.INEXACT
        elif q < self.min:
            under = neg and not self.signed
            q = self.min
            flags = (FPFlags.UNDERFLOW if under else FPFlags.OVERFLOW) | FPFlags.INEXACT
        return self(q), flags

    def __str__(self) -> str:
        name = f"{'i' if self.signed else 'u'}{self.bits}"
        if self.frac_bits:
            name += f", frac={self.frac_bits}"
        if self.rounding is not Rounding.RNE:
            name += f", {self.rounding.value}"
        return name


@dataclass(frozen=True)
class Pow2Format:
    """Exponent-only scale: ``value = 2 ** (code - bias)``, like OCP E8M0.
    With ``nan=True`` (E8M0) the all-ones code is NaN, not a scale."""
    bits: int = 8
    bias: int = 127
    nan: bool = True

    def __post_init__(self):
        if not isinstance(self.bits, int) or self.bits < 1:
            raise ValueError("bits must be a positive integer")

    @property
    def max_code(self) -> int:
        return (1 << self.bits) - 1 - int(self.nan)

    @property
    def max(self) -> Fraction:
        return self.value(self.max_code)

    @property
    def size(self) -> int:
        return self.bits

    def value(self, code: int) -> Fraction:
        if code > self.max_code:
            raise ValueError(f"scale code {code} is NaN in {self}")
        return _pow2(code - self.bias)

    def from_raw(self, raw: int) -> UINT:
        return UINT(raw, self.bits)

    def __str__(self) -> str:
        name = f"ue{self.bits}m0"
        if self.bias != (1 << (self.bits - 1)) - 1:
            name += f", bias={self.bias}"
        if not self.nan:
            name += ", no-nan"
        return name


@dataclass(frozen=True)
class BlockFormat:
    """How a tensor is block-quantized.

    elem          element format (FPFormat or IntFormat)
    block_size    elements per block, along the blocking axis
    scale         per-block scale format (FPFormat, Pow2Format, or None for 1)
    scale_max     optional upper clamp on block scales
    zero_point    optional per-block additive zero-point (IntFormat elements)
    tensor_scale  optional per-tensor scale format (e.g. FP32 for NVFP4)

    Recipe options (defaults give the exact recipe):
    compute       format of the recipe's intermediate arithmetic; None is
                  exact, FP32 reproduces frameworks that compute in float32
    scale_min     lower clamp on nonzero block scales (ModelOpt uses 2**-9)
    zero_scale    scale stored for an all-zero block (default: 0; ModelOpt 1)
    """
    elem: FPFormat | IntFormat = E2M1
    block_size: int = 16
    scale: FPFormat | Pow2Format | None = E4M3
    _: KW_ONLY
    scale_max: Fraction | int | None = None
    zero_point: IntFormat | None = None
    tensor_scale: FPFormat | None = None
    compute: FPFormat | None = None
    scale_min: Fraction | int | float | None = None
    zero_scale: Fraction | int | None = None

    def __post_init__(self):
        if not isinstance(self.elem, (FPFormat, IntFormat)):
            raise TypeError("elem must be an FPFormat or IntFormat")
        if not isinstance(self.block_size, int) or self.block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if self.scale is not None and not isinstance(self.scale, (FPFormat, Pow2Format)):
            raise TypeError("scale must be an FPFormat, Pow2Format or None")
        for name in ("tensor_scale", "compute"):
            v = getattr(self, name)
            if v is not None and not isinstance(v, FPFormat):
                raise TypeError(f"{name} must be an FPFormat or None")
        if self.zero_point is not None:
            if not isinstance(self.zero_point, IntFormat):
                raise TypeError("zero_point must be an IntFormat")
            if not isinstance(self.elem, IntFormat):
                raise ValueError("zero_point requires IntFormat elements")
            if not isinstance(self.scale, FPFormat):
                raise ValueError("zero_point requires an FPFormat scale")
        for name in ("scale_max", "scale_min", "zero_scale"):
            v = getattr(self, name)
            if v is not None:
                v = Fraction(v)
                if v < 0 or (v == 0 and name != "zero_scale"):
                    raise ValueError(f"{name} must be positive")
                object.__setattr__(self, name, v)

    @property
    def elem_max(self) -> Fraction:
        e = self.elem
        return e.max_value if isinstance(e, IntFormat) else e.max

    @property
    def scale_cap(self) -> Fraction:
        """Largest block scale the quantizer will use."""
        if self.scale is None:
            return Fraction(1)
        cap = self.scale.max
        return min(cap, self.scale_max) if self.scale_max is not None else cap

    def _native(self) -> tuple:
        """Parameters for the native quantizer (computed once)."""
        spec = self.__dict__.get("_native_spec")
        if spec is None:
            e, sc, zp = self.elem, self.scale, self.zero_point
            elem_fp = e if isinstance(e, FPFormat) else None
            ints = lambda f: (f.bits, f.signed, _ROUNDINGS.index(f.rounding), f.frac_bits)
            kind = 0 if sc is None else (1 if isinstance(sc, FPFormat) else 2)
            spec = (elem_fp, None if elem_fp else ints(e), Fraction(self.elem_max),
                    self.block_size, kind, sc if kind == 1 else None,
                    (sc.bits, sc.bias, sc.max_code) if kind == 2 else None,
                    Fraction(self.scale_cap), None if zp is None else ints(zp),
                    self.tensor_scale, self.compute, self.scale_min, self.zero_scale,
                    _to_fraction)
            object.__setattr__(self, "_native_spec", spec)
        return spec

    def _c(self, x: Fraction) -> Fraction:
        """One step of the recipe's intermediate arithmetic."""
        if self.compute is None or x == 0:
            return x
        return FP._round(x, self.compute)[0].exact

    def __str__(self) -> str:
        parts = [f"elem {self.elem}", f"block {self.block_size}"]
        if self.scale is not None:
            s = f"scale {self.scale}"
            if self.scale_max is not None:
                s += f" (max {self.scale_max})"
            parts.append(s)
        if self.zero_point is not None:
            parts.append(f"zero-point {self.zero_point}")
        if self.tensor_scale is not None:
            parts.append(f"tensor scale {self.tensor_scale}")
        if self.compute is not None:
            parts.append(f"computed in {self.compute}")
        return "; ".join(parts)


# NVIDIA NVFP4: E2M1 x16, OCP E4M3 block scale (max 448), FP32 tensor scale.
NVFP4 = BlockFormat(E2M1, 16, E4M3, tensor_scale=FP32)
# The same, following NVIDIA ModelOpt's recipe exactly: float32 intermediate
# arithmetic, block scales clamped to [2**-9, 448], all-zero blocks scale 1.
NVFP4_MODELOPT = BlockFormat(E2M1, 16, E4M3, tensor_scale=FP32, compute=FP32,
                             scale_min=Fraction(1, 512), zero_scale=1)
# OCP Microscaling v1.0: 32-element blocks with an E8M0 shared exponent.
E8M0 = Pow2Format(8, 127)
MXFP4 = BlockFormat(E2M1, 32, E8M0)
MXINT8 = BlockFormat(IntFormat(8, frac_bits=6), 32, E8M0)


# Nested-list helpers (no numpy dependency)
def _shape(values) -> tuple[int, ...]:
    if isinstance(values, BlockTensor):
        return values.shape
    if isinstance(values, (list, tuple)):
        if not values:
            raise ValueError("empty tensor")
        inner = [_shape(v) for v in values]
        if any(s != inner[0] for s in inner):
            raise ValueError("ragged nested lists: all rows must have the same shape")
        return (len(values),) + inner[0]
    return ()


def _as_lists(values):
    """NumPy arrays and other array-likes as nested lists (their numbers
    become Python floats and ints, exactly)."""
    if (not isinstance(values, (list, tuple, BlockTensor, FPArray, FP))
            and hasattr(values, "tolist") and hasattr(values, "shape")):
        return values.tolist()
    return values


def _shape_flat(values) -> tuple[tuple[int, ...], list]:
    """(_shape(values), _flatten(values)) in one native pass."""
    values = _as_lists(values)
    r = _core.shape_flatten(values, BlockTensor)
    return r if r is not None else (_shape(values), _flatten(values))


def _flatten(values) -> list:
    if isinstance(values, (list, tuple)):
        return [x for v in values for x in _flatten(v)]
    return [values]


def _nest(flat: list, shape: tuple[int, ...]):
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    step = len(flat) // shape[0]
    return [_nest(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def _strides(shape):
    out, acc = [], 1
    for n in reversed(shape):
        out.append(acc)
        acc *= n
    return tuple(reversed(out))


def _permute(flat: list, shape: tuple, perm: tuple) -> list:
    """Flat C-order data of ``shape`` transposed by ``perm``."""
    return _core.permute(flat, shape, perm)


def _indices(shape):
    if not shape:
        yield ()
        return
    for i in range(shape[0]):
        for rest in _indices(shape[1:]):
            yield (i,) + rest


def _values_of(values) -> tuple[tuple[int, ...], list]:
    """Shape and flat C-order values (converted with _to_fraction by the
    native quantizer, which raises the same errors)."""
    if isinstance(values, BlockTensor):
        return values.shape, list(values._flat_values())
    shape, flat = _shape_flat(values)
    if not shape:
        raise ValueError("expected a tensor (nested lists), not a scalar")
    return shape, flat


class BlockTensor:
    """An N-D tensor quantized with a BlockFormat along one axis."""

    __slots__ = ("fmt", "shape", "axis", "_elems", "_scales", "_zeros", "_tscale",
                 "_flags", "_source", "_cache")

    # Layout: the tensor is stored as rows running along ``axis``; rows are
    # ordered over the remaining axes in C order.
    @property
    def _perm(self) -> tuple:
        return tuple(i for i in range(len(self.shape)) if i != self.axis) + (self.axis,)

    @staticmethod
    def _norm_axis(axis: int, ndim: int) -> int:
        if not -ndim <= axis < ndim:
            raise ValueError(f"axis {axis} out of range for {ndim}-D tensor")
        return axis % ndim

    def _to_rows(self, flat: list) -> list[list]:
        data = _permute(flat, self.shape, self._perm)
        n = self.shape[self.axis]
        return [data[i:i + n] for i in range(0, len(data), n)]

    def _from_rows(self, rows: list[list]) -> list:
        """Rows back to flat C-order data of self.shape."""
        perm = self._perm
        permuted_shape = tuple(self.shape[p] for p in perm)
        inverse = tuple(perm.index(i) for i in range(len(perm)))
        return _permute([x for r in rows for x in r], permuted_shape, inverse)

    # Construction
    @classmethod
    def quantize(cls, values, fmt: BlockFormat = NVFP4, axis: int = -1) -> "BlockTensor":
        """Quantize nested numbers (or FP/UINT/INT/FIXED values, or another
        BlockTensor's exact values) in blocks along ``axis``."""
        shape, flat = _values_of(values)
        t = object.__new__(cls)
        t.fmt, t.shape, t._cache = fmt, shape, None
        t.axis = cls._norm_axis(axis, len(shape))
        t._source = flat
        t._tscale, t._elems, t._scales, t._zeros, flags = _core.block_quantize(
            flat, shape, t.axis, fmt._native())
        t._flags = FPFlags(flags)
        return t

    @classmethod
    def from_raw(cls, elem_codes, scale_codes=None, fmt: BlockFormat = NVFP4,
                 zero_codes=None, tensor_scale: int | None = None,
                 axis: int = -1) -> "BlockTensor":
        """Build from raw codes, e.g. captured from RTL. ``elem_codes`` is
        nested like the tensor; scale and zero-point codes are nested like it
        too, with the blocking axis holding one entry per block."""
        shape = _shape(elem_codes)
        t = object.__new__(cls)
        t.fmt, t.shape, t._source, t._flags, t._cache = fmt, shape, None, FPFlags(0), None
        t.axis = cls._norm_axis(axis, len(shape))
        n = shape[t.axis]
        nblocks = -(-n // fmt.block_size)
        bshape = shape[:t.axis] + (nblocks,) + shape[t.axis + 1:]
        erows = t._to_rows(_flatten(elem_codes))
        t._elems = [[fmt.elem.from_raw(c) for c in row] for row in erows]

        def block_rows(codes, name):
            if _shape(codes) != bshape:
                raise ValueError(f"{name} codes must have shape {bshape}")
            data = _permute(_flatten(codes), bshape, t._perm)
            return [data[i:i + nblocks] for i in range(0, len(data), nblocks)]

        if fmt.scale is None:
            t._scales = [[None] * nblocks for _ in erows]
        else:
            t._scales = [[fmt.scale.from_raw(c) for c in r]
                         for r in block_rows(scale_codes, "scale")]
        t._zeros = None
        if fmt.zero_point is not None:
            t._zeros = [[fmt.zero_point.from_raw(c) for c in r]
                        for r in block_rows(zero_codes, "zero-point")]
        t._tscale = None
        if fmt.tensor_scale is not None:
            if tensor_scale is None:
                raise ValueError("this format needs a tensor_scale code")
            t._tscale = fmt.tensor_scale.from_raw(tensor_scale)
        return t

    # Raw data (for driving and checking RTL)
    def _nested(self, rows):
        return _nest(self._from_rows(rows), self.shape)

    def _nested_blocks(self, rows):
        n = len(rows[0]) if rows else 0
        bshape = self.shape[:self.axis] + (n,) + self.shape[self.axis + 1:]
        perm = self._perm
        permuted = tuple(bshape[p] for p in perm)
        inverse = tuple(perm.index(i) for i in range(len(perm)))
        return _nest(_permute([x for r in rows for x in r], permuted, inverse), bshape)

    @property
    def elem_raw(self):
        return self._nested([[e.raw for e in row] for row in self._elems])

    @property
    def scale_raw(self):
        return self._nested_blocks([[None if s is None else s.raw for s in row]
                                    for row in self._scales])

    @property
    def zero_raw(self):
        if self._zeros is None:
            return None
        return self._nested_blocks([[z.raw for z in row] for row in self._zeros])

    @property
    def tensor_scale(self) -> FP | None:
        return self._tscale

    @property
    def flags(self) -> FPFlags:
        """OR of the status flags of every element quantization."""
        return self._flags

    # Values (computed once and cached)
    def _scale_value(self, r: int, b: int) -> Fraction:
        s = self._scales[r][b]
        if s is None:
            return Fraction(1)
        if isinstance(self.fmt.scale, Pow2Format):
            return self.fmt.scale.value(s.val)
        return s.exact

    def _flat_values(self) -> list[Fraction]:
        if self._cache is None:
            vals = _core.block_values(self._elems, self._scales, self._zeros, self._tscale,
                                      self.fmt._native())
            if vals is not None:
                perm = self._perm
                inverse = tuple(perm.index(i) for i in range(len(perm)))
                self._cache = _permute(vals, tuple(self.shape[p] for p in perm), inverse)
                return self._cache
            # inf/NaN codes: the reference path raises the right error
            s_t = self._tscale.exact if self._tscale is not None else Fraction(1)
            bs = self.fmt.block_size
            rows = []
            for r, row in enumerate(self._elems):
                out = []
                for c, e in enumerate(row):
                    b = c // bs
                    if isinstance(e, FP):
                        q = e.exact
                    else:
                        z = self._zeros[r][b].val if self._zeros is not None else 0
                        q = (e.val - z) * self.fmt.elem.unit
                    out.append(s_t * self._scale_value(r, b) * q)
                rows.append(out)
            self._cache = self._from_rows(rows)
        return self._cache

    def dequantize(self):
        """Exact values as nested lists of Fraction."""
        return _nest(self._flat_values(), self.shape)

    def to_float(self):
        return _nest([float(x) for x in self._flat_values()], self.shape)

    def error(self) -> Fraction | None:
        """Max |source - dequantized| if built by quantize(), else None."""
        if self._source is None:
            return None
        src = [_to_fraction(x) for x in self._source]
        return max(abs(s - q) for s, q in zip(src, self._flat_values()))

    def __getitem__(self, idx):
        vals = self._flat_values()
        if not isinstance(idx, tuple):
            idx = (idx,)
        if len(idx) > len(self.shape) or not all(isinstance(i, int) for i in idx):
            raise IndexError("index with integers, one per axis at most")
        idx = tuple(i % n if -n <= i < n else _bad_index(i, n)
                    for i, n in zip(idx, self.shape))
        st = _strides(self.shape)
        base = sum(i * s for i, s in zip(idx, st))
        rest = self.shape[len(idx):]
        size = 1
        for n in rest:
            size *= n
        return _nest(vals[base:base + size], rest) if rest else vals[base]

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    # Ops: exact math, then requantize into the left operand's format
    def _check(self, other: "BlockTensor", what: str) -> None:
        if other.fmt != self.fmt:
            _warn(BlockFormatWarning, f"mixing block formats [{self.fmt}] and "
                  f"[{other.fmt}] in {what}, result uses [{self.fmt}] (left)")

    def _elementwise(self, other, fn, what):
        a = self._flat_values()
        if isinstance(other, BlockTensor):
            if other.shape != self.shape:
                raise ValueError(f"shape mismatch: {self.shape} vs {other.shape}")
            self._check(other, what)
            out = [fn(x, y) for x, y in zip(a, other._flat_values())]
        elif isinstance(other, (int, float, Fraction, FP, UINT)) or hasattr(other, "exact"):
            k = _to_fraction(other)
            out = [fn(x, k) for x in a]
        else:
            return NotImplemented
        return BlockTensor.quantize(_nest(out, self.shape), self.fmt, self.axis)

    def __add__(self, o):      return self._elementwise(o, lambda a, b: a + b, "+")
    def __radd__(self, o):     return self._elementwise(o, lambda a, b: b + a, "+")
    def __sub__(self, o):      return self._elementwise(o, lambda a, b: a - b, "-")
    def __rsub__(self, o):     return self._elementwise(o, lambda a, b: b - a, "-")
    def __mul__(self, o):      return self._elementwise(o, lambda a, b: a * b, "*")
    def __rmul__(self, o):     return self._elementwise(o, lambda a, b: b * a, "*")
    def __truediv__(self, o):  return self._elementwise(o, lambda a, b: a / b, "/")
    def __neg__(self):         return self._elementwise(-1, lambda a, b: a * b, "-")

    def __matmul__(self, other: "BlockTensor"):
        if not isinstance(other, BlockTensor):
            return NotImplemented
        self._check(other, "@")
        if self.ndim == 1 and other.ndim == 1:
            return dot(self, other)
        result = matmul(self, other)
        return BlockTensor.quantize(result, self.fmt)

    def transpose(self, *axes: int) -> "BlockTensor":
        """Permute axes (default: reverse them). Blocks move with their axis,
        so this is exact data movement: no requantization."""
        n = self.ndim
        perm = tuple(a % n for a in axes) if axes else tuple(reversed(range(n)))
        if sorted(perm) != list(range(n)):
            raise ValueError(f"{axes} is not a permutation of the axes")
        t = object.__new__(BlockTensor)
        t.fmt, t._tscale, t._flags, t._cache = self.fmt, self._tscale, self._flags, None
        t.shape = tuple(self.shape[p] for p in perm)
        t.axis = perm.index(self.axis)
        # Same blocks, reordered: rows follow the other axes in the new order.
        old_other = [i for i in range(n) if i != self.axis]
        new_other = [p for p in perm if p != self.axis]
        order = [old_other.index(p) for p in new_other]
        lead = tuple(self.shape[i] for i in old_other)
        idx = list(_indices(tuple(lead[i] for i in order)))
        st = _strides(lead)
        src = [sum(ix[order.index(k)] * st[k] for k in range(len(lead))) for ix in idx] \
            if lead else [0]
        t._elems = [self._elems[i] for i in src]
        t._scales = [self._scales[i] for i in src]
        t._zeros = None if self._zeros is None else [self._zeros[i] for i in src]
        t._source = None if self._source is None else \
            _permute(self._source, self.shape, perm)
        return t

    @property
    def T(self) -> "BlockTensor":
        return self.transpose()

    def reblock(self, axis: int = -1, fmt: BlockFormat | None = None) -> "BlockTensor":
        """Requantize with blocks along another axis (or another format).
        Raises CastWarning if that changes any value."""
        t = BlockTensor.quantize(self, fmt or self.fmt, axis)
        if t._flat_values() != self._flat_values():
            _warn(CastWarning, f"reblocking a {self.shape} tensor along axis "
                  f"{t.axis} changed its values")
        return t

    def dot(self, other, acc=None):
        return dot(self, other, acc)

    # Comparison: bit-exact
    def __eq__(self, other) -> bool:
        if not isinstance(other, BlockTensor):
            return NotImplemented
        ts = lambda t: None if t._tscale is None else t._tscale.raw
        return (self.fmt == other.fmt and self.shape == other.shape
                and self.axis == other.axis
                and self.elem_raw == other.elem_raw
                and self.scale_raw == other.scale_raw
                and self.zero_raw == other.zero_raw and ts(self) == ts(other))

    __hash__ = None

    def __repr__(self) -> str:
        ax = "" if self.axis == self.ndim - 1 else f", axis={self.axis}"
        return f"BlockTensor(shape={self.shape}{ax}, [{self.fmt}])"


def _bad_index(i, n):
    raise IndexError(f"index {i} out of range for axis of length {n}")


def _operand(v) -> tuple[tuple[int, ...], list]:
    """(shape, flat values) keeping FP operands as FP (for inf/NaN)."""
    if isinstance(v, BlockTensor):
        return v.shape, list(v._flat_values())
    if isinstance(v, FPArray):
        return v.shape, v._flat()
    shape, flat = _shape_flat(v)
    if _core.plain_numbers(flat):   # used as they are (same exact values)
        return shape, flat
    return shape, [x if isinstance(x, FP) else _to_fraction(x) for x in flat]


def _reduce(a: list, b: list, acc):
    """sum(a[i]*b[i]): exact Fraction, or FP per acc (FPFormat: exact then one
    rounding; Accumulator: that accumulation model)."""
    if acc is None:
        r = _core.exact_dot(a, b)
        if r is not None:
            return r
        if any(isinstance(x, FP) and not x.is_finite for x in a + b):
            raise ValueError("inf/NaN operands need an accumulator format (acc=...)")
        return sum((_to_fraction(x) * _to_fraction(y) for x, y in zip(a, b)), Fraction(0))
    if isinstance(acc, FPFormat):
        acc = Accumulator(acc, order="exact")
    return acc.sum_products(a, b)


def dot(a, b, acc: FPFormat | Accumulator | None = None):
    """Dot product of two vectors: exact (a Fraction), rounded once into an
    FPFormat, or accumulated as an Accumulator models it (an FP)."""
    sa, va = _operand(a)
    sb, vb = _operand(b)
    if len(sa) != 1 or len(sb) != 1:
        raise ValueError(f"expected 1-D vectors, got shapes {sa} and {sb}")
    if sa != sb:
        raise ValueError(f"length mismatch: {sa[0]} vs {sb[0]}")
    return _reduce(va, vb, acc)


def matmul(a, b, acc: FPFormat | Accumulator | None = None,
           out: BlockFormat | None = None, *, transpose_b: bool = False):
    """Matrix product. 1-D operands act as a row/column vector; N-D operands
    are batches of matrices over their leading axes (which must match).

    Each output is exact (Fraction), or rounded per ``acc`` (an FPFormat for
    one rounding, or an Accumulator). With ``out`` the result is quantized
    into a BlockTensor. ``transpose_b=True`` computes a @ b.T from b's exact
    values (for tensor cores where both operands are blocked along K).

    Two FPArray operands with an ``acc`` give an FPArray (same values and
    flags as the list form)."""
    arrays = (isinstance(a, FPArray) and isinstance(b, FPArray)
              and out is None and _acc_spec(acc) is not None)
    if arrays:
        sa, sb = a.shape, b.shape
    else:
        sa, fa = _operand(a)
        sb, fb = _operand(b)
    if transpose_b and len(sb) >= 2:
        perm = tuple(range(len(sb) - 2)) + (len(sb) - 1, len(sb) - 2)
        if arrays:
            b = b.transpose(*perm)
        else:
            fb = _permute(fb, sb, perm)
        sb = tuple(sb[p] for p in perm)
    a_vec, b_vec = len(sa) == 1, len(sb) == 1
    if a_vec:
        sa = (1,) + sa
    if b_vec:
        sb = sb + (1,)
    if sa[-1] != sb[-2]:
        raise ValueError(f"inner dimensions differ: {sa[-1]} vs {sb[-2]}")
    batch_a, batch_b = sa[:-2], sb[:-2]
    if batch_a and batch_b and batch_a != batch_b:
        raise ValueError(f"batch shapes differ: {batch_a} vs {batch_b}")
    batch = batch_a or batch_b
    m, k, n = sa[-2], sa[-1], sb[-1]
    nb = 1
    for d in batch:
        nb *= d
    spec = _acc_spec(acc)
    shape = batch + ((m,) if not a_vec else ()) + ((n,) if not b_vec else ())
    if arrays:
        r = _core.array_matmul(a, b, nb, m, k, n, bool(batch_a), bool(batch_b), spec, shape or (1,))
        return r if shape else r[0]
    res = None
    if acc is None or spec is not None:
        res = _core.matmul_reduce(fa, fb, nb, m, k, n, bool(batch_a), bool(batch_b), spec)
    if res is None:
        res = _matmul_loop(fa, fb, nb, m, k, n, batch_a, batch_b, acc)
    result = _nest(res, shape) if shape else res[0]
    if out is not None:
        return BlockTensor.quantize(result, out)
    return result


def _acc_spec(acc):
    """Accumulator parameters for the native reduction (None: not native)."""
    if isinstance(acc, FPFormat):
        return (acc, "exact", None, 1, None)
    if isinstance(acc, Accumulator):
        return acc._spec()
    return None


def _matmul_loop(fa, fb, nb, m, k, n, batch_a, batch_b, acc):
    res = []
    for bi in range(nb):
        oa = bi * m * k if batch_a else 0
        ob = bi * k * n if batch_b else 0
        for i in range(m):
            row = fa[oa + i * k: oa + (i + 1) * k]
            for j in range(n):
                col = [fb[ob + t * n + j] for t in range(k)]
                res.append(_reduce(row, col, acc))
    return res
