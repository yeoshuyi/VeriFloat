"""Models of how hardware accumulates a dot product.

An exact dot product rounded once (``Accumulator(fmt, order="exact")``) is the
ideal. Real datapaths round as they go, so ``Accumulator`` also models:

* ``order="sequential"``: s = round(s + g) for each group g of products,
  in index order (an FMA chain when products are exact and group=1);
* ``order="pairwise"``: a balanced tree, rounding after every addition;
* ``product``: round each product into this format before adding;
* ``group`` and ``align_bits``: sum ``group`` products at a time in a fixed-
  point adder that aligns them to the largest exponent in the group and keeps
  ``align_bits`` bits below it (the rest is truncated toward zero), as many
  tensor-core and MX dot-product units do.

Every rounding uses the accumulator format's own rounding mode; the result is
an FP whose ``flags`` are the OR of all the roundings (like a sticky fflags).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .fp import FP, FPFlags, FPFormat, _floor_log2

_F = FPFlags


@dataclass(frozen=True)
class Accumulator:
    fmt: FPFormat
    order: str = "sequential"
    product: FPFormat | None = None
    group: int = 1
    align_bits: int | None = None

    def __post_init__(self):
        if self.order not in ("exact", "sequential", "pairwise"):
            raise ValueError("order must be 'exact', 'sequential' or 'pairwise'")
        if not isinstance(self.group, int) or self.group < 1:
            raise ValueError("group must be a positive integer")
        if self.align_bits is not None and self.align_bits < 0:
            raise ValueError("align_bits must be >= 0")

    def __str__(self) -> str:
        parts = [f"{self.order} in {self.fmt}"]
        if self.product is not None:
            parts.append(f"products in {self.product}")
        if self.group > 1:
            parts.append(f"groups of {self.group}")
        if self.align_bits is not None:
            parts.append(f"{self.align_bits} bits below the group max")
        return ", ".join(parts)

    def sum_products(self, a, b, init=None) -> FP:
        """sum(a[i] * b[i]) (+ init) as this accumulator computes it.
        Operands may be FP values or exact numbers."""
        terms = [_mul(x, y, self.product) for x, y in zip(a, b, strict=True)]
        return self._sum(terms, init)

    def _sum(self, terms: list, init) -> FP:
        fmt = self.fmt
        flags = _F(0)
        for t in terms:
            if isinstance(t, FP):
                flags |= t.flags
        special = _special_sum(terms + ([init] if isinstance(init, FP) else []), fmt)
        if special is not None:
            r, f = special
            return FP._finish(r, f | flags, None, None, fmt)
        vals = [_signed(t) for t in terms]
        adds = []
        for i in range(0, len(vals), self.group):
            (v, sign), inexact = _aligned_sum(vals[i:i + self.group], self.align_bits, fmt)
            flags |= _F.INEXACT if inexact else _F(0)
            adds.append((v, sign))
        start = _signed(init) if init is not None else None
        if self.order == "exact":
            total = (Fraction(0), False) if start is None else start
            for t in adds:
                total = _add(total, t, fmt)
            r = FP._encode(total[0], fmt, total[1])
            return FP._finish(r, r.flags | flags, None, total[0], fmt)
        if self.order == "sequential":
            v0, s0 = start or (Fraction(0), False)
            acc = FP._encode(v0, fmt, s0)
            flags |= acc.flags
            for t in adds:
                if not acc.is_finite:      # inf + finite stays inf (IEEE)
                    break
                v, sign = _add((acc.exact, acc.sign), t, fmt)
                acc = FP._encode(v, fmt, sign)
                flags |= acc.flags
            return FP._finish(acc, flags, None, None, fmt)
        # pairwise tree
        level = [FP._encode(v, fmt, sign) for v, sign in adds]
        if start is not None:
            level.insert(0, FP._encode(start[0], fmt, start[1]))
        if not level:
            level = [FP._encode(Fraction(0), fmt)]
        for v in level:
            flags |= v.flags
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level) - 1, 2):
                x, y = level[i], level[i + 1]
                if x.is_finite and y.is_finite:
                    v, sign = _add((x.exact, x.sign), (y.exact, y.sign), fmt)
                    r = FP._encode(v, fmt, sign)
                else:                      # an overflowed node: IEEE rules
                    r = FP._arith("+", x, y, fmt)
                flags |= r.flags
                nxt.append(r)
            if len(level) % 2:
                nxt.append(level[-1])
            level = nxt
        return FP._finish(level[0], flags, None, None, fmt)


def _signed(t) -> tuple[Fraction, bool]:
    """(value, sign); the sign matters only for zeros (-0 vs +0)."""
    if isinstance(t, FP):
        return t.exact, t.sign
    if isinstance(t, tuple):
        return t
    v = Fraction(t)
    return v, v < 0


def _add(a: tuple, b: tuple, fmt: FPFormat) -> tuple[Fraction, bool]:
    """Exact sum with the IEEE sign for zero results."""
    v = a[0] + b[0]
    return v, (v < 0 if v else FP._sum_zero_sign(a[1], b[1], fmt))


def _mul(x, y, product: FPFormat | None):
    if isinstance(x, FP) and isinstance(y, FP) and not (x.is_finite and y.is_finite):
        return x * y if product is None else (x * y).convert(product)
    xv, xs = _signed(x)
    yv, ys = _signed(y)
    if product is None:
        return xv * yv, xs != ys
    return FP._encode(xv * yv, product, xs != ys)


def _special_sum(terms: list, fmt: FPFormat):
    """IEEE result when any term is inf or NaN: (FP, flags), else None."""
    specials = [t for t in terms if isinstance(t, FP) and not t.is_finite]
    if not specials:
        return None
    nans = [t for t in specials if t.is_nan]
    if nans:
        return FP._nan_result(fmt, *nans)
    signs = {t.sign for t in specials}
    if len(signs) == 2:                      # +inf + -inf
        return FP._nan_result(fmt, invalid=True)
    r, f, _ = FP._inf_result(signs.pop(), fmt)
    return r, f


def _aligned_sum(values: list[tuple], align_bits: int | None, fmt: FPFormat):
    """Exact (value, sign) sum of a group, or the sum after truncating each
    value toward zero to ``align_bits`` bits below the group's largest
    exponent. Returns ((value, sign), inexact)."""
    inexact = False
    if align_bits is not None:
        nonzero = [v for v, _ in values if v]
        if nonzero:
            top = max(_floor_log2(abs(v).numerator, abs(v).denominator) for v in nonzero)
            step = Fraction(2) ** (top - align_bits)
            cut = []
            for v, sign in values:
                q = math.trunc(v / step) * step
                inexact |= q != v
                cut.append((q, sign))
            values = cut
    total = values[0] if values else (Fraction(0), False)
    for t in values[1:]:
        total = _add(total, t, fmt)
    return total, inexact
