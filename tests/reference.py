"""Independent reference models and constrained random generators."""

from __future__ import annotations

import bisect
import functools
from fractions import Fraction

from customtypes import FP


# Integer reference
def wrap_u(x: int, bits: int) -> int:
    return x & ((1 << bits) - 1)


def wrap_s(x: int, bits: int) -> int:
    x = wrap_u(x, bits)
    return x - (1 << bits) if x >> (bits - 1) else x


def trunc_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def trunc_mod(a: int, b: int) -> int:
    return a - trunc_div(a, b) * b


# Constrained generators: uniform, biased toward boundary values
EDGE_BIAS = 0.25


def rand_uint(rng, bits: int) -> int:
    hi = (1 << bits) - 1
    if rng.random() < EDGE_BIAS:
        return rng.choice([0, 1 & hi, hi, max(hi - 1, 0), 1 << (bits - 1)])
    return rng.randint(0, hi)


def rand_sint(rng, bits: int) -> int:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if rng.random() < EDGE_BIAS:
        return rng.choice([lo, lo + 1, -1, 0, 1 if hi else 0, hi - 1, hi])
    return rng.randint(lo, hi)


def rand_code(rng, E: int, M: int) -> int:
    """Random raw FP code, biased toward zeros, subnormals and max-magnitude."""
    width = 1 + E + M
    if rng.random() < EDGE_BIAS:
        mag = rng.choice([0, 1, (1 << M) - 1, 1 << M, (1 << (E + M)) - 1])
        return (rng.getrandbits(1) << (E + M)) | mag
    return rng.getrandbits(width)


# FP reference: enumerate the non-negative grid, round by nearest-even search
@functools.lru_cache(maxsize=None)
def grid(E: int, M: int) -> list[Fraction]:
    bias = (1 << (E - 1)) - 1
    vals = []
    for code in range(1 << (E + M)):
        e, m = code >> M, code & ((1 << M) - 1)
        if e == 0:
            vals.append(Fraction(m, 1 << M) * Fraction(2) ** (1 - bias))
        else:
            vals.append((1 + Fraction(m, 1 << M)) * Fraction(2) ** (e - bias))
    return vals


def ref_value(code: int, E: int, M: int) -> Fraction:
    v = grid(E, M)[code & ((1 << (E + M)) - 1)]
    return -v if code >> (E + M) else v


def ref_round(x: Fraction, E: int, M: int, zero_sign: bool = False) -> int:
    """Raw code of x rounded to nearest-even, saturating at max magnitude."""
    vals = grid(E, M)
    ax = abs(x)
    if ax >= vals[-1]:
        c = len(vals) - 1
    else:
        i = bisect.bisect_right(vals, ax) - 1
        lo, hi = vals[i], vals[i + 1]
        if ax - lo < hi - ax:
            c = i
        elif ax - lo > hi - ax:
            c = i + 1
        else:
            c = i if i % 2 == 0 else i + 1
    sign = (x < 0) if x != 0 else zero_sign
    return (int(sign) << (E + M)) | c


def check_fp_ops(a: int, b: int, E: int, M: int) -> None:
    fa, fb = FP.from_raw(a, E, M), FP.from_raw(b, E, M)
    va, vb = ref_value(a, E, M), ref_value(b, E, M)
    sa, sb = fa.sign, fb.sign
    ctx = f"a={a:#x} b={b:#x} E{E}M{M}"
    assert (fa + fb).raw == ref_round(va + vb, E, M, sa and sb), ctx
    assert (fa - fb).raw == ref_round(va - vb, E, M, sa and not sb), ctx
    assert (fa * fb).raw == ref_round(va * vb, E, M, sa != sb), ctx
    if vb != 0:
        assert (fa / fb).raw == ref_round(va / vb, E, M, sa != sb), ctx
    assert (fa < fb) == (va < vb), ctx
    assert (fa == fb) == (va == vb), ctx
    assert (fa & fb).raw == a & b, ctx
    assert (fa | fb).raw == a | b, ctx
    assert (fa ^ fb).raw == a ^ b, ctx
