"""Accumulation models for dot products.

External references: the C library's fma()/fmaf() (correctly rounded fused
multiply-add) for FMA chains, and numpy float32 arithmetic (correctly rounded
per operation) for separately rounded products and pairwise trees. The
aligned-group model has no external reference and is checked against a direct
transcription of its definition.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import struct
import sys
from fractions import Fraction

import pytest

from verifloat import (FP16, FP32, FP64, NVFP4, Accumulator, BlockTensor, FPFlags,
                       Rounding, dot)
from test_external import rand_bits

np = pytest.importorskip("numpy")

# The C library's fma and fmaf are the reference for the FMA-chain model. They
# live in libm on POSIX systems and in the Universal C Runtime on Windows.
_libm_name = ctypes.util.find_library("m") or ("ucrtbase" if sys.platform == "win32" else None)
if _libm_name is None:
    pytest.skip("no C math library to take fma from", allow_module_level=True)
libm = ctypes.CDLL(_libm_name)
libm.fmaf.restype = ctypes.c_float
libm.fmaf.argtypes = [ctypes.c_float] * 3
libm.fma.restype = ctypes.c_double
libm.fma.argtypes = [ctypes.c_double] * 3


def finite_vector(rng, fmt, width, n):
    out = []
    while len(out) < n:
        x = fmt.from_raw(rand_bits(rng, width, fmt))
        if x.is_finite:
            out.append(x)
    return out


@pytest.mark.parametrize("fmt, width, fma", [(FP32, 32, "fmaf"), (FP64, 64, "fma")])
def test_fma_chain_vs_libm(rng, iters, fmt, width, fma):
    fn = getattr(libm, fma)
    acc = Accumulator(fmt, "sequential")
    for _ in range(max(iters // 10, 100)):
        n = rng.randint(1, 12)
        a, b = finite_vector(rng, fmt, width, n), finite_vector(rng, fmt, width, n)
        s = 0.0
        for x, y in zip(a, b):
            s = fn(float(x), float(y), s)
        got = acc.sum_products(a, b)
        assert float(got) == s or (math.isnan(s) and got.is_nan), (a, b)
        if s == 0:
            assert math.copysign(1, s) == math.copysign(1, float(got))


def test_rounded_products_vs_numpy(rng, iters):
    acc = Accumulator(FP32, "sequential", product=FP32)
    tree = Accumulator(FP32, "pairwise", product=FP32)
    for _ in range(max(iters // 10, 100)):
        n = rng.randint(1, 12)
        a, b = finite_vector(rng, FP32, 32, n), finite_vector(rng, FP32, 32, n)
        with np.errstate(all="ignore"):
            p = [np.float32(float(x)) * np.float32(float(y)) for x, y in zip(a, b)]
            s = np.float32(0)
            for v in p:
                s = np.float32(s + v)
            level = list(p)
            while len(level) > 1:
                level = [np.float32(level[i] + level[i + 1]) for i in range(0, len(level) - 1, 2)] \
                    + ([level[-1]] if len(level) % 2 else [])
        seq, par = acc.sum_products(a, b), tree.sum_products(a, b)
        assert float(seq) == float(s) or (seq.is_nan and math.isnan(s)), (a, b)
        assert float(par) == float(level[0]) or (par.is_nan and math.isnan(level[0])), (a, b)


def ref_aligned(vals, group, align_bits, fmt):
    """Direct transcription: truncate each group's terms toward zero to
    align_bits bits below the group's largest exponent, sum exactly, then add
    to the running sum with one rounding."""
    s = FP32.replace(rounding=fmt.rounding)(0) if fmt is FP32 else fmt(0)
    for i in range(0, len(vals), group):
        g = vals[i:i + group]
        top = max((math.floor(math.log2(abs(float(v)))) for v in g if v), default=None)
        if top is not None:
            # exact floor(log2) for Fractions
            top = max(_flog2(abs(v)) for v in g if v)
            step = Fraction(2) ** (top - align_bits)
            g = [Fraction(math.trunc(v / step)) * step for v in g]
        if not s.is_finite:
            break                      # inf + finite stays inf
        s = fmt(s.exact + sum(g, Fraction(0)))
    return s


def _flog2(x: Fraction) -> int:
    e = x.numerator.bit_length() - x.denominator.bit_length()
    return e if x >= Fraction(2) ** e else e - 1


def test_aligned_groups(rng, iters):
    for _ in range(max(iters // 10, 100)):
        fmt = FP16.replace(rounding=rng.choice([Rounding.RNE, Rounding.RTZ]))
        group, bits = rng.randint(1, 8), rng.randint(0, 12)
        n = rng.randint(1, 20)
        a = [Fraction(rng.randint(-2000, 2000), 1 << rng.randint(0, 10)) for _ in range(n)]
        b = [Fraction(rng.randint(-2000, 2000), 1 << rng.randint(0, 10)) for _ in range(n)]
        got = Accumulator(fmt, "sequential", group=group, align_bits=bits).sum_products(a, b)
        want = ref_aligned([x * y for x, y in zip(a, b)], group, bits, fmt)
        assert got.raw == want.raw, (a, b, group, bits)


def test_accumulator_anchors():
    big = [FP32(1e8), FP32(1), FP32(-1e8)]
    ones = [FP32(1)] * 3
    assert float(Accumulator(FP32, "exact").sum_products(big, ones)) == 1.0
    seq = Accumulator(FP32, "sequential").sum_products(big, ones)
    assert float(seq) == 0.0 and seq.flags == FPFlags.INEXACT
    inf = FP32(float("inf"))
    r = Accumulator(FP32).sum_products([inf, -inf], [FP32(1), FP32(1)])
    assert r.is_nan and r.flags & FPFlags.INVALID
    # dot()/matmul() take an Accumulator where they took an FPFormat
    t = BlockTensor.quantize([1.5, -2, 3, 0.25], NVFP4)
    assert dot(t, t, Accumulator(FP32, "exact")).raw == dot(t, t, FP32).raw
    assert str(Accumulator(FP32, "sequential", FP16, group=4, align_bits=24)) == \
        "sequential in e8m23, products in e5m10, groups of 4, 24 bits below the group max"
