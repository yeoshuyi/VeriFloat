"""Fixed-point (FIXED) and saturating integers.

APyTypes (Linköping University) is the external reference: its APyFixed is a
bit-accurate two's-complement fixed-point type with the same bit-growth rules
and the same cast = quantize-then-overflow order.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from verifloat import (FIXED, FP16, INT, UINT, CastWarning, FixedFormat, FPFlags,
                       FP, IntCastWarning, Rounding)
from reference import ROUNDINGS, caught, ref_int_round

apy = pytest.importorskip("apytypes")
Q = apy.QuantizationMode
APY_Q = {Rounding.RNE: Q.TIES_EVEN, Rounding.RNA: Q.TIES_AWAY, Rounding.RTZ: Q.TO_ZERO,
         Rounding.RDN: Q.TO_NEG, Rounding.RUP: Q.TO_POS}
APY_O = {"wrap": apy.OverflowMode.WRAP, "saturate": apy.OverflowMode.SAT}


def rand_fmt(rng, signed=True, **kw):
    ib = rng.randint(1 if signed else 0, 10)
    fb = rng.randint(-2, 10)
    if ib + fb < 1:
        fb = 1 - ib
    return FixedFormat(ib, fb, signed, **kw)


def rand_fixed(rng, fmt):
    return fmt.from_raw(rng.getrandbits(fmt.bits))


def to_apy(x: FIXED):
    return apy.APyFixed(x.raw, int_bits=x.int_bits, frac_bits=x.frac_bits)


def test_cast_vs_apytypes(rng, iters):
    for _ in range(iters):
        src = rand_fmt(rng)
        dst = rand_fmt(rng, rounding=rng.choice(ROUNDINGS),
                       overflow=rng.choice(["wrap", "saturate"]))
        x = rand_fixed(rng, src)
        got = x.cast(dst)
        want = to_apy(x).cast(int_bits=dst.int_bits, frac_bits=dst.frac_bits,
                              quantization=APY_Q[dst.rounding], overflow=APY_O[dst.overflow])
        assert got.raw == want.to_bits(), (x, dst)
        # flags: INEXACT if bits were dropped, OVERFLOW if out of range
        q = ref_int_round(x.exact / dst.ulp, dst.rounding, -(1 << 99), 1 << 99)
        want_flags = (FPFlags.INEXACT if q * dst.ulp != x.exact else FPFlags(0)) | \
            (FPFlags.OVERFLOW if not dst.min_raw <= q <= dst.max_raw else FPFlags(0))
        assert got.flags == want_flags, (x, dst)


@pytest.mark.parametrize("op", ["+", "-", "*"])
def test_arith_growth_vs_apytypes(rng, iters, op):
    for _ in range(iters):
        a, b = rand_fixed(rng, rand_fmt(rng)), rand_fixed(rng, rand_fmt(rng))
        fn = {"+": lambda p, q: p + q, "-": lambda p, q: p - q, "*": lambda p, q: p * q}[op]
        got, want = fn(a, b), fn(to_apy(a), to_apy(b))
        assert (got.int_bits, got.frac_bits, got.raw) == \
            (want.int_bits, want.frac_bits, want.to_bits()), (a, op, b)
        assert got.exact == fn(a.exact, b.exact)          # growth is exact


@pytest.mark.parametrize("op", ["+", "-", "*"])
def test_saturating_int_vs_apytypes(rng, iters, op):
    fn = {"+": lambda p, q: p + q, "-": lambda p, q: p - q, "*": lambda p, q: p * q}[op]
    for _ in range(iters):
        n = rng.randint(2, 32)
        a, b = rng.randint(-(1 << (n - 1)), (1 << (n - 1)) - 1), \
            rng.randint(-(1 << (n - 1)), (1 << (n - 1)) - 1)
        got = fn(INT(a, n, saturate=True), INT(b, n, saturate=True))
        want = fn(apy.APyFixed.from_float(a, n, 0), apy.APyFixed.from_float(b, n, 0)) \
            .cast(int_bits=n, frac_bits=0, overflow=apy.OverflowMode.SAT)
        assert got.raw == want.to_bits() and got.saturate, (a, op, b, n)


def test_saturating_int_anchors():
    assert (UINT(250, 8, saturate=True) + 10).val == 255
    assert (UINT(3, 8, saturate=True) - 5).val == 0
    assert (INT(-128, 8, saturate=True) - 1).val == -128
    assert (-INT(-128, 8, saturate=True)).val == 127          # -(-128) saturates
    assert (INT(-128, 8, saturate=True) // -1).val == 127
    assert (UINT(0xF0, 8, saturate=True) << 4).val == 0      # bit ops still wrap
    assert INT(300, 8, saturate=True).val == 127            # construction clamps
    assert INT(-300, 8, saturate=True).resize(4).val == -8   # narrowing clamps
    assert repr(UINT(9, 4, saturate=True)) == "UINT(9, u4, sat)"
    with pytest.warns(IntCastWarning, match="mismatched saturate"):
        UINT(1, 8, saturate=True) + UINT(1, 8)


def test_fixed_anchors():
    q = FixedFormat(4, 4)                    # signed Q3.4: -8 .. 7.9375
    x = q(1.3)
    assert (x.raw, x.exact, x.flags) == (21, Fraction(21, 16), FPFlags.INEXACT)
    assert (q.min, q.max, q.bits) == (-8, Fraction(127, 16), 8)
    assert (q(100).raw, q(100).flags) == (0x40, FPFlags.OVERFLOW)   # 1600 wraps to 64
    assert q.replace(overflow="saturate")(100).exact == q.max
    assert FixedFormat(4, 4, rounding=Rounding.RDN)(-0.01).exact == Fraction(-1, 16)
    y = x * x
    assert (y.int_bits, y.frac_bits, y.exact) == (8, 8, Fraction(441, 256))
    assert (x << 2).exact == x.exact * 4 and (x >> 1).exact == x.exact / 2
    assert x.div(3, FixedFormat(2, 8)).exact == Fraction(112, 256)   # 1.3125/3 -> 0.4375
    assert str(q) == "sfix4.4" and repr(x) == "FIXED(1.3125, sfix4.4, flags=INEXACT)"
    assert float(FP.from_value(x, FP16)) == 1.3125 and q(FP16(2.5)).exact == Fraction(5, 2)
    with pytest.warns(CastWarning, match="lossy"):
        x + 0.01
    u = FixedFormat(4, 2, signed=False)
    with pytest.warns(IntCastWarning, match="mixing signedness"):
        r = u(3) - q(1)
    assert r.signed and r.exact == 2
