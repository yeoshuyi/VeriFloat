"""Floating point against APyTypes (Linköping University).

APyFloat is a bit-accurate model of IEEE-754-style formats of any exponent
and mantissa width and any bias, with the five IEEE rounding directions. It
was written independently of VeriFloat, so it checks the C++ core's rounding
kernel on formats outside the IEEE/OCP set.

Where the two disagree, the case is settled by the exact references: the
frozen pure-Python 0.1 (Fraction arithmetic), and for formats small enough to
enumerate, the grid reference model in tests/reference.py. Only these known
APyTypes 0.5 behaviours may differ:

* ``APyFloat.cast`` into a format's subnormal range: when a result carries up
  from the subnormals it can come out wrong (e.g. 1.5 * 2**-21 cast with ties
  to even into e6m17/bias 5 gives 2**-4 instead of 2**-20).
* ``APyFloat.from_float`` always rounds to nearest-even (the quantization
  context does not apply), so directed modes are checked through ``cast`` from
  binary64.
* ``!=`` with a NaN operand is False in APyFloat; IEEE 754 (and Python floats)
  say True, so comparisons involving NaN are checked against IEEE directly.
"""

from __future__ import annotations

import operator
import os
import struct

import pytest

apy = pytest.importorskip("apytypes")

from conftest import COMPARED  # noqa: E402
from verifloat import FPFormat, Rounding  # noqa: E402
from reference import Fmt, ref_round  # noqa: E402

try:
    import verifloat_py as py
except ImportError:                     # the arbiter is optional
    py = None

Q = apy.QuantizationMode
APY_MODE = {Rounding.RNE: Q.TIES_EVEN, Rounding.RNA: Q.TIES_AWAY, Rounding.RTZ: Q.TO_ZERO,
            Rounding.RUP: Q.TO_POS, Rounding.RDN: Q.TO_NEG}
OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


def rand_format(rng, max_e=11, max_m=30):
    """IEEE-style format (APyFloat always has inf/NaN), random widths and bias."""
    E, M = rng.randint(2, max_e), rng.randint(1, max_m)
    bias = (1 << (E - 1)) - 1 if rng.random() < 0.6 else rng.randint(0, (1 << E) - 2)
    return E, M, bias


def rand_code(rng, E, M, bias):
    """Biased encodings: specials, subnormals, extremes, values near 1, exponents
    close together (cancellation, rounding at every position)."""
    top = (1 << E) - 1
    sign = rng.getrandbits(1) << (E + M)
    r = rng.random()
    if r < 0.15:
        return sign | rng.choice([0, 1, (1 << M) - 1, 1 << M, (top << M) - 1, top << M,
                                  (top << M) | 1, (top << M) | (1 << (M - 1)), (1 << (M + 1)) - 1])
    if r < 0.45:
        e = min(max(bias + rng.randint(-3, 3), 0), top)
        return sign | (e << M) | rng.getrandbits(M)
    if r < 0.6:
        return sign | (rng.randint(0, 2) << M) | rng.getrandbits(M)
    if r < 0.7:
        return sign | (rng.randint(top - 2, top) << M) | rng.getrandbits(M)
    return rng.getrandbits(1 + E + M)


def ours(E, M, bias, rounding=Rounding.RNE) -> FPFormat:
    return FPFormat(E, M, bias, rounding=rounding)


def same(want, got) -> bool:
    COMPARED["APyTypes (result)"] += 1
    return (want.is_nan and got.is_nan) or want.to_bits() == got.raw


def arbiter(x, E, M, B, rounding) -> int | None:
    """Raw code of the exact finite value x rounded into (E, M, B): from the
    pure-Python 0.1, cross-checked with the grid model when it is small."""
    ref = None
    if E + M <= 14:
        ref = ref_round(x.exact, Fmt.make(E, M, B, rounding=rounding), x.sign).raw
    if py is not None:
        r = py.FPFormat(E, M, B, rounding=py.Rounding(rounding.value))(x.exact)
        if x.exact == 0:
            r = -r if x.sign else r
        assert ref is None or ref == r.raw
        ref = r.raw
    return ref


@pytest.mark.parametrize("rounding", APY_MODE, ids=lambda r: r.name)
@pytest.mark.parametrize("op", OPS)
def test_arith_vs_apyfloat(rng, iters, op, rounding):
    for _ in range(iters):
        E, M, bias = rand_format(rng)
        ca, cb = rand_code(rng, E, M, bias), rand_code(rng, E, M, bias)
        a = apy.APyFloat.from_bits(ca, E, M, bias)
        b = apy.APyFloat.from_bits(cb, E, M, bias)
        with apy.APyFloatQuantizationContext(APY_MODE[rounding]):
            want = OPS[op](a, b)
        assert (want.exp_bits, want.man_bits, want.bias) == (E, M, bias)
        f = ours(E, M, bias, rounding)
        got = OPS[op](f.from_raw(ca), f.from_raw(cb))
        assert same(want, got), (E, M, bias, hex(ca), op, hex(cb), want, got)


DESIGNED = [(5, 10), (4, 3), (3, 4), (8, 23), (6, 17), (11, 52)]


@pytest.mark.parametrize("EM", DESIGNED, ids=lambda em: f"e{em[0]}m{em[1]}")
@pytest.mark.parametrize("rounding", APY_MODE, ids=lambda r: r.name)
@pytest.mark.parametrize("op", OPS)
def test_designed_cases_vs_apyfloat(iters, op, rounding, EM):
    """The designed cases of tests/directed.py: results on and next to
    rounding ties, the overflow threshold and the subnormal range, and every
    pair of special values, in standard and non-standard widths."""
    import directed as D
    E, M = EM
    f = ours(E, M, None, rounding)
    bias = f.bias
    pairs = {"+": D.add_cases, "-": D.add_cases, "*": D.mul_cases, "/": D.div_cases}[op](f)
    if iters < 20000:                       # an evenly spread part unless asked for everything
        pairs = pairs[::max(len(pairs) // (4 * max(iters, 200)), 1)]
    edges = D.edge_codes(f)
    pairs = pairs + [(a, b) for a in edges for b in edges]
    bad = []
    with apy.APyFloatQuantizationContext(APY_MODE[rounding]):
        for ca, cb in pairs:
            want = OPS[op](apy.APyFloat.from_bits(ca, E, M, bias), apy.APyFloat.from_bits(cb, E, M, bias))
            got = OPS[op](f.from_raw(ca), f.from_raw(cb))
            if not same(want, got):
                bad.append((hex(ca), hex(cb), want, got))
    assert not bad, (E, M, op, rounding, len(bad), len(pairs), bad[:5])


@pytest.mark.parametrize("rounding", APY_MODE, ids=lambda r: r.name)
def test_cast_vs_apyfloat(rng, iters, rounding):
    diverged = 0
    for _ in range(iters):
        (E, M, B), (E2, M2, B2) = rand_format(rng), rand_format(rng)
        c = rand_code(rng, E, M, B)
        want = apy.APyFloat.from_bits(c, E, M, B).cast(E2, M2, B2, APY_MODE[rounding])
        x = ours(E, M, B).from_raw(c)
        got = x.convert(ours(E2, M2, B2, rounding))
        if same(want, got):
            continue
        # Disagreement: the independent reference model decides, and only a
        # result in the subnormal range may differ (APyTypes cast bug).
        assert x.is_finite, (E, M, B, hex(c), want, got)
        ctx = (E, M, B, hex(c), E2, M2, B2, want, got)
        ref = arbiter(x, E2, M2, B2, rounding)
        assert ref is not None and got.raw == ref, ctx
        assert got.exp.val <= 1, ctx      # subnormal, or carried into the lowest binade
        diverged += 1
    assert diverged <= max(iters // 200, 3)


def rand_double(rng) -> float:
    r = rng.random()
    if r < 0.6:
        return struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
    return rng.uniform(-1, 1) * 2.0 ** rng.randint(-40, 40)


@pytest.mark.parametrize("rounding", APY_MODE, ids=lambda r: r.name)
def test_from_float_vs_apyfloat(rng, iters, rounding):
    for _ in range(iters):
        E, M, B = rand_format(rng)
        d = rand_double(rng)
        if rounding is Rounding.RNE:
            want = apy.APyFloat.from_float(d, E, M, B)
        else:
            want = apy.APyFloat.from_float(d, 11, 52).cast(E, M, B, APY_MODE[rounding])
        got = ours(E, M, B, rounding)(d)
        if same(want, got):
            continue
        ref = arbiter(ours(11, 52, 1023)(d), E, M, B, rounding)
        assert ref is not None and got.raw == ref, (d, E, M, B, want, got)
        assert got.exp.val <= 1, (d, E, M, B, want, got)


def test_queries_vs_apyfloat(rng, iters):
    for _ in range(iters):
        E, M, B = rand_format(rng)
        c = rand_code(rng, E, M, B)
        a = apy.APyFloat.from_bits(c, E, M, B)
        x = ours(E, M, B).from_raw(c)
        assert (a.is_nan, a.is_inf, a.is_finite, a.is_zero, a.is_subnormal) == \
            (x.is_nan, x.is_inf, x.is_finite, x.is_zero, x.is_subnormal), (E, M, B, hex(c))
        if x.is_finite:
            assert a.to_fraction() == x.exact, (E, M, B, hex(c))
        c2 = rand_code(rng, E, M, B)
        b, y = apy.APyFloat.from_bits(c2, E, M, B), ours(E, M, B).from_raw(c2)
        for op in (operator.lt, operator.le, operator.eq, operator.ne, operator.gt, operator.ge):
            if x.is_nan or y.is_nan:   # IEEE: unordered, so only != holds
                assert op(x, y) == (op is operator.ne), (E, M, B, hex(c), hex(c2), op)
            else:
                assert op(a, b) == op(x, y), (E, M, B, hex(c), hex(c2), op)


# ---------------------------------------------------------------- arrays

needs_arrays = pytest.mark.skipif(os.environ.get("VERIFLOAT_IMPL", "cpp") != "cpp",
                                  reason="FPArray is part of the C++ package")

def rand_codes(rng, E, M, bias, shape):
    if len(shape) == 1:
        return [rand_code(rng, E, M, bias) for _ in range(shape[0])]
    return [rand_codes(rng, E, M, bias, shape[1:]) for _ in range(shape[0])]


def same_array(want, got: "FPArray") -> bool:
    w = want.to_bits()
    flat = lambda x: [v for r in x for v in flat(r)] if isinstance(x, list) else [x]  # noqa: E731
    f = got.format
    for a, b in zip(flat(w), flat(got.raw), strict=True):
        if a != b and not (f.from_raw(a).is_nan and f.from_raw(b).is_nan):
            return False
    return True


@needs_arrays
@pytest.mark.parametrize("rounding", APY_MODE, ids=lambda r: r.name)
@pytest.mark.parametrize("op", OPS)
def test_fparray_vs_apyfloatarray(rng, iters, op, rounding):
    """FPArray element-wise arithmetic against APyFloatArray."""
    from verifloat import FPArray
    for _ in range(max(iters // 40, 25)):
        E, M, bias = rand_format(rng)
        shape = rng.choice([(40,), (5, 8), (2, 3, 4)])
        ca, cb = rand_codes(rng, E, M, bias, shape), rand_codes(rng, E, M, bias, shape)
        a = apy.APyFloatArray.from_bits(ca, E, M, bias)
        b = apy.APyFloatArray.from_bits(cb, E, M, bias)
        with apy.APyFloatQuantizationContext(APY_MODE[rounding]):
            want = OPS[op](a, b)
        f = ours(E, M, bias, rounding)
        got = OPS[op](FPArray.from_raw(ca, f), FPArray.from_raw(cb, f))
        assert got.shape == tuple(want.shape)
        assert same_array(want, got), (E, M, bias, op, ca, cb)


@needs_arrays
@pytest.mark.parametrize("rounding", APY_MODE, ids=lambda r: r.name)
def test_fparray_quantize_and_cast_vs_apyfloatarray(rng, iters, rounding):
    from verifloat import FPArray
    for _ in range(max(iters // 40, 25)):
        E, M, B = rand_format(rng)
        vals = [rand_double(rng) for _ in range(40)]
        vals = [v for v in vals if v == v]          # APyTypes keeps NaN payloads its own way
        f = ours(E, M, B, rounding)
        src = apy.APyFloatArray.from_float(vals, 11, 52)
        want = src.cast(E, M, B, APY_MODE[rounding])
        got = f.array(vals)
        got2 = FPArray(vals, ours(11, 52, 1023)).convert(f)
        assert got == got2
        w = want.to_bits()
        for i, v in enumerate(vals):
            if w[i] == got.raw[i]:
                continue
            # APyTypes' cast into the subnormal range (see the module docstring)
            ref = arbiter(ours(11, 52, 1023)(v), E, M, B, rounding)
            assert ref is not None and got.raw[i] == ref and got[i].exp.val <= 1, (v, E, M, B, rounding)


@needs_arrays
def test_matmul_vs_apyfloatarray(rng, iters):
    """APyFloatArray's matmul rounds each product, then adds them in order:
    Accumulator(fmt, "sequential", product=fmt). The two libraries treat a sum
    that overflows part way differently, so those outputs only need to be
    non-finite in both."""
    from verifloat import Accumulator, FPArray, matmul
    checked = 0
    for _ in range(max(iters // 20, 60)):
        E, M = rng.randint(3, 11), rng.randint(2, 30)
        f = FPFormat(E, M)
        m, k, n = rng.randint(1, 5), rng.randint(1, 12), rng.randint(1, 5)
        near_one = lambda: ((rng.getrandbits(1) << (E + M))  # noqa: E731
                            | (min(max(f.bias + rng.randint(-3, 3), 0), f._top - 1) << M) | rng.getrandbits(M))
        ca = [[near_one() for _ in range(k)] for _ in range(m)]
        cb = [[near_one() for _ in range(n)] for _ in range(k)]
        want = (apy.APyFloatArray.from_bits(ca, E, M) @ apy.APyFloatArray.from_bits(cb, E, M)).to_bits()
        got = matmul(FPArray.from_raw(ca, f), FPArray.from_raw(cb, f), acc=Accumulator(f, "sequential", product=f))
        for i in range(m):
            for j in range(n):
                x, y = f.from_raw(want[i][j]), got[i, j]
                if x.is_finite and y.is_finite:
                    assert x.raw == y.raw, (E, M, ca, cb, i, j)
                    checked += 1
                else:
                    assert not x.is_finite and not y.is_finite, (E, M, ca, cb, i, j)
    assert checked > 100
