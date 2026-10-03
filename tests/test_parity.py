"""Differential tests: the C++-backed ``verifloat`` against the frozen
pure-Python ``verifloat_py`` (archive/python, git tag python-v0.1.0).

Every case is plain data (format fields, raw codes, Python numbers) that is
built separately in each implementation, and the two must agree exactly on:
results (raw codes, values, types), status flags, unrounded values, reprs,
hashes, exceptions (type and message), warnings (category and message), and
the number of stochastic-rounding draws consumed.
"""

from __future__ import annotations

import copy
import enum
import os
import pickle
import sys
import warnings
from decimal import Decimal
from fractions import Fraction

import pytest

import verifloat as V
import verifloat.bus  # noqa: F401  (m.bus below)
from reference import ROUNDINGS, rand_code, rand_fmt

pytestmark = pytest.mark.skipif(os.environ.get("VERIFLOAT_IMPL", "cpp") != "cpp",
                                reason="compares the C++ package against verifloat_py")
P = pytest.importorskip("verifloat_py")
import verifloat_py.bus  # noqa: E402,F401  (m.bus below)

ALL_ROUNDINGS = ["rne", "rna", "rtz", "rup", "rdn", "sr"]
NAN_MODES = ["canonical", "propagate", "x86", "arm"]


# ---------------------------------------------------------------- harness

def snap(x):
    """Implementation-independent picture of a result."""
    name = type(x).__name__
    if name == "FP":
        return ("FP", str(x.format), x.raw, int(x.flags), x.unrounded, repr(x), str(x),
                None if x.is_nan else hash(x))
    if name in ("UINT", "INT"):
        return (name, x.val, x.bits, x.saturate, repr(x), hash(x))
    if name == "FIXED":
        return ("FIXED", str(x.format), x.val, int(x.flags), repr(x))
    if name == "BlockTensor":
        return ("BT", repr(x), x.shape, x.axis, x.elem_raw, x.scale_raw, x.zero_raw,
                snap(x.tensor_scale), int(x.flags), x.dequantize(), x.error())
    if name in ("FPFormat", "FixedFormat", "BlockFormat", "IntFormat", "Pow2Format", "Accumulator"):
        return (name, str(x), repr(x))
    if name == "BlockBuses":
        return snap(tuple(x.__dict__.values()))
    if isinstance(x, enum.Enum):
        return (type(x).__name__, x.value if not isinstance(x, enum.Flag) else int(x))
    if isinstance(x, float):
        return ("float", repr(x))
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, tuple(snap(v) for v in x))
    if isinstance(x, dict):
        return ("dict", tuple((k, snap(v)) for k, v in x.items()))
    if x is None or isinstance(x, (bool, int, str, Fraction)):
        return x
    raise TypeError(f"cannot snapshot {type(x)}")


SR_SEED = 1234


def outcome(m, fn):
    """(result or exception, warnings, next SR draw) of fn(m)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.set_sr_source(SR_SEED)
        try:
            r = snap(fn(m))
        except Exception as e:  # noqa: BLE001 - exceptions are compared too
            r = ("raise", type(e).__name__, str(e))
        drawn = m.fp._sr_state["source"](24)
    return r, [(c.category.__name__, str(c.message)) for c in w], drawn


def outcomes(fn):
    """(C++ outcome, 0.1 outcome), to be equal.

    One case is put on equal terms first. 0.1 writes the exact value into an
    overflow or underflow warning; when that value has more digits than
    Python converts to text (4300: any overflow of binary128), the conversion
    fails and the operation raises ValueError instead of returning its
    result. The C++ package writes an approximation and returns the result.
    There, 0.1 is rerun with the digit limit lifted: result, flags, random
    draws and warning categories must still match; the warning texts differ.
    """
    got, want = outcome(V, fn), outcome(P, fn)
    r = want[0]
    if got != want and isinstance(r, tuple) and r[:2] == ("raise", "ValueError") and "Exceeds the limit" in r[2]:
        limit = sys.get_int_max_str_digits()
        sys.set_int_max_str_digits(0)
        try:
            want = outcome(P, fn)
        finally:
            sys.set_int_max_str_digits(limit)
        got, want = ((o[0], [c for c, _ in o[1]], o[2]) for o in (got, want))
    return got, want


def check(fn, what=""):
    got, want = outcomes(fn)
    assert got == want, what


# ---------------------------------------------------------------- formats

def spec(E, M, **kw):
    """Plain-data format description."""
    return (E, M, tuple(sorted(kw.items())))


def spec_of(f) -> tuple:   # reference.Fmt -> spec
    return spec(f.E, f.M, bias=f.bias, signed=f.signed, wrap=f.wrap, ftz=f.ftz,
                rounding=f.rounding.value, inf_nan=f.inf_nan, saturate=f.saturate,
                nan_mode=f.nan_mode.value, has_zero=f.has_zero, sr_bits=f.sr_bits,
                tininess=f.tininess)


def fmt(m, s):
    E, M, kw = s
    kw = dict(kw)
    if "rounding" in kw:
        kw["rounding"] = m.Rounding(kw["rounding"])
    if "nan_mode" in kw:
        kw["nan_mode"] = m.NaNMode(kw["nan_mode"])
    return m.FPFormat(E, M, **kw)


def with_modes(s, **kw):
    E, M, old = s
    d = dict(old)
    d.update(kw)
    return (E, M, tuple(sorted(d.items())))


SMALL = [spec(2, 1, inf_nan=False), spec(2, 3, inf_nan=False), spec(3, 2, inf_nan=False),
         spec(2, 1), spec(3, 1, inf_nan="fn"), spec(2, 1, signed=False, inf_nan=False),
         spec(2, 0, inf_nan="fn"), spec(3, 0, signed=False, inf_nan="fn", has_zero=False),
         spec(2, 2, bias=0, inf_nan=False), spec(3, 1, bias=-2)]
BYTE = [spec(4, 3, inf_nan="fn"), spec(5, 2), spec(4, 3, signed=False, inf_nan="fn"),
        spec(8, 0, signed=False, inf_nan="fn", has_zero=False), spec(4, 3), spec(3, 4)]
IEEE = [spec(5, 10), spec(8, 7), spec(8, 23), spec(11, 52)]
WIDE = [spec(15, 112), spec(13, 70), spec(8, 23, rounding="sr", sr_bits=60),
        spec(11, 52, rounding="sr", sr_bits=40), spec(6, 130, inf_nan="fn"),
        spec(14, 3, bias=-5)]
MODES = [dict(), dict(saturate=True), dict(wrap=True), dict(ftz=True),
         dict(tininess="before"), dict(nan_mode="propagate"), dict(nan_mode="x86"),
         dict(nan_mode="arm"), dict(signed=False)]


def size_of(s):
    E, M, kw = s
    return int(dict(kw).get("signed", True)) + E + M


def binary_ops(a, b):
    return (a + b, a - b, a * b, a / b, a.fma(b, a), b.fma(a, b),
            a.minimum(b), a.maximum(b), a.minimum_number(b), a.maximum_number(b),
            a.compare(b), a.compare(b, signaling=True), a.eq(b), a.lt(b), a.le(b),
            a < b, a <= b, a == b, a != b, a > b, a >= b, a & b, a | b, a ^ b)


def unary_ops(m, a, others):
    out = [a.sqrt(), -a, +a, abs(a), ~a, bool(a), a.to_bin(), a.to_bin(""), a.to_hex(),
           a.exp, a.mantissa, a.is_nan, a.is_inf, a.is_finite, a.is_snan, a.is_zero,
           a.is_subnormal, a.size, a.sign, float(a)]
    for r in ALL_ROUNDINGS:
        rr = m.Rounding(r)
        out += [a.round_to_integral(rr), a.round_to_integral(rr, exact=True),
                a.to_int(8, True, rr), a.to_int(5, False, rr, exact=False)]
    out += [a.round_to_integral(), a.to_int(), a.to_int(64, signed=False)]
    for o in others:
        out.append(a.convert(o))
    for attr in ("exact", "ulp", "unrounded"):
        try:
            out.append(getattr(a, attr))
        except ValueError as e:
            out.append(str(e))
    try:
        out.append(int(a))
    except (ValueError, OverflowError) as e:
        out.append((type(e).__name__, str(e)))
    return tuple(out)


# ---------------------------------------------------------------- FP tests

@pytest.mark.parametrize("base", SMALL, ids=str)
def test_small_formats_all_pairs(base, rng):
    """Every pair of codes of the 4-6 bit formats, under every rounding."""
    for r in ALL_ROUNDINGS:
        mode = rng.choice(MODES)
        s = with_modes(base, rounding=r, **mode)
        try:
            fmt(P, s)
        except ValueError:
            continue
        n = 1 << size_of(s)
        pairs = [(i, j) for i in range(n) for j in range(n)]
        if len(pairs) > 1200:
            pairs = rng.sample(pairs, 1200)
        for i, j in pairs:
            check(lambda m: binary_ops(fmt(m, s).from_raw(i), fmt(m, s).from_raw(j)), (s, i, j))


@pytest.mark.parametrize("base", SMALL + BYTE, ids=str)
def test_all_codes_unary(base, rng):
    """Every code of the small and byte formats through the unary ops."""
    targets = [spec(2, 1, inf_nan=False), spec(5, 10), spec(4, 3, inf_nan="fn", saturate=True),
               spec(8, 0, signed=False, inf_nan="fn", has_zero=False), spec(3, 2, wrap=True)]
    for _ in range(3):
        s = with_modes(base, rounding=rng.choice(ALL_ROUNDINGS), **rng.choice(MODES))
        try:
            fmt(P, s)
        except ValueError:
            continue
        for code in range(1 << size_of(s)):
            check(lambda m: unary_ops(m, fmt(m, s).from_raw(code), [fmt(m, t) for t in targets]),
                  (s, code))


def test_random_formats_mixed(rng, iters):
    """Random formats (all modes), mixed-format operands and promotions."""
    for _ in range(iters):
        fa, fb, fc = rand_fmt(rng), rand_fmt(rng), rand_fmt(rng)
        if rng.random() < 0.5:
            fb = fa
        sa, sb, sc = spec_of(fa), spec_of(fb), spec_of(fc)
        ca, cb, cc = (rand_code(rng, f.E, f.M, f.signed, f) for f in (fa, fb, fc))
        check(lambda m: binary_ops(fmt(m, sa).from_raw(ca), fmt(m, sb).from_raw(cb)), (sa, sb, ca, cb))
        check(lambda m: fmt(m, sa).from_raw(ca).fma(fmt(m, sb).from_raw(cb), fmt(m, sc).from_raw(cc)),
              (sa, sb, sc, ca, cb, cc))
        check(lambda m: unary_ops(m, fmt(m, sa).from_raw(ca), [fmt(m, sb), fmt(m, sc)]), (sa, ca))


def rand_number(rng):
    """A Python number of any supported kind."""
    k = rng.randrange(9)
    if k == 0:
        return rng.randint(-40, 40)
    if k == 1:
        return rng.choice([0, 1, -1, 2 ** 70 + 1, -(2 ** 130), 3 ** 50, True, False])
    if k == 2:
        return rng.uniform(-50, 50)
    if k == 3:
        return rng.choice([0.0, -0.0, 5e-324, -1e-310, 1e308, -1.7e308, 1e-40, 0.1,
                           float("inf"), -float("inf"), float("nan"), -float("nan")])
    if k == 4:
        return Fraction(rng.randint(-1000, 1000), rng.randint(1, 1000))
    if k == 5:
        return Fraction(rng.getrandbits(200) - (1 << 199), rng.getrandbits(150) + 1)
    if k == 6:
        return Fraction(rng.randint(-9, 9), 1 << rng.randint(0, 200))
    if k == 7:
        return rng.uniform(-1, 1) * 2.0 ** rng.randint(-160, 160)
    return Fraction(1, 3) * rng.choice([1, -1, 1000, Fraction(1, 1 << 40)])


def test_scalars_and_from_value(rng, iters):
    """Python-number operands (implicit casts and their warnings) and
    from_value on every kind of input."""
    for _ in range(iters):
        f = rand_fmt(rng)
        if rng.random() < 0.2:
            f = f._replace(rounding=V.Rounding.SR, sr_bits=rng.randint(1, 12))
        s = spec_of(f)
        code = rand_code(rng, f.E, f.M, f.signed, f)
        x = rand_number(rng)
        sr = rng.choice([None, None, 0, rng.getrandbits(f.sr_bits), 1 << f.sr_bits])
        check(lambda m: fmt(m, s).from_raw(code) + x, (s, code, x))
        check(lambda m: x - fmt(m, s).from_raw(code), (s, code, x))
        check(lambda m: (fmt(m, s).from_raw(code) * x, x / fmt(m, s).from_raw(code),
                         fmt(m, s).from_raw(code) < x, x == fmt(m, s).from_raw(code),
                         fmt(m, s).from_raw(code).fma(x, x)), (s, code, x))
        check(lambda m: (fmt(m, s)(x, sr_rand=sr), m.FP.from_value(x, fmt=fmt(m, s)),
                         fmt(m, s)(x)), (s, x, sr))
    for x in ["0.1", "-1e-5", "1/7", Decimal("2.5"), None, "abc", [1]]:
        check(lambda m: m.FP16(x), x)
        check(lambda m: m.FP16(1) + x, x)
    check(lambda m: (m.FP16(m.UINT(300, 9)), m.FP16(m.INT(-3, 4)),
                     m.E4M3(m.FixedFormat(4, 4)(1.3125)), m.FP32(m.FP16(0.1))))


def test_stochastic_rounding(rng, iters):
    """SR in every path, with explicit and drawn random bits."""
    for _ in range(iters // 2):
        f = rand_fmt(rng)._replace(rounding=V.Rounding.SR, sr_bits=rng.randint(1, 40))
        s = spec_of(f)
        ca, cb = (rand_code(rng, f.E, f.M, f.signed, f) for _ in range(2))
        x = rand_number(rng)
        check(lambda m: (binary_ops(fmt(m, s).from_raw(ca), fmt(m, s).from_raw(cb)),
                         unary_ops(m, fmt(m, s).from_raw(ca), [fmt(m, s)]),
                         fmt(m, s)(x), fmt(m, s)(x, sr_rand=7),
                         fmt(m, s).from_raw(ca).convert(fmt(m, s), sr_rand=3)), (s, ca, cb, x))
    seq = lambda m: [m.FP16(0.1).convert(m.FPFormat(2, 1, rounding=m.Rounding.SR)) for _ in range(50)]
    check(seq)


def test_sr_custom_source():
    def run(m):
        state = [0x5A5A]

        def lfsr(n):
            v = 0
            for _ in range(n):
                b = ((state[0] >> 0) ^ (state[0] >> 2) ^ (state[0] >> 3) ^ (state[0] >> 5)) & 1
                state[0] = (state[0] >> 1) | (b << 15)
                v = (v << 1) | b
            return v
        m.set_sr_source(lfsr)
        f = m.FPFormat(4, 3, rounding=m.Rounding.SR, sr_bits=5)
        return [f(x / 37) for x in range(-40, 40)]
    check(run)


@pytest.mark.parametrize("base", IEEE + WIDE, ids=str)
def test_wide_and_ieee(base, rng, iters):
    """IEEE binary16/bfloat16/binary32/binary64, and formats wider than the
    native fast path (binary128, long mantissas, many SR bits)."""
    E, M, _ = base
    for _ in range(max(iters // 4, 50)):
        s = with_modes(base, **rng.choice(MODES))
        if dict(base[2]).get("rounding") is None:
            s = with_modes(s, rounding=rng.choice(ALL_ROUNDINGS[:5]))
        try:
            fmt(P, s)
        except ValueError:
            continue
        f = fmt(P, s)
        def code():
            sign = rng.getrandbits(1) << (E + M) if f.signed else 0
            if rng.random() < 0.3:
                e = rng.choice([0, 1, f._top, f._top - 1, rng.randint(0, f._top)])
                return sign | (e << M) | rng.choice([0, 1, f._mask, rng.getrandbits(M) if M else 0])
            return sign | rng.getrandbits(E + M)
        ca, cb, cc = code(), code(), code()
        check(lambda m: binary_ops(fmt(m, s).from_raw(ca), fmt(m, s).from_raw(cb)), (s, ca, cb))
        check(lambda m: (fmt(m, s).from_raw(ca).fma(fmt(m, s).from_raw(cb), fmt(m, s).from_raw(cc)),
                         unary_ops(m, fmt(m, s).from_raw(ca), [m.FP16, m.FP64, m.E4M3])),
              (s, ca, cb, cc))
        x = rand_number(rng)
        check(lambda m: (fmt(m, s)(x), fmt(m, s).from_raw(ca) * x), (s, ca, x))


def test_add_cancellation_and_gaps(rng, iters):
    """Additions with every exponent gap and near-total cancellation (the
    native adder keeps only a window of bits plus a sticky bit)."""
    for _ in range(iters):
        base = rng.choice(IEEE + [spec(4, 3), spec(5, 2), spec(8, 23, rounding="sr", sr_bits=12)])
        s = with_modes(base, rounding=rng.choice(ALL_ROUNDINGS), **rng.choice(MODES[:5]))
        E, M, _ = s
        f = fmt(P, s)
        ea = rng.randint(0, f._top)
        eb = min(max(ea + rng.randint(-M - 8, M + 8), 0), f._top)
        ma = rng.getrandbits(M) if M else 0
        mb = rng.choice([ma, ma ^ 1, rng.getrandbits(M) if M else 0, (ma + 1) & f._mask])
        sign = 1 << (E + M) if f.signed else 0
        a, b = (ea << M) | ma, sign | (eb << M) | mb
        c = rng.getrandbits(E + M)
        check(lambda m: (fmt(m, s).from_raw(a) + fmt(m, s).from_raw(b),
                         fmt(m, s).from_raw(a) - fmt(m, s).from_raw(b ^ sign),
                         fmt(m, s).from_raw(a).fma(fmt(m, s).from_raw(c), fmt(m, s).from_raw(b)),
                         fmt(m, s).from_raw(b).fma(fmt(m, s).from_raw(a), fmt(m, s).from_raw(a))),
              (s, a, b, c))


def test_format_api(rng):
    names = []
    for _ in range(300):
        f = rand_fmt(rng)
        s = spec_of(f)
        check(lambda m: (fmt(m, s), str(fmt(m, s)), repr(fmt(m, s)), fmt(m, s).kw,
                         fmt(m, s).max, fmt(m, s).min_normal, fmt(m, s).min_subnormal,
                         fmt(m, s).emin, fmt(m, s).emax, fmt(m, s).size,
                         m.FPFormat.parse(str(fmt(m, s))) == fmt(m, s),
                         hash(fmt(m, s)) == hash(m.FPFormat.parse(str(fmt(m, s)))),
                         fmt(m, s).replace(exp_bits=f.E + 1), fmt(m, s).replace(rounding=m.Rounding.RTZ),
                         [fmt(m, s).max_value(sg) for sg in (False, True)]), s)
        for meth in ("zero", "inf", "nan"):
            check(lambda m: [getattr(fmt(m, s), meth)(sg) for sg in (False, True)], (s, meth))
        payload = rng.choice([0, 1, 3])
        check(lambda m: fmt(m, s).nan(payload=payload), s)
        names.append(str(fmt(P, s)))
    for name in names[:50] + ["e4m3, bogus", "x", "ue8m0, fn, no-zero"]:
        check(lambda m: m.FPFormat.parse(name), name)
    bad = [dict(exp_bits=0), dict(mantissa_bits=-1), dict(exp_bits=1), dict(mantissa_bits=0),
           dict(inf_nan="x"), dict(tininess="never"), dict(rounding="rne"), dict(sr_bits=0),
           dict(bias=1.5), dict(exp_bits=2.0), dict(exp_bits=2, mantissa_bits=0, inf_nan=False,
                                                    has_zero=False, signed=False, bias=100)]
    for kw in bad:
        check(lambda m: m.FPFormat(**kw), kw)
    check(lambda m: (list(m.FPFormat(2, 1).all_values()), list(m.FP.all_values(3, 1, signed=False)),
                     m.FP.zero(3, 2, sign=True), m.FP.max_value(m.E4M3, sign=True),
                     m.FP.from_raw(0x3c, 5, 2), m.FP.from_raw(-1, fmt=m.FP16),
                     m.FP.from_raw(1 << 70, m.FP16)))
    check(lambda m: m.FP.from_value(1, m.FPFormat(2, 1), 3))
    check(lambda m: m.FP(True, m.UINT(3, 4), m.UINT(5, 3), bias=4, rounding=m.Rounding.RTZ))
    check(lambda m: m.FP(True, m.UINT(3, 4), m.UINT(5, 3), signed=False))
    check(lambda m: m.FP(False, 3, m.UINT(5, 3)))
    check(lambda m: (m.FP16(1.5).exp_bits, m.FP16(1.5).rounding, m.FP16(1.5).nan_mode,
                     m.FP16(1.5).tininess, m.FP16(1.5).inf_nan, m.FP16(1.5).format))
    check(lambda m: m.FP16(1.5).nonexistent)
    check(lambda m: (m.FP16(1.5).error_ulps(), m.FP16(0.1).error_ulps(),
                     m.FP16(0.1).error_ulps(m.FP32(0.1)), m.FP16(1).sqrt().error_ulps()))
    check(lambda m: m.FP16(2).sqrt().error_ulps())


def test_pickle_and_copy():
    def run(m):
        xs = [m.FP16(0.1), m.FP16(1) / m.FP16(0), m.E4M3(1000), m.FP16(-0.0), m.FP16(2).sqrt(),
              m.UINT(5, 3, saturate=True), m.INT(-2, 70), m.FP16]
        return [(pickle.loads(pickle.dumps(x)), copy.copy(x), copy.deepcopy(x)) for x in xs]
    check(run)


# ---------------------------------------------------------------- integers

def rand_int_obj(rng):
    signed = rng.random() < 0.5
    bits = rng.choice([rng.randint(1 + signed, 16), rng.randint(17, 64), rng.randint(65, 140)])
    v = rng.choice([rng.getrandbits(bits + 2) - (1 << bits), 0, -1, (1 << bits) - 1,
                    1 << (bits - 1), -(1 << (bits - 1)), rng.randint(-5, 5)])
    return ("INT" if signed else "UINT", v, bits, rng.random() < 0.3)


def build_int(m, t):
    name, v, bits, sat = t
    return getattr(m, name)(v, bits, sat)


def test_integers(rng, iters):
    for _ in range(iters):
        a = rand_int_obj(rng)
        b = rand_int_obj(rng) if rng.random() < 0.6 else (a[0], rng.randint(-9, 9), a[2], a[3])
        lit = rng.choice([rng.randint(-300, 300), 2 ** 100, -(2 ** 65), True, 0, 1])
        n = rng.choice([0, 1, 3, 63, 64, 65, 200])
        width = rng.randint(0, 70)

        def ops(m):
            x, y = build_int(m, a), build_int(m, b)
            out = []
            for f in (lambda: x + y, lambda: x - y, lambda: x * y, lambda: x // y, lambda: x % y,
                      lambda: x & y, lambda: x | y, lambda: x ^ y, lambda: x + lit, lambda: lit - x,
                      lambda: x * lit, lambda: lit // x, lambda: x % lit, lambda: lit & x,
                      lambda: x < y, lambda: x <= y, lambda: x > y, lambda: x >= y, lambda: x == y,
                      lambda: x != y, lambda: x < lit, lambda: lit < x, lambda: x == lit,
                      lambda: -x, lambda: ~x, lambda: x << n, lambda: x >> n, lambda: x << -1,
                      lambda: x.resize(width), lambda: int(x), lambda: bool(x),
                      lambda: [1, 2, 3, 4][x.raw % 4], lambda: x.to_bin(), lambda: x.to_hex(),
                      lambda: (x.min, x.max, x.mask, x.raw, x.size, x.signed),
                      lambda: x + 1.5, lambda: x + m.FP16(1)):
                try:
                    out.append(snap(f()))
                except Exception as e:  # noqa: BLE001
                    out.append(("raise", type(e).__name__, str(e)))
            return out
        check(ops, (a, b, lit, n))
    for args in [(5,), (5, 0), (5, -1), (5, True), (5, 8.0), ("7", 8), (3.9, 8), (300, 8, True),
                 (-300, 8, 1), (), (1, 2 ** 7)]:
        check(lambda m: (m.UINT(*args), m.INT(*args)), args)
    check(lambda m: (m.UINT.signed, m.INT.signed, issubclass(m.INT, m.UINT), isinstance(m.INT(1, 4), m.UINT)))


# ---------------------------------------------------------------- fixed point

def rand_fixed_spec(rng):
    signed = rng.random() < 0.7
    ib = rng.randint(1 if signed else 0, 10)
    fb = rng.randint(-2, 10)
    if ib + fb < 1:
        fb = 1 - ib
    return (ib, fb, signed, rng.choice(ROUNDINGS).value, rng.choice(["wrap", "saturate"]))


def fixed_fmt(m, s):
    ib, fb, signed, r, o = s
    return m.FixedFormat(ib, fb, signed, rounding=m.Rounding(r), overflow=o)


def test_fixed(rng, iters):
    for _ in range(iters):
        sa, sb, sc = rand_fixed_spec(rng), rand_fixed_spec(rng), rand_fixed_spec(rng)
        ra, rb = rng.getrandbits(sa[0] + sa[1]), rng.getrandbits(sb[0] + sb[1])
        x = rand_number(rng)
        n = rng.randint(-3, 3)

        def ops(m):
            a, b = fixed_fmt(m, sa).from_raw(ra), fixed_fmt(m, sb).from_raw(rb)
            out = []
            for f in (lambda: a + b, lambda: a - b, lambda: a * b, lambda: -a, lambda: a.cast(fixed_fmt(m, sc)),
                      lambda: a.cast(sc[0], sc[1]), lambda: a.div(b, fixed_fmt(m, sc)), lambda: a + x,
                      lambda: x * a, lambda: fixed_fmt(m, sc)(x), lambda: a << n, lambda: a >> n,
                      lambda: a == b, lambda: a < b, lambda: a <= x, lambda: a > b, lambda: a >= 1,
                      lambda: hash(a), lambda: float(a), lambda: int(a), lambda: bool(a),
                      lambda: a.to_bin(), lambda: a.to_hex(), lambda: (a.raw, a.exact, a.bits),
                      lambda: m.FP16(a), lambda: fixed_fmt(m, sc)(m.FP16(1.5)),
                      lambda: fixed_fmt(m, sc)(m.INT(-3, 5)), lambda: (fixed_fmt(m, sc).min, fixed_fmt(m, sc).max)):
                try:
                    out.append(snap(f()))
                except Exception as e:  # noqa: BLE001
                    out.append(("raise", type(e).__name__, str(e)))
            return out
        check(ops, (sa, sb, sc, ra, rb, x))


# ---------------------------------------------------------------- block tensors

def rand_block_spec(rng):
    block = rng.choice([1, 2, 4, 8, 16, 32])
    if rng.random() < 0.5:
        f = rand_fmt(rng)
        while not f.has_zero:
            f = rand_fmt(rng)
        elem, zp = ("fp", spec_of(f)), None
    else:
        elem = ("int", rng.randint(2, 8), rng.random() < 0.7, rng.choice(ROUNDINGS).value,
                rng.choice([0, 0, 2, 6]))
        zp = ("int", rng.randint(2, 8), rng.random() < 0.5, "rne", 0) if rng.random() < 0.4 else None
    kind = rng.random()
    if zp is not None or kind < 0.5:
        scale = ("fp", rng.choice([spec(4, 3, inf_nan="fn"), spec(4, 3, signed=False, inf_nan="fn"),
                                   spec(5, 2), spec(3, 2, bias=6)]))
    elif kind < 0.85:
        bits = rng.randint(4, 8)
        scale = ("pow2", bits, rng.randint(0, (1 << bits) - 1), rng.random() < 0.8)
    else:
        scale = None
    kw = {}
    if scale and scale[0] == "fp":
        if rng.random() < 0.3:
            kw["scale_max"] = rng.choice([Fraction(1, 2), 3, 100, 448])
        if rng.random() < 0.5:
            kw["tensor_scale"] = spec(8, 23)
        if rng.random() < 0.4:
            kw.update(compute=rng.choice([spec(8, 23), None, spec(5, 10)]),
                      scale_min=rng.choice([None, Fraction(1, 512), Fraction(1, 8)]),
                      zero_scale=rng.choice([None, 1, Fraction(1, 4)]))
    return (elem, block, scale, zp, tuple(sorted(kw.items())))


def sub_fmt(m, s):
    if s is None:
        return None
    if s[0] == "fp":
        return fmt(m, s[1])
    if s[0] == "int":
        return m.IntFormat(s[1], s[2], m.Rounding(s[3]), s[4])
    if s[0] == "pow2":
        return m.Pow2Format(s[1], s[2], s[3])
    return fmt(m, s)


def block_fmt(m, s):
    elem, block, scale, zp, kw = s
    kw = {k: (sub_fmt(m, v) if isinstance(v, tuple) else v) for k, v in kw}
    return m.BlockFormat(sub_fmt(m, elem), block, sub_fmt(m, scale), zero_point=sub_fmt(m, zp), **kw)


def rand_values(rng, shape):
    if len(shape) == 1:
        return [rand_value(rng) for _ in range(shape[0])]
    return [rand_values(rng, shape[1:]) for _ in range(shape[0])]


def rand_value(rng):
    k = rng.random()
    if k < 0.1:
        return 0
    if k < 0.2:
        return Fraction(rng.randint(-50, 50), rng.randint(1, 16))
    return rng.choice([1, -1]) * rng.uniform(0, 1) * 2.0 ** rng.randint(-12, 12)


def accumulators(m):
    f = m.FP32
    return [None, f, m.FP16, m.Accumulator(f), m.Accumulator(f, "exact"), m.Accumulator(f, "pairwise"),
            m.Accumulator(f, product=m.FP16), m.Accumulator(f, "pairwise", product=m.BF16),
            m.Accumulator(f, group=4, align_bits=10), m.Accumulator(m.FP16, group=16, align_bits=25),
            m.Accumulator(m.BF16, "exact", group=8, align_bits=4),
            m.Accumulator(m.FPFormat(5, 2, saturate=True), group=2)]


def test_block_tensors(rng, iters, kernel_level):
    for _ in range(max(iters // 40, 20)):
        bs = rand_block_spec(rng)
        shape = rng.choice([(rng.randint(1, 40),), (rng.randint(1, 4), rng.randint(1, 24)),
                            (2, rng.randint(1, 3), rng.randint(1, 9))])
        axis = rng.randrange(-len(shape), len(shape))
        va, vb = rand_values(rng, shape), rand_values(rng, shape)
        k2 = rng.randint(1, 4)
        vc = rand_values(rng, (shape[-1], k2))
        acc_i = rng.randrange(len(accumulators(V)))

        def ops(m):
            bf = block_fmt(m, bs)
            a = m.BlockTensor.quantize(va, bf, axis)
            b = m.BlockTensor.quantize(vb, bf)
            out = [bf, a, b]
            for f in (lambda: a + b, lambda: a - b, lambda: a * b, lambda: a * 3, lambda: -a,
                      lambda: a / 2, lambda: 2 - a, lambda: a + m.FP16(0.5), lambda: a.T, lambda: a.T.T == a,
                      lambda: a.reblock(-1), lambda: a.transpose(), lambda: a[0], lambda: a[-1],
                      lambda: len(a), lambda: a.to_float(), lambda: a == b,
                      lambda: m.matmul(a, vc, acc=accumulators(m)[acc_i]),
                      lambda: m.matmul(a, vc, acc=accumulators(m)[acc_i], out=bf),
                      lambda: m.matmul(b, b, transpose_b=True),
                      lambda: a @ m.BlockTensor.quantize(vc, bf),
                      lambda: m.dot(m.BlockTensor.quantize(va if len(shape) == 1 else va[0], bf),
                                    vb if len(shape) == 1 else vb[0], accumulators(m)[acc_i]),
                      lambda: m.BlockTensor.from_raw(a.elem_raw, a.scale_raw, bf, a.zero_raw,
                                                    a.tensor_scale.raw if a.tensor_scale else None,
                                                    axis=a.axis),
                      lambda: m.bus.pack_block_row(b), lambda: m.bus.unpack_block_row(
                          m.bus.pack_block_row(b), bf, shape[-1]) if b.ndim == 1 else None):
                try:
                    out.append(snap(f()))
                except Exception as e:  # noqa: BLE001
                    out.append(("raise", type(e).__name__, str(e)))
            return out
        check(ops, (bs, shape, axis))


def test_block_scales_that_underflow(kernel_level):
    """With a compute format, an intermediate of the scale recipe can round to
    zero (tiny values under a tensor scale); the reference then fails with a
    ZeroDivisionError from its Fraction arithmetic, and so must the core."""
    def formats(m):
        return [m.BlockFormat(m.IntFormat(2, frac_bits=6, rounding=m.Rounding.RTZ), 4, m.FPFormat(5, 2),
                              zero_point=m.IntFormat(8, signed=False), tensor_scale=m.FP32, compute=m.FP16,
                              zero_scale=1),
                m.BlockFormat(m.E2M1, 4, m.E4M3, tensor_scale=m.FP32, compute=m.FP16),
                m.BlockFormat(m.IntFormat(8), 8, m.FP16, tensor_scale=m.FP16, compute=m.FPFormat(4, 3)),
                m.BlockFormat(m.E4M3, 16, m.FP32, compute=m.FPFormat(5, 2)),
                m.NVFP4_MODELOPT]
    for k in range(0, 150):         # every binade down to where the tensor scale itself underflows
        u = 2.0 ** -k
        vals = [[1.5 * u, -2.0 * u, 3.0 * u, 0.0], [u, u * 2.0 ** -17, -u * 2.0 ** -30, 7.0 * u]]
        for i in range(5):
            def run(m):
                t = m.BlockTensor.quantize(vals, formats(m)[i])
                return (t, t * 2, m.matmul(t, [[1.0], [2.0], [3.0], [4.0]], acc=m.FP32))
            check(run, (k, i))


def test_named_block_formats(rng, kernel_level):
    for name in ("NVFP4", "NVFP4_MODELOPT", "MXFP4", "MXINT8"):
        vals = rand_values(rng, (3, 64))
        check(lambda m: (m.BlockTensor.quantize(vals, getattr(m, name)),
                         m.matmul(m.BlockTensor.quantize(vals, getattr(m, name)), vals,
                                  acc=m.FP32, transpose_b=True)), name)


def test_accumulator(rng, iters, kernel_level):
    for _ in range(max(iters // 10, 50)):
        fs = spec_of(rand_fmt(rng))
        n = rng.randint(0, 40)
        codes_a = [rng.getrandbits(size_of(fs)) for _ in range(n)]
        nums_b = [rand_number(rng) if rng.random() < 0.3 else rng.uniform(-4, 4) for _ in range(n)]
        nums_b = [x for x in nums_b]
        init = rng.choice([None, 0, 1.5, -2, Fraction(1, 3)])
        order = rng.choice(["exact", "sequential", "pairwise"])
        group = rng.choice([1, 1, 2, 3, 8])
        align = rng.choice([None, None, 0, 3, 12, 30])
        acc_s = spec_of(rand_fmt(rng))
        prod = rng.choice([None, spec_of(rand_fmt(rng)), spec(8, 23)])

        def run(m):
            f = fmt(m, fs)
            a = [f.from_raw(c) for c in codes_a]
            b = [x if isinstance(x, (int, Fraction)) or x == x and abs(x) != float("inf") else 1.0
                 for x in nums_b]
            acc = m.Accumulator(fmt(m, acc_s), order, None if prod is None else fmt(m, prod), group, align)
            init_v = init if not isinstance(init, str) else None
            return (acc, acc.sum_products(a, b, init_v), acc.sum_products(b, a),
                    acc.sum_products([x for x in a if x.is_finite], [x for x in a if x.is_finite]))
        check(run, (fs, codes_a, nums_b, init, order, group, align, acc_s, prod))
    check(lambda m: (m.Accumulator(m.FP32, "bogus")))
    check(lambda m: (m.Accumulator(m.FP32, group=0)))
    check(lambda m: (m.Accumulator(m.FP32, align_bits=-1)))
    check(lambda m: m.Accumulator(m.FP16).sum_products([1, 2], [3]))


def test_bus(rng):
    for _ in range(200):
        width = rng.randint(1, 12)
        count = rng.randint(1, 10)
        codes = [rng.getrandbits(width) for _ in range(count)]
        msb = rng.random() < 0.5
        check(lambda m: (m.bus.pack(codes, width, msb_first=msb),
                         m.bus.unpack(m.bus.pack(codes, width, msb_first=msb), width, count, msb_first=msb),
                         m.bus.pack(codes + [1 << width], width)), (codes, width))
        check(lambda m: m.bus.unpack_fp(m.bus.pack_fp([m.E2M1.from_raw(c & 15) for c in codes]),
                                        m.E2M1, count), codes)


# ---------------------------------------------------------------- edge grids
# Deterministic sweeps over the encodings where formats break: every pair of
# edge values, under every rounding, NaN convention and underflow option.

EDGE_FORMATS = [spec(5, 10), spec(8, 7), spec(8, 23), spec(11, 52), spec(4, 3, inf_nan="fn"),
                spec(5, 2), spec(8, 0, signed=False, inf_nan="fn", has_zero=False),
                spec(2, 1, inf_nan=False), spec(4, 3, signed=False, inf_nan="fn")]
EDGE_OPTIONS = [dict(), dict(ftz=True), dict(tininess="before"), dict(saturate=True)]


def edge_codes(s) -> list[int]:
    """Magnitude edge encodings, both signs (when signed)."""
    f = fmt(P, s)
    E, M = f.exp_bits, f.mantissa_bits
    top, mask = f._top, f._mask
    one = f.bias << M
    mags = {0, 1, mask, 1 << M, (1 << M) | 1, f._max_code, f._max_code - 1,
            one, one - 1, one + 1, (f.bias + 1) << M, max(f.bias - 1, 0) << M}
    if f.inf_nan is True:
        mags |= {top << M, (top << M) | 1, (top << M) | (1 << (M - 1)), (top << M) | mask,
                 (top << M) | (mask >> 1)}
    elif f.inf_nan == "fn":
        mags.add((top << M) | mask)
    mags = sorted(m for m in mags if 0 <= m < 1 << (E + M))
    sb = 1 << (E + M)
    return mags + ([m | sb for m in mags] if f.signed else [])


@pytest.mark.parametrize("base", EDGE_FORMATS, ids=str)
def test_edge_grid_binary(base):
    codes = edge_codes(base)
    for r in ALL_ROUNDINGS:
        for nm in NAN_MODES:
            for opt in EDGE_OPTIONS:
                s = with_modes(base, rounding=r, nan_mode=nm, **opt)
                for i in codes:
                    check(lambda m: [binary_ops(fmt(m, s).from_raw(i), fmt(m, s).from_raw(j))
                                     for j in codes], (s, i))


@pytest.mark.parametrize("base", EDGE_FORMATS, ids=str)
def test_edge_grid_fma_and_unary(base):
    codes = edge_codes(base)
    targets = [spec(5, 10), spec(4, 3, inf_nan="fn"), spec(8, 0, signed=False, inf_nan="fn", has_zero=False)]
    for r in ALL_ROUNDINGS:
        for nm in NAN_MODES:
            s = with_modes(base, rounding=r, nan_mode=nm)
            for i in codes:
                check(lambda m: unary_ops(m, fmt(m, s).from_raw(i), [fmt(m, t) for t in targets]), (s, i))
                for j in codes[::3]:
                    check(lambda m: [fmt(m, s).from_raw(i).fma(fmt(m, s).from_raw(j), fmt(m, s).from_raw(k))
                                     for k in codes], (s, i, j))


def test_edge_scalars():
    """Python numbers at the edges of every format, as operands and as values."""
    nums = [0, -0.0, 0.0, 1, -1, 2 ** 1024, -(2 ** 1024), 2 ** 64 - 1, -(2 ** 63), 5e-324, -5e-324,
            1.7976931348623157e308, float("inf"), -float("inf"), float("nan"), Fraction(1, 3),
            Fraction(-2, 3), Fraction(1, 2 ** 1100), Fraction(3, 2 ** 1075), Fraction(2 ** 1100 + 1, 3),
            2 ** -149, 2 ** -150, 1.5 * 2 ** -150, 2 ** -24, 2 ** -25, 3 * 2 ** -26, 65504, 65520, 65519.99,
            448, 464, 480, 57344, 61440, 2 ** 127, 2 ** 128 - 2 ** 103, 2 ** 128]
    for base in EDGE_FORMATS:
        for r in ALL_ROUNDINGS:
            for opt in EDGE_OPTIONS:
                s = with_modes(base, rounding=r, **opt)
                check(lambda m: [(fmt(m, s)(x), fmt(m, s)(x, sr_rand=1), fmt(m, s).max_value() + x,
                                  x * fmt(m, s).max_value(), fmt(m, s).from_raw(1) - x) for x in nums], s)


INT_WIDTHS = [1, 2, 7, 8, 31, 32, 63, 64, 65, 127, 128, 129]


def test_edge_grid_integers():
    for bits in INT_WIDTHS:
        for name in ("UINT", "INT"):
            if name == "INT" and bits < 2:
                continue
            lo = -(1 << (bits - 1)) if name == "INT" else 0
            hi = (1 << (bits - (name == "INT"))) - 1
            vals = sorted(v for v in {lo, lo + 1, -1, 0, 1, 2, hi - 1, hi} if lo <= v <= hi)
            for sat in (False, True):
                for other_bits in (bits, bits + 1, max(bits - 1, 2), 64):
                    for oname in ("UINT", "INT"):
                        def run(m):
                            out = []
                            for v in vals:
                                x = getattr(m, name)(v, bits, sat)
                                ov = [w for w in (-1, 0, 1, (1 << (other_bits - 1)) - 1)]
                                for w in ov:
                                    y = getattr(m, oname)(w, other_bits, sat)
                                    for f in (lambda: x + y, lambda: x - y, lambda: x * y, lambda: x // y,
                                              lambda: x % y, lambda: x & y, lambda: x | y, lambda: x ^ y,
                                              lambda: x < y, lambda: x >= y, lambda: x == y,
                                              lambda: x + w, lambda: w - x, lambda: x * (hi + 1),
                                              lambda: -x, lambda: ~x, lambda: x << bits, lambda: x >> 1,
                                              lambda: x.resize(other_bits)):
                                        try:
                                            out.append(snap(f()))
                                        except Exception as e:  # noqa: BLE001
                                            out.append(("raise", type(e).__name__, str(e)))
                            return out
                        check(run, (name, bits, sat, oname, other_bits))


# ---------------------------------------------------------------- every setting
# The full cross product of the format options, so that no combination is
# left to chance: signed x inf_nan x has_zero x saturate x wrap x ftz x
# tininess x nan_mode x rounding (4608 combinations per base format; the
# invalid ones must be rejected identically).

CROSS = [(sg, inn, hz) for sg in (True, False) for inn in (True, False, "fn") for hz in (True, False)]
CROSS_BASES = [(2, 1, None), (3, 2, None), (2, 2, 0), (3, 0, None), (3, 1, 5), (4, 3, None)]


@pytest.mark.parametrize("signed, inf_nan, has_zero", CROSS, ids=str)
def test_every_setting(signed, inf_nan, has_zero, rng, iters):
    npairs = max(3, iters // 500)
    targets = [spec(5, 10), spec(2, 1, inf_nan=False), spec(4, 3, inf_nan="fn", saturate=True)]
    for saturate in (False, True):
        for wrap in (False, True):
            for ftz in (False, True):
                for tininess in ("after", "before"):
                    for nm in NAN_MODES:
                        for r in ALL_ROUNDINGS:
                            E, M, bias = rng.choice(CROSS_BASES)
                            kw = dict(signed=signed, inf_nan=inf_nan, has_zero=has_zero, saturate=saturate,
                                      wrap=wrap, ftz=ftz, tininess=tininess, nan_mode=nm, rounding=r)
                            if bias is not None:
                                kw["bias"] = bias
                            s = spec(E, M, **kw)
                            check(lambda m: fmt(m, s), s)
                            try:
                                fmt(P, s)
                            except ValueError:
                                continue
                            n = 1 << size_of(s)
                            for _ in range(npairs):
                                i, j, k = rng.randrange(n), rng.randrange(n), rng.randrange(n)
                                check(lambda m: binary_ops(fmt(m, s).from_raw(i), fmt(m, s).from_raw(j)), (s, i, j))
                                check(lambda m: fmt(m, s).from_raw(i).fma(fmt(m, s).from_raw(j),
                                                                          fmt(m, s).from_raw(k)), (s, i, j, k))
                            i = rng.randrange(n)
                            x = rand_number(rng)
                            check(lambda m: unary_ops(m, fmt(m, s).from_raw(i), [fmt(m, t) for t in targets]),
                                  (s, i))
                            check(lambda m: (fmt(m, s)(x), fmt(m, s).from_raw(i) + x, x * fmt(m, s).from_raw(i),
                                             m.FP16(x if isinstance(x, (int, Fraction)) else 0.75)
                                             .convert(fmt(m, s))), (s, i, x))


@pytest.mark.parametrize("base", [(2, 1, None), (2, 2, 0), (3, 1, 5)], ids=str)
def test_every_setting_all_pairs(base, rng, iters):
    """All pairs of codes of a 4- or 5-bit format, in each setting. The
    default run takes a random third of the settings; VERIFLOAT_ITERS >= 20000
    takes them all."""
    E, M, bias = base
    take = 1.0 if iters >= 20000 else 0.04
    for signed, inf_nan, has_zero in CROSS:
        for saturate, wrap, ftz in [(a, b, c) for a in (False, True) for b in (False, True) for c in (False, True)]:
            for tininess in ("after", "before"):
                for nm in NAN_MODES:
                    for r in ALL_ROUNDINGS:
                        if rng.random() >= take:
                            continue
                        kw = dict(signed=signed, inf_nan=inf_nan, has_zero=has_zero, saturate=saturate,
                                  wrap=wrap, ftz=ftz, tininess=tininess, nan_mode=nm, rounding=r)
                        if bias is not None:
                            kw["bias"] = bias
                        s = spec(E, M, **kw)
                        try:
                            fmt(P, s)
                        except ValueError:
                            continue
                        n = 1 << size_of(s)

                        def table(m):
                            f = fmt(m, s)
                            vals = [f.from_raw(c) for c in range(n)]
                            out = []
                            for a in vals:
                                for b in vals:
                                    for op in (lambda: a + b, lambda: a - b, lambda: a * b, lambda: a / b,
                                               lambda: a.fma(b, a), lambda: a.minimum(b),
                                               lambda: a.maximum_number(b), lambda: a.compare(b, signaling=True)):
                                        try:
                                            out.append(snap(op()))
                                        except Exception as e:  # noqa: BLE001
                                            out.append(("raise", type(e).__name__, str(e)))
                            return out
                        check(table, s)


# ---------------------------------------------------------------- NaN handling

NAN_FORMATS = [spec(2, 1), spec(3, 2), spec(3, 3), spec(5, 10), spec(8, 23), spec(11, 52), spec(15, 112),
               spec(3, 2, inf_nan="fn"), spec(4, 3, inf_nan="fn"), spec(3, 3, signed=False),
               spec(4, 3, signed=False, inf_nan="fn")]


def nan_operands(s) -> list[int]:
    """Raw codes: every kind of NaN (quiet and signaling, payloads at each
    end, both signs), the infinities, zeros and a few finite values."""
    f = fmt(P, s)
    E, M, top, mask = f.exp_bits, f.mantissa_bits, f._top, f._mask
    mags = {0, 1, f._max_code, f.bias << M if 0 < f.bias < top else 1 << M}
    if f.inf_nan is True:
        q = 1 << (M - 1)
        mags |= {top << M, (top << M) | q, (top << M) | mask, (top << M) | 1, (top << M) | (q >> 1),
                 (top << M) | (q | 1), (top << M) | (mask >> 1), (top << M) | (q | (q >> 1))}
    elif f.inf_nan == "fn":
        mags |= {(top << M) | mask, (top << M) | (mask >> 1), top << M}
    mags = sorted(m for m in mags if 0 <= m < 1 << (E + M))
    return mags + ([m | (1 << (E + M)) for m in mags] if f.signed else [])


@pytest.mark.parametrize("base", NAN_FORMATS, ids=str)
@pytest.mark.parametrize("nan_mode", NAN_MODES)
def test_nan_handling(base, nan_mode, rng):
    """Every NaN kind through every operation under each NaN convention:
    which NaN comes back (sign, payload, quieting), and INVALID."""
    s = with_modes(base, nan_mode=nan_mode)
    codes = nan_operands(s)
    others = [with_modes(t, nan_mode=nan_mode) for t in NAN_FORMATS if t != base]
    targets = [spec(5, 10, nan_mode=nan_mode), spec(3, 2, nan_mode=nan_mode), spec(15, 112, nan_mode=nan_mode),
               spec(4, 3, inf_nan="fn", nan_mode=nan_mode), spec(8, 23), spec(2, 1, inf_nan=False),
               spec(8, 0, signed=False, inf_nan="fn", has_zero=False)]
    for i in codes:
        check(lambda m: unary_ops(m, fmt(m, s).from_raw(i), [fmt(m, t) for t in targets]), (s, i))
        check(lambda m: [binary_ops(fmt(m, s).from_raw(i), fmt(m, s).from_raw(j)) for j in codes], (s, i))
        # Mixed formats: the NaN is widened (payload kept) before the operation.
        o = rng.choice(others)
        ocodes = nan_operands(o)
        check(lambda m: [binary_ops(fmt(m, s).from_raw(i), fmt(m, o).from_raw(j)) for j in ocodes[::2]],
              (s, o, i))
        check(lambda m: [(fmt(m, s).from_raw(i) + x, x * fmt(m, s).from_raw(i), fmt(m, s).from_raw(i) == x,
                          fmt(m, s).from_raw(i).minimum_number(x))
                         for x in (float("nan"), -float("nan"), float("inf"), 0, -0.0, 1.5)], (s, i))
    # fma: a NaN (or 0 * inf) in each position
    for i in codes:
        for j in rng.sample(codes, min(len(codes), 8)):
            check(lambda m: [fmt(m, s).from_raw(i).fma(fmt(m, s).from_raw(j), fmt(m, s).from_raw(k))
                             for k in codes], (s, i, j))
    check(lambda m: [fmt(m, s)(x) for x in (float("nan"), -float("nan"), float("inf"), -float("inf"))], s)
    check(lambda m: [fmt(m, s).nan(sg, p) for sg in (False, True) for p in (None, 1, 2, 3)], s)
    check(lambda m: fmt(m, s).nan(payload=0), s)


def test_nan_in_accumulators(rng, kernel_level):
    for nm in NAN_MODES:
        s = spec(5, 10, nan_mode=nm)
        codes = nan_operands(s)
        for _ in range(60):
            a = [rng.choice(codes) for _ in range(rng.randint(1, 6))]
            b = [rng.choice(codes) for _ in a]
            order = rng.choice(["exact", "sequential", "pairwise"])
            prod = rng.choice([None, spec(8, 23, nan_mode=nm), spec(4, 3, inf_nan="fn")])

            def run(m):
                f = fmt(m, s)
                acc = m.Accumulator(f, order, None if prod is None else fmt(m, prod))
                x, y = [f.from_raw(c) for c in a], [f.from_raw(c) for c in b]
                return (acc.sum_products(x, y), m.dot(x, y, acc), m.matmul([x], [[v] for v in y], acc=acc),
                        m.matmul([x], [[v] for v in y], acc=f))
            check(run, (nm, a, b, order, prod))


# ---------------------------------------------------------------- rounding

ROUND_FORMATS = [spec(5, 10), spec(8, 23), spec(11, 52), spec(4, 3, inf_nan="fn"), spec(2, 1, inf_nan=False),
                 spec(3, 2), spec(8, 7), spec(4, 3, signed=False, inf_nan="fn"), spec(15, 112),
                 spec(8, 0, signed=False, inf_nan="fn", has_zero=False), spec(3, 0, inf_nan="fn"),
                 spec(6, 70), spec(3, 2, has_zero=False, inf_nan="fn")]
ROUND_OPTIONS = [dict(), dict(saturate=True), dict(wrap=True), dict(ftz=True), dict(tininess="before")]


def boundary_values(f, rng) -> list[Fraction]:
    """Exact values on and around every kind of rounding boundary of f."""
    tiny = Fraction(1, 1 << 400)
    out = [Fraction(0)]
    M = f.mantissa_bits
    codes = {0, 1, 2, f._max_code, f._max_code - 1, 1 << M, (1 << M) - 1, (1 << M) + 1}
    codes |= {rng.getrandbits(f.exp_bits + M) for _ in range(6)}
    for c in sorted(codes):
        if c < 0 or c > f._max_code:
            continue
        x = P.FP._from_raw_fmt(c, f)
        if not x.is_finite:
            continue
        v = x.exact
        nxt = P.FP._from_raw_fmt(c + 1, f).exact if c < f._max_code else None
        if nxt is None:             # above the largest value: the overflow thresholds
            ulp = v - P.FP._from_raw_fmt(c - 1, f).exact if c else v
            nxt = v + ulp
            out += [nxt, nxt + tiny, nxt * 2, nxt * 4 + tiny]
        gap = nxt - v
        out += [v, v + gap / 2, v + gap / 2 - tiny, v + gap / 2 + tiny, v + gap / 4, v + 3 * gap / 4,
                v + tiny, nxt - tiny, v + gap / 3]
    lo = f.min_subnormal
    out += [lo / 2, lo / 2 - tiny, lo / 2 + tiny, lo / 4, lo - tiny, tiny, f.min_normal - tiny,
            f.min_normal - lo / 2, f.min_normal - lo / 4, f.min_normal * (1 - Fraction(1, 1 << (M + 2)))]
    return out


@pytest.mark.parametrize("base", ROUND_FORMATS, ids=str)
@pytest.mark.parametrize("rounding", ALL_ROUNDINGS)
def test_rounding_boundaries(base, rounding, rng):
    """Values exactly on, just below and just above each rounding boundary
    (ties, the overflow threshold, the subnormal range, tininess), reached
    three ways: from an exact number, by conversion, and as a sum."""
    wide = spec(15, 600)
    for opt in ROUND_OPTIONS:
        s = with_modes(base, rounding=rounding, **opt)
        try:
            f = fmt(P, s)
        except ValueError:
            continue
        for v in boundary_values(f, rng):
            for x in ((v, -v) if f.signed or v == 0 else (v, -v / 8)):
                sr = rng.choice([None, 0, 1, (1 << f.sr_bits) - 1, 1 << (f.sr_bits - 1)]) \
                    if rounding == "sr" else None
                check(lambda m: (fmt(m, s)(x, sr_rand=sr), m.FP.from_value(x, fmt(m, s)),
                                 fmt(m, wide)(x).convert(fmt(m, s), sr_rand=sr)), (s, x, sr))
                # the same value as a sum of two exact parts, and as a product
                part = Fraction(rng.randint(1, 7), 8) * x
                check(lambda m: (fmt(m, wide)(part) + fmt(m, wide)(x - part)).convert(fmt(m, s)), (s, x))
                check(lambda m: (fmt(m, s)(part) + (x - part), fmt(m, s)(1) * x, fmt(m, s)(x) / 1,
                                 fmt(m, s)(2).fma(x / 2, 0), fmt(m, s)(1).fma(part, x - part)), (s, x))


def test_rounding_to_integers(rng, iters):
    """round_to_integral and to_int on ties and near-ties, every mode."""
    for _ in range(max(iters // 20, 60)):
        s = rng.choice([spec(5, 10), spec(8, 23), spec(11, 52), spec(4, 3, inf_nan="fn"), spec(6, 70)])
        s = with_modes(s, nan_mode=rng.choice(NAN_MODES))
        n = rng.randint(-70, 70)
        x = Fraction(2 * n + 1, 2) + rng.choice([0, 0, Fraction(1, 1 << 9), -Fraction(1, 1 << 9), Fraction(1, 4)])
        x *= rng.choice([1, 1, 1 << 20, 1 << 40, Fraction(1, 64)])
        bits = rng.choice([1, 2, 8, 16, 32, 64, 65])
        sg = rng.random() < 0.5

        def run(m):
            a = fmt(m, s)(x)
            out = []
            for r in ALL_ROUNDINGS:
                rr = m.Rounding(r)
                out += [a.round_to_integral(rr), a.round_to_integral(rr, exact=True),
                        a.to_int(bits, sg, rr), a.to_int(bits, sg, rr, exact=False)]
            return out
        check(run, (s, x, bits, sg))


# ---------------------------------------------------------------- dot-product kernels

KERNEL_FORMATS = [spec(5, 10), spec(8, 7), spec(8, 23), spec(11, 52), spec(4, 3, inf_nan="fn"), spec(5, 2),
                  spec(2, 1, inf_nan=False), spec(8, 23, saturate=True), spec(5, 10, ftz=True),
                  spec(8, 23, tininess="before"), spec(6, 30), spec(10, 31), spec(3, 60), spec(8, 23, bias=100),
                  spec(4, 3, signed=False, inf_nan="fn"),
                  # few exponent bits: products and sums land on the edges of the range all the time
                  spec(4, 6, tininess="before"), spec(3, 12, tininess="before"), spec(4, 23),
                  spec(3, 10, inf_nan="fn"),
                  # ...and few mantissa bits: rounding often carries into the next binade, which at the
                  # bottom of the range is where tininess before and after rounding part ways
                  spec(4, 1, tininess="before"), spec(3, 2, tininess="before"), spec(4, 2),
                  spec(5, 3, tininess="before"), spec(4, 1)]


def kernel_codes(rng, f, n: int, style: int) -> list[int]:
    E, M, top = f.exp_bits, f.mantissa_bits, f._top
    out = []
    for _ in range(n):
        sign = (rng.getrandbits(1) << (E + M)) if f.signed else 0
        if style == 1:
            out.append(rng.getrandbits(f.size))
            continue
        if style == 0:
            e = min(max(f.bias + rng.randint(-4, 4), 1), max(top - 1, 1))
        elif style == 2:
            e = rng.randint(0, top)
        elif style == 3:
            if rng.random() < 0.6:
                out.append(sign)
                continue
            e = min(max(f.bias + rng.randint(-2, 2), 1), max(top - 1, 1))
        else:
            e = rng.randint(0, min(3, top))
        out.append(sign | (min(e, top) << M) | (rng.getrandbits(M) if M else 0))
    return out


# (under VERIFLOAT_IMPL=py this module is skipped and V has no _core)
KERNEL_LEVELS = ["general", "scalar"] + list(V._core.simd_available()) if hasattr(V, "_core") else ["general"]


@pytest.fixture(params=KERNEL_LEVELS)
def kernel_level(request):
    """Run the C++ core on one of its paths: the general one, the scalar fast
    kernels, or one of the SIMD instruction sets this CPU has."""
    level = request.param
    was = (V._core.set_fast(level != "general"), V._core.simd())
    V._core.set_simd("none" if level in ("general", "scalar") else level)
    yield level
    V._core.set_fast(was[0])
    V._core.set_simd(was[1])


@pytest.mark.parametrize("order", ["sequential", "pairwise", "exact"])
def test_dot_product_kernels(order, rng, iters, kernel_level):
    """matmul, dot and sum_products where the C++ core takes its fast kernels
    (ordinary formats, finite operands), and where it leaves them part way
    (overflow, subnormal results, zeros, cancellation, specials). Each path of
    the core must match the pure-Python model."""
    for it in range(max(iters // 12, 60)):
        sa, sb = rng.choice(KERNEL_FORMATS), rng.choice(KERNEL_FORMATS)
        sf = with_modes(rng.choice(KERNEL_FORMATS), rounding=rng.choice(ALL_ROUNDINGS))
        sp = rng.choice([None, with_modes(rng.choice(KERNEL_FORMATS), rounding=rng.choice(ALL_ROUNDINGS[:5]))])
        m_, k, n = rng.randint(1, 3), rng.randint(1, 10), rng.randint(1, 5)
        style = rng.choice([0, 0, 1, 2, 3, 4])
        ca = kernel_codes(rng, fmt(P, sa), m_ * k, style)
        cb = kernel_codes(rng, fmt(P, sb), k * n, rng.choice([style, 0]))
        if rng.random() < 0.3 and n > 1:        # exact cancellation between two columns
            sbit = 1 << (fmt(P, sb).exp_bits + fmt(P, sb).mantissa_bits)
            for t in range(k):
                cb[t * n + 1] = cb[t * n] ^ sbit if fmt(P, sb).signed else cb[t * n]
        group = rng.choice([1, 1, 1, 2, 4])
        align = rng.choice([None, None, None, 6])

        def run(m):
            fa, fb = fmt(m, sa), fmt(m, sb)
            a = [[fa.from_raw(ca[i * k + t]) for t in range(k)] for i in range(m_)]
            b = [[fb.from_raw(cb[t * n + j]) for j in range(n)] for t in range(k)]
            acc = m.Accumulator(fmt(m, sf), order, None if sp is None else fmt(m, sp), group, align)
            col = [row[0] for row in b]
            out = [m.matmul(a, b, acc=acc), m.dot(a[0], col, acc), acc.sum_products(a[0], col)]
            if order == "exact" and sp is None and group == 1:
                out.append(m.matmul(a, b, acc=fmt(m, sf)))
            return out
        check(run, (it, sa, sb, sf, sp, order, group, align, ca, cb))


# ---------------------------------------------------------------- calling conventions

KEYWORD_CALLS = [
    # every parameter of every public callable, passed by keyword
    lambda m: m.UINT(val=5, bits=8, saturate=True),
    lambda m: m.UINT(300, saturate=True, bits=8),
    lambda m: m.INT(val=-5, bits=4),
    lambda m: (m.UINT(), m.INT(), m.UINT(7), m.INT(-1, 3)),
    lambda m: m.UINT(200, 8).resize(bits=4),
    lambda m: m.INT(-3, 8, True).resize(bits=2),
    lambda m: m.FP(sign=True, exp=m.UINT(3, 4), mantissa=m.UINT(5, 3)),
    lambda m: m.FP(False, m.UINT(3, 4), mantissa=m.UINT(5, 3), bias=4, signed=False, saturate=True),
    lambda m: m.FP(False, exp=m.UINT(3, 4), mantissa=m.UINT(5, 3), rounding=m.Rounding.RTZ, inf_nan="fn"),
    lambda m: m.FP16(value=0.1),
    lambda m: m.FPFormat(4, 3, rounding=m.Rounding.SR)(value=0.3, sr_rand=200),
    lambda m: m.FPFormat(exp_bits=4, mantissa_bits=3, bias=5, signed=False),
    lambda m: m.FP16.from_raw(raw=0x3c01),
    lambda m: (m.FP16.zero(sign=True), m.FP16.max_value(sign=True), m.FP16.inf(sign=True),
               m.FP16.nan(sign=True, payload=3), m.FP16.replace(mantissa_bits=3)),
    lambda m: m.FPFormat.parse(name="ue4m3, fn"),
    lambda m: m.FP.from_value(value=0.1, exp_bits=4, mantissa_bits=3, bias=5, signed=True, inf_nan="fn"),
    lambda m: m.FP.from_value(value=0.1, fmt=m.E4M3, sr_rand=1),
    lambda m: m.FP.from_value(0.1, m.E4M3),
    lambda m: m.FP.from_raw(raw=9, exp_bits=3, mantissa_bits=2, bias=1, signed=False, ftz=True),
    lambda m: m.FP.from_raw(raw=9, fmt=m.E5M2),
    lambda m: m.FP.zero(exp_bits=3, mantissa_bits=2, sign=True, bias=2, signed=True, fmt=None),
    lambda m: m.FP.max_value(exp_bits=3, mantissa_bits=2, sign=True, bias=2, signed=True),
    lambda m: list(m.FP.all_values(exp_bits=2, mantissa_bits=1, signed=False)),
    lambda m: m.FP16(1.5).fma(b=m.FP16(2), c=m.FP16(0.1)),
    lambda m: m.FP16(1.5).fma(c=3, b=2),
    lambda m: (m.FP16(1.5).compare(other=m.FP16(2), signaling=True), m.FP16(1.5).compare(other=2),
               m.FP16(1.5).eq(other=2, signaling=True), m.FP16(1.5).lt(other=2, signaling=False),
               m.FP16(1.5).le(other=m.FP16.nan(), signaling=False)),
    lambda m: (m.FP16(1.5).minimum(other=2), m.FP16(1.5).maximum(other=m.FP16(2)),
               m.FP16(1.5).minimum_number(other=m.FP16.nan()), m.FP16(1.5).maximum_number(other=2)),
    lambda m: (m.FP16(2.5).to_int(bits=8, signed=False, rounding=m.Rounding.RUP, exact=False),
               m.FP16(2.5).to_int(signed=False), m.FP16(2.5).round_to_integral(rounding=m.Rounding.RDN, exact=True),
               m.FP16(2.5).round_to_integral(exact=True), m.FP16(2.5).to_bin(sep="_")),
    lambda m: (m.FP16(0.1).convert(exp_bits=4, mantissa_bits=3, bias=6, signed=True, saturate=True),
               m.FP16(0.1).convert(fmt=m.E4M3), m.FP16(0.1).convert(m.E4M3, sr_rand=3),
               m.FP16(0.1).convert(fmt=m.FPFormat(4, 3, rounding=m.Rounding.SR), sr_rand=255),
               m.FP16(0.1).convert(4, 3), m.FP16(0.1).convert(exp_bits=m.E5M2)),
    lambda m: (m.FP16(0.1).error_ulps(ref=m.FP32(0.1)), m.FP16(0.1).error_ulps(bogus=1)),
    lambda m: m.FixedFormat(int_bits=4, frac_bits=4, signed=False, rounding=m.Rounding.RTZ, overflow="saturate"),
    lambda m: (m.FixedFormat(4, 4)(value=1.3), m.FixedFormat(4, 4).from_raw(raw=0x93),
               m.FixedFormat(4, 4).replace(frac_bits=2)),
    lambda m: m.FixedFormat(4, 4)(1.5).div(other=m.FixedFormat(4, 4)(0.75), fmt=m.FixedFormat(6, 6)),
    lambda m: m.FixedFormat(4, 4)(1.5).div(fmt=m.FixedFormat(6, 6), other=3),
    lambda m: (m.FixedFormat(4, 4)(1.5).cast(int_bits=2, frac_bits=1, signed=True, overflow="saturate"),
               m.FixedFormat(4, 4)(1.5).cast(int_bits=m.FixedFormat(3, 1)),
               m.FIXED.cast_value(value=1.3, fmt=m.FixedFormat(4, 4))),
    lambda m: m.Accumulator(fmt=m.FP16, order="pairwise", product=m.FP16, group=2, align_bits=4)
    .sum_products(a=[1.5, 2], b=[m.FP16(3), 4], init=1),
    lambda m: (m.dot(a=[1.5, 2], b=[3, 4], acc=m.FP16), m.matmul(a=[[1.5, 2]], b=[[3], [4]], acc=m.FP16,
                                                                out=None, transpose_b=False)),
    lambda m: m.BlockTensor.quantize(values=[0.5, 1, 2, 3], fmt=m.NVFP4, axis=-1),
    lambda m: m.BlockTensor.quantize([0.5, 1, 2, 3]).dot(other=[1, 2, 3, 4], acc=m.FP32),
    lambda m: m.bus.pack(codes=[1, 2, 3], width=4, msb_first=True),
    lambda m: m.bus.unpack(value=0x123, width=4, count=3, msb_first=True),
    lambda m: (m.IntFormat(bits=8, signed=True, frac_bits=6)(v=3), m.IntFormat(8).from_raw(raw=200),
               m.Pow2Format(bits=8, bias=127).from_raw(raw=3), m.Pow2Format(8, 127).value(code=130)),
    lambda m: m.BlockFormat(elem=m.E2M1, block_size=16, scale=m.E4M3, tensor_scale=m.FP32),
    lambda m: m.set_sr_source(source=5),
]

BAD_CALLS = [
    # wrong arity or unknown keywords: a TypeError in both (messages may differ)
    lambda m: m.UINT(1, 2, False, 4), lambda m: m.UINT(value=1), lambda m: m.UINT(1, 8).resize(),
    lambda m: m.UINT(1, 8).resize(4, 5), lambda m: m.UINT(1, 8).resize(width=4), lambda m: m.UINT(1, 8).to_hex(1),
    lambda m: m.FP(True), lambda m: m.FP(), lambda m: m.FP(True, m.UINT(1, 2), m.UINT(1, 2), 1, True, 5),
    lambda m: m.FP16(), lambda m: m.FP16(1, 2), lambda m: m.FP16(1, sr=3), lambda m: m.FP16(1).fma(1),
    lambda m: m.FP16(1).fma(1, 2, 3), lambda m: m.FP16(1).fma(1, d=2), lambda m: m.FP16(1).sqrt(1),
    lambda m: m.FP16(1).minimum(), lambda m: m.FP16(1).minimum(1, 2), lambda m: m.FP16(1).minimum(o=1),
    lambda m: m.FP16(1).compare(), lambda m: m.FP16(1).compare(1, True, 3), lambda m: m.FP16(1).to_int(1, 2, 3, 4, 5),
    lambda m: m.FP16(1).to_int(width=8), lambda m: m.FP16(1).round_to_integral(1, 2, 3),
    lambda m: m.FP16(1).to_bin(" ", 1), lambda m: m.FP16(1).to_hex(1), lambda m: m.FP16(1).convert(1, 2, 3, 4, 5),
    lambda m: m.FixedFormat(4, 4)(), lambda m: m.FixedFormat(4, 4)(1, 2), lambda m: m.FixedFormat(4, 4).from_raw(),
    lambda m: m.FixedFormat(4, 4)(1).div(1), lambda m: m.FixedFormat(4, 4)(1).div(1, m.FixedFormat(4, 4), 3),
    lambda m: m.FixedFormat(4, 4)(1).to_bin(1), lambda m: m.FIXED(1), lambda m: m.FP.from_value(),
]


def test_keyword_arguments():
    """Every parameter can be passed by name, exactly as in 0.1."""
    for i, call in enumerate(KEYWORD_CALLS):
        check(call, f"KEYWORD_CALLS[{i}]")
    for i, call in enumerate(BAD_CALLS):
        got, want = outcomes(call)
        assert got[0][:2] == want[0][:2] == ("raise", "TypeError"), (f"BAD_CALLS[{i}]", got, want)
