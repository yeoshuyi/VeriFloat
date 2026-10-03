"""Constrained random tests for UINT, INT and generic FP formats."""

from __future__ import annotations

import math
import operator
from fractions import Fraction

import pytest

from verifloat import (E2M1, E4M3, E5M2, FP, FP16, FP32, INT, UINT, CastWarning,
                       FPFlags, FPFormat, FPFormatWarning, FPModeWarning,
                       FPOverflowWarning, FPUnderflowWarning, IntCastWarning,
                       NaNMode, Rounding)
from reference import (ROUNDINGS, Fmt, Val, assert_ref, caught, check_fp_ops, default_bias,
                       NoNaN, ref_compare, ref_fma, ref_minmax, ref_round_int, ref_sqrt,
                       ref_to_int,
                       finite_grid, fmt_of, ref_arith, ref_decode, ref_inf, ref_nan,
                       grid, grid_set, rand_code, rand_fmt, rand_sint, rand_uint,
                       ref_round, ref_value, trunc_div, trunc_mod, wrap_s,
                       wrap_u)

INT_OPS = [operator.add, operator.sub, operator.mul,
           operator.and_, operator.or_, operator.xor]


# UINT
@pytest.mark.parametrize("op", INT_OPS)
def test_uint_binops(rng, iters, op):
    for _ in range(iters):
        bits = rng.randint(1, 64)
        a, b = rand_uint(rng, bits), rand_uint(rng, bits)
        k = rng.randint(-(2 << bits), 2 << bits)
        r = op(UINT(a, bits), UINT(b, bits))
        assert type(r) is UINT and r.bits == bits
        assert r.val == wrap_u(op(a, b), bits)
        lossy = {IntCastWarning} if not 0 <= k < 1 << bits else set()
        r, warned = caught(lambda: op(UINT(a, bits), k))
        assert r.val == wrap_u(op(a, k), bits) and warned == lossy
        r, warned = caught(lambda: op(k, UINT(a, bits)))
        assert r.val == wrap_u(op(k, a), bits) and warned == lossy


def test_uint_div_mod(rng, iters):
    for _ in range(iters):
        bits = rng.randint(1, 64)
        a, b = rand_uint(rng, bits), rand_uint(rng, bits)
        if b == 0:
            with pytest.raises(ZeroDivisionError):
                UINT(a, bits) // UINT(b, bits)
            continue
        assert (UINT(a, bits) // UINT(b, bits)).val == a // b
        assert (UINT(a, bits) % UINT(b, bits)).val == a % b


def test_uint_unary_shift_resize(rng, iters):
    for _ in range(iters):
        bits = rng.randint(1, 64)
        a = rand_uint(rng, bits)
        n = rng.randint(0, bits + 2)
        new = rng.randint(1, 64)
        x = UINT(a, bits)
        assert x.size == bits
        assert (~x).val == wrap_u(~a, bits)
        assert (x << n).val == wrap_u(a << n, bits)
        assert (x >> n).val == a >> n
        assert x.resize(new).val == wrap_u(a, new)
        assert x.resize(new).size == new
        assert x.raw == a and int(x) == a and x.to_bin() == format(a, f"0{bits}b")


def test_uint_compare(rng, iters):
    for _ in range(iters):
        bits = rng.randint(1, 64)
        a, b = rand_uint(rng, bits), rand_uint(rng, bits)
        x, y = UINT(a, bits), UINT(b, bits)
        assert (x < y) == (a < b)
        assert (x == y) == (a == b)
        assert (x >= y) == (a >= b)


# INT
@pytest.mark.parametrize("op", INT_OPS)
def test_int_binops(rng, iters, op):
    for _ in range(iters):
        bits = rng.randint(2, 64)
        a, b = rand_sint(rng, bits), rand_sint(rng, bits)
        r = op(INT(a, bits), INT(b, bits))
        assert type(r) is INT and r.bits == bits
        assert r.val == wrap_s(op(a, b), bits)
        assert r.raw == wrap_u(op(a, b), bits)


def test_int_div_mod_truncate(rng, iters):
    for _ in range(iters):
        bits = rng.randint(2, 64)
        a, b = rand_sint(rng, bits), rand_sint(rng, bits)
        if b == 0:
            continue
        assert (INT(a, bits) // INT(b, bits)).val == wrap_s(trunc_div(a, b), bits)
        assert (INT(a, bits) % INT(b, bits)).val == wrap_s(trunc_mod(a, b), bits)


def test_int_unary_shift_resize(rng, iters):
    for _ in range(iters):
        bits = rng.randint(2, 64)
        a = rand_sint(rng, bits)
        n = rng.randint(0, bits + 2)
        new = rng.randint(2, 64)
        x = INT(a, bits)
        assert isinstance(x, UINT) and x.size == bits
        assert (x.min, x.max) == (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        assert x.val == a and x.raw == wrap_u(a, bits)
        assert (~x).val == wrap_s(~a, bits)
        assert (x << n).val == wrap_s(a << n, bits)
        assert (x >> n).val == a >> n  # arithmetic
        assert x.resize(new).val == wrap_s(a, new)  # sign-extends / truncates


def test_int_compare(rng, iters):
    for _ in range(iters):
        bits = rng.randint(2, 64)
        a, b = rand_sint(rng, bits), rand_sint(rng, bits)
        assert (INT(a, bits) < INT(b, bits)) == (a < b)
        assert (INT(a, bits) == INT(b, bits)) == (a == b)


# Mixed signedness: any UINT operand makes the result UINT over raw bits
@pytest.mark.parametrize("op", INT_OPS)
def test_mixed_signedness(rng, iters, op):
    for _ in range(iters):
        ub, sb = rng.randint(1, 64), rng.randint(2, 64)
        u = UINT(rand_uint(rng, ub), ub)
        s = INT(rand_sint(rng, sb), sb)
        bits = max(ub, sb)
        for x, y in ((u, s), (s, u)):
            r, warned = caught(lambda: op(x, y))
            assert type(r) is UINT and r.bits == bits
            assert r.val == wrap_u(op(x.raw, y.raw), bits)
            assert warned == {IntCastWarning}      # signedness always warns
        s2 = INT(rand_sint(rng, ub + 1), ub + 1)
        r, warned = caught(lambda: op(s, s2))
        assert type(r) is INT and r.bits == max(sb, ub + 1)
        assert r.val == wrap_s(op(s.val, s2.val), r.bits)
        assert warned == ({IntCastWarning} if sb != ub + 1 else set())


# FP, random formats: custom bias, unsigned, wrap, ftz, rounding modes and
# IEEE / finite / OCP-fn special-value encodings
NX, UF, OF = FPFlags.INEXACT, FPFlags.UNDERFLOW, FPFlags.OVERFLOW
DZ, NV = FPFlags.DIVZERO, FPFlags.INVALID


def test_fp_ops(rng, iters):
    for _ in range(iters):
        f = rand_fmt(rng)
        check_fp_ops(rand_code(rng, *f.code, f=f), rand_code(rng, *f.code, f=f), f)


def expected_convert(D, f: Fmt):
    """Reference result of converting decoded value D into format f."""
    if D.kind == "fin":
        return ref_round(D.v, f, D.sign)
    if D.kind == "inf":
        return ref_inf(D.sign, f)
    return None if f.inf_nan is False else ref_nan(f, D)


def test_fp_raw_roundtrip(rng, iters):
    for _ in range(iters):
        f = rand_fmt(rng)
        code = rand_code(rng, *f.code, f=f)
        x = f.raw(code)
        D = ref_decode(code, f)
        assert fmt_of(x.format) == f
        assert x.size == int(f.signed) + f.E + f.M
        assert x.raw == code and x.flags == 0
        assert (x.is_nan, x.is_inf) == (D.kind == "nan", D.kind == "inf")
        if D.kind == "fin":
            assert x.exact == D.v
        assert x.to_bin(sep="") == format(code, f"0{x.size}b")
        assert int(x.to_hex(), 16) == code and len(x.to_hex()) == (x.size + 3) // 4
        assert (~x).raw == wrap_u(~code, x.size)
        r, warned = caught(lambda: FP.from_value(x, f.E, f.M, **f.kw))
        assert_ref(r, warned, expected_convert(D, f))
        neg, warned = caught(lambda: -x)
        if f.signed:
            assert neg.raw == code ^ f.sbit and not warned
        elif D.kind == "nan":
            assert neg.raw == code and not warned
        else:
            want = ref_inf(True, f) if D.kind == "inf" else ref_round(-D.v, f)
            assert_ref(neg, warned, want)


def test_fp_from_value_rounds(rng, iters):
    for _ in range(iters):
        f = rand_fmt(rng)
        top = 2 * finite_grid(f)[-1]
        x = Fraction(rng.randint(-10**6, 10**6), 10**6) * top
        if rng.random() < 0.3:  # stress the underflow region
            x /= 1 << (f.E + f.M + rng.randint(0, 8))
        r, warned = caught(lambda: FP.from_value(x, f.E, f.M, **f.kw))
        assert_ref(r, warned, ref_round(x, f), ctx=f"{x} in {f}")


def test_fp_convert(rng, iters):
    for _ in range(iters):
        f1, f2 = rand_fmt(rng), rand_fmt(rng)
        x = f1.raw(rand_code(rng, *f1.code, f=f1))
        want = expected_convert(ref_decode(x.raw, f1), f2)
        if want is None:
            with pytest.raises(ValueError):
                x.convert(f2.E, f2.M, **f2.kw)
            continue
        r, warned = caught(lambda: x.convert(f2.E, f2.M, **f2.kw))
        assert_ref(r, warned, want, ctx=f"{x!r} -> {f2}")


def test_fp_mixed_formats_promote(rng, iters):
    for _ in range(iters):
        f1, f2 = rand_fmt(rng), rand_fmt(rng)
        a, b = rand_code(rng, *f1.code, f=f1), rand_code(rng, *f2.code, f=f2)
        fa, fb = f1.raw(a), f2.raw(b)
        r, warned = caught(lambda: fa + fb)
        rf = fmt_of(r.format)
        # Common format holds both operand formats exactly; signed and
        # specials if either has them; other modes from the left.
        assert rf.signed == (f1.signed or f2.signed)
        assert rf.M == max(f1.M, f2.M)
        kinds = {f1.inf_nan, f2.inf_nan}
        assert rf.inf_nan == (True if True in kinds else "fn" if "fn" in kinds else False)
        assert (rf.wrap, rf.ftz, rf.rounding, rf.saturate, rf.nan_mode) == \
            (f1.wrap, f1.ftz, f1.rounding, f1.saturate, f1.nan_mode)
        assert set(finite_grid(rf)) >= set(finite_grid(f1)) | set(finite_grid(f2))
        if (f1.bias, f2.bias) == (default_bias(f1.E), default_bias(f2.E)) \
                and f1.inf_nan == f2.inf_nan and f1.has_zero and f2.has_zero:
            assert (rf.E, rf.bias) == (max(f1.E, f2.E), default_bias(rf.E))
        want = ref_arith("+", ref_decode(a, f1), ref_decode(b, f2), rf)
        mixed = set()
        if f1[:3] != f2[:3]:
            mixed.add(FPFormatWarning)
        if f1[3:] != f2[3:]:
            mixed.add(FPModeWarning)
        assert_ref(r, warned - mixed, want, ctx=f"{f1} + {f2}")
        assert mixed <= warned


def test_fp_python_specials(rng, iters):
    for _ in range(iters // 10):
        f = rand_fmt(rng)
        for v in (math.inf, -math.inf, math.nan):
            D = Val("nan", None, False) if math.isnan(v) else Val("inf", None, v < 0)
            want = expected_convert(D, f)
            if want is None:
                with pytest.raises(ValueError):
                    FP.from_value(v, f.E, f.M, **f.kw)
                continue
            r, warned = caught(lambda: FP.from_value(v, f.E, f.M, **f.kw))
            assert r.raw == want.raw and r.flags == want.flags and warned == want.warnings


# Anchors: custom bias / unsigned
def test_fp_custom_bias():
    x = FP.max_value(2, 1, bias=0, inf_nan=False)
    assert [float(FP.from_raw(c, 2, 1, bias=0, inf_nan=False)) for c in range(8)] == \
        [0, 1, 2, 3, 4, 6, 8, 12]
    assert repr(x) == "FP(12.0, e2m1, bias=0, finite)"
    assert float(FP.max_value(4, 3, bias=10, inf_nan=False)) == 60.0
    assert FP.from_value(1, inf_nan=False).bias == default_bias(2)
    assert float(FP.from_value(0.001, 2, 1, bias=-3, inf_nan=False)) == 0.0
    assert float(FP.from_value(5, 2, 1, bias=-3, inf_nan=False)) == 8.0   # subnormal step is 8


def test_fp_unsigned():
    u = FP.from_value(4, 2, 1, signed=False, inf_nan=False)
    assert (u.size, u.signed, repr(u), u.to_bin()) == (3, False, "FP(4.0, ue2m1, finite)", "11 0")
    assert FP.from_raw(0b1111, 2, 1, signed=False, inf_nan=False).raw == 0b111
    assert FP.from_value(-0.0, 2, 1, signed=False, inf_nan=False).raw == 0
    assert float(u - 1) == 3.0            # scalar keeps its sign
    with pytest.raises(ValueError):
        FP(True, UINT(1, 2), UINT(0, 1), signed=False)


def UE(code, **modes):
    return FP.from_raw(code, 2, 1, signed=False, **modes, inf_nan=False)


@pytest.mark.parametrize("fn, want, flags, warns", [
    (lambda u: u * 2, 6.0, OF | NX, {FPOverflowWarning}),     # 8 saturates
    (lambda u: u + 1, 4.0, NX, set()),                        # 5 ties to even
    (lambda u: u * 100, 6.0, OF | NX, {CastWarning, FPOverflowWarning}),  # 100 -> 6 cast
    (lambda u: u - 6, 0.0, UF | NX, {FPUnderflowWarning}),    # negative clamps
    (lambda u: 1 - u, 0.0, UF | NX, {FPUnderflowWarning}),
    (lambda u: -u, 0.0, UF | NX, {FPUnderflowWarning}),
    # Codes: 1 = 0.5 (subnormal), 2 = 1.0 (min normal), 3 = 1.5
    (lambda u: UE(1) * UE(2), 0.5, 0, set()),                 # tiny but exact
    (lambda u: UE(1) * UE(1), 0.0, UF | NX, {FPUnderflowWarning}),  # 0.25 ties to 0
    (lambda u: UE(1) * UE(3), 1.0, UF | NX, {FPUnderflowWarning}),  # 0.75 up to min normal
])
def test_fp_unsigned_overflow_underflow(fn, want, flags, warns):
    u = FP.from_value(4, 2, 1, signed=False, inf_nan=False)
    r, got = caught(lambda: fn(u))
    assert float(r) == want and r.signed is False
    assert r.flags == flags and got == warns


def test_fp_warnings_point_at_caller():
    u = FP.from_value(4, 2, 1, signed=False, inf_nan=False)
    for fn in (lambda: u * 2, lambda: u + FP.from_value(1, inf_nan=False), lambda: u > FP.from_value(1, inf_nan=False)):
        with pytest.warns(RuntimeWarning) as rec:
            fn()
        assert len(rec) == 1 and rec[0].filename == __file__


def test_signed_fp_never_warns(rng, iters):
    big = FP.from_value(1000, 5, 4, inf_nan=False)
    for _ in range(iters):
        E, M = rng.randint(2, 5), rng.randint(1, 4)
        fa = FP.from_raw(rand_code(rng, E, M), E, M, inf_nan=False)
        big.convert(E, M), FP.from_value(1e-9, E, M, inf_nan=False)  # error filter would fail
        FP.from_value(1e-9, E, M, ftz=True, inf_nan=False)
        fa * fa, fa - fa
    assert FP.from_value(100, inf_nan=False).flags == OF | NX            # flags still set


# Anchors: exponent wrap
@pytest.mark.parametrize("fn, want, flags, warns", [
    (lambda: FP.from_value(4, 2, 1, signed=False, wrap=True, inf_nan=False) * 2, 0.0, OF | NX, {FPOverflowWarning}),
    (lambda: FP.from_value(4, 2, 1, signed=False, wrap=True, inf_nan=False) * 3, 0.5, OF | NX, {FPOverflowWarning}),
    (lambda: FP.from_value(0.25, 2, 1, signed=False, wrap=True, inf_nan=False), 4.0, UF | NX, {FPUnderflowWarning}),
    (lambda: FP.from_value(0.5, 2, 1, signed=False, wrap=True, inf_nan=False), 0.0, UF | NX, {FPUnderflowWarning}),
    (lambda: FP.from_value(6, wrap=True, inf_nan=False) + 6, 0.5, OF | NX, {FPOverflowWarning}),
    (lambda: FP.from_value(-6, wrap=True, inf_nan=False) * 2, -0.5, OF | NX, {FPOverflowWarning}),
    (lambda: FP.from_value(4, 2, 1, signed=False, wrap=True, inf_nan=False) - 6, 0.0, UF | NX, {FPUnderflowWarning}),
    (lambda: FP.from_value(0.25, 2, 1, signed=False, wrap=True, ftz=True, inf_nan=False), 0.0, UF | NX, {FPUnderflowWarning}),
    (lambda: FP.from_value(3, wrap=True, inf_nan=False) * 2, 6.0, 0, set()),
])
def test_fp_wrap(fn, want, flags, warns):
    r, got = caught(fn)
    assert float(r) == want and r.wrap
    assert r.flags == flags and got == warns


# Anchors: rounding modes (finite E2M1 grid 0 .5 1 1.5 2 3 4 6)
R = Rounding


@pytest.mark.parametrize("x, rne, rna, rtz, rup, rdn", [
    (1.25, 1.0, 1.5, 1.0, 1.5, 1.0),
    (-1.25, -1.0, -1.5, -1.0, -1.0, -1.5),
    (1.75, 2.0, 2.0, 1.5, 2.0, 1.5),
    (-1.75, -2.0, -2.0, -1.5, -1.5, -2.0),
    (5.0, 4.0, 6.0, 4.0, 6.0, 4.0),
    (1.1, 1.0, 1.0, 1.0, 1.5, 1.0),
    (0.1, 0.0, 0.0, 0.0, 0.5, 0.0),
    (-0.1, -0.0, -0.0, -0.0, -0.0, -0.5),
])
def test_fp_rounding_modes(x, rne, rna, rtz, rup, rdn):
    for mode, want in zip([R.RNE, R.RNA, R.RTZ, R.RUP, R.RDN], [rne, rna, rtz, rup, rdn]):
        r = FP.from_value(x, rounding=mode, inf_nan=False)
        assert float(r) == want and str(float(r)) == str(want), mode
        assert r.rounding is mode


@pytest.mark.parametrize("mode, overflow", [
    (R.RNE, True), (R.RNA, True), (R.RTZ, False), (R.RUP, True), (R.RDN, False)])
def test_fp_rounding_overflow_threshold(mode, overflow):
    r = FP.from_value(7, rounding=mode, inf_nan=False)  # midway between max 6 and virtual 8
    assert float(r) == 6.0
    assert bool(r.flags & OF) == overflow and bool(r.flags & NX)


# Anchors: flush-to-zero / denormals-are-zero
def test_fp_ftz():
    sub = FP.from_raw(0b0001, ftz=True, inf_nan=False)          # 0.5 subnormal encoding
    assert float(sub) == 0.0 and sub.is_zero and sub.raw == 1 and sub.is_subnormal
    assert float(FP.from_raw(0b0001, inf_nan=False)) == 0.5    # gradual decodes it
    r = FP.from_value(0.5, ftz=True, inf_nan=False)
    assert (r.raw, r.flags) == (0, UF | NX)
    r = FP.from_value(0.9, ftz=True, inf_nan=False)             # rounds up to min normal: kept
    assert (float(r), r.flags) == (1.0, NX)
    r = FP.from_value(-0.3, ftz=True, inf_nan=False)
    assert (r.raw, r.flags) == (0b1000, UF | NX)
    with pytest.warns(CastWarning, match="lossy"):
        assert float(FP.from_value(1, ftz=True, inf_nan=False) - 0.5) == 1.0  # 0.5 literal flushed
    with pytest.raises(ZeroDivisionError):
        FP.from_value(1, ftz=True, inf_nan=False) / sub
    with pytest.warns(FPUnderflowWarning, match="flushed"):
        FP.from_value(0.3, 2, 1, signed=False, ftz=True, inf_nan=False)


# Anchors: verification data and formatting
def test_fp_introspection():
    x = FP.from_value(Fraction(51, 10), inf_nan=False)
    assert (float(x), x.unrounded, x.exact, x.ulp) == (6.0, Fraction(51, 10), 6, 2)
    assert x.error_ulps() == Fraction(9, 20)
    assert x.error_ulps(6) == 0 and x.error_ulps(FP.from_value(4, inf_nan=False)) == 1
    assert FP.from_raw(1, inf_nan=False).ulp == Fraction(1, 2)
    with pytest.raises(ValueError):
        FP.from_raw(1, inf_nan=False).error_ulps()
    assert x.flags == NX and int(FP.from_value(100, inf_nan=False).flags) == 0b101
    assert repr(FP.from_value(100, inf_nan=False)) == "FP(6.0, e2m1, finite, flags=INEXACT|OVERFLOW)"
    assert repr(FP.from_value(1, 4, 3, bias=10, signed=False, wrap=True, ftz=True,
                              rounding=R.RTZ, inf_nan=False)) == "FP(1.0, ue4m3, bias=10, finite, wrap, ftz, rtz)"
    assert (-FP.from_value(100, inf_nan=False)).flags == 0      # flags describe one rounding


def test_fp_all_values():
    vals = list(FP.all_values(2, 1, inf_nan=False))
    assert [v.raw for v in vals] == list(range(16))
    assert [float(v) for v in vals[:8]] == [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    assert len(list(FP.all_values(4, 3, signed=False, inf_nan=False))) == 128
    assert all(v.ftz for v in FP.all_values(2, 1, ftz=True, inf_nan=False))


@pytest.mark.parametrize("x, hex_, bits", [
    (FP.from_value(-6, inf_nan=False), "f", "1111"),
    (FP.from_value(1.0, 5, 2, inf_nan=False), "3c", "00111100"),
    (FP.from_value(-1, 5, 3, inf_nan=False), "178", "101111000"),
    (FP.from_value(0.5, 4, 4, signed=False, inf_nan=False), "60", "01100000"),
])
def test_fp_hex_bin(x, hex_, bits):
    assert x.to_hex() == hex_ and x.to_bin(sep="") == bits


def test_uint_hex():
    assert UINT(10, 12).to_hex() == "00a" and INT(-1, 5).to_hex() == "1f"


def test_fp_mixed_modes_warn_and_take_left():
    a, b = FP.from_value(1, wrap=True, inf_nan=False), FP.from_value(1, inf_nan=False)
    with pytest.warns(FPModeWarning, match=r"mismatched wrap: True \(left\) vs False"):
        assert (a + b).wrap
    with pytest.warns(FPModeWarning, match="using False"):
        assert not (b + a).wrap


@pytest.mark.parametrize("a, b, msgs", [
    (FPFormat(2, 1, signed=False), FPFormat(2, 1), ["mismatched signed: ue2m1 (left) vs e2m1 (right), result is signed"]),
    (FPFormat(2, 1, ftz=True), FPFormat(2, 1), ["mismatched ftz: True (left) vs False (right), using True"]),
    (FPFormat(2, 1, rounding=Rounding.RTZ), FPFormat(2, 1), ["mismatched rounding: rtz (left) vs rne (right), using rtz"]),
    (FPFormat(2, 1, wrap=True, rounding=Rounding.RUP), FPFormat(4, 3),
     ["mixing FP formats e2m1, wrap, rup and e4m3, promoted to e4m3, wrap, rup",
      "mismatched wrap: True (left) vs False (right), using True",
      "mismatched rounding: rup (left) vs rne (right), using rup"]),
])
def test_fp_each_mismatch_warns(a, b, msgs):
    with pytest.warns(CastWarning) as rec:
        a(1) * b(1)
    assert [str(w.message) for w in rec] == msgs
    assert all(issubclass(w.category, FPFormatWarning) for w in rec)


@pytest.mark.parametrize("fn, warns", [
    (lambda: FP.from_value(3, inf_nan=False) * 2, set()),                   # exact literal: silent
    (lambda: FP.from_value(3, inf_nan=False) * 0.1, {CastWarning}),         # 0.1 -> 0.0
    (lambda: FP.from_value(3, inf_nan=False) == 3.2, {CastWarning}),        # comparisons cast too
    (lambda: FP.from_value(1, inf_nan=False) + 7, {CastWarning}),           # 7 saturates to 6
    (lambda: UINT(3, 8) + 1, set()),
    (lambda: UINT(3, 8) + 300, {IntCastWarning}),            # 300 -> 44
    (lambda: UINT(3, 8) + -1, {IntCastWarning}),
    (lambda: INT(3, 8) + -1, set()),
    (lambda: UINT(3, 4) + UINT(3, 8), {IntCastWarning}),     # width
    (lambda: INT(3, 8) + UINT(3, 8), {IntCastWarning}),      # signedness
    (lambda: INT(3, 4) + INT(3, 8), {IntCastWarning}),
])
def test_automatic_casts_warn(fn, warns):
    assert caught(fn)[1] == warns


def test_int_literal_is_cast_before_compare_and_divide():
    with pytest.warns(IntCastWarning, match="300 to u8 is lossy: 44"):
        assert not UINT(50, 8) < 300                      # compares 50 < 44
    with pytest.warns(IntCastWarning):
        assert (UINT(200, 8) // 300).val == 200 // 44


# Anchors: IEEE 754 special values (binary16 and OCP FP8)
def test_ieee_overflow_by_rounding_mode():
    assert FP16.max == 65504
    for mode, pos, neg in [(R.RNE, "inf", "-inf"), (R.RNA, "inf", "-inf"),
                           (R.RTZ, 65504.0, -65504.0), (R.RUP, "inf", -65504.0),
                           (R.RDN, 65504.0, "-inf")]:
        f = FP16.replace(rounding=mode)
        assert float(f(1e6)) == float(pos) and float(f(-1e6)) == float(neg), mode
        assert f(1e6).flags == OF | NX
    r = FP16(65520)                      # exactly halfway to 2**16: ties to even -> inf
    assert r.is_inf and r.flags == OF | NX
    assert FP16(65519).raw == 0x7BFF    # just below halfway stays max
    assert float(FP16.replace(saturate=True)(1e6)) == 65504.0   # satfinite


def test_ieee_invalid_and_divzero():
    one, zero, inf = FP16(1), FP16(0), FP16(float("inf"))
    for r, raw, flags in [(one / zero, 0x7C00, DZ), (-one / zero, 0xFC00, DZ),
                          (zero / zero, 0x7E00, NV), (inf - inf, 0x7E00, NV),
                          (zero * inf, 0x7E00, NV), (inf / inf, 0x7E00, NV),
                          (inf + one, 0x7C00, 0), (one / inf, 0x0000, 0),
                          (-one / inf, 0x8000, 0), (inf * -2, 0xFC00, 0)]:
        assert (r.raw, r.flags) == (raw, flags), r
    assert (1 / zero).is_inf


def test_ieee_nan_semantics():
    nan, one = FP16(float("nan")), FP16(1)
    assert nan.raw == 0x7E00 and nan.is_nan and not nan.is_snan
    assert not (nan == nan) and nan != nan and not (nan < one) and not (nan >= one)
    assert math.isnan(float(nan)) and bool(nan)
    snan = FP16.from_raw(0x7C01)
    assert snan.is_snan and ((snan + 1).raw, (snan + 1).flags) == (0x7E00, NV)
    prop = FP16.replace(nan_mode=NaNMode.PROPAGATE)
    s2 = prop.from_raw(0xFC01)                           # negative sNaN, payload 1
    assert ((s2 * 2).raw, (s2 * 2).flags) == (0xFE01, NV)   # quieted, payload kept
    q = prop.from_raw(0x7E05)
    assert (q + prop.nan()).raw == 0x7E05 and (q + 1).flags == 0
    assert FP32(float("nan")).convert(FP16).is_nan
    assert prop.from_raw(0x7E05).convert(FP32.replace(nan_mode=NaNMode.PROPAGATE)).raw \
        == 0x7FC00000 | (0x205 << 13)                    # payload aligned to the MSB
    with pytest.raises(ValueError):
        FP16(float("nan")).convert(E2M1)                 # no NaN in OCP FP4
    with pytest.raises(ValueError):
        int(nan)
    with pytest.raises(OverflowError):
        int(FP16(float("inf")))


def test_ocp_fp8_encodings():
    assert (E4M3.max, E5M2.max) == (448, 57344)
    assert E4M3.from_raw(0x7F).is_nan and E4M3.from_raw(0xFF).is_nan
    assert E4M3.from_raw(0x7E).exact == 448 and not any(v.is_inf for v in E4M3.all_values())
    assert (E4M3(464).raw, E4M3(464).flags) == (0x7E, NX)       # tie goes to 448
    assert E4M3(470).is_nan and E4M3(470).flags == OF | NX      # OCP: overflow is NaN
    assert float(E4M3.replace(saturate=True)(470)) == 448.0
    assert E4M3(float("inf")).is_nan and E4M3(float("inf")).flags == NV
    assert E5M2(61440).is_inf and E5M2.from_raw(0x7C).is_inf    # E5M2 is IEEE-like
    assert sum(v.is_nan for v in E5M2.all_values()) == 6


def test_unsigned_ieee():
    u = FPFormat(5, 10, signed=False)
    with pytest.warns(FPOverflowWarning, match="rounded to inf"):
        assert u(1e6).is_inf and u.from_raw(0x7C00).is_inf
    with pytest.warns(FPUnderflowWarning, match="clamped"):
        r = u(float("-inf"))
    assert r.raw == 0 and r.flags == UF | NX


def test_inf_nan_mismatch_warns():
    with pytest.warns(FPModeWarning, match="mismatched inf_nan: ieee .left. vs finite "
                                           r"\(right\), result is ieee"):
        r = FP16(1) + FPFormat(5, 10, inf_nan=False)(1)
    assert r.inf_nan is True and r == 2


# FPFormat descriptor
def test_fpformat_basics():
    f = FPFormat(4, 3, ftz=True, rounding=Rounding.RTZ)
    assert f(1.7) == FP.from_value(1.7, 4, 3, ftz=True, rounding=Rounding.RTZ)
    assert f(1.7).format == f and f.from_raw(0x3c).raw == 0x3c
    assert FPFormat(2, 1) == FPFormat(2, 1, bias=1) and hash(FPFormat()) == hash(FPFormat(2, 1))
    assert f.replace(rounding=Rounding.RNE) == FPFormat(4, 3, ftz=True)
    assert FPFormat(4, 3).replace(exp_bits=5) == FPFormat(5, 3)          # default bias follows
    assert FPFormat(4, 3, bias=2).replace(exp_bits=5).bias == 2         # custom bias kept
    assert (f.size, f.emin, f.emax, f.max, f.min_normal, f.min_subnormal) == \
        (8, -6, 7, 240, Fraction(1, 64), Fraction(1, 512))       # IEEE-style E4M3
    assert str(FPFormat(4, 3, 10, False, wrap=True, ftz=True, rounding=Rounding.RTZ)) == \
        "ue4m3, bias=10, wrap, ftz, rtz"
    assert repr(f) == "FPFormat(4, 3, ftz=True, rounding=Rounding.RTZ)"
    assert FP.from_value(0.1, 4, 3).convert(FPFormat(2, 1)).format == FPFormat(2, 1)
    assert FP.from_raw(0x3c, fmt=FPFormat(5, 2)).exact == 1
    assert len(list(FP.all_values(FPFormat(3, 1)))) == 32
    with pytest.raises(TypeError):
        FP.from_value(1, FPFormat(2, 1), 3)
    with pytest.raises(ValueError):
        FPFormat(1, 1)
    with pytest.raises(ValueError):
        FPFormat.parse("fp8")


def test_fpformat_random(rng, iters):
    for _ in range(iters):
        f = rand_fmt(rng)
        ff = f.to_fpformat()
        assert FPFormat.parse(str(ff)) == ff and eval(repr(ff)) == ff
        vals = finite_grid(f)
        first = 1 if f.has_zero else 0          # code 0 is zero only if there is one
        assert ff.max == vals[-1] and ff.min_subnormal == vals[first]
        assert ff.min_normal == vals[(1 << f.M) if f.has_zero else 0]
        assert ff.size == f.raw(0).size
        assert list(ff.all_values())[-1].raw == (1 << ff.size) - 1



# New FPU operations on random formats (fn, unsigned, no-zero, custom bias,
# wrap, ftz, every NaN mode). IEEE binary16/32/64 are also checked against
# Berkeley TestFloat (test_testfloat.py) and x86 hardware (test_hwfpu.py).
def rand_operands(rng, f, k):
    return [rand_code(rng, *f.code, f=f) for _ in range(k)]


def test_fp_fma_random(rng, iters):
    for _ in range(iters):
        f = rand_fmt(rng)
        a, b, c = rand_operands(rng, f, 3)
        r, warned = caught(lambda: f.raw(a).fma(f.raw(b), f.raw(c)))
        want = ref_fma(ref_decode(a, f), ref_decode(b, f), ref_decode(c, f), f)
        assert (r.raw, r.flags, warned) == (want.raw, want.flags, want.warnings), \
            f"fma {a:#x} {b:#x} {c:#x} in {f}"


def test_fp_sqrt_random(rng, iters):
    for _ in range(iters):
        f = rand_fmt(rng)
        (a,) = rand_operands(rng, f, 1)
        try:
            want = ref_sqrt(ref_decode(a, f), f)
        except NoNaN:
            with pytest.raises(ValueError):
                f.raw(a).sqrt()
            continue
        r, warned = caught(lambda: f.raw(a).sqrt())
        assert (r.raw, r.flags, warned) == (want.raw, want.flags, want.warnings), \
            f"sqrt {a:#x} in {f}"


@pytest.mark.parametrize("name, pick_max, number", [
    ("minimum", False, False), ("maximum", True, False),
    ("minimum_number", False, True), ("maximum_number", True, True)])
def test_fp_minmax_random(rng, iters, name, pick_max, number):
    for _ in range(iters):
        f = rand_fmt(rng)
        a, b = rand_operands(rng, f, 2)
        r = getattr(f.raw(a), name)(f.raw(b))
        want = ref_minmax(a, b, f, pick_max, number)
        assert (r.raw, r.flags) == (want.raw, want.flags), f"{name} {a:#x} {b:#x} in {f}"


def test_fp_to_int_random(rng, iters):
    for _ in range(iters):
        f = rand_fmt(rng)
        (a,) = rand_operands(rng, f, 1)
        bits, signed = rng.randint(2, 16), rng.random() < 0.5
        mode, exact = rng.choice(ROUNDINGS), rng.random() < 0.5
        v, flags = f.raw(a).to_int(bits, signed, mode, exact)
        want = ref_to_int(ref_decode(a, f), f, bits, signed, mode, exact)
        assert (v.raw, flags) == want, f"to_int {a:#x} {bits} {signed} {mode} in {f}"


def test_fp_round_to_integral_random(rng, iters):
    for _ in range(iters):
        f = rand_fmt(rng)
        (a,) = rand_operands(rng, f, 1)
        mode, exact = rng.choice(ROUNDINGS), rng.random() < 0.5
        r, warned = caught(lambda: f.raw(a).round_to_integral(mode, exact))
        want = ref_round_int(ref_decode(a, f), f, mode, exact)
        assert (r.raw, r.flags) == (want.raw, want.flags), f"rint {a:#x} {mode} in {f}"


def test_fp_compare_random(rng, iters):
    for _ in range(iters):
        f = rand_fmt(rng)
        a, b = rand_operands(rng, f, 2)
        signaling = rng.random() < 0.5
        got = f.raw(a).compare(f.raw(b), signaling)
        assert got == ref_compare(ref_decode(a, f), ref_decode(b, f), signaling)


def test_fp_stochastic_ops(rng, iters):
    """Operations in an SR format draw sr_bits from the configured source."""
    from verifloat import set_sr_source
    try:
        for _ in range(iters // 4):
            f = rand_fmt(rng)._replace(rounding=Rounding.SR, sr_bits=rng.randint(1, 8))
            r = rng.getrandbits(f.sr_bits)
            set_sr_source(lambda n, r=r: r)
            a, b = rand_operands(rng, f, 2)
            A, B = ref_decode(a, f), ref_decode(b, f)
            if A.kind != "fin" or B.kind != "fin" or A.v * B.v == 0:
                continue      # specials and signed zeros don't round
            got, _ = caught(lambda: f.raw(a) * f.raw(b))
            x = A.v * B.v
            # Reference: SR as gfloat defines it (checked in test_external.py
            # for conversions): truncate, then round away when the discarded
            # fraction, rounded to sr_bits bits, plus r carries.
            want, _ = caught(lambda: FP.from_value(x, f.to_fpformat(), sr_rand=r))
            assert got.raw == want.raw and got.flags == want.flags
    finally:
        set_sr_source(0)


@pytest.mark.skipif(__import__("os").environ.get("VERIFLOAT_IMPL", "cpp") == "py",
                    reason="a 0.1 bug (fixed in the C++ core): see the docstring")
@pytest.mark.parametrize("M", [1, 2, 3])
@pytest.mark.parametrize("inf_nan", ["fn", False])
def test_one_exponent_bit_formats(M, inf_nan):
    """Formats with a single exponent bit: the subnormals and the only normal
    binade share one exponent, so in an 'fn' format the subnormal with an
    all-ones mantissa sits right below the NaN code and must not be taken for
    it (version 0.1 returned NaN with OVERFLOW for that exactly representable
    value). Every pair of codes in every rounding mode against the reference
    model, and the value itself."""
    for rounding in ROUNDINGS:
        for tininess in ("after", "before"):
            f = Fmt.make(1, M, inf_nan=inf_nan, rounding=rounding, tininess=tininess)
            for a in range(1 << (2 + M)):
                for b in range(1 << (2 + M)):
                    check_fp_ops(a, b, f)
    fmt = FPFormat(1, M, inf_nan=inf_nan)
    top_subnormal = fmt.from_raw((1 << M) - 1)
    again = fmt(top_subnormal.exact)
    assert (again.raw, again.flags) == (top_subnormal.raw, FPFlags(0))
    for raw in range(1 << (1 + M)):
        x = fmt.from_raw(raw)
        if x.is_finite:
            assert fmt(x.exact).raw == raw and fmt(float(x)).raw == raw and (x * 1).raw == raw, raw


def test_integer_square_root(rng, iters):
    """The core's integer square roots (64-bit, 128-bit, big) against
    Python's math.isqrt: perfect squares and their neighbours, one below the
    next square, powers of two, and random integers of every size."""
    import os
    if os.environ.get("VERIFLOAT_IMPL", "cpp") != "cpp":
        pytest.skip("part of the C++ core")
    from verifloat import _core
    cases = [0, 1, 2, 3, 4, (1 << 64) - 1, 1 << 64, (1 << 64) + 1, (1 << 128) - 1, 1 << 128, (1 << 128) + 1]
    for bits in [*range(1, 140), 190, 191, 192, 230, 255, 256, 257, 400, 601, 1000]:
        r = rng.getrandbits(bits) | (1 << (bits - 1))
        cases += [r * r, r * r + 1, r * r - 1, (r + 1) * (r + 1) - 1, r * r + r, r * r + 2 * r, 1 << bits,
                  (1 << bits) - 1, (1 << bits) + 1]
    for _ in range(iters):
        cases.append(rng.getrandbits(rng.choice([8, 31, 32, 53, 63, 64, 65, 100, 126, 127, 128, 129, 200, 300])))
    for n in cases:
        assert _core._isqrt(n) == math.isqrt(n), n
    with pytest.raises(ValueError):
        _core._isqrt(-1)


def test_float_of_a_binary_format_is_its_encoding():
    """float() of a binary64 value is that value's own bits (the sign of a
    NaN included), and of binary32 / binary16 the exact widening."""
    import os
    import struct
    from verifloat import FP64
    bits = lambda x: struct.unpack("<Q", struct.pack("<d", x))[0]      # noqa: E731
    for c in (0, 1, 2, (1 << 52) - 1, 1 << 52, (1 << 52) | 1, (1 << 53) - 1, 0x3FF0000000000000, 0x3FFFFFFFFFFFFFFF,
              0x4340000000000001, 0x7FEFFFFFFFFFFFFF, 0x7FF0000000000000):
        for code in (c, c | (1 << 63)):
            assert bits(float(FP64.from_raw(code))) == code, hex(code)
    for width, f, pack in ((32, FP32, "<If"), (16, FP16, "<He")):
        top = (1 << f.exp_bits) - 1
        for field in (0, 1, 2, top // 2, top - 1, top):
            for mant in (0, 1, f._mask >> 1, f._mask - 1, f._mask):
                for sign in (0, 1):
                    code = (sign << (width - 1)) | (field << f.mantissa_bits) | mant
                    want = struct.unpack(pack[0] + pack[2], struct.pack(pack[:2], code))[0]
                    got = float(f.from_raw(code))
                    if field == top and mant:
                        assert got != got                                # a NaN
                    else:
                        assert bits(got) == bits(want), hex(code)
    if os.environ.get("VERIFLOAT_IMPL", "cpp") == "cpp":
        for f in (FP16, FP32, FP64):
            nan = f.from_raw((f._top << f.mantissa_bits) | (1 << (f.mantissa_bits - 1)))
            assert bits(float(nan)) >> 63 == 0 and bits(float(-nan)) >> 63 == 1
