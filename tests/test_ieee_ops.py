"""The remaining IEEE 754 operations: remainder, fmod, next_up / next_down,
scaleb, logb, fclass and sign injection.

None of these exist in version 0.1, so there is no frozen model to compare
with. Each is checked against references VeriFloat did not compute:

* a model written here from the definitions, on exact Fractions and on the
  enumerated value grid of small formats (tests/reference.py), over every
  format option, with result code, flags, warnings and unrounded value;
* the host's C library through ``math`` and NumPy (IEEE double, single, half);
* GNU MPFR for wide and odd formats;
* Berkeley TestFloat / SoftFloat for the remainder.
"""

from __future__ import annotations

import itertools
import math
import os
import struct
import warnings
from fractions import Fraction

import pytest

if os.environ.get("VERIFLOAT_IMPL", "cpp") != "cpp":
    pytest.skip("these operations are part of the C++ package", allow_module_level=True)

from reference import (F, Fmt, NoNaN, Ref, assert_ref, caught, finite_grid, rand_code, rand_fmt,  # noqa: E402
                       ref_decode, ref_inf, ref_nan, ref_round)
from verifloat import (E4M3, FP16, FP32, FP64, FPFlags, FPFormat, FPFormatWarning, NaNMode,  # noqa: E402
                       Rounding, set_sr_source)


# ----------------------------------------------------------------- references

def ref_rem(A, B, f: Fmt, truncate: bool) -> Ref:
    """x - n*y, n = x/y rounded to nearest-even (IEEE remainder) or truncated (fmod)."""
    if A.kind == "nan" or B.kind == "nan":
        return ref_nan(f, A, B)
    if A.kind == "inf" or (B.kind == "fin" and B.v == 0):
        return ref_nan(f, invalid=True)
    if B.kind == "inf":
        return ref_round(A.v, f, A.sign)
    q = A.v / B.v
    n = math.trunc(q) if truncate else round(q)       # round(Fraction): ties to even
    return ref_round(A.v - n * B.v, f, A.sign)


def ref_next(code: int, f: Fmt, up: bool) -> Ref:
    """The neighbouring value, found by sorting every value the format has."""
    A = ref_decode(code, f)
    if A.kind == "nan":
        return ref_nan(f, A)
    plain = lambda raw: Ref(raw, F(0), set(), None)   # noqa: E731
    sbit = f.sbit
    if A.kind == "inf":
        away = up != A.sign
        return plain(code if away else (int(A.sign) * sbit | f.max_code))
    # every finite value with the code that holds it (a subnormal code of an
    # ftz format reads as zero, so it holds no value of its own)
    codes = {}
    for m, v in enumerate(finite_grid(f)):
        if f.ftz and f.has_zero and m >> f.M == 0 and m:
            continue
        codes[v] = m
        if f.signed and v:
            codes[-v] = sbit | m
    x = A.v
    beyond = [v for v in codes if (v > x if up else v < x)]
    if not beyond:
        if x == 0:
            return plain(0)                            # unsigned: nothing below zero
        if f.inf_nan is True and not f.saturate and (up != A.sign):
            return plain(int(A.sign) * sbit | (f.top << f.M))
        return plain(code)
    w = min(beyond) if up else max(beyond)
    if w == 0:
        return plain(int(A.sign and f.signed) * sbit)  # the zero keeps x's sign
    return plain(codes[w])


def ref_scaleb(code: int, n: int, f: Fmt) -> Ref:
    A = ref_decode(code, f)
    if A.kind == "nan":
        return ref_nan(f, A)
    if A.kind == "inf":
        return Ref(code, F(0), set(), None)
    if A.v == 0:
        return Ref(int(A.sign) * f.sbit, F(0), set(), None)
    return ref_round(A.v * Fraction(2) ** n, f, A.sign)


def floor_log2(x: Fraction) -> int:
    e = x.numerator.bit_length() - x.denominator.bit_length()
    while Fraction(2) ** e > x:
        e -= 1
    while Fraction(2) ** (e + 1) <= x:
        e += 1
    return e


def ref_logb(code: int, f: Fmt) -> Ref:
    A = ref_decode(code, f)
    if A.kind == "nan":
        return ref_nan(f, A)
    if A.kind == "inf":
        return Ref(f.top << f.M, F(0), set(), None)
    if A.v == 0:
        r = ref_inf(True, f)
        return r._replace(flags=r.flags | F.DIVZERO)
    return ref_round(Fraction(floor_log2(abs(A.v))), f, False)


def ref_fclass(code: int, f: Fmt) -> int:
    A = ref_decode(code, f)
    if A.kind == "nan":
        return 1 << (8 if A.snan else 9)
    if A.kind == "inf":
        return 1 << (0 if A.sign else 7)
    if A.v == 0:
        return 1 << (3 if A.sign else 4)
    if f.has_zero and (code & (f.sbit - 1)) >> f.M == 0:
        return 1 << (2 if A.sign else 5)
    return 1 << (1 if A.sign else 6)


def expect(fn, want_fn, ctx):
    """Run the operation and the reference; they agree on the result or on
    the error (a NaN or a zero the format cannot hold)."""
    try:
        want = want_fn()
    except NoNaN:
        with pytest.raises(ValueError):
            fn()
        return
    got, warned = caught(fn)
    assert_ref(got, warned, want, ctx=ctx)


SMALL = [
    Fmt.make(2, 1), Fmt.make(3, 2), Fmt.make(2, 3), Fmt.make(4, 3, inf_nan="fn"),
    Fmt.make(3, 2, signed=False), Fmt.make(3, 2, inf_nan=False), Fmt.make(3, 2, ftz=True),
    Fmt.make(3, 2, wrap=True), Fmt.make(3, 2, saturate=True), Fmt.make(3, 2, has_zero=False, inf_nan="fn"),
    Fmt.make(3, 0, inf_nan="fn"), Fmt.make(1, 2, inf_nan="fn"), Fmt.make(1, 3, inf_nan=False),
    Fmt.make(3, 2, signed=False, has_zero=False, inf_nan="fn"), Fmt.make(3, 2, 1, tininess="before"),
    Fmt.make(3, 3, nan_mode=NaNMode.X86), Fmt.make(3, 3, nan_mode=NaNMode.ARM),
    Fmt.make(3, 3, nan_mode=NaNMode.PROPAGATE), Fmt.make(3, 2, rounding=Rounding.RDN),
    Fmt.make(3, 2, rounding=Rounding.RUP, ftz=True), Fmt.make(2, 2, 5, rounding=Rounding.RTZ, wrap=True),
    Fmt.make(4, 2, -3, signed=False, saturate=True), Fmt.make(2, 0, inf_nan=False),
]


def all_codes(f: Fmt):
    return range(1 << (f.E + f.M + (1 if f.signed else 0)))


def ids(f):
    return str(f.to_fpformat()).replace(", ", "_")


# ----------------------------------------------------------------- remainder

@pytest.mark.parametrize("truncate", [False, True], ids=["remainder", "fmod"])
@pytest.mark.parametrize("f", SMALL, ids=ids)
def test_remainder_every_pair_of_a_small_format(f, truncate):
    op = "fmod" if truncate else "remainder"
    for a, b in itertools.product(all_codes(f), repeat=2):
        A, B = ref_decode(a, f), ref_decode(b, f)
        x, y = f.raw(a), f.raw(b)
        expect(lambda: getattr(x, op)(y), lambda: ref_rem(A, B, f, truncate),
               f"{a:#x} {op} {b:#x} in {f}")


@pytest.mark.parametrize("truncate", [False, True], ids=["remainder", "fmod"])
def test_remainder_random_formats(rng, iters, truncate):
    op = "fmod" if truncate else "remainder"
    for _ in range(iters):
        f = rand_fmt(rng)
        a, b = rand_code(rng, f.E, f.M, f.signed, f), rand_code(rng, f.E, f.M, f.signed, f)
        A, B = ref_decode(a, f), ref_decode(b, f)
        x, y = f.raw(a), f.raw(b)
        expect(lambda: getattr(x, op)(y), lambda: ref_rem(A, B, f, truncate),
               f"{a:#x} {op} {b:#x} in {f}")


def bits64(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def rand64(rng) -> int:
    r = rng.random()
    if r < 0.1:
        return rng.choice([0, 1, 2, (1 << 52) - 1, 1 << 52, 0x7FEFFFFFFFFFFFFF, 0x7FF0000000000000,
                           0x7FF8000000000000, 0x7FF0000000000001, 0x3FF0000000000000]) | rng.getrandbits(1) << 63
    if r < 0.4:     # near 1: quotients with few bits, so ties and sign changes happen
        return (rng.getrandbits(1) << 63) | ((1023 + rng.randint(-3, 3)) << 52) | (rng.getrandbits(6) << 46)
    return rng.getrandbits(64)


@pytest.mark.parametrize("truncate", [False, True], ids=["remainder", "fmod"])
def test_remainder_against_libm_double(rng, iters, truncate):
    theirs = math.fmod if truncate else math.remainder
    for _ in range(iters * 5):
        ca, cb = rand64(rng), rand64(rng)
        a, b = FP64.from_raw(ca), FP64.from_raw(cb)
        got = a.fmod(b) if truncate else a.remainder(b)
        x, y = float(a), float(b)
        invalid = a.is_snan or b.is_snan
        try:
            want = theirs(x, y)
        except ValueError:      # math raises for inf rem y and x rem 0
            want, invalid = math.nan, True
        if math.isnan(want):
            assert got.is_nan, (hex(ca), hex(cb))
        else:
            assert got.raw == bits64(want), (hex(ca), hex(cb), got, want)
            assert got.unrounded == Fraction(want)
        assert got.flags == (FPFlags.INVALID if invalid else FPFlags(0)), (hex(ca), hex(cb))


def test_remainder_against_numpy_single_and_half(rng, iters):
    np = pytest.importorskip("numpy")
    for fmt, dtype, utype in ((FP32, np.float32, np.uint32), (FP16, np.float16, np.uint16)):
        n = iters * 3
        ca = np.array([rng.getrandbits(fmt.size) for _ in range(n)], dtype=utype)
        cb = np.array([rng.getrandbits(fmt.size) for _ in range(n)], dtype=utype)
        if fmt is FP16:     # close exponents, as random halves rarely are
            cb = (cb & 0x83FF) | (ca & 0x7C00)
        x, y = ca.view(dtype), cb.view(dtype)
        with np.errstate(all="ignore"):
            want = np.fmod(x, y)            # C fmod on the dtype itself
        for i in range(n):
            a, b = fmt.from_raw(int(ca[i])), fmt.from_raw(int(cb[i]))
            got = a.fmod(b)
            if np.isnan(want[i]):
                assert got.is_nan
            else:
                assert got.raw == int(want[i:i + 1].view(utype)[0]), (fmt, hex(ca[i]), hex(cb[i]))
            # The IEEE remainder of two singles is a single, and so exact in double.
            r = a.remainder(b)
            try:
                w = math.remainder(float(x[i]), float(y[i]))
            except ValueError:
                w = math.nan
            if math.isnan(w):
                assert r.is_nan
            else:
                assert r.exact == Fraction(w) and r.sign == (math.copysign(1, w) < 0), (fmt, hex(ca[i]), hex(cb[i]))
                assert r.flags == FPFlags(0)


def test_remainder_against_mpfr(rng, iters):
    gmpy2 = pytest.importorskip("gmpy2")
    from test_mpfr import agree, context, rand_code as mp_code, rand_format, to_mpfr
    for _ in range(iters):
        f = rand_format(rng, Rounding.RNE)
        ca, cb = mp_code(rng, f), mp_code(rng, f)
        if rng.random() < 0.5:      # close exponents
            top = f._top << f.mantissa_bits
            cb = (cb & ~top) | (ca & top)
        a, b = f.from_raw(ca), f.from_raw(cb)
        for ours, theirs in ((a.remainder, gmpy2.remainder), (a.fmod, gmpy2.fmod)):
            got = ours(b)
            with context(f) as ctx:
                x, y = to_mpfr(a), to_mpfr(b)
                ctx.clear_flags()
                want = theirs(x, y)
                if a.is_nan or b.is_nan:
                    ctx.invalid = a.is_snan or b.is_snan
                agree(got, want, ctx, (f, hex(ca), hex(cb)))


def sig_exp(x):
    """A finite nonzero FP as (integer significand, exponent)."""
    f = x.format
    M = f.mantissa_bits
    field, mant = int(x.exp), int(x.mantissa)
    if field == 0 and f.has_zero:
        return mant, 1 - f.bias - M
    return mant | 1 << M, field - f.bias - M


def test_remainder_far_apart_exponents(rng):
    """The quotient can have thousands of bits (or far more); only its parity
    and the remainder are ever computed."""
    f = FPFormat(15, 70)
    big = f.max_value()
    for y in (f(3), f(7.5), f(Fraction(2**70 - 1, 2**69)), f.from_raw(1), f(1e-300)):
        for x in (big, -big, big.next_down(), f(Fraction(1, 3))):
            q = x.exact / y.exact
            for op, n in (("remainder", round(q)), ("fmod", math.trunc(q))):
                got = getattr(x, op)(y)
                assert got.exact == x.exact - n * y.exact, (op, x, y)
                assert got.flags == FPFlags(0)
    # |x| far below |y| comes back unchanged
    tiny = f.from_raw(5)
    assert tiny.remainder(big).raw == tiny.raw and tiny.fmod(-big).raw == tiny.raw

    # Exponents near 2**39: the values cannot be written out. Python's
    # modular power gives the remainder and the quotient's parity.
    g = FPFormat(40, 70)
    for _ in range(300):
        x, y = g.from_raw(rng.getrandbits(g.size)), g.from_raw(rng.getrandbits(g.size))
        if not (x.is_finite and y.is_finite) or x.is_zero or y.is_zero:
            continue
        (sx, ex), (sy, ey) = sig_exp(x), sig_exp(y)
        if ex < ey:
            x, y, sx, ex, sy, ey = y, x, sy, ey, sx, ex
        t = sx * pow(2, ex - ey, 2 * sy) % (2 * sy)          # |x| mod 2|y|, in units of 2**ey
        odd, r = divmod(t, sy)
        for op, want in (("fmod", r), ("remainder", r - sy if 2 * r > sy or (2 * r == sy and odd) else r)):
            got = getattr(x, op)(y)
            assert got.flags == FPFlags(0), (op, x, y)
            if want == 0:
                assert got.is_zero and got.sign == x.sign
                continue
            sg, eg = sig_exp(got)
            assert eg <= ey and sg == abs(want) << (ey - eg), (op, hex(x.raw), hex(y.raw))
            assert got.sign == (x.sign != (want < 0))


def test_exact_remainder_of_any_two_dyadics(rng, iters):
    """The kernel under remainder and fmod, on operands no single format
    gives it: any significand widths and any exponent gap."""
    from verifloat import _core
    for _ in range(iters * 3):
        xs = rng.getrandbits(rng.choice([1, 2, 5, 24, 53, 64, 70, 130])) * rng.choice([1, -1])
        ys = (rng.getrandbits(rng.choice([1, 2, 5, 24, 53, 64, 70, 130])) or 1) * rng.choice([1, -1])
        xe = rng.randint(-60, 60)
        ye = xe + rng.choice([0, 1, -1, 2, -2, rng.randint(-8, 8), rng.randint(-150, 150)])
        if rng.random() < 0.3:      # an exact tie: x = (n + 1/2) y
            n = rng.randint(-9, 9)
            xs, xe = (2 * n + 1) * ys, ye - 1
        x, y = xs * Fraction(2) ** xe, ys * Fraction(2) ** ye
        q = x / y
        for truncate, n in ((False, round(q)), (True, math.trunc(q))):
            rs, re = _core._rem_exact(xs, xe, ys, ye, truncate)
            assert rs * Fraction(2) ** re == x - n * y, (xs, xe, ys, ye, truncate)
    with pytest.raises(ZeroDivisionError):
        _core._rem_exact(1, 0, 0, 0, False)


def test_remainder_operands():
    a, b = FP32(5.5), FP32(2.0)
    assert float(a.remainder(b)) == -0.5 and float(a.fmod(b)) == 1.5
    assert float(a.remainder(2)) == -0.5 and float(a.remainder(other=2.0)) == -0.5
    assert float(a.fmod(Fraction(3, 2))) == 1.0
    assert a.remainder(b).unrounded == Fraction(-1, 2)
    # zero results take the sign of the dividend
    assert FP32(-4.0).remainder(b).raw == 0x80000000 and FP32(4.0).fmod(-b).raw == 0
    # mixed formats are promoted like any binary operation
    with pytest.warns(FPFormatWarning):
        r = FP16(5.5).remainder(b)
    assert r.format == FP32 and float(r) == -0.5
    with pytest.raises(TypeError, match="needs an FP or a number"):
        a.remainder("2")
    with pytest.raises(TypeError):
        a.fmod()
    # a format without NaN cannot answer x rem 0
    g = FPFormat(3, 2, inf_nan=False)
    with pytest.raises(ValueError, match="NaN is not representable"):
        g(1).remainder(g(0))


# ----------------------------------------------------------------- next_up / next_down

@pytest.mark.parametrize("f", SMALL, ids=ids)
def test_next_every_code_of_a_small_format(f):
    for c in all_codes(f):
        x = f.raw(c)
        expect(x.next_up, lambda: ref_next(c, f, True), f"next_up {c:#x} in {f}")
        expect(x.next_down, lambda: ref_next(c, f, False), f"next_down {c:#x} in {f}")


def test_next_random_formats(rng, iters):
    for _ in range(iters):
        f = rand_fmt(rng)
        c = rand_code(rng, f.E, f.M, f.signed, f)
        x = f.raw(c)
        expect(x.next_up, lambda: ref_next(c, f, True), f"next_up {c:#x} in {f}")
        expect(x.next_down, lambda: ref_next(c, f, False), f"next_down {c:#x} in {f}")


def test_next_every_half_against_numpy():
    np = pytest.importorskip("numpy")
    codes = np.arange(1 << 16, dtype=np.uint16)
    x = codes.view(np.float16)
    with np.errstate(all="ignore"):
        up = np.nextafter(x, np.float16(np.inf)).view(np.uint16)
        down = np.nextafter(x, np.float16(-np.inf)).view(np.uint16)
    for c in range(1 << 16):
        v = FP16.from_raw(c)
        u, d = v.next_up(), v.next_down()
        if v.is_nan:
            assert u.is_nan and d.is_nan
            assert u.flags == d.flags == (FPFlags.INVALID if v.is_snan else FPFlags(0))
        else:
            assert (u.raw, d.raw) == (int(up[c]), int(down[c])), hex(c)
            assert u.flags == d.flags == FPFlags(0) and u.unrounded is None


def test_next_against_libm(rng, iters):
    np = pytest.importorskip("numpy")
    for _ in range(iters * 5):
        c = rand64(rng)
        v = FP64.from_raw(c)
        if not v.is_nan:
            assert v.next_up().raw == bits64(math.nextafter(float(v), math.inf)), hex(c)
            assert v.next_down().raw == bits64(math.nextafter(float(v), -math.inf)), hex(c)
        c = rng.getrandbits(32) if rng.random() < 0.8 else rng.choice([0, 1, 0x7F7FFFFF, 0x7F800000, 0x007FFFFF, 0x00800000]) | rng.getrandbits(1) << 31
        v = FP32.from_raw(c)
        if not v.is_nan:
            x = np.array([c], dtype=np.uint32).view(np.float32)
            with np.errstate(all="ignore"):
                up = np.nextafter(x, np.float32(np.inf)).view(np.uint32)
                down = np.nextafter(x, np.float32(-np.inf)).view(np.uint32)
            assert (v.next_up().raw, v.next_down().raw) == (int(up[0]), int(down[0])), hex(c)


def test_next_wide_formats(rng, iters):
    """Mantissas past 64 bits take the other storage: the neighbour is one
    ulp away, in the right direction, and the two operations invert."""
    for _ in range(iters // 4):
        f = FPFormat(rng.randint(3, 15), rng.choice([64, 65, 80, 112, 128, 200]))
        size = f.size
        r = rng.random()
        c = rng.getrandbits(size)
        if r < 0.3:     # all-ones or all-zeros mantissas: the carry and the borrow
            c = (c >> f.mantissa_bits << f.mantissa_bits) | rng.choice([0, f._mask, 1, f._mask - 1])
        x = f.from_raw(c)
        if not x.is_finite:
            continue
        u, d = x.next_up(), x.next_down()
        for y, up in ((u, True), (d, False)):
            if y.is_inf:
                assert abs(x.exact) == f.max and y.sign == (not up)
                continue
            gap = y.exact - x.exact
            assert (gap > 0) == up and gap != 0
            assert abs(gap) == min(x.ulp, y.ulp), (f, hex(c), up)
            back = y.next_down() if up else y.next_up()
            assert back.exact == x.exact
        assert u.flags == d.flags == FPFlags(0)


def test_next_special_cases():
    assert FP32(0.0).next_up().raw == 1 and FP32(-0.0).next_up().raw == 1
    assert FP32(0.0).next_down().raw == 0x80000001 and FP32(-0.0).next_down().raw == 0x80000001
    assert FP32.from_raw(0x80000001).next_up().raw == 0x80000000       # -min -> -0
    assert FP32.from_raw(1).next_down().raw == 0                        # +min -> +0
    inf, ninf = FP32.inf(), FP32.inf(True)
    assert inf.next_up().raw == inf.raw and ninf.next_down().raw == ninf.raw
    assert inf.next_down().raw == 0x7F7FFFFF and ninf.next_up().raw == 0xFF7FFFFF
    assert FP32.from_raw(0x7F7FFFFF).next_up().raw == inf.raw
    snan = FP32.from_raw(0x7F800001)
    assert snan.next_up().is_nan and snan.next_up().flags == FPFlags.INVALID
    # no infinity to step to: the largest value stays
    top = E4M3.max_value()
    assert top.next_up().raw == top.raw and (-top).next_down().raw == (-top).raw
    # an unsigned format has nothing below zero
    u = FPFormat(4, 3, signed=False)
    assert u(0).next_down().raw == 0 and u(0).next_up().raw == 1


# ----------------------------------------------------------------- scaleb / logb

@pytest.mark.parametrize("f", SMALL, ids=ids)
def test_scaleb_logb_every_code_of_a_small_format(f):
    span = (1 << f.E) + f.M + 3
    shifts = sorted({0, 1, -1, 2, -2, span, -span, 2 * span + 1, -2 * span - 1, 100, -100, 517, -517,
                     *range(-span, span, 3)})
    for c in all_codes(f):
        x = f.raw(c)
        expect(x.logb, lambda: ref_logb(c, f), f"logb {c:#x} in {f}")
        for n in shifts:
            expect(lambda: x.scaleb(n), lambda: ref_scaleb(c, n, f), f"scaleb({c:#x}, {n}) in {f}")


def test_scaleb_logb_random_formats(rng, iters):
    for _ in range(iters):
        f = rand_fmt(rng)
        c = rand_code(rng, f.E, f.M, f.signed, f)
        x = f.raw(c)
        n = rng.choice([rng.randint(-8, 8), rng.randint(-80, 80), rng.randint(-5000, 5000)])
        expect(lambda: x.scaleb(n), lambda: ref_scaleb(c, n, f), f"scaleb({c:#x}, {n}) in {f}")
        expect(x.logb, lambda: ref_logb(c, f), f"logb {c:#x} in {f}")


def test_scaleb_is_the_rounding_of_the_exact_product(rng, iters):
    """Any format and rounding mode: scaleb(x, n) is fmt(x * 2**n), flags included."""
    for _ in range(iters):
        E, M = rng.randint(2, 12), rng.choice([1, 3, 10, 23, 52, 63, 64, 65, 100])
        f = FPFormat(E, M, rounding=rng.choice(list(Rounding)[:5]), tininess=rng.choice(["before", "after"]),
                     ftz=rng.random() < 0.2, saturate=rng.random() < 0.2)
        x = f.from_raw(rng.getrandbits(f.size))
        if not x.is_finite or x.is_zero:
            continue
        span = f.emax - f.emin + M
        n = rng.choice([rng.randint(-3, 3), rng.randint(-span - 5, span + 5), rng.randint(-3 * span, 3 * span)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got, want = x.scaleb(n), f(x.exact * Fraction(2) ** n)
        assert (got.raw, got.flags) == (want.raw, want.flags), (f, x, n)
        assert got.unrounded == x.exact * Fraction(2) ** n


def test_scaleb_stochastic_rounding(rng, iters):
    f = FPFormat(4, 3, rounding=Rounding.SR, sr_bits=6)
    try:
        for _ in range(iters):
            x = f.from_raw(rng.getrandbits(f.size))
            if not x.is_finite or x.is_zero:
                continue
            n, k = rng.randint(-25, 12), rng.getrandbits(6)
            set_sr_source(lambda bits: k)
            got = x.scaleb(n)
            want = f(x.exact * Fraction(2) ** n, sr_rand=k)
            assert (got.raw, got.flags) == (want.raw, want.flags), (x, n, k)
    finally:
        set_sr_source(None)


def test_scaleb_against_libm(rng, iters):
    np = pytest.importorskip("numpy")
    for _ in range(iters * 3):
        c = rand64(rng)
        v = FP64.from_raw(c)
        n = rng.choice([rng.randint(-5, 5), rng.randint(-1100, 1100), rng.randint(-2200, 2200)])
        got = v.scaleb(n)
        if v.is_nan:
            assert got.is_nan
        else:
            try:
                want = math.ldexp(float(v), n)
            except OverflowError:
                want = math.copysign(math.inf, float(v))
            assert got.raw == bits64(want), (hex(c), n)
        c = rng.getrandbits(32)
        v = FP32.from_raw(c)
        n = rng.choice([rng.randint(-5, 5), rng.randint(-160, 160), rng.randint(-300, 300)])
        got = v.scaleb(n)
        if not v.is_nan:
            with np.errstate(all="ignore"):
                want = np.ldexp(np.array([c], dtype=np.uint32).view(np.float32), n)
            assert got.raw == int(want.view(np.uint32)[0]), (hex(c), n)


def test_scaleb_huge_exponents():
    one = FP32(1.0)
    r = one.scaleb(1 << 62)
    assert r.is_inf and r.flags == FPFlags.OVERFLOW | FPFlags.INEXACT
    r = (-one).scaleb(-(1 << 62))
    assert r.raw == 0x80000000 and r.flags == FPFlags.UNDERFLOW | FPFlags.INEXACT
    with pytest.raises(OverflowError):
        one.scaleb((1 << 62) + 1)
    with pytest.raises(TypeError):
        one.scaleb(1.0)
    with pytest.raises(TypeError):
        one.scaleb(True)
    assert one.scaleb(n=3) == 8


def test_logb_against_libm(rng, iters):
    for _ in range(iters * 3):
        c = rand64(rng)
        v = FP64.from_raw(c)
        got = v.logb()
        if v.is_nan:
            assert got.is_nan
        elif v.is_inf:
            assert got.is_inf and not got.sign and got.flags == FPFlags(0)
        elif v.is_zero:
            assert got.is_inf and got.sign and got.flags == FPFlags.DIVZERO
        else:
            m, e = math.frexp(abs(float(v)))           # |v| = m * 2**e, 0.5 <= m < 1
            assert got == e - 1 and got.flags == FPFlags(0), hex(c)
            assert got.unrounded == e - 1


# ----------------------------------------------------------------- fclass, sign injection

@pytest.mark.parametrize("f", SMALL, ids=ids)
def test_fclass_every_code_of_a_small_format(f):
    for c in all_codes(f):
        x = f.raw(c)
        k = x.fclass()
        assert k == ref_fclass(c, f), (hex(c), f)
        assert x.is_normal == bool(k & 0b0001000010)
        assert x.is_nan == bool(k & 0b1100000000) and x.is_snan == bool(k & 0b0100000000)
        assert x.is_inf == bool(k & 0b0010000001) and x.is_zero == bool(k & 0b0000011000)


def test_fclass_against_numpy():
    np = pytest.importorskip("numpy")
    codes = np.arange(1 << 16, dtype=np.uint16)
    x = codes.view(np.float16)
    tiny = np.finfo(np.float16).tiny
    with np.errstate(all="ignore"):
        neg = np.signbit(x)
        kind = np.where(np.isnan(x), 9, np.where(np.isinf(x), 7, np.where(x == 0, 4,
                        np.where(np.abs(x) < tiny, 5, 6))))
    for c in range(1 << 16):
        k = FP16.from_raw(c).fclass()
        if kind[c] == 9:
            assert k == (1 << 9 if c & 0x0200 else 1 << 8)
        else:
            assert k == 1 << (7 - int(kind[c]) if neg[c] else int(kind[c])), hex(c)


@pytest.mark.parametrize("f", SMALL, ids=ids)
def test_sign_injection_every_pair_of_a_small_format(f):
    mag = f.sbit - 1
    for a, b in itertools.product(all_codes(f), repeat=2):
        x, y = f.raw(a), f.raw(b)
        sx, sy = bool(a & f.sbit), bool(b & f.sbit)
        for name, s in (("copysign", sy), ("fsgnj", sy), ("fsgnjn", not sy), ("fsgnjx", sx != sy)):
            if s and not f.signed:
                with pytest.raises(ValueError, match="unsigned FP cannot have its sign set"):
                    getattr(x, name)(y)
                continue
            r = getattr(x, name)(y)
            assert r.raw == (a & mag) | (f.sbit if s else 0), (name, hex(a), hex(b), f)
            assert r.flags == FPFlags(0) and r.unrounded is None and r.format == x.format


def test_sign_injection_operands():
    x = FP32(1.5)
    assert x.copysign(-0.0).raw == 0xBFC00000 and x.copysign(0.0).raw == x.raw
    assert x.copysign(-3).sign and not x.copysign(7).sign and x.copysign(Fraction(-1, 3)).sign
    assert x.copysign(FP16(-2)).sign and x.copysign(other=E4M3(-1)).format == FP32
    assert x.fsgnjn(x).raw == 0xBFC00000 and x.fsgnjx(-x).sign and not (-x).fsgnjx(-x).sign
    # bits only: a signaling NaN stays signaling and raises nothing
    snan = FP32.from_raw(0x7F800001)
    r = snan.copysign(-1.0)
    assert r.raw == 0xFF800001 and r.flags == FPFlags(0)
    with pytest.raises(TypeError, match="needs an FP or a number"):
        x.copysign("-")


# ----------------------------------------------------------------- TestFloat

def test_remainder_against_testfloat():
    """SoftFloat's remainder under each of its NaN conventions."""
    try:
        import testfloat_build
        from conftest import COMPARED
        from test_testfloat import FORMATS, SPECS, cases
        gens = {spec: testfloat_build.testfloat_gen(spec) for spec in SPECS}
    except Exception as e:      # no network / compiler: skip, don't fail
        pytest.skip(f"TestFloat unavailable: {e}")
    for spec, mode in SPECS.items():
        for fname, base in FORMATS.items():
            fmt = base.replace(nan_mode=mode)
            for a, b, r, fl in cases(gens[spec], f"{fname}_rem", n=4000):
                x = fmt.from_raw(int(a, 16)).remainder(fmt.from_raw(int(b, 16)))
                assert (x.raw, int(x.flags)) == (int(r, 16), int(fl, 16)), (spec, fname, a, b)
                COMPARED["Berkeley TestFloat (result and flags)"] += 1
