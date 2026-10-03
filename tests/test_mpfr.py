"""Floating point against GNU MPFR (through gmpy2).

MPFR computes correctly rounded results at any precision, with an exponent
range and subnormal emulation (``mpfr_subnormalize``) that reproduce an
IEEE-style format of any width. It is an implementation VeriFloat did not
write, and it covers what the other references cannot: formats that are
neither IEEE interchange formats nor OCP ones, including mantissas far wider
than 64 bits, with every result and flag checked in four rounding modes.

Compared: + - * / fma sqrt, rounding of exact rationals and of doubles,
conversion between formats; result (value, zero sign, inf, NaN) and the
inexact, overflow, underflow, invalid and divide-by-zero flags.

MPFR has no round-to-nearest-ties-away, so RNA is not covered here. Its
underflow flag follows tininess detected after rounding, and it is also raised
for exact subnormal results, where IEEE 754's default handling raises no flag;
the comparison takes that into account.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

gmpy2 = pytest.importorskip("gmpy2")

from conftest import COMPARED  # noqa: E402
from verifloat import FP16, FP32, FP64, FPFlags, FPFormat, Rounding  # noqa: E402

MODES = {Rounding.RNE: gmpy2.RoundToNearest, Rounding.RTZ: gmpy2.RoundToZero,
         Rounding.RUP: gmpy2.RoundUp, Rounding.RDN: gmpy2.RoundDown}
WIDTHS = [1, 2, 3, 4, 7, 10, 23, 31, 52, 61, 62, 63, 64, 65, 80, 112, 127, 128, 129, 236, 500]


def rand_format(rng, rounding) -> FPFormat:
    E = rng.randint(3, 15)
    M = rng.choice(WIDTHS) if rng.random() < 0.7 else rng.randint(1, 140)
    bias = None if rng.random() < 0.6 else (1 << (E - 1)) - 1 + rng.randint(-3, 3)
    return FPFormat(E, M, bias, rounding=rounding)


def context(f: FPFormat):
    """An MPFR context that behaves as the IEEE-style format f."""
    return gmpy2.context(precision=f.mantissa_bits + 1, round=MODES[f.rounding], emax=f.emax + 1,
                         emin=f.emin - f.mantissa_bits + 1, subnormalize=True)


def rand_code(rng, f: FPFormat) -> int:
    E, M, top = f.exp_bits, f.mantissa_bits, f._top
    sign = rng.getrandbits(1) << (E + M)
    r = rng.random()
    if r < 0.12:
        return sign | (rng.choice([0, top]) << M) | rng.choice([0, 0, 1, f._mask, 1 << (M - 1)])
    if r < 0.5:
        e = min(max(f.bias + rng.randint(-3, 3), 0), top - 1)
    elif r < 0.65:
        e = rng.randint(0, 2)
    elif r < 0.8:
        e = rng.randint(max(top - 3, 0), top - 1)
    else:
        e = rng.randint(0, top - 1)
    m = rng.choice([0, 1, f._mask, f._mask - 1]) if rng.random() < 0.15 else rng.getrandbits(M)
    return sign | (e << M) | m


def to_mpfr(x):
    """The exact value of an FP as an mpfr of the current context."""
    if x.is_nan:
        return gmpy2.nan()
    if x.is_inf:
        return gmpy2.inf(-1 if x.sign else 1)
    if x.is_zero:
        return gmpy2.zero(-1 if x.sign else 1)
    q = x.exact
    return gmpy2.mpfr(gmpy2.mpq(q.numerator, q.denominator))


def agree(got, want, ctx, what):
    """An FP result against an mpfr result and the context's flags."""
    if gmpy2.is_nan(want):
        assert got.is_nan, what
    elif gmpy2.is_infinite(want):
        assert got.is_inf and got.sign == (want < 0), what
    elif gmpy2.is_zero(want):
        assert got.is_zero and got.sign == gmpy2.is_signed(want), what
    else:
        q = gmpy2.mpq(want)
        assert got.is_finite and got.exact == Fraction(int(q.numerator), int(q.denominator)), what
    flags = FPFlags(0)
    for name, flag in (("inexact", FPFlags.INEXACT), ("overflow", FPFlags.OVERFLOW),
                       ("underflow", FPFlags.UNDERFLOW), ("invalid", FPFlags.INVALID),
                       ("divzero", FPFlags.DIVZERO)):
        if getattr(ctx, name):
            flags |= flag
    if not ctx.inexact:     # MPFR also flags exact subnormal results; IEEE's flag needs a loss
        flags &= ~FPFlags.UNDERFLOW
    assert got.flags == flags, (what, got.flags, flags)
    COMPARED["GNU MPFR (result and flags)"] += 1


OPS = {
    "add": (lambda a, b, c: a + b, lambda a, b, c: gmpy2.add(a, b)),
    "sub": (lambda a, b, c: a - b, lambda a, b, c: gmpy2.sub(a, b)),
    "mul": (lambda a, b, c: a * b, lambda a, b, c: gmpy2.mul(a, b)),
    "div": (lambda a, b, c: a / b, lambda a, b, c: gmpy2.div(a, b)),
    "fma": (lambda a, b, c: a.fma(b, c), lambda a, b, c: gmpy2.fma(a, b, c)),
    "sqrt": (lambda a, b, c: a.sqrt(), lambda a, b, c: gmpy2.sqrt(a)),
}


def check_op(f: FPFormat, op: str, ca: int, cb: int, cc: int) -> None:
    """One operation on raw codes of f against MPFR: result and flags."""
    ours, theirs = OPS[op]
    a, b, c = f.from_raw(ca), f.from_raw(cb), f.from_raw(cc)
    got = ours(a, b, c)
    with context(f) as ctx:
        x, y, z = to_mpfr(a), to_mpfr(b), to_mpfr(c)
        ctx.clear_flags()
        want = theirs(x, y, z)
        # MPFR raises its NaN flag whenever the result is NaN. IEEE's INVALID
        # is for signaling NaN operands and invalid operations, not for a
        # quiet NaN passing through.
        used = [a] if op == "sqrt" else [a, b, c] if op == "fma" else [a, b]
        if any(v.is_nan for v in used):
            ctx.invalid = any(v.is_snan for v in used)
            # fma(0, inf, qNaN): IEEE leaves INVALID to the implementation
            # (VeriFloat's default follows RISC-V and signals).
            if op == "fma" and ((a.is_zero and b.is_inf) or (a.is_inf and b.is_zero)):
                ctx.invalid = True
        agree(got, want, ctx, (op, f, hex(ca), hex(cb), hex(cc)))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("rounding", MODES, ids=lambda r: r.name)
def test_arithmetic(rng, iters, op, rounding):
    for _ in range(iters):
        f = rng.choice([FP16, FP32, FP64]).replace(rounding=rounding) if rng.random() < 0.15 \
            else rand_format(rng, rounding)
        ca, cb, cc = (rand_code(rng, f) for _ in range(3))
        if op == "sqrt" and rng.random() < 0.8:
            ca &= ~(1 << (f.exp_bits + f.mantissa_bits))
        if op in ("add", "sub", "fma") and rng.random() < 0.3:    # close exponents: cancellation
            cb = (ca ^ (rng.getrandbits(1) << (f.exp_bits + f.mantissa_bits))) ^ rng.choice([0, 1, 2, f._mask])
            cc = cb if rng.random() < 0.5 else cc
        check_op(f, op, ca, cb, cc)


@pytest.mark.parametrize("rounding", MODES, ids=lambda r: r.name)
def test_round_exact_numbers(rng, iters, rounding):
    """Rationals and doubles rounded into a format, around every boundary."""
    for _ in range(iters):
        f = rand_format(rng, rounding)
        kind = rng.randrange(5)
        if kind == 0:       # a neighbour's midpoint, nudged or not
            c = rand_code(rng, f) & ~(1 << (f.exp_bits + f.mantissa_bits))
            lo = f.from_raw(c)
            if not lo.is_finite:
                continue
            x = lo.exact + lo.ulp / 2 + rng.choice([0, 1, -1]) * Fraction(1, 1 << 900)
            x = rng.choice([1, -1]) * x
        elif kind == 1:     # near the overflow threshold
            x = f.max * (1 + Fraction(rng.randint(-3, 3), 1 << (f.mantissa_bits + 2)))
        elif kind == 2:     # in and below the subnormal range
            x = f.min_subnormal * Fraction(rng.randint(-40, 40), 16) + rng.choice([0, 1, -1]) * Fraction(1, 1 << 900)
        elif kind == 3:
            x = Fraction(rng.getrandbits(300) - (1 << 299), rng.getrandbits(250) + 1) \
                * Fraction(2) ** rng.randint(f.emin - 5, min(f.emax, f.emin + 400))
        else:
            x = rng.uniform(-1, 1) * 2.0 ** rng.randint(-300, 300)
        got = f(x)
        with context(f) as ctx:
            ctx.clear_flags()
            if isinstance(x, float):
                want = gmpy2.mpfr(x)
            else:
                want = gmpy2.mpfr(gmpy2.mpq(x.numerator, x.denominator))
            agree(got, want, ctx, (f, x))


@pytest.mark.parametrize("rounding", MODES, ids=lambda r: r.name)
def test_convert(rng, iters, rounding):
    for _ in range(iters):
        src, dst = rand_format(rng, Rounding.RNE), rand_format(rng, rounding)
        c = rand_code(rng, src)
        x = src.from_raw(c)
        if x.is_nan:
            continue        # NaN payloads are an IEEE matter; MPFR has one NaN
        got = x.convert(dst)
        with context(dst) as ctx:
            ctx.clear_flags()
            if x.is_inf:
                want = gmpy2.inf(-1 if x.sign else 1)
            elif x.is_zero:
                want = gmpy2.zero(-1 if x.sign else 1)
            else:
                q = x.exact
                want = gmpy2.mpfr(gmpy2.mpq(q.numerator, q.denominator))
            agree(got, want, ctx, (src, dst, hex(c)))


# Formats no standard defines, and binary128 and wider: there MPFR is the
# only outside reference for the designed cases.
DESIGNED = [FPFormat(3, 2), FPFormat(4, 3), FPFormat(4, 7), FPFormat(5, 10), FPFormat(6, 17), FPFormat(8, 23),
            FPFormat(7, 40, 60), FPFormat(11, 52), FPFormat(15, 64), FPFormat(15, 112), FPFormat(12, 150)]


def designed(f: FPFormat, op: str) -> list[tuple[int, int, int]]:
    import directed as D
    edges = D.edge_codes(f)
    if op in ("add", "sub"):
        return [(a, b, 0) for a, b in D.add_cases(f)] + [(a, b, 0) for a in edges for b in edges]
    if op == "mul":
        return [(a, b, 0) for a, b in D.mul_cases(f)] + [(a, b, 0) for a in edges for b in edges]
    if op == "div":
        return [(a, b, 0) for a, b in D.div_cases(f)] + [(a, b, 0) for a in edges for b in edges]
    if op == "sqrt":
        return [(a, 0, 0) for a in [*D.sqrt_cases(f), *edges]]
    few = D.few_edges(f)
    return D.fma_cases(f) + [(a, b, c) for a in few for b in few for c in few]


def part(cases: list, iters: int, seed_text: str) -> list:
    """All cases with VERIFLOAT_ITERS >= 20000, else an evenly spread part
    that changes with the seed."""
    want = len(cases) if iters >= 20000 else max(iters, 200) * 2
    if len(cases) <= want:
        return cases
    stride = len(cases) // want
    return cases[sum(seed_text.encode()) % stride::stride]


@pytest.mark.parametrize("f", DESIGNED, ids=str)
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("rounding", MODES, ids=lambda r: r.name)
def test_designed_cases(request, iters, f, op, rounding):
    """The designed cases of tests/directed.py (ties and their neighbours,
    the overflow threshold, the subnormal range, special values) in formats
    of any width, in MPFR's four rounding modes."""
    f = f.replace(rounding=rounding)
    seed = request.getfixturevalue("rng").random()
    for ca, cb, cc in part(designed(f, op), iters, repr(seed)):
        check_op(f, op, ca, cb, cc)


@pytest.mark.parametrize("src, dst", [(FPFormat(11, 52), FPFormat(4, 7)), (FPFormat(15, 112), FPFormat(6, 17)),
                                      (FPFormat(12, 150), FPFormat(15, 112)), (FPFormat(15, 112), FPFormat(15, 64)),
                                      (FPFormat(8, 23), FPFormat(3, 2)), (FPFormat(6, 17), FPFormat(7, 40, 60))],
                         ids=lambda f: str(f))
@pytest.mark.parametrize("rounding", MODES, ids=lambda r: r.name)
def test_designed_conversions(src, dst, rounding):
    """Values of one format on every rounding boundary of another."""
    import directed as D
    dst = dst.replace(rounding=rounding)
    for c in D.convert_cases(src, dst):
        x = src.from_raw(c)
        if x.is_nan:
            continue
        got = x.convert(dst)
        with context(dst) as ctx:
            ctx.clear_flags()
            want = to_mpfr(x)
            agree(got, want, ctx, (src, dst, hex(c)))
