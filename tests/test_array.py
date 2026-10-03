"""FPArray and the fast kernels against the scalar FP type.

An FPArray operation must give, element by element, exactly what the scalar
operation gives: the same code, flags, result format, warnings and errors.
The scalar type is itself checked against the pure-Python 0.1, the reference
model and the external references, so these tests carry that to the arrays
and to the fast kernels behind ``matmul``/``dot``:

* each array operation against a loop of scalar operations, over random
  formats in every mode, exhaustive small formats and IEEE edge values;
* each fast kernel against the general path (``_core.set_fast(False)``).
"""

from __future__ import annotations

import copy
import itertools
import operator
import os
import pickle
import warnings
from fractions import Fraction

import pytest

if os.environ.get("VERIFLOAT_IMPL", "cpp") != "cpp":
    pytest.skip("FPArray is part of the C++ package", allow_module_level=True)

import verifloat as vf
from verifloat import (BF16, E2M1, E4M3, E5M2, FP, FP16, FP32, FP64, Accumulator, FPArray, FPFlags,
                       FPFormat, NaNMode, Rounding, _core, dot, matmul)
from reference import ROUNDINGS, rand_code, rand_fmt

OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
ALL_ROUNDINGS = list(Rounding)


# Every way an operation can be computed: the general path (the scalar code
# on exact integers), the scalar fast kernels, and each SIMD instruction set
# this CPU has. All must give identical results.
FAST_LEVELS = ["scalar"] + list(_core.simd_available())
LEVELS = ["general"] + FAST_LEVELS


def use_level(level: str):
    """Select a level; returns what to pass to restore_level."""
    was = (_core.set_fast(level != "general"), _core.simd())
    _core.set_simd("none" if level in ("general", "scalar") else level)
    return was


def restore_level(was) -> None:
    _core.set_fast(was[0])
    _core.set_simd(was[1])


@pytest.fixture(params=LEVELS)
def kernels(request):
    """Run a test once per level."""
    was = use_level(request.param)
    yield request.param
    restore_level(was)


@pytest.fixture(params=FAST_LEVELS)
def fast_level(request):
    """Run a test once per fast level (it compares with the general path itself)."""
    was = use_level(request.param)
    yield request.param
    restore_level(was)


def outcome(fn):
    """(result or exception, warnings) of fn()."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vf.set_sr_source(99)
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001 - errors are compared too
            r = ("raise", type(e).__name__, str(e))
    return r, [(c.category.__name__, str(c.message)) for c in w]


def flat(x):
    return [v for row in x for v in flat(row)] if isinstance(x, list) else [x]


def pic(x):
    """Comparable picture of an array, or of a (nested) list of FP."""
    if isinstance(x, FPArray):
        vals = flat(x.tolist())
        assert flat(x.raw) == [v.raw for v in vals]
        assert all(v.format == x.format and v.unrounded is None for v in vals)
        assert int(x.flags) == sum_flags(vals), "flags must be the OR over the elements"
        return (str(x.format), x.shape, [(v.raw, int(v.flags)) for v in vals])
    if isinstance(x, list):
        vals = flat(x)
        return (str(vals[0].format), shape_of(x), [(v.raw, int(v.flags)) for v in vals])
    if isinstance(x, FP):
        return (str(x.format), x.raw, int(x.flags))
    return x


def sum_flags(vals) -> int:
    f = 0
    for v in vals:
        f |= int(v.flags)
    return f


def shape_of(x) -> tuple:
    return (len(x),) + shape_of(x[0]) if isinstance(x, list) else ()


def nested(flat_vals, shape):
    if len(shape) == 1:
        return list(flat_vals)
    step = len(flat_vals) // shape[0]
    return [nested(flat_vals[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def same(array_fn, scalar_fn, ctx=""):
    got, want = outcome(array_fn), outcome(scalar_fn)
    if isinstance(want[0], list) and flat(want[0])[0].format.size > 64:
        # Mixed formats promoted past 64 bits: an array cannot hold the result.
        assert got[0][:2] == ("raise", "ValueError") and "at most 64 bits" in got[0][2], ctx
        return
    if isinstance(got[0], tuple) and got[0][:2] == ("raise", "ValueError") and "at most 64 bits" in got[0][2]:
        # ...also when the scalar loop itself ends in an error (say a division
        # by zero part way): the array refuses the format before computing.
        assert FPFormat.parse(got[0][2].split(", not ", 1)[1]).size > 64 and isinstance(want[0], tuple), ctx
        return
    assert (pic(got[0]), got[1]) == (pic(want[0]), want[1]), ctx


def narrow_fmt(rng) -> FPFormat:
    """A random format of at most 64 bits, in any combination of modes: the
    small ones of the reference model half the time, else any widths."""
    if rng.random() < 0.5:
        return rand_fmt(rng).to_fpformat()
    while True:
        E = rng.randint(2, 12)
        M = rng.choice([0, 1, 2, 7, 10, 23, 24, 29, 30, 31, 32, 40, 51, 52, rng.randint(0, 61)])
        if E + M > 63:
            continue
        inf_nan = rng.choice([True, True, False, "fn"])
        bias = None if rng.random() < 0.6 else (1 << (E - 1)) - 1 + rng.randint(-3, 3)
        try:
            return FPFormat(E, M, bias, rng.random() < 0.8, inf_nan=inf_nan, has_zero=rng.random() < 0.9,
                            saturate=rng.random() < 0.2, wrap=rng.random() < 0.1, ftz=rng.random() < 0.2,
                            rounding=rng.choice(ALL_ROUNDINGS), sr_bits=rng.choice([1, 8, 20]),
                            nan_mode=rng.choice(list(NaNMode)), tininess=rng.choice(["after", "before"]))
        except ValueError:
            continue


def codes_of(rng, f: FPFormat, n: int) -> list[int]:
    """Raw codes biased towards the edges (specials, subnormals, near 1)."""
    out = []
    top, M, E = f._top, f.mantissa_bits, f.exp_bits
    for _ in range(n):
        r = rng.random()
        sign = (rng.getrandbits(1) << (E + M)) if f.signed else 0
        if r < 0.25:
            e = rng.choice([0, 1, top, max(top - 1, 0), min(max(f.bias, 0), top)])
            m = rng.choice([0, 1, f._mask, f._mask >> 1, (1 << M) >> 1])
            out.append(sign | (e << M) | (m & f._mask))
        elif r < 0.6:
            e = min(max(f.bias + rng.randint(-3, 3), 0), top)
            out.append(sign | (e << M) | (rng.getrandbits(M) if M else 0))
        else:
            out.append(rng.getrandbits(f.size))
    return out


IEEE = [FP16, BF16, FP32, FP64]


# ---------------------------------------------------------------- construction

def test_quantize_matches_scalar(rng, iters, kernels):
    numbers = [0.0, -0.0, 1.0, -1.5, 0.1, 1e-320, 5e-324, 1e308, -1e308, 65504.0, 65520.0, 2.0 ** -149,
               3.0e-45, float("inf"), -float("inf"), float("nan"), 7, -3, 2 ** 80, True,
               Fraction(1, 3), Fraction(-7, 1 << 70), FP16(0.1), FP32(-2.5), E4M3(448), FP64(1e300)]
    for _ in range(max(iters // 20, 30)):
        f = rng.choice(IEEE + [E2M1, E4M3, E5M2]) if rng.random() < 0.4 else narrow_fmt(rng)
        vals = [rng.choice(numbers) if rng.random() < 0.3 else
                rng.uniform(-1, 1) * 2.0 ** rng.randint(-f.bias - 4, f.bias + 4)
                if abs(f.bias) < 900 else rng.uniform(-8, 8) for _ in range(rng.randint(1, 24))]
        same(lambda: f.array(vals), lambda: [f(v) for v in vals], (f, vals))
        same(lambda: FPArray(vals, f), lambda: [f(v) for v in vals], (f, vals))
        raws = codes_of(rng, f, 12) + [-1, 1 << 70]
        same(lambda: FPArray.from_raw(raws, f), lambda: [f.from_raw(r) for r in raws], (f, raws))
        # FP values of the array's own format: every kind of code, with flags to drop
        own = [f.from_raw(c) for c in codes_of(rng, f, 24)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            own += [x * x for x in own[:6] if x.is_finite]     # values that carry flags
        same(lambda: f.array(own), lambda: [f(v) for v in own], (f, [v.raw for v in own]))
    for f in (FPFormat(5, 10, ftz=True), FPFormat(3, 2, ftz=True), FPFormat(4, 3, ftz=True, inf_nan="fn"),
              FPFormat(3, 2, ftz=True, signed=False), FPFormat(3, 2, ftz=True, has_zero=False, inf_nan="fn")):
        own = [f.from_raw(c) for c in range(min(1 << f.size, 1 << 12))]      # subnormals read as zero
        same(lambda: f.array(own), lambda: [f(v) for v in own], f)


def test_shapes_and_access(rng):
    vals = [[[rng.uniform(-4, 4) for _ in range(5)] for _ in range(3)] for _ in range(2)]
    a = FP16.array(vals)
    ref = [[[FP16(v) for v in row] for row in m] for m in vals]
    assert a.shape == (2, 3, 5) and a.ndim == 3 and len(a) == 2 and a.format == FP16
    assert a.raw == [[[v.raw for v in row] for row in m] for m in ref]
    assert a.to_float() == [[[float(v) for v in row] for row in m] for m in ref]
    assert repr(a) == "FPArray(shape=(2, 3, 5), [e5m10])"
    for i, j, k in itertools.product(range(-2, 2), range(-3, 3), range(-5, 5)):
        assert a[i, j, k].raw == ref[i][j][k].raw and a[i][j][k].flags == ref[i][j][k].flags
        assert isinstance(a[i, j, k], FP)
    assert a[1].shape == (3, 5) and a[1, 2].shape == (5,) and a[1, 2].raw == [v.raw for v in ref[1][2]]
    assert [row.raw for row in a] == a.raw
    assert a.T.shape == (5, 3, 2) and a.T.T == a
    assert a.transpose(1, 0, 2).raw == [[a.raw[i][j] for i in range(2)] for j in range(3)]
    assert a.transpose(0, -1, 1)[1, 4, 2].raw == a[1, 2, 4].raw
    assert a.reshape(6, 5).raw == [row for m in a.raw for row in m]
    assert a.reshape((30,)).raw == flat(a.raw)
    assert a == FPArray.from_raw(a.raw, FP16) and not (a != FPArray.from_raw(a.raw, FP16))
    assert a != FPArray.from_raw(a.raw, FP16.replace(rounding=Rounding.RTZ))
    assert a != a.reshape(6, 5) and a != a.raw and (a == 3) is False
    for clone in (pickle.loads(pickle.dumps(a)), copy.copy(a), copy.deepcopy(a)):
        assert clone == a and pic(clone) == pic(a)
    with pytest.raises(TypeError):
        hash(a)
    # Flags survive indexing, reshaping, transposing and pickling.
    q = FP16.array([0.1, 1.0, 1e9])
    assert [int(v.flags) for v in q.tolist()] == [1, 0, 5] and q.flags == FPFlags.INEXACT | FPFlags.OVERFLOW
    assert [int(v.flags) for v in pickle.loads(pickle.dumps(q)).reshape(3, 1).T.tolist()[0]] == [1, 0, 5]


@pytest.mark.parametrize("fn, exc, msg", [
    (lambda: FP16.array([]), ValueError, "empty tensor"),
    (lambda: FP16.array([[1.0, 2.0], [3.0]]), ValueError, "ragged"),
    (lambda: FP16.array([[1.0], 2.0]), ValueError, "ragged"),
    (lambda: FP16.array([1.0, [2.0]]), ValueError, "ragged"),
    (lambda: FP16.array(1.0), TypeError, "not a scalar"),
    (lambda: FPArray([1.0], 16), TypeError, "FPFormat"),
    (lambda: FPFormat(15, 112).array([1.0]), ValueError, "at most 64 bits"),
    (lambda: FP16.array([1.0, 2.0]) + FP16.array([1.0, 2.0, 3.0]), ValueError, "shape mismatch"),
    (lambda: FP16.array([1.0, 2.0])[2], IndexError, "out of range"),
    (lambda: FP16.array([1.0, 2.0])[-3], IndexError, "out of range"),
    (lambda: FP16.array([1.0, 2.0])[0, 0], IndexError, "too many indices"),
    (lambda: FP16.array([1.0, 2.0])["0"], TypeError, "integers"),
    (lambda: FP16.array([1.0, 2.0]) + "x", TypeError, "unsupported operand"),
    (lambda: FP16.array([1.0, 2.0]).reshape(3), ValueError, "cannot reshape"),
    (lambda: FP16.array([[1.0, 2.0]]).transpose(0, 0), ValueError, "permutation"),
    (lambda: FPArray.from_raw([1.5], FP16), TypeError, "ints"),
    (lambda: FPFormat(4, 3, inf_nan=False).array([float("nan")]), ValueError, "NaN is not representable"),
    (lambda: FPFormat(4, 3, inf_nan=False).array([1.0]) / FPFormat(4, 3, inf_nan=False).array([0.0]),
     ZeroDivisionError, "division by zero"),
])
def test_errors(fn, exc, msg):
    with pytest.raises(exc, match=msg):
        fn()


# ---------------------------------------------------------------- arithmetic

def check_binops(f: FPFormat, ca: list[int], cb: list[int], ctx=""):
    a, b = FPArray.from_raw(ca, f), FPArray.from_raw(cb, f)
    xs, ys = [f.from_raw(c) for c in ca], [f.from_raw(c) for c in cb]
    for name, op in OPS.items():
        same(lambda: op(a, b), lambda: [op(x, y) for x, y in zip(xs, ys)], (ctx, f, name))


@pytest.mark.parametrize("rounding", ALL_ROUNDINGS, ids=lambda r: r.name)
def test_exhaustive_small_formats(rounding, kernels):
    """Every pair of codes of 4- to 7-bit formats, in every mode."""
    bases = [FPFormat(2, 1), FPFormat(2, 2), FPFormat(3, 2), FPFormat(3, 3), FPFormat(2, 3, bias=3),
             FPFormat(3, 1, inf_nan="fn"), FPFormat(2, 2, inf_nan=False), FPFormat(3, 2, signed=False),
             FPFormat(3, 0, inf_nan="fn"), FPFormat(3, 1, has_zero=False, inf_nan=False),
             FPFormat(1, 3, inf_nan=False), FPFormat(1, 2, inf_nan="fn")]
    modes = [dict(), dict(saturate=True), dict(wrap=True), dict(ftz=True), dict(tininess="before"),
             dict(nan_mode=NaNMode.PROPAGATE), dict(nan_mode=NaNMode.X86), dict(nan_mode=NaNMode.ARM)]
    for base in bases:
        for mode in modes:
            f = base.replace(rounding=rounding, **mode)
            n = 1 << f.size
            pairs = list(itertools.product(range(n), repeat=2))
            check_binops(f, [p[0] for p in pairs], [p[1] for p in pairs])


@pytest.mark.parametrize("base", IEEE + [E4M3, E5M2, E2M1], ids=str)
def test_edge_pairs(base, kernels):
    """Every pair of edge encodings of the standard formats, in every rounding
    mode and underflow option."""
    E, M = base.exp_bits, base.mantissa_bits
    top, mask, one = base._top, base._mask, base.bias << M
    mags = {0, 1, mask, 1 << M, (1 << M) | 1, base._max_code, base._max_code - 1, one, one - 1, one + 1,
            (base.bias + 1) << M, max(base.bias - 1, 0) << M, (base.bias + M) << M, ((base.bias + M) << M) | 1,
            ((base.bias - M) << M) | mask if base.bias > M else 1,
            top << M, (top << M) | 1, (top << M) | (1 << (M - 1)), (top << M) | mask}
    mags = sorted(m for m in mags if 0 <= m < 1 << (E + M))
    codes = mags + [m | (1 << (E + M)) for m in mags]
    pairs = list(itertools.product(codes, repeat=2))
    for r in ALL_ROUNDINGS:
        for mode in (dict(), dict(ftz=True), dict(tininess="before"), dict(saturate=True),
                     dict(nan_mode=NaNMode.ARM)):
            f = base.replace(rounding=r, **mode)
            check_binops(f, [p[0] for p in pairs], [p[1] for p in pairs], (r, mode))


def test_random_formats(rng, iters, kernels):
    """Random formats in every mode: arrays with arrays, FP scalars, numbers
    and arrays of other formats."""
    for _ in range(max(iters // 10, 60)):
        f = rng.choice(IEEE + [E4M3]).replace(rounding=rng.choice(ALL_ROUNDINGS)) \
            if rng.random() < 0.4 else narrow_fmt(rng)
        g = f if rng.random() < 0.5 else narrow_fmt(rng)
        n = rng.randint(1, 40)
        ca, cb = codes_of(rng, f, n), codes_of(rng, g, n)
        a, b = FPArray.from_raw(ca, f), FPArray.from_raw(cb, g)
        xs, ys = [f.from_raw(c) for c in ca], [g.from_raw(c) for c in cb]
        s = g.from_raw(codes_of(rng, g, 1)[0])
        num = rng.choice([2, -1, 0, 0.5, -0.0, 0.1, 1e30, Fraction(1, 3), float("inf"), 3.0])
        for name, op in OPS.items():
            ctx = (f, g, name, ca, cb)
            same(lambda: op(a, b), lambda: [op(x, y) for x, y in zip(xs, ys)], ctx)
            same(lambda: op(a, s), lambda: [op(x, s) for x in xs], (ctx, s.raw))
            same(lambda: op(s, a), lambda: [op(s, x) for x in xs], (ctx, s.raw))
            same(lambda: op(a, num), lambda: [op(x, num) for x in xs], (ctx, num))
            same(lambda: op(num, a), lambda: [op(num, x) for x in xs], (ctx, num))
        same(lambda: -a, lambda: [-x for x in xs], f)
        same(lambda: +a, lambda: [+x for x in xs], f)
        same(lambda: abs(a), lambda: [abs(x) for x in xs], f)
        same(lambda: a.convert(g), lambda: [x.convert(g) for x in xs], (f, g, ca))
        same(lambda: FPArray(a, g), lambda: [x.convert(g) for x in xs], (f, g, ca))


def test_wide_values_and_division(rng, iters, kernels):
    """Operands across the whole exponent range of each IEEE format: every
    exponent gap in additions, near-total cancellation, and quotients whose
    remainders sit on rounding boundaries."""
    for _ in range(max(iters // 20, 40)):
        f = rng.choice(IEEE).replace(rounding=rng.choice(ALL_ROUNDINGS[:5]))
        E, M, top, mask = f.exp_bits, f.mantissa_bits, f._top, f._mask
        ca, cb = [], []
        for _ in range(64):
            ea = rng.randint(0, top - 1)
            eb = min(max(ea + rng.randint(-M - 6, M + 6), 0), top - 1)
            ma = rng.getrandbits(M)
            mb = rng.choice([ma, ma ^ 1, rng.getrandbits(M), (ma + 1) & mask, 0, mask])
            sa, sb = (rng.getrandbits(1) << (E + M) for _ in range(2))
            ca.append(sa | (ea << M) | ma)
            cb.append(sb | (eb << M) | mb)
        check_binops(f, ca, cb)


# ---------------------------------------------------------------- scalar fast paths

def scalar_snapshots(f: FPFormat, pairs, floats):
    """Results of every scalar operation that has a fast path, as plain data
    (code, flags and the unrounded value)."""
    snap = lambda r: (r.raw, int(r.flags), r.unrounded)  # noqa: E731
    out = []
    for i, j in pairs:
        a, b = f.from_raw(i), f.from_raw(j)
        for op in OPS.values():
            try:
                out.append(snap(op(a, b)))
            except (ValueError, ZeroDivisionError) as e:
                out.append(str(e))
        try:
            out.append((snap(a.fma(b, a)), snap(b.fma(a, b)), snap(a.fma(a, b))))
        except ValueError as e:
            out.append(str(e))
    for x in floats:
        try:
            out.append(snap(f(x)))
        except ValueError as e:
            out.append(str(e))
    return out


def check_scalar_paths(f: FPFormat, pairs, floats=()):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = {}
        for level in ("general", "scalar"):
            was = use_level(level)
            vf.set_sr_source(99)       # the same random bits on both paths
            try:
                results[level] = scalar_snapshots(f, pairs, floats)
            finally:
                restore_level(was)
    if results["scalar"] != results["general"]:
        k = next(i for i, (x, y) in enumerate(zip(results["scalar"], results["general"])) if x != y)
        raise AssertionError((f, k, results["scalar"][k], results["general"][k]))


FLOATS = [0.0, -0.0, 1.0, -1.5, 0.1, 1 / 3, 1e-320, 5e-324, 1e308, -1e308, 65504.0, 65519.99, 65520.0, 2.0 ** -149,
          2.0 ** -150, 3.0e-45, 448.0, 464.0, 480.0, 1e-5, 6e-8, 2.0 ** -24, 2.0 ** -25, float("inf"), float("nan")]


@pytest.mark.parametrize("rounding", ALL_ROUNDINGS, ids=lambda r: r.name)
def test_scalar_fast_paths_exhaustive(rounding):
    """a + b, a - b, a * b, a / b, fma and fmt(float) take a fast path for
    same-format finite operands. Every pair of codes of 4- to 7-bit formats, in
    every mode: the fast path against the general one, unrounded values
    included."""
    bases = [FPFormat(2, 1), FPFormat(2, 2), FPFormat(3, 2), FPFormat(3, 3), FPFormat(2, 3, bias=3),
             FPFormat(3, 1, inf_nan="fn"), FPFormat(2, 2, inf_nan=False), FPFormat(4, 2), FPFormat(3, 3, inf_nan="fn"),
             # one normal binade: the smallest normal exponent is also the largest
             FPFormat(1, 3, inf_nan=False), FPFormat(1, 2, inf_nan="fn"), FPFormat(1, 4, bias=1, inf_nan=False)]
    modes = [dict(), dict(saturate=True), dict(wrap=True), dict(ftz=True), dict(tininess="before"),
             dict(nan_mode=NaNMode.X86), dict(nan_mode=NaNMode.ARM)]
    for base in bases:
        for mode in modes:
            f = base.replace(rounding=rounding, **mode)
            n = 1 << f.size
            check_scalar_paths(f, list(itertools.product(range(n), repeat=2)), FLOATS)


def test_scalar_fast_paths_random(rng, iters):
    """...and on the standard and random wider formats: edge encodings, every
    exponent gap, near-total cancellation."""
    for _ in range(max(iters // 20, 40)):
        f = rng.choice(IEEE + [E4M3, E5M2]).replace(rounding=rng.choice(ALL_ROUNDINGS)) \
            if rng.random() < 0.5 else narrow_fmt(rng)
        E, M, top, mask = f.exp_bits, f.mantissa_bits, f._top, f._mask
        codes = codes_of(rng, f, 40)
        pairs = list(zip(codes, codes_of(rng, f, 40)))
        for _ in range(60):
            ea = rng.randint(0, top)
            eb = min(max(ea + rng.randint(-M - 6, M + 6), 0), top)
            ma = rng.getrandbits(M) if M else 0
            mb = rng.choice([ma, ma ^ 1, rng.getrandbits(M) if M else 0, (ma + 1) & mask, 0, mask])
            sa, sb = ((rng.getrandbits(1) << (E + M)) if f.signed else 0 for _ in range(2))
            pairs.append((sa | (ea << M) | ma, sb | (eb << M) | mb))
        floats = FLOATS + [rng.uniform(-1, 1) * 2.0 ** rng.randint(-f.bias - 3, f.bias + 3)
                           if abs(f.bias) < 900 else rng.uniform(-8, 8) for _ in range(40)]
        check_scalar_paths(f, pairs, floats)


# ---------------------------------------------------------------- matmul and dot

def accumulators(rng):
    f = rng.choice([FP32, FP16, BF16, FP64, E4M3, FPFormat(5, 10, rounding=rng.choice(ALL_ROUNDINGS[:5])),
                    FPFormat(8, 23, rounding=rng.choice(ALL_ROUNDINGS)), FPFormat(4, 3, saturate=True),
                    FPFormat(6, 9, ftz=True), FPFormat(8, 23, signed=False), FPFormat(11, 40, tininess="before")])
    p = rng.choice([None, None, FP16, FP32, BF16, FP64, E4M3, FPFormat(8, 23, rounding=Rounding.RTZ)])
    return rng.choice([f, Accumulator(f), Accumulator(f, "exact"), Accumulator(f, "pairwise"),
                       Accumulator(f, product=p), Accumulator(f, "pairwise", product=p),
                       Accumulator(f, "exact", product=p), Accumulator(f, group=rng.choice([2, 4])),
                       Accumulator(f, group=4, align_bits=rng.choice([0, 6, 20])),
                       Accumulator(f, "pairwise", product=p, group=3)])


def rand_matrix(rng, f: FPFormat, shape, style: int):
    """Codes for a matrix: 0 moderate values, 1 any code (specials included),
    2 wide exponent range, 3 sparse with signed zeros, 4 near the subnormals."""
    E, M, top = f.exp_bits, f.mantissa_bits, f._top
    n = 1
    for d in shape:
        n *= d
    out = []
    for _ in range(n):
        sign = (rng.getrandbits(1) << (E + M)) if f.signed else 0
        if style == 0:
            e = min(max(f.bias + rng.randint(-4, 4), 1), max(top - 1, 1))
        elif style == 1:
            out.append(rng.getrandbits(f.size))
            continue
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
    return nested(out, shape)


def test_matmul_array_vs_lists(rng, iters, kernels):
    """matmul on FPArrays against matmul on lists of FP, which the parity
    tests tie to the pure-Python implementation."""
    for it in range(max(iters // 10, 80)):
        fa = rng.choice([FP16, FP32, BF16, E4M3, E5M2, E2M1, FP64, FPFormat(6, 30), FPFormat(3, 2, ftz=True),
                         FPFormat(4, 3, signed=False, inf_nan="fn")])
        fb = fa if rng.random() < 0.6 else rng.choice([FP16, FP32, E4M3, FP64])
        k = rng.randint(1, 12)
        sa = rng.choice([(k,), (rng.randint(1, 5), k), (2, rng.randint(1, 3), k)])
        sb = rng.choice([(k,), (k, rng.randint(1, 5))] + ([(2, k, rng.randint(1, 3))] if len(sa) == 3 else []))
        transpose_b = len(sb) >= 2 and rng.random() < 0.2
        if transpose_b:
            sb = sb[:-2] + (sb[-1], sb[-2])
        style = rng.randrange(5)
        ra, rb = rand_matrix(rng, fa, sa, style), rand_matrix(rng, fb, sb, rng.choice([style, 0]))
        acc = accumulators(rng)
        a, b = FPArray.from_raw(ra, fa), FPArray.from_raw(rb, fb)
        la, lb = a.tolist(), b.tolist()
        ctx = (it, fa, fb, sa, sb, style, acc, ra, rb)
        same(lambda: matmul(a, b, acc=acc, transpose_b=transpose_b),
             lambda: matmul(la, lb, acc=acc, transpose_b=transpose_b), ctx)
        if len(sa) == 1 and len(sb) == 1:
            same(lambda: dot(a, b, acc), lambda: dot(la, lb, acc), ctx)
    a = FP16.array([[1.0, 2.0], [3.0, 4.0]])
    assert matmul(a, a) == matmul(a.tolist(), a.tolist())            # exact: Fractions
    assert isinstance(matmul(a[0], a[1], acc=FP32), FP) and isinstance(matmul(a, a, acc=FP32), FPArray)
    with pytest.raises(ValueError, match="inner dimensions differ"):
        matmul(FP16.array([[1.0, 2.0, 3.0]]), a, acc=FP32)
    with pytest.raises(ValueError, match="at most 64 bits"):
        matmul(a, a, acc=FPFormat(15, 112))


@pytest.mark.parametrize("order", ["sequential", "pairwise", "exact"])
@pytest.mark.parametrize("rounding", ROUNDINGS, ids=lambda r: r.name)
def test_matmul_fast_vs_general(rng, iters, order, rounding, fast_level):
    """The fast kernels (64- and 128-bit, the tight ones for narrow formats,
    scalar and SIMD) against the general path, on lists of FP and on arrays."""
    formats = [FP16, BF16, FP32, FP64, E4M3, E5M2, FPFormat(5, 2, ftz=True), FPFormat(8, 30), FPFormat(11, 40),
               FPFormat(4, 3, inf_nan=False), FPFormat(8, 23, saturate=True), FPFormat(8, 23, tininess="before"),
               FPFormat(6, 57), FPFormat(3, 60), FPFormat(10, 31), FPFormat(8, 23, bias=100),
               # few exponent bits: products and sums land on the edges of the range all the time
               FPFormat(4, 6, tininess="before"), FPFormat(3, 12, tininess="before"), FPFormat(4, 23),
               FPFormat(3, 10, inf_nan="fn"), FPFormat(4, 10, tininess="before", saturate=True),
               # ...and few mantissa bits: rounding often carries into the next binade, which at the
               # bottom of the range is where tininess before and after rounding part ways
               FPFormat(4, 1, tininess="before"), FPFormat(3, 2, tininess="before"), FPFormat(4, 2),
               FPFormat(5, 3, tininess="before"), FPFormat(4, 1)]
    for it in range(max(iters // 8, 100)):
        fa, fb = rng.choice(formats), rng.choice(formats)
        F = rng.choice(formats).replace(rounding=rounding)
        P = rng.choice([None, rng.choice(formats).replace(rounding=rng.choice(ROUNDINGS))])
        # one case in three sums the products in groups, half of those aligned first
        group = rng.choice([1, 1, 1, 1, 2, 3, 8])
        align = rng.choice([None, 0, 3, 12, 30, 70]) if group > 1 and rng.random() < 0.5 else None
        acc = Accumulator(F, order, product=P, group=group, align_bits=align)
        m, k, n = rng.randint(1, 4), rng.randint(1, 20), rng.randint(1, 9)
        style = rng.choice([0, 0, 2, 3, 4])
        ra, rb = rand_matrix(rng, fa, (m, k), style), rand_matrix(rng, fb, (k, n), rng.choice([style, 0]))
        if rng.random() < 0.3:      # exact cancellation: a column of b negated in the next
            for row in rb:
                if len(row) > 1:
                    row[1] = row[0] ^ (1 << (fb.exp_bits + fb.mantissa_bits))
        la = [[fa.from_raw(c) for c in row] for row in ra]
        lb = [[fb.from_raw(c) for c in row] for row in rb]
        a, b = FPArray.from_raw(ra, fa), FPArray.from_raw(rb, fb)
        if rng.random() < 0.25:     # exact numbers among the operands of the list form
            numbers = [0, 1, -3, 0.5, -0.0, 0.375, 1e-3, Fraction(-5, 8), Fraction(7, 1 << 20), 2 ** 40, 3.0e10]
            for row in la:
                for i in range(len(row)):
                    if row[i].is_finite and rng.random() < 0.5:
                        row[i] = rng.choice(numbers)
            a = None
        ctx = (it, fa, fb, acc, ra, rb)
        was = _core.set_fast(False)
        try:
            want = outcome(lambda: matmul(la, lb, acc=acc))
            want_dot = outcome(lambda: acc.sum_products(la[0], [row[0] for row in lb]))
            want_exact = outcome(lambda: (matmul(la, lb), dot(la[0], [row[0] for row in lb])))
        finally:
            _core.set_fast(was)
        # exact results (Fractions) come from the same kernels, unrounded
        assert outcome(lambda: (matmul(la, lb), dot(la[0], [row[0] for row in lb]))) == want_exact, ctx
        got = outcome(lambda: matmul(la, lb, acc=acc))
        assert (pic(got[0]), got[1]) == (pic(want[0]), want[1]), ctx
        if not isinstance(got[0], tuple):       # unrounded values survive the fast kernel
            assert [v.unrounded for v in flat(got[0])] == [v.unrounded for v in flat(want[0])], ctx
        if a is not None:
            got_arr = outcome(lambda: matmul(a, b, acc=acc))
            assert (pic(got_arr[0]), got_arr[1]) == (pic(want[0]), want[1]), ctx
        got_dot = outcome(lambda: acc.sum_products(la[0], [row[0] for row in lb]))
        assert (pic(got_dot[0]), got_dot[1]) == (pic(want_dot[0]), want_dot[1]), ctx
        if not isinstance(got_dot[0], tuple):
            assert (got_dot[0].raw, got_dot[0].flags, got_dot[0].unrounded) == \
                (want_dot[0].raw, want_dot[0].flags, want_dot[0].unrounded), ctx


def test_long_dot_products(rng, kernels):
    """Long sums: the interleaved lanes, lengths that are not a multiple of
    the lane count, and totals that overflow part way."""
    for f, acc_f in [(FP32, FP32), (FP16, FP16), (BF16, FP32), (E4M3, FP16), (FP32, FP64)]:
        for n in (1, 2, 3, 4, 5, 7, 8, 9, 63, 64, 65, 200):
            for cols in (1, 3, 4, 5, 9):
                ra = rand_matrix(rng, f, (2, n), 0)
                rb = rand_matrix(rng, f, (n, cols), 0)
                a, b = FPArray.from_raw(ra, f), FPArray.from_raw(rb, f)
                for acc in (Accumulator(acc_f, product=acc_f), Accumulator(acc_f), acc_f,
                            Accumulator(acc_f, "pairwise")):
                    same(lambda: matmul(a, b, acc=acc), lambda: matmul(a.tolist(), b.tolist(), acc=acc),
                         (f, acc, n, cols))
    big = FP16.array([[60000.0] * 40])
    for acc in (Accumulator(FP16), Accumulator(FP16, product=FP16), Accumulator(FP16, "pairwise"), FP16):
        r = matmul(big, big.T, acc=acc)
        assert r[0, 0].is_inf and FPFlags.OVERFLOW in r.flags
        assert pic(r) == pic(matmul(big.tolist(), big.T.tolist(), acc=acc))


@pytest.mark.parametrize("rounding", ROUNDINGS, ids=lambda r: r.name)
@pytest.mark.parametrize("tininess", ["after", "before"])
def test_kernel_range_edges(rounding, tininess):
    """Every product of a small operand format, as the second term of a sum,
    on every fast level against the general path: products and sums on each
    side of the smallest normal value and of the largest value, in each
    rounding mode. (A result just below the smallest normal that rounds up to
    it is where tininess before and after rounding differ.)"""
    O = FPFormat(4, 3)
    rows = [c for c in range(1 << 7) if O.from_raw(c).is_finite and c >> 3] + [0]
    cols = rows + [c | (1 << 7) for c in rows]
    for E, M, inf_nan in [(3, 1, True), (4, 1, True), (3, 2, "fn"), (4, 3, True), (3, 3, False), (5, 2, True)]:
        f = FPFormat(E, M, inf_nan=inf_nan, rounding=rounding, tininess=tininess)
        wide = FPFormat(8, 23, rounding=rounding)
        for first in (O(1.0), O(2.0 ** -6), O(-448.0)):
            a = FPArray.from_raw([[first.raw, c] for c in rows], O)
            b = FPArray.from_raw([[O(1.0).raw] * len(cols), cols], O)
            for acc in (Accumulator(wide, product=f), Accumulator(f), Accumulator(f, product=f),
                        Accumulator(f, "pairwise", product=f), Accumulator(f, "exact", product=f), f):
                results = {}
                for level in LEVELS:
                    was = use_level(level)
                    try:
                        r = outcome(lambda: matmul(a, b, acc=acc))
                    finally:
                        restore_level(was)
                    results[level] = (pic(r[0]), r[1])
                for level in FAST_LEVELS:
                    assert results[level] == results["general"], (level, f, first, acc)


def test_tininess_before_rounding_carry(fast_level):
    """1.875 * 2**-7 rounds up to the smallest normal of e4m1 (2**-6). With
    tininess detected before rounding that is an underflow; it must not be
    lost when the product is not the first term of the sum."""
    P = FPFormat(4, 1, tininess="before")
    acc = Accumulator(FP32, "sequential", product=P)
    a = [[FP16(1.0), FP16(1.5 * 2 ** -4)]]
    b = [[FP16(1.0)], [FP16(1.25 * 2 ** -3)]]
    r = matmul(a, b, acc=acc)[0][0]
    assert (float(r), r.flags) == (1.015625, FPFlags.INEXACT | FPFlags.UNDERFLOW)
    r = matmul(FPArray(a, FP16), FPArray(b, FP16), acc=acc)[0, 0]
    assert (float(r), r.flags) == (1.015625, FPFlags.INEXACT | FPFlags.UNDERFLOW)
    after = Accumulator(FP32, "sequential", product=P.replace(tininess="after"))
    assert matmul(a, b, acc=after)[0][0].flags == FPFlags.INEXACT


# ---------------------------------------------------------------- block quantizer

def test_block_quantize_fast_vs_general(rng, iters):
    """BlockTensor.quantize keeps dyadic values (floats, FP values, ints) on
    machine integers; the same tensors with that switched off must give the
    same element codes, scale codes, tensor scale and flags."""
    from verifloat import (E2M3, E3M2, MXFP4, MXINT8, NVFP4, NVFP4_MODELOPT, BlockFormat, BlockTensor, IntFormat,
                           Pow2Format)

    def formats():
        elem = rng.choice([E2M1, E2M3, E3M2, E4M3, E5M2, FP16, FPFormat(3, 2), FPFormat(4, 3, signed=False, inf_nan="fn"),
                           FPFormat(2, 1, inf_nan=False, rounding=rng.choice(ALL_ROUNDINGS)),
                           FPFormat(4, 3, inf_nan="fn", rounding=rng.choice(ALL_ROUNDINGS[:5])), IntFormat(8),
                           IntFormat(4, frac_bits=2)])
        scale = rng.choice([E4M3, FP16, FP32, FPFormat(4, 3, signed=False, inf_nan="fn"), Pow2Format(8, 127), None])
        kw = {}
        if rng.random() < 0.5:
            kw["tensor_scale"] = rng.choice([FP32, FP16])
        if rng.random() < 0.4:
            kw["compute"] = rng.choice([FP32, FP16, FP64])
        if rng.random() < 0.3 and isinstance(scale, FPFormat):
            kw["scale_min"] = Fraction(1, 512)
            kw["zero_scale"] = 1
        if isinstance(elem, IntFormat) and isinstance(scale, FPFormat) and rng.random() < 0.5:
            kw["zero_point"] = IntFormat(elem.bits, signed=elem.signed)
        return BlockFormat(elem, rng.choice([1, 4, 16, 32]), scale, **kw)

    def values(n):
        style = rng.randrange(5)
        out = []
        for _ in range(n):
            r = rng.random()
            if r < 0.1:
                out.append(rng.choice([0.0, -0.0, 0]))
            elif style == 0:
                out.append(rng.uniform(-8, 8))
            elif style == 1:
                out.append(rng.uniform(-1, 1) * 2.0 ** rng.randint(-30, 30))
            elif style == 2:
                out.append(rng.choice([FP16, FP32, E4M3])(rng.uniform(-4, 4)))
            elif style == 3:
                out.append(rng.choice([rng.randint(-20, 20), rng.uniform(-3, 3), Fraction(rng.randint(-40, 40), 16)]))
            else:       # one value that is not dyadic sends the tensor to the general path
                out.append(Fraction(rng.randint(-50, 50), rng.randint(1, 9)) if r < 0.2 else rng.uniform(-2, 2))
        return out

    def snapshot(t):
        return (t.elem_raw, t.scale_raw, t.zero_raw, None if t.tensor_scale is None else t.tensor_scale.raw,
                int(t.flags))

    def both_paths(bf, vals, ctx):
        results = {}
        for level in ("general", "scalar"):
            was = use_level(level)
            try:
                results[level] = outcome(lambda: snapshot(BlockTensor.quantize(vals, bf)))
            finally:
                restore_level(was)
        assert results["scalar"] == results["general"], ctx

    # Elements far below their block's scale: the quotient underflows in the
    # compute format (to a subnormal, or to zero, which then has no sign).
    tiny = [1.0, -1e-9, 1e-9, -1e-30, 3e-39, -3e-39, 1e-45, -1e-45, -6e-8, 6e-8, -3e-8, 0.0, -0.0, 2.9e-8, -2.9e-8, 0.5]
    for elem in (E2M1, E4M3, E5M2, FP16, FPFormat(3, 2), FPFormat(2, 1, inf_nan=False, rounding=Rounding.RDN)):
        for compute in (None, FP16, FP32, FP64, FPFormat(5, 10, rounding=Rounding.RDN), FPFormat(8, 23, rounding=Rounding.RUP)):
            for scale in (E4M3, FP32, None):
                for ts in (None, FP32):
                    bf = BlockFormat(elem, 16, scale, compute=compute, tensor_scale=ts)
                    both_paths(bf, tiny, (bf, "tiny"))
                    both_paths(bf, [-v for v in tiny], (bf, "-tiny"))

    for it in range(max(iters // 10, 100)):
        try:
            bf = rng.choice([NVFP4, NVFP4_MODELOPT, MXFP4, MXINT8]) if rng.random() < 0.4 else formats()
        except (ValueError, TypeError):
            continue
        shape = rng.choice([(rng.randint(1, 70),), (rng.randint(1, 4), rng.randint(1, 40))])
        vals = values(shape[0]) if len(shape) == 1 else [values(shape[1]) for _ in range(shape[0])]
        results = {}
        for level in ("general", "scalar"):
            was = use_level(level)
            try:
                r = outcome(lambda: snapshot(BlockTensor.quantize(vals, bf)))
            finally:
                restore_level(was)
            results[level] = r
        assert results["scalar"] == results["general"], (it, bf, vals)


# ---------------------------------------------------------------- the kernels are used

def test_kernel_selfcheck(fast_level, rng, iters):
    """The built-in check of every kernel against exact integer arithmetic
    (src/cpp/selfcheck.cpp): random formats, operands and lengths, each
    accumulation order, element-wise operations and conversions. A kernel may
    decline a case, but must not decline everything."""
    r = _core.kernel_selfcheck(rng.getrandbits(60), iters * 10)
    assert r["dots"] > iters * 30 and r["dots_on_kernel"] > r["dots"] // 5
    if fast_level != "scalar":
        assert r["elements_on_kernel"] > r["elements"] // 5


def test_ordinary_data_stays_on_the_fast_kernels(rng, fast_level):
    """Finite, normal data in ordinary formats must not fall back to the
    general path (a silent fallback would be correct but 20x slower)."""
    def run(fn):
        _core.fast_stats(True)
        fn()
        return _core.fast_stats(True)

    for f in (FP16, BF16, FP32, FP64, E5M2):
        vals = [rng.uniform(1, 2) * rng.choice([1, -1]) for _ in range(500)]
        hits, misses = run(lambda: f.array(vals))
        assert (hits, misses) == (500, 0), f
        a, b = f.array(vals), f.array(vals[::-1])
        for op in OPS.values():
            assert run(lambda: op(a, b)) == (500, 0), f
        assert run(lambda: a * f(1.5)) == (500, 0), f
        # (positive values: a sum of mixed signs can cancel down to a subnormal,
        # which the dot-product kernels rightly hand to the general path)
        m = f.array([[abs(v) for v in vals[i * 20:(i + 1) * 20]] for i in range(10)])
        for acc in (f, Accumulator(f), Accumulator(f, product=f), Accumulator(f, "pairwise"),
                    Accumulator(FP64, "exact", product=f)):
            assert run(lambda: matmul(m, m.T, acc=acc)) == (100, 0), (f, acc)
            assert run(lambda: matmul(m.tolist(), m.T.tolist(), acc=acc)) == (100, 0), (f, acc)
    # Subnormal results of single operations stay on the fast path too...
    sub, inf, one = FP16.array([1e-7] * 4), FP16.array([float("inf"), 1.0]), FP16.array([1.0, 1.0])
    assert run(lambda: FP16.array([1.0, 1e-7])) == (2, 0)
    assert run(lambda: sub * sub) == (4, 0) and run(lambda: sub + sub) == (4, 0)
    # ...and what the kernels cannot do is counted as handed on, not lost.
    big, large = FP16.array([[1e4, 1e4]]), FP16.array([1e4, 1.0])
    assert run(lambda: FP16.array([float("inf"), 1.0, 1e-7])) == (2, 1)
    assert run(lambda: inf + one) == (1, 1)
    assert run(lambda: large * large) == (1, 1)     # overflow
    assert run(lambda: matmul(big, big.T, acc=FP16)) == (0, 1)
