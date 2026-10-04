"""NumPy arrays (and other numeric buffers) in and out of FPArray.

A buffer must give exactly what the same numbers give as nested lists: the
same codes and the same per-element flags, whatever the dtype, memory layout
or kernel level. NumPy stays optional for the package; these tests need it.
"""

from __future__ import annotations

import array
import warnings

import pytest

np = pytest.importorskip("numpy")

from test_array import LEVELS, narrow_fmt, restore_level, use_level  # noqa: E402  (skips under VERIFLOAT_IMPL=py)
from verifloat import (BF16, E2M1, E4M3, FP16, FP32, FP64, NVFP4, Accumulator, BlockTensor, FPArray, FPFlags,  # noqa: E402
                       FPFormat, Rounding, dot, matmul, set_sr_source)


@pytest.fixture(params=LEVELS)
def kernels(request):
    was = use_level(request.param)
    yield request.param
    restore_level(was)


def picture(a: FPArray):
    """Codes and per-element flags, nested like the array."""
    def walk(x):
        return [walk(v) for v in x] if isinstance(x, list) else (x.raw, int(x.flags))
    return str(a.format), a.shape, walk(a.tolist())


def same_as_lists(values, fmt, ctx=""):
    """fmt.array(buffer) against fmt.array(buffer.tolist()): the same array,
    or the same error. Warnings are ignored on both sides (test_array.py
    compares them for lists)."""
    def run(v):
        set_sr_source(99)        # stochastic rounding: the same random bits for both
        try:
            return fmt.array(v)
        except (ValueError, TypeError, ZeroDivisionError) as e:
            return (type(e).__name__, str(e))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got, want = run(values), run(values.tolist())
    if isinstance(want, tuple) or isinstance(got, tuple):
        assert got == want, ctx
    else:
        assert picture(got) == picture(want), ctx
    return got


FLOATS = [0.0, -0.0, 1.0, -1.5, 0.1, 1 / 3, 1e-320, 5e-324, 1e308, -1e308, 65504.0, 65519.99, 65520.0, 2.0 ** -149,
          3.0e-45, 448.0, 464.0, 1e-5, 6e-8, 2.0 ** -24, float("inf"), -float("inf"), float("nan")]


def test_float_buffers_match_lists(rng, iters, kernels):
    for _ in range(max(iters // 20, 40)):
        f = rng.choice([FP16, BF16, FP32, FP64, E4M3, E2M1]).replace(rounding=rng.choice(list(Rounding)[:5])) \
            if rng.random() < 0.5 else narrow_fmt(rng)
        n = rng.randint(1, 70)
        vals = [rng.choice(FLOATS) if rng.random() < 0.3 else rng.uniform(-1, 1) * 2.0 ** rng.randint(-40, 40)
                for _ in range(n)]
        x = np.array(vals, dtype=np.float64)
        same_as_lists(x, f, (f, "float64"))
        with np.errstate(over="ignore", invalid="ignore"):
            same_as_lists(x.astype(np.float32), f, (f, "float32"))      # each narrower float widens exactly
            same_as_lists(x.astype(np.float16), f, (f, "float16"))


def test_layouts_and_dtypes(rng, kernels):
    base = np.array([[rng.uniform(-9, 9) for _ in range(7)] for _ in range(6)])
    for f in (FP16, FP32, E4M3, FPFormat(4, 3, signed=False, inf_nan="fn"), FPFormat(5, 10, rounding=Rounding.SR)):
        for view in (base, base.T, base[::-1], base[:, ::-2], base[1:5, 2:6], np.asfortranarray(base),
                     base.reshape(2, 3, 7), base.reshape(2, 3, 7).transpose(2, 0, 1), base.astype(">f8"),
                     base.astype("<f4"), np.ascontiguousarray(base.T)[::2, 1::3]):
            got = same_as_lists(view, f, (f, view.dtype, view.shape, view.strides))
            assert isinstance(got, FPArray) and got.shape == view.shape
        ro = base.copy()
        ro.setflags(write=False)
        same_as_lists(ro, f)
    ints = np.array([[-5, 0, 3], [127, -128, 1]])
    for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.bool_):
        same_as_lists(ints.astype(dtype), FP32, dtype)
        same_as_lists(ints.astype(dtype), FPFormat(4, 3, signed=False, inf_nan="fn"), dtype)
    big = np.array([2 ** 63, 2 ** 64 - 1, 0], dtype=np.uint64)
    same_as_lists(big, FP64)
    same_as_lists(np.array([-2 ** 63, 2 ** 63 - 1], dtype=np.int64), FP64)
    # other buffer providers, and array-likes that are not numeric buffers
    assert FP16.array(array.array("d", [1.0, 2.5, 0.1])) == FP16.array([1.0, 2.5, 0.1])
    assert FP16.array(array.array("i", [1, -2, 3])) == FP16.array([1, -2, 3])
    assert FP16.array(memoryview(np.array([[1.5, 2.5]]))) == FP16.array([[1.5, 2.5]])
    obj = np.array([1, 2.5, FP16(3), FP32(0.1)], dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert picture(FP16.array(obj)) == picture(FP16.array(obj.tolist()))
    assert FPArray(np.array([1.5, 2.5]), FP16) == FP16.array([1.5, 2.5])


def test_raw_codes_from_buffers(rng):
    for f in (FP16, FP32, FP64, E4M3, E2M1, FPFormat(4, 3, signed=False, inf_nan="fn"), FPFormat(11, 40)):
        codes = [rng.getrandbits(f.size) for _ in range(40)]
        want = FPArray.from_raw(codes, f)
        dtype = np.uint8 if f.size <= 8 else np.uint16 if f.size <= 16 else np.uint32 if f.size <= 32 else np.uint64
        x = np.array(codes, dtype=dtype)
        assert FPArray.from_raw(x, f) == want
        assert FPArray.from_raw(x.astype(np.uint64), f) == want
        assert FPArray.from_raw(x.reshape(5, 8).T, f) == FPArray.from_raw(x.reshape(5, 8).T.tolist(), f)
        # Codes that do not fit the format are refused, from a buffer as from a
        # list; a signed array exactly as wide as the format holds bit patterns.
        for bad in ([-1], [1 << f.size] if f.size < 64 else [1 << 64]):
            if f.size < 64:      # an int64 array is a 64-bit format's bit patterns
                with pytest.raises(ValueError, match="does not fit"):
                    FPArray.from_raw(np.array(bad, dtype=np.int64), f)
            with pytest.raises(ValueError, match="does not fit"):
                FPArray.from_raw(bad, f)
        if f.size in (8, 16, 32, 64):
            as_signed = x.view({8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}[f.size])
            assert FPArray.from_raw(as_signed, f) == want
    with pytest.raises(TypeError, match="ints"):
        FPArray.from_raw(np.array([1.5]), FP16)


def test_outputs(rng, kernels):
    for f in (FP16, FP32, FP64, E4M3, E2M1, FPFormat(6, 9, ftz=True), FPFormat(11, 40)):
        vals = [[rng.choice(FLOATS) if rng.random() < 0.3 else rng.uniform(-4, 4) for _ in range(5)] for _ in range(4)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not f.has_nan:
                vals = [[0.5 if v != v else v for v in row] for row in vals]
            a = f.array(vals)
        raw, flags, value = a.to_numpy("raw"), a.to_numpy("flags"), a.to_numpy()
        assert raw.shape == flags.shape == value.shape == (4, 5)
        assert raw.dtype == (np.uint8 if f.size <= 8 else np.uint16 if f.size <= 16 else np.uint32 if f.size <= 32
                             else np.uint64) and flags.dtype == np.uint8 and value.dtype == np.float64
        assert raw.tolist() == a.raw
        assert flags.tolist() == [[int(x.flags) for x in row] for row in a.tolist()]
        assert int(np.bitwise_or.reduce(flags, axis=None)) == int(a.flags)
        want = np.array(a.to_float())
        assert np.array_equal(value, want, equal_nan=True) and np.array_equal(np.signbit(value), np.signbit(want))
        # each output is a fresh, writable array; the round trip through raw codes is exact
        raw[0, 0] ^= 1
        assert a.to_numpy("raw")[0, 0] == raw[0, 0] ^ 1
        assert FPArray.from_raw(a.to_numpy("raw"), f) == a
    with pytest.raises(ValueError, match="what must be"):
        FP16.array([1.0]).to_numpy("codes")


@pytest.mark.parametrize("bad, exc, msg", [
    (lambda: FP16.array(np.float64(1.5)), TypeError, "not a scalar"),
    (lambda: FP16.array(np.array(2.0)), TypeError, "not a scalar"),
    (lambda: FP16.array(np.zeros((0, 3))), ValueError, "empty tensor"),
    (lambda: FP16.array(np.zeros((3, 0))), ValueError, "empty tensor"),
    (lambda: FPArray.from_raw(np.zeros((0,), dtype=np.uint8), FP16), ValueError, "empty tensor"),
    (lambda: FP16.array(np.array([1.5 + 2j])), TypeError, None),
    (lambda: FP16.array(np.array(["a", "b"])), (TypeError, ValueError), None),
    (lambda: FPFormat(15, 112).array(np.array([1.0])), ValueError, "at most 64 bits"),
    (lambda: FPFormat(4, 3, inf_nan=False).array(np.array([1.0, np.nan])), ValueError, "NaN is not representable"),
])
def test_errors(bad, exc, msg):
    with pytest.raises(exc, match=msg):
        bad()


def test_other_entry_points(rng, kernels):
    """Wherever nested lists are accepted, an array gives the same result."""
    a = np.array([[rng.uniform(-2, 2) for _ in range(8)] for _ in range(3)])
    b = np.array([[rng.uniform(-2, 2) for _ in range(4)] for _ in range(8)])
    for acc in (FP32, Accumulator(FP16), Accumulator(FP32, "pairwise", product=FP16), None):
        got, want = matmul(a, b, acc=acc), matmul(a.tolist(), b.tolist(), acc=acc)
        if acc is None:
            assert got == want
        else:
            assert [[(x.raw, x.flags) for x in row] for row in got] == [[(x.raw, x.flags) for x in row] for row in want]
        assert dot(a[0], b[:, 0], acc) == dot(a[0].tolist(), b[:, 0].tolist(), acc) or acc is not None
    x = np.array([[rng.uniform(-6, 6) for _ in range(32)] for _ in range(2)], dtype=np.float32)
    t, u = BlockTensor.quantize(x, NVFP4), BlockTensor.quantize(x.tolist(), NVFP4)
    assert t == u and t.flags == u.flags
    fa, fb = FP16.array(a), FP16.array(b)
    assert matmul(fa, fb, acc=FP32) == matmul(FP16.array(a.tolist()), FP16.array(b.tolist()), acc=FP32)
    assert FPArray.from_raw(fa.to_numpy("raw"), FP16) + 1.5 == fa + 1.5
    assert fa.flags == FPFlags(int(np.bitwise_or.reduce(fa.to_numpy("flags"), axis=None)))
