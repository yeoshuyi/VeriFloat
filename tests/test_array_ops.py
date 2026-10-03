"""FPArray indexing, broadcasting, element-wise methods, sums and constructors.

The rule is the one of tests/test_array.py: an array operation gives,
element by element, exactly what the scalar FP operation gives (code, flags,
format, warnings, errors). Which element goes where is checked against
NumPy, which is used here only to index and broadcast object arrays.
"""

from __future__ import annotations

import inspect
import operator
import os
import pickle
import warnings
from fractions import Fraction

import pytest

if os.environ.get("VERIFLOAT_IMPL", "cpp") != "cpp":
    pytest.skip("FPArray is part of the C++ package", allow_module_level=True)
np = pytest.importorskip("numpy")

from test_array import (flat, kernels, narrow_fmt, nested, outcome, pic, same)  # noqa: E402,F401
from verifloat import (E4M3, FP, FP16, FP32, UE4M3, Accumulator, FPArray, FPFlags, FPFormat, Rounding, dot,  # noqa: E402
                       vectors)
from verifloat.array import _PAIR_METHODS, _PLAIN_METHODS, _VALUE_METHODS  # noqa: E402

OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


def rand_shape(rng, max_dims=4, max_len=4) -> tuple:
    return tuple(rng.randint(1, max_len) for _ in range(rng.randint(1, max_dims)))


def count(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def rand_codes(rng, f: FPFormat, n: int) -> list[int]:
    edges = vectors.edge_codes(f)
    return [rng.choice(edges) if rng.random() < 0.3 else rng.getrandbits(f.size) for _ in range(n)]


def rand_array(rng, f: FPFormat, shape) -> FPArray:
    return FPArray.from_raw(nested(rand_codes(rng, f, count(shape)), shape), f)


def flagged_array(rng, shape) -> FPArray:
    """An FP16 array whose elements carry assorted flags."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rand_array(rng, FP16, shape) * rand_array(rng, FP16, shape)


def objects(x: FPArray):
    """The elements as a NumPy object array of FP values, same shape."""
    out = np.empty(count(x.shape), dtype=object)
    out[:] = flat(x.tolist())
    return out.reshape(x.shape)


def as_lists(a) -> list:
    """Nested lists of an object ndarray (tolist would also unpack nothing: FP is a leaf)."""
    return [as_lists(v) for v in a] if a.ndim > 1 else list(a)


# ---------------------------------------------------------------- indexing

def rand_key(rng, shape):
    items = []
    for d in shape[:rng.randint(0, len(shape))]:
        r = rng.random()
        if r < 0.4:
            items.append(rng.randint(-d, d - 1))
        else:
            lim = d + 2
            start, stop = (rng.choice([None, rng.randint(-lim, lim)]) for _ in range(2))
            step = rng.choice([None, 1, 1, -1, 2, -2, 3, rng.randint(-4, 4) or 1])
            items.append(slice(start, stop, step))
    if rng.random() < 0.3:
        items.insert(rng.randint(0, len(items)), Ellipsis)
    for _ in range(rng.choice([0, 0, 0, 1, 2])):
        items.insert(rng.randint(0, len(items)), None)
    return items[0] if len(items) == 1 and rng.random() < 0.5 else tuple(items)


def test_basic_indexing_matches_numpy(rng, iters):
    """Integers, slices, `...` and None select the elements NumPy selects, each with its flags."""
    scalars = arrays = empties = 0
    for _ in range(iters):
        x = flagged_array(rng, rand_shape(rng))
        ref = objects(x)
        key = rand_key(rng, x.shape)
        try:
            want = ref[key]
        except IndexError:          # an integer past its axis (a `...` moved it to another one)
            with pytest.raises(IndexError, match="out of range|no elements"):
                x[key]
            continue
        if isinstance(want, np.ndarray) and want.ndim == 0:     # NumPy's 0-d array (an index with `...`)
            want = want[()]                                     # is the element itself here
        if isinstance(want, FP):
            got = x[key]
            assert isinstance(got, FP) and pic(got) == pic(want), key
            scalars += 1
        elif want.size == 0:
            with pytest.raises(IndexError, match="no elements"):
                x[key]
            empties += 1
        else:
            got = x[key]
            assert isinstance(got, FPArray) and got.shape == want.shape, key
            assert pic(got) == pic(as_lists(want)), key
            assert got.format is x.format
            arrays += 1
    assert min(scalars, arrays) > 0 and (min(scalars, arrays, empties) > iters // 50 or iters < 2000)
    x = FP16.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert x[:, 1].to_float() == [2.0, 5.0] and x[::-1, ::2].to_float() == [[4.0, 6.0], [1.0, 3.0]]
    assert x[None].shape == (1, 2, 3) and x[..., None].shape == (2, 3, 1) and x[1, ..., 2] == FP16(6.0)
    y = x[:, 1:]
    assert y == FP16.array([[2.0, 3.0], [5.0, 6.0]]) and pickle.loads(pickle.dumps(y)) == y
    assert [r.to_float() for r in x] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]      # iteration: rows


@pytest.mark.parametrize("key, exc, msg", [
    ((0, 0, 0), IndexError, "too many indices"),
    ((slice(None), 3), IndexError, "out of range"),
    ((..., ...), IndexError, "single ellipsis"),
    ((None, 0, 0, 0), IndexError, "too many indices"),
    ("a", TypeError, "integers or slices"),
    (True, TypeError, "integers or slices"),
    (1.0, TypeError, "integers or slices"),
    ([0, 1], TypeError, "integers or slices"),
    (slice(2, 2), IndexError, "no elements"),
    (slice(None, None, 0), ValueError, "slice step cannot be zero"),
])
def test_indexing_errors(key, exc, msg):
    with pytest.raises(exc, match=msg):
        FP16.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])[key]


# ---------------------------------------------------------------- broadcasting

def compatible_shapes(rng):
    """Two shapes that broadcast (90%), or two random ones."""
    out = rand_shape(rng)
    if rng.random() < 0.1:
        return rand_shape(rng), out

    def part():
        s = [1 if rng.random() < 0.35 else d for d in out]
        return tuple(s[rng.randint(0, len(s) - 1):])
    return part(), part()


def test_operators_broadcast(rng, iters, kernels):
    done = refused = 0
    for _ in range(iters // 8):
        fx = rng.choice([FP16, FP16, E4M3, FP32, UE4M3, narrow_fmt(rng)])
        fy = fx if rng.random() < 0.8 else rng.choice([FP16, E4M3, narrow_fmt(rng)])
        sx, sy = compatible_shapes(rng)
        x, y = rand_array(rng, fx, sx), rand_array(rng, fy, sy)
        try:
            bx, by = np.broadcast_arrays(objects(x), objects(y))
        except ValueError:
            for op in OPS.values():
                with pytest.raises(ValueError, match="shape mismatch"):
                    op(x, y)
            refused += 1
            continue
        for name, op in OPS.items():
            same(lambda: op(x, y),
                 lambda: nested([op(a, b) for a, b in zip(bx.ravel(), by.ravel())], bx.shape), (name, str(fx), sx, sy))
        done += 1
    assert done > 0 and (refused > 0 or iters < 2000)      # a short run may draw no mismatch
    for op in OPS.values():
        with pytest.raises(ValueError, match="shape mismatch"):
            op(FP16.array([[1.0, 2.0], [3.0, 4.0]]), FP16.array([1.0, 2.0, 3.0]))
    a, b = FP16.array([[1.0], [2.0]]), FP16.array([10.0, 20.0, 30.0])
    assert (a + b).to_float() == [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]
    assert (b - a).shape == (2, 3) and (a * a).shape == (2, 1)


def test_broadcast_to_and_reshape(rng, iters):
    for _ in range(iters // 4):
        x = flagged_array(rng, rand_shape(rng, 3))
        shape = tuple(rng.randint(1, 3) for _ in range(rng.randint(0, 2))) + tuple(
            d if d != 1 else rng.randint(1, 3) for d in x.shape)
        got = x.broadcast_to(shape)
        assert pic(got) == pic(as_lists(np.broadcast_to(objects(x), shape)))
        n = count(x.shape)
        assert x.reshape(-1).shape == (n,) and pic(x.reshape(-1)) == pic(flat(x.tolist()))
        assert x.reshape(1, -1).shape == (1, n) and x.reshape((-1, 1)).shape == (n, 1)
        assert x.reshape(-1, *x.shape[1:]) == x
    x = FP16.array([[1.0, 2.0, 3.0]])
    assert x.broadcast_to((2, 3)) == FP16.array([1.0, 2.0, 3.0]).broadcast_to((2, 3)) != x
    assert FP16.array([7.0]).broadcast_to(3) == FP16.array([7.0, 7.0, 7.0])
    for shape, msg in (((2, 2), "cannot broadcast"), ((3,), "cannot broadcast"), ((1, 3, 1), "cannot broadcast"), ((0, 3), "positive"),
                       ((), "at least one axis"), ((1, 1, 3), None)):
        if msg is None:
            x.broadcast_to(shape)
        else:
            with pytest.raises(ValueError, match=msg):
                x.broadcast_to(shape)
    with pytest.raises(ValueError, match="cannot reshape"):
        x.reshape(-1, 2)
    with pytest.raises(ValueError, match="positive"):
        x.reshape(-1, -1)


# ---------------------------------------------------------------- element-wise methods

ARITY = {name: len(inspect.signature(getattr(FP, name)).parameters) - 1
         for name in (*_VALUE_METHODS, *_PAIR_METHODS, *_PLAIN_METHODS)}
ROUNDINGS = [None, Rounding.RNE, Rounding.RNA, Rounding.RTZ, Rounding.RUP, Rounding.RDN]


def method_args(rng, name: str, f: FPFormat, shape):
    """Arguments for FPArray.<name>, and for each element the scalar's arguments."""
    n = count(shape)
    if name == "scaleb":
        k = rng.choice([0, 1, -1, rng.randint(-40, 40), rng.randint(-5000, 5000)])
        return (k,), [(k,)] * n
    if name == "round_to_integral":
        args = (rng.choice(ROUNDINGS), rng.random() < 0.5)
        return args, [args] * n
    if name == "to_int":
        args = (rng.choice([1, 8, 32, 64, rng.randint(1, 70)]), rng.random() < 0.6, rng.choice(ROUNDINGS),
                rng.random() < 0.5)
        return args, [args] * n
    operands = {"fma": 2}.get(name, 1 if name not in ("sqrt", "next_up", "next_down", "logb", "fclass") else 0)
    args, per = [], [[] for _ in range(n)]
    for _ in range(operands):
        r = rng.random()
        g = f if rng.random() < 0.85 else rng.choice([FP16, E4M3])
        if r < 0.5:                                         # an array of the same shape
            y = rand_array(rng, g, shape)
            vals = flat(y.tolist())
        elif r < 0.7:                                       # an array that broadcasts
            s = tuple(1 if rng.random() < 0.5 else d for d in shape)[rng.randint(0, len(shape) - 1):]
            y = rand_array(rng, g, s)
            vals = list(np.broadcast_to(objects(y), shape).ravel())
        elif r < 0.9:                                       # one FP for every element
            y = g.from_raw(rand_codes(rng, g, 1)[0])
            vals = [y] * n
        else:                                               # a number
            y = rng.choice([0, 1, -2, 0.5, 3.75, Fraction(1, 3)])
            vals = [y] * n
        args.append(y)
        for i in range(n):
            per[i].append(vals[i])
    if name in _PAIR_METHODS:                               # eq, lt, le, compare: signaling
        s = rng.random() < 0.5
        args.append(s)
        for p in per:
            p.append(s)
    return tuple(args), [tuple(p) for p in per]


@pytest.mark.parametrize("name", sorted(ARITY))
def test_method_is_the_scalar_method_on_every_element(name, rng, iters):
    kinds = set()
    for _ in range(max(iters // 10, 40)):
        f = rng.choice([FP16, E4M3, narrow_fmt(rng), narrow_fmt(rng)])
        shape = rand_shape(rng, 3, 3)
        x = rand_array(rng, f, shape)
        elems = flat(x.tolist())
        args, per = method_args(rng, name, f, shape)

        def scalars():
            return [getattr(e, name)(*a) for e, a in zip(elems, per)]
        got, want = outcome(lambda: getattr(x, name)(*args)), outcome(scalars)
        ctx = (name, str(f), shape, x.raw, args)
        r = want[0]
        raised = isinstance(r, tuple) and r[:1] == ("raise",)
        if isinstance(got[0], tuple) and got[0][:2] == ("raise", "ValueError") and "at most 64 bits" in got[0][2]:
            # Mixed formats promoted past 64 bits: an array cannot hold the
            # result and says so at the first element.
            assert name in _VALUE_METHODS and (raised or r[0].format.size > 64), ctx
            continue
        assert got[1] == want[1], ctx                       # the same warnings, in the same order
        if raised:                                          # the scalar raises: so does the array
            assert got[0] == r, ctx
            kinds.add("raise")
        elif name in _VALUE_METHODS:
            assert pic(got[0]) == pic(nested(r, shape)), ctx
            kinds.add("value")
        elif name in _PAIR_METHODS:
            flags = FPFlags(0)
            for _, fl in r:
                flags |= fl
            assert got[0] == (nested([v for v, _ in r], shape), flags), ctx
            assert type(got[0][1]) is FPFlags
            kinds.add("pair")
        else:
            assert got[0] == nested(r, shape), ctx
            kinds.add("plain")
    assert kinds - {"raise"}


def test_methods_have_the_scalar_signatures():
    for name in ARITY:
        m = getattr(FPArray, name)
        assert inspect.signature(m) == inspect.signature(getattr(FP, name)), name
        assert m.__name__ == name and m.__qualname__ == f"FPArray.{name}" and m.__doc__.startswith(f"FP.{name} of")
    x = FP16.array([1.5, 2.5, -0.5])
    assert x.round_to_integral(exact=True) == x.round_to_integral(None, True) != x.round_to_integral(Rounding.RTZ)
    assert x.round_to_integral(exact=True).flags == FPFlags.INEXACT and x.round_to_integral().flags == FPFlags(0)
    assert x.fma(c=1, b=2).to_float() == [4.0, 6.0, 0.0]
    assert x.lt(1.0, signaling=False) == ([False, False, True], FPFlags(0))
    assert x.maximum(FP16.array([2.0])).to_float() == [2.0, 2.5, 2.0]
    with pytest.raises(TypeError):
        x.sqrt(1)
    with pytest.raises(TypeError):
        x.fma(1)
    with pytest.raises(ValueError, match="shape mismatch"):
        x.minimum(FP16.array([1.0, 2.0]))


# ---------------------------------------------------------------- sums

def rand_acc(rng, f: FPFormat):
    r = rng.random()
    if r < 0.15:
        return None
    if r < 0.3:
        return rng.choice([f, FP32, FP16])
    order = rng.choice(["exact", "sequential", "pairwise"])
    product = rng.choice([None, None, f, FP16])
    group = rng.choice([1, 1, 2, 4])
    align = rng.choice([None, 8, 24]) if group > 1 else None
    return Accumulator(rng.choice([f, FP32, FP16]), order, product, group, align)


def test_sum_is_dot_with_ones(rng, iters, kernels):
    for _ in range(iters // 8):
        f = rng.choice([FP16, FP16, E4M3, FP32, narrow_fmt(rng)])
        shape = rand_shape(rng, 3, 5)
        x = rand_array(rng, f, shape)
        acc = rand_acc(rng, f)
        ref = objects(x)
        ctx = (str(f), shape, x.raw, str(acc))

        def one(lane):
            lane = list(lane)
            if isinstance(acc, Accumulator):
                return acc.sum_products(lane, [1] * len(lane))
            return dot(lane, [1] * len(lane), acc)

        def show(o):        # outcome with a comparable result
            r = o[0]
            if isinstance(r, (FP, FPArray)) or (isinstance(r, list) and isinstance(flat(r)[0], FP)):
                r = pic(r)
            return r, o[1]

        got, want = outcome(lambda: x.sum(acc=acc)), outcome(lambda: one(ref.ravel()))
        assert show(got) == show(want), ctx
        axis = rng.randint(-len(shape), len(shape) - 1)
        moved = np.moveaxis(ref, axis, -1)
        lanes = moved.reshape(-1, moved.shape[-1])

        def along():
            sums = [one(lane) for lane in lanes]
            return sums[0] if len(shape) == 1 else nested(sums, moved.shape[:-1])
        got, want = outcome(lambda: x.sum(axis, acc)), outcome(along)
        assert show(got) == show(want), (ctx, axis)
        if len(shape) > 1 and acc is not None and not isinstance(got[0], tuple):
            assert isinstance(got[0], FPArray) and got[0].shape == moved.shape[:-1]


def test_sum_known_values():
    x = FP16.array([[1.0, 2.0 ** -11, 2.0 ** -11], [2048.0, 1.0, -2048.0]])
    assert x.sum() == Fraction(2) + Fraction(2, 2048) and x.sum(1) == [1 + Fraction(1, 1024), Fraction(1)]
    assert x.sum(0) == [Fraction(2049), 1 + Fraction(1, 2048), Fraction(1, 2048) - 2048]
    # In binary16 each 2**-11 is half an ulp of 1.0: added one at a time it is
    # lost (ties to even); the exact sum rounded once keeps both.
    seq, once = x.sum(1, Accumulator(FP16, "sequential")), x.sum(1, FP16)
    # (2048 + 1 is a tie too: it rounds to 2048, and the last row sums to 0.)
    assert seq.to_float() == [1.0, 0.0] and once.to_float() == [1.0 + 2.0 ** -10, 1.0]
    assert seq[0].flags == seq[1].flags == FPFlags.INEXACT and once.flags == FPFlags(0)
    assert x.sum(acc=Accumulator(FP16, "pairwise")) == dot(flat(x.tolist()), [1] * 6, Accumulator(FP16, "pairwise"))
    assert FP16.array([1.0, 2.0, 3.0]).sum(0, FP32) == FP32(6.0) == FP16.array([1.0, 2.0, 3.0]).sum(-1, FP32)
    for axis, exc in ((2, ValueError), (-3, ValueError), (1.0, TypeError), (True, TypeError)):
        with pytest.raises(exc, match="axis"):
            x.sum(axis)


# ---------------------------------------------------------------- constructors

def test_constructors(rng, iters):
    for _ in range(iters // 20):
        f = narrow_fmt(rng)
        shape = rand_shape(rng, 3)
        v = rng.choice([0, 1, 0.1, -2.5, 1e30, Fraction(1, 3), float("inf")])
        for build, value in ((lambda: FPArray.full(shape, v, f), v), (lambda: FPArray.zeros(shape, f), 0),
                             (lambda: FPArray.ones(shape, f), 1)):
            got, want = outcome(build), outcome(lambda: f(value))
            if isinstance(want[0], tuple):                  # the format cannot take the value: the same error
                assert got == want
                continue
            assert got[1] == want[1]                        # one rounding, one warning
            assert got[0].shape == shape and got[0].format is f
            assert set(pic(got[0])[2]) == {(want[0].raw, int(want[0].flags))}
    assert FPArray.full(3, 0.1, FP16).to_float() == [float(FP16(0.1))] * 3
    assert FPArray.full(2, 0.1, FP16).flags == FPFlags.INEXACT and FPArray.zeros((2, 2), FP16).raw == [[0, 0], [0, 0]]
    assert FPArray.ones((1, 2), E4M3) == E4M3.array([[1.0, 1.0]])
    for bad, exc in (((), ValueError), ((2, 0), ValueError), ("ab", TypeError), (1.5, TypeError)):
        with pytest.raises(exc):
            FPArray.zeros(bad, FP16)
