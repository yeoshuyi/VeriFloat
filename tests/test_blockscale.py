"""Constrained random tests for block-scaled tensors across formats."""

from __future__ import annotations

from fractions import Fraction

import pytest

from verifloat import (E2M1, E4M3, E8M23, FP16, FP32, MXFP4, MXINT8, UE4M3, BlockFormat,
                       BlockFormatWarning, BlockTensor, CastWarning, FPFormat,
                       IntFormat, Pow2Format, Rounding, dot, matmul)
from reference import (ROUNDINGS, caught, rand_fmt, rand_tensor, ref_matmul, ref_quantize,
                       rows_of)


def rand_block_format(rng) -> BlockFormat:
    """Random element/scale/zero-point/tensor-scale combination."""
    block = rng.choice([1, 2, 4, 8, 16, 32])
    if rng.random() < 0.5:
        elem = rand_fmt(rng)
        while not elem.has_zero:                # a quantizer needs zero
            elem = rand_fmt(rng)
        elem = elem.to_fpformat()
        zp = None
    else:
        elem = IntFormat(rng.randint(2, 8), rng.random() < 0.7,
                         rng.choice(ROUNDINGS))
        zp = IntFormat(rng.randint(2, 8), rng.random() < 0.5) if rng.random() < 0.4 else None
    kind = rng.random()
    if zp is not None or kind < 0.5:
        scale = rng.choice([E4M3, UE4M3, FPFormat(5, 2), FPFormat(3, 2, bias=6)])
    elif kind < 0.85:
        bits = rng.randint(4, 8)
        scale = Pow2Format(bits, rng.randint(0, (1 << bits) - 1))
    else:
        scale = None
    scale_max = None
    if isinstance(scale, FPFormat) and rng.random() < 0.3:
        scale_max = rng.choice([Fraction(1, 2), 3, 100, 448])
    # A per-tensor scale pairs with FP block scales (NVFP4 style).
    tensor = E8M23 if isinstance(scale, FPFormat) and rng.random() < 0.5 else None
    recipe = {}
    if isinstance(scale, FPFormat) and rng.random() < 0.4:
        recipe = dict(compute=rng.choice([FP32, None]),
                      scale_min=rng.choice([None, Fraction(1, 512), Fraction(1, 8)]),
                      zero_scale=rng.choice([None, 1, Fraction(1, 4)]))
    return BlockFormat(elem, block, scale, scale_max=scale_max, zero_point=zp,
                       tensor_scale=tensor, **recipe)


def rand_shape(rng):
    return (rng.randint(1, 40),) if rng.random() < 0.3 else (rng.randint(1, 4), rng.randint(1, 40))


def assert_matches_ref(t: BlockTensor, rows, bf):
    ref = ref_quantize(rows, bf)
    one = len(t.shape) == 1
    unwrap = (lambda x: x[0]) if one else (lambda x: x)
    assert t.elem_raw == unwrap(ref.elem_raw), bf
    assert t.scale_raw == unwrap(ref.scale_raw), bf
    assert t.zero_raw == (None if ref.zero_raw is None else unwrap(ref.zero_raw)), bf
    assert (t.tensor_scale.raw if t.tensor_scale else None) == ref.tensor_raw, bf
    assert t.dequantize() == unwrap(ref.values), bf


def block_iters(iters):
    return max(iters // 20, 20)


def test_quantize_random(rng, iters):
    for _ in range(block_iters(iters)):
        bf, shape = rand_block_format(rng), rand_shape(rng)
        rows = rand_tensor(rng, shape)
        vals = rows[0] if len(shape) == 1 else rows
        t = BlockTensor.quantize(vals, bf)
        assert t.shape == shape
        assert_matches_ref(t, rows, bf)
        # Raw codes roundtrip through from_raw bit-exactly.
        t2 = BlockTensor.from_raw(t.elem_raw, t.scale_raw, bf, t.zero_raw,
                                  t.tensor_scale.raw if t.tensor_scale else None)
        assert t2 == t and t2.dequantize() == t.dequantize()


def test_ops_random(rng, iters):
    for _ in range(block_iters(iters)):
        bf = rand_block_format(rng)
        shape = (rng.randint(1, 4), rng.randint(1, 24))
        a = BlockTensor.quantize(rand_tensor(rng, shape), bf)
        b = BlockTensor.quantize(rand_tensor(rng, shape), bf)
        A, B = a.dequantize(), b.dequantize()
        for got, exact in ((a + b, [[x + y for x, y in zip(r, s)] for r, s in zip(A, B)]),
                           (a - b, [[x - y for x, y in zip(r, s)] for r, s in zip(A, B)]),
                           (a * b, [[x * y for x, y in zip(r, s)] for r, s in zip(A, B)]),
                           (a * 3, [[x * 3 for x in r] for r in A]),
                           (-a, [[-x for x in r] for r in A])):
            assert_matches_ref(got, exact, bf)
        # Transpose moves blocks with their axis: exact, no requantization.
        cols = [list(c) for c in zip(*A)]
        t, warned = caught(lambda: a.T)
        assert t.dequantize() == cols and not warned and t.axis == 0
        assert t.T == a                                  # bit-exact round trip
        # reblock() requantizes along the new last axis; warns iff lossy.
        r, warned = caught(lambda: t.reblock(-1))
        assert_matches_ref(r, cols, bf)
        assert warned == ({CastWarning} if ref_quantize(cols, bf).values != cols else set())
        # Matmul / dot: exact accumulation.
        c = BlockTensor.quantize(rand_tensor(rng, (shape[1], rng.randint(1, 5))), bf)
        C = c.dequantize()
        assert matmul(a, c) == ref_matmul(A, C)
        assert_matches_ref(a @ c, ref_matmul(A, C), bf)
        acc = [[x.raw for x in r] for r in matmul(a, c, acc=E8M23)]
        assert acc == [[FPFormat(8, 23)(x).raw for x in r] for r in ref_matmul(A, C)]
        v = BlockTensor.quantize(rand_tensor(rng, (shape[1],))[0], bf)
        assert dot(a[0], v) == sum((x * y for x, y in zip(A[0], v.dequantize())), Fraction(0))
        assert matmul(a, v) == [row[0] for row in ref_matmul(A, [[x] for x in v.dequantize()])]


def test_mxfp4_anchor():
    t = BlockTensor.quantize([1, 2, 3], MXFP4)
    assert (t.elem_raw, t.scale_raw) == ([4, 6, 7], [126])   # scale 2**-1
    assert t.dequantize() == [1, 2, 3]
    t = BlockTensor.quantize([7], MXFP4)                    # 7 / 2**0 -> 6 saturates
    assert (t.scale_raw, t.dequantize()) == ([127], [6])
    z = BlockTensor.quantize([0, 0], MXFP4)                 # all-zero block: smallest scale
    assert z.scale_raw == [0] and z.dequantize() == [0, 0]
    assert t.flags


def test_mxint8_anchor():
    # OCP MX: INT8 elements are 1.6 fixed point, so the E8M0 code is
    # floor(log2(amax)) + 127 (checked against gfloat in test_external.py).
    m = BlockTensor.quantize([[1, 2], [3, 4]], MXINT8)
    assert m.elem_raw == [[32, 64], [48, 64]] and m.scale_raw == [[128], [129]]
    assert (m @ m).dequantize() == [[7, 10], [15, 22]]


def test_zero_point_anchor():
    u4 = IntFormat(4, signed=False)
    bf = BlockFormat(u4, 4, E4M3, zero_point=u4)
    t = BlockTensor.quantize([[-1, 0.5, 3, 7]], bf)
    # scale (7 - -1) / 15 -> 0.5625 in E4M3, zero-point round(1/0.5625) = 2
    assert (t.elem_raw, t.scale_raw, t.zero_raw) == ([[0, 3, 7, 14]], [[49]], [[2]])
    assert t.dequantize() == [[Fraction(-9, 8), Fraction(9, 16), Fraction(45, 16), Fraction(27, 4)]]


def test_format_mismatch_warns():
    a = BlockTensor.quantize([1, 2, 3], MXFP4)
    b = BlockTensor.quantize([1, 2, 3], BlockFormat(E2M1, 32, E4M3))
    with pytest.warns(BlockFormatWarning, match=r"result uses \[elem e2m1, finite; block 32; scale ue8m0\] \(left\)"):
        assert (a + b).fmt == MXFP4
    with pytest.warns(BlockFormatWarning):
        a @ b


def test_errors():
    with pytest.raises(ValueError, match="underflowed"):   # 2**255 scale cap
        BlockTensor.quantize([1], BlockFormat(E2M1, 4, Pow2Format(8, 0), tensor_scale=E8M23))
    with pytest.raises(ValueError):
        BlockFormat(E2M1, 16, E4M3, zero_point=IntFormat(4))    # zp needs int elements
    with pytest.raises(ValueError):
        BlockFormat(IntFormat(4), 16, Pow2Format(), zero_point=IntFormat(4))
    with pytest.raises(ValueError):
        BlockFormat(E2M1, 0)
    a = BlockTensor.quantize([[1, 2], [3, 4]], MXFP4)
    with pytest.raises(ValueError):
        a + BlockTensor.quantize([[1, 2, 3]], MXFP4)
    with pytest.raises(ValueError):
        matmul(a, [1, 2, 3])
    with pytest.raises(ValueError):
        BlockTensor.quantize([[1, 2], [3]], MXFP4)
    with pytest.raises(ValueError):
        BlockTensor.quantize([1, float("nan")], MXFP4)


def test_indexing_and_accessors():
    t = BlockTensor.quantize([[1, 2], [3, 4]], MXINT8)
    assert (t[1, 0], t[0], len(t), t.ndim) == (3, [1, 2], 2, 2)
    assert t.to_float() == [[1.0, 2.0], [3.0, 4.0]] and t.error() == 0
    assert BlockTensor.from_raw(t.elem_raw, t.scale_raw, MXINT8).error() is None
    assert repr(t) == "BlockTensor(shape=(2, 2), [elem i8, frac=6; block 32; scale ue8m0])"
    assert rows_of(t.elem_raw) == t.elem_raw
