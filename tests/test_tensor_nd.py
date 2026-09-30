"""N-D block tensors blocked along any axis (numpy is used only as a test
oracle for reshapes and transposes)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from verifloat import MXFP4, NVFP4, BlockTensor, matmul
from reference import ref_quantize
from test_blockscale import rand_block_format

np = pytest.importorskip("numpy")


def rand_nd(rng, ndim):
    shape = tuple(rng.randint(1, 5) for _ in range(ndim))
    n = int(np.prod(shape))
    vals = [Fraction(rng.randint(-(1 << 12), 1 << 12), 1 << rng.randint(0, 12)) for _ in range(n)]
    return shape, np.array(vals, dtype=object).reshape(shape)


def test_nd_quantize_any_axis(rng, iters):
    for _ in range(max(iters // 20, 30)):
        bf = rand_block_format(rng)
        shape, arr = rand_nd(rng, rng.randint(1, 4))
        axis = rng.randrange(-len(shape), len(shape))
        t = BlockTensor.quantize(arr.tolist(), bf, axis)
        # Reference: move the axis last, quantize rows, move it back.
        moved = np.moveaxis(arr, axis, -1)
        rows = moved.reshape(-1, shape[axis]).tolist()
        ref = ref_quantize(rows, bf)
        back = lambda x, n: np.moveaxis(np.array(x, dtype=object).reshape(
            moved.shape[:-1] + (n,)), -1, axis).tolist()
        nb = -(-shape[axis] // bf.block_size)
        assert t.shape == shape and t.axis == axis % len(shape)
        assert t.elem_raw == back(ref.elem_raw, shape[axis])
        assert t.scale_raw == back(ref.scale_raw, nb)
        assert t.dequantize() == back(ref.values, shape[axis])
        # from_raw round trip with the same axis
        t2 = BlockTensor.from_raw(t.elem_raw, t.scale_raw, bf, t.zero_raw,
                                  t.tensor_scale.raw if t.tensor_scale else None, axis=axis)
        assert t2 == t


def test_nd_transpose_indexing_and_ops(rng, iters):
    for _ in range(max(iters // 20, 30)):
        shape, arr = rand_nd(rng, 3)
        t = BlockTensor.quantize(arr.tolist(), MXFP4, axis=rng.randrange(3))
        vals = np.array(t.dequantize(), dtype=object)
        perm = tuple(rng.sample(range(3), 3))
        tt = t.transpose(*perm)
        assert tt.dequantize() == np.transpose(vals, perm).tolist()
        assert tt.axis == perm.index(t.axis)
        inverse = tuple(perm.index(i) for i in range(3))
        assert tt.transpose(*inverse) == t                     # exact round trip
        i, j, k = (rng.randrange(n) for n in shape)
        assert t[i, j, k] == vals[i, j, k] and t[i] == vals[i].tolist()
        assert t[-1, j] == vals[-1, j].tolist()
        s = t + t
        assert s.axis == t.axis and s.shape == shape


def test_batched_matmul(rng, iters):
    for _ in range(max(iters // 40, 20)):
        b, m, k, n = (rng.randint(1, 3) for _ in range(4))
        _, x = rand_nd(rng, 0) if False else (None, None)
        A = [[[Fraction(rng.randint(-50, 50), 8) for _ in range(k)] for _ in range(m)]
             for _ in range(b)]
        B = [[[Fraction(rng.randint(-50, 50), 8) for _ in range(n)] for _ in range(k)]
             for _ in range(b)]
        a, bb = BlockTensor.quantize(A, NVFP4), BlockTensor.quantize(B, NVFP4, axis=-2)
        va = np.array(a.dequantize(), dtype=object)
        vb = np.array(bb.dequantize(), dtype=object)
        want = [(va[i] @ vb[i]).tolist() for i in range(b)]
        assert matmul(a, bb) == want
        assert matmul(a, bb.transpose(0, 2, 1), transpose_b=True) == want


def test_nd_errors():
    with pytest.raises(ValueError, match="ragged"):
        BlockTensor.quantize([[1, 2], [3]], MXFP4)
    with pytest.raises(ValueError, match="axis"):
        BlockTensor.quantize([[1, 2]], MXFP4, axis=2)
    t = BlockTensor.quantize([[1, 2], [3, 4]], MXFP4)
    with pytest.raises(IndexError):
        t[2, 0]
    with pytest.raises(ValueError, match="permutation"):
        t.transpose(0, 0)
    with pytest.raises(ValueError, match="scale codes"):
        BlockTensor.from_raw(t.elem_raw, [[1, 2]], MXFP4)
