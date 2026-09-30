"""Dedicated tests for NVFP4: the E2M1 element format and NVFP4 block scaling."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from verifloat import (E2M1, E4M3, E8M23, FP, NVFP4, BlockTensor, CastWarning, FPFlags,
                       dot, matmul)
from reference import (Fmt, caught, check_fp_ops, grid, rand_code, rand_tensor,
                       ref_matmul, ref_quantize, ref_round, ref_value)

E, M = 2, 1
MAGNITUDES = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
E2M1_REF = Fmt.make(E, M, inf_nan=False)   # reference model of the element format


# Spec anchors
def test_e2m1_preset_is_ocp_fp4():
    x = E2M1(1)
    assert (x.exp_bits, x.mantissa_bits, x.size, x.bias, x.inf_nan) == (2, 1, 4, 1, False)
    assert (E2M1.max, E2M1.min_subnormal) == (6, Fraction(1, 2))


def test_decode_table():
    for code in range(16):
        want = MAGNITUDES[code & 7] * (-1 if code & 8 else 1)
        assert float(E2M1.from_raw(code)) == want
        assert E2M1.from_raw(code).raw == code


@pytest.mark.parametrize("x, want", [
    (0.25, 0.0), (0.75, 1.0), (1.25, 1.0), (1.75, 2.0),
    (2.5, 2.0), (3.5, 4.0), (5.0, 4.0), (5.1, 6.0), (-2.5, -2.0),
])
def test_round_ties_even(x, want):
    assert float(E2M1(x)) == want


# Constrained random
def rand_real(rng) -> float:
    """Random real, biased toward rounding midpoints and saturation."""
    r = rng.random()
    sign = rng.choice([-1, 1])
    if r < 0.3:
        i = rng.randrange(7)
        return sign * (MAGNITUDES[i] + MAGNITUDES[i + 1]) / 2
    if r < 0.45:
        return sign * rng.uniform(6, 1e6)
    if r < 0.55:
        return sign * rng.uniform(0, 0.5)
    return rng.uniform(-8, 8)


def test_from_value_random(rng, iters):
    for _ in range(iters):
        x = rand_real(rng)
        got = E2M1(x)
        want = ref_round(Fraction(x), E2M1_REF, math.copysign(1, x) < 0)
        assert (got.raw, got.flags) == (want.raw, want.flags), x
        assert -6 <= float(got) <= 6


def test_ops_random(rng, iters):
    for _ in range(iters):
        check_fp_ops(rand_code(rng, E, M), rand_code(rng, E, M), E2M1_REF)


def test_scalar_operand_random(rng, iters):
    for _ in range(iters):
        a = rand_code(rng, E, M)
        k = rng.randint(-8, 8)
        fa, va = E2M1.from_raw(a), ref_value(a, E2M1_REF)
        vk = ref_value(ref_round(Fraction(k), E2M1_REF).raw, E2M1_REF)
        lossy = {CastWarning} if vk != k else set()
        r, warned = caught(lambda: fa + k)
        assert r.raw == ref_round(va + vk, E2M1_REF, fa.sign and vk < 0).raw and warned == lossy
        r, warned = caught(lambda: fa * k)
        assert r.raw == ref_round(va * vk, E2M1_REF, fa.sign != (vk < 0)).raw and warned == lossy
        r, warned = caught(lambda: k - fa)
        assert r.raw == ref_round(vk - va, E2M1_REF, vk < 0 and not fa.sign).raw and warned == lossy


def test_saturates_random(rng, iters):
    for _ in range(iters):
        x = rng.choice([-1, 1]) * rng.uniform(6, 1e9)
        assert float(E2M1(x)) == math.copysign(6.0, x)
        a = E2M1.from_raw(rand_code(rng, E, M))
        assert float(E2M1(6) + abs(a)) == 6.0
        assert float(E2M1(-6) - abs(a)) == -6.0


def test_signed_zero():
    assert E2M1(-0.0).raw == 0b1000
    assert (E2M1(-0.0) + E2M1(-0.0)).sign
    assert not (E2M1(1) + E2M1(-1)).sign
    assert E2M1(-0.0) == E2M1(0.0)
    assert math.copysign(1, float(E2M1.from_raw(0b1000))) < 0


def test_div_zero(rng):
    # E2M1 has no inf/NaN, so there is nothing IEEE-correct to return.
    a = E2M1.from_raw(rand_code(rng, E, M))
    for z in (0, E2M1.from_raw(0b0000), E2M1.from_raw(0b1000)):
        with pytest.raises(ZeroDivisionError):
            a / z


def test_no_specials():
    assert all(v.is_finite for v in E2M1.all_values())
    with pytest.raises(ValueError):
        E2M1(float("nan"))
    r = E2M1(float("-inf"))                      # saturates, like ml_dtypes
    assert float(r) == -6.0 and r.flags == FPFlags.OVERFLOW | FPFlags.INEXACT


def test_grid_matches_spec():
    assert [float(v) for v in grid(E, M)] == MAGNITUDES


def test_status_flags():
    NX, OF = FPFlags.INEXACT, FPFlags.OVERFLOW
    assert E2M1(1.5).flags == 0
    assert E2M1(5.1).flags == NX
    assert E2M1(100).flags == OF | NX
    assert E2M1(-100).flags == OF | NX
    assert (E2M1(6) + 6).flags == OF | NX
    assert (E2M1(3) * 1.5).flags == NX                  # 4.5 -> 4
    assert E2M1(0.25).flags == FPFlags.UNDERFLOW | NX   # tiny, inexact


# NVFP4 block scaling: E2M1 x16 elements, E4M3 block scale (max 448), FP32 tensor scale
def block_iters(iters):
    return max(iters // 20, 20)


def rand_shape(rng):
    return (rng.randint(1, 64),) if rng.random() < 0.3 else (rng.randint(1, 4), rng.randint(1, 64))


def assert_nvfp4_matches_ref(t, rows):
    ref = ref_quantize(rows, NVFP4)
    unwrap = (lambda x: x[0]) if t.ndim == 1 else (lambda x: x)
    assert t.elem_raw == unwrap(ref.elem_raw)
    assert t.scale_raw == unwrap(ref.scale_raw)
    assert t.tensor_scale.raw == ref.tensor_raw
    assert t.dequantize() == unwrap(ref.values)


def test_block_anchor():
    t = BlockTensor.quantize([6, 3, 1.5, 0, -0.5, -6], NVFP4)
    assert t.elem_raw == [7, 5, 3, 0, 9, 15]
    assert t.scale_raw == [126]                             # E4M3 448, the clamp
    assert t.tensor_scale.to_hex() == "3b124925"            # FP32(6 / (6 * 448))
    assert t.tensor_scale.exact == FP.from_value(Fraction(1, 448), E8M23).exact
    z = BlockTensor.quantize([[0] * 16, [1] * 16], NVFP4)
    assert z.scale_raw == [[0], [126]] and z.dequantize()[0] == [0] * 16


def test_block_quantize_random(rng, iters):
    for _ in range(block_iters(iters)):
        shape = rand_shape(rng)
        rows = rand_tensor(rng, shape)
        t = BlockTensor.quantize(rows[0] if len(shape) == 1 else rows, NVFP4)
        assert_nvfp4_matches_ref(t, rows)
        # Scales never exceed the OCP E4M3 max, elements stay on the E2M1 grid.
        assert all(s <= 0x7E for r in rows_of_raw(t.scale_raw) for s in r)
        assert all(0 <= e < 16 for r in rows_of_raw(t.elem_raw) for e in r)
        # Per block, error is at most one element step (rounding: the widest
        # E2M1 gap is 2, so half of it) or the saturation excess when the
        # scale rounded down: err <= max(step, amax - 6 * step).
        scales = rows_of_raw(t.scale_raw)
        for src, deq, srow in zip(rows, rows_of_raw(t.dequantize()), scales):
            for b in range(0, len(src), 16):
                step = E4M3.from_raw(srow[b // 16]).exact * t.tensor_scale.exact
                amax = max(abs(x) for x in src[b:b + 16])
                err = max(abs(x - q) for x, q in zip(src[b:b + 16], deq[b:b + 16]))
                assert err <= max(step, amax - 6 * step)
        t2 = BlockTensor.from_raw(t.elem_raw, t.scale_raw, NVFP4,
                                  tensor_scale=t.tensor_scale.raw)
        assert t2 == t


def rows_of_raw(x):
    return [x] if x and not isinstance(x[0], list) else x


def test_block_ops_random(rng, iters):
    for _ in range(block_iters(iters)):
        m, k, n = rng.randint(1, 3), rng.randint(1, 48), rng.randint(1, 3)
        a = BlockTensor.quantize(rand_tensor(rng, (m, k)), NVFP4)
        b = BlockTensor.quantize(rand_tensor(rng, (m, k)), NVFP4)
        w = BlockTensor.quantize(rand_tensor(rng, (n, k)), NVFP4)  # weights, K-blocked
        A, B, W = a.dequantize(), b.dequantize(), w.dequantize()
        assert_nvfp4_matches_ref(a + b, [[x + y for x, y in zip(r, s)] for r, s in zip(A, B)])
        assert_nvfp4_matches_ref(a - b, [[x - y for x, y in zip(r, s)] for r, s in zip(A, B)])
        # Y = A @ W.T with both operands blocked along K, like a tensor core.
        WT = [list(c) for c in zip(*W)]
        exact = ref_matmul(A, WT)
        assert matmul(a, [list(c) for c in zip(*W)]) == exact
        got = matmul(a, WT, acc=E8M23)
        assert [[x.raw for x in r] for r in got] == [[FP.from_value(x, E8M23).raw for x in r] for r in exact]
        assert matmul(a, w, transpose_b=True) == exact
        assert dot(a[0], w[0]) == exact[0][0]
        assert dot(a[0], w[0], acc=E8M23).raw == FP.from_value(exact[0][0], E8M23).raw
        t = w.T                          # (K, N), still blocked along K: exact
        assert t.dequantize() == WT and t.axis == 0
        assert matmul(a, t) == exact
        assert_nvfp4_matches_ref(a @ t, exact)          # requantized product
