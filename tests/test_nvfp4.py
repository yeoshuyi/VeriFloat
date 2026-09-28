"""Dedicated tests for NVFP4 (E2M1), the default FP format."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from customtypes import FP
from reference import check_fp_ops, grid, rand_code, ref_round, ref_value

E, M = 2, 1
MAGNITUDES = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


# Spec anchors
def test_default_format_is_e2m1():
    x = FP.from_value(1)
    assert (x.exp_bits, x.mantissa_bits, x.size, x.bias) == (2, 1, 4, 1)


def test_decode_table():
    for code in range(16):
        want = MAGNITUDES[code & 7] * (-1 if code & 8 else 1)
        assert float(FP.from_raw(code)) == want
        assert FP.from_raw(code).raw == code


@pytest.mark.parametrize("x, want", [
    (0.25, 0.0), (0.75, 1.0), (1.25, 1.0), (1.75, 2.0),
    (2.5, 2.0), (3.5, 4.0), (5.0, 4.0), (5.1, 6.0), (-2.5, -2.0),
])
def test_round_ties_even(x, want):
    assert float(FP.from_value(x)) == want


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
        got = FP.from_value(x)
        assert got.raw == ref_round(Fraction(x), E, M, math.copysign(1, x) < 0), x
        assert -6 <= float(got) <= 6


def test_ops_random(rng, iters):
    for _ in range(iters):
        check_fp_ops(rand_code(rng, E, M), rand_code(rng, E, M), E, M)


def test_scalar_operand_random(rng, iters):
    for _ in range(iters):
        a = rand_code(rng, E, M)
        k = rng.randint(-8, 8)
        fa, va = FP.from_raw(a), ref_value(a, E, M)
        vk = ref_value(ref_round(Fraction(k), E, M), E, M)
        assert (fa + k).raw == ref_round(va + vk, E, M, fa.sign and vk < 0)
        assert (fa * k).raw == ref_round(va * vk, E, M, fa.sign != (vk < 0))
        assert (k - fa).raw == ref_round(vk - va, E, M, vk < 0 and not fa.sign)


def test_saturates_random(rng, iters):
    for _ in range(iters):
        x = rng.choice([-1, 1]) * rng.uniform(6, 1e9)
        assert float(FP.from_value(x)) == math.copysign(6.0, x)
        a = FP.from_raw(rand_code(rng, E, M))
        assert float(FP.from_value(6) + abs(a)) == 6.0
        assert float(FP.from_value(-6) - abs(a)) == -6.0


def test_signed_zero():
    assert FP.from_value(-0.0).raw == 0b1000
    assert (FP.from_value(-0.0) + FP.from_value(-0.0)).sign
    assert not (FP.from_value(1) + FP.from_value(-1)).sign
    assert FP.from_value(-0.0) == FP.from_value(0.0)
    assert math.copysign(1, float(FP.from_raw(0b1000))) < 0


def test_div_zero(rng):
    a = FP.from_raw(rand_code(rng, E, M))
    for z in (0, FP.from_raw(0b0000), FP.from_raw(0b1000)):
        with pytest.raises(ZeroDivisionError):
            a / z


def test_grid_matches_spec():
    assert [float(v) for v in grid(E, M)] == MAGNITUDES
