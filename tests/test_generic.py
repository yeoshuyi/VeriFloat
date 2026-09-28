"""Constrained random tests for UINT, INT and generic FP formats."""

from __future__ import annotations

import operator
from fractions import Fraction

import pytest

from customtypes import FP, INT, UINT
from reference import (check_fp_ops, rand_code, rand_sint, rand_uint,
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
        assert op(UINT(a, bits), k).val == wrap_u(op(a, k), bits)
        assert op(k, UINT(a, bits)).val == wrap_u(op(k, a), bits)


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
            r = op(x, y)
            assert type(r) is UINT and r.bits == bits
            assert r.val == wrap_u(op(x.raw, y.raw), bits)
        s2 = INT(rand_sint(rng, ub + 1), ub + 1)
        r = op(s, s2)
        assert type(r) is INT and r.bits == max(sb, ub + 1)
        assert r.val == wrap_s(op(s.val, s2.val), r.bits)


# FP, random formats
def rand_fmt(rng):
    return rng.randint(2, 5), rng.randint(1, 4)


def test_fp_ops(rng, iters):
    for _ in range(iters):
        E, M = rand_fmt(rng)
        check_fp_ops(rand_code(rng, E, M), rand_code(rng, E, M), E, M)


def test_fp_raw_roundtrip(rng, iters):
    for _ in range(iters):
        E, M = rand_fmt(rng)
        code = rand_code(rng, E, M)
        x = FP.from_raw(code, E, M)
        assert x.size == 1 + E + M
        assert x.raw == code
        assert FP.from_value(float(x), E, M).raw == code
        assert (-x).raw == code ^ (1 << (E + M))
        assert (~x).raw == wrap_u(~code, x.size)


def test_fp_from_value_rounds(rng, iters):
    for _ in range(iters):
        E, M = rand_fmt(rng)
        top = 2 * ref_value((1 << (E + M)) - 1, E, M)
        x = Fraction(rng.randint(-10**6, 10**6), 10**6) * top
        assert FP.from_value(x, E, M).raw == ref_round(x, E, M)


def test_fp_convert(rng, iters):
    for _ in range(iters):
        (E1, M1), (E2, M2) = rand_fmt(rng), rand_fmt(rng)
        code = rand_code(rng, E1, M1)
        x = FP.from_raw(code, E1, M1)
        want = ref_round(ref_value(code, E1, M1), E2, M2, x.sign)
        assert x.convert(E2, M2).raw == want


def test_fp_mixed_formats_promote(rng, iters):
    for _ in range(iters):
        (E1, M1), (E2, M2) = rand_fmt(rng), rand_fmt(rng)
        a, b = rand_code(rng, E1, M1), rand_code(rng, E2, M2)
        fa, fb = FP.from_raw(a, E1, M1), FP.from_raw(b, E2, M2)
        E, M = max(E1, E2), max(M1, M2)
        r = fa + fb
        assert (r.exp_bits, r.mantissa_bits, r.size) == (E, M, 1 + E + M)
        want = ref_value(a, E1, M1) + ref_value(b, E2, M2)
        assert r.raw == ref_round(want, E, M, fa.sign and fb.sign)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_fp_rejects_nan_inf(rng, bad):
    with pytest.raises(ValueError):
        FP.from_value(bad, *rand_fmt(rng))
