"""Independent reference models and constrained random generators."""

from __future__ import annotations

import bisect
import functools
import math
import warnings
from fractions import Fraction
from typing import NamedTuple

import pytest

from verifloat import (FP, FPFlags, FPFormat, FPOverflowWarning,
                       FPUnderflowWarning, NaNMode, Rounding)


# Integer reference
def wrap_u(x: int, bits: int) -> int:
    return x & ((1 << bits) - 1)


def wrap_s(x: int, bits: int) -> int:
    x = wrap_u(x, bits)
    return x - (1 << bits) if x >> (bits - 1) else x


def trunc_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def trunc_mod(a: int, b: int) -> int:
    return a - trunc_div(a, b) * b


# Constrained generators: uniform, biased toward boundary values
EDGE_BIAS = 0.25


def rand_uint(rng, bits: int) -> int:
    hi = (1 << bits) - 1
    if rng.random() < EDGE_BIAS:
        return rng.choice([0, 1 & hi, hi, max(hi - 1, 0), 1 << (bits - 1)])
    return rng.randint(0, hi)


def rand_sint(rng, bits: int) -> int:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if rng.random() < EDGE_BIAS:
        return rng.choice([lo, lo + 1, -1, 0, 1 if hi else 0, hi - 1, hi])
    return rng.randint(lo, hi)


def default_bias(E: int) -> int:
    return (1 << (E - 1)) - 1


class Fmt(NamedTuple):
    E: int
    M: int
    bias: int
    signed: bool = True
    wrap: bool = False
    ftz: bool = False
    rounding: Rounding = Rounding.RNE
    inf_nan: bool | str = True
    saturate: bool = False
    nan_mode: NaNMode = NaNMode.CANONICAL
    has_zero: bool = True
    sr_bits: int = 8
    tininess: str = "after"

    @classmethod
    def make(cls, E, M, bias=None, **kw) -> "Fmt":
        return cls(E, M, default_bias(E) if bias is None else bias, **kw)

    @property
    def kw(self) -> dict:
        """Keyword arguments for the FP API."""
        return dict(bias=self.bias, signed=self.signed, wrap=self.wrap,
                    ftz=self.ftz, rounding=self.rounding, inf_nan=self.inf_nan,
                    saturate=self.saturate, nan_mode=self.nan_mode,
                    has_zero=self.has_zero, sr_bits=self.sr_bits,
                    tininess=self.tininess)

    @property
    def code(self) -> tuple:
        return self.E, self.M, self.signed

    def raw(self, code: int) -> FP:
        return FP.from_raw(code, self.E, self.M, **self.kw)

    def to_fpformat(self) -> FPFormat:
        return FPFormat(self.E, self.M, **self.kw)

    # Encoding geometry, written from the IEEE 754 / OCP definitions
    @property
    def top(self) -> int:           # all-ones exponent field
        return (1 << self.E) - 1

    @property
    def mask(self) -> int:
        return (1 << self.M) - 1

    @property
    def max_code(self) -> int:      # magnitude code of the largest finite value
        all_ones = (1 << (self.E + self.M)) - 1
        if self.inf_nan is True:    # whole top binade is inf/NaN
            return all_ones - (1 << self.M)
        if self.inf_nan == "fn":    # only the all-ones code is NaN
            return all_ones - 1
        return all_ones

    @property
    def emin(self) -> int:          # exponent of the lowest normal binade
        return (1 if self.has_zero else 0) - self.bias

    @property
    def sbit(self) -> int:
        return 1 << (self.E + self.M)


ROUNDINGS = [Rounding.RNE, Rounding.RNA, Rounding.RTZ, Rounding.RUP, Rounding.RDN]


def rand_fmt(rng) -> Fmt:
    """Random valid format: default bias 40%, unsigned 30%, wrap 20%, ftz 20%,
    rounding RNE 50%, specials ieee/finite/fn 40/40/20%, saturate 20%,
    NaN mode canonical 50%, tininess before 30%, zero-width mantissa 10%,
    no zero 10%. (Stochastic rounding is tested separately.)"""
    while True:
        E, M = rng.randint(2, 5), rng.randint(1, 4)
        bias = default_bias(E)
        if rng.random() >= 0.4:
            bias = rng.choice([bias + rng.randint(-4, 4), rng.randint(-3, (1 << E) + 2)])
        rounding = Rounding.RNE if rng.random() < 0.5 else rng.choice(ROUNDINGS)
        inf_nan = rng.choices([True, False, "fn"], [4, 4, 2])[0]
        nan_mode = NaNMode.CANONICAL if rng.random() < 0.5 else rng.choice(list(NaNMode))
        has_zero = rng.random() >= 0.1
        if rng.random() < 0.1:
            M = 0
            inf_nan = "fn" if inf_nan is True else inf_nan
        if not has_zero and inf_nan is False:
            inf_nan = "fn"          # a no-zero format needs NaN for zero results
        f = Fmt(E, M, bias, rng.random() < 0.7, rng.random() < 0.2,
                rng.random() < 0.2, rounding, inf_nan, rng.random() < 0.2,
                nan_mode, has_zero, 8, "before" if rng.random() < 0.3 else "after")
        try:
            f.to_fpformat()
            return f
        except ValueError:
            continue


def special_codes(f: Fmt) -> list[int]:
    """Magnitude codes of inf and NaN encodings (including signaling NaNs)."""
    if f.inf_nan is True:
        return [(f.top << f.M) | m for m in range(f.mask + 1)]  # m=0 is inf
    if f.inf_nan == "fn":
        return [(f.top << f.M) | f.mask]
    return []


def rand_code(rng, E: int, M: int, signed: bool = True, f: Fmt | None = None) -> int:
    """Random raw FP code, biased toward zeros, subnormals, max-magnitude and
    (when a format with specials is given) inf/NaN encodings."""
    sign = rng.getrandbits(1) << (E + M) if signed else 0
    if f is not None and special_codes(f) and rng.random() < 0.15:
        return sign | rng.choice(special_codes(f))
    if rng.random() < EDGE_BIAS:
        top = f.max_code if f is not None else (1 << (E + M)) - 1
        return sign | rng.choice([0, 1, (1 << M) - 1, 1 << M, top])
    return sign | rng.getrandbits(E + M)


# FP reference: enumerate the non-negative grid, round between neighbours
@functools.lru_cache(maxsize=None)
def grid(E: int, M: int, bias: int | None = None, has_zero: bool = True) -> list[Fraction]:
    """Value of every magnitude code as if all were finite."""
    if bias is None:
        bias = default_bias(E)
    vals = []
    for code in range(1 << (E + M)):
        e, m = code >> M, code & ((1 << M) - 1)
        if e == 0 and has_zero:
            vals.append(Fraction(m, 1 << M) * Fraction(2) ** (1 - bias))
        else:
            vals.append((1 + Fraction(m, 1 << M)) * Fraction(2) ** (e - bias))
    return vals


@functools.lru_cache(maxsize=None)
def grid_set(E: int, M: int, bias: int | None = None) -> frozenset[Fraction]:
    return frozenset(grid(E, M, bias))


def finite_grid(f: Fmt) -> list[Fraction]:
    return grid(f.E, f.M, f.bias, f.has_zero)[:f.max_code + 1]


class Val(NamedTuple):
    """A decoded operand: kind is 'fin', 'inf' or 'nan'."""
    kind: str
    v: Fraction | None
    sign: bool
    snan: bool = False
    payload: int = 0
    M: int = 0
    fn: bool = False


def ref_decode(code: int, f: Fmt) -> Val:
    mag = code & (f.sbit - 1)
    sign = bool(f.signed and code & f.sbit)
    e, m = mag >> f.M, mag & f.mask
    if f.inf_nan is True and e == f.top:
        if m == 0:
            return Val("inf", None, sign)
        return Val("nan", None, sign, not m >> (f.M - 1), m, f.M)
    if f.inf_nan == "fn" and e == f.top and m == f.mask:
        return Val("nan", None, sign, False, m, f.M, fn=True)
    v = Fraction(0) if (f.ftz and f.has_zero and e == 0) else \
        grid(f.E, f.M, f.bias, f.has_zero)[mag]
    return Val("fin", -v if sign else v, sign)


def ref_value(code: int, f: Fmt) -> Fraction:
    """Exact value of a finite code."""
    d = ref_decode(code, f)
    assert d.kind == "fin", f"code {code:#x} is {d.kind} in {f}"
    return d.v


def _choose(lo, hi, x, neg, mode, lo_even):
    """Pick lo or hi (magnitudes, lo <= x < hi) under a rounding mode."""
    if x == lo:
        return lo
    if mode is Rounding.RTZ:
        return lo
    if mode is Rounding.RUP:
        return lo if neg else hi
    if mode is Rounding.RDN:
        return hi if neg else lo
    if x - lo != hi - x:
        return lo if x - lo < hi - x else hi
    if mode is Rounding.RNA:
        return hi
    return lo if lo_even else hi


def _floor_log2(x: Fraction) -> int:
    e = 0
    while x >= 2:
        x /= 2
        e += 1
    while x < 1:
        x *= 2
        e -= 1
    return e


def _unbounded(ax: Fraction, neg: bool, f: Fmt):
    """Round ax to M+1 significant bits with an unbounded exponent.
    Returns (sig, e) with 2**M <= sig < 2**(M+1)."""
    e = _floor_log2(ax)
    scaled = ax / Fraction(2) ** (e - f.M)
    lo = int(scaled)
    # Ties-to-even compares the parity of the whole code; with M=0 that is
    # the exponent field's parity.
    lo_even = (lo % 2 == 0) if f.M else ((e + f.bias) % 2 == 0)
    sig = _choose(lo, lo + 1, scaled, neg, f.rounding, lo_even)
    if sig == 1 << (f.M + 1):
        sig, e = 1 << f.M, e + 1
    return sig, e


class Ref(NamedTuple):
    raw: int
    flags: FPFlags
    warnings: set
    unrounded: object = None


_WARN = {"saturated": FPOverflowWarning, "overflowed": FPOverflowWarning,
         "wrapped-up": FPOverflowWarning, "wrapped-down": FPUnderflowWarning,
         "clamped": FPUnderflowWarning, "underflowed": FPUnderflowWarning,
         "flushed": FPUnderflowWarning}
F = FPFlags


def _warns(event, f: Fmt) -> set:
    if event and (not f.signed or event.startswith("wrapped")):
        return {_WARN[event]}
    return set()


def ref_canonical_nan(f: Fmt, sign: bool = False) -> int:
    m = f.mask if f.inf_nan == "fn" else 1 << (f.M - 1)
    return (int(sign and f.signed) * f.sbit) | (f.top << f.M) | m


def ref_default_nan(f: Fmt) -> int:
    """NaN an invalid operation creates: negative on x86, else canonical."""
    return ref_canonical_nan(f, f.nan_mode is NaNMode.X86)


def _nan_raw(f: Fmt, src: Val | None) -> int:
    """NaN encoding for a result in f: default, or src's payload quieted."""
    if f.nan_mode is NaNMode.CANONICAL or src is None:
        return ref_default_nan(f)
    if f.inf_nan == "fn":
        return ref_canonical_nan(f, src.sign)
    payload = 1 << max(src.M - 1, 0) if (src.fn or src.M == 0) else src.payload
    src_m = max(src.M, 1)
    payload = payload << (f.M - src_m) if f.M >= src_m else payload >> (src_m - f.M)
    payload |= 1 << (f.M - 1)
    return (int(src.sign and f.signed) * f.sbit) | (f.top << f.M) | payload


class NoNaN(Exception):
    """The operation needs a NaN the format cannot encode (VeriFloat raises
    ValueError; division by zero raises ZeroDivisionError instead)."""


def ref_nan(f: Fmt, *ops: Val, invalid: bool = False) -> Ref:
    if f.inf_nan is False:
        raise NoNaN
    flags = F.INVALID if invalid or any(o.snan for o in ops) else F(0)
    nans = [o for o in ops if o.kind == "nan"]
    src = nans[0] if nans else None
    if f.nan_mode is NaNMode.ARM and nans:
        src = next((o for o in nans if o.snan), src)
    return Ref(_nan_raw(f, src), flags, set(), None)


def _no_zero(f: Fmt) -> Ref:
    return Ref(ref_canonical_nan(f), F.INVALID, set(), None)


def ref_inf(sign: bool, f: Fmt) -> Ref:
    """An exact infinity delivered into f."""
    unr = -math.inf if sign else math.inf
    if sign and not f.signed:
        if not f.has_zero:
            return _no_zero(f)._replace(unrounded=unr)
        return Ref(0, F.UNDERFLOW | F.INEXACT, _warns("clamped", f), unr)
    s = int(sign) * f.sbit
    if f.inf_nan is True and not f.saturate:
        return Ref(s | (f.top << f.M), F(0), set(), unr)
    if f.inf_nan == "fn" and not f.saturate:
        return Ref(ref_canonical_nan(f, sign), F.INVALID, set(), unr)
    return Ref(s | f.max_code, F.OVERFLOW | F.INEXACT, _warns("saturated", f), unr)


def _overflow_raw(sign: bool, f: Fmt):
    """IEEE 754-2019 7.4: RNE/RNA go to inf, RTZ to max, RUP/RDN to inf only
    in their own direction. 'fn' formats give NaN in place of inf."""
    to_inf = {Rounding.RNE: True, Rounding.RNA: True, Rounding.RTZ: False,
              Rounding.RUP: not sign, Rounding.RDN: sign}[f.rounding]
    s = int(sign) * f.sbit
    if f.inf_nan is False or f.saturate or not to_inf:
        return s | f.max_code, "saturated"
    if f.inf_nan is True:
        return s | (f.top << f.M), "overflowed"
    return ref_canonical_nan(f, sign), "overflowed"


def ref_round(x, f: Fmt, zero_sign: bool = False) -> Ref:
    """Reference rounding of finite x into f: raw code, flags and warnings."""
    x = Fraction(x)
    M, bias = f.M, f.bias
    min_normal = Fraction(2) ** f.emin
    event, flags = None, F(0)
    if x == 0:
        if not f.has_zero:
            return _no_zero(f)._replace(unrounded=x)
        return Ref(int(zero_sign and f.signed) * f.sbit, F(0), set(), x)
    if x < 0 and not f.signed:
        if not f.has_zero:
            return _no_zero(f)._replace(unrounded=x)
        return Ref(0, F.UNDERFLOW | F.INEXACT, _warns("clamped", f), x)
    neg = x < 0
    sbit = int(neg) * f.sbit
    ax = abs(x)
    u_sig, u_e = _unbounded(ax, neg, f)
    tiny = ax < min_normal if f.tininess == "before" else u_e < f.emin
    vals = finite_grid(f)
    # The code after the largest finite one sits one top-binade ulp up.
    virtual = vals[-1] + Fraction(2) ** ((f.max_code >> M) - bias - M)
    if f.has_zero and not (f.wrap or f.ftz):
        # Gradual underflow: round between neighbouring grid values.
        if ax >= virtual:
            i = len(vals)
        else:
            i = bisect.bisect_right(vals, ax) - 1
            hi = vals[i + 1] if i + 1 < len(vals) else virtual
            r = _choose(vals[i], hi, ax, neg, f.rounding, i % 2 == 0)
            i = i if r == vals[i] else i + 1
        if i == len(vals):
            raw, event = _overflow_raw(neg, f)
            flags = F.OVERFLOW | F.INEXACT
        else:
            raw = sbit | i
            if vals[i] != ax:
                flags = F.INEXACT
                if tiny:
                    flags |= F.UNDERFLOW
                    event = "underflowed"
        return Ref(raw, flags, _warns(event, f), x)
    # No subnormals (wrap, ftz, or no zero): the rounded value has M bits.
    mant = u_sig & f.mask
    field = u_e + bias
    if f.ftz and tiny:
        return Ref(sbit, F.UNDERFLOW | F.INEXACT, _warns("flushed", f), x)
    if field > f.top or (field >= 0 and ((field << M) | mant) > f.max_code):
        flags = F.OVERFLOW | F.INEXACT
        if f.wrap:
            raw, event = sbit | ((field % (1 << f.E)) << M) | mant, "wrapped-up"
        else:
            raw, event = _overflow_raw(neg, f)
        return Ref(raw, flags, _warns(event, f), x)
    if u_e < f.emin:
        if f.wrap:
            raw = sbit | ((field % (1 << f.E)) << M) | mant
            return Ref(raw, F.UNDERFLOW | F.INEXACT, _warns("wrapped-down", f), x)
        # No zero: everything below the smallest value rounds up to it.
        return Ref(sbit | ((0 if not f.has_zero else 1) << M), F.UNDERFLOW | F.INEXACT,
                   _warns("underflowed", f), x)
    raw = sbit | (field << M) | mant
    if u_sig * Fraction(2) ** (u_e - M) != ax:
        flags = F.INEXACT
        if tiny and not f.has_zero:
            flags |= F.UNDERFLOW
            event = "underflowed"
    return Ref(raw, flags, _warns(event, f), x)


def _sum_zero_sign(sa: bool, sb: bool, f: Fmt) -> bool:
    """IEEE 754-2019 6.3: an exact zero sum of opposite signs is +0, except
    under roundTowardNegative, where it is -0."""
    return sa if sa == sb else f.rounding is Rounding.RDN


def ref_arith(op: str, A: Val, B: Val, f: Fmt):
    """IEEE 754 result of A op B in f, or ZeroDivisionError for formats
    without NaN (VeriFloat raises instead of inventing a result)."""
    if A.kind == "nan" or B.kind == "nan":
        return ref_nan(f, A, B)
    if op in "+-":
        sb = B.sign ^ (op == "-")
        if A.kind == "inf" and B.kind == "inf":
            return ref_inf(A.sign, f) if A.sign == sb else ref_nan(f, invalid=True)
        if A.kind == "inf":
            return ref_inf(A.sign, f)
        if B.kind == "inf":
            return ref_inf(sb, f)
        x = A.v + B.v if op == "+" else A.v - B.v
        return ref_round(x, f, _sum_zero_sign(A.sign, sb, f))
    s = A.sign != B.sign
    a_zero = A.kind == "fin" and A.v == 0
    b_zero = B.kind == "fin" and B.v == 0
    if op == "*":
        if "inf" in (A.kind, B.kind):
            return ref_nan(f, invalid=True) if (a_zero or b_zero) else ref_inf(s, f)
        return ref_round(A.v * B.v, f, s)
    if A.kind == "inf":
        return ref_nan(f, invalid=True) if B.kind == "inf" else ref_inf(s, f)
    if B.kind == "inf":
        return ref_round(Fraction(0), f, s)
    if b_zero:
        if f.inf_nan is False:
            return ZeroDivisionError
        if a_zero:
            return ref_nan(f, invalid=True)
        r = ref_inf(s, f)
        return r._replace(flags=r.flags | F.DIVZERO)
    return ref_round(A.v / B.v, f, s)


def ref_less(A: Val, B: Val):
    key = lambda v: (-math.inf if v.sign else math.inf) if v.kind == "inf" else v.v
    return key(A) < key(B), key(A) == key(B)


def ref_fma(A: Val, B: Val, C: Val, f: Fmt) -> Ref:
    """IEEE 754 fusedMultiplyAdd with VeriFloat's per-NaNMode choices for the
    implementation-defined cases (0*inf + qNaN, which NaN is returned)."""
    if A.kind == "nan" or B.kind == "nan":
        if f.nan_mode is NaNMode.ARM and C.snan:
            return Ref(_nan_raw(f, C), F.INVALID, set(), None)
        return ref_nan(f, A, B, C)
    zero = lambda v: v.kind == "fin" and v.v == 0
    if (A.kind == "inf" and zero(B)) or (zero(A) and B.kind == "inf"):
        if f.nan_mode is NaNMode.X86 and C.kind == "nan" and not C.snan:
            return Ref(_nan_raw(f, C), F(0), set(), None)
        if f.nan_mode is NaNMode.ARM and C.snan:
            return Ref(_nan_raw(f, C), F.INVALID, set(), None)
        return Ref(ref_default_nan(f), F.INVALID, set(), None)
    if C.kind == "nan":
        return ref_nan(f, C)
    s = A.sign != B.sign
    if "inf" in (A.kind, B.kind):
        if C.kind == "inf" and C.sign != s:
            return ref_nan(f, invalid=True)
        return ref_inf(s, f)
    if C.kind == "inf":
        return ref_inf(C.sign, f)
    return ref_round(A.v * B.v + C.v, f, _sum_zero_sign(s, C.sign, f))


def ref_sqrt(A: Val, f: Fmt) -> Ref:
    if A.kind == "nan":
        return ref_nan(f, A)
    if A.kind == "fin" and A.v == 0:
        return Ref(int(A.sign) * f.sbit, F(0), set(), None)
    if A.sign:
        return ref_nan(f, invalid=True)
    if A.kind == "inf":
        return Ref(f.top << f.M, F(0), set(), None)
    # sqrt to 400 fractional bits plus a sticky half bit: far more precision
    # than any tested format needs to decide the rounding.
    P = 400
    n, d = A.v.numerator, A.v.denominator
    scaled, rem = divmod(n * 4 ** P, d)
    r = math.isqrt(scaled)
    exact = rem == 0 and r * r == scaled
    y = Fraction(r, 2 ** P) if exact else Fraction(2 * r + 1, 2 ** (P + 1))
    ref = ref_round(y, f)
    return ref._replace(flags=ref.flags | (F(0) if exact else F.INEXACT), unrounded=None)


def ref_minmax(a: int, b: int, f: Fmt, pick_max: bool, number: bool) -> Ref:
    """IEEE 754-2019 9.6 minimum/maximum (NaN wins) and minimumNumber /
    maximumNumber (the number wins); -0 < +0. The chosen operand is returned
    as is (a zero, including a DAZ-flushed subnormal, as a clean zero)."""
    A, B = ref_decode(a, f), ref_decode(b, f)
    snan = A.snan or B.snan
    keep = lambda code, v: int(v.sign) * f.sbit if v.kind == "fin" and v.v == 0 else code
    if A.kind == "nan" or B.kind == "nan":
        if number and not (A.kind == "nan" and B.kind == "nan"):
            code, v = (b, B) if A.kind == "nan" else (a, A)
            return Ref(keep(code, v), F.INVALID if snan else F(0), set(), None)
        return ref_nan(f, A, B)
    key = lambda cv: ((-math.inf if cv[1].sign else math.inf) if cv[1].kind == "inf"
                      else cv[1].v, cv[1].sign is False)   # -0 < +0
    code, v = (max if pick_max else min)((a, A), (b, B), key=key)
    return Ref(keep(code, v), F(0), set(), None)


def _val_raw(v: Val, f: Fmt) -> int:
    """Encoding of an infinity or zero."""
    if v.kind == "inf":
        return int(v.sign) * f.sbit | (f.top << f.M)
    return int(v.sign) * f.sbit


_INT_POLICY = {NaNMode.CANONICAL: ("max", "max", "min"), NaNMode.PROPAGATE: ("max", "max", "min"),
               NaNMode.X86: ("ind", "ind", "ind"), NaNMode.ARM: ("zero", "max", "min")}


def ref_to_int(A: Val, f: Fmt, bits: int, signed: bool, rounding, exact: bool):
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)

    def invalid(kind):
        which = _INT_POLICY[f.nan_mode][kind]
        v = {"max": hi, "min": lo, "zero": 0, "ind": lo if signed else (1 << bits) - 1}[which]
        return v & ((1 << bits) - 1), F.INVALID

    if A.kind == "nan":
        return invalid(0)
    if A.kind == "inf":
        return invalid(2 if A.sign else 1)
    q = ref_int_round(A.v, rounding, -(1 << 200), 1 << 200)
    if q > hi:
        return invalid(1)
    if q < lo:
        return invalid(2)
    return q & ((1 << bits) - 1), (F.INEXACT if exact and q != A.v else F(0))


def ref_round_int(A: Val, f: Fmt, rounding, exact: bool) -> Ref:
    if A.kind == "nan":
        return ref_nan(f, A)
    if A.kind == "inf":
        return Ref(_val_raw(A, f), F(0), set(), None)
    if A.v == 0:
        return Ref(int(A.sign) * f.sbit, F(0), set(), None)
    q = ref_int_round(A.v, rounding, -(1 << 200), 1 << 200)
    if q == 0:
        if not f.has_zero:
            return _no_zero(f)
        return Ref(int(A.sign) * f.sbit, F.INEXACT if exact else F(0), set(), None)
    return Ref(ref_round(Fraction(q), f._replace(rounding=Rounding.RNE, wrap=False,
                                                 ftz=False)).raw,
               F.INEXACT if exact and q != A.v else F(0), set(), None)


def ref_compare(A: Val, B: Val, signaling: bool):
    if A.kind == "nan" or B.kind == "nan":
        return "unordered", F.INVALID if (signaling or A.snan or B.snan) else F(0)
    lt, eq = ref_less(A, B)
    return ("lt" if lt else "eq" if eq else "gt"), F(0)


def caught(fn):
    """Run fn, returning (result, set of warning categories raised)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = fn()
    return r, {type(x.message) for x in w}


def assert_ref(got: FP, warned: set, want: Ref, unrounded=..., ctx="") -> None:
    assert got.raw == want.raw, f"raw {got.raw:#x} != {want.raw:#x} {ctx}"
    assert got.flags == want.flags, f"flags {got.flags!r} != {want.flags!r} {ctx}"
    assert warned == want.warnings, f"warnings {warned} != {want.warnings} {ctx}"
    expect = want.unrounded if unrounded is ... else unrounded
    if isinstance(expect, float) and math.isnan(expect):
        expect = None
    assert got.unrounded == expect, f"unrounded {got.unrounded} != {expect} {ctx}"


FP_ARITH = [
    ("+", lambda a, b: a + b),
    ("-", lambda a, b: a - b),
    ("*", lambda a, b: a * b),
    ("/", lambda a, b: a / b),
]


def check_fp_ops(a: int, b: int, f: Fmt) -> None:
    fa, fb = f.raw(a), f.raw(b)
    A, B = ref_decode(a, f), ref_decode(b, f)
    for name, op in FP_ARITH:
        ctx = f"{a:#x} {name} {b:#x} in {f}"
        want = ref_arith(name, A, B, f)
        if want is ZeroDivisionError:
            with pytest.raises(ZeroDivisionError):
                op(fa, fb)
            continue
        r, warned = caught(lambda: op(fa, fb))
        assert_ref(r, warned, want, ctx=ctx)
    ctx = f"a={a:#x} b={b:#x} {f}"
    if "nan" in (A.kind, B.kind):
        assert not (fa < fb) and not (fa > fb) and not (fa == fb) and (fa != fb), ctx
    else:
        lt, eq = ref_less(A, B)
        assert (fa < fb) == lt and (fa == fb) == eq and (fa != fb) == (not eq), ctx
        assert (fa > fb) == (not lt and not eq), ctx
    assert (fa & fb).raw == a & b, ctx
    assert (fa | fb).raw == a | b, ctx
    assert (fa ^ fb).raw == a ^ b, ctx


# Block-scaling reference. Small FP formats (elements, block scales) are
# rounded with the grid-based ref_round above; the FP32 tensor scale is too
# large to enumerate, so it uses FP.from_value, verified by the FP tests.
def fmt_of(f: FPFormat) -> Fmt:
    return Fmt(f.exp_bits, f.mantissa_bits, f.bias, f.signed, f.wrap, f.ftz,
               f.rounding, f.inf_nan, f.saturate, f.nan_mode, f.has_zero,
               f.sr_bits, f.tininess)


def ref_int_round(x: Fraction, mode: Rounding, lo: int, hi: int) -> int:
    ax = abs(x)
    m = int(ax)  # floor of the magnitude
    q = _choose(Fraction(m), Fraction(m + 1), ax, x < 0, mode, m % 2 == 0)
    q = int(-q if x < 0 else q)
    return min(max(q, lo), hi)


def _int_range(f):
    return (-(1 << (f.bits - 1)), (1 << (f.bits - 1)) - 1) if f.signed else (0, (1 << f.bits) - 1)


def _unit(f) -> Fraction:
    return Fraction(1, 1 << f.frac_bits)


def _elem_max(bf):
    e = bf.elem
    if isinstance(e, FPFormat):
        return finite_grid(fmt_of(e))[-1]
    return _int_range(e)[1] * _unit(e)


def _pow2_max_code(sf) -> int:
    return (1 << sf.bits) - 1 - int(sf.nan)


class RefBlocks(NamedTuple):
    elem_raw: list
    scale_raw: list
    zero_raw: list | None
    tensor_raw: int | None
    values: list


def ref_quantize(rows: list[list[Fraction]], bf) -> RefBlocks:
    from verifloat import Pow2Format  # noqa: local to keep the FP reference standalone
    emax = _elem_max(bf)
    sf = bf.scale

    def c(x):  # one step of the recipe's intermediate arithmetic
        return x if bf.compute is None or x == 0 else FP.from_value(x, bf.compute).exact

    if sf is None:
        cap = Fraction(1)
    elif isinstance(sf, Pow2Format):
        cap = Fraction(2) ** (_pow2_max_code(sf) - sf.bias)
    else:
        cap = finite_grid(fmt_of(sf))[-1]
    if bf.scale_max is not None:
        cap = min(cap, bf.scale_max)
    tensor_raw, s_t = None, Fraction(1)
    if bf.tensor_scale is not None:
        amax = max(abs(x) for r in rows for x in r)
        ts = FP.from_value(c(amax / c(emax * cap)) if amax else 1, bf.tensor_scale)
        tensor_raw, s_t = ts.raw, ts.exact
    E_raw, S_raw, Z_raw, V = [], [], [] if bf.zero_point else None, []
    for row in rows:
        er, sr, zr, vr = [], [], [], []
        for start in range(0, len(row), bf.block_size):
            block = row[start:start + bf.block_size]
            z = 0
            if sf is None:
                s_b, sr_code = Fraction(1), None
            elif isinstance(sf, Pow2Format):
                amax = max(abs(x) for x in block) / s_t
                code = 0 if amax == 0 else min(max(
                    _floor_log2(amax) - _floor_log2(emax) + sf.bias, 0), _pow2_max_code(sf))
                s_b, sr_code = Fraction(2) ** (code - sf.bias), code
            else:
                sfmt = fmt_of(sf)
                if bf.zero_point is None:
                    span = c(max(abs(x) for x in block) / c(emax * s_t))
                else:
                    qlo, qhi = _int_range(bf.elem)
                    lo, hi = min(min(block), 0), max(max(block), 0)
                    span = c((hi - lo) / c((qhi - qlo) * _unit(bf.elem) * s_t))
                if span == 0 and bf.zero_scale is not None:
                    span = bf.zero_scale
                elif span and bf.scale_min is not None:
                    span = max(span, bf.scale_min)
                sr_code = ref_round(min(span, cap), sfmt).raw
                s_b = ref_value(sr_code, sfmt)
                if bf.zero_point is not None:
                    zlo, zhi = _int_range(bf.zero_point)
                    z = 0 if s_b == 0 else ref_int_round(
                        qlo - lo / (s_b * s_t * _unit(bf.elem)),
                        bf.zero_point.rounding, zlo, zhi)
                    zr.append(z & ((1 << bf.zero_point.bits) - 1))
            sr.append(sr_code)
            step = c(s_b * s_t)
            for x in block:
                v = Fraction(0) if step == 0 else c(x / step)
                if isinstance(bf.elem, FPFormat):
                    ef = fmt_of(bf.elem)._replace(saturate=True, wrap=False)  # quantizers saturate
                    code = ref_round(v, ef).raw
                    val = ref_value(code, ef)
                else:
                    qlo, qhi = _int_range(bf.elem)
                    q = ref_int_round(v / _unit(bf.elem) + z, bf.elem.rounding, qlo, qhi)
                    code = q & ((1 << bf.elem.bits) - 1)
                    val = (q - z) * _unit(bf.elem)
                er.append(code)
                vr.append(s_b * s_t * val)      # the encoding's exact value
        E_raw.append(er)
        S_raw.append(sr)
        V.append(vr)
        if Z_raw is not None:
            Z_raw.append(zr)
    return RefBlocks(E_raw, S_raw, Z_raw, tensor_raw, V)


def rand_tensor(rng, shape) -> list[list[Fraction]]:
    """Constrained random values: per-block magnitudes spanning 2**-12..2**12,
    with zeros, all-zero blocks and outliers."""
    rows, cols = shape if len(shape) == 2 else (1, shape[0])
    out = []
    for _ in range(rows):
        row = []
        exp = rng.randint(-12, 12)
        for c in range(cols):
            if c % 8 == 0 and rng.random() < 0.3:  # re-draw the local scale
                exp = rng.randint(-12, 12)
            r = rng.random()
            if r < 0.1:
                x = Fraction(0)
            else:
                x = Fraction(rng.randint(-(1 << 20), 1 << 20), 1 << 20) * Fraction(2) ** exp
                if r > 0.97:
                    x *= 1 << rng.randint(4, 16)  # outlier
            row.append(x)
        if rng.random() < 0.1:
            row = [Fraction(0)] * cols
        out.append(row)
    return out


def rows_of(t) -> list[list]:
    return [t] if t and not isinstance(t[0], list) else t


def ref_matmul(a: list[list[Fraction]], b: list[list[Fraction]]) -> list[list[Fraction]]:
    return [[sum((x * y for x, y in zip(r, c)), Fraction(0)) for c in zip(*b)] for r in a]
