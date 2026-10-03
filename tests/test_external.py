"""Checks against references VeriFloat did not write.

* Known-answer tests: encodings published in IEEE 754-2019 and the OCP
  8-bit FP / Microscaling (MX) v1.0 specifications.
* Hardware IEEE 754: binary64/binary32 arithmetic executed by the host FPU
  (via numpy), including the overflow/underflow/divide/invalid flags.
* numpy float16 and Google's ml_dtypes (bfloat16, OCP FP8/FP6/FP4): these
  compute in float32 and round once, which is correctly rounded for these
  narrow formats (float32 has at least 2p+2 significand bits).
* Graphcore's gfloat: OCP MX block encoding (scale and element codes).

Where a reference uses a different but valid convention, the difference is
stated next to the check (e.g. x86 returns a negative default NaN).
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction

import pytest

np = pytest.importorskip("numpy")
md = pytest.importorskip("ml_dtypes")

from verifloat import (BF16, E2M1, E2M3, E3M2, E4M3, E5M2, FP16, FP32, FP64,  # noqa: E402
                       MXFP4, MXINT8, BlockFormat, BlockTensor, FPFlags, FPFormat)
from conftest import COMPARED  # noqa: E402
from reference import caught  # noqa: E402

NX, UF, OF = FPFlags.INEXACT, FPFlags.UNDERFLOW, FPFlags.OVERFLOW
DZ, NV = FPFlags.DIVZERO, FPFlags.INVALID


# Known-answer tests from the specifications
@pytest.mark.parametrize("fmt, raw, value", [
    # IEEE 754-2019 binary16
    (FP16, 0x3C00, 1), (FP16, 0xC000, -2), (FP16, 0x7BFF, 65504),
    (FP16, 0x0400, Fraction(1, 2**14)), (FP16, 0x0001, Fraction(1, 2**24)),
    (FP16, 0x3555, Fraction(1365, 4096)),              # nearest to 1/3
    (FP16, 0x7C00, math.inf), (FP16, 0xFC00, -math.inf), (FP16, 0x8000, 0),
    # binary32 / bfloat16
    (FP32, 0x3F800000, 1), (FP32, 0x7F7FFFFF, (2 - Fraction(1, 2**23)) * 2**127),
    (FP32, 0x00000001, Fraction(1, 2**149)), (FP32, 0x7F800000, math.inf),
    (BF16, 0x3F80, 1), (BF16, 0x7F7F, (2 - Fraction(1, 2**7)) * 2**127),
    # OCP FP8 E4M3: no inf, S.1111.111 is NaN, max 448
    (E4M3, 0x7E, 448), (E4M3, 0x08, Fraction(1, 64)), (E4M3, 0x01, Fraction(1, 512)),
    # OCP FP8 E5M2: IEEE-like, max 57344
    (E5M2, 0x7B, 57344), (E5M2, 0x04, Fraction(1, 2**14)), (E5M2, 0x01, Fraction(1, 2**16)),
    (E5M2, 0x7C, math.inf),
    # OCP MX v1.0 FP6 / FP4 (no inf/NaN)
    (E2M3, 0x1F, Fraction(15, 2)), (E2M3, 0x08, 1), (E2M3, 0x01, Fraction(1, 8)),
    (E3M2, 0x1F, 28), (E3M2, 0x04, Fraction(1, 4)), (E3M2, 0x01, Fraction(1, 16)),
    (E2M1, 0x7, 6), (E2M1, 0x2, 1), (E2M1, 0x1, Fraction(1, 2)),
])
def test_known_answers(fmt: FPFormat, raw, value):
    x = fmt.from_raw(raw)
    if isinstance(value, float):
        assert x.is_inf and float(x) == value
    else:
        assert x.exact == value
        assert fmt(value).raw == raw or value == 0     # and it encodes back


@pytest.mark.parametrize("fmt, max_raw, max_value", [
    (FP16, 0x7BFF, 65504), (BF16, 0x7F7F, (2 - Fraction(1, 2**7)) * 2**127),
    (FP32, 0x7F7FFFFF, (2 - Fraction(1, 2**23)) * 2**127),
    (E4M3, 0x7E, 448), (E5M2, 0x7B, 57344),
    (E2M3, 0x1F, Fraction(15, 2)), (E3M2, 0x1F, 28), (E2M1, 0x7, 6),
])
def test_known_max(fmt, max_raw, max_value):
    assert fmt.max == max_value and fmt.max_value().raw == max_raw
    r = fmt(max_value * 2)                         # overflow is flagged
    assert r.flags & OF


@pytest.mark.parametrize("fmt, nan_codes", [
    (E4M3, {0x7F, 0xFF}), (E5M2, {0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF}),
    (E2M1, set()), (E2M3, set()), (E3M2, set()),
])
def test_known_nan_encodings(fmt, nan_codes):
    assert {v.raw for v in fmt.all_values() if v.is_nan} == nan_codes


def test_known_rounding_results():
    assert FP16(1 / 3).raw == 0x3555 and FP32(1 / 3).raw == 0x3EAAAAAB
    assert BF16(Fraction(1, 3)).raw == 0x3EAB
    assert FP16(65520).raw == 0x7C00          # halfway to 2**16 rounds to inf (IEEE)
    assert E4M3(464).raw == 0x7E              # OCP: halfway to NaN code ties to 448


# Host FPU (IEEE 754 hardware) for binary64 / binary32
def rand_bits(rng, width: int, fmt: FPFormat) -> int:
    """Biased random encodings: specials, subnormals, extremes, close values."""
    r = rng.random()
    M = fmt.mantissa_bits
    top = (1 << fmt.exp_bits) - 1
    sign = rng.getrandbits(1) << (width - 1)
    if r < 0.1:
        return sign | rng.choice([0, 1, (1 << M) - 1, 1 << M, (top << M) - 1,
                                  top << M, (top << M) | 1, (top << M) | (1 << (M - 1))])
    if r < 0.4:  # moderate exponents, so + and - often cancel or round
        e = rng.randint(fmt.bias - 8, fmt.bias + 8)
        return sign | (e << M) | rng.getrandbits(M)
    if r < 0.5:  # near the subnormal range
        return sign | (rng.randint(0, 3) << M) | rng.getrandbits(M)
    return rng.getrandbits(width)


NP_FLAG = {"overflow": OF, "underflow": UF, "divide by zero": DZ, "invalid value": NV}


def hw_op(op, a, b):
    """Run a op b on the FPU; return (result array, flags it raised)."""
    raised = set()
    with np.errstate(all="call", call=lambda kind, _: raised.add(NP_FLAG[kind])):
        r = op(a, b)
    flags = FPFlags(0)
    for f in raised:
        flags |= f
    return r, flags


OPS = [("+", np.add, lambda x, y: x + y), ("-", np.subtract, lambda x, y: x - y),
       ("*", np.multiply, lambda x, y: x * y), ("/", np.divide, lambda x, y: x / y)]


@pytest.mark.parametrize("fmt, dtype, utype, width", [
    (FP64, np.float64, np.uint64, 64), (FP32, np.float32, np.uint32, 32)])
def test_host_fpu(rng, iters, fmt, dtype, utype, width):
    for _ in range(iters // 2):
        ra, rb = rand_bits(rng, width, fmt), rand_bits(rng, width, fmt)
        a = np.array([ra], dtype=utype).view(dtype)
        b = np.array([rb], dtype=utype).view(dtype)
        fa, fb = fmt.from_raw(ra), fmt.from_raw(rb)
        for name, npop, op in OPS:
            hw, hw_flags = hw_op(npop, a, b)
            got, _ = caught(lambda: op(fa, fb))
            ctx = f"{ra:#x} {name} {rb:#x}"
            hw_raw = int(hw.view(utype)[0])
            if math.isnan(hw[0]):
                # x86 returns a negative "default NaN"; VeriFloat's canonical
                # NaN is positive (RISC-V style). Both are NaN.
                assert got.is_nan, ctx
            else:
                assert got.raw == hw_raw, f"{ctx}: {got.raw:#x} != hw {hw_raw:#x}"
            # Flags: numpy reports all but INEXACT. x86 detects tininess
            # after rounding, VeriFloat's default too.
            mask = OF | UF | DZ | NV
            assert got.flags & mask == hw_flags & mask, \
                f"{ctx}: flags {got.flags!r} != hw {hw_flags!r}"


@pytest.mark.parametrize("fmt, dtype, utype", [(FP32, np.float32, np.uint32),
                                               (FP16, np.float16, np.uint16)])
def test_host_conversions(rng, iters, fmt, dtype, utype):
    for _ in range(iters):
        x = struct.unpack("<d", struct.pack("<Q", rand_bits(rng, 64, FP64)))[0]
        if not math.isfinite(x) and not math.isinf(x):
            continue
        with np.errstate(all="ignore"):
            hw = np.array([x], dtype=np.float64).astype(dtype)
        got = fmt(x)
        if math.isnan(x):
            assert got.is_nan
        else:
            assert got.raw == int(hw.view(utype)[0]), f"{x!r}"


# numpy float16 and ml_dtypes narrow formats
ML_FORMATS = [
    (FP16, np.float16, np.uint16),
    (BF16, md.bfloat16, np.uint16),
    (E5M2, md.float8_e5m2, np.uint8),
    (E4M3, md.float8_e4m3fn, np.uint8),                 # OCP FN: overflow -> NaN
    (FPFormat(4, 3), md.float8_e4m3, np.uint8),         # IEEE-style E4M3
    (FPFormat(3, 4), md.float8_e3m4, np.uint8),         # IEEE-style E3M4
    (E3M2, md.float6_e3m2fn, np.uint8),
    (E2M3, md.float6_e2m3fn, np.uint8),
    (E2M1, md.float4_e2m1fn, np.uint8),
]


def ml_from_raw(raw, dtype, utype):
    if dtype in (md.float6_e3m2fn, md.float6_e2m3fn, md.float4_e2m1fn):
        # Sub-byte types: build from the value (no raw view available).
        return None
    return np.array([raw], dtype=utype).view(dtype)


@pytest.mark.parametrize("fmt, dtype, utype", ML_FORMATS,
                         ids=[str(f[0]) for f in ML_FORMATS])
def test_ml_dtypes_encodings(fmt, dtype, utype):
    """Every encoding decodes to the same value as ml_dtypes."""
    for x in fmt.all_values():
        ref = np.array([float(x)], dtype=np.float32).astype(dtype).astype(np.float32)[0]
        if x.is_nan:
            assert math.isnan(ref), x
        else:
            assert float(ref) == float(x) and math.copysign(1, ref) == math.copysign(1, float(x)), x


@pytest.mark.parametrize("fmt, dtype, utype", ML_FORMATS,
                         ids=[str(f[0]) for f in ML_FORMATS])
def test_ml_dtypes_conversion_and_arithmetic(rng, iters, fmt, dtype, utype):
    values = [v for v in fmt.all_values()]
    finite_ops = fmt.inf_nan is False
    for _ in range(iters // 4):
        # Conversion of a random float32 (exact source, so no double rounding).
        x32 = np.float32(struct.unpack("<f", struct.pack("<I", rand_bits(rng, 32, FP32)))[0])
        if math.isfinite(x32) or fmt.has_nan:
            with np.errstate(all="ignore"):
                ref = np.array([x32], dtype=np.float32).astype(dtype).astype(np.float32)[0]
            if math.isnan(x32) and not fmt.has_nan:
                continue
            got, _ = caught(lambda: fmt(float(x32)))
            if math.isnan(ref):
                assert got.is_nan, f"{x32!r}"
            else:
                assert float(got) == float(ref) and \
                    math.copysign(1, float(got)) == math.copysign(1, float(ref)), f"{x32!r}"
        # Arithmetic on two random encodings.
        a, b = rng.choice(values), rng.choice(values)
        if (a.is_nan or b.is_nan) and not fmt.has_nan:
            continue
        na = np.array([float(a)], dtype=np.float32).astype(dtype)
        nb = np.array([float(b)], dtype=np.float32).astype(dtype)
        for name, npop, op in OPS:
            if name == "/" and finite_ops and b.is_zero:
                continue  # no inf/NaN to return: VeriFloat raises instead
            with np.errstate(all="ignore"):
                ref = npop(na, nb).astype(np.float32)[0]
            got, _ = caught(lambda: op(a, b))
            ctx = f"{a!r} {name} {b!r}"
            if math.isnan(ref):
                assert got.is_nan, ctx
            else:
                assert float(got) == float(ref), f"{ctx}: {float(got)} != {ref}"
                if float(ref) == 0:
                    assert math.copysign(1, float(got)) == math.copysign(1, float(ref)), ctx


@pytest.mark.parametrize("fmt, dtype, utype", [m for m in ML_FORMATS if m[0].size <= 8],
                         ids=[str(m[0]) for m in ML_FORMATS if m[0].size <= 8])
def test_ml_dtypes_every_pair(fmt, dtype, utype):
    """Formats of 8 bits or fewer: every pair of encodings through + - * /
    (65,536 pairs for an 8-bit format), value and sign of zero."""
    values = list(fmt.all_values())
    arr = np.array([float(v) for v in values], dtype=np.float32).astype(dtype)
    finite_ops = fmt.inf_nan is False
    bad = []
    for name, npop, op in OPS:
        with np.errstate(all="ignore"):
            ref = npop(arr[:, None], arr[None, :]).astype(np.float32)
        for i, a in enumerate(values):
            row = ref[i]
            for j, b in enumerate(values):
                if (a.is_nan or b.is_nan) and not fmt.has_nan:
                    continue
                if name == "/" and finite_ops and b.is_zero:
                    continue  # no inf/NaN to return: VeriFloat raises instead
                got, _ = caught(lambda: op(a, b))
                r = float(row[j])
                g = float(got)
                if not (got.is_nan if math.isnan(r) else g == r and math.copysign(1, g) == math.copysign(1, r)):
                    bad.append((repr(a), name, repr(b), g, r))
    COMPARED["ml_dtypes and NumPy (result)"] += 4 * len(values) ** 2
    assert not bad, (len(bad), bad[:5])


def boundary_floats(fmt: FPFormat, step: int = 1) -> list[float]:
    """Doubles on every rounding boundary of a format: each value, each
    midpoint of two neighbouring values, half the smallest value, and the
    first points past the largest (for big formats, every step-th value)."""
    vals = sorted({abs(float(v)) for v in fmt.all_values() if v.is_finite})
    out = [0.0]
    picked = [i for i in range(len(vals) - 1) if i % step == 0 or i < 40 or i > len(vals) - 40]
    for i in picked:
        out += [vals[i], (vals[i] + vals[i + 1]) / 2]
    top = vals[-1]
    out += [vals[0] / 2, vals[0] / 4, top, top + (top - vals[-2]) / 2, top + (top - vals[-2]), top * 2]
    return [x for x in dict.fromkeys(out) if math.isfinite(x)]


@pytest.mark.parametrize("fmt, dtype, utype", ML_FORMATS, ids=[str(f[0]) for f in ML_FORMATS])
def test_ml_dtypes_rounding_boundaries(fmt, dtype, utype):
    """Conversion from float32 on every boundary: each midpoint between
    neighbouring values and the float32 on either side of it, half the
    smallest value, and the first points past the largest. As single values
    and through the array kernels."""
    pts = []
    for x in boundary_floats(fmt):
        with np.errstate(all="ignore"):
            x32 = np.float32(x)
        if float(x32) != x:
            continue                        # not a float32 (bfloat16's overflow points)
        pts += [x32, np.nextafter(x32, np.float32(np.inf)), np.nextafter(x32, np.float32(-np.inf))]
    src = np.array(pts + [-p for p in pts], dtype=np.float32)
    src = src[np.isfinite(src)]
    with np.errstate(all="ignore"):
        ref = src.astype(dtype).astype(np.float32)
    if hasattr(fmt, "array"):
        got, _ = caught(lambda: fmt.array(src).to_numpy())
    else:                                   # the pure-Python 0.1 has no arrays: single values only
        got = [caught(lambda: float(fmt(float(x))))[0] for x in src]
    step = max(len(src) // 4000, 1)
    single = {i: caught(lambda: float(fmt(float(src[i]))))[0] for i in range(0, len(src), step)}
    bad = []
    for i, (x, g, r) in enumerate(zip(src, got, ref)):
        ok = math.isnan(g) if math.isnan(r) else g == r and math.copysign(1, g) == math.copysign(1, r)
        if i in single:
            s1 = single[i]
            ok = ok and (math.isnan(s1) if math.isnan(g) else s1 == g and math.copysign(1, s1) == math.copysign(1, g))
        if not ok:
            bad.append((float(x), float(g), float(r)))
    COMPARED["ml_dtypes and NumPy (result)"] += len(src)
    assert len(src) > 60 and not bad, (len(bad), len(src), bad[:5])


# Graphcore gfloat: OCP MX block encoding
gfloat = pytest.importorskip("gfloat")
from gfloat import compute_scale_amax, encode_block  # noqa: E402
from gfloat.formats import format_info_mxfp4_e2m1, format_info_mxint8  # noqa: E402


def rand_block(rng, n=32):
    exp = rng.randint(-20, 20)
    vals = [rng.uniform(-1, 1) * 2.0 ** exp for _ in range(n)]
    r = rng.random()
    if r < 0.1:
        vals = [0.0] * n
    elif r < 0.3:  # values right at powers of two stress floor(log2(amax))
        vals[rng.randrange(n)] = rng.choice([-1, 1]) * 2.0 ** (exp + 1)
    return vals


@pytest.mark.parametrize("ours, theirs", [(MXFP4, format_info_mxfp4_e2m1),
                                          (MXINT8, format_info_mxint8)],
                         ids=["MXFP4", "MXINT8"])
def test_gfloat_mx_blocks(rng, iters, ours: BlockFormat, theirs):
    for _ in range(max(iters // 20, 20)):
        vals = rand_block(rng)
        scale = compute_scale_amax(theirs.etype.emax, np.array(vals))
        codes = list(encode_block(theirs, scale, [v / scale for v in vals]))
        t = BlockTensor.quantize(vals, ours)
        assert t.scale_raw == [codes[0]], (vals, scale)
        assert t.elem_raw == codes[1:], vals


# gfloat as the reference for directed rounding, saturation and stochastic
# rounding on the small OCP formats (conversions of float64 values).
import gfloat.formats as gff  # noqa: E402
from gfloat import RoundMode, round_float  # noqa: E402

from verifloat import UE8M0, Rounding  # noqa: E402

GF_FORMATS = [(E2M1, gff.format_info_ocp_e2m1), (E2M3, gff.format_info_ocp_e2m3),
              (E3M2, gff.format_info_ocp_e3m2), (E4M3, gff.format_info_ocp_e4m3),
              (E5M2, gff.format_info_ocp_e5m2), (FP16, gff.format_info_binary16),
              (BF16, gff.format_info_bfloat16)]
GF_MODES = {Rounding.RNE: RoundMode.TiesToEven, Rounding.RNA: RoundMode.TiesToAway,
            Rounding.RTZ: RoundMode.TowardZero, Rounding.RUP: RoundMode.TowardPositive,
            Rounding.RDN: RoundMode.TowardNegative}


def rand_value(rng, fmt: FPFormat) -> float:
    """A float64 around the format's range, biased to ties and extremes."""
    r = rng.random()
    top = float(fmt.max)
    if r < 0.3:  # exactly halfway between two neighbouring encodings
        code = rng.randrange(fmt._max_code)
        lo, hi = float(fmt.from_raw(code)), float(fmt.from_raw(code + 1))
        return rng.choice([-1, 1]) * (lo + hi) / 2
    if r < 0.4:
        return rng.choice([-1, 1]) * rng.uniform(top, 4 * top)       # overflow
    if r < 0.5:
        return rng.choice([-1, 1]) * rng.uniform(0, 4 * float(fmt.min_normal))
    return rng.choice([-1, 1]) * rng.uniform(0, top)


def same(got, ref: float, ctx):
    COMPARED["gfloat (result)"] += 1
    if math.isnan(ref):
        assert got.is_nan, ctx
    else:
        assert float(got) == ref and math.copysign(1, float(got)) == math.copysign(1, ref), \
            f"{ctx}: {float(got)} != {ref}"


@pytest.mark.parametrize("fmt, fi", GF_FORMATS, ids=[str(f[0]) for f in GF_FORMATS])
@pytest.mark.parametrize("rounding", GF_MODES)
@pytest.mark.parametrize("sat", [False, True])
def test_gfloat_rounding_modes(rng, iters, fmt, fi, rounding, sat):
    if not sat and fmt.inf_nan is False:
        sat = True   # formats without inf/NaN always saturate
    f = fmt.replace(rounding=rounding, saturate=sat)
    for _ in range(max(iters // 10, 100)):
        x = rand_value(rng, fmt)
        got, _ = caught(lambda: f(x))
        same(got, round_float(fi, x, GF_MODES[rounding], sat), f"{x!r} {rounding} sat={sat}")


@pytest.mark.parametrize("fmt, fi", GF_FORMATS, ids=[str(f[0]) for f in GF_FORMATS])
@pytest.mark.parametrize("rounding", GF_MODES)
@pytest.mark.parametrize("sat", [False, True])
def test_gfloat_rounding_boundaries(fmt, fi, rounding, sat):
    """Every rounding boundary of the format (each midpoint and the doubles
    on either side of it, half the smallest value, the overflow points), in
    the five rounding modes, saturating or not. binary16 and bfloat16 take
    every 97th value."""
    if not sat and fmt.inf_nan is False:
        sat = True   # formats without inf/NaN always saturate
    f = fmt.replace(rounding=rounding, saturate=sat)
    # One point is not taken from gfloat 0.5: the double just below half the
    # smallest value, under ties-away. It is nearer to zero, and gfloat
    # returns the smallest value. SoftFloat gives zero for binary16
    # (f64_to_f16, near_maxMag: 0 with INEXACT|UNDERFLOW), as VeriFloat does
    # in every format.
    just_below_half = math.nextafter(float(fmt.min_subnormal if fmt.has_zero else 0) / 2, 0)
    n = 0
    for x in boundary_floats(fmt, 1 if fmt.size <= 8 else 97):
        for y in (x, math.nextafter(x, math.inf), math.nextafter(x, -math.inf)):
            for v in (y, -y):
                got, _ = caught(lambda: f(v))
                if rounding is Rounding.RNA and abs(v) == just_below_half and v != 0:
                    assert got.is_zero and math.copysign(1, float(got)) == math.copysign(1, v), v
                    continue
                same(got, round_float(fi, v, GF_MODES[rounding], sat), f"{v!r} {rounding} sat={sat}")
                n += 1
    assert n > 60


@pytest.mark.parametrize("fmt, fi", GF_FORMATS, ids=[str(f[0]) for f in GF_FORMATS])
@pytest.mark.parametrize("sr_bits", [1, 3, 8])
def test_gfloat_stochastic_rounding(rng, iters, fmt, fi, sr_bits):
    sat = fmt.inf_nan is False
    f = fmt.replace(rounding=Rounding.SR, sr_bits=sr_bits, saturate=sat)
    for _ in range(max(iters // 10, 100)):
        x, r = rand_value(rng, fmt), rng.getrandbits(sr_bits)
        got, _ = caught(lambda: f(x, sr_rand=r))
        ref = round_float(fi, x, RoundMode.Stochastic, sat, srbits=r, srnumbits=sr_bits)
        same(got, ref, f"SR {x!r} r={r}/{sr_bits}")


def test_e8m0():
    # Known answers (OCP MX v1.0): value 2**(code - 127), 0xFF is NaN, no zero.
    assert (UE8M0.from_raw(127).exact, UE8M0.from_raw(0).exact) == (1, Fraction(1, 2**127))
    assert UE8M0.from_raw(254).exact == 2**127 and UE8M0.from_raw(255).is_nan
    assert UE8M0.max == 2**127 and UE8M0.size == 8
    # Just above the smallest value 2**-127 rounds to it (ml_dtypes gives
    # 2**-126 for such float32-subnormal inputs; see below).
    assert UE8M0(Fraction(102, 100) * Fraction(1, 2**127)).raw == 0


def test_e8m0_vs_ml_dtypes(rng, iters):
    """ml_dtypes float8_e8m0fnu: zero, negatives and overflow are NaN; values
    below the minimum round up to it. It rounds exact ties up, whereas
    VeriFloat (like gfloat and IEEE P3109) breaks ties by code parity, so
    ties are compared with gfloat instead."""
    for _ in range(iters):
        e = rng.randint(-130, 126)
        x = np.float32(rng.choice([0.0, rng.uniform(0.5, 2) * 2.0 ** e, -1.0]))
        ref = np.array([x], dtype=np.float32).astype(md.float8_e8m0fnu).astype(np.float32)[0]
        mant = float(x) / 2.0 ** math.floor(math.log2(float(x))) if x > 0 else 0
        if mant == 1.5:
            continue   # exact tie: see test_e8m0_ties_vs_gfloat
        if 0 < x < 2.0 ** -126:
            continue   # ml_dtypes maps every float32 subnormal to 2**-126 (code 1)
        got, _ = caught(lambda: UE8M0(float(x)))
        same(got, float(ref), f"{x!r}")


def test_e8m0_ties_vs_gfloat():
    fi = gff.format_info_ocp_e8m0
    for e in range(-126, 126):
        x = 1.5 * 2.0 ** e
        same(UE8M0(x), round_float(fi, x), f"{x!r}")


@pytest.mark.parametrize("ours, theirs", [
    ("minimum", np.minimum), ("maximum", np.maximum),        # NaN propagates
    ("minimum_number", np.fmin), ("maximum_number", np.fmax)])  # NaN ignored
@pytest.mark.parametrize("fmt, dtype, utype, width", [
    (FP32, np.float32, np.uint32, 32), (FP64, np.float64, np.uint64, 64)])
def test_numpy_min_max(rng, iters, ours, theirs, fmt, dtype, utype, width):
    """numpy follows IEEE 754-2019 minimum/maximum and C fmin/fmax (=
    minimumNumber/maximumNumber) except that C leaves min(-0, +0) open, so
    pairs of zeros are skipped. Which NaN is returned is also not compared."""
    for _ in range(iters // 4):
        ra, rb = rand_bits(rng, width, fmt), rand_bits(rng, width, fmt)
        a, b = fmt.from_raw(ra), fmt.from_raw(rb)
        if a.is_zero and b.is_zero:
            continue
        if ours.endswith("_number") and (a.is_snan or b.is_snan):
            # IEEE 754-2019 minimumNumber (and RISC-V fmin) return the number
            # for a signaling NaN; C fmin keeps the 2008 minNum rule (NaN).
            continue
        with np.errstate(all="ignore"):
            ref = theirs(np.array([ra], utype).view(dtype), np.array([rb], utype).view(dtype))
        got = getattr(a, ours)(b)
        if math.isnan(ref[0]):
            assert got.is_nan, (ours, hex(ra), hex(rb))
        else:
            assert got.raw == int(ref.view(utype)[0]), (ours, hex(ra), hex(rb))
