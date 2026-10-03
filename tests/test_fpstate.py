"""The processor's floating-point state must not change a result.

Other code in the same process can leave that state in any condition: a
library built with -ffast-math switches on flush-to-zero and
denormals-are-zero for the whole process when it is loaded, numerical code
changes the rounding mode, and MMX code that does not clean up leaves the x87
registers unusable. A golden model has to give the same bits regardless, so
the C++ core computes with integers only and reads and writes doubles through
their encoding.

Each test computes a broad set of results once in the default state and once
in a disturbed one, and compares them. Results are observed as raw codes and
as the bits of doubles: formatting a float is Python's own floating-point
code, which is not what is under test. x86-64 only (the state is set through
MXCSR and the x87 control word); needs gcc.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import struct
import subprocess
import warnings
from fractions import Fraction
from pathlib import Path

import pytest

if os.environ.get("VERIFLOAT_IMPL", "cpp") != "cpp":
    pytest.skip("the pure-Python 0.1 computes floats with the interpreter's own arithmetic",
                allow_module_level=True)
pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64") or not shutil.which("gcc"),
    reason="needs an x86-64 host and gcc")

from verifloat import (E4M3, FP16, FP32, FP64, NVFP4, Accumulator, BlockTensor, FixedFormat, FPArray,  # noqa: E402
                       FPFormat, Rounding, dot, dpi, matmul)

SRC = Path(__file__).parent / "native" / "fpstate.c"
DEFAULT_MXCSR, DEFAULT_CW = 0x1F80, 0x037F
DAZ, FTZ = 1 << 6, 1 << 15
STATES = {
    "flush-to-zero and denormals-are-zero (-ffast-math)": dict(mxcsr=DEFAULT_MXCSR | DAZ | FTZ),
    "SSE rounding toward +inf": dict(mxcsr=DEFAULT_MXCSR | (2 << 13)),
    "SSE rounding toward -inf": dict(mxcsr=DEFAULT_MXCSR | (1 << 13)),
    "SSE rounding toward zero, with FTZ and DAZ": dict(mxcsr=DEFAULT_MXCSR | (3 << 13) | DAZ | FTZ),
    "x87 at single precision, rounding toward zero": dict(cw=0x0C7F),
    "x87 rounding toward +inf": dict(cw=0x0B7F),
    "x87 registers left in use by MMX code": dict(mmx=True),
    "all of them": dict(mxcsr=DEFAULT_MXCSR | (2 << 13) | DAZ | FTZ, cw=0x0C7F, mmx=True),
}


@pytest.fixture(scope="module")
def fp():
    cache = Path(os.environ.get("VERIFLOAT_CACHE", Path.home() / ".cache" / "verifloat"))
    lib = cache / f"fpstate-{hashlib.sha256(SRC.read_bytes()).hexdigest()[:12]}.so"
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcc", "-O1", "-shared", "-fPIC", "-o", str(lib), str(SRC)], check=True)
    so = ctypes.CDLL(str(lib))
    so.get_mxcsr.restype = ctypes.c_uint
    so.get_x87_cw.restype = ctypes.c_ushort
    so.set_x87_cw.argtypes = [ctypes.c_ushort]
    return so


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def of_bits(b: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", b))[0]


# Doubles that exercise every way a float is read: subnormals of every size,
# the edges of the normal range, ordinary values, both signs. Built from bit
# patterns, before any state is disturbed.
FLOATS = [of_bits(b) for b in (
    0x0000000000000001, 0x8000000000000001, 0x0000000000000003, 0x000FFFFFFFFFFFFF, 0x800FFFFFFFFFFFFF,
    0x0000123456789ABC, 0x0010000000000000, 0x8010000000000001, 0x001FFFFFFFFFFFFF, 0x3FF0000000000000,
    0x3FF8000000000001, 0xBFE5555555555555, 0x4340000000000001, 0x7FEFFFFFFFFFFFFF, 0xFFEFFFFFFFFFFFFF,
    0x36A0000000000000, 0x3690000000000001, 0x380FFFFFFFFFFFFF, 0x0000000000000000, 0x8000000000000000)]
SMALL = [FLOATS[i] for i in (0, 1, 2, 5, 6, 9, 10, 11, 18, 19)]      # finite in every format used with them
FP128 = FPFormat(15, 112)
FORMATS = [FP16, FP32, FP64, FP128, E4M3, FP32.replace(rounding=Rounding.RUP), FP64.replace(rounding=Rounding.RTZ),
           FPFormat(11, 52, ftz=True), FPFormat(12, 150), FPFormat(8, 23, rounding=Rounding.RNA, tininess="before")]


def results(lib, arrays) -> dict:
    """A broad set of results, as integers only."""
    f32s, f64s, f16s = arrays
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in FORMATS:
            name = str(f)
            vals = [f(x) for x in FLOATS]
            out[name, "from float"] = [(v.raw, int(v.flags)) for v in vals]
            out[name, "to float"] = [bits(float(v)) for v in vals]
            out[name, "sqrt"] = [(r.raw, int(r.flags)) for r in (abs(v).sqrt() for v in vals)]
            out[name, "sqrt of codes"] = [(r.raw, int(r.flags)) for r in
                                          (f.from_raw(c).sqrt() for c in (1, 2, 3, f._mask, (f.bias << f.mantissa_bits) | 1,
                                                                          (f._top - 1) << f.mantissa_bits | f._mask))]
            a = vals[5]
            out[name, "number operands"] = [(r.raw, int(r.flags)) for x in FLOATS[:12] for r in
                                            (a + x, x - a, a * x, a.fma(x, FLOATS[2]), a.minimum(x))]
            out[name, "compare with numbers"] = [(a < x, a == x, a.compare(x)[0]) for x in FLOATS[:12]]
            out[name, "arithmetic"] = [(r.raw, int(r.flags)) for v, w in zip(vals, vals[3:])
                                       for r in (v + w, v * w, v.fma(w, v), v.remainder(w) if not w.is_zero else v)]
        for f in (FP16, FP32, FP64, E4M3):
            name = str(f)
            for label, src in (("list", FLOATS), ("float64 buffer", f64s), ("float32 buffer", f32s),
                               ("float16 buffer", f16s)):
                arr = FPArray(src, f)
                out[name, "array from", label] = (arr.raw, arr._bytes(1))
                out[name, "array values", label] = arr._bytes(2)                    # float64 values, as bytes
                out[name, "array to_float", label] = [bits(x) for x in arr.to_float()]
            arr = FPArray(FLOATS, f)
            out[name, "array ops"] = [(r.raw, r._bytes(1)) for r in (arr + arr, arr * 0.5, arr - FLOATS[5], arr.sqrt())]
            acc = Accumulator(f, "sequential")
            out[name, "dot"] = [dot(SMALL, [f(x) for x in SMALL], acc).raw, dot(SMALL, SMALL[::-1], acc).raw,
                                acc.sum_products([f(x) for x in SMALL], SMALL).raw]
            out[name, "matmul"] = matmul(FPArray([SMALL[:4], SMALL[4:8]], f), FPArray([SMALL[8:10]] * 4, f),
                                         acc=Accumulator(f, "pairwise")).raw
        out["exact dot"] = dot(SMALL, SMALL[::-1])
        fixed = FixedFormat(2, 1100)
        out["fixed"] = [fixed(x).val for x in FLOATS[:9]]
        t = BlockTensor.quantize([0.5, FLOATS[5], -1.75, 3.0, FLOATS[0], 0.0, 6.0, FLOATS[11]] * 2, NVFP4)
        out["block tensor"] = (t.elem_raw, t.scale_raw, int(t.flags))
        flags = ctypes.c_int(0)
        h = lib.vf_format(b"e11m52")
        out["C library"] = [(lib.vf_from_double(h, x, flags), flags.value, bits(lib.vf_to_double(h, bits(x))))
                            for x in FLOATS]
        h32 = lib.vf_format(b"e8m23")
        out["C library, binary32"] = [(lib.vf_from_double(h32, x, flags), flags.value) for x in FLOATS]
        out["C library, sqrt"] = [lib.vf_sqrt(h, bits(x) & ~(1 << 63), flags) for x in FLOATS]
    return out


@pytest.fixture(scope="module")
def clib():
    L = ctypes.CDLL(str(dpi.library()))
    L.vf_format.restype, L.vf_format.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
    PINT = ctypes.POINTER(ctypes.c_int)
    L.vf_from_double.restype, L.vf_from_double.argtypes = ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_double, PINT]
    L.vf_to_double.restype, L.vf_to_double.argtypes = ctypes.c_double, [ctypes.c_void_p, ctypes.c_uint64]
    L.vf_sqrt.restype, L.vf_sqrt.argtypes = ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint64, PINT]
    return L


@pytest.fixture(scope="module")
def buffers():
    """Float buffers of the three widths, made before any state is disturbed
    (NumPy's own conversions depend on that state)."""
    np = pytest.importorskip("numpy")
    f64s = np.array(FLOATS, dtype=np.float64)
    f32s = np.array([0x00000001, 0x80000003, 0x007FFFFF, 0x00800000, 0x3F800001, 0xBFC00000, 0x7F7FFFFF, 0x00012345],
                    dtype=np.uint32).view(np.float32)
    f16s = np.array([0x0001, 0x8003, 0x03FF, 0x0400, 0x3C01, 0xBE00, 0x7BFF, 0x0155], dtype=np.uint16).view(np.float16)
    return f32s, f64s, f16s


@pytest.fixture(scope="module")
def clean(fp, clib, buffers):
    assert (fp.get_mxcsr() & 0xFFC0, fp.get_x87_cw()) == (DEFAULT_MXCSR, DEFAULT_CW), "the test must start in the default state"
    return results(clib, buffers)


@pytest.mark.parametrize("state", STATES)
def test_results_do_not_depend_on_the_floating_point_state(fp, clib, buffers, clean, state):
    change = STATES[state]
    try:
        if "mxcsr" in change:
            fp.set_mxcsr(change["mxcsr"])
        if "cw" in change:
            fp.set_x87_cw(change["cw"])
        if change.get("mmx"):
            fp.dirty_mmx()
        got = results(clib, buffers)
    finally:
        fp.clean_mmx()
        fp.set_x87_cw(DEFAULT_CW)
        fp.set_mxcsr(DEFAULT_MXCSR)
    different = [k for k in clean if got[k] != clean[k]]
    assert not different, (state, different[:6], [(clean[k], got[k]) for k in different[:1]])


def test_the_disturbed_states_do_disturb(fp):
    """The states are real: in each, the processor itself computes differently."""
    tiny, one = of_bits(1), of_bits(0x3FF0000000000001)
    try:
        fp.set_mxcsr(DEFAULT_MXCSR | DAZ | FTZ)
        halved = (of_bits(0x0010000000000000) / 2, tiny + tiny)
    finally:
        fp.set_mxcsr(DEFAULT_MXCSR)
    assert halved == (0.0, 0.0) and tiny + tiny == of_bits(2)       # flushed, read as zero; normally exact
    try:
        fp.set_mxcsr(DEFAULT_MXCSR | (2 << 13))
        up = one * one
    finally:
        fp.set_mxcsr(DEFAULT_MXCSR)
    assert bits(up) == bits(one * one) + 1                           # rounded up instead of to nearest
    assert Fraction(tiny) == Fraction(1, 2 ** 1074)
