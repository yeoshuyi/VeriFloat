"""The host's x86 SSE/FMA unit as a golden reference, driven directly through
MXCSR so that rounding modes, flush-to-zero (FTZ) and denormals-are-zero (DAZ)
can be selected. Compares result bits and all five exception flags.

VeriFloat's ``ftz=True`` models FTZ and DAZ together, with tininess detected
after rounding, which is what x86 does. NaN results follow NaNMode.X86.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

from verifloat import FP32, FP64, FPFlags, NaNMode, Rounding
from test_external import rand_bits

pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64") or not shutil.which("gcc"),
    reason="needs an x86-64 host and gcc")

SRC = Path(__file__).parent / "native" / "hwfpu.c"
RC = {Rounding.RNE: 0, Rounding.RDN: 1, Rounding.RUP: 2, Rounding.RTZ: 3}
OPS = {"add": 0, "sub": 1, "mul": 2, "div": 3, "sqrt": 4, "fma": 5}
MXCSR_FLAGS = [(0, FPFlags.INVALID), (2, FPFlags.DIVZERO), (3, FPFlags.OVERFLOW),
               (4, FPFlags.UNDERFLOW), (5, FPFlags.INEXACT)]   # bit 1 (DE) ignored


@pytest.fixture(scope="module")
def hw():
    cache = Path(os.environ.get("VERIFLOAT_CACHE", Path.home() / ".cache" / "verifloat"))
    digest = hashlib.sha256(SRC.read_bytes()).hexdigest()[:12]
    lib = cache / f"hwfpu-{digest}.so"
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcc", "-O1", "-shared", "-fPIC", "-o", str(lib), str(SRC)],
                       check=True)
    so = ctypes.CDLL(str(lib))
    for name, t in (("op32", ctypes.c_uint32), ("op64", ctypes.c_uint64)):
        f = getattr(so, name)
        f.restype = t
        f.argtypes = [ctypes.c_int, t, t, t, ctypes.c_uint, ctypes.c_int, ctypes.c_int,
                      ctypes.POINTER(ctypes.c_uint)]
    return so


def hw_run(hw, width, op, a, b, c, rounding, ftz):
    flags = ctypes.c_uint()
    fn = hw.op32 if width == 32 else hw.op64
    r = fn(OPS[op], a, b, c, RC[rounding], int(ftz), int(ftz), ctypes.byref(flags))
    out = FPFlags(0)
    for bit, flag in MXCSR_FLAGS:
        if flags.value >> bit & 1:
            out |= flag
    return r, out


def model_run(fmt, op, a, b, c):
    fa, fb, fc = fmt.from_raw(a), fmt.from_raw(b), fmt.from_raw(c)
    return {"add": lambda: fa + fb, "sub": lambda: fa - fb, "mul": lambda: fa * fb,
            "div": lambda: fa / fb, "sqrt": lambda: fa.sqrt(),
            "fma": lambda: fa.fma(fb, fc)}[op]()


@pytest.mark.parametrize("fmt, width", [(FP32, 32), (FP64, 64)], ids=["fp32", "fp64"])
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("rounding", RC)
@pytest.mark.parametrize("ftz", [False, True], ids=["ieee", "ftz_daz"])
def test_x86_sse(hw, rng, iters, fmt, width, op, rounding, ftz):
    if op == "fma" and not hw.has_fma():
        pytest.skip("CPU has no FMA")
    f = fmt.replace(rounding=rounding, ftz=ftz, nan_mode=NaNMode.X86)
    for _ in range(max(iters // 4, 200)):
        a, b, c = (rand_bits(rng, width, fmt) for _ in range(3))
        if ftz and rng.random() < 0.5:   # make subnormal inputs and outputs common
            a = (a & (1 << (width - 1))) | rng.getrandbits(fmt.mantissa_bits + 1)
        want, want_flags = hw_run(hw, width, op, a, b, c, rounding, ftz)
        got = model_run(f, op, a, b, c)
        ctx = f"{op} {a:#x} {b:#x} {c:#x} {rounding} ftz={ftz}"
        if op == "fma" and got.is_nan:
            # Which NaN an x86 FMA returns depends on the instruction form
            # (132/213/231) the compiler picked, so only check it is a NaN.
            assert fmt.from_raw(want).is_nan, ctx
        else:
            assert got.raw == want, f"{ctx}: model {got.raw:#x} != hw {want:#x}"
        assert got.flags == want_flags, f"{ctx}: model {got.flags!r} != hw {want_flags!r}"


@pytest.mark.parametrize("rounding", RC)
def test_fma_chain_signed_zeros(hw, rng, iters, rounding):
    """Accumulator('sequential') is an FMA chain: check it against the FPU on
    vectors full of +-0 and exact cancellations, where IEEE's zero-sign rules
    (and -0 under round-toward-negative) decide the result."""
    from verifloat import Accumulator
    if not hw.has_fma():
        pytest.skip("CPU has no FMA")
    f = FP32.replace(rounding=rounding, nan_mode=NaNMode.X86)
    acc = Accumulator(f, "sequential")
    pool = [0x0, 0x80000000, 0x3F800000, 0xBF800000, 0x40000000, 0xC0000000]
    for _ in range(max(iters // 10, 100)):
        n = rng.randint(1, 8)
        a = [rng.choice(pool) for _ in range(n)]
        b = [rng.choice(pool) for _ in range(n)]
        init = rng.choice([0x0, 0x80000000])
        s = init
        for x, y in zip(a, b):
            s, _ = hw_run(hw, 32, "fma", x, y, s, rounding, False)
        got = acc.sum_products([f.from_raw(x) for x in a], [f.from_raw(y) for y in b],
                               init=f.from_raw(init))
        assert got.raw == s, ([hex(x) for x in a], [hex(y) for y in b], hex(init), rounding)
