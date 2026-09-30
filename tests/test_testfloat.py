"""Berkeley TestFloat 3e vectors (with SoftFloat 3e as the reference).

testfloat_gen enumerates IEEE edge cases (zeros, subnormals, extremes,
NaNs, random significands) and SoftFloat computes the expected result and
exception flags. Three SoftFloat specializations fix the implementation-
defined parts (which NaN is returned, what an invalid float-to-integer
conversion gives), matching VeriFloat's NaNMode.CANONICAL (RISC-V),
NaNMode.X86 (8086-SSE) and NaNMode.ARM (ARM-VFPv2).

Every result bit pattern and every flag must match. The first run downloads
and builds TestFloat (see testfloat_build.py); without a C toolchain or
network access these tests are skipped.
"""

from __future__ import annotations

import itertools
import os
import subprocess

import pytest

from verifloat import FP16, FP32, FP64, INT, UINT, FPFlags, NaNMode, Rounding

N = int(os.environ.get("VERIFLOAT_TESTFLOAT_N", 300))

SPECS = {"RISCV": NaNMode.CANONICAL, "8086-SSE": NaNMode.X86, "ARM-VFPv2": NaNMode.ARM}
MODES = {Rounding.RNE: "-rnear_even", Rounding.RTZ: "-rminMag", Rounding.RDN: "-rmin",
         Rounding.RUP: "-rmax", Rounding.RNA: "-rnear_maxMag"}
FORMATS = {"f16": FP16, "f32": FP32, "f64": FP64}
INTS = {"i32": (32, True), "ui32": (32, False), "i64": (64, True), "ui64": (64, False)}


@pytest.fixture(scope="session")
def generators():
    try:
        import testfloat_build
        return {spec: testfloat_build.testfloat_gen(spec) for spec in SPECS}
    except Exception as e:  # no network / compiler: skip, don't fail
        pytest.skip(f"TestFloat unavailable: {e}")


def cases(gen, function, *opts, n=N):
    """n test cases spread across TestFloat's level-1 sequence."""
    proc = subprocess.Popen([str(gen), *opts, function], stdout=subprocess.PIPE, text=True)
    try:
        # Level-1 sequences are ordered. One-operand functions have only a
        # few hundred cases, so take them all; for two operands (~46k cases)
        # and fused multiply-add (millions) take an evenly spaced sample.
        unary = function.endswith(("sqrt", "roundToInt")) or "_to_" in function
        stride = 1 if unary else 997 if function.endswith("mulAdd") else 37
        limit = None if unary else n
        lines = itertools.islice(proc.stdout, 0, None, stride)
        return [line.split() for line in itertools.islice(lines, limit)]
    finally:
        proc.kill()
        proc.wait()


def fmt_for(name, spec, rounding, tininess="after"):
    return FORMATS[name].replace(nan_mode=SPECS[spec], rounding=rounding,
                                 tininess=tininess)


def check(got_raw, got_flags, want_raw, want_flags, ctx):
    assert (got_raw, int(got_flags)) == (want_raw, want_flags), \
        f"{ctx}: got {got_raw:#x} flags {int(got_flags):#x}, " \
        f"SoftFloat {want_raw:#x} flags {want_flags:#x}"


ARITH = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b,
         "mul": lambda a, b: a * b, "div": lambda a, b: a / b}


def run_arith(gen, spec, fname, op, rounding, tininess):
    fmt = fmt_for(fname, spec, rounding, tininess)
    for a, b, r, fl in cases(gen, f"{fname}_{op}", MODES[rounding], f"-tininess{tininess}"):
        x = ARITH[op](fmt.from_raw(int(a, 16)), fmt.from_raw(int(b, 16)))
        check(x.raw, x.flags, int(r, 16), int(fl, 16), f"{spec} {fname}_{op} {a} {b}")


# Arithmetic: every rounding mode and both tininess conventions (RISC-V),
# RNE for the x86 and ARM NaN conventions.
@pytest.mark.parametrize("fname", FORMATS)
@pytest.mark.parametrize("op", ARITH)
@pytest.mark.parametrize("rounding", MODES)
@pytest.mark.parametrize("tininess", ["after", "before"])
def test_arith_riscv(generators, fname, op, rounding, tininess):
    run_arith(generators["RISCV"], "RISCV", fname, op, rounding, tininess)


@pytest.mark.parametrize("spec", ["8086-SSE", "ARM-VFPv2"])
@pytest.mark.parametrize("fname", FORMATS)
@pytest.mark.parametrize("op", ARITH)
def test_arith_nan_conventions(generators, spec, fname, op):
    run_arith(generators[spec], spec, fname, op, Rounding.RNE, "after")


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("fname", FORMATS)
@pytest.mark.parametrize("rounding", MODES)
def test_fma(generators, spec, fname, rounding):
    if spec != "RISCV" and rounding is not Rounding.RNE:
        pytest.skip("NaN conventions are checked with RNE only")
    fmt = fmt_for(fname, spec, rounding)
    for a, b, c, r, fl in cases(generators[spec], f"{fname}_mulAdd", MODES[rounding]):
        fa, fb, fc = (fmt.from_raw(int(v, 16)) for v in (a, b, c))
        if (spec == "8086-SSE" and fc.is_nan and not fc.is_snan
                and ((fa.is_inf and fb.is_zero) or (fa.is_zero and fb.is_inf))):
            # 0 * inf + qNaN: SoftFloat's x86 model signals INVALID, x86
            # hardware does not (IEEE leaves it open); VeriFloat follows the
            # hardware, which test_hwfpu.py checks.
            continue
        x = fa.fma(fb, fc)
        check(x.raw, x.flags, int(r, 16), int(fl, 16), f"{spec} {fname}_mulAdd {a} {b} {c}")


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("fname", FORMATS)
@pytest.mark.parametrize("rounding", MODES)
def test_sqrt(generators, spec, fname, rounding):
    if spec != "RISCV" and rounding is not Rounding.RNE:
        pytest.skip("NaN conventions are checked with RNE only")
    fmt = fmt_for(fname, spec, rounding)
    for a, r, fl in cases(generators[spec], f"{fname}_sqrt", MODES[rounding]):
        x = fmt.from_raw(int(a, 16)).sqrt()
        check(x.raw, x.flags, int(r, 16), int(fl, 16), f"{spec} {fname}_sqrt {a}")


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("fname", FORMATS)
@pytest.mark.parametrize("rounding", MODES)
@pytest.mark.parametrize("exact", [False, True])
def test_round_to_int(generators, spec, fname, rounding, exact):
    fmt = fmt_for(fname, spec, Rounding.RNE)
    opt = "-exact" if exact else "-notexact"
    for a, r, fl in cases(generators[spec], f"{fname}_roundToInt", MODES[rounding], opt):
        x = fmt.from_raw(int(a, 16)).round_to_integral(rounding, exact=exact)
        check(x.raw, x.flags, int(r, 16), int(fl, 16), f"{spec} {fname}_roundToInt {a}")


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("fname", FORMATS)
@pytest.mark.parametrize("iname", INTS)
@pytest.mark.parametrize("rounding", MODES)
def test_float_to_int(generators, spec, fname, iname, rounding):
    bits, signed = INTS[iname]
    fmt = fmt_for(fname, spec, Rounding.RNE)
    for exact in (True, False):
        opt = "-exact" if exact else "-notexact"
        for a, r, fl in cases(generators[spec], f"{fname}_to_{iname}", MODES[rounding],
                              opt, n=N // 2):
            v, flags = fmt.from_raw(int(a, 16)).to_int(bits, signed, rounding, exact)
            check(v.raw, flags, int(r, 16), int(fl, 16), f"{spec} {fname}_to_{iname} {a}")


@pytest.mark.parametrize("fname", FORMATS)
@pytest.mark.parametrize("iname", INTS)
@pytest.mark.parametrize("rounding", MODES)
def test_int_to_float(generators, fname, iname, rounding):
    bits, signed = INTS[iname]
    fmt = fmt_for(fname, "RISCV", rounding)
    for a, r, fl in cases(generators["RISCV"], f"{iname}_to_{fname}", MODES[rounding]):
        v = (INT if signed else UINT)(int(a, 16), bits).val
        x = fmt(v)
        check(x.raw, x.flags, int(r, 16), int(fl, 16), f"{iname}_to_{fname} {a}")


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("src, dst", [(s, d) for s in FORMATS for d in FORMATS if s != d])
@pytest.mark.parametrize("rounding", MODES)
def test_float_to_float(generators, spec, src, dst, rounding):
    if spec != "RISCV" and rounding is not Rounding.RNE:
        pytest.skip("NaN conventions are checked with RNE only")
    fs, fd = fmt_for(src, spec, rounding), fmt_for(dst, spec, rounding)
    for a, r, fl in cases(generators[spec], f"{src}_to_{dst}", MODES[rounding]):
        x = fs.from_raw(int(a, 16)).convert(fd)
        check(x.raw, x.flags, int(r, 16), int(fl, 16), f"{spec} {src}_to_{dst} {a}")


COMPARE = {"eq": ("eq", False), "le": ("le", True), "lt": ("lt", True),
           "eq_signaling": ("eq", True), "le_quiet": ("le", False),
           "lt_quiet": ("lt", False)}


@pytest.mark.parametrize("fname", FORMATS)
@pytest.mark.parametrize("op", COMPARE)
def test_compare(generators, fname, op):
    fmt = fmt_for(fname, "RISCV", Rounding.RNE)
    method, signaling = COMPARE[op]
    for a, b, r, fl in cases(generators["RISCV"], f"{fname}_{op}"):
        res, flags = getattr(fmt.from_raw(int(a, 16)), method)(fmt.from_raw(int(b, 16)),
                                                               signaling=signaling)
        check(int(res), flags, int(r, 16), int(fl, 16), f"{fname}_{op} {a} {b}")
