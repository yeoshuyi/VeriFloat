"""Designed cases against Berkeley SoftFloat 3e.

tests/test_testfloat.py takes the operands TestFloat generates. Here the
operands are ours (tests/directed.py): built from each format's parameters to
sit on rounding ties and next to them, on the overflow threshold, at the
bottom of the normal range where the two tininess conventions disagree, in
the subnormal range, and on every kind of special value and NaN. SoftFloat
computes the expected result and flags for exactly those operands
(tests/native/softfloat_ref.c), for binary16, binary32, binary64 and
binary128, in the five rounding modes, with tininess detected before and
after rounding, and under the NaN conventions of RISC-V, x86 SSE and ARM.

Every way VeriFloat can compute a result is held to the same answers: the
general code (exact integers of any size), the scalar fast paths, the array
kernels at each SIMD level, and the C library.

A run takes every case, about 24 million operations in under a minute.
With VERIFLOAT_ITERS below its default (as on an emulated CPU), and under
the pure-Python implementation, each set is sampled instead, a different
part of it for each seed. Without gcc, make and (on the first run) network
access these tests are skipped.
"""

from __future__ import annotations

import collections
import ctypes
import functools
import os
import struct
import subprocess

import pytest

import directed as D
from conftest import COMPARED, ITERS, SEED
from verifloat import FP16, FP32, FP64, FPFormat, NaNMode, Rounding

IMPL = os.environ.get("VERIFLOAT_IMPL", "cpp")
if IMPL == "cpp":
    from verifloat import FPArray, _core, dpi

FP128 = FPFormat(15, 112)
FORMATS = {16: FP16, 32: FP32, 64: FP64, 128: FP128}
SPECS = {"RISCV": NaNMode.CANONICAL, "8086-SSE": NaNMode.X86, "ARM-VFPv2": NaNMode.ARM}
MODES = {Rounding.RNE: 0, Rounding.RTZ: 1, Rounding.RDN: 2, Rounding.RUP: 3, Rounding.RNA: 4}
TININESS = {"before": 0, "after": 1}
# Cases taken from each set: all of them, unless the run asked for fewer
# iterations than the default or uses the (slow) pure-Python implementation.
SAMPLE = None if ITERS >= (2000 if IMPL == "cpp" else 20000) else max(ITERS, 200)


@pytest.fixture(scope="session")
def softfloat():
    try:
        import testfloat_build
        return {spec: testfloat_build.softfloat_ref(spec) for spec in SPECS}
    except Exception as e:  # no network / compiler: skip, don't fail
        pytest.skip(f"SoftFloat unavailable: {e}")


# ---------------------------------------------------------------- the cases

@functools.lru_cache(maxsize=None)
def cases(width: int, name: str, other: int = 0):
    """A designed set of operand tuples (always three operands) for a format."""
    f = FORMATS[width]
    pad2 = lambda pairs: [(a, b, 0) for a, b in pairs]      # noqa: E731
    pad1 = lambda ones: [(a, 0, 0) for a in ones]           # noqa: E731
    edges = D.edge_codes(f)
    if name == "edge pairs":
        return [(a, b, 0) for a in edges for b in edges]
    if name == "edge triples":
        few = D.few_edges(f)
        return [(a, b, c) for a in few for b in few for c in few]
    if name == "edges":
        return pad1(edges)
    full = {"add": lambda: pad2(D.add_cases(f)), "mul": lambda: pad2(D.mul_cases(f)),
            "div": lambda: pad2(D.div_cases(f)), "rem": lambda: pad2(D.rem_cases(f)),
            "mulAdd": lambda: D.fma_cases(f), "sqrt": lambda: pad1(D.sqrt_cases(f)),
            "roundToInt": lambda: pad1(D.round_int_cases(f)),
            "convert": lambda: pad1(D.convert_cases(f, FORMATS[other])),
            "to_i32": lambda: pad1(D.to_int_cases(f, 32, True)), "to_ui32": lambda: pad1(D.to_int_cases(f, 32, False)),
            "to_i64": lambda: pad1(D.to_int_cases(f, 64, True)), "to_ui64": lambda: pad1(D.to_int_cases(f, 64, False)),
            "from_i32": lambda: pad1(D.from_int_cases(f, 32, True)),
            "from_ui32": lambda: pad1(D.from_int_cases(f, 32, False)),
            "from_i64": lambda: pad1(D.from_int_cases(f, 64, True)),
            "from_ui64": lambda: pad1(D.from_int_cases(f, 64, False))}[name]()
    if SAMPLE is None or len(full) <= SAMPLE:
        return full
    stride = len(full) // SAMPLE
    return full[SEED % stride::stride]         # an evenly spread part; another seed, another part


@functools.lru_cache(maxsize=None)
def reference(exe, width, op, rounding, tininess, exact, name, other=0):
    """SoftFloat's (result, flags) for every case of a set."""
    ops = cases(width, name, other)
    text = "".join(f"{a:x} {b:x} {c:x}\n" for a, b, c in ops)
    run = subprocess.run([str(exe), str(width), op, str(MODES[rounding]), str(TININESS[tininess]), str(int(exact))],
                         input=text, capture_output=True, text=True, check=True)
    out = [(int(r, 16), int(fl, 16)) for r, fl in (line.split() for line in run.stdout.splitlines())]
    assert len(out) == len(ops), run.stderr
    return out


def fmt_of(width, spec, rounding=Rounding.RNE, tininess="after") -> FPFormat:
    return FORMATS[width].replace(nan_mode=SPECS[spec], rounding=rounding, tininess=tininess)


def x86_fma_open_case(f, a, b, c) -> bool:
    """0 * inf + qNaN: SoftFloat's x86 model signals INVALID, x86 hardware
    does not (IEEE leaves it open). VeriFloat follows the hardware, which
    tests/test_hwfpu.py checks."""
    fa, fb, fc = f.from_raw(a), f.from_raw(b), f.from_raw(c)
    return fc.is_nan and not fc.is_snan and ((fa.is_inf and fb.is_zero) or (fa.is_zero and fb.is_inf))


BINARY = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
          "div": lambda a, b: a / b, "rem": lambda a, b: a.remainder(b)}
if IMPL != "cpp":
    del BINARY["rem"]           # version 0.1 has no remainder
COMPARE = {"eq": ("eq", False), "le": ("le", True), "lt": ("lt", True), "eq_signaling": ("eq", True),
           "le_quiet": ("le", False), "lt_quiet": ("lt", False)}


def scalar(f: FPFormat, op: str, ops):
    """VeriFloat's (result, flags) for every case, through the FP type."""
    raw = f.from_raw
    if op in BINARY:
        fn = BINARY[op]
        res = [fn(raw(a), raw(b)) for a, b, _ in ops]
    elif op == "mulAdd":
        res = [raw(a).fma(raw(b), raw(c)) for a, b, c in ops]
    elif op == "sqrt":
        res = [raw(a).sqrt() for a, _, _ in ops]
    else:
        method, signaling = COMPARE[op]
        out = []
        for a, b, _ in ops:
            v, fl = getattr(raw(a), method)(raw(b), signaling=signaling)
            out.append((int(v), int(fl)))
        return out
    return [(r.raw, int(r.flags)) for r in res]


def differences(ops, got, want, limit=5):
    bad = [(tuple(hex(v) for v in o), (hex(g[0]), g[1]), (hex(w[0]), w[1]))
           for o, g, w in zip(ops, got, want) if g != w]
    return len(bad), bad[:limit]


def check(ops, got, want, what):
    COMPARED["Berkeley SoftFloat on designed cases (result and flags)"] += len(ops)
    n, first = differences(ops, got, want)
    assert n == 0, f"{what}: {n} of {len(ops)} differ; operands, VeriFloat, SoftFloat: {first}"


@pytest.fixture(params=["general", "fast"])
def path(request):
    """The scalar operations on the general code, then on the fast paths."""
    if IMPL != "cpp":
        if request.param == "fast":
            pytest.skip("the pure-Python implementation has one path")
        yield request.param
        return
    was = _core.set_fast(request.param == "fast")
    yield request.param
    _core.set_fast(was)


# ---------------------------------------------------------------- the design itself

@pytest.mark.parametrize("width", [16, 32])
def test_the_designed_cases_reach_every_boundary(width):
    """Exact rational arithmetic on the full sets: each boundary an operation
    can reach is reached, many times."""
    f = FORMATS[width]

    def reached(pairs, fn):
        seen = collections.Counter()
        for ops in pairs:
            vals = [D.exact(f, c) for c in ops]
            if None not in vals:
                r = fn(*vals)
                if r is not None:
                    seen[D.classify(f, r)] += 1
        return seen
    rounding = {"exact", "tie", "just off a tie", "inexact"}
    subnormal = {"subnormal, exact", "subnormal, tie", "subnormal, inexact"}
    tiny = {"tiny, rounds to the smallest normal"}
    want = {
        # a sum below the normal range is exact, so it has no subnormal ties
        "add": (D.add_cases(f), lambda a, b: a + b,
                rounding | {"zero", "subnormal, exact", "over", "over, below the threshold", "over, on the threshold"}),
        "mul": (D.mul_cases(f), lambda a, b: a * b, rounding | subnormal | tiny | {"over"}),
        # a quotient of two p-bit numbers is never a tie in the normal range,
        # nor within a quarter ulp below the smallest normal
        "div": (D.div_cases(f), lambda a, b: a / b if b else None,
                (rounding - {"tie"}) | subnormal | {"over"}),
        "fma": (D.fma_cases(f), lambda a, b, c: a * b + c,
                rounding | subnormal | tiny | {"zero", "over", "over, below the threshold", "over, on the threshold"}),
    }
    for op, (sets, fn, classes) in want.items():
        seen = reached(sets, fn)
        assert classes <= set(seen), (op, classes - set(seen))
        assert all(seen[c] >= 12 for c in classes), (op, {c: seen[c] for c in classes if seen[c] < 12})
    for src in (64, 128):
        seen = collections.Counter(D.classify(f, v) for v in (D.exact(FORMATS[src], c)
                                                               for c in D.convert_cases(FORMATS[src], f)) if v is not None)
        assert rounding - {"just off a tie"} | subnormal | tiny | {
            "zero", "over", "over, below the threshold", "over, on the threshold"} <= set(seen), (src, seen)


# ---------------------------------------------------------------- arithmetic

@pytest.mark.parametrize("width", FORMATS)
@pytest.mark.parametrize("op", ["add", "sub", "mul", "div", "mulAdd"])
@pytest.mark.parametrize("rounding", MODES, ids=lambda r: r.value)
@pytest.mark.parametrize("tininess", TININESS)
def test_rounding_and_range_boundaries(softfloat, path, width, op, rounding, tininess):
    name = "add" if op == "sub" else op
    ops = cases(width, name)
    want = reference(softfloat["RISCV"], width, op, rounding, tininess, False, name)
    f = fmt_of(width, "RISCV", rounding, tininess)
    check(ops, scalar(f, op, ops), want, (op, str(f), path))


@pytest.mark.parametrize("width", FORMATS)
@pytest.mark.parametrize("op", ["sqrt", "rem"] if IMPL == "cpp" else ["sqrt"])
@pytest.mark.parametrize("rounding", MODES, ids=lambda r: r.value)
def test_sqrt_and_remainder(softfloat, path, width, op, rounding):
    ops = cases(width, op)
    want = reference(softfloat["RISCV"], width, op, rounding, "after", False, op)
    f = fmt_of(width, "RISCV", rounding)
    check(ops, scalar(f, op, ops), want, (op, str(f), path))


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("width", FORMATS)
@pytest.mark.parametrize("rounding", MODES, ids=lambda r: r.value)
def test_special_values_and_nans(softfloat, path, spec, width, rounding):
    """Every pair (for fma: triple) of zeros, subnormals, extremes,
    infinities and quiet and signaling NaNs, through every operation, under
    each NaN convention: which NaN comes back, with which payload and sign,
    the sign of zero results, and the flags."""
    exe = softfloat[spec]
    f = fmt_of(width, spec, rounding)
    pairs = cases(width, "edge pairs")
    for op in (*BINARY, *COMPARE):
        want = reference(exe, width, op, rounding, "after", False, "edge pairs")
        check(pairs, scalar(f, op, pairs), want, (spec, op, str(f), path))
    ones = cases(width, "edges")
    check(ones, scalar(f, "sqrt", ones), reference(exe, width, "sqrt", rounding, "after", False, "edges"),
          (spec, "sqrt", str(f), path))
    triples = cases(width, "edge triples")
    want = reference(exe, width, "mulAdd", rounding, "after", False, "edge triples")
    if spec == "8086-SSE":
        keep = [i for i, t in enumerate(triples) if not x86_fma_open_case(f, *t)]
        triples, want = [triples[i] for i in keep], [want[i] for i in keep]
    check(triples, scalar(f, "mulAdd", triples), want, (spec, "mulAdd", str(f), path))


# ---------------------------------------------------------------- conversions

@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("src, dst", [(s, d) for s in FORMATS for d in FORMATS if s != d])
@pytest.mark.parametrize("rounding", MODES, ids=lambda r: r.value)
@pytest.mark.parametrize("tininess", TININESS)
def test_float_to_float(softfloat, path, spec, src, dst, rounding, tininess):
    ops = cases(src, "convert", dst)
    want = reference(softfloat[spec], src, f"to_f{dst}", rounding, tininess, False, "convert", dst)
    fs, fd = fmt_of(src, spec, rounding, tininess), fmt_of(dst, spec, rounding, tininess)
    got = [(r.raw, int(r.flags)) for r in (fs.from_raw(a).convert(fd) for a, _, _ in ops)]
    check(ops, got, want, (spec, str(fs), "->", str(fd), path))


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("width", FORMATS)
@pytest.mark.parametrize("itype", ["i32", "ui32", "i64", "ui64"])
@pytest.mark.parametrize("rounding", MODES, ids=lambda r: r.value)
def test_float_to_integer(softfloat, path, spec, width, itype, rounding):
    """Around the integer type's limits and every half: the value, INEXACT
    when asked for, and what an invalid conversion returns on each platform."""
    bits, signed = int(itype.lstrip("ui")), itype[0] == "i"
    f = fmt_of(width, spec)
    ops = cases(width, f"to_{itype}")
    for exact in (False, True):
        want = reference(softfloat[spec], width, f"to_{itype}", rounding, "after", exact, f"to_{itype}")
        got = []
        for a, _, _ in ops:
            v, fl = f.from_raw(a).to_int(bits, signed, rounding, exact)
            got.append((v.raw, int(fl)))
        check(ops, got, want, (spec, str(f), itype, rounding, exact, path))


@pytest.mark.parametrize("width", FORMATS)
@pytest.mark.parametrize("itype", ["i32", "ui32", "i64", "ui64"])
@pytest.mark.parametrize("rounding", MODES, ids=lambda r: r.value)
def test_integer_to_float(softfloat, path, width, itype, rounding):
    bits, signed = int(itype.lstrip("ui")), itype[0] == "i"
    f = fmt_of(width, "RISCV", rounding)
    ops = cases(width, f"from_{itype}")
    want = reference(softfloat["RISCV"], width, f"from_{itype}", rounding, "after", False, f"from_{itype}")
    got = []
    for a, _, _ in ops:
        r = f(a - (1 << bits) if signed and a >> (bits - 1) else a)
        got.append((r.raw, int(r.flags)))
    check(ops, got, want, (str(f), itype, path))


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("width", FORMATS)
@pytest.mark.parametrize("rounding", MODES, ids=lambda r: r.value)
def test_round_to_integral(softfloat, path, spec, width, rounding):
    f = fmt_of(width, spec)
    ops = cases(width, "roundToInt")
    for exact in (False, True):
        want = reference(softfloat[spec], width, "roundToInt", rounding, "after", exact, "roundToInt")
        got = [(r.raw, int(r.flags)) for r in (f.from_raw(a).round_to_integral(rounding, exact=exact)
                                               for a, _, _ in ops)]
        check(ops, got, want, (spec, str(f), rounding, exact, path))


# ---------------------------------------------------------------- the other implementations

needs_cpp = pytest.mark.skipif(IMPL != "cpp", reason="arrays and the C library are part of the C++ package")
LEVELS = ["general", "scalar", *(_core.simd_available() if IMPL == "cpp" else [])]
ARRAY_OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
             "div": lambda a, b: a / b}


@pytest.fixture(params=LEVELS)
def level(request):
    was = (_core.set_fast(request.param != "general"), _core.simd())
    _core.set_simd("none" if request.param in ("general", "scalar") else request.param)
    yield request.param
    _core.set_fast(was[0])
    _core.set_simd(was[1])


def array_result(r):
    """(code, flags) of every element."""
    flags = r._bytes(1)
    return list(zip(struct.unpack(f"={len(flags)}Q", r._bytes(0)), flags))


@needs_cpp
@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("op", ARRAY_OPS)
@pytest.mark.parametrize("rounding", MODES, ids=lambda r: r.value)
@pytest.mark.parametrize("tininess", TININESS)
def test_array_kernels(softfloat, level, width, op, rounding, tininess):
    """The element-wise kernels (general, scalar, AVX2, AVX-512) on the same
    cases: each element's code and flags."""
    f = fmt_of(width, "RISCV", rounding, tininess)
    for name in ("add" if op == "sub" else op, "edge pairs"):
        ops = cases(width, name)
        want = reference(softfloat["RISCV"], width, op, rounding, tininess if name != "edge pairs" else "after",
                         False, name)
        if name == "edge pairs" and tininess == "before":
            continue
        a = FPArray.from_raw([o[0] for o in ops], f)
        b = FPArray.from_raw([o[1] for o in ops], f)
        check(ops, array_result(ARRAY_OPS[op](a, b)), want, (op, str(f), level, name))


@needs_cpp
@pytest.mark.parametrize("src, dst", [(s, d) for s in (16, 32, 64) for d in (16, 32, 64) if s != d])
@pytest.mark.parametrize("rounding", MODES, ids=lambda r: r.value)
@pytest.mark.parametrize("tininess", TININESS)
def test_array_conversion(softfloat, level, src, dst, rounding, tininess):
    ops = cases(src, "convert", dst)
    want = reference(softfloat["RISCV"], src, f"to_f{dst}", rounding, tininess, False, "convert", dst)
    fs, fd = fmt_of(src, "RISCV", rounding, tininess), fmt_of(dst, "RISCV", rounding, tininess)
    got = array_result(FPArray.from_raw([o[0] for o in ops], fs).convert(fd))
    check(ops, got, want, (str(fs), "->", str(fd), level))


@pytest.fixture(scope="module")
def lib():
    L = ctypes.CDLL(str(dpi.library()))
    U64, PINT, PW = ctypes.c_uint64, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint32)
    L.vf_format.restype, L.vf_format.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
    L.vf_op.restype, L.vf_op.argtypes = U64, [ctypes.c_void_p, ctypes.c_int, U64, U64, U64, PINT]
    L.vf_convert.restype, L.vf_convert.argtypes = U64, [ctypes.c_void_p, ctypes.c_void_p, U64, PINT]
    L.vf_op128.restype, L.vf_op128.argtypes = None, [ctypes.c_void_p, ctypes.c_int, PW, PW, PW, PW, PINT]
    return L


C_OPS = {"add": 0, "sub": 1, "mul": 2, "div": 3, "rem": 4, "mulAdd": 13, "sqrt": 14}     # VF_OP_* in verifloat.h


@needs_cpp
@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("width", FORMATS)
@pytest.mark.parametrize("op", C_OPS)
@pytest.mark.parametrize("rounding", [Rounding.RNE, Rounding.RDN, Rounding.RNA], ids=lambda r: r.value)
def test_c_library(softfloat, lib, spec, width, op, rounding):
    """libverifloat (what a SystemVerilog testbench calls through DPI-C)."""
    f = fmt_of(width, spec, rounding, "before")
    h = lib.vf_format(str(f).encode())
    assert h, str(f)
    flags = ctypes.c_int(0)
    sets = ["edge triples" if op == "mulAdd" else "edges" if op == "sqrt" else "edge pairs"]
    if spec == "RISCV":
        sets.append({"sub": "add"}.get(op, op))
    for name in sets:
        ops = cases(width, name)
        want = reference(softfloat[spec], width, op, rounding, "before", False, name)
        if spec == "8086-SSE" and op == "mulAdd":
            keep = [i for i, t in enumerate(ops) if not x86_fma_open_case(f, *t)]
            ops, want = [ops[i] for i in keep], [want[i] for i in keep]
        got = []
        if width <= 64:
            for a, b, c in ops:
                r = lib.vf_op(h, C_OPS[op], a, b, c, flags)
                got.append((r, flags.value))
        else:
            words = lambda x: (ctypes.c_uint32 * 4)(*[(x >> (32 * i)) & 0xFFFFFFFF for i in range(4)])   # noqa: E731
            out = (ctypes.c_uint32 * 4)()
            for a, b, c in ops:
                lib.vf_op128(h, C_OPS[op], words(a), words(b), words(c), out, flags)
                got.append((sum(int(v) << (32 * i) for i, v in enumerate(out)), flags.value))
        check(ops, got, want, (spec, op, str(f), name))
