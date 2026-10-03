"""The C library (libverifloat, verifloat.h) against the Python package.

Both are built from the same core, but the library goes through its own
entry points, raw-code packing, format-name parser and error path, and it
always runs the general kernel. Every function is called through ctypes on
random formats and codes and must return what the Python object model
returns: the same code, the same flags, and an error exactly where Python
raises.
"""

from __future__ import annotations

import ctypes as C
import math
import os
import struct
import warnings

import pytest

if os.environ.get("VERIFLOAT_IMPL", "cpp") != "cpp":
    pytest.skip("the C library is part of the C++ package", allow_module_level=True)

from reference import rand_code, rand_fmt  # noqa: E402
from verifloat import FP16, FP32, FP64, FPFormat, NaNMode, Rounding, dpi, set_sr_source  # noqa: E402

U64, I64, INT, PINT, FMT = C.c_uint64, C.c_int64, C.c_int, C.POINTER(C.c_int), C.c_void_p
PW = C.POINTER(C.c_uint32)
VF_ERROR = 256
ROUNDINGS = [Rounding.RNE, Rounding.RNA, Rounding.RTZ, Rounding.RUP, Rounding.RDN]
OPS = ["add", "sub", "mul", "div", "rem", "fmod", "min", "max", "minnum", "maxnum", "sgnj", "sgnjn", "sgnjx",
       "fma", "sqrt", "neg", "abs", "next_up", "next_down", "logb"]
BINARY, UNARY = OPS[:13], OPS[14:]

PY = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
    "rem": lambda a, b: a.remainder(b), "fmod": lambda a, b: a.fmod(b),
    "min": lambda a, b: a.minimum(b), "max": lambda a, b: a.maximum(b),
    "minnum": lambda a, b: a.minimum_number(b), "maxnum": lambda a, b: a.maximum_number(b),
    "sgnj": lambda a, b: a.copysign(b), "sgnjn": lambda a, b: a.fsgnjn(b), "sgnjx": lambda a, b: a.fsgnjx(b),
    "fma": lambda a, b, c: a.fma(b, c),
    "sqrt": lambda a: a.sqrt(), "neg": lambda a: -a, "abs": lambda a: abs(a),
    "next_up": lambda a: a.next_up(), "next_down": lambda a: a.next_down(), "logb": lambda a: a.logb(),
}


@pytest.fixture(scope="module")
def lib():
    L = C.CDLL(str(dpi.library()))
    sig = {
        "vf_format": (FMT, [C.c_char_p]), "vf_format_size": (INT, [FMT]), "vf_format_name": (C.c_char_p, [FMT]),
        "vf_last_error": (C.c_char_p, []), "vf_version": (C.c_char_p, []),
        "vf_fma": (U64, [FMT, U64, U64, U64, PINT]),
        "vf_scaleb": (U64, [FMT, U64, I64, PINT]), "vf_fclass": (INT, [FMT, U64]),
        "vf_op": (U64, [FMT, INT, U64, U64, U64, PINT]),
        "vf_convert": (U64, [FMT, FMT, U64, PINT]),
        "vf_round_to_integral": (U64, [FMT, U64, INT, INT, PINT]),
        "vf_to_int": (U64, [FMT, U64, INT, INT, INT, INT, PINT]),
        "vf_from_int": (U64, [FMT, I64, PINT]), "vf_from_uint": (U64, [FMT, U64, PINT]),
        "vf_from_double": (U64, [FMT, C.c_double, PINT]), "vf_to_double": (C.c_double, [FMT, U64]),
        "vf_set_sr": (None, [U64]),
        "vf_op_w": (None, [FMT, INT, PW, PW, PW, PW, PINT]), "vf_scaleb_w": (None, [FMT, PW, I64, PW, PINT]),
        "vf_fclass_w": (INT, [FMT, PW]), "vf_compare_w": (INT, [FMT, PW, PW, INT, PINT]),
        "vf_convert_w": (None, [FMT, FMT, PW, PW, PINT]),
        "vf_round_to_integral_w": (None, [FMT, PW, INT, INT, PW, PINT]),
        "vf_op128": (None, [FMT, INT, PW, PW, PW, PW, PINT]), "vf_scaleb128": (None, [FMT, PW, I64, PW, PINT]),
        "vf_fclass128": (INT, [FMT, PW]), "vf_compare128": (INT, [FMT, PW, PW, INT, PINT]),
        "vf_convert128": (None, [FMT, FMT, PW, PW, PINT]),
        "vf_round_to_integral128": (None, [FMT, PW, INT, INT, PW, PINT]),
    }
    for name in BINARY:
        sig[f"vf_{name}"] = (U64, [FMT, U64, U64, PINT])
    for name in UNARY:
        sig[f"vf_{name}"] = (U64, [FMT, U64, PINT])
    for name in ("compare", "eq", "lt", "le"):
        sig[f"vf_{name}"] = (INT, [FMT, U64, U64, INT, PINT])
    for name, (res, args) in sig.items():
        fn = getattr(L, name)
        fn.restype, fn.argtypes = res, args
    return L


def handle(lib, fmt: FPFormat):
    h = lib.vf_format(str(fmt).encode())
    assert h, lib.vf_last_error()
    return h


def model(fn):
    """The Python model's answer: (raw, flags) or the exception it raises."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = fn()
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            return e
    return r.raw, int(r.flags)


def agree(lib, got_raw, got_flags, want, ctx):
    if isinstance(want, Exception):
        assert got_flags == VF_ERROR and got_raw == 0, (ctx, want)
        assert lib.vf_last_error().decode() == str(want), ctx
    else:
        assert (got_raw, got_flags) == want, (ctx, hex(got_raw), got_flags, hex(want[0]), want[1])


def some_format(rng) -> FPFormat:
    """Small formats with every option, the IEEE formats, and anything up to 64 bits."""
    r = rng.random()
    if r < 0.5:
        return rand_fmt(rng).to_fpformat()
    if r < 0.7:
        return rng.choice([FP16, FP32, FP64]).replace(
            rounding=rng.choice(ROUNDINGS), nan_mode=rng.choice(list(NaNMode)),
            tininess=rng.choice(["before", "after"]))
    while True:
        E = rng.randint(1, 20)
        M = rng.randint(0, 63 - E)
        try:
            return FPFormat(E, M, None if rng.random() < 0.6 else rng.randint(-50, 1 << E),
                            signed=rng.random() < 0.8, inf_nan=rng.choice([True, True, False, "fn"]),
                            has_zero=rng.random() < 0.9, saturate=rng.random() < 0.2, wrap=rng.random() < 0.15,
                            ftz=rng.random() < 0.15, rounding=rng.choice(ROUNDINGS),
                            nan_mode=rng.choice(list(NaNMode)), tininess=rng.choice(["before", "after"]))
        except ValueError:
            continue


def some_code(rng, f: FPFormat) -> int:
    E, M = f.exp_bits, f.mantissa_bits
    r = rng.random()
    if r < 0.5:
        return rng.getrandbits(f.size)
    sign = (rng.getrandbits(1) << (E + M)) if f.signed else 0
    if r < 0.7:     # exponent near the middle: sums and products stay in range
        e = min(max(f.bias + rng.randint(-2, 2), 0), f._top)
        return sign | (e << M) | rng.getrandbits(M)
    field = rng.choice([0, 0, 1, f._top, f._top - 1 if f._top else 0])
    mant = rng.choice([0, 1, f._mask, f._mask >> 1, (f._mask >> 1) + 1 if M else 0, rng.getrandbits(M)])
    return sign | (field << M) | (mant & f._mask)


# ----------------------------------------------------------------- formats

def test_format_names_round_trip(lib, rng, iters):
    for _ in range(iters):
        f = some_format(rng) if rng.random() < 0.8 else FPFormat(
            rng.randint(2, 30), rng.randint(1, 300), rounding=rng.choice(list(Rounding)),
            sr_bits=rng.choice([8, 1, 13]))
        h = handle(lib, f)
        assert lib.vf_format_name(h).decode() == str(f)
        assert lib.vf_format_size(h) == f.size
        assert lib.vf_format(str(f).encode()) == h                      # one handle per format
        assert lib.vf_format(str(f).replace(", ", " ,  ").encode()) == h   # spacing is free


def test_format_errors(lib):
    for bad, why in (("", "not an FP format name"), ("fp32", "not an FP format name"),
                     ("e8", "not an FP format name"), ("e8m23x", "not an FP format name"),
                     ("e8m23, banana", "unknown format tag 'banana'"), ("e8m23, bias=x", "not an FP format name"),
                     ("e0m3", "need at least 1 exponent bit"), ("e3m0", "needs a mantissa bit"),
                     ("e1m3", "at least 2 exponent bits"), ("e61m3", "format too large"),
                     ("e8m23, sr_bits=0", "sr_bits must be >= 1"), ("e1m0, fn", "no positive finite value")):
        assert not lib.vf_format(bad.encode()), bad
        assert why in lib.vf_last_error().decode(), (bad, lib.vf_last_error())
        with pytest.raises(ValueError):                # the Python package refuses the same names
            FPFormat.parse(bad)
    assert not lib.vf_format(None)
    flags = C.c_int(0)
    assert lib.vf_add(None, 1, 2, flags) == 0 and flags.value == VF_ERROR
    assert b"null format" in lib.vf_last_error()
    assert lib.vf_version().decode().count(".") >= 1


# ----------------------------------------------------------------- operations

def test_arithmetic_matches_the_python_model(lib, rng, iters):
    flags = C.c_int(0)
    for _ in range(iters):
        f = some_format(rng)
        h = handle(lib, f)
        ca, cb, cc = (some_code(rng, f) for _ in range(3))
        a, b, c = f.from_raw(ca), f.from_raw(cb), f.from_raw(cc)
        for i, name in enumerate(OPS):
            n = 3 if name == "fma" else 2 if name in BINARY else 1
            args = (ca, cb, cc)[:n]
            want = model(lambda: PY[name](*(a, b, c)[:n]))
            got = getattr(lib, f"vf_{name}")(h, *args, flags)
            agree(lib, got, flags.value, want, (name, str(f), [hex(x) for x in args]))
            got = lib.vf_op(h, i, ca, cb, cc, flags)   # the same operation by number
            agree(lib, got, flags.value, want, ("vf_op", name, str(f)))


def test_scaleb_fclass_compare(lib, rng, iters):
    flags = C.c_int(0)
    rel = {"lt": 0, "eq": 1, "gt": 2, "unordered": 3}
    for _ in range(iters):
        f = some_format(rng)
        h = handle(lib, f)
        ca, cb = some_code(rng, f), some_code(rng, f)
        if rng.random() < 0.2:
            cb = ca
        a, b = f.from_raw(ca), f.from_raw(cb)
        n = rng.choice([rng.randint(-4, 4), rng.randint(-300, 300), rng.randint(-(1 << 62), 1 << 62)])
        got = lib.vf_scaleb(h, ca, n, flags)
        agree(lib, got, flags.value, model(lambda: a.scaleb(n)), ("scaleb", str(f), hex(ca), n))
        assert lib.vf_fclass(h, ca) == a.fclass()
        for signaling in (0, 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                want, wf = a.compare(b, signaling=bool(signaling))
                eq, lt, le = (getattr(a, m)(b, signaling=bool(signaling)) for m in ("eq", "lt", "le"))
            assert (lib.vf_compare(h, ca, cb, signaling, flags), flags.value) == (rel[want], int(wf))
            assert (lib.vf_eq(h, ca, cb, signaling, flags), flags.value) == (int(eq[0]), int(eq[1]))
            assert (lib.vf_lt(h, ca, cb, signaling, flags), flags.value) == (int(lt[0]), int(lt[1]))
            assert (lib.vf_le(h, ca, cb, signaling, flags), flags.value) == (int(le[0]), int(le[1]))
    assert lib.vf_scaleb(handle(lib, FP32), 0x3F800000, (1 << 62) + 1, flags) == 0 and flags.value == VF_ERROR


def test_conversions_match_the_python_model(lib, rng, iters):
    flags = C.c_int(0)
    for _ in range(iters):
        f, g = some_format(rng), some_format(rng)
        h, hg = handle(lib, f), handle(lib, g)
        ca = some_code(rng, f)
        a = f.from_raw(ca)
        got = lib.vf_convert(hg, h, ca, flags)
        agree(lib, got, flags.value, model(lambda: a.convert(g)), ("convert", str(f), str(g), hex(ca)))

        mode = rng.choice([None, *ROUNDINGS])
        exact = rng.random() < 0.5
        got = lib.vf_round_to_integral(h, ca, -1 if mode is None else ROUNDINGS.index(mode), exact, flags)
        agree(lib, got, flags.value, model(lambda: a.round_to_integral(mode, exact)),
              ("round_to_integral", str(f), hex(ca), mode, exact))

        bits, signed = rng.choice([1, 2, 8, 16, 31, 32, 33, 63, 64, rng.randint(1, 64)]), rng.random() < 0.6
        got = lib.vf_to_int(h, ca, bits, signed, -1 if mode is None else ROUNDINGS.index(mode), exact, flags)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v, fl = a.to_int(bits, signed, mode, exact)
        assert (got, flags.value) == (int(v) & ((1 << bits) - 1), int(fl)), (str(f), hex(ca), bits, signed, mode)

        v = rng.choice([0, 1, -1, rng.getrandbits(64) - (1 << 63), rng.randint(-5000, 5000),
                        (1 << rng.randint(0, 62)) + rng.randint(-2, 2)])
        v = max(min(v, (1 << 63) - 1), -(1 << 63))
        got = lib.vf_from_int(h, v, flags)
        agree(lib, got, flags.value, model(lambda: f(v)), ("from_int", str(f), v))
        u = v & ((1 << 64) - 1)
        got = lib.vf_from_uint(h, u, flags)
        agree(lib, got, flags.value, model(lambda: f(u)), ("from_uint", str(f), u))

        d = rng.choice([0.0, -0.0, math.inf, -math.inf, math.nan, 5e-324, 1.7976931348623157e308,
                        rng.uniform(-4, 4), rng.uniform(-1, 1) * 2.0 ** rng.randint(-1074, 1023),
                        float(rng.randint(-40, 40)) / 8])
        got = lib.vf_from_double(h, d, flags)
        agree(lib, got, flags.value, model(lambda: f(d)), ("from_double", str(f), d))

        back, want = lib.vf_to_double(h, ca), float(a)
        assert struct.pack("<d", back) == struct.pack("<d", want) or (math.isnan(back) and math.isnan(want)), \
            (str(f), hex(ca))
    assert lib.vf_to_int(handle(lib, FP32), 0, 65, 1, -1, 1, flags) == 0 and flags.value == VF_ERROR
    assert lib.vf_round_to_integral(handle(lib, FP32), 0, 9, 1, flags) == 0 and flags.value == VF_ERROR


def test_stochastic_rounding(lib, rng, iters):
    flags = C.c_int(0)
    try:
        for _ in range(iters):
            sr_bits = rng.choice([1, 4, 8, 13])
            f = FPFormat(rng.randint(2, 6), rng.randint(1, 12), rounding=Rounding.SR, sr_bits=sr_bits,
                         inf_nan=rng.choice([True, "fn", False]))
            h = handle(lib, f)
            ca, cb, cc = (some_code(rng, f) for _ in range(3))
            a, b, c = f.from_raw(ca), f.from_raw(cb), f.from_raw(cc)
            k = rng.getrandbits(sr_bits)
            lib.vf_set_sr(k)
            set_sr_source(lambda bits: k)
            for name in ("add", "sub", "mul", "div", "fma", "sqrt"):
                n = 3 if name == "fma" else 1 if name == "sqrt" else 2
                want = model(lambda: PY[name](*(a, b, c)[:n]))
                got = getattr(lib, f"vf_{name}")(h, *(ca, cb, cc)[:n], flags)
                agree(lib, got, flags.value, want, (name, str(f), hex(ca), hex(cb), hex(cc), k))
            x = FP32.from_raw(rng.getrandbits(32))
            got = lib.vf_convert(h, handle(lib, FP32), x.raw, flags)
            agree(lib, got, flags.value, model(lambda: x.convert(f)), ("convert", str(f), hex(x.raw), k))
            got = lib.vf_round_to_integral(h, ca, 5, 1, flags)
            agree(lib, got, flags.value, model(lambda: a.round_to_integral(Rounding.SR, True)),
                  ("round_to_integral sr", str(f), hex(ca), k))
    finally:
        set_sr_source(None)
        lib.vf_set_sr(0)


# ----------------------------------------------------------------- any width

def words(x: int, n: int):
    return (C.c_uint32 * n)(*[(x >> (32 * i)) & 0xFFFFFFFF for i in range(n)])


def value(w) -> int:
    return sum(int(v) << (32 * i) for i, v in enumerate(w))


def test_wide_formats(lib, rng, iters):
    flags = C.c_int(0)
    FP128 = FPFormat(15, 112)
    for _ in range(iters // 2):
        f = rng.choice([FP128, FP128, FP32, FPFormat(11, 53), FPFormat(15, 64), FPFormat(8, 200),
                        FPFormat(rng.randint(2, 20), rng.randint(1, 180), rounding=rng.choice(ROUNDINGS),
                                 nan_mode=rng.choice(list(NaNMode)))])
        h = handle(lib, f)
        n = (f.size + 31) // 32
        ca, cb, cc = (some_code(rng, f) for _ in range(3))
        a, b, c = f.from_raw(ca), f.from_raw(cb), f.from_raw(cc)
        out = (C.c_uint32 * n)()
        for i, name in enumerate(OPS):
            k = 3 if name == "fma" else 2 if name in BINARY else 1
            want = model(lambda: PY[name](*(a, b, c)[:k]))
            lib.vf_op_w(h, i, words(ca, n), words(cb, n), words(cc, n), out, flags)
            if isinstance(want, Exception):
                assert flags.value == VF_ERROR
            else:
                assert (value(out), flags.value) == want, (name, str(f), hex(ca), hex(cb), hex(cc))
        s = rng.randint(-40000, 40000)
        lib.vf_scaleb_w(h, words(ca, n), s, out, flags)
        assert (value(out), flags.value) == model(lambda: a.scaleb(s)), ("scaleb", str(f), hex(ca), s)
        assert lib.vf_fclass_w(h, words(ca, n)) == a.fclass()
        rel, fl = a.compare(b)
        assert (lib.vf_compare_w(h, words(ca, n), words(cb, n), 0, flags), flags.value) == \
            ({"lt": 0, "eq": 1, "gt": 2, "unordered": 3}[rel], int(fl))
        g = rng.choice([FP128, FP64, FPFormat(15, 64), f])
        outg = (C.c_uint32 * ((g.size + 31) // 32))()
        lib.vf_convert_w(handle(lib, g), h, words(ca, n), outg, flags)
        assert (value(outg), flags.value) == model(lambda: a.convert(g)), ("convert", str(f), str(g), hex(ca))
        mode = rng.choice(ROUNDINGS)
        lib.vf_round_to_integral_w(h, words(ca, n), ROUNDINGS.index(mode), 1, out, flags)
        assert (value(out), flags.value) == model(lambda: a.round_to_integral(mode, True))
        if f.size > 64:     # the 64-bit entry points refuse a wider format
            assert lib.vf_add(h, 1, 2, flags) == 0 and flags.value == VF_ERROR
            assert b"wider than 64 bits" in lib.vf_last_error()


def test_known_values(lib):
    """A few results worked out by hand, so the test does not rest on the
    Python model alone."""
    flags = C.c_int(0)
    fp32 = lib.vf_format(b"e8m23")
    assert lib.vf_add(fp32, 0x3F800000, 0x40000000, flags) == 0x40400000 and flags.value == 0   # 1 + 2 = 3
    assert lib.vf_div(fp32, 0x3F800000, 0x40400000, flags) == 0x3EAAAAAB and flags.value == 1   # 1/3, inexact
    assert lib.vf_div(fp32, 0x3F800000, 0, flags) == 0x7F800000 and flags.value == 8            # 1/0 = inf
    assert lib.vf_sqrt(fp32, 0xBF800000, flags) == 0x7FC00000 and flags.value == 16             # sqrt(-1)
    assert lib.vf_mul(fp32, 0x7F7FFFFF, 0x40000000, flags) == 0x7F800000 and flags.value == 5   # overflow
    rtz = lib.vf_format(b"e8m23, rtz")
    assert lib.vf_mul(rtz, 0x7F7FFFFF, 0x40000000, flags) == 0x7F7FFFFF and flags.value == 5
    assert lib.vf_to_int(fp32, 0xC0200000, 32, 1, 2, 1, flags) == 0xFFFFFFFE and flags.value == 1   # -2.5 -> -2
    assert lib.vf_from_double(fp32, 0.1, flags) == 0x3DCCCCCD and flags.value == 1
    assert lib.vf_to_double(fp32, 0x3DCCCCCD) == 0.10000000149011612
    assert lib.vf_fclass(fp32, 0x80000000) == 1 << 3 and lib.vf_fclass(fp32, 0x7FC00000) == 1 << 9
    e4m3 = lib.vf_format(b"e4m3, fn")
    assert lib.vf_format_size(e4m3) == 8
    assert lib.vf_add(e4m3, 0x7E, 0x7E, flags) == 0x7F and flags.value == 5                    # 448 + 448: NaN (no inf)


def test_128_bit_vectors(lib, rng, iters):
    """The *128 functions: any format of up to 128 bits in four words, every
    result word written (zero above the format, zero on an error)."""
    flags = C.c_int(0)
    junk = lambda: (C.c_uint32 * 4)(*[0xDEADBEEF] * 4)     # noqa: E731 - what a simulator may pass as `result`
    for _ in range(iters // 4):
        f = rng.choice([FPFormat(15, 112), FP16, FP64, FPFormat(15, 64), FPFormat(11, 90),
                        FPFormat(rng.randint(2, 20), rng.randint(1, 107), rounding=rng.choice(ROUNDINGS))])
        h = handle(lib, f)
        ca, cb, cc = (some_code(rng, f) for _ in range(3))
        a, b, c = f.from_raw(ca), f.from_raw(cb), f.from_raw(cc)
        for i, name in enumerate(OPS):
            k = 3 if name == "fma" else 2 if name in BINARY else 1
            want = model(lambda: PY[name](*(a, b, c)[:k]))
            out = junk()
            lib.vf_op128(h, i, words(ca, 4), words(cb, 4), words(cc, 4), out, flags)
            assert (value(out), flags.value) == ((0, VF_ERROR) if isinstance(want, Exception) else want), \
                (name, str(f), hex(ca), hex(cb), hex(cc))
        s, out = rng.randint(-40000, 40000), junk()
        lib.vf_scaleb128(h, words(ca, 4), s, out, flags)
        assert (value(out), flags.value) == model(lambda: a.scaleb(s))
        assert lib.vf_fclass128(h, words(ca, 4)) == a.fclass()
        rel, fl = a.compare(b, signaling=True)
        assert (lib.vf_compare128(h, words(ca, 4), words(cb, 4), 1, flags), flags.value) == \
            ({"lt": 0, "eq": 1, "gt": 2, "unordered": 3}[rel], int(fl))
        g, out = rng.choice([FPFormat(15, 112), FP32, FPFormat(15, 64), f]), junk()
        lib.vf_convert128(handle(lib, g), h, words(ca, 4), out, flags)
        assert (value(out), flags.value) == model(lambda: a.convert(g)), (str(f), str(g), hex(ca))
        mode, out = rng.choice(ROUNDINGS), junk()
        lib.vf_round_to_integral128(h, words(ca, 4), ROUNDINGS.index(mode), 1, out, flags)
        assert (value(out), flags.value) == model(lambda: a.round_to_integral(mode, True))
    wide, out = handle(lib, FPFormat(8, 200)), junk()
    lib.vf_op128(wide, 0, words(0, 4), words(0, 4), words(0, 4), out, flags)
    assert (value(out), flags.value) == (0, VF_ERROR) and b"wider than 128 bits" in lib.vf_last_error()
    out = junk()
    lib.vf_convert128(handle(lib, FP32), wide, words(0, 4), out, flags)
    assert (value(out), flags.value) == (0, VF_ERROR)
    assert lib.vf_fclass128(wide, words(0, 4)) == 0
