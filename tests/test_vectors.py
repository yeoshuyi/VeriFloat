"""Test-vector export (verifloat.vectors): generation, file formats, verify, CLI.

Expected values always come from the model, so most checks here are about the
plumbing (round trips, layout, determinism, error handling). The independent
check on the numbers is the last section: real Berkeley TestFloat output must
pass ``verify`` (skipped without TestFloat, like test_testfloat.py).
"""

from __future__ import annotations

import io
import itertools
import json
import shutil
import subprocess
import sys
import warnings

import os

import pytest

if os.environ.get("VERIFLOAT_IMPL", "cpp") != "cpp":
    pytest.skip("vector export is part of the C++ package", allow_module_level=True)

from verifloat import (BF16, E2M1, E4M3, E5M2, FP16, FP32, FP64, UE4M3, UE8M0, FPFlags,
                       FPFormat, NaNMode, Rounding, set_sr_source, vectors as V)
from verifloat import fp as _fp
from verifloat.vectors import Vector, VectorFormatError, VectorSpec

NO_NAN = FPFormat(3, 2, inf_nan=False)
UNSIGNED = FPFormat(4, 3, signed=False)
SR8 = E4M3.replace(rounding=Rounding.SR, sr_bits=5)
VARIED = FPFormat(4, 3, inf_nan="fn", rounding=Rounding.RTZ, nan_mode=NaNMode.X86,
                  tininess="before")
ALL_FORMATS = [FP16, E4M3, E5M2, BF16, NO_NAN, UNSIGNED, UE8M0, E2M1, VARIED, SR8,
               FPFormat(1, 2, inf_nan="fn"), FPFormat(2, 0, inf_nan="fn"),
               FPFormat(5, 10, bias=12, ftz=True, rounding=Rounding.RUP)]


def kwargs_for(op: str, fmt: FPFormat | None = None) -> dict:
    kw = {"convert": {"to": E5M2}, "to_int": {"bits": 12}, "from_int": {"bits": 10},
          "round_to_integral": {}}.get(op, {})
    if op in ("to_int", "round_to_integral") and fmt is not None and fmt.rounding is Rounding.SR:
        kw = {**kw, "rounding": Rounding.RTZ}      # SR is refused for these two
    return kw


def collect(op, fmt, count, **kw):
    stream = V.generate(op, fmt, count, **{**kwargs_for(op, fmt), **kw})
    return stream, list(stream)


# ------------------------------------------------------------ generation
@pytest.mark.parametrize("op", V.OPS)
def test_every_op_generates_and_matches_model(op):
    """Each vector re-computes to itself, and mixed output exercises flags."""
    stream, vecs = collect(op, FP16, 300, **({"bits": 20} if op == "from_int" else {}))
    assert len(vecs) == 300 and stream.stats.generated == 300
    assert stream.stats.skipped == 0
    for v in vecs:
        assert stream.spec.compute(v.operands, v.sr) == v
    if op != "copysign":                    # sign injection raises no flag
        assert len({v.flags for v in vecs}) >= 2


@pytest.mark.parametrize("fmt", ALL_FORMATS, ids=str)
@pytest.mark.parametrize("style", ["random", "mixed", "edges"])
def test_every_format_style_runs(fmt, style):
    for op in V.OPS:
        count = 40 if style != "edges" else 25
        stream, vecs = collect(op, fmt, count, style=style)
        assert len(vecs) <= count
        if style != "edges":
            assert len(vecs) == count
        assert stream.stats.generated == len(vecs)


def test_aliases_and_spec_validation():
    assert VectorSpec("mulAdd", FP16).op == "fma"
    assert VectorSpec("roundToInt", FP16).op == "round_to_integral"
    assert VectorSpec("min", FP16).op == "minimum" and VectorSpec("maxnum", FP16).op == "maximum_number"
    with pytest.raises(ValueError, match="unknown operation"):
        VectorSpec("pow", FP16)
    with pytest.raises(TypeError, match="target format"):
        VectorSpec("convert", FP16)
    with pytest.raises(TypeError, match="does not apply"):
        VectorSpec("add", FP16, bits=8)
    with pytest.raises(TypeError, match="does not apply"):
        VectorSpec("add", FP16, to=FP32)
    with pytest.raises(TypeError, match="does not apply"):
        VectorSpec("sqrt", FP16, rounding=Rounding.RTZ)
    with pytest.raises(TypeError):
        VectorSpec("add", "e5m10")
    with pytest.raises(ValueError):
        VectorSpec("to_int", FP16, bits=0)
    s = VectorSpec("to_int", FP16)
    assert (s.bits, s.signed, s.rounding, s.exact) == (32, True, Rounding.RNE, True)
    with pytest.raises(ValueError, match="stochastic"):
        VectorSpec("to_int", SR8)
    with pytest.raises(ValueError, match="stochastic"):
        VectorSpec("round_to_integral", FP16, rounding=Rounding.SR)
    assert VectorSpec("to_int", SR8, rounding=Rounding.RTZ).rounding is Rounding.RTZ


def test_generate_argument_errors():
    with pytest.raises(ValueError, match="needs count"):
        V.generate("add", FP16)
    with pytest.raises(ValueError, match="style"):
        V.generate("add", FP16, 3, style="weird")
    with pytest.raises(ValueError):
        V.generate("add", FP16, -1)
    with pytest.raises(TypeError):
        V.generate("add", FP16, 3, seed="x")
    assert list(V.generate("add", FP16, 0)) == []


def test_determinism_by_seed():
    for op in ("add", "fma", "to_int", "from_int", "convert"):
        a = collect(op, FP16, 200, seed=7)[1]
        b = collect(op, FP16, 200, seed=7)[1]
        c = collect(op, FP16, 200, seed=8)[1]
        assert a == b and a != c
    # edges / exhaustive sampling too, and the seed is the only thing that matters
    a = collect("add", FP16, 100, style="edges", seed=3)[1]
    assert a == collect("add", FP16, 100, style="edges", seed=3)[1]
    assert a != collect("add", FP16, 100, style="edges", seed=4)[1]
    a = collect("add", FP16, 100, style="exhaustive", seed=3)[1]
    assert a == collect("add", FP16, 100, style="exhaustive", seed=3)[1]


def test_known_stream_is_stable():
    """Pins the random stream: a change here silently invalidates stored
    vector files, so it must be a deliberate decision."""
    vecs = collect("add", E4M3, 4, seed=1)[1]
    assert vecs == [Vector((0x01, 0x75), 0x75, 0x01), Vector((0x30, 0xB1), 0x98, 0x00),
                    Vector((0x81, 0x07), 0x06, 0x00), Vector((0x01, 0x00), 0x01, 0x00)]


def test_exhaustive_equals_direct_enumeration():
    fmt = FPFormat(2, 1)             # 4-bit IEEE-style: 16 codes
    for op, f in [("add", lambda a, b: a + b), ("mul", lambda a, b: a * b),
                  ("sub", lambda a, b: a - b), ("minimum", lambda a, b: a.minimum(b))]:
        stream, vecs = collect(op, fmt, None, style="exhaustive")
        want = []
        for a, b in itertools.product(range(16), repeat=2):
            r = f(fmt.from_raw(a), fmt.from_raw(b))
            want.append(Vector((a, b), r.raw, int(r.flags)))
        assert vecs == want
    stream, vecs = collect("fma", fmt, None, style="exhaustive")
    assert len(vecs) == 16 ** 3
    for (a, b, c), v in zip(itertools.product(range(16), repeat=3), vecs):
        r = fmt.from_raw(a).fma(fmt.from_raw(b), fmt.from_raw(c))
        assert v == Vector((a, b, c), r.raw, int(r.flags))
    stream, vecs = collect("sqrt", fmt, None, style="exhaustive")
    assert [v.operands[0] for v in vecs] == list(range(16))
    # integer conversions enumerate their integer domain
    stream, vecs = collect("from_int", E4M3, None, style="exhaustive", bits=8, signed=False)
    assert [v.operands[0] for v in vecs] == list(range(256))
    for v in vecs:
        r = E4M3(v.operands[0])
        assert (v.result, v.flags) == (r.raw, int(r.flags))


def test_exhaustive_counts_skipped_cases():
    fmt = NO_NAN                      # x/0 raises ZeroDivisionError here
    stream, vecs = collect("div", fmt, None, style="exhaustive")
    n = 1 << fmt.size
    zeros = [a for a in range(n) if fmt.from_raw(a).is_zero]
    assert stream.stats.skipped_by == {"ZeroDivisionError": n * len(zeros)}
    assert stream.stats.stimuli == n * n
    assert len(vecs) == n * n - n * len(zeros) == stream.stats.generated
    assert all(not fmt.from_raw(v.operands[1]).is_zero for v in vecs)
    assert "skipped" in str(stream.stats)


def test_random_styles_replace_skipped_stimuli():
    stream, vecs = collect("div", NO_NAN, 500)
    assert len(vecs) == 500
    assert stream.stats.skipped > 0 and set(stream.stats.skipped_by) <= {"ZeroDivisionError",
                                                                          "ValueError"}


def test_hopeless_operation_gives_up_clearly(monkeypatch):
    """If the model raises for (nearly) every stimulus, generation must stop."""
    def always_raises(self, operands, sr):
        raise ZeroDivisionError("always")
    monkeypatch.setattr(VectorSpec, "_compute", always_raises)
    with pytest.raises(RuntimeError, match="no usable vectors"):
        list(V.generate("div", NO_NAN, 5))


def test_exhaustive_and_edges_size_limit():
    with pytest.raises(ValueError, match="exhaustive.*limit"):
        list(V.generate("add", FP16, style="exhaustive"))
    with pytest.raises(ValueError, match="limit"):
        list(V.generate("fma", FP16, style="edges", limit=1000))
    # a count turns it into a sample of distinct stimuli in ascending order
    vecs = list(V.generate("add", FP16, 50, style="exhaustive"))
    keys = [v.operands[0] << 16 | v.operands[1] for v in vecs]
    assert keys == sorted(set(keys)) and len(keys) == 50
    # count above the set size gives the whole set
    assert len(list(V.generate("add", E2M1, 10 ** 6, style="exhaustive"))) == 256


@pytest.mark.parametrize("fmt", ALL_FORMATS, ids=str)
def test_edge_codes_are_valid_distinct_and_cover_specials(fmt):
    codes = V.edge_codes(fmt)
    assert codes == sorted(set(codes))
    assert all(0 <= c < 1 << fmt.size for c in codes)
    vals = [fmt.from_raw(c) for c in codes]
    has = lambda pred: any(pred(x) for x in vals)
    assert has(lambda x: x.raw == fmt.max_value().raw)
    if fmt.signed:
        assert has(lambda x: x.raw == fmt.max_value(True).raw)
    if fmt.has_zero:
        assert has(lambda x: x.is_zero and not x.sign) and has(lambda x: x.is_zero and x.sign) == fmt.signed
        if fmt.mantissa_bits:
            assert has(lambda x: x.raw == 1)
            assert has(lambda x: x.is_subnormal and x.raw == (1 << fmt.mantissa_bits) - 1)
    assert has(lambda x: x.is_inf) == fmt.has_inf
    assert has(lambda x: x.is_nan) == fmt.has_nan
    if fmt.inf_nan is True and fmt.mantissa_bits >= 2:
        assert has(lambda x: x.is_snan) and has(lambda x: x.is_nan and not x.is_snan)
    if fmt.inf_nan is True and fmt.mantissa_bits >= 2 and fmt.signed:
        assert has(lambda x: x.is_nan and x.sign)
    if fmt.bias in range(fmt._max_field + 1):
        assert has(lambda x: x.exact == 1)


def test_edges_style_contains_the_specials_in_operands():
    ops = {x for v in V.generate("add", FP16, style="edges") for x in v.operands}
    assert ops == set(V.edge_codes(FP16))
    for want in (0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x7BFF, 0xFBFF,
                 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0x7DFF):   # +-0, subnormals, 1, max, inf, qNaN, sNaNs
        assert want in ops
    # all pairs: edges for add is the full product of the edge list
    n = len(V.edge_codes(FP16))
    vecs = list(V.generate("add", FP16, style="edges"))
    assert len(vecs) == n * n
    assert {v.operands for v in vecs} == set(itertools.product(V.edge_codes(FP16), repeat=2))
    # and the results include every flag the edges can raise
    flags = {v.flags for v in vecs}
    assert {FPFlags.INVALID, FPFlags.INEXACT | FPFlags.OVERFLOW, 0} <= flags
    # fma edges are triples
    n = len(V.edge_codes(E2M1))
    assert len(list(V.generate("fma", E2M1, style="edges"))) == n ** 3
    # integer sources: edges include extremes and powers of two
    ints = {v.operands[0] for v in V.generate("from_int", FP16, style="edges", bits=16, signed=True)}
    assert {0, 1, 0x7FFF, 0x8000, 0xFFFF, 0x4000, 0x3FFF, 0x4001} <= ints


def test_mixed_style_hits_specials_boundaries_and_cancellation():
    _, vecs = collect("add", FP16, 3000)
    cls = lambda c: FP16.from_raw(c)
    ops = [x for v in vecs for x in v.operands]
    assert any(cls(x).is_inf for x in ops) and any(cls(x).is_snan for x in ops)
    assert any(cls(x).is_subnormal for x in ops) and any(cls(x).is_zero for x in ops)
    assert any(v.operands[0] ^ v.operands[1] == 0x8000 for v in vecs)           # exact cancellation
    assert any(FPFlags.OVERFLOW & v.flags for v in vecs)
    _, vecs = collect("mul", FP16, 3000)
    assert any(FPFlags.UNDERFLOW & v.flags for v in vecs)
    assert any(FPFlags.OVERFLOW & v.flags for v in vecs)
    _, vecs = collect("div", FP16, 3000)
    assert any(FPFlags.DIVZERO & v.flags for v in vecs)
    _, vecs = collect("fma", FP16, 3000)
    assert any(not FP16.from_raw(v.result).is_nan and FP16.from_raw(v.result).is_zero for v in vecs)
    _, vecs = collect("from_int", FP16, 3000, bits=32)
    assert any(v.flags == int(FPFlags.INEXACT) for v in vecs) and any(v.flags == 0 for v in vecs)
    _, vecs = collect("to_int", FP16, 3000, bits=8)
    assert {FPFlags.INVALID, FPFlags.INEXACT, 0} <= {v.flags for v in vecs}


def test_random_style_is_uniform_over_codes():
    _, vecs = collect("add", FPFormat(2, 1), 4000, style="random")
    seen = {x for v in vecs for x in v.operands}
    assert seen == set(range(16))


def test_warnings_are_counted_not_emitted():
    fmt = UNSIGNED                    # negative results clamp with a warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stream, vecs = collect("sub", fmt, 300)
    assert stream.stats.warned > 0 and len(vecs) == 300
    assert "warnings" in str(stream.stats)


# ------------------------------------------------------------ stochastic rounding
def test_stochastic_rounding_exports_the_random_bits():
    assert VectorSpec("add", SR8).uses_sr and not VectorSpec("minimum", SR8).uses_sr
    assert not VectorSpec("lt", SR8).uses_sr and not VectorSpec("add", E4M3).uses_sr
    assert VectorSpec("convert", E4M3, to=SR8).uses_sr and not VectorSpec("convert", SR8, to=E4M3).uses_sr
    assert VectorSpec("add", SR8).fields == (("a", 8), ("b", 8), ("sr", 5), ("result", 8), ("flags", 5))
    sentinel = lambda nbits: 0
    set_sr_source(sentinel)
    try:
        _, vecs = collect("from_int", SR8, 400, bits=12)
        assert _fp._sr_state["source"] is sentinel      # restored
        assert len({v.sr for v in vecs}) > 10 and all(0 <= v.sr < 32 for v in vecs)
        for v in vecs:                                   # the bits are what rounds the value
            val = v.operands[0] - (1 << 12) if v.operands[0] >> 11 else v.operands[0]
            r = SR8(val, sr_rand=v.sr)
            assert (v.result, v.flags) == (r.raw, int(r.flags))
        for op in ("add", "mul", "div", "fma", "sqrt"):
            stream, vecs = collect(op, SR8, 200)
            assert V.verify(_lines(stream.spec, vecs), stream.spec).ok
            assert any(v.sr for v in vecs)
        # same operands, different bits can round differently
        outs = {VectorSpec("mul", SR8).compute((0x39, 0x39), sr=s).result for s in range(32)}
        assert len(outs) == 2
        assert _fp._sr_state["source"] is sentinel
        # an exception inside the model still restores the source
        with pytest.raises(ZeroDivisionError):
            VectorSpec("div", FPFormat(3, 2, inf_nan=False, rounding=Rounding.SR)).compute((1, 0), sr=0)
        assert _fp._sr_state["source"] is sentinel
    finally:
        set_sr_source(0)
    with pytest.raises(ValueError, match="sr"):
        VectorSpec("add", SR8).compute((1, 2))
    with pytest.raises(ValueError, match="does not use"):
        VectorSpec("add", E4M3).compute((1, 2), sr=1)
    with pytest.raises(ValueError, match="sr"):
        VectorSpec("add", SR8).compute((1, 2), sr=32)


def _lines(spec, vecs, fmt="testfloat"):
    buf = io.StringIO()
    V.write(buf, vecs, spec, format=fmt)
    return buf.getvalue().splitlines()


# ------------------------------------------------------------ file formats
EXT = {"testfloat": "txt", "csv": "csv", "jsonl": "jsonl", "readmemh": "hex"}


@pytest.mark.parametrize("fmt_name", V.FORMATS)
@pytest.mark.parametrize("op", V.OPS)
def test_roundtrip_every_op_and_file_format(op, fmt_name, tmp_path):
    stream, vecs = collect(op, E4M3, 120, seed=2)
    path = tmp_path / ("v." + EXT[fmt_name])
    assert V.write(path, vecs, stream.spec, format=fmt_name) == len(vecs)
    assert list(V.read(path, stream.spec)) == vecs                    # format by extension
    assert list(V.read(path, stream.spec, fmt_name)) == vecs
    assert list(V.read(path.read_text().splitlines(), stream.spec, fmt_name)) == vecs
    rep = V.verify(path, stream.spec)
    assert rep.ok and rep.checked == len(vecs)


@pytest.mark.parametrize("fmt_name", V.FORMATS)
@pytest.mark.parametrize("fmt", [FP16, NO_NAN, UNSIGNED, UE8M0, SR8, VARIED, E2M1], ids=str)
def test_roundtrip_formats_with_headers(fmt, fmt_name, tmp_path):
    for op in ("add", "fma", "sqrt", "lt"):
        stream, vecs = collect(op, fmt, 80)
        p = tmp_path / "v.out"
        for header in (False, True):
            V.write(p, vecs, stream.spec, format=fmt_name, header=header)
            assert list(V.read(p, stream.spec, fmt_name)) == vecs
            assert V.verify(p, stream.spec, format=fmt_name).ok


def test_write_a_stream_directly_and_read_jsonl_without_spec(tmp_path):
    stream = V.generate("convert", E5M2, 50, seed=9, to=E4M3.replace(rounding=Rounding.RUP))
    p = tmp_path / "c.jsonl"
    V.write(p, stream)                                   # spec comes from the stream
    meta = V.read_meta(p)
    assert meta["op"] == "convert" and meta["seed"] == 9 and meta["style"] == "mixed"
    assert meta["fmt"] == str(E5M2) and meta["to"] == "e4m3, fn, rup"
    assert meta["verifloat_vectors"] == 1 and "version" in meta and meta["count"] == 50
    assert FPFormat.parse(meta["to"]) == stream.spec.to
    vecs = list(V.read(p))                               # no spec needed
    assert len(vecs) == 50
    rep = V.verify(p)                                    # nor for verify
    assert rep.ok and rep.spec == stream.spec
    assert V.verify(p, "convert", E5M2, to=stream.spec.to).ok
    with pytest.raises(ValueError, match="generated for"):
        V.verify(p, "convert", E5M2, to=E4M3)
    with pytest.raises(ValueError, match="generated for"):
        V.verify(p, "add", E5M2)
    with pytest.raises(ValueError, match="spec"):
        V.write(io.StringIO(), [])
    with pytest.raises(ValueError, match="spec"):
        list(V.read(["1 2 3"], format="testfloat"))


def test_jsonl_layout(tmp_path):
    stream, vecs = collect("add", FP16, 3, seed=1)
    lines = _lines(stream.spec, vecs, "jsonl")
    meta, *rows = [json.loads(x) for x in lines]
    assert meta["op"] == "add" and meta["fmt"] == "e5m10"
    assert set(rows[0]) == {"a", "b", "result", "flags"}
    assert rows[0]["a"] == f"{vecs[0].operands[0]:04X}" and len(rows[0]["flags"]) == 2
    assert len(rows) == 3


def test_testfloat_layout_matches_testfloat_gen_conventions():
    spec = VectorSpec("add", FP16)
    assert _lines(spec, [spec.compute((0x3C00, 0x4000))]) == ["3C00 4000 4200 00"]
    assert _lines(spec, [spec.compute((0x7C00, 0xFC00))]) == ["7C00 FC00 7E00 10"]
    spec = VectorSpec("fma", FP32)
    (line,) = _lines(spec, [spec.compute((0x3F800000, 1, 0))])
    assert line.split() == ["3F800000", "00000001", "00000000", "00000001", "00"]
    spec = VectorSpec("lt", FP16)                                     # booleans are one digit
    assert _lines(spec, [spec.compute((0x3C00, 0x4000))]) == ["3C00 4000 1 00"]
    assert _lines(spec, [spec.compute((0x4000, 0x3C00))]) == ["4000 3C00 0 00"]
    assert _lines(spec, [spec.compute((0x7E00, 0x3C00))]) == ["7E00 3C00 0 10"]
    spec = VectorSpec("to_int", FP16, bits=32)                        # integers are 8 digits
    assert _lines(spec, [spec.compute((0x4248,))]) == ["4248 00000003 01"]     # 3.14 -> 3, inexact
    spec = VectorSpec("to_int", FP16, bits=12, signed=True)           # 12 bits: 3 digits
    assert _lines(spec, [spec.compute((0xC000,))]) == ["C000 FFE 00"]          # -2 in 12 bits
    spec = VectorSpec("from_int", FP16, bits=32, signed=True)
    assert _lines(spec, [spec.compute((0xFFFFFFFF,))]) == ["FFFFFFFF BC00 00"]
    spec = VectorSpec("from_int", FP16, bits=32, signed=False)
    assert _lines(spec, [spec.compute((0xFFFFFFFF,))]) == ["FFFFFFFF 7C00 05"]
    spec = VectorSpec("add", E2M1)                                    # 4-bit: one digit per operand
    assert _lines(spec, [spec.compute((2, 2))]) == ["2 2 4 00"]
    spec = VectorSpec("add", FPFormat(5, 7))                          # 13 bits: 4 digits
    assert all(len(t) == 4 for t in _lines(spec, [spec.compute((1, 2))])[0].split()[:3])
    spec = VectorSpec("compare", FP16)
    assert _lines(spec, [spec.compute((0x3C00, 0x4000)), spec.compute((0x3C00, 0x3C00)),
                         spec.compute((0x4000, 0x3C00)), spec.compute((0x7E00, 0x3C00))]) == [
        "3C00 4000 0 00", "3C00 3C00 1 00", "4000 3C00 2 00", "7E00 3C00 3 00"]
    # no header, no comments by default: byte-for-byte TestFloat's layout
    stream, vecs = collect("mul", FP32, 20)
    text = "\n".join(_lines(stream.spec, vecs))
    assert "#" not in text and all(len(ln.split()) == 4 for ln in text.splitlines())


def test_testfloat_header_is_opt_in_and_readable():
    spec = VectorSpec("add", FP16)
    vecs = [spec.compute((1, 2))]
    plain, buf = io.StringIO(), io.StringIO()
    V.write(plain, vecs, spec)
    V.write(buf, vecs, spec, header=True, meta={"note": "hello"})
    assert not plain.getvalue().startswith("#") and buf.getvalue().startswith("# ")
    assert "# op: add" in buf.getvalue() and "# note: hello" in buf.getvalue()
    assert list(V.read(buf.getvalue().splitlines(), spec)) == vecs


def test_csv_layout():
    spec = VectorSpec("fma", E4M3)
    lines = _lines(spec, [spec.compute((0x38, 0x38, 0x00))], "csv")
    assert lines == ["a,b,c,result,flags", "38,38,00,38,00"]
    spec = VectorSpec("add", SR8)
    assert _lines(spec, [spec.compute((0x38, 0x38), sr=3)], "csv")[0] == "a,b,sr,result,flags"
    with pytest.raises(VectorFormatError, match="header"):
        list(V.read(["x,y,z,result,flags", "1,2,3,4,5"], VectorSpec("fma", E4M3), "csv"))


def test_readmemh_packing_and_systemverilog():
    spec = VectorSpec("add", FP16)
    v = spec.compute((0x3C00, 0x4000))                                 # result 0x4200, flags 0
    assert spec.word_width == 16 + 16 + 16 + 5 == 53
    body = [ln for ln in _lines(spec, [v], "readmemh") if not ln.startswith("//")]
    assert body == ["%014X" % (0x3C00 << 37 | 0x4000 << 21 | 0x4200 << 5)]
    assert len(body[0]) == 14                                          # ceil(53/4)
    # the flags are the low 5 bits, the first operand the high bits
    spec = VectorSpec("add", FP16)
    v = spec.compute((0x7C00, 0xFC00))
    (word,) = [int(x, 16) for x in _lines(spec, [v], "readmemh") if not x.startswith("//")]
    assert word & 31 == 0x10 and (word >> 5) & 0xFFFF == 0x7E00 and word >> 37 == 0x7C00
    sv = V.systemverilog(spec)
    assert "VEC_W = 53" in sv and "VEC_A_LSB = 37" in sv and "VEC_B_LSB = 21" in sv
    assert "VEC_RESULT_LSB = 5" in sv and "VEC_FLAGS_LSB = 0" in sv
    assert "logic [15:0] a;" in sv and "logic [4:0] flags;" in sv and "typedef struct packed" in sv
    text = "\n".join(_lines(spec, [v], "readmemh"))
    assert text.startswith("// ") and "[52:37] a, 16 bits" in text and "op: add" in text
    # comments are legal $readmemh syntax: nothing but comments and hex words
    for ln in text.splitlines():
        assert ln.startswith("//") or all(c in "0123456789ABCDEF" for c in ln)
    # reader accepts several words per line, underscores, comments, lowercase
    w = "%014x" % word
    assert list(V.read([f"{w[:7]}_{w[7:]} {w} // c", ""], spec, "readmemh")) == [v, v]
    with pytest.raises(VectorFormatError, match="address"):
        list(V.read(["@10"], spec, "readmemh"))
    with pytest.raises(VectorFormatError, match="fit"):
        list(V.read(["F" * 14], spec, "readmemh"))


@pytest.mark.skipif(not shutil.which("verilator"), reason="verilator not installed")
def test_readmemh_and_snippet_load_in_systemverilog(tmp_path):
    """$readmemh into the generated struct yields the same fields as the testfloat file."""
    n = 12
    stream = V.generate("fma", E4M3, n, seed=2)
    vecs = list(stream)
    V.write(tmp_path / "v.hex", vecs, stream.spec)
    (tmp_path / "vec.svh").write_text(V.systemverilog(stream.spec))
    (tmp_path / "tb.sv").write_text(f"""module tb;
  `include "vec.svh"
  vec_t mem [{n}];
  initial begin
    $readmemh("v.hex", mem);
    for (int i = 0; i < {n}; i++)
      $display("%02X %02X %02X %02X %02X", mem[i].a, mem[i].b, mem[i].c, mem[i].result, mem[i].flags);
    $finish;
  end
endmodule
""")
    build = subprocess.run(["verilator", "--binary", "--timing", "-Wno-fatal", "--top-module", "tb",
                            "tb.sv"], cwd=tmp_path, capture_output=True, text=True)
    if build.returncode:
        pytest.skip(f"verilator could not build the testbench: {build.stderr[-300:]}")
    sim = subprocess.run([str(tmp_path / "obj_dir" / "Vtb")], cwd=tmp_path, capture_output=True,
                         text=True)
    got = [ln.upper() for ln in sim.stdout.splitlines() if ln[:1] in "0123456789abcdefABCDEF"][:n]
    assert got == _lines(stream.spec, vecs)


def test_read_errors_name_the_line():
    spec = VectorSpec("add", FP16)
    with pytest.raises(VectorFormatError, match="line 2: expected 4 fields"):
        list(V.read(["3C00 4000 4200 00", "3C00 4000 4200"], spec))
    with pytest.raises(VectorFormatError, match="line 1.*hex"):
        list(V.read(["3C0G 4000 4200 00"], spec))
    with pytest.raises(VectorFormatError, match="line 1.*fit"):
        list(V.read(["13C00 4000 4200 00"], spec))
    with pytest.raises(VectorFormatError, match="line 1.*fit"):
        list(V.read(["3C00 4000 4200 20"], spec))                      # flags are 5 bits
    with pytest.raises(VectorFormatError, match="metadata"):
        list(V.read(['{"a": "1"}'], spec, "jsonl"))
    with pytest.raises(VectorFormatError, match="keys"):
        list(V.read(['{"verifloat_vectors": 1}', '{"a": "1"}'], spec, "jsonl"))
    with pytest.raises(VectorFormatError, match="JSON"):
        list(V.read(["nope"], spec, "jsonl"))
    with pytest.raises(ValueError, match="format"):
        V.write(io.StringIO(), [], spec, format="xml")
    with pytest.raises(ValueError, match="does not fit"):
        V.write(io.StringIO(), [Vector((0x1FFFF, 0), 0, 0)], spec)
    assert list(V.read(["", "# comment", "3C00 4000 4200 00"], spec))[0].result == 0x4200


def test_format_detection():
    assert V.detect_format("x.csv") == "csv" and V.detect_format("x.JSONL") == "jsonl"
    assert V.detect_format("x.hex") == "readmemh" and V.detect_format("x.mem") == "readmemh"
    assert V.detect_format("x.txt") == "testfloat" and V.detect_format(["a"]) == "testfloat"


# ------------------------------------------------------------ verify
def corrupt(lines, fmt_name, spec, index, what):
    """Return the lines with vector ``index`` changed in its result or flags."""
    vecs = list(V.read(lines, spec, fmt_name))
    v = vecs[index]
    vecs[index] = (v._replace(result=v.result ^ 1) if what == "result"
                   else v._replace(flags=v.flags ^ 1))
    return _lines(spec, vecs, fmt_name)


@pytest.mark.parametrize("fmt_name", V.FORMATS)
@pytest.mark.parametrize("what", ["result", "flags"])
@pytest.mark.parametrize("op", ["add", "fma", "sqrt", "lt", "convert", "to_int", "from_int",
                                "compare"])
def test_verify_catches_corruption(op, what, fmt_name):
    stream, vecs = collect(op, E4M3, 60, seed=5)
    spec = stream.spec
    good = _lines(spec, vecs, fmt_name)
    assert V.verify(good, spec, format=fmt_name).ok
    bad = corrupt(good, fmt_name, spec, 17, what)
    out = io.StringIO()
    rep = V.verify(bad, spec, format=fmt_name, out=out)
    assert not rep.ok and rep.mismatch_count == 1 and rep.checked == 60
    (m,) = rep.mismatches
    (changed,) = [i for i, (g, b) in enumerate(zip(good, bad), 1) if g != b]
    assert m.line == changed                 # reported against the file's own line numbers
    assert m.expected == vecs[17]
    text = out.getvalue()
    assert "1 mismatches" in text and "model" in text and "file" in text


def test_verify_reports_the_values_and_flag_names():
    spec = VectorSpec("add", FP16)
    out = io.StringIO()
    rep = V.verify(["3C00 4000 4200 00", "7C00 FC00 7E00 00", "3C00 3C00 4000 01"], spec, out=out)
    assert rep.mismatch_count == 2 and rep.checked == 3 and not rep.ok
    text = out.getvalue()
    assert "line 2: add(0x7c00, 0xfc00): file 0x7e00 [none]; model 0x7e00 [INVALID]" in text
    assert "line 3: add(0x3c00, 0x3c00): file 0x4000 [INEXACT]; model 0x4000 [none]" in text
    assert "checked 3 vectors, 2 mismatches" in text


def test_verify_max_report_keeps_counting():
    spec = VectorSpec("add", FP16)
    rep = V.verify(["3C00 3C00 0000 00"] * 30, spec, max_report=5)
    assert rep.mismatch_count == 30 and len(rep.mismatches) == 5
    out = io.StringIO()
    V.verify(["3C00 3C00 0000 00"] * 30, spec, max_report=5, out=out)
    assert "25 more" in out.getvalue()


def test_verify_treats_model_errors_as_mismatches():
    spec = VectorSpec("div", NO_NAN)
    rep = V.verify(["08 00 00 00"], spec)
    assert rep.mismatch_count == 1 and "ZeroDivisionError" in rep.mismatches[0].describe(spec)


def test_verify_argument_errors():
    with pytest.raises(TypeError, match="fmt"):
        V.verify([], "add")
    with pytest.raises(TypeError, match="op and fmt"):
        V.verify([])
    assert V.verify([], VectorSpec("add", FP16)).checked == 0
    assert V.verify([], "add", FP16).ok


def test_verify_accepts_vectors_made_elsewhere():
    """Hand-written TestFloat-style lines (f16 1+1, 0/0, max+max RNE, RTZ)."""
    spec = VectorSpec("add", FP16)
    assert V.verify(["3C00 3C00 4000 00", "7BFF 7BFF 7C00 05", "0001 0001 0002 00"], spec).ok
    spec = VectorSpec("div", FP16)
    assert V.verify(["0000 0000 7E00 10", "3C00 0000 7C00 08"], spec).ok
    spec = VectorSpec("add", FP16.replace(rounding=Rounding.RTZ))
    assert V.verify(["7BFF 7BFF 7BFF 05"], spec).ok


# ------------------------------------------------------------ formats from names
def test_parse_format():
    assert V.parse_format("FP16") is FP16 and V.parse_format("E4M3") is E4M3
    assert V.parse_format("e4m3, fn") == E4M3
    assert V.parse_format("e4m3") == FPFormat(4, 3) != E4M3          # lowercase: IEEE-style
    assert V.parse_format("FP16, rtz") == FP16.replace(rounding=Rounding.RTZ)
    assert V.parse_format("E4M3, rtz, tininess=before") == E4M3.replace(
        rounding=Rounding.RTZ, tininess="before")
    assert V.parse_format("BF16, x86") == BF16.replace(nan_mode=NaNMode.X86)
    for fmt in ALL_FORMATS:
        assert V.parse_format(str(fmt)) == fmt
    with pytest.raises(ValueError):
        V.parse_format("nonsense")


def test_every_format_roundtrips_through_metadata(tmp_path):
    for fmt in ALL_FORMATS:
        for op in ("add", "convert", "to_int", "from_int", "round_to_integral"):
            spec = VectorSpec(op, fmt, **kwargs_for(op, fmt))
            assert VectorSpec.from_meta(json.loads(json.dumps(spec.to_meta()))) == spec


# ------------------------------------------------------------ operations semantics
def test_operations_match_the_documented_model_calls():
    f = FP16
    a, b, c = 0x4248, 0xC0A0, 0x3C00
    A, B, C = (f.from_raw(x) for x in (a, b, c))

    def check(op, want, ops=None, **kw):
        spec = VectorSpec(op, f, **kw)
        v = spec.compute(ops or (a, b, c)[:spec.arity])
        assert (v.result, v.flags) == want, op

    def pack(x):
        return (x.raw, int(x.flags))
    check("add", pack(A + B))
    check("sub", pack(A - B))
    check("mul", pack(A * B))
    check("div", pack(A / B))
    check("fma", pack(A.fma(B, C)))
    check("sqrt", pack(A.sqrt()))
    check("minimum", pack(A.minimum(B)))
    check("maximum", pack(A.maximum(B)))
    check("minimum_number", pack(A.minimum_number(B)))
    check("maximum_number", pack(A.maximum_number(B)))
    check("round_to_integral", pack(A.round_to_integral()))            # the model's default: not exact
    check("round_to_integral", pack(A.round_to_integral(Rounding.RNE, True)), exact=True)
    check("round_to_integral", pack(A.round_to_integral(Rounding.RTZ, False)),
          rounding=Rounding.RTZ, exact=False)
    r, fl = A.to_int(12, False, Rounding.RUP, False)
    check("to_int", (int(r), int(fl)), bits=12, signed=False, rounding=Rounding.RUP, exact=False)
    r, fl = B.to_int(12, True, Rounding.RNE, True)
    check("to_int", (int(r) & 0xFFF, int(fl)), ops=(b,), bits=12)
    for op, meth, sig in [("eq", "eq", False), ("eq_signaling", "eq", True), ("lt", "lt", True),
                          ("lt_quiet", "lt", False), ("le", "le", True), ("le_quiet", "le", False)]:
        for x, y in [(A, B), (B, A), (A, A)]:
            v = VectorSpec(op, f).compute((x.raw, y.raw))
            res, fl = getattr(x, meth)(y, signaling=sig)
            assert (v.result, v.flags) == (int(res), int(fl))
    qnan, snan = 0x7E00, 0x7D00
    assert VectorSpec("eq", f).compute((qnan, qnan)).flags == 0
    assert VectorSpec("eq_signaling", f).compute((qnan, qnan)).flags == FPFlags.INVALID
    assert VectorSpec("eq", f).compute((snan, 0)).flags == FPFlags.INVALID
    assert VectorSpec("lt", f).compute((qnan, 0)).flags == FPFlags.INVALID
    assert VectorSpec("lt_quiet", f).compute((qnan, 0)).flags == 0
    # convert: the flags of the conversion, in the target format's modes
    t = E4M3.replace(rounding=Rounding.RTZ)
    v = VectorSpec("convert", f, to=t).compute((a,))
    x = A.convert(t)
    assert (v.result, v.flags) == pack(x)
    # from_int: signed / unsigned interpretation of the same pattern
    s = VectorSpec("from_int", f, bits=8, signed=True).compute((0xFF,))
    u = VectorSpec("from_int", f, bits=8, signed=False).compute((0xFF,))
    assert (s.result, u.result) == (f(-1).raw, f(255).raw)


def test_format_modes_are_part_of_the_vectors():
    """Tininess and NaN convention change vectors: same operands, different flags / NaN."""
    fmt = FPFormat(3, 2)
    after = list(V.generate("mul", fmt, style="exhaustive"))
    before = list(V.generate("mul", fmt.replace(tininess="before"), style="exhaustive"))
    assert [v.operands for v in after] == [v.operands for v in before]
    assert [v.result for v in after] == [v.result for v in before]
    assert any(x.flags != y.flags for x, y in zip(after, before))
    canonical = VectorSpec("add", FP16).compute((0x7E01, 0x3C00))
    x86 = VectorSpec("add", FP16.replace(nan_mode=NaNMode.X86)).compute((0x7E01, 0x3C00))
    assert (canonical.result, x86.result) == (0x7E00, 0x7E01)


# ------------------------------------------------------------ CLI
def run(*args, stdin=None):
    return subprocess.run([sys.executable, "-m", "verifloat.vectors", *map(str, args)],
                          capture_output=True, text=True, input=stdin, timeout=120)


def test_cli_generate_then_verify(tmp_path):
    out = tmp_path / "add.txt"
    r = run("generate", "--op", "add", "--fmt", "e4m3, fn", "--count", 1000, "--seed", 1,
            "--style", "mixed", "--format", "testfloat", "-o", out)
    assert r.returncode == 0, r.stderr
    assert len(out.read_text().splitlines()) == 1000 and "1000 vectors" in r.stderr
    r = run("verify", out, "--op", "add", "--fmt", "e4m3, fn")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "checked 1000 vectors, 0 mismatches" in r.stdout
    # same seed, same file; the library API produces the very same text
    out2 = tmp_path / "add2.txt"
    assert run("generate", "--op", "add", "--fmt", "e4m3, fn", "--count", 1000, "--seed", 1,
               "-o", out2).returncode == 0
    assert out.read_text() == out2.read_text()
    assert out.read_text().splitlines() == _lines(*(lambda s: (s.spec, list(s)))(
        V.generate("add", FPFormat.parse("e4m3, fn"), 1000, seed=1)))


def test_cli_verify_fails_with_nonzero_status(tmp_path):
    out = tmp_path / "mul.txt"
    assert run("generate", "--op", "mul", "--fmt", "FP16", "--count", 200, "-o", out).returncode == 0
    lines = out.read_text().splitlines()
    a, b, res, fl = lines[3].split()
    lines[3] = f"{a} {b} {int(res, 16) ^ 1:04X} {fl}"
    bad = tmp_path / "bad.txt"
    bad.write_text("\n".join(lines) + "\n")
    r = run("verify", bad, "--op", "mul", "--fmt", "FP16")
    assert r.returncode == 1
    assert "line 4" in r.stdout and "1 mismatches" in r.stdout
    # wrong rounding mode: many vectors differ, still exit status 1
    r = run("verify", out, "--op", "mul", "--fmt", "FP16, rtz")
    assert r.returncode == 1 and "mismatches" in r.stdout
    # wrong operation entirely
    assert run("verify", out, "--op", "add", "--fmt", "FP16").returncode == 1


def test_cli_errors_have_distinct_status(tmp_path):
    assert run("verify", tmp_path / "missing.txt", "--op", "add", "--fmt", "FP16").returncode == 2
    junk = tmp_path / "junk.txt"
    junk.write_text("not a vector\n")
    r = run("verify", junk, "--op", "add", "--fmt", "FP16")
    assert r.returncode == 2 and "line 1" in r.stderr
    r = run("generate", "--op", "pow", "--fmt", "FP16", "--count", 3)
    assert r.returncode == 2 and "unknown operation" in r.stderr
    r = run("generate", "--op", "add", "--fmt", "FP16")                 # mixed needs --count
    assert r.returncode == 2 and "count" in r.stderr
    r = run("generate", "--op", "add", "--fmt", "FP16", "--style", "exhaustive")
    assert r.returncode == 2 and "limit" in r.stderr
    r = run("generate", "--op", "add", "--fmt", "garbage", "--count", 3)
    assert r.returncode == 2
    r = run("generate", "--op", "add", "--fmt", "FP16", "--count", 3, "--bits", 8)
    assert r.returncode == 2 and "does not apply" in r.stderr
    r = run("generate", "--op", "convert", "--fmt", "FP16", "--count", 3)
    assert r.returncode == 2 and "target format" in r.stderr
    assert run().returncode == 2


def test_cli_help_is_clear():
    r = run("--help")
    assert r.returncode == 0
    assert "generate" in r.stdout and "verify" in r.stdout and "operations:" in r.stdout
    assert "testfloat" in r.stdout and "readmemh" in r.stdout
    for sub in ("generate", "verify"):
        r = run(sub, "--help")
        assert r.returncode == 0 and "--op" in r.stdout and "--fmt" in r.stdout
    assert "--style" in run("generate", "--help").stdout
    assert "--max-report" in run("verify", "--help").stdout


@pytest.mark.parametrize("fmt_name, ext", [("csv", "csv"), ("jsonl", "jsonl"), ("readmemh", "hex"),
                                           ("testfloat", "txt")])
def test_cli_every_file_format(fmt_name, ext, tmp_path):
    out = tmp_path / f"v.{ext}"
    r = run("generate", "--op", "fma", "--fmt", "BF16, rtz", "--count", 100, "--seed", 4, "-o", out)
    assert r.returncode == 0, r.stderr                                 # format from extension
    r = run("verify", out, "--op", "fma", "--fmt", "BF16, rtz")
    assert r.returncode == 0, r.stdout
    out2 = tmp_path / "other.dat"                                      # explicit --format
    assert run("generate", "--op", "fma", "--fmt", "BF16, rtz", "--count", 100, "--seed", 4,
               "--format", fmt_name, "-o", out2).returncode == 0
    assert run("verify", out2, "--op", "fma", "--fmt", "BF16, rtz",
               "--format", fmt_name).returncode == 0
    assert list(V.read(out, VectorSpec("fma", BF16.replace(rounding=Rounding.RTZ)))) == \
        list(V.read(out2, VectorSpec("fma", BF16.replace(rounding=Rounding.RTZ)), fmt_name))


def test_cli_jsonl_is_self_describing_and_sidecars(tmp_path):
    out, meta, sv = tmp_path / "c.jsonl", tmp_path / "m.json", tmp_path / "w.svh"
    r = run("generate", "--op", "convert", "--fmt", "FP32", "--to", "E4M3, saturate",
            "--count", 80, "--seed", 3, "-o", out, "--meta", meta, "--sv", sv)
    assert r.returncode == 0, r.stderr
    r = run("verify", out)                                              # nothing else needed
    assert r.returncode == 0 and "checked 80" in r.stdout
    info = json.loads(meta.read_text())
    assert info["op"] == "convert" and info["seed"] == 3 and info["to"] == "e4m3, fn, saturate"
    assert "typedef struct packed" in sv.read_text()
    assert run("verify", out, "--op", "convert", "--fmt", "FP32", "--to", "E4M3").returncode == 2


def test_cli_readmemh_and_stdout(tmp_path):
    r = run("generate", "--op", "sqrt", "--fmt", "FP16", "--count", 5, "--format", "readmemh")
    assert r.returncode == 0
    assert r.stdout.startswith("// ") and "typedef struct packed" in r.stdout
    words = [ln for ln in r.stdout.splitlines() if not ln.startswith("//")]
    assert len(words) == 5 and all(len(w) == 10 for w in words)       # 16+16+5 = 37 bits
    r = run("generate", "--op", "sqrt", "--fmt", "FP16", "--count", 5)
    assert r.returncode == 0 and "#" not in r.stdout
    again = run("verify", "-", "--op", "sqrt", "--fmt", "FP16", stdin=r.stdout)
    assert again.returncode == 0 and "checked 5 " in again.stdout


def test_cli_int_options_and_edges_and_exhaustive(tmp_path):
    out = tmp_path / "t.txt"
    args = ["--op", "to_int", "--fmt", "E5M2", "--bits", 6, "--unsigned", "--rounding", "rtz",
            "--no-exact"]
    assert run("generate", *args, "--style", "exhaustive", "-o", out).returncode == 0
    assert len(out.read_text().splitlines()) == 256
    assert run("verify", out, *args).returncode == 0
    # the same file fails when verified as signed / exact
    assert run("verify", out, "--op", "to_int", "--fmt", "E5M2", "--bits", 6,
               "--rounding", "rtz", "--no-exact").returncode == 1
    r = run("generate", "--op", "add", "--fmt", "e2m1", "--style", "edges", "-o", out)
    assert r.returncode == 0 and "wrote" in r.stderr
    n = len(V.edge_codes(FPFormat(2, 1)))
    assert len(out.read_text().splitlines()) == n * n


# ------------------------------------------------------------ real TestFloat
TF_MODES = {Rounding.RNE: "-rnear_even", Rounding.RTZ: "-rminMag", Rounding.RDN: "-rmin",
            Rounding.RUP: "-rmax", Rounding.RNA: "-rnear_maxMag"}
TF_FORMATS = {"f16": FP16, "f32": FP32, "f64": FP64}
SPECS = {"RISCV": NaNMode.CANONICAL, "8086-SSE": NaNMode.X86, "ARM-VFPv2": NaNMode.ARM}


@pytest.fixture(scope="module")
def tf_gens():
    try:
        import testfloat_build
        return {spec: testfloat_build.testfloat_gen(spec) for spec in SPECS}
    except Exception as e:  # no network / compiler: skip, don't fail
        pytest.skip(f"TestFloat unavailable: {e}")


def tf_lines(gen, function, *opts, n=400):
    """n lines of testfloat_gen output, spread over its level-1 sequence."""
    proc = subprocess.Popen([str(gen), *opts, function], stdout=subprocess.PIPE, text=True)
    try:
        unary = function.endswith(("sqrt", "roundToInt")) or "_to_" in function
        stride = 1 if unary else 997 if function.endswith("mulAdd") else 37
        return list(itertools.islice(itertools.islice(proc.stdout, 0, None, stride),
                                     None if unary else n))
    finally:
        proc.kill()
        proc.wait()


def tf_fmt(name, spec, rounding):
    return TF_FORMATS[name].replace(nan_mode=SPECS[spec], rounding=rounding)


@pytest.mark.parametrize("rounding", TF_MODES)
@pytest.mark.parametrize("fname", TF_FORMATS)
@pytest.mark.parametrize("op, tf", [("add", "add"), ("mul", "mul"), ("div", "div"),
                                    ("fma", "mulAdd"), ("sqrt", "sqrt")])
def test_verify_accepts_real_testfloat_output(tf_gens, op, tf, fname, rounding):
    lines = tf_lines(tf_gens["RISCV"], f"{fname}_{tf}", TF_MODES[rounding])
    assert len(lines) > 100
    rep = V.verify(lines, op, tf_fmt(fname, "RISCV", rounding), format="testfloat")
    assert rep.ok, "\n".join(m.describe(rep.spec) for m in rep.mismatches)
    assert rep.checked == len(lines)


@pytest.mark.parametrize("spec", ["8086-SSE", "ARM-VFPv2"])
@pytest.mark.parametrize("fname", TF_FORMATS)
@pytest.mark.parametrize("op", ["add", "mul", "div", "sqrt"])
def test_verify_accepts_testfloat_nan_conventions(tf_gens, spec, fname, op):
    lines = tf_lines(tf_gens[spec], f"{fname}_{op}", "-rnear_even")
    assert V.verify(lines, op, tf_fmt(fname, spec, Rounding.RNE)).ok


@pytest.mark.parametrize("fname", TF_FORMATS)
def test_verify_accepts_testfloat_tininess_before(tf_gens, fname):
    fmt = TF_FORMATS[fname].replace(tininess="before")
    for op in ("mul", "div"):
        lines = tf_lines(tf_gens["RISCV"], f"{fname}_{op}", "-tininessbefore")
        assert V.verify(lines, op, fmt).ok


@pytest.mark.parametrize("fname", TF_FORMATS)
def test_verify_accepts_testfloat_conversions_and_comparisons(tf_gens, fname):
    gen = tf_gens["RISCV"]
    fmt = TF_FORMATS[fname]
    for tf, kw in [("to_i32", dict(bits=32)), ("to_ui32", dict(bits=32, signed=False)),
                   ("to_i64", dict(bits=64))]:
        lines = tf_lines(gen, f"{fname}_{tf}", "-rminMag", "-exact")
        rep = V.verify(lines, "to_int", fmt, rounding=Rounding.RTZ, exact=True, **kw)
        assert rep.ok, rep.mismatches[:1]
    lines = tf_lines(gen, f"{fname}_roundToInt", "-rmax", "-notexact")
    assert V.verify(lines, "roundToInt", fmt, rounding=Rounding.RUP, exact=False).ok
    for tf, op in [("eq", "eq"), ("lt", "lt"), ("le", "le"), ("eq_signaling", "eq_signaling"),
                   ("lt_quiet", "lt_quiet"), ("le_quiet", "le_quiet")]:
        assert V.verify(tf_lines(gen, f"{fname}_{tf}"), op, fmt).ok
    lines = tf_lines(gen, f"i32_to_{fname}", "-rmin")
    assert V.verify(lines, "from_int", fmt.replace(rounding=Rounding.RDN), bits=32).ok
    lines = tf_lines(gen, f"ui64_to_{fname}", "-rmax")
    assert V.verify(lines, "from_int", fmt.replace(rounding=Rounding.RUP), bits=64,
                    signed=False).ok
    other = next(n for n in TF_FORMATS if n != fname)
    lines = tf_lines(gen, f"{fname}_to_{other}", "-rnear_even")
    assert V.verify(lines, "convert", fmt, to=TF_FORMATS[other]).ok


def test_verify_is_sensitive_to_the_rounding_mode_of_real_output(tf_gens):
    lines = tf_lines(tf_gens["RISCV"], "f32_mul", "-rmin")
    assert V.verify(lines, "mul", FP32.replace(rounding=Rounding.RDN)).ok
    wrong = V.verify(lines, "mul", FP32)               # RNE model against round-down data
    assert not wrong.ok and wrong.mismatch_count > 5


def test_verify_is_sensitive_to_the_nan_convention_of_real_output(tf_gens):
    lines = tf_lines(tf_gens["8086-SSE"], "f32_add", "-rnear_even")
    assert V.verify(lines, "add", tf_fmt("f32", "8086-SSE", Rounding.RNE)).ok
    assert not V.verify(lines, "add", FP32).ok               # RISC-V canonical NaNs


def test_generated_f16_files_agree_with_testfloat_layout(tf_gens, tmp_path):
    """Our files for IEEE formats are line-for-line the same shape as testfloat_gen's."""
    real = tf_lines(tf_gens["RISCV"], "f16_mulAdd", "-rnear_even", n=50)
    ours = _lines(*(lambda s: (s.spec, list(s)))(V.generate("fma", FP16, 50)))
    assert [len(x) for x in real[0].split()] == [len(x) for x in ours[0].split()]
    assert all(len(ln.split()) == 5 for ln in ours)
