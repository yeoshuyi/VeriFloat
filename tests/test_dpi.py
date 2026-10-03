"""The SystemVerilog package (verifloat_pkg.sv) in a real simulator.

A testbench imports the package, reads vectors written by verifloat.vectors
and calls the DPI-C functions on every one: each result and its flags must be
what the Python model exported. This checks what tests/test_capi.py cannot
reach through ctypes: the import declarations, the argument types as
Verilator passes them (chandle, longint, bit [127:0], string), and linking
with the flags `python -m verifloat.dpi` prints. Needs Verilator; skipped
without it.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

if os.environ.get("VERIFLOAT_IMPL", "cpp") != "cpp":
    pytest.skip("the C library is part of the C++ package", allow_module_level=True)
if not shutil.which("verilator"):
    pytest.skip("verilator not installed", allow_module_level=True)

from verifloat import E4M3, FP16, FP32, FP64, FPFormat, NaNMode, Rounding, dpi, vectors  # noqa: E402

FP128 = FPFormat(15, 112)
FORMATS = [FP16, FP32, FP64, E4M3,
           FPFormat(5, 2, rounding=Rounding.RTZ),
           FPFormat(4, 3, 10, signed=False, inf_nan="fn", saturate=True, tininess="before"),
           FPFormat(8, 7, nan_mode=NaNMode.X86, rounding=Rounding.RUP, tininess="before"),
           FPFormat(5, 10, ftz=True, rounding=Rounding.RDN, nan_mode=NaNMode.ARM),
           FPFormat(3, 2, inf_nan=False, rounding=Rounding.RNA),
           FPFormat(4, 6, rounding=Rounding.SR, sr_bits=8)]
WIDE = [FP128, FPFormat(15, 64), FPFormat(11, 90, rounding=Rounding.RTZ, nan_mode=NaNMode.PROPAGATE)]

# Operation numbers of vf_op (VF_OP_* in verifloat.h), by vectors' names.
OP = {"add": 0, "sub": 1, "mul": 2, "div": 3, "remainder": 4, "fmod": 5, "minimum": 6, "maximum": 7,
      "minimum_number": 8, "maximum_number": 9, "copysign": 10, "fma": 13, "sqrt": 14, "next_up": 17,
      "next_down": 18, "logb": 19}
CMP = {"compare": (0, 0), "compare_signaling": (0, 1), "eq": (1, 0), "eq_signaling": (1, 1),
       "lt_quiet": (2, 0), "lt": (2, 1), "le_quiet": (3, 0), "le": (3, 1)}
ROUND = {Rounding.RNE: 0, Rounding.RNA: 1, Rounding.RTZ: 2, Rounding.RUP: 3, Rounding.RDN: 4}

TB = r"""
module tb;
  import verifloat_pkg::*;

  chandle fmts [int];
  int fd, n, idx, kind, p1, p2, p3, p4, fi, ti, flags, ef, line, checked, bad, r32;
  string tag, name, path;
  bit [127:0] a, b, c, er, r;
  bit [63:0] sr;
  longint unsigned r64;

  initial begin
    if (!$value$plusargs("vectors=%s", path)) path = "vectors.txt";
    fd = $fopen(path, "r");
    if (fd == 0) begin $display("CANNOT OPEN %s", path); $finish; end
    $display("VERSION %s", vf_version());
    while ($fscanf(fd, "%s", tag) == 1) begin
      line++;
      if (tag == "F") begin
        n = $fscanf(fd, "%d %s", idx, name);
        fmts[idx] = vf_format(name);
        if (fmts[idx] == null) $display("FORMAT ERROR %0d %s", idx, vf_last_error());
        else $display("FMT %0d %0d %s", idx, vf_format_size(fmts[idx]), vf_format_name(fmts[idx]));
      end else begin
        n = $fscanf(fd, "%d %d %d %d %d %d %d %h %h %h %h %h %h", kind, p1, p2, p3, p4, fi, ti, sr, a, b, c, er, ef);
        if (n != 13) begin $display("BAD LINE %0d", line); bad++; continue; end
        vf_set_sr(sr);
        r = '0;
        case (kind)
          0: r[63:0] = vf_op(fmts[fi], p1, a[63:0], b[63:0], c[63:0], flags);
          1: begin
            case (p1)
              0: r32 = vf_compare(fmts[fi], a[63:0], b[63:0], p2, flags);
              1: r32 = vf_eq(fmts[fi], a[63:0], b[63:0], p2, flags);
              2: r32 = vf_lt(fmts[fi], a[63:0], b[63:0], p2, flags);
              default: r32 = vf_le(fmts[fi], a[63:0], b[63:0], p2, flags);
            endcase
            r[31:0] = r32;
          end
          2: r[63:0] = vf_convert(fmts[ti], fmts[fi], a[63:0], flags);
          3: r[63:0] = vf_round_to_integral(fmts[fi], a[63:0], p1, p2, flags);
          4: r[63:0] = vf_to_int(fmts[fi], a[63:0], p1, p2, p3, p4, flags);
          5: r[63:0] = p2 != 0 ? vf_from_int(fmts[fi], a[63:0], flags) : vf_from_uint(fmts[fi], a[63:0], flags);
          6: vf_op128(fmts[fi], p1, a, b, c, r, flags);
          7: begin r32 = vf_compare128(fmts[fi], a, b, p2, flags); r[31:0] = r32; end
          8: vf_round_to_integral128(fmts[fi], a, p1, p2, r, flags);
          10: vf_convert128(fmts[ti], fmts[fi], a, r, flags);
          // The named functions, one by one (kind 0 goes through vf_op).
          9: begin
            case (p1)
              VF_OP_ADD: r64 = vf_add(fmts[fi], a[63:0], b[63:0], flags);
              VF_OP_SUB: r64 = vf_sub(fmts[fi], a[63:0], b[63:0], flags);
              VF_OP_MUL: r64 = vf_mul(fmts[fi], a[63:0], b[63:0], flags);
              VF_OP_DIV: r64 = vf_div(fmts[fi], a[63:0], b[63:0], flags);
              VF_OP_REM: r64 = vf_rem(fmts[fi], a[63:0], b[63:0], flags);
              VF_OP_FMOD: r64 = vf_fmod(fmts[fi], a[63:0], b[63:0], flags);
              VF_OP_MIN: r64 = vf_min(fmts[fi], a[63:0], b[63:0], flags);
              VF_OP_MAX: r64 = vf_max(fmts[fi], a[63:0], b[63:0], flags);
              VF_OP_MINNUM: r64 = vf_minnum(fmts[fi], a[63:0], b[63:0], flags);
              VF_OP_MAXNUM: r64 = vf_maxnum(fmts[fi], a[63:0], b[63:0], flags);
              VF_OP_SGNJ: r64 = vf_sgnj(fmts[fi], a[63:0], b[63:0], flags);
              VF_OP_FMA: r64 = vf_fma(fmts[fi], a[63:0], b[63:0], c[63:0], flags);
              VF_OP_SQRT: r64 = vf_sqrt(fmts[fi], a[63:0], flags);
              VF_OP_NEXT_UP: r64 = vf_next_up(fmts[fi], a[63:0], flags);
              VF_OP_NEXT_DOWN: r64 = vf_next_down(fmts[fi], a[63:0], flags);
              default: r64 = vf_logb(fmts[fi], a[63:0], flags);
            endcase
            r[63:0] = r64;
          end
          default: begin $display("BAD KIND %0d", kind); bad++; end
        endcase
        checked++;
        if (r !== er || flags != ef) begin
          bad++;
          if (bad <= 20)
            $display("MISMATCH line %0d kind %0d: got %h [%s], expected %h [%s]", line, kind, r,
                     vf_flags_str(flags), er, vf_flags_str(ef));
        end
      end
    end
    // Errors: an unknown format, and an operation the format cannot answer.
    if (vf_format("banana") != null) $display("NO FORMAT ERROR");
    $display("ERRTEXT %s", vf_last_error());
    r64 = vf_div(vf_format("e3m2, finite"), 0, 0, flags);
    $display("ERRFLAGS %0d %0d %s | %s", r64, flags, vf_flags_str(flags), vf_last_error());
    $display("FLAGSTR %s %s", vf_flags_str(0), vf_flags_str(VF_INEXACT | VF_OVERFLOW));
    $display("REAL %h %.17g", vf_from_double(vf_format("e8m23"), 0.1, flags), vf_to_double(vf_format("e8m23"), 64'h3DCCCCCD));
    $display("CLASS %0d %0d", vf_fclass(vf_format("e8m23"), 64'h7F800000), vf_fclass128(vf_format("e15m112"), '0));
    r = '1;
    vf_op128(vf_format("e8m300"), VF_OP_ADD, a, b, c, r, flags);
    $display("TOOWIDE %h %0d %s", r, flags, vf_last_error());
    vf_scaleb128(vf_format("e15m112"), 128'h3FFF0000000000000000000000000000, 3, r, flags);
    $display("SCALEB %h %h", vf_scaleb(vf_format("e8m23"), 64'h3F800000, -1, flags), r);
    $display("CHECKED %0d MISMATCHES %0d", checked, bad);
    $finish;
  end
endmodule
"""


def name_arg(fmt: FPFormat) -> str:
    return str(fmt).replace(" ", "")        # one %s token; the parser ignores spacing


def vector_lines(seed: int, count: int):
    """(format list, lines): every DPI entry point on vectors from the Python model."""
    fmts = [*FORMATS, *WIDE]
    index = {f: i for i, f in enumerate(fmts)}
    lines = [f"F {i} {name_arg(f)}" for i, f in enumerate(fmts)]

    def emit(kind, p, f, t, spec_kw, op, n=count, style="mixed"):
        stream = vectors.generate(op, f, n, seed=seed, style=style, **spec_kw)
        for v in stream:
            ops = list(v.operands) + [0, 0]
            if kind == 5 and spec_kw.get("signed", True) and ops[0] >> (spec_kw["bits"] - 1):
                ops[0] = (ops[0] - (1 << spec_kw["bits"])) & ((1 << 64) - 1)     # sign-extend to longint
            p4 = [*p, 0, 0, 0, 0][:4]
            lines.append("V %d %d %d %d %d %d %d %x %x %x %x %x %x" % (
                kind, *p4, index[f], index[t if t is not None else f], v.sr or 0, *ops[:3], v.result, v.flags))

    for f in FORMATS:
        sr = f.rounding is Rounding.SR
        for op, num in OP.items():
            emit(0, [num], f, None, {}, op)
            emit(9, [num], f, None, {}, op, n=max(count // 4, 8))
        for op, (which, signaling) in CMP.items():
            emit(1, [which, signaling], f, None, {}, op, n=max(count // 2, 8))
        for g in (FP16, FP32, FORMATS[5], FORMATS[9]):
            emit(2, [], f, g, {"to": g}, "convert")
        for mode, exact in ((Rounding.RNE, True), (Rounding.RTZ, False), (Rounding.RUP, True), (None, False)):
            if mode is None and sr:
                continue
            kw = {"rounding": mode, "exact": exact}
            m = -1 if mode is None else ROUND[mode]
            emit(3, [m, int(exact)], f, None, kw, "round_to_integral", n=count // 2)
            for bits, signed in ((32, True), (8, False), (64, True), (64, False), (1, True)):
                emit(4, [bits, int(signed), m, int(exact)], f, None, {**kw, "bits": bits, "signed": signed},
                     "to_int", n=count // 4)
        for bits, signed in ((64, True), (64, False), (20, True), (8, False)):
            emit(5, [bits, int(signed)], f, None, {"bits": bits, "signed": signed}, "from_int", n=count // 2)
        emit(0, [OP["add"]], f, None, {}, "add", n=None, style="edges")
    for f in WIDE:
        for op, num in OP.items():
            emit(6, [num], f, None, {}, op)
        emit(7, [0, 0], f, None, {}, "compare")
        emit(7, [0, 1], f, None, {}, "compare_signaling")
        emit(8, [ROUND[Rounding.RNE], 1], f, None, {"rounding": Rounding.RNE, "exact": True}, "round_to_integral")
        for g in (*WIDE, FP32):
            emit(10, [], f, g, {"to": g}, "convert")
            emit(10, [], g, f, {"to": f}, "convert")
    return fmts, lines


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    d = tmp_path_factory.mktemp("dpi")
    (d / "tb.sv").write_text(TB)
    build = subprocess.run(
        ["verilator", "--binary", "--timing", "-Wno-fatal", "--top-module", "tb", str(dpi.sv_package()), "tb.sv",
         "-CFLAGS", dpi.cflags(), "-LDFLAGS", dpi.ldflags()], cwd=d, capture_output=True, text=True)
    assert build.returncode == 0, build.stdout[-3000:] + build.stderr[-3000:]

    def run(lines):
        (d / "vectors.txt").write_text("\n".join(lines) + "\n")
        r = subprocess.run([str(d / "obj_dir" / "Vtb"), "+vectors=vectors.txt"], cwd=d, capture_output=True,
                           text=True, timeout=600)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        return r.stdout.splitlines()
    return run


def tail(out, tag):
    return [ln[len(tag) + 1:] for ln in out if ln.startswith(tag + " ")]


def test_every_dpi_function_matches_exported_vectors(sim):
    seed = int(os.environ.get("VERIFLOAT_SEED", 1)) & 0xFFFF
    count = max(int(os.environ.get("VERIFLOAT_ITERS", 2000)) // 10, 16)
    fmts, lines = vector_lines(seed, count)
    out = sim(lines)
    assert tail(out, "FMT") == [f"{i} {f.size} {f}" for i, f in enumerate(fmts)]
    assert not tail(out, "MISMATCH") and not tail(out, "BAD"), [ln for ln in out if "MISMATCH" in ln][:5]
    n = sum(ln.startswith("V ") for ln in lines)
    assert n > 20000
    assert tail(out, "CHECKED") == [f"{n} MISMATCHES 0"]
    # Entry points that vectors do not cover, on known values.
    assert tail(out, "VERSION")[0].count(".") >= 1
    assert tail(out, "ERRTEXT") == ["not an FP format name: 'banana'"]
    assert tail(out, "ERRFLAGS")[0].startswith("0 256 ERROR | ")
    assert tail(out, "FLAGSTR") == ["none OVERFLOW|INEXACT"]
    assert tail(out, "REAL") == ["000000003dcccccd 0.10000000149011612"]
    assert tail(out, "TOOWIDE") == ["0" * 32 + " 256 e8m300 is wider than 128 bits: use the _w functions"]
    assert tail(out, "CLASS") == [f"{1 << 7} {1 << 4}"]
    assert tail(out, "SCALEB") == ["000000003f000000 40020000000000000000000000000000"]


def test_the_testbench_reports_a_wrong_vector(sim):
    """The check has teeth: one flipped result bit and one flipped flag are both reported."""
    _, lines = vector_lines(3, 16)
    vec = [i for i, ln in enumerate(lines) if ln.startswith("V ")]
    i, j = vec[len(vec) // 3], vec[len(vec) // 2]
    f = lines[i].split()
    f[-2] = "%x" % (int(f[-2], 16) ^ 1)
    lines[i] = " ".join(f)
    f = lines[j].split()
    f[-1] = "%x" % (int(f[-1], 16) ^ 1)
    lines[j] = " ".join(f)
    out = sim(lines)
    assert tail(out, "CHECKED") == [f"{len(vec)} MISMATCHES 2"]
    assert len(tail(out, "MISMATCH")) == 2
