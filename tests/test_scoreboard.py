"""verifloat.scoreboard: comparison options, mismatch kinds, reports, DUT value
types (int, bit string, cocotb LogicArray, handles), arrays, block tensors and
the Scoreboard counters. Nothing here needs a simulator."""

from __future__ import annotations

import logging
import os

import pytest

if os.environ.get("VERIFLOAT_IMPL", "cpp") != "cpp":
    pytest.skip("the scoreboard is part of the C++ package", allow_module_level=True)

from verifloat import (E4M3, FP16, FP32, NVFP4, MXINT8, BlockTensor, FPArray,
                       FPFlags)
from verifloat.scoreboard import Kind, Mismatch, Scoreboard, compare  # noqa: E402

NX, OF, NV = FPFlags.INEXACT, FPFlags.OVERFLOW, FPFlags.INVALID


class Handle:
    """A signal handle: just a .value."""
    def __init__(self, value):
        self.value = value


class Bits:
    """A LogicArray look-alike, for when cocotb is not installed."""
    def __init__(self, text):
        self.text = text
        self.is_resolvable = set(text) <= set("01")

    def __len__(self):
        return len(self.text)

    def to_unsigned(self):
        if not self.is_resolvable:
            raise ValueError("unresolved")
        return int(self.text, 2)

    def __str__(self):
        return self.text


def sum32(a: float, b: float):
    return FP32(a) + FP32(b)


# --- passing and the basic mismatch ------------------------------------------

def test_match_returns_none():
    want = sum32(1.5, 2.25)
    assert compare(want, want.raw) is None
    assert compare(want, want.raw, flags=int(want.flags)) is None


def test_value_mismatch_fields():
    want = sum32(0.1, 0.2)
    m = compare(want, want.raw - 1)
    assert isinstance(m, Mismatch) and m.kind is Kind.VALUE
    assert m.expected_raw == want.raw and m.actual_raw == want.raw - 1
    assert m.expected is want and m.actual.raw == want.raw - 1
    assert m.ulps == -1 and m.fmt == FP32
    assert m.expected_flags is None and m.flag_diff == 0


def test_wrong_value_never_passes_without_tolerance():
    one = FP16(1.0)
    for code in (0x3C01, 0x3BFF, 0xBC00, 0x7C00, 0x7E00, 0x0000):
        assert compare(one, code).kind in (Kind.VALUE, Kind.NAN)


def test_ulps_sign_and_binade_boundary():
    one = FP16(1.0)
    assert compare(one, 0x3C03).ulps == 3
    # one step below 1.0 is half a binade-above ulp, still exactly one ulp
    assert compare(one, 0x3BFF).ulps == -1
    assert compare(FP16.from_raw(0x3BFF), 0x3C00).ulps == 1


def test_no_ulps_for_non_finite():
    assert compare(FP16(1.0), 0x7C00).ulps is None
    assert compare(FP16.inf(), 0x3C00).ulps is None
    assert compare(FP16.inf(), 0xFC00).kind is Kind.VALUE


# --- tolerance ---------------------------------------------------------------

def test_tol_ulps():
    one = FP16(1.0)
    assert compare(one, 0x3C02, tol_ulps=2) is None
    assert compare(one, 0x3BFE, tol_ulps=2) is None
    m = compare(one, 0x3C03, tol_ulps=2)
    assert m.kind is Kind.VALUE and m.ulps == 3
    assert compare(one, 0x3C03, tol_ulps=2.5) is not None
    assert compare(one, 0x3C03, tol_ulps=3) is None


def test_tol_does_not_forgive_non_finite_or_nan():
    assert compare(FP16.inf(), 0x7BFF, tol_ulps=100) is not None
    assert compare(FP16(1.0), 0x7C00, tol_ulps=100) is not None
    assert compare(FP16(1.0), 0x7E00, tol_ulps=100).kind is Kind.NAN


def test_tol_does_not_forgive_zero_sign():
    assert compare(FP16.zero(), 0x8000, tol_ulps=5).kind is Kind.ZERO_SIGN


def test_tol_does_not_waive_flags():
    want = FP16(1 / 3)
    assert compare(want, want.raw + 1, flags=0, tol_ulps=1).kind is Kind.FLAGS


def test_tol_validation():
    with pytest.raises(ValueError):
        compare(FP16(1.0), 0x3C00, tol_ulps=-1)


# --- NaN policies ------------------------------------------------------------

QNAN, QNAN2, SNAN = 0x7E00, 0xFE01, 0x7D00


@pytest.mark.parametrize("policy, got, ok", [
    ("exact", QNAN, True), ("exact", QNAN2, False), ("exact", SNAN, False),
    ("quiet", QNAN, True), ("quiet", QNAN2, True), ("quiet", SNAN, False),
    ("any", QNAN, True), ("any", QNAN2, True), ("any", SNAN, True),
])
def test_nan_policies_ieee(policy, got, ok):
    m = compare(FP16.nan(), got, nan=policy)
    assert (m is None) == ok
    if not ok:
        assert m.kind is Kind.NAN


@pytest.mark.parametrize("policy", ["exact", "quiet", "any"])
def test_nan_against_number_is_always_a_nan_mismatch(policy):
    assert compare(FP16.nan(), 0x3C00, nan=policy).kind is Kind.NAN
    assert compare(FP16(1.0), QNAN, nan=policy).kind is Kind.NAN
    assert compare(FP16.nan(), 0x7C00, nan=policy).kind is Kind.NAN   # inf


def test_nan_fn_format():
    # E4M3 has two NaN codes (0x7F, 0xFF) and none of them signals
    assert compare(E4M3.nan(), 0x7F) is None
    assert compare(E4M3.nan(), 0xFF).kind is Kind.NAN
    assert compare(E4M3.nan(), 0xFF, nan="quiet") is None
    assert compare(E4M3.nan(), 0xFF, nan="any") is None
    assert compare(E4M3.nan(), 0x7E).kind is Kind.NAN                 # 448 is not NaN
    assert compare(E4M3.nan(sign=True), 0x7F, nan="exact").kind is Kind.NAN


def test_nan_policy_validation():
    with pytest.raises(ValueError):
        compare(FP16.nan(), QNAN, nan="loose")


def test_expected_nan_with_equal_code_passes_in_exact():
    assert compare(FP16.from_raw(SNAN), SNAN) is None


def test_nan_flags_still_compared():
    want = FP16.inf() - FP16.inf()
    assert want.flags & NV
    assert compare(want, 0xFE00, nan="any", flags=0).kind is Kind.FLAGS
    assert compare(want, 0xFE00, nan="any", flags=int(NV)) is None


# --- zero sign ---------------------------------------------------------------

def test_zero_sign_strict_by_default():
    m = compare(FP16.zero(), 0x8000)
    assert m.kind is Kind.ZERO_SIGN
    assert compare(FP16.zero(sign=True), 0x0000).kind is Kind.ZERO_SIGN


def test_zero_sign_relaxed():
    assert compare(FP16.zero(), 0x8000, zero_sign=False) is None
    assert compare(FP16.zero(sign=True), 0x0000, zero_sign=False) is None
    # relaxing the sign of zero does not relax anything else
    assert compare(FP16.zero(), 0x0001, zero_sign=False).kind is Kind.VALUE
    assert compare(FP16.zero(), 0x8001, zero_sign=False).kind is Kind.VALUE


def test_zero_against_tiny_is_value_not_zero_sign():
    assert compare(FP16.zero(), 0x0001).kind is Kind.VALUE


# --- flags -------------------------------------------------------------------

def test_flags_default_riscv_order():
    want = FP16(1 / 3)                      # INEXACT
    assert want.flags == NX
    assert compare(want, want.raw, flags=0b00001) is None
    m = compare(want, want.raw, flags=0b00000)
    assert m.kind is Kind.FLAGS and m.flag_diff == NX
    assert m.expected_flags == NX and m.actual_flags == 0
    m = compare(want, want.raw, flags=0b10001)
    assert m.flag_diff == NV


def test_flags_every_bit_position():
    for flag, bit in zip(FPFlags, range(5)):
        want = FP16.from_raw(0x3C00)
        # a flag raised by the DUT only
        m = compare(want, want.raw, flags=1 << bit)
        assert m.kind is Kind.FLAGS and m.actual_flags == flag
        assert m.flag_diff == flag


def test_flags_ignore_bits_above_the_five():
    want = FP16(1.0)
    assert compare(want, want.raw, flags=0b1100000) is None


def test_flag_map_reorders():
    want = FP16(1 / 3)                      # INEXACT
    fmap = {NX: 4, NV: 0}                   # DUT: bit 0 = invalid, bit 4 = inexact
    assert compare(want, want.raw, flags=0b10000, flag_map=fmap) is None
    m = compare(want, want.raw, flags=0b00001, flag_map=fmap)
    assert m.kind is Kind.FLAGS and m.actual_flags == NV
    assert m.flag_diff == NX | NV
    assert m.expected_flags == NX


def test_flag_map_partial_ignores_missing_flags():
    # DUT only has invalid and inexact; the expected overflow/underflow are not compared
    want = FP16(70000.0)
    assert want.flags == (OF | NX)
    fmap = {NX: 0, NV: 1}
    assert compare(want, want.raw, flags=0b01, flag_map=fmap) is None
    m = compare(want, want.raw, flags=0b11, flag_map=fmap)
    assert m.flag_diff == NV and m.expected_flags == NX


def test_flag_map_ignores_unmapped_unresolved_bits():
    want = FP16(1.0)
    fmap = {NX: 0}
    assert compare(want, want.raw, flags="xxxx0", flag_map=fmap) is None
    m = compare(want, want.raw, flags="xxxxx", flag_map=fmap)
    assert m.kind is Kind.UNRESOLVED and m.flags_text == "xxxxx"


def test_flag_map_validation():
    with pytest.raises(ValueError):
        compare(FP16(1.0), 0x3C00, flags=0, flag_map={NX | NV: 0})
    with pytest.raises(ValueError):
        compare(FP16(1.0), 0x3C00, flags=0, flag_map={NX: 0, NV: 0})
    with pytest.raises(ValueError):
        compare(FP16(1.0), 0x3C00, flags=0, flag_map={NX: -1})


def test_flags_not_checked_unless_given():
    want = FP16(1 / 3)
    assert compare(want, want.raw) is None


def test_value_and_flags_both_wrong_reports_both():
    want = FP16(1 / 3)
    m = compare(want, want.raw + 1, flags=0)
    assert m.kind is Kind.VALUE and m.flag_diff == NX
    assert "differ: INEXACT" in m.report()


def test_flags_given_as_flags_enum_and_handle():
    want = FP16(1 / 3)
    assert compare(want, want.raw, flags=NX) is None
    assert compare(want, want.raw, flags=Handle(1)) is None
    assert compare(want, want.raw, flags=Handle(Bits("00001"))) is None


# --- DUT value types and X/Z ---------------------------------------------------

@pytest.mark.parametrize("wrap", [
    lambda v, w: v,
    lambda v, w: Handle(v),
    lambda v, w: Handle(Handle(v)),
    lambda v, w: format(v, f"0{w}b"),
    lambda v, w: "0b" + format(v, f"0{w}b"),
    lambda v, w: format(v, f"0{w}b")[:4] + "_" + format(v, f"0{w}b")[4:],
    lambda v, w: Bits(format(v, f"0{w}b")),
    lambda v, w: Handle(Bits(format(v, f"0{w}b"))),
    lambda v, w: FP16.from_raw(v),
], ids="int handle handle2 str 0b str_ bits handle-bits FP".split())
def test_dut_value_types_match_and_mismatch(wrap):
    want = FP16(1.5)
    assert compare(want, wrap(want.raw, 16)) is None
    m = compare(want, wrap(want.raw + 1, 16))
    assert m.kind is Kind.VALUE and m.actual_raw == want.raw + 1


def test_unresolved_is_its_own_kind():
    m = compare(FP16(1.5), Bits("0011111000000x00"))
    assert m.kind is Kind.UNRESOLVED
    assert m.actual_raw is None and m.actual is None
    assert m.actual_text == "0011111000000x00"
    assert "0b0011111000000x00" in m.report()


@pytest.mark.parametrize("bad", ["x" * 16, "z" * 16, "0011110000000zzz", "U" * 16])
def test_unresolved_strings(bad):
    assert compare(FP16(1.5), bad).kind is Kind.UNRESOLVED
    assert compare(FP16(1.5), Handle(bad)).kind is Kind.UNRESOLVED


def test_unresolved_wins_over_nan_policy():
    assert compare(FP16.nan(), "x" * 16, nan="any").kind is Kind.UNRESOLVED


def test_unresolved_flags_with_good_value():
    want = FP16(1.0)
    m = compare(want, want.raw, flags="0000x")
    assert m.kind is Kind.UNRESOLVED and m.flags_text == "0000x"
    assert m.actual_raw == want.raw and m.actual_flags is None
    assert "flags    unresolved bits 0b0000x" in m.report()


def test_unresolved_value_and_flags_both_reported():
    m = compare(FP16(1.0), "x" * 16, flags="xxxxx")
    assert m.kind is Kind.UNRESOLVED
    assert m.actual_text and m.flags_text


def test_width_and_range_errors_are_not_mismatches():
    with pytest.raises(ValueError):
        compare(FP16(1.0), 0x10000)
    with pytest.raises(ValueError):
        compare(FP16(1.0), -1)
    with pytest.raises(ValueError):
        compare(FP16(1.0), "0" * 15)
    with pytest.raises(ValueError):
        compare(FP16(1.0), Bits("0" * 32))
    with pytest.raises(TypeError):
        compare(FP16(1.0), object())


def test_with_cocotb_logic_array():
    types = pytest.importorskip("cocotb.types")
    LA = types.LogicArray
    want = FP16(1.5)
    assert compare(want, LA(want.raw, 16)) is None
    assert compare(want, Handle(LA(want.raw, 16))) is None
    assert compare(want, LA(want.raw + 1, 16)).kind is Kind.VALUE
    m = compare(want, LA("0011111000000X00"))
    assert m.kind is Kind.UNRESOLVED and m.actual_text == "0011111000000X00"
    assert compare(want, LA("Z" * 16)).kind is Kind.UNRESOLVED
    with pytest.raises(ValueError):
        compare(want, LA(1, 8))
    # flags from a flag bus; only unmapped bits unresolved
    assert compare(want, want.raw, flags=LA("00000")) is None
    assert compare(want, want.raw, flags=LA("0000X")).kind is Kind.UNRESOLVED
    fm = {NX: 0}
    assert compare(want, want.raw, flags=LA("XXXX0"), flag_map=fm) is None
    # a one-bit Logic as a one-bit flag bus
    assert compare(want, want.raw, flags=types.Logic("0"), flag_map=fm) is None
    assert compare(want, want.raw, flags=types.Logic("1"), flag_map=fm).kind is Kind.FLAGS
    assert compare(want, want.raw, flags=types.Logic("X"), flag_map=fm).kind is Kind.UNRESOLVED


def test_cocotb_range_directions():
    types = pytest.importorskip("cocotb.types")
    r = types.Range(0, "to", 15)
    la = types.LogicArray(FP16(1.5).raw, r)
    assert compare(FP16(1.5), la) is None


# --- reports -----------------------------------------------------------------

def test_report_contents():
    a, b = FP32(0.1), FP32(0.2)
    want = a + b
    m = compare(want, want.raw - 1, flags=0, op="add", a=a, b=b)
    r = m.report()
    lines = r.splitlines()
    assert lines[0] == "value mismatch (e8m23)"
    assert any(l.startswith("  op") and l.endswith("add") for l in lines)
    for tag in ("a ", "b "):
        assert any(l.strip().startswith(tag) for l in lines)
    exp = next(l for l in lines if l.strip().startswith("expected"))
    act = next(l for l in lines if l.strip().startswith("actual"))
    assert "0x3e99999a" in exp and "0|01111101|00110011001100110011010" in exp
    assert "0x3e999999" in act and "0|01111101|00110011001100110011001" in act
    assert "0.30000001192092896" in exp
    assert "-1 ulp (actual - expected)" in r
    assert "exact    0.30000000447034836" in r
    assert "off by: expected +0.25 ulp, actual -0.75 ulp" in r
    assert "flags    expected INEXACT, actual none; differ: INEXACT" in r
    assert str(m) == r


def test_report_operands_are_in_the_same_form_as_results():
    a = FP32(0.1)
    r = compare(FP32(1.0), 0, a=a).report()
    line = next(l for l in r.splitlines() if l.strip().startswith("a "))
    assert "0x3dcccccd" in line and "0|01111011|10011001100110011001101" in line


def test_report_columns_align():
    r = compare(FP32(1.0), 0x3F800001, x=FP32(123456.0)).report().splitlines()[1:]
    cols = [l.index("0x") for l in r if "0x" in l]
    assert len(set(cols)) == 1


def test_report_non_fp_context_is_truncated_text():
    r = compare(FP16(1.0), 0, note="x" * 500, n=3).report()
    assert "  n        3" in r
    assert "..." in r and "x" * 500 not in r


def test_report_hides_exact_when_the_result_is_exact():
    r = compare(FP16(1.0), 0x3C01).report()
    assert "exact" not in r.replace("expected", "")
    assert "flags" not in r                          # not compared: not listed


def test_report_exact_line_for_non_finite_expected():
    want = FP16(70000.0)                              # overflows to inf, unrounded kept
    r = compare(want, 0x7BFF).report()
    assert "exact" in r and "70000.0" in r


def test_report_nan_and_inf_text():
    r = compare(FP16.from_raw(0x7D00), 0x7C00).report()
    assert "snan" in r and "inf" in r
    r = compare(FP16.inf(sign=True), 0x3C00).report()
    assert "-inf" in r


def test_report_zero_sign():
    r = compare(FP16.zero(), 0x8000).report()
    assert r.splitlines()[0] == "zero sign mismatch (e5m10)"
    assert "-0.0" in r and "diff" not in r


def test_report_fn_format_and_unsigned_format():
    r = compare(E4M3.nan(), 0xFF).report()
    assert "NaN mismatch (e4m3, fn)" in r and "1|1111|111" in r
    from verifloat import UE4M3
    r = compare(UE4M3.from_raw(0x38), 0x39).report()
    assert "0x38" in r and "0111|000" in r


def test_report_prefix_and_context_switch():
    m = compare(FP16(1.0), 0, a=FP16(2.0))
    assert m.report("dut #4: ").startswith("dut #4: value mismatch")
    assert "a " not in m.report(context=False).replace("actual", "")


# --- Scoreboard: counters, policies, logging -----------------------------------

def test_scoreboard_fail_fast_default_raises():
    sb = Scoreboard("t")
    sb.check(FP16(1.0), 0x3C00)
    with pytest.raises(AssertionError) as e:
        sb.check(FP16(1.0), 0x3C01, a=FP16(2.0))
    assert "value mismatch" in str(e.value) and "t #1:" in str(e.value)
    assert (sb.checked, sb.passed, sb.failed) == (2, 1, 1)
    assert sb.by_kind == {"value": 1}


def test_scoreboard_collects_and_counts_by_kind():
    sb = Scoreboard("t", fail_fast=False)
    assert sb.check(FP16(1.0), 0x3C00) is None
    m1 = sb.check(FP16(1.0), 0x3C01)
    m2 = sb.check(FP16.nan(), 0x3C00)
    m3 = sb.check(FP16.zero(), 0x8000)
    m4 = sb.check(FP16(1 / 3), FP16(1 / 3).raw, flags=0)
    m5 = sb.check(FP16(1.0), "x" * 16)
    assert [m.kind for m in (m1, m2, m3, m4, m5)] == [
        Kind.VALUE, Kind.NAN, Kind.ZERO_SIGN, Kind.FLAGS, Kind.UNRESOLVED]
    assert (sb.checked, sb.passed, sb.failed) == (6, 1, 5)
    assert sb.by_kind == {"value": 1, "nan": 1, "zero_sign": 1, "flags": 1, "unresolved": 1}
    assert sb.mismatches == [m1, m2, m3, m4, m5]
    with pytest.raises(AssertionError) as e:
        sb.assert_clean()
    s = str(e.value)
    assert "6 checked, 1 passed, 5 failed" in s and "flags 1" in s and "value 1" in s


def test_scoreboard_options_are_applied():
    sb = Scoreboard(nan="any", zero_sign=False, tol_ulps=1,
                    flag_map={NX: 0})
    sb.check(FP16.nan(), 0xFE01)
    sb.check(FP16.zero(), 0x8000)
    sb.check(FP16(1.0), 0x3C01)
    sb.check(FP16(1 / 3), FP16(1 / 3).raw, flags=1)
    sb.check(FP16(1 / 3), FP16(1 / 3).raw, flags=0b11110 | 1)   # unmapped bits ignored
    sb.assert_clean()
    with pytest.raises(AssertionError):
        sb.check(FP16(1.0), 0x3C02)


def test_scoreboard_option_validation():
    with pytest.raises(ValueError):
        Scoreboard(nan="nope")
    with pytest.raises(ValueError):
        Scoreboard(max_report=0)
    with pytest.raises(ValueError):
        Scoreboard(tol_ulps=-1)


def test_assert_clean_pass_and_empty():
    sb = Scoreboard("t")
    with pytest.raises(AssertionError, match="no checks"):
        sb.assert_clean()
    sb.assert_clean(allow_empty=True)
    sb.check(FP16(1.0), 0x3C00)
    sb.assert_clean()
    assert "1 checked, 1 passed, 0 failed" in sb.summary()


def test_scoreboard_logging(caplog):
    log = logging.getLogger("test.scoreboard")
    sb = Scoreboard("t", fail_fast=False, logger=log, max_report=2)
    with caplog.at_level(logging.ERROR, logger="test.scoreboard"):
        for k in range(5):
            sb.check(FP16(1.0), 0x3C01 + k, a=FP16(2.0))
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 3                                  # two reports, one notice
    assert msgs[0].startswith("t #0: value mismatch") and msgs[1].startswith("t #1:")
    assert "further mismatches not logged" in msgs[2]
    assert sb.failed == 5 and len(sb.mismatches) == 2


def test_scoreboard_fail_fast_does_not_log(caplog):
    sb = Scoreboard("t", logger=logging.getLogger("test.scoreboard"))
    with caplog.at_level(logging.DEBUG), pytest.raises(AssertionError):
        sb.check(FP16(1.0), 0)
    assert not caplog.records


def test_default_logger_without_simulation():
    assert Scoreboard().logger().name == "verifloat.scoreboard"


def test_default_logger_in_simulation(monkeypatch):
    cocotb = pytest.importorskip("cocotb")
    monkeypatch.setattr(cocotb, "is_simulation", True)
    assert Scoreboard().logger().name == "cocotb.scoreboard"


# --- arrays ------------------------------------------------------------------

def make_array():
    return FP16.array([[1.5, 2.25], [0.1, -3.0]])


def test_check_array_pass_with_lists_flat_and_array():
    arr = make_array()
    sb = Scoreboard("t")
    assert sb.check_array(arr, arr.raw) == []
    assert sb.check_array(arr, [c for row in arr.raw for c in row]) == []
    assert sb.check_array(arr, FPArray.from_raw(arr.raw, FP16)) == []
    assert sb.check_array(arr, [[Handle(c) for c in row] for row in arr.raw]) == []
    assert (sb.checked, sb.passed) == (16, 16)


def test_check_array_reports_each_index():
    arr = make_array()
    got = [list(r) for r in arr.raw]
    got[0][1] ^= 1
    got[1][0] ^= 4
    sb = Scoreboard("t", fail_fast=False)
    ms = sb.check_array(arr, got, op="load")
    assert [m.index for m in ms] == [(0, 1), (1, 0)]
    assert (sb.checked, sb.passed, sb.failed) == (4, 2, 2)
    assert ms[0].context == {"op": "load"}
    with pytest.raises(AssertionError) as e:
        Scoreboard().check_array(arr, got, op="load")
    s = str(e.value)
    assert "value mismatch at [0, 1]" in s and "at [1, 0]" in s
    assert s.count("op ") == 1                      # context printed once


def test_check_array_fail_fast_reports_cap():
    arr = FP16.array([float(i) for i in range(1, 21)])
    got = [c ^ 1 for c in arr.raw]
    with pytest.raises(AssertionError) as e:
        Scoreboard("t", max_report=3).check_array(arr, got)
    s = str(e.value)
    assert s.count("value mismatch") == 3
    assert "and 17 more mismatches in this check" in s
    sb = Scoreboard("t", fail_fast=False, max_report=3)
    ms = sb.check_array(arr, got)
    assert len(ms) == 20 and sb.failed == 20 and len(sb.mismatches) == 3


def test_check_array_xz_and_nan_policy():
    arr = FP16.array([1.0, float("nan")])
    sb = Scoreboard(fail_fast=False, nan="any")
    ms = sb.check_array(arr, [Bits("x" * 16), 0xFE01])
    assert [m.kind for m in ms] == [Kind.UNRESOLVED]
    assert sb.passed == 1


def test_check_array_per_element_flags():
    arr = FP16.array([1.0, 1 / 3, 70000.0])
    flags = [f.flags for f in arr.tolist()]
    assert flags == [FPFlags(0), NX, OF | NX]
    sb = Scoreboard(fail_fast=False)
    assert sb.check_array(arr, arr.raw, flags=[int(f) for f in flags]) == []
    ms = sb.check_array(arr, arr.raw, flags=[0, 0, int(OF | NX)])
    assert [(m.index, m.kind, m.flag_diff) for m in ms] == [((1,), Kind.FLAGS, NX)]
    with pytest.raises(ValueError):
        sb.check_array(arr, arr.raw, flags=[0, 0])


def test_check_array_aggregate_flags():
    arr = FP16.array([1.0, 1 / 3, 70000.0])
    sb = Scoreboard(fail_fast=False)
    assert arr.flags == OF | NX
    assert sb.check_array(arr, arr.raw, flags=int(OF | NX)) == []
    ms = sb.check_array(arr, arr.raw, flags=int(NX))
    assert len(ms) == 1 and ms[0].label == "flags" and ms[0].index is None
    assert ms[0].kind is Kind.FLAGS and ms[0].flag_diff == OF
    assert "flags mismatch at flags" in ms[0].report()
    ms = sb.check_array(arr, arr.raw, flags="0000x")
    assert ms[0].kind is Kind.UNRESOLVED
    assert sb.checked == 12                         # three elements + the aggregate, x3
    assert sb.by_kind == {"flags": 1, "unresolved": 1}


def test_check_array_from_fp_lists():
    vals = [[FP16(1.0), FP16(1 / 3)], [FP16(2.0), FP16(70000.0)]]
    raw = [[v.raw for v in row] for row in vals]
    sb = Scoreboard(fail_fast=False)
    assert sb.check_array(vals, raw, flags=int(NX | OF)) == []
    assert sb.check_array(vals, raw, flags=[[0, 1], [0, 5]]) == []
    ms = sb.check_array(vals, raw, flags=[[0, 0], [0, 5]])
    assert [m.index for m in ms] == [(0, 1)]
    ms = sb.check_array(vals, [[0, 0], [0, 0]])
    assert len(ms) == 4 and "exact" in ms[3].report()    # unrounded is kept for FP lists


def test_check_array_shape_errors():
    arr = make_array()
    sb = Scoreboard()
    with pytest.raises(ValueError):
        sb.check_array(arr, [[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        sb.check_array(arr, [1, 2, 3])
    with pytest.raises(ValueError):
        sb.check_array(arr, 5)
    with pytest.raises(ValueError):
        sb.check_array([], [])
    assert sb.checked == 0


def test_check_array_width_error():
    with pytest.raises(ValueError):
        Scoreboard().check_array(make_array(), [[0x10000, 0], [0, 0]])


# --- block tensors ---------------------------------------------------------------

def nvfp4():
    return BlockTensor.quantize([1.0] * 16 + [6.0] * 16, NVFP4)


def test_check_block_pass():
    t = nvfp4()
    sb = Scoreboard("t")
    assert sb.check_block(t, t.elem_raw, t.scale_raw, t.tensor_scale.raw) == []
    assert sb.checked == 32 + 2 + 1
    assert sb.check_block(t, elems=t.elem_raw) == []
    assert sb.check_block(t, scales=t.scale_raw) == []
    assert sb.check_block(t, tensor_scale=Handle(t.tensor_scale.raw)) == []


def test_check_block_reports_label_and_index():
    t = nvfp4()
    elems = list(t.elem_raw)
    elems[20] ^= 1
    scales = list(t.scale_raw)
    scales[1] -= 1
    sb = Scoreboard(fail_fast=False)
    ms = sb.check_block(t, elems, scales, t.tensor_scale.raw + 1)
    assert [(m.label, m.index) for m in ms] == [
        ("elem", (20,)), ("scale", (1,)), ("tensor_scale", None)]
    assert all(m.kind is Kind.VALUE for m in ms)
    assert (sb.checked, sb.failed) == (35, 3)
    assert ms[0].report().startswith("value mismatch at elem[20] (e2m1, finite)")
    assert ms[1].report().startswith("value mismatch at scale[1] (e4m3, fn)")
    assert ms[2].report().startswith("value mismatch at tensor_scale (e8m23)")


def test_check_block_nan_scale_policy():
    t = BlockTensor.from_raw([0] * 16, [0x7F], NVFP4, tensor_scale=FP32(1.0).raw)
    assert t.scale_raw == [0x7F]
    assert Scoreboard(nan="any").check_block(t, scales=[0xFF]) == []
    with pytest.raises(AssertionError):
        Scoreboard().check_block(t, scales=[0xFF])


def test_check_block_2d_and_flat_actual_and_xz():
    t = BlockTensor.quantize([[1.0] * 16, [2.0] * 16], NVFP4)
    assert len(t.scale_raw) == 2 and t.scale_raw[0] != t.scale_raw[1]
    flat = [c for row in t.elem_raw for c in row]
    sb = Scoreboard(fail_fast=False)
    assert sb.check_block(t, flat) == []
    bad = list(t.elem_raw[1])
    bad[3] = Bits("xxxx")
    ms = sb.check_block(t, [t.elem_raw[0], bad])
    assert [(m.label, m.index, m.kind) for m in ms] == [("elem", (1, 3), Kind.UNRESOLVED)]
    with pytest.raises(ValueError):
        sb.check_block(t, t.elem_raw[:1])


def test_check_block_integer_elements_and_pow2_scales():
    t = BlockTensor.quantize([1.0] * 32, MXINT8)
    sb = Scoreboard(fail_fast=False)
    assert sb.check_block(t, t.elem_raw, t.scale_raw) == []
    got = list(t.elem_raw)
    got[5] = 0x7F
    ms = sb.check_block(t, got, [t.scale_raw[0] + 1])
    assert [(m.label, m.index) for m in ms] == [("elem", (5,)), ("scale", (0,))]
    r = ms[0].report()
    assert "expected 1.0       0x40  01000000" in r and "actual   1.984375  0x7f  01111111" in r
    assert ms[1].report().startswith("value mismatch at scale[0]")
    ms = sb.check_block(t, ["x" * 8] + got[1:5] + [t.elem_raw[5]] + t.elem_raw[6:])
    assert ms[0].kind is Kind.UNRESOLVED


def test_check_block_errors():
    t = BlockTensor.quantize([1.0] * 32, MXINT8)
    with pytest.raises(ValueError, match="tensor scale"):
        Scoreboard().check_block(t, tensor_scale=1)
    with pytest.raises(ValueError, match="zero"):
        Scoreboard().check_block(t, zeros=[0])


def test_check_block_zero_points():
    from verifloat import UE4M3, BlockFormat, IntFormat
    fmt = BlockFormat(IntFormat(4, False), 4, UE4M3, zero_point=IntFormat(4, False))
    t = BlockTensor.from_raw([1, 2, 3, 4], [0x38], fmt, [3])
    assert t.zero_raw == [3]
    sb = Scoreboard(fail_fast=False)
    assert sb.check_block(t, zeros=[3]) == []
    assert sb.check_block(t, zeros=[4])[0].label == "zero"


def test_check_block_counts_into_summary():
    t = nvfp4()
    sb = Scoreboard("blk", fail_fast=False)
    sb.check_block(t, t.elem_raw, [0, 0])
    assert "blk: 34 checked, 32 passed, 2 failed (value 2)" in sb.summary()
