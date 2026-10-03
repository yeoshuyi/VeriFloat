"""Compare DUT outputs with the golden model and report what differs.

Every testbench ends up hand-writing this glue; it lives here once. Nothing
imports cocotb: a DUT value can be an ``int`` (raw code), a bit string, a
cocotb ``LogicArray`` / ``Logic``, or a signal handle (anything with
``.value``). X/Z bits are a mismatch of their own kind instead of a crash::

    from verifloat.scoreboard import Scoreboard, compare

    sb = Scoreboard("adder", nan="any")            # raises AssertionError on the first mismatch
    want = FP32.from_raw(a) + FP32.from_raw(b)
    sb.check(want, dut.z, a=FP32.from_raw(a), b=FP32.from_raw(b), op="add")
    sb.assert_clean()                              # at the end of the test

    compare(want, got_code)                        # the pure function: Mismatch or None

Flags are compared when the DUT's flag value is given. They are read in RISC-V
fflags order unless ``flag_map`` says otherwise; flags missing from the map
are not compared, which is how a DUT with a partial flag bus is checked.
"""

from __future__ import annotations

import enum
import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction

from .array import FPArray
from .blockscale import BlockTensor, IntFormat, Pow2Format
from .fp import FP, FPFlags, FPFormat

__all__ = ["Kind", "Mismatch", "Scoreboard", "compare"]

_NAN_POLICIES = ("exact", "quiet", "any")


class Kind(str, enum.Enum):
    """What differs. Only one kind is reported per check; flag differences
    are always listed in the report, and are the kind when nothing else is."""
    VALUE = "value"            # the codes differ (beyond tol_ulps)
    NAN = "nan"                # NaN against non-NaN, or NaNs the policy rejects
    ZERO_SIGN = "zero_sign"    # +0 against -0 under zero_sign=True
    FLAGS = "flags"            # the value matches, the status flags do not
    UNRESOLVED = "unresolved"  # the DUT value has X/Z bits


_TITLE = {Kind.VALUE: "value mismatch", Kind.NAN: "NaN mismatch",
          Kind.ZERO_SIGN: "zero sign mismatch", Kind.FLAGS: "flags mismatch",
          Kind.UNRESOLVED: "unresolved bits (X/Z)"}

_RISCV = {f: f.value.bit_length() - 1 for f in FPFlags}


@dataclass(frozen=True)
class Mismatch:
    """One failed check. ``expected_raw`` / ``actual_raw`` are None for a
    flags-only check (array-level flags) or when the DUT value has X/Z bits.
    ``ulps`` is actual - expected in ulps (the smaller of the two spacings),
    when both are finite."""
    kind: Kind
    label: str = ""                       # "elem", "scale", ... in block checks
    index: tuple[int, ...] | None = None  # position in an array check
    fmt: object = None                    # FPFormat, IntFormat or Pow2Format
    expected_raw: int | None = None
    actual_raw: int | None = None
    actual_text: str | None = None        # the DUT bits, when some are X/Z
    expected: FP | None = None
    actual: FP | None = None
    ulps: Fraction | None = None
    expected_flags: FPFlags | None = None  # masked to the flags the DUT has
    actual_flags: FPFlags | None = None
    flags_text: str | None = None         # the DUT flag bits, when some are X/Z
    context: Mapping = field(default_factory=dict)

    @property
    def flag_diff(self) -> FPFlags:
        if self.expected_flags is None or self.actual_flags is None:
            return FPFlags(0)
        return self.expected_flags ^ self.actual_flags

    def report(self, prefix: str = "", context: bool = True) -> str:
        """Multi-line description: the context, expected and actual as value,
        hex and sign|exponent|mantissa, the ulp difference, the flags that
        differ. ``context=False`` leaves the context out (repeated mismatches)."""
        where = self.label + ("" if self.index is None
                              else "[" + ", ".join(map(str, self.index)) + "]")
        head = f"{prefix}{_TITLE[self.kind]}" + (f" at {where}" if where else "")
        if isinstance(self.fmt, FPFormat):
            head += f" ({self.fmt})"
        rows = []                  # (label, cells) for values, (label, str) for text
        for k, v in (self.context.items() if context else ()):
            rows.append((k, self._cells(v.raw, v.format, v) if isinstance(v, FP)
                         else _short(v)))
        if self.expected_raw is not None:
            rows.append(("expected", self._cells(self.expected_raw, self.fmt, self.expected)))
        if self.actual_text is not None:
            rows.append(("actual", "unresolved 0b" + self.actual_text))
        elif self.actual_raw is not None:
            rows.append(("actual", self._cells(self.actual_raw, self.fmt, self.actual)))
        cells = [c for _, c in rows if isinstance(c, tuple)]
        widths = [max(len(c[i]) for c in cells) for i in range(3)] if cells else []
        lw = max([len(k) for k, _ in rows] + [8])
        lines = [head]
        for k, c in rows:
            lines.append(f"  {k:<{lw}} {self._line(c, widths) if isinstance(c, tuple) else c}")
        if self.ulps is not None:
            lines.append(f"  {'diff':<{lw}} {_ulps(self.ulps)} ulp (actual - expected)")
        exp = self.expected
        if exp is not None and isinstance(exp.unrounded, Fraction) and (
                not exp.is_finite or exp.unrounded != exp.exact):
            lines.append(f"  {'exact':<{lw}} {_num(exp.unrounded)}{self._errors(exp.unrounded)}")
        if self.flags_text is not None:
            lines.append(f"  {'flags':<{lw}} unresolved bits 0b{self.flags_text}")
        elif self.flag_diff:
            lines.append(f"  {'flags':<{lw}} expected {_names(self.expected_flags)}, "
                         f"actual {_names(self.actual_flags)}; "
                         f"differ: {_names(self.flag_diff)}")
        return "\n".join(lines)

    __str__ = report

    def _errors(self, u) -> str:
        if self.expected is None or self.actual is None:
            return ""
        parts = []
        for name, fp in (("expected", self.expected), ("actual", self.actual)):
            if fp.is_finite:
                try:
                    parts.append(f"{name} {_ulps(fp.error_ulps(u))} ulp")
                except (ValueError, OverflowError):
                    pass
        return f"  (off by: {', '.join(parts)})" if parts else ""

    @staticmethod
    def _cells(raw, fmt, fp: FP | None = None):
        if isinstance(fmt, FPFormat):
            fp = fp if fp is not None else fmt.from_raw(raw)
            return _num(fp), "0x" + fp.to_hex(), fp.to_bin("|")
        width = getattr(fmt, "size", None) or max(raw.bit_length(), 1)
        return (_int_value(fmt, raw), f"0x{raw:0{(width + 3) // 4}x}", f"{raw:0{width}b}")

    @staticmethod
    def _line(c, widths) -> str:
        return f"{c[0]:<{widths[0]}}  {c[1]:<{widths[1]}}  {c[2]}".rstrip()


def _short(v, limit: int = 160) -> str:
    s = str(v)
    return s if len(s) <= limit else s[:limit - 3] + "..."


def _num(x) -> str:
    """Short exact-enough text for a value, a Fraction or a float."""
    if isinstance(x, FP):
        if x.is_snan:
            return "snan"
        if x.is_nan:
            return "nan"
        if x.is_inf:
            return "-inf" if x.sign else "inf"
        try:
            return repr(float(x))
        except OverflowError:
            return str(x.exact)
    try:
        return repr(float(x))
    except OverflowError:
        return str(x)


def _int_value(fmt, raw: int) -> str:
    try:
        if isinstance(fmt, IntFormat):
            return _num(fmt.value(fmt.from_raw(raw)))
        if isinstance(fmt, Pow2Format):
            return _num(fmt.value(raw))
    except ValueError:
        return "nan"
    return str(raw)


def _ulps(x) -> str:
    x = Fraction(x)
    s = str(int(x)) if x.denominator == 1 else f"{float(x):.3g}"
    return "+" + s if x > 0 else s


def _names(f) -> str:
    f = FPFlags(f)
    return "|".join(m.name for m in FPFlags if m in f) or "none"


# Reading DUT values ---------------------------------------------------------

def _read(value, _depth: int = 0) -> tuple[int, int, str | None, int | None]:
    """DUT value -> (int, unknown-bit mask, bit text if any bit is X/Z, width).
    Unknown bits read as 0 in the int. The width is None when the type has none."""
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"negative DUT value {value}")
        return int(value), 0, None, None
    if isinstance(value, FP):
        return value.raw, 0, None, None
    if isinstance(value, str):
        return _from_text(value.replace("_", ""))
    if hasattr(value, "value") and not hasattr(value, "is_resolvable") and _depth < 3:
        return _read(value.value, _depth + 1)       # a signal handle
    if getattr(value, "is_resolvable", True):
        for conv in ("to_unsigned", "integer", "__int__"):
            if hasattr(value, conv):
                v = getattr(value, conv)
                v = v() if callable(v) else v
                try:
                    width = len(value)
                except TypeError:
                    width = None
                return int(v), 0, None, width
        raise TypeError(f"cannot read a DUT value from {type(value).__name__}")
    return _from_text(str(value))


def _from_text(s: str) -> tuple[int, int, str | None, int | None]:
    s = s.strip()
    if s[:2].lower() == "0b":
        s = s[2:]
    if not s:
        raise ValueError("empty DUT bit string")
    val = unk = 0
    for ch in s:
        val, unk = val << 1, unk << 1
        if ch == "1":
            val |= 1
        elif ch != "0":
            unk |= 1
    return val, unk, (s if unk else None), len(s)


def _code(value, width: int, what: str = "value"):
    """Read a DUT code of ``width`` bits -> (int, bit text or None)."""
    v, unk, text, w = _read(value)
    if w is not None and w != width:
        raise ValueError(f"DUT {what} is {w} bits wide, the format has {width}")
    if v >> width:
        raise ValueError(f"DUT {what} {v:#x} does not fit in {width} bits")
    return v, (text if unk else None)


# Comparison -----------------------------------------------------------------

@dataclass(frozen=True)
class _Options:
    nan: str = "exact"
    zero_sign: bool = True
    tol_ulps: Fraction = Fraction(0)
    flag_map: Mapping | None = None

    def __post_init__(self):
        if self.nan not in _NAN_POLICIES:
            raise ValueError(f"nan must be one of {_NAN_POLICIES}")
        tol = Fraction(self.tol_ulps)
        if tol < 0:
            raise ValueError("tol_ulps must be >= 0")
        object.__setattr__(self, "tol_ulps", tol)
        fm = self.flag_map
        if fm is not None:
            fm = {FPFlags(k): v for k, v in fm.items()}
            if any(len(list(f)) != 1 and f != 0 for f in fm) or 0 in fm:
                raise ValueError("flag_map keys must be single FPFlags members")
            if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in fm.values()):
                raise ValueError("flag_map values must be bit positions >= 0")
            if len(set(fm.values())) != len(fm):
                raise ValueError("flag_map bit positions must be distinct")
            object.__setattr__(self, "flag_map", fm)

    @property
    def positions(self) -> Mapping:
        return self.flag_map if self.flag_map is not None else _RISCV


def _flag_state(expected, flags, opts: _Options):
    """-> (expected flags masked to the DUT's, DUT flags decoded, bit text)."""
    pos = opts.positions
    exp = FPFlags(0)
    for f in pos:
        exp |= FPFlags(expected) & f
    v, unk, text, _ = _read(flags)
    mask = sum(1 << p for p in pos.values())
    if unk & mask:
        return exp, None, text
    act = FPFlags(0)
    for f, p in pos.items():
        if v >> p & 1:
            act |= f
    return exp, act, None


def _compare_fp(exp: FP, act: FP, opts: _Options):
    """-> (kind or None, ulps). Value part only."""
    if exp.raw == act.raw:
        return None, None
    if exp.is_nan or act.is_nan:
        if exp.is_nan and act.is_nan and (
                opts.nan == "any" or (opts.nan == "quiet" and not act.is_snan)):
            return None, None
        return Kind.NAN, None
    if exp.is_zero and act.is_zero:
        sign = 1 << (exp.format.size - 1) if exp.format.signed else 0
        if sign and exp.raw ^ act.raw == sign:
            return (None if not opts.zero_sign else Kind.ZERO_SIGN), None
    ulps = None
    if exp.is_finite and act.is_finite:
        ulps = (act.exact - exp.exact) / min(act.ulp, exp.ulp)
        if opts.tol_ulps and abs(ulps) <= opts.tol_ulps and not (
                exp.is_zero and act.is_zero):
            return None, ulps
    return Kind.VALUE, ulps


def _check_fp(exp: FP, actual, flags, opts, context, label="", index=None):
    fmt = exp.format
    raw, text = _code(actual, fmt.size)
    base = dict(label=label, index=index, fmt=fmt, expected_raw=exp.raw,
                actual_raw=None if text else raw, actual_text=text,
                expected=exp, context=context)
    kind = ulps = None
    act = None
    if text:
        kind = Kind.UNRESOLVED
    else:
        act = fmt.from_raw(raw)
        kind, ulps = _compare_fp(exp, act, opts)
    fl = {}
    if flags is not None:
        e, a, ftext = _flag_state(exp.flags, flags, opts)
        fl = dict(expected_flags=e, actual_flags=a, flags_text=ftext)
        if kind is None and (ftext is not None or a != e):
            kind = Kind.UNRESOLVED if ftext is not None else Kind.FLAGS
    if kind is None:
        return None
    return Mismatch(kind, actual=act, ulps=ulps, **base, **fl)


def _check_int(fmt, expected: int, actual, context, label="", index=None):
    raw, text = _code(actual, fmt.size)
    if text:
        return Mismatch(Kind.UNRESOLVED, label, index, fmt, expected, None, text,
                        context=context)
    if raw == expected:
        return None
    return Mismatch(Kind.VALUE, label, index, fmt, expected, raw, context=context)


def compare(expected: FP, actual, *, flags=None, nan: str = "exact",
            zero_sign: bool = True, tol_ulps=0, flag_map: Mapping | None = None,
            **context) -> Mismatch | None:
    """Compare a DUT output with the model's; None when it passes.

    expected   the model's FP: its code and, if ``flags`` is given, its flags.
    actual     the DUT value: int, bit string, LogicArray, or a handle.
    flags      the DUT's flag value (same types); None skips the flag check.
    nan        "exact": NaN codes must match bit for bit. "quiet": any quiet NaN
               matches any NaN. "any": any NaN matches any NaN. Only used when the
               codes differ; a NaN against a non-NaN is always a mismatch.
    zero_sign  False accepts +0 for -0 (and the reverse).
    tol_ulps   accept finite results within this many ulps (0: bit-exact).
    flag_map   {FPFlags member: bit position in the DUT flag bus}. Default is
               RISC-V fflags order; flags not listed are not compared.
    context    shown in the report (operands as FP values, op names, ...).

    A width or range error (a code that cannot be this format's) raises
    ValueError: that is a testbench bug, not a DUT failure.
    """
    return _check_fp(expected, actual, flags,
                     _Options(nan, zero_sign, tol_ulps, flag_map), context)


# Scoreboard -----------------------------------------------------------------

def _leaves(x, index=()):
    """Yield (index, leaf) for nested lists/tuples."""
    if isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _leaves(v, index + (i,))
    else:
        yield index, x


def _shape(x) -> tuple[int, ...]:
    shape = []
    while isinstance(x, (list, tuple)):
        shape.append(len(x))
        x = x[0] if x else None
    return tuple(shape)


def _pair(expected, actual, what: str):
    """Align the DUT leaves with the expected ones. The DUT value is nested like
    the expected one, or a flat list (what unpacking a bus gives)."""
    ex = list(_leaves(expected))
    ac = [v for _, v in _leaves(actual)]
    if isinstance(actual, (list, tuple)) and (
            _shape(actual) == _shape(expected) or len(_shape(actual)) == 1) \
            and len(ac) == len(ex):
        return [(i, e, a) for (i, e), a in zip(ex, ac)]
    raise ValueError(f"{what}: the DUT gave shape {_shape(actual)} for "
                     f"{len(ex)} expected values of shape {_shape(expected)}")


class Scoreboard:
    """A stream of checks with counters and a failure policy.

    name        in logs and reports.
    fail_fast   True: a failing check raises AssertionError (which fails a
                cocotb test). False: log it, count it and go on; call
                ``assert_clean()`` at the end.
    max_report  mismatches reported in total (logged, kept in ``mismatches``,
                listed by ``summary()``), and per failing array check. The
                counters always count every element.
    nan, zero_sign, tol_ulps, flag_map   as in ``compare``.
    logger      a ``logging.Logger``; default is the cocotb logger in a running
                simulation, else the ``verifloat.scoreboard`` logger.
    """

    def __init__(self, name: str = "", *, fail_fast: bool = True, max_report: int = 10,
                 nan: str = "exact", zero_sign: bool = True, tol_ulps=0,
                 flag_map: Mapping | None = None, logger: logging.Logger | None = None):
        if max_report < 1:
            raise ValueError("max_report must be >= 1")
        self.name, self.fail_fast, self.max_report = name, fail_fast, max_report
        self._opts = _Options(nan, zero_sign, tol_ulps, flag_map)
        self._logger = logger
        self.checked = self.passed = self.calls = 0
        self.by_kind: Counter = Counter()
        self.mismatches: list[Mismatch] = []
        self._silenced = False

    @property
    def failed(self) -> int:
        return self.checked - self.passed

    def logger(self) -> logging.Logger:
        if self._logger is not None:
            return self._logger
        try:
            import cocotb
            if cocotb.is_simulation:
                return logging.getLogger("cocotb.scoreboard")
        except (ImportError, AttributeError):
            pass
        return logging.getLogger("verifloat.scoreboard")

    # Checks
    def check(self, expected: FP, actual, flags=None, **context) -> Mismatch | None:
        """Check one value (see ``compare``). Returns the Mismatch, if any."""
        m = _check_fp(expected, actual, flags, self._opts, context)
        return self._finish([m] if m else [], 1)

    def check_array(self, expected, actual, flags=None, **context) -> list[Mismatch]:
        """Check an FPArray, or nested lists of FP, element by element.

        actual   nested lists of codes (LogicArrays, ints, ...) of the same shape,
                 or a flat list in row-major order, or an FPArray.
        flags    a nested list of per-element flag values, or one value compared with
                 the OR of the expected flags (reported with label "flags").
        """
        per_elem = isinstance(flags, (list, tuple))
        if isinstance(expected, FPArray):
            fmt, total, nested = expected.format, expected.flags, expected.raw
            fps = [f for _, f in _leaves(expected.tolist())] if per_elem else None
        else:
            leaves = [f for _, f in _leaves(expected)]
            if not leaves:
                raise ValueError("check_array: nothing to check")
            fmt, nested, fps = leaves[0].format, expected, None
            total = FPFlags(0)
            for f in leaves:
                total |= f.flags
        if isinstance(actual, FPArray):
            actual = actual.raw
        pairs = _pair(nested, actual, "check_array")
        fl = [v for _, v in _leaves(flags)] if per_elem else None
        if per_elem and len(fl) != len(pairs):
            raise ValueError("flags must have one value per element")
        found = []
        for k, (idx, e, a) in enumerate(pairs):
            f = fl[k] if per_elem else None
            fp = e if isinstance(e, FP) else (fps[k] if fps else None)
            if f is None:       # fast path: only the code matters
                raw, text = _code(a, fmt.size)
                if text is None and raw == (fp.raw if fp else e):
                    continue
            fp = fp or fmt.from_raw(e)
            m = _check_fp(fp, a, f, self._opts, context, "", idx or None)
            if m:
                found.append(m)
        n = len(pairs)
        if flags is not None and not per_elem:
            n += 1
            e, a, text = _flag_state(total, flags, self._opts)
            if text is not None or a != e:
                found.append(Mismatch(Kind.UNRESOLVED if text is not None else Kind.FLAGS,
                                      "flags", None, fmt, expected_flags=e,
                                      actual_flags=a, flags_text=text, context=context))
        return self._finish(found, n, True)

    def check_block(self, expected: BlockTensor, elems=None, scales=None,
                    tensor_scale=None, zeros=None, **context) -> list[Mismatch]:
        """Check the DUT's codes of a BlockTensor: element codes, block scale
        codes, zero-point codes and the tensor scale code. Pass only what the
        DUT outputs; each is nested like the tensor (elements) or like its
        blocks (scales, zeros), or a flat list. Reports use the labels "elem",
        "scale", "zero" and "tensor_scale" and the index within that list."""
        fmt = expected.fmt
        found, n = [], 0
        parts = [("elem", fmt.elem, expected.elem_raw, elems),
                 ("scale", fmt.scale, expected.scale_raw, scales),
                 ("zero", fmt.zero_point, expected.zero_raw, zeros)]
        for label, f, raws, act in parts:
            if act is None:
                continue
            if f is None:
                raise ValueError(f"the format has no {label} codes to check")
            for idx, raw, a in _pair(raws, act, f"check_block {label}"):
                n += 1
                m = _check_code(f, raw, a, self._opts, context, label, idx)
                if m:
                    found.append(m)
        if tensor_scale is not None:
            ts = expected.tensor_scale
            if ts is None:
                raise ValueError("the format has no tensor scale")
            n += 1
            m = _check_fp(ts, tensor_scale, None, self._opts, context, "tensor_scale")
            if m:
                found.append(m)
        return self._finish(found, n, True)

    def _finish(self, found, n, many=False):
        self.calls += 1
        self.checked += n
        self.passed += n - len(found)
        self.by_kind.update(m.kind.value for m in found)
        shown = found[:self.max_report]
        room = self.max_report - len(self.mismatches)
        self.mismatches.extend(shown[:max(room, 0)])
        if found:
            prefix = (f"{self.name} " if self.name else "") + f"#{self.calls - 1}: "
            text = "\n".join(m.report(prefix if i == 0 else "", i == 0) for i, m in enumerate(shown))
            if len(found) > len(shown):
                text += f"\n  ... and {len(found) - len(shown)} more mismatches in this check"
            if self.fail_fast:
                raise AssertionError(text)
            if room > 0:
                self.logger().error(text)
            elif not self._silenced:
                self._silenced = True
                self.logger().error(f"{self.name or 'scoreboard'}: further mismatches "
                                    f"not logged (max_report={self.max_report})")
        if many:
            return found
        return found[0] if found else None

    # Results
    def summary(self) -> str:
        head = (f"{self.name or 'scoreboard'}: {self.checked} checked, "
                f"{self.passed} passed, {self.failed} failed")
        if self.by_kind:
            head += " (" + ", ".join(f"{k} {v}" for k, v in sorted(self.by_kind.items())) + ")"
        if not self.mismatches:
            return head
        lines = [head, f"first {len(self.mismatches)} mismatches:"]
        lines += [m.report() for m in self.mismatches]
        return "\n".join(lines)

    def assert_clean(self, *, allow_empty: bool = False) -> None:
        """Raise AssertionError if any check failed, or if none ran (a test
        that compared nothing must not pass)."""
        if self.failed:
            raise AssertionError(self.summary())
        if not self.checked and not allow_empty:
            raise AssertionError(f"{self.name or 'scoreboard'}: no checks were made")


def _check_code(fmt, expected_raw: int, actual, opts, context, label, index):
    if isinstance(fmt, FPFormat):
        return _check_fp(fmt.from_raw(expected_raw), actual, None, opts, context,
                         label, index)
    return _check_int(fmt, expected_raw, actual, context, label, index)

