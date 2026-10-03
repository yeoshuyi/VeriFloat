"""Floating point: formats, rounding modes and the FP type.

FP values and rounding run in the C++ core (``verifloat._core``); this module
holds the format description (``FPFormat``), the enums, and the few
operations that are naturally Python (name parsing, the common-format search,
warning messages).
"""

from __future__ import annotations

import dataclasses
import enum
import math
import random
import re
from dataclasses import KW_ONLY, dataclass
from fractions import Fraction

from . import _core
from ._warnings import (CastWarning, FPFormatWarning, FPModeWarning,
                        FPOverflowWarning, FPUnderflowWarning, IntCastWarning)
from ._warnings import warn as _warn


def _pow2(k: int) -> Fraction:
    return Fraction(1 << k) if k >= 0 else Fraction(1, 1 << -k)


def _default_bias(exp_bits: int) -> int:
    return (1 << (exp_bits - 1)) - 1


class Rounding(enum.Enum):
    """IEEE-754 rounding-direction attributes, plus stochastic rounding."""
    RNE = "rne"  # nearest, ties to even (IEEE default)
    RNA = "rna"  # nearest, ties away from zero
    RTZ = "rtz"  # toward zero (truncate)
    RUP = "rup"  # toward +infinity
    RDN = "rdn"  # toward -infinity
    SR = "sr"    # stochastic: round away when (fraction + random) carries


class NaNMode(enum.Enum):
    """Which NaN an operation returns (IEEE 754 leaves this to the
    implementation). Also selects the integer returned by an invalid
    float-to-integer conversion."""
    CANONICAL = "canonical"  # RISC-V: always the canonical quiet NaN
    PROPAGATE = "propagate"  # first NaN operand, quieted; else canonical
    X86 = "x86"              # x86 SSE: first NaN operand; else negative default NaN
    ARM = "arm"              # ARM VFP: first signaling NaN, else first NaN operand


class FPFlags(enum.IntFlag):
    """IEEE-754 status flags, laid out like RISC-V fflags (NX = bit 0)."""
    INEXACT = 1 << 0
    UNDERFLOW = 1 << 1
    OVERFLOW = 1 << 2
    DIVZERO = 1 << 3
    INVALID = 1 << 4


_F = FPFlags
_ROUNDINGS = list(Rounding)
_NAN_MODES = list(NaNMode)


# Stochastic rounding draws its random bits from this source (nbits -> int).
_sr_state = {"source": random.Random(0).getrandbits}


def set_sr_source(source) -> None:
    """Set the random-bit source for Rounding.SR: a callable taking a bit
    count and returning that many random bits (e.g. a model of the DUT's LFSR),
    or an int seed for a private random.Random."""
    if isinstance(source, int):
        source = random.Random(source).getrandbits
    _sr_state["source"] = source


def _show(x) -> str:
    if x is None:
        return "an irrational result"
    if x is ...:        # the core did not build it (see fp_finish)
        return "a value too large to write out"
    try:
        return repr(float(x))
    except OverflowError:
        pass
    try:
        return str(x)
    except ValueError:  # more digits than Python converts to text
        return _approx(x)


def _approx(x) -> str:
    """Scientific notation for a huge rational, from its leading bits."""
    x = Fraction(x)
    n, d = abs(x.numerator), x.denominator
    sn, sd = max(n.bit_length() - 64, 0), max(d.bit_length() - 64, 0)
    lg = math.log10((n >> sn) / (d >> sd)) + (sn - sd) * math.log10(2)
    e = math.floor(lg)
    return f"about {'-' if x < 0 else ''}{10 ** (lg - e):.6f}e{e:+d}"


_NAME_RE = re.compile(r"(u?)e(\d+)m(\d+)((?:, [a-z0-9_-]+(?:=-?[a-z0-9]+)?)*)")
_INF_NAN_NAME = {True: "ieee", False: "finite", "fn": "fn"}
_INF_NAN_CODE = {False: 0, True: 1, "fn": 2}

# Equal formats share one id, so the core compares formats in O(1).
_INTERN: dict[tuple, int] = {}


@dataclass(frozen=True)
class FPFormat(_core.FormatBase):
    """A complete FP format: field widths, bias, signedness and modes.

    Calling a format rounds a value into it: ``FPFormat(4, 3)(0.1)``.

    inf_nan    True: IEEE 754 specials (all-ones exponent is inf/NaN).
               "fn": OCP "finite + NaN" (only the all-ones code is NaN, no inf).
               False: every code is a finite number (overflow saturates).
    has_zero   False: exponent field 0 is an ordinary binade, so there is no
               zero and no subnormal (OCP E8M0).
    saturate   overflow gives +-max instead of inf/NaN (like cvt.satfinite).
    tininess   "after" (x86, RISC-V) or "before" (ARM) rounding.
    nan_mode   which NaN results are returned (see NaNMode).
    sr_bits    random bits used by Rounding.SR.
    """
    exp_bits: int = 2
    mantissa_bits: int = 1
    bias: int | None = None
    signed: bool = True
    _: KW_ONLY
    inf_nan: bool | str = True
    has_zero: bool = True
    saturate: bool = False
    wrap: bool = False
    ftz: bool = False
    rounding: Rounding = Rounding.RNE
    sr_bits: int = 8
    nan_mode: NaNMode = NaNMode.CANONICAL
    tininess: str = "after"

    def __post_init__(self):
        for name in ("exp_bits", "mantissa_bits", "sr_bits"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool):
                raise TypeError(f"{name} must be an int")
        if self.exp_bits < 1:
            raise ValueError("need at least 1 exponent bit")
        if self.mantissa_bits < 0:
            raise ValueError("mantissa_bits must be >= 0")
        if self.sr_bits < 1:
            raise ValueError("sr_bits must be >= 1")
        if self.bias is None:
            object.__setattr__(self, "bias", _default_bias(self.exp_bits))
        elif not isinstance(self.bias, int) or isinstance(self.bias, bool):
            raise TypeError("bias must be an int or None")
        if self.inf_nan not in (True, False, "fn"):
            raise ValueError("inf_nan must be True, False or 'fn'")
        if self.inf_nan is True and self.mantissa_bits == 0:
            raise ValueError("IEEE inf/NaN needs a mantissa bit; use inf_nan='fn'")
        if self.inf_nan is True and self.exp_bits < 2:
            raise ValueError("IEEE inf/NaN needs at least 2 exponent bits")
        if self.tininess not in ("before", "after"):
            raise ValueError("tininess must be 'before' or 'after'")
        if not isinstance(self.rounding, Rounding):
            raise TypeError("rounding must be a Rounding member")
        if not isinstance(self.nan_mode, NaNMode):
            raise TypeError("nan_mode must be a NaNMode member")
        for name in ("signed", "has_zero", "saturate", "wrap", "ftz"):
            object.__setattr__(self, name, bool(getattr(self, name)))
        if self._max_code < (0 if not self.has_zero else 1):
            raise ValueError(f"format {self} has no positive finite value")
        if (self.exp_bits > 60 or abs(self.bias) > 1 << 60 or self.mantissa_bits > 1 << 24
                or self.sr_bits > 1 << 20):
            raise ValueError("format too large for the C++ core: needs exp_bits <= 60, "
                             "|bias| <= 2**60, mantissa_bits <= 2**24, sr_bits <= 2**20")
        key = (self.exp_bits, self.mantissa_bits, self.bias, self.signed,
               _INF_NAN_CODE[self.inf_nan], self.has_zero, self.saturate, self.wrap,
               self.ftz, self.rounding, self.sr_bits, self.nan_mode, self.tininess)
        fid = _INTERN.setdefault(key, len(_INTERN))
        self._setup(self.exp_bits, self.mantissa_bits, self.bias, self.signed,
                    _INF_NAN_CODE[self.inf_nan], self.has_zero, self.saturate, self.wrap,
                    self.ftz, _ROUNDINGS.index(self.rounding), self.sr_bits,
                    _NAN_MODES.index(self.nan_mode), self.tininess == "before", fid)

    # Construction: calling a format (``fmt(value, sr_rand=None)``) rounds a
    # value into it; that is implemented by the native base class.
    def from_raw(self, raw: int) -> "FP":
        return _core.fp_from_raw(self, raw)

    def array(self, values) -> "_core.FPArray":
        """Round nested lists of values into an FPArray of this format."""
        return _core.FPArray(values, self)

    def all_values(self):
        """Yield every encoding of this format in raw-code order."""
        from_raw = _core.fp_from_raw
        for raw in range(1 << self.size):
            yield from_raw(self, raw)

    def zero(self, sign: bool = False) -> "FP":
        if not self.has_zero:
            raise ValueError(f"{self} has no zero")
        return _core.fp_make(self, sign, 0, 0)

    def max_value(self, sign: bool = False) -> "FP":
        """Largest finite value."""
        return _core.fp_make(self, sign, self._max_field, self._max_mant)

    def inf(self, sign: bool = False) -> "FP":
        if not self.has_inf:
            raise ValueError(f"{self} has no infinity")
        return _core.fp_make(self, sign, self._top, 0)

    def nan(self, sign: bool = False, payload: int | None = None) -> "FP":
        """A NaN: the canonical quiet NaN, or one with the given mantissa."""
        if not self.has_nan:
            raise ValueError(f"{self} has no NaN")
        if self.inf_nan == "fn":
            payload = self._mask
        elif payload is None:
            payload = 1 << (self.mantissa_bits - 1)
        elif not payload & self._mask:
            raise ValueError("a NaN payload must be nonzero")
        return _core.fp_make(self, sign and self.signed, self._top, payload)

    def replace(self, **changes) -> "FPFormat":
        """Copy with some fields changed. A default bias follows exp_bits."""
        if ("exp_bits" in changes and "bias" not in changes
                and self.bias == _default_bias(self.exp_bits)):
            changes["bias"] = None
        return dataclasses.replace(self, **changes)

    # Encoding geometry
    @property
    def has_inf(self) -> bool:
        return self.inf_nan is True

    @property
    def has_nan(self) -> bool:
        return self.inf_nan is not False

    @property
    def _top(self) -> int:
        return (1 << self.exp_bits) - 1

    @property
    def _mask(self) -> int:
        return (1 << self.mantissa_bits) - 1

    @property
    def _max_code(self) -> int:
        """Magnitude code (exponent and mantissa fields) of the largest finite value."""
        all_ones = (1 << (self.exp_bits + self.mantissa_bits)) - 1
        if self.inf_nan is True:
            return all_ones - (1 << self.mantissa_bits)   # top exponent reserved
        if self.inf_nan == "fn":
            return all_ones - 1                            # all-ones code is NaN
        return all_ones

    @property
    def _max_field(self) -> int:
        return self._max_code >> self.mantissa_bits

    @property
    def _max_mant(self) -> int:
        return self._max_code & self._mask

    @property
    def _min_field(self) -> int:
        """Exponent field of the lowest normal binade."""
        return 1 if self.has_zero else 0

    # Range data
    @property
    def size(self) -> int:
        return int(self.signed) + self.exp_bits + self.mantissa_bits

    @property
    def emin(self) -> int:
        """Exponent of the smallest normal binade."""
        return self._min_field - self.bias

    @property
    def emax(self) -> int:
        """Exponent of the largest finite binade."""
        return self._max_field - self.bias

    @property
    def max(self) -> Fraction:
        """Largest finite value."""
        if self.has_zero and self._max_field == 0:   # only subnormals
            return Fraction(self._max_mant, 1 << self.mantissa_bits) * _pow2(self.emin)
        return (1 + Fraction(self._max_mant, 1 << self.mantissa_bits)) * _pow2(self.emax)

    @property
    def min_normal(self) -> Fraction:
        return _pow2(self.emin)

    @property
    def min_subnormal(self) -> Fraction:
        """Smallest positive value (decodes as 0 when ftz is set)."""
        if self.has_zero and self.mantissa_bits:
            return _pow2(self.emin - self.mantissa_bits)
        return _pow2(self.emin)

    @property
    def kw(self) -> dict:
        """Keyword arguments equivalent to this format for the FP APIs."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
                if f.name not in ("exp_bits", "mantissa_bits", "_")}

    # Names
    def _tags(self):
        if self.bias != _default_bias(self.exp_bits):
            yield f"bias={self.bias}"
        if self.inf_nan is not True:
            yield _INF_NAN_NAME[self.inf_nan]
        if not self.has_zero:
            yield "no-zero"
        for flag in ("saturate", "wrap", "ftz"):
            if getattr(self, flag):
                yield flag
        if self.rounding is not Rounding.RNE:
            yield self.rounding.value
        if self.sr_bits != 8:
            yield f"sr_bits={self.sr_bits}"
        if self.nan_mode is not NaNMode.CANONICAL:
            yield self.nan_mode.value
        if self.tininess != "after":
            yield "tininess=before"

    def __str__(self) -> str:
        name = f"{'' if self.signed else 'u'}e{self.exp_bits}m{self.mantissa_bits}"
        return ", ".join([name, *self._tags()])

    def __repr__(self) -> str:
        args = [str(self.exp_bits), str(self.mantissa_bits)]
        if self.bias != _default_bias(self.exp_bits):
            args.append(f"bias={self.bias}")
        if not self.signed:
            args.append("signed=False")
        if self.inf_nan is not True:
            args.append(f"inf_nan={self.inf_nan!r}")
        if not self.has_zero:
            args.append("has_zero=False")
        for flag in ("saturate", "wrap", "ftz"):
            if getattr(self, flag):
                args.append(f"{flag}=True")
        if self.rounding is not Rounding.RNE:
            args.append(f"rounding=Rounding.{self.rounding.name}")
        if self.sr_bits != 8:
            args.append(f"sr_bits={self.sr_bits}")
        if self.nan_mode is not NaNMode.CANONICAL:
            args.append(f"nan_mode=NaNMode.{self.nan_mode.name}")
        if self.tininess != "after":
            args.append(f"tininess={self.tininess!r}")
        return f"FPFormat({', '.join(args)})"

    def __reduce__(self):
        return (_rebuild_format, (self.exp_bits, self.mantissa_bits, self.kw))

    @classmethod
    def parse(cls, name: str) -> "FPFormat":
        """Inverse of str(): e.g. 'ue4m3, bias=10, fn, saturate, rtz'."""
        m = _NAME_RE.fullmatch(name.strip())
        if not m:
            raise ValueError(f"not an FP format name: {name!r}")
        kw: dict = dict(signed=not m[1])
        for tag in filter(None, m[4].split(", ")):
            key, _, val = tag.partition("=")
            if key == "bias":
                kw["bias"] = int(val)
            elif key == "sr_bits":
                kw["sr_bits"] = int(val)
            elif key == "tininess":
                kw["tininess"] = val
            elif tag == "finite":
                kw["inf_nan"] = False
            elif tag == "fn":
                kw["inf_nan"] = "fn"
            elif tag == "no-zero":
                kw["has_zero"] = False
            elif tag in ("saturate", "wrap", "ftz"):
                kw[tag] = True
            elif tag in {r.value for r in Rounding}:
                kw["rounding"] = Rounding(tag)
            elif tag in {n.value for n in NaNMode}:
                kw["nan_mode"] = NaNMode(tag)
            else:
                raise ValueError(f"unknown format tag {tag!r} in {name!r}")
        return cls(int(m[2]), int(m[3]), **kw)


def _rebuild_format(exp_bits, mantissa_bits, kw):
    return FPFormat(exp_bits, mantissa_bits, **kw)


# Presets. OCP formats follow the OCP 8-bit FP / MX specifications.
E2M1 = FPFormat(2, 1, inf_nan=False)           # OCP FP4 (NVFP4/MXFP4 element)
E2M3 = FPFormat(2, 3, inf_nan=False)           # OCP FP6
E3M2 = FPFormat(3, 2, inf_nan=False)           # OCP FP6
E4M3 = FPFormat(4, 3, inf_nan="fn")            # OCP FP8 E4M3 (NaN only, max 448)
E5M2 = FPFormat(5, 2)                          # OCP FP8 E5M2 (IEEE-like)
UE4M3 = FPFormat(4, 3, signed=False, inf_nan="fn")
UE8M0 = FPFormat(8, 0, signed=False, inf_nan="fn", has_zero=False)  # OCP E8M0
FP16 = FPFormat(5, 10)                         # IEEE binary16
BF16 = FPFormat(8, 7)                          # bfloat16
FP32 = E8M23 = FPFormat(8, 23)                 # IEEE binary32
FP64 = FPFormat(11, 52)                        # IEEE binary64


def _resolve(exp_bits, mantissa_bits, bias, signed, fmt, modes) -> FPFormat:
    """Build an FPFormat from either an FPFormat or the field arguments."""
    if isinstance(exp_bits, FPFormat):
        fmt, exp_bits = exp_bits, 2
    if fmt is not None:
        if (exp_bits, mantissa_bits, bias, signed) != (2, 1, None, True) or modes:
            raise TypeError("pass either an FPFormat or format fields, not both")
        return fmt
    return FPFormat(exp_bits, mantissa_bits, bias, signed, **modes)


# Pure-Python rounding helpers, kept for code that rounds Python numbers
# outside the FP type (fixed point, integer quantizers).
def _round_sig(n: int, d: int, shift: int, neg: bool, rounding: Rounding,
               odd: bool | None = None, sr: int = 0, sr_bits: int = 8):
    """Round n/d / 2**shift to an integer. Returns (sig, inexact).

    ``odd`` overrides the parity used by ties-to-even (for zero-width
    mantissas, where the code's parity is the exponent's). ``sr`` holds the
    random bits for stochastic rounding."""
    num, den = (n, d << shift) if shift >= 0 else (n << -shift, d)
    sig, rem = divmod(num, den)
    if rem:
        if rounding is Rounding.RNE:
            is_odd = sig & 1 if odd is None else odd
            up = 2 * rem > den or (2 * rem == den and is_odd)
        elif rounding is Rounding.RNA:
            up = 2 * rem >= den
        elif rounding is Rounding.RTZ:
            up = False
        elif rounding is Rounding.RUP:
            up = not neg
        elif rounding is Rounding.RDN:
            up = neg
        else:
            # Stochastic: the discarded fraction, rounded to sr_bits bits
            # (ties to even), plus the random bits; carry out rounds away.
            q, r = divmod(rem << sr_bits, den)
            q += 2 * r > den or (2 * r == den and q & 1)
            up = q + sr >= 1 << sr_bits
        sig += up
    return sig, rem != 0


def _round_to_int(x: Fraction, rounding: Rounding, sr: int = 0, sr_bits: int = 8):
    """Round x to an integer. Returns (q, inexact)."""
    neg = x < 0
    q, inexact = _round_sig(abs(x).numerator, abs(x).denominator, 0, neg,
                            rounding, sr=sr, sr_bits=sr_bits)
    return (-q if neg else q), inexact


def _floor_log2(n: int, d: int) -> int:
    e = n.bit_length() - d.bit_length()
    if (n << max(0, -e)) < (d << max(0, e)):
        e -= 1
    return e


# Rounding events that can raise a warning, and the warning they raise.
_EVENTS = {
    "saturated": FPOverflowWarning,
    "overflowed": FPOverflowWarning,
    "wrapped-up": FPOverflowWarning,
    "wrapped-down": FPUnderflowWarning,
    "clamped": FPUnderflowWarning,
    "underflowed": FPUnderflowWarning,
    "flushed": FPUnderflowWarning,
}


def _report(event: str, x, fmt: FPFormat, result: "FP") -> None:
    if event == "clamped":
        msg = f"negative value {_show(x)} clamped to {result} in {fmt}"
    else:
        kind = "overflowed" if _EVENTS[event] is FPOverflowWarning else "underflowed"
        how = {"saturated": "saturated to", "overflowed": "rounded to",
               "wrapped-up": "exponent wrapped to",
               "wrapped-down": "exponent wrapped to", "underflowed": "rounded to",
               "flushed": "flushed to"}[event]
        msg = f"value {_show(x)} {kind} {fmt}, {how} {result}"
    _warn(_EVENTS[event], msg)


_MODE_FIELDS = ("saturate", "wrap", "ftz", "rounding", "sr_bits", "nan_mode", "tininess")


def _common_of(fa: FPFormat, fb: FPFormat) -> FPFormat:
    """Smallest format holding every value of both formats. Signed if either
    is; has inf/NaN if either does; the remaining modes come from fa."""
    if fa == fb:
        return fa
    specials = {fa.inf_nan, fb.inf_nan}
    inf_nan = True if True in specials else ("fn" if "fn" in specials else False)
    M = max(fa.mantissa_bits, fb.mantissa_bits, 1 if inf_nan is True else 0)
    # With at least as many mantissa bits and the lower emin, every
    # finite value of either format (subnormals included) is exact.
    emin = min(fa.emin, fb.emin)
    need = max(fa.max, fb.max)
    E = 2
    while FPFormat(E, M, 1 - emin, inf_nan=inf_nan).max < need:
        E += 1
    return FPFormat(E, M, 1 - emin, fa.signed or fb.signed, inf_nan=inf_nan,
                    **{k: getattr(fa, k) for k in _MODE_FIELDS})


def _warn_mismatch(fa: FPFormat, fb: FPFormat, fmt: FPFormat) -> None:
    if ((fa.exp_bits, fa.mantissa_bits, fa.bias)
            != (fb.exp_bits, fb.mantissa_bits, fb.bias)):
        _warn(FPFormatWarning,
              f"mixing FP formats {fa} and {fb}, promoted to {fmt}")
    if fa.signed != fb.signed:
        _warn(FPModeWarning, f"mismatched signed: {fa} (left) vs {fb} "
              f"(right), result is signed")
    if fa.inf_nan != fb.inf_nan:
        _warn(FPModeWarning, f"mismatched inf_nan: {_INF_NAN_NAME[fa.inf_nan]} "
              f"(left) vs {_INF_NAN_NAME[fb.inf_nan]} (right), result is "
              f"{_INF_NAN_NAME[fmt.inf_nan]}")
    if fa.has_zero != fb.has_zero:
        _warn(FPModeWarning, f"mismatched has_zero: {fa.has_zero} (left) vs "
              f"{fb.has_zero} (right), result has zero")
    for tag in _MODE_FIELDS:
        lv, rv = getattr(fa, tag), getattr(fb, tag)
        if lv != rv:
            show = (lambda v: v.value) if isinstance(lv, enum.Enum) else (lambda v: v)
            _warn(FPModeWarning, f"mismatched {tag}: {show(lv)} (left) vs "
                  f"{show(rv)} (right), using {show(lv)}")


def _cast_warn(other, fmt: FPFormat, value: float, flags: FPFlags) -> None:
    try:
        shown = repr(other)
    except ValueError:      # more digits than Python converts to text
        shown = _approx(other)
    _warn(CastWarning, f"implicit cast of {shown} to {fmt} is lossy: "
          f"{value!r} ({flags.name})")


_core._init(Fraction, [FPFlags(i) for i in range(32)], _ROUNDINGS, _NAN_MODES,
            _warn, IntCastWarning, _report, _warn_mismatch, _common_of, _cast_warn,
            _sr_state, FPFormat)


# ---------------------------------------------------------------- FP
# The type is native; the constructors that take format fields, and a few
# rarely used helpers, are added here.

FP = _core.FP


def _from_value(cls, value, exp_bits: int | FPFormat = 2, mantissa_bits: int = 1,
                bias: int | None = None, signed: bool = True, *,
                fmt: FPFormat | None = None, sr_rand: int | None = None, **modes) -> FP:
    """Round a number (int, float, Fraction, FP, UINT/INT) into a format.
    ``sr_rand`` supplies the random bits for Rounding.SR explicitly."""
    return _core.fp_from_value(value, _resolve(exp_bits, mantissa_bits, bias, signed, fmt, modes),
                               sr_rand)


def _zero(cls, exp_bits: int | FPFormat = 2, mantissa_bits: int = 1,
          sign: bool = False, bias=None, signed=True, *, fmt=None, **modes) -> FP:
    return _resolve(exp_bits, mantissa_bits, bias, signed, fmt, modes).zero(sign)


def _max_value(cls, exp_bits: int | FPFormat = 2, mantissa_bits: int = 1,
               sign: bool = False, bias=None, signed=True, *, fmt=None, **modes) -> FP:
    return _resolve(exp_bits, mantissa_bits, bias, signed, fmt, modes).max_value(sign)


def _from_raw(cls, raw: int, exp_bits: int | FPFormat = 2, mantissa_bits: int = 1,
              bias: int | None = None, signed: bool = True, *,
              fmt: FPFormat | None = None, **modes) -> FP:
    return _core.fp_from_raw(_resolve(exp_bits, mantissa_bits, bias, signed, fmt, modes), raw)


def _all_values(cls, exp_bits: int | FPFormat = 2, mantissa_bits: int = 1,
                bias: int | None = None, signed: bool = True, *,
                fmt: FPFormat | None = None, **modes):
    """Yield every encoding of a format in raw-code order."""
    return _resolve(exp_bits, mantissa_bits, bias, signed, fmt, modes).all_values()


def _convert(self, exp_bits: int | FPFormat = 2, mantissa_bits: int = 1,
             bias: int | None = None, signed: bool = True, *,
             fmt: FPFormat | None = None, sr_rand: int | None = None, **modes) -> FP:
    return self._convert_to(_resolve(exp_bits, mantissa_bits, bias, signed, fmt, modes), sr_rand)


def _ulp(self) -> Fraction:
    """Spacing of adjacent encodings in this value's binade."""
    f = self.format
    return _pow2(max(self.exp.val, f._min_field) - f.bias - f.mantissa_bits)


def _error_ulps(self, ref=None) -> Fraction:
    """(exact - ref) in ulps of this value; ref defaults to unrounded."""
    if ref is None:
        ref = self.unrounded
        if ref is None:
            raise ValueError("no unrounded value; pass ref explicitly")
    if isinstance(ref, FP):
        ref = ref.exact
    return (self.exact - Fraction(ref)) / self.ulp


def _key(self):
    """Value for ordering: exact, or +-inf as a float (not for NaN)."""
    if self.is_inf:
        return -math.inf if self.sign else math.inf
    return self.exact


def _restore(fmt: FPFormat, raw: int, flags: int, unrounded) -> FP:
    """Unpickle an FP (see FP.__reduce__)."""
    return _core.fp_restore(fmt, raw, flags, unrounded)


# Internal hooks with the reference implementation's signatures.
def _round(x, fmt: FPFormat, zero_sign: bool = False, sr=None):
    r, flags, event = _core.fp_round(x, fmt, zero_sign)
    return r, FPFlags(flags), event


def _encode(x, fmt: FPFormat, zero_sign: bool = False, sr=None) -> FP:
    return _core.fp_encode(x, fmt, zero_sign, sr)


def _finish(result: FP, flags, event, unrounded, fmt: FPFormat) -> FP:
    return _core.fp_finish(result, int(flags), event, unrounded, fmt)


def _nan_result(fmt: FPFormat, *operands: FP, invalid: bool = False):
    return _core.fp_nan_result(fmt, list(operands), invalid)


def _sum_zero_sign(sa: bool, sb: bool, fmt: FPFormat) -> bool:
    """IEEE 754 6.3: an exact zero sum of opposite-signed operands is +0,
    except -0 when rounding toward negative."""
    return sa if sa == sb else fmt.rounding is Rounding.RDN


# FP.from_value and FP.convert are native for (value, FPFormat) / (FPFormat);
# other signatures (format fields, fmt=) go through these resolvers.
_core._init_slow(_from_value, _convert)

for _name, _obj in {
    "zero": classmethod(_zero),
    "max_value": classmethod(_max_value),
    "from_raw": classmethod(_from_raw),
    "all_values": classmethod(_all_values),
    "ulp": property(_ulp),
    "error_ulps": _error_ulps,
    "_key": _key,
    "_restore": staticmethod(_restore),
    "_make": staticmethod(lambda sign, field, mant, fmt: _core.fp_make(fmt, sign, field, mant)),
    "_from_raw_fmt": staticmethod(lambda raw, fmt: _core.fp_from_raw(fmt, raw)),
    "_round": staticmethod(_round),
    "_encode": staticmethod(_encode),
    "_finish": staticmethod(_finish),
    "_arith": staticmethod(_core.fp_arith),
    "_nan_result": staticmethod(_nan_result),
    "_inf_result": staticmethod(_core.fp_inf_result),
    "_sum_zero_sign": staticmethod(_sum_zero_sign),
    "_common_of": staticmethod(_common_of),
}.items():
    _fn = getattr(_obj, "__func__", None) or getattr(_obj, "fget", None) or _obj
    if getattr(_fn, "__module__", None) == __name__ and _fn.__name__ != "<lambda>":
        _fn.__name__, _fn.__qualname__ = _name, f"FP.{_name}"   # as tracebacks and help() show them
    setattr(FP, _name, _obj)
del _name, _obj, _fn
_from_value.__name__, _from_value.__qualname__ = "from_value", "FP.from_value"
_convert.__name__, _convert.__qualname__ = "convert", "FP.convert"

# Names the 0.1 module also exposed.
from .sint import INT  # noqa: E402,F401
from .uint import UINT  # noqa: E402,F401
