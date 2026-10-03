from __future__ import annotations

import dataclasses
import enum
import math
import random
import re
from dataclasses import KW_ONLY, dataclass
from fractions import Fraction

from ._warnings import (CastWarning, FPFormatWarning, FPModeWarning,
                        FPOverflowWarning, FPUnderflowWarning)
from ._warnings import warn as _warn
from .sint import INT
from .uint import UINT


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
    try:
        return repr(float(x))
    except OverflowError:
        return str(x)


_NAME_RE = re.compile(r"(u?)e(\d+)m(\d+)((?:, [a-z0-9_-]+(?:=-?[a-z0-9]+)?)*)")
_INF_NAN_NAME = {True: "ieee", False: "finite", "fn": "fn"}


@dataclass(frozen=True)
class FPFormat:
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

    # Construction
    def __call__(self, value, *, sr_rand: int | None = None) -> "FP":
        return FP.from_value(value, self, sr_rand=sr_rand)

    def from_raw(self, raw: int) -> "FP":
        return FP._from_raw_fmt(raw, self)

    def all_values(self):
        """Yield every encoding of this format in raw-code order."""
        for raw in range(1 << self.size):
            yield FP._from_raw_fmt(raw, self)

    def zero(self, sign: bool = False) -> "FP":
        if not self.has_zero:
            raise ValueError(f"{self} has no zero")
        return FP._make(sign, 0, 0, self)

    def max_value(self, sign: bool = False) -> "FP":
        """Largest finite value."""
        return FP._make(sign, self._max_field, self._max_mant, self)

    def inf(self, sign: bool = False) -> "FP":
        if not self.has_inf:
            raise ValueError(f"{self} has no infinity")
        return FP._make(sign, self._top, 0, self)

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
        return FP._make(sign and self.signed, self._top, payload, self)

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


def _should_warn(event: str | None, fmt: FPFormat) -> bool:
    # Unsigned formats warn on every event; signed formats follow standard
    # FPU behaviour silently, except for the opt-in exponent wrap.
    return event is not None and (not fmt.signed or event.startswith("wrapped"))


_MODE_FIELDS = ("saturate", "wrap", "ftz", "rounding", "sr_bits", "nan_mode", "tininess")


class FP:
    __slots__ = ("_sign", "_exp", "_mantissa", "_format", "_flags", "_unrounded")

    def __init__(self, sign: bool, exp: UINT, mantissa: UINT,
                 bias: int | None = None, signed: bool = True, **modes):
        if not isinstance(exp, UINT) or not isinstance(mantissa, UINT):
            raise TypeError("exp and mantissa must be UINT")
        fmt = FPFormat(exp.bits, mantissa.bits, bias, signed, **modes)
        if sign and not fmt.signed:
            raise ValueError("unsigned FP cannot have its sign set")
        self._sign = bool(sign)
        self._exp = exp
        self._mantissa = mantissa
        self._format = fmt
        self._flags = FPFlags(0)
        self._unrounded = None

    # Construction
    @classmethod
    def _make(cls, sign: bool, field: int, mant: int, fmt: FPFormat) -> "FP":
        if sign and not fmt.signed:
            raise ValueError("unsigned FP cannot have its sign set")
        x = object.__new__(cls)
        x._sign = bool(sign)
        x._exp = UINT(field, fmt.exp_bits)
        x._mantissa = UINT(mant, fmt.mantissa_bits)
        x._format = fmt
        x._flags = FPFlags(0)
        x._unrounded = None
        return x

    @classmethod
    def _from_raw_fmt(cls, raw: int, fmt: FPFormat) -> "FP":
        E, M = fmt.exp_bits, fmt.mantissa_bits
        s = fmt.signed and (raw >> (E + M)) & 1
        return cls._make(bool(s), raw >> M, raw, fmt)

    @classmethod
    def from_value(cls, value, exp_bits: int | FPFormat = 2, mantissa_bits: int = 1,
                   bias: int | None = None, signed: bool = True, *,
                   fmt: FPFormat | None = None, sr_rand: int | None = None,
                   **modes) -> "FP":
        """Round a number (int, float, Fraction, FP, UINT/INT) into a format.
        ``sr_rand`` supplies the random bits for Rounding.SR explicitly."""
        fmt = _resolve(exp_bits, mantissa_bits, bias, signed, fmt, modes)
        if isinstance(value, FP):
            return value._convert_to(fmt, sr_rand)
        if isinstance(value, UINT):
            value = value.val
        elif not isinstance(value, (int, float, Fraction)) and hasattr(value, "exact"):
            value = value.exact        # e.g. a FIXED value
        if isinstance(value, float) and not math.isfinite(value):
            return cls._finish(*cls._special_in(value, fmt), fmt)
        zero_sign = isinstance(value, float) and math.copysign(1.0, value) < 0
        return cls._encode(Fraction(value), fmt, zero_sign, sr_rand)

    @classmethod
    def zero(cls, exp_bits: int | FPFormat = 2, mantissa_bits: int = 1,
             sign: bool = False, bias=None, signed=True, *, fmt=None,
             **modes) -> "FP":
        return _resolve(exp_bits, mantissa_bits, bias, signed, fmt, modes).zero(sign)

    @classmethod
    def max_value(cls, exp_bits: int | FPFormat = 2, mantissa_bits: int = 1,
                  sign: bool = False, bias=None, signed=True, *, fmt=None,
                  **modes) -> "FP":
        return _resolve(exp_bits, mantissa_bits, bias, signed, fmt,
                        modes).max_value(sign)

    @classmethod
    def from_raw(cls, raw: int, exp_bits: int | FPFormat = 2, mantissa_bits: int = 1,
                 bias: int | None = None, signed: bool = True, *,
                 fmt: FPFormat | None = None, **modes) -> "FP":
        return cls._from_raw_fmt(raw, _resolve(exp_bits, mantissa_bits, bias,
                                               signed, fmt, modes))

    @classmethod
    def all_values(cls, exp_bits: int | FPFormat = 2, mantissa_bits: int = 1,
                   bias: int | None = None, signed: bool = True, *,
                   fmt: FPFormat | None = None, **modes):
        """Yield every encoding of a format in raw-code order."""
        return _resolve(exp_bits, mantissa_bits, bias, signed, fmt,
                        modes).all_values()

    # Getters
    @property
    def format(self) -> FPFormat:
        return self._format

    @property
    def sign(self) -> bool:
        return self._sign

    @property
    def exp(self) -> UINT:
        return self._exp

    @property
    def mantissa(self) -> UINT:
        return self._mantissa

    def __getattr__(self, name):
        # Format fields (exp_bits, bias, rounding, ...) read through.
        if name in FPFormat.__dataclass_fields__ and name != "_":
            return getattr(self._format, name)
        raise AttributeError(name)

    @property
    def size(self) -> int:
        """Storage width in bits: [sign +] exponent + mantissa."""
        return self._format.size

    @property
    def raw(self) -> int:
        return ((int(self._sign) << (self.exp_bits + self.mantissa_bits))
                | (self._exp.val << self.mantissa_bits)
                | self._mantissa.val)

    # Classification
    @property
    def is_nan(self) -> bool:
        f = self._format
        if not f.has_nan or self._exp.val != f._top:
            return False
        if f.inf_nan == "fn":
            return self._mantissa.val == f._mask
        return self._mantissa.val != 0

    @property
    def is_inf(self) -> bool:
        return (self._format.has_inf and self._exp.val == self._format._top
                and self._mantissa.val == 0)

    @property
    def is_finite(self) -> bool:
        return not (self.is_nan or self.is_inf)

    @property
    def is_snan(self) -> bool:
        """Signaling NaN: IEEE formats, quiet bit (mantissa MSB) clear."""
        return (self.is_nan and self._format.inf_nan is True
                and not self._mantissa.val >> (self.mantissa_bits - 1))

    @property
    def is_zero(self) -> bool:
        return self.is_finite and self.exact == 0

    @property
    def is_subnormal(self) -> bool:
        return (self._format.has_zero and self._exp.val == 0
                and self._mantissa.val != 0)

    # Verification data
    @property
    def flags(self) -> FPFlags:
        """Status flags raised by the operation that produced this value."""
        return self._flags

    @property
    def unrounded(self):
        """Infinitely precise result before rounding: a Fraction, +-inf as a
        float, or None (NaN results and values built from bits)."""
        return self._unrounded

    @property
    def exact(self) -> Fraction:
        """The exact value this encoding represents (finite values only)."""
        if not self.is_finite:
            raise ValueError(f"{self!r} is not finite")
        f = self._format
        M, b, m = f.mantissa_bits, self._exp.val, self._mantissa.val
        if b == 0 and f.has_zero:
            if f.ftz:  # denormals-are-zero on input
                return Fraction(0)
            v = m * _pow2(1 - f.bias - M)
        else:
            v = ((1 << M) | m) * _pow2(b - f.bias - M)
        return -v if self._sign else v

    def _key(self):
        """Value for ordering: exact, or +-inf as a float (not for NaN)."""
        if self.is_inf:
            return -math.inf if self._sign else math.inf
        return self.exact

    @property
    def ulp(self) -> Fraction:
        """Spacing of adjacent encodings in this value's binade."""
        f = self._format
        return _pow2(max(self._exp.val, f._min_field) - f.bias - f.mantissa_bits)

    def error_ulps(self, ref=None) -> Fraction:
        """(exact - ref) in ulps of this value; ref defaults to unrounded."""
        if ref is None:
            ref = self._unrounded
            if ref is None:
                raise ValueError("no unrounded value; pass ref explicitly")
        if isinstance(ref, FP):
            ref = ref.exact
        return (self.exact - Fraction(ref)) / self.ulp

    def _fmt(self) -> FPFormat:
        return self._format

    # Rounding core (finite values)
    @classmethod
    def _overflow(cls, sign: bool, fmt: FPFormat):
        """Result of an overflow, per IEEE 754 7.4: inf or max by rounding
        direction (NaN instead of inf in 'fn' formats)."""
        to_inf = {Rounding.RNE: True, Rounding.RNA: True, Rounding.SR: True,
                  Rounding.RTZ: False, Rounding.RUP: not sign,
                  Rounding.RDN: sign}[fmt.rounding]
        if not fmt.has_nan or fmt.saturate or not to_inf:
            return cls._make(sign, fmt._max_field, fmt._max_mant, fmt), "saturated"
        if fmt.has_inf:
            return cls._make(sign, fmt._top, 0, fmt), "overflowed"
        return fmt.nan(sign), "overflowed"

    @classmethod
    def _no_zero(cls, fmt: FPFormat):
        """A zero (or negative-into-unsigned) result in a format without zero."""
        if not fmt.has_nan:
            raise ValueError(f"{fmt} has no zero and no NaN to return instead")
        return fmt.nan(), _F.INVALID, None

    @classmethod
    def _round(cls, x: Fraction, fmt: FPFormat, zero_sign: bool = False,
               sr: int | None = None):
        """Round finite x into fmt. Returns (result, flags, event)."""
        M, bias = fmt.mantissa_bits, fmt.bias
        emin, emax = fmt.emin, fmt.emax
        if x == 0:
            if not fmt.has_zero:
                return cls._no_zero(fmt)
            return cls._make(zero_sign and fmt.signed, 0, 0, fmt), _F(0), None
        if x < 0 and not fmt.signed:
            if not fmt.has_zero:
                return cls._no_zero(fmt)
            # Unsigned formats clamp negative results to zero.
            return cls._make(False, 0, 0, fmt), _F.UNDERFLOW | _F.INEXACT, "clamped"

        if fmt.rounding is Rounding.SR and sr is None:
            sr = _sr_state["source"](fmt.sr_bits)
        sr = sr or 0
        sign = x < 0
        n, d = abs(x).numerator, abs(x).denominator
        e = _floor_log2(n, d)

        def rnd(exp):
            # Zero-width mantissa: ties-to-even uses the code's parity.
            odd = None
            if M == 0:
                trunc = (n << max(0, -exp)) // (d << max(0, exp))
                odd = bool(trunc) and bool((exp + bias) & 1)
            s, inexact = _round_sig(n, d, exp - M, sign, fmt.rounding, odd, sr,
                                    fmt.sr_bits)
            if s >> (M + 1):
                return s >> 1, exp + 1, inexact
            return s, exp, inexact

        # Tininess: before rounding, or after rounding to M bits with an
        # unbounded exponent (IEEE 754-2019 7.5).
        e_after = rnd(e)[1]
        tiny = (e if fmt.tininess == "before" else e_after) < emin

        gradual = fmt.has_zero and not (fmt.wrap or fmt.ftz)
        sig, e, inexact = rnd(max(e, emin) if gradual else e)

        if fmt.ftz and tiny:
            return cls._make(sign, 0, 0, fmt), _F.UNDERFLOW | _F.INEXACT, "flushed"
        # In 'fn' formats the all-ones code of the top binade is NaN.
        over = e > emax or (e == emax and (sig & fmt._mask) > fmt._max_mant)
        if over:
            if fmt.wrap:
                result = cls._make(sign, e + bias, sig, fmt)  # UINT wraps field
                return result, _F.OVERFLOW | _F.INEXACT, "wrapped-up"
            result, event = cls._overflow(sign, fmt)
            return result, _F.OVERFLOW | _F.INEXACT, event
        if e < emin:
            if fmt.wrap:
                result = cls._make(sign, e + bias, sig, fmt)
                return result, _F.UNDERFLOW | _F.INEXACT, "wrapped-down"
            # No zero and no subnormals: the smallest value is the floor.
            result = cls._make(sign, fmt._min_field, 0, fmt)
            return result, _F.UNDERFLOW | _F.INEXACT, "underflowed"
        field = e + bias if (sig >> M or not fmt.has_zero) else 0
        result = cls._make(sign, field, sig, fmt)
        flags, event = _F(0), None
        if inexact:
            flags |= _F.INEXACT
            if tiny and (gradual or not fmt.has_zero):
                flags |= _F.UNDERFLOW
                event = "underflowed"
        return result, flags, event

    @classmethod
    def _finish(cls, result: "FP", flags: FPFlags, event, unrounded, fmt) -> "FP":
        result._flags = flags
        result._unrounded = unrounded
        if _should_warn(event, fmt):
            _report(event, unrounded, fmt, result)
        return result

    @classmethod
    def _encode(cls, x: Fraction, fmt: FPFormat, zero_sign: bool = False,
                sr: int | None = None) -> "FP":
        result, flags, event = cls._round(x, fmt, zero_sign, sr)
        return cls._finish(result, flags, event, x, fmt)

    # Special values (inf / NaN)
    def _nan_to(self, fmt: FPFormat):
        """This NaN re-encoded in fmt: canonical, or payload kept and quieted."""
        invalid = _F.INVALID if self.is_snan else _F(0)
        if not fmt.has_nan:
            raise ValueError(f"NaN is not representable in {fmt}")
        keep = fmt.nan_mode is not NaNMode.CANONICAL
        if not keep or fmt.inf_nan == "fn":
            return fmt.nan(self._sign and keep), invalid
        payload = (self._mantissa.val if self.inf_nan is True and self.mantissa_bits
                   else 1 << max(self.mantissa_bits - 1, 0))
        shift = fmt.mantissa_bits - max(self.mantissa_bits, 1)
        payload = payload << shift if shift >= 0 else payload >> -shift
        payload |= 1 << (fmt.mantissa_bits - 1)  # quiet it
        return fmt.nan(self._sign, payload), invalid

    @classmethod
    def _default_nan(cls, fmt: FPFormat) -> "FP":
        return fmt.nan(fmt.nan_mode is NaNMode.X86)

    @classmethod
    def _nan_result(cls, fmt: FPFormat, *operands: "FP", invalid: bool = False):
        """NaN produced by an operation. Returns (result, flags)."""
        flags = _F.INVALID if invalid or any(o.is_snan for o in operands) else _F(0)
        if not fmt.has_nan:
            raise ValueError(f"invalid operation: NaN is not representable in {fmt}")
        nans = [o for o in operands if o.is_nan]
        if fmt.nan_mode is not NaNMode.CANONICAL and nans:
            src = nans[0]
            if fmt.nan_mode is NaNMode.ARM:
                src = next((o for o in nans if o.is_snan), src)
            return src._nan_to(fmt)[0], flags
        return cls._default_nan(fmt), flags

    @classmethod
    def _inf_result(cls, sign: bool, fmt: FPFormat):
        """An exact infinity delivered into fmt. Returns (result, flags, event)."""
        if sign and not fmt.signed:
            if not fmt.has_zero:
                return cls._no_zero(fmt)
            return cls._make(False, 0, 0, fmt), _F.UNDERFLOW | _F.INEXACT, "clamped"
        if fmt.has_inf and not fmt.saturate:
            return cls._make(sign, fmt._top, 0, fmt), _F(0), None
        if fmt.has_nan and not fmt.saturate:  # 'fn': no infinity -> NaN
            return fmt.nan(sign), _F.INVALID, None
        return fmt.max_value(sign), _F.OVERFLOW | _F.INEXACT, "saturated"

    @classmethod
    def _special_in(cls, value: float, fmt: FPFormat):
        """Python float inf/NaN into fmt: (result, flags, event, unrounded)."""
        if math.isnan(value):
            if not fmt.has_nan:
                raise ValueError(f"NaN is not representable in {fmt}")
            keep = fmt.nan_mode is not NaNMode.CANONICAL
            return (fmt.nan(math.copysign(1, value) < 0 and keep), _F(0), None, None)
        return (*cls._inf_result(value < 0, fmt), value)

    def _convert_to(self, fmt: FPFormat, sr: int | None = None) -> "FP":
        if self.is_nan:
            result, flags = self._nan_to(fmt)
            return FP._finish(result, flags, None, None, fmt)
        if self.is_inf:
            v = -math.inf if self._sign else math.inf
            return FP._finish(*FP._inf_result(self._sign, fmt), v, fmt)
        return FP._encode(self.exact, fmt, self._sign, sr)

    def convert(self, exp_bits: int | FPFormat = 2, mantissa_bits: int = 1,
                bias: int | None = None, signed: bool = True, *,
                fmt: FPFormat | None = None, sr_rand: int | None = None,
                **modes) -> "FP":
        return self._convert_to(_resolve(exp_bits, mantissa_bits, bias, signed,
                                         fmt, modes), sr_rand)

    # Mixed formats
    @staticmethod
    def _common_fmt(a: "FP", b: "FP") -> FPFormat:
        """Smallest format holding every value of both a's and b's formats.
        Signed if either is; has inf/NaN if either does; the remaining
        modes come from the left operand."""
        return FP._common_of(a._format, b._format)

    @staticmethod
    def _common_of(fa: FPFormat, fb: FPFormat) -> FPFormat:
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

    @staticmethod
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

    def _coerce(self, other):
        """Return (self, other, fmt) with both FPs exact in fmt, or None."""
        if isinstance(other, FP):
            fmt = FP._common_fmt(self, other)
            if self._format != other._format:
                FP._warn_mismatch(self._format, other._format, fmt)
            return self._widen(fmt), other._widen(fmt), fmt
        if isinstance(other, (int, float, Fraction)) and not isinstance(other, bool):
            return self, self._cast_scalar(other), self._format
        return None

    def _coerce3(self, b, c):
        """Common format and widened copies of self, b, c (for fma)."""
        b = b if isinstance(b, FP) else self._cast_scalar(b)
        c = c if isinstance(c, FP) else self._cast_scalar(c)
        f_ab = FP._common_of(self._format, b._format)
        if self._format != b._format:
            FP._warn_mismatch(self._format, b._format, f_ab)
        fmt = FP._common_of(f_ab, c._format)
        if f_ab != c._format:
            FP._warn_mismatch(f_ab, c._format, fmt)
        return self._widen(fmt), b._widen(fmt), c._widen(fmt), fmt

    def _cast_scalar(self, other) -> "FP":
        """Implicitly cast a Python number into this value's format. Warns
        with CastWarning only when the cast changes the value."""
        fmt = self._format
        if isinstance(other, float) and not math.isfinite(other):
            target = fmt if fmt.signed or not other < 0 else fmt.replace(signed=True)
            b, flags, _, _ = FP._special_in(other, target)
        else:
            zero_sign = isinstance(other, float) and math.copysign(1.0, other) < 0
            x = Fraction(other)
            # A negative literal keeps its sign even against an unsigned
            # format, so that e.g. unsigned - 1 is not clamped to unsigned - 0.
            target = fmt if fmt.signed or x >= 0 else fmt.replace(signed=True)
            if x == 0 and not target.has_zero:
                target = target.replace(has_zero=True)
            b, flags, _ = FP._round(x, target, zero_sign)
        if flags:
            _warn(CastWarning, f"implicit cast of {other!r} to {fmt} is lossy: "
                  f"{float(b)!r} ({flags.name})")
        return b

    def _widen(self, fmt: FPFormat) -> "FP":
        """Re-encode exactly into a (wider) common format, keeping the sign.
        The copy uses gradual underflow and no wrap/ftz, so no operand is
        altered (e.g. a subnormal is not read as 0 because the *left*
        operand has ftz); the op result is still rounded with fmt's modes."""
        if self._format == fmt:
            return self
        plain = fmt.replace(wrap=False, ftz=False, saturate=False, has_zero=True,
                            rounding=Rounding.RNE)
        if self.is_nan:
            # Keep the payload (and signaling state) so the operation itself
            # raises INVALID; widening never shrinks the mantissa.
            if plain.inf_nan == "fn" or self.inf_nan == "fn" or not self.mantissa_bits:
                return plain.nan(self._sign)
            payload = self._mantissa.val << (plain.mantissa_bits - self.mantissa_bits)
            return FP._make(self._sign and plain.signed, plain._top, payload, plain)
        if self.is_inf:
            return FP._make(self._sign, plain._top, 0, plain)
        return FP._round(self.exact, plain, self._sign)[0]

    def _fields(self):
        return self._sign, self._exp.val, self._mantissa.val

    # Arithmetic
    def _binop(self, other, op, reverse=False):
        c = self._coerce(other)
        if c is None:
            return NotImplemented
        a, b, fmt = c
        if reverse:
            a, b = b, a
        return FP._arith(op, a, b, fmt)

    @staticmethod
    def _sum_zero_sign(sa: bool, sb: bool, fmt: FPFormat) -> bool:
        """IEEE 754 6.3: an exact zero sum of opposite-signed operands is +0,
        except -0 when rounding toward negative."""
        return sa if sa == sb else fmt.rounding is Rounding.RDN

    @staticmethod
    def _arith(op: str, a: "FP", b: "FP", fmt: FPFormat) -> "FP":
        """a op b rounded into fmt, following IEEE 754 for inf/NaN operands."""
        if a.is_nan or b.is_nan:
            return FP._finish(*FP._nan_result(fmt, a, b), None, None, fmt)
        sa, sb = a.sign, b.sign
        if op in "+-":
            sb ^= op == "-"
            if a.is_inf and b.is_inf and sa != sb:     # inf - inf
                return FP._finish(*FP._nan_result(fmt, invalid=True), None, None, fmt)
            if a.is_inf or b.is_inf:
                s = sa if a.is_inf else sb
                return FP._finish(*FP._inf_result(s, fmt), -math.inf if s else math.inf, fmt)
            x = a.exact + b.exact if op == "+" else a.exact - b.exact
            return FP._encode(x, fmt, FP._sum_zero_sign(sa, sb, fmt))
        s = sa != sb
        if op == "*":
            if a.is_inf or b.is_inf:
                if a.is_zero or b.is_zero:             # 0 * inf
                    return FP._finish(*FP._nan_result(fmt, invalid=True), None, None, fmt)
                return FP._finish(*FP._inf_result(s, fmt), -math.inf if s else math.inf, fmt)
            return FP._encode(a.exact * b.exact, fmt, s)
        # division
        if a.is_inf and b.is_inf:                      # inf / inf
            return FP._finish(*FP._nan_result(fmt, invalid=True), None, None, fmt)
        if a.is_inf:
            return FP._finish(*FP._inf_result(s, fmt), -math.inf if s else math.inf, fmt)
        if b.is_inf:
            return FP._encode(Fraction(0), fmt, s)
        if b.is_zero:
            if not fmt.has_nan:
                raise ZeroDivisionError(f"FP division by zero in {fmt}")
            if a.is_zero:                              # 0 / 0
                return FP._finish(*FP._nan_result(fmt, invalid=True), None, None, fmt)
            r, flags, event = FP._inf_result(s, fmt)
            return FP._finish(r, flags | _F.DIVZERO, event, -math.inf if s else math.inf, fmt)
        return FP._encode(a.exact / b.exact, fmt, s)

    def __add__(self, o):       return self._binop(o, "+")
    def __radd__(self, o):      return self._binop(o, "+", True)
    def __sub__(self, o):       return self._binop(o, "-")
    def __rsub__(self, o):      return self._binop(o, "-", True)
    def __mul__(self, o):       return self._binop(o, "*")
    def __rmul__(self, o):      return self._binop(o, "*", True)
    def __truediv__(self, o):   return self._binop(o, "/")
    def __rtruediv__(self, o):  return self._binop(o, "/", True)

    def fma(self, b, c) -> "FP":
        """Fused multiply-add: self * b + c with a single rounding."""
        a, b, c, fmt = self._coerce3(b, c)
        fin = lambda r: FP._finish(*r, None, None, fmt)
        if a.is_nan or b.is_nan:
            if fmt.nan_mode is NaNMode.ARM and c.is_snan:
                # ARM picks among (a, b) first, then against c, so a
                # signaling c wins over the (by then quiet) a/b NaN.
                return fin((c._nan_to(fmt)[0], _F.INVALID))
            return fin(FP._nan_result(fmt, a, b, c))
        s = a.sign != b.sign
        if (a.is_inf and b.is_zero) or (a.is_zero and b.is_inf):
            # IEEE 754-2019 7.2: 0 * inf + c is invalid; when c is a quiet
            # NaN, signaling is implementation-defined. RISC-V and ARM signal
            # (ARM returns a signaling c, quieted); x86 hardware does not and
            # returns c.
            if fmt.nan_mode is NaNMode.X86 and c.is_nan and not c.is_snan:
                return fin((c._nan_to(fmt)[0], _F(0)))
            if fmt.nan_mode is NaNMode.ARM and c.is_snan:
                return fin((c._nan_to(fmt)[0], _F.INVALID))
            return fin((FP._default_nan(fmt), _F.INVALID))
        if c.is_nan:
            return fin(FP._nan_result(fmt, c))
        if a.is_inf or b.is_inf:
            if c.is_inf and c.sign != s:
                return fin(FP._nan_result(fmt, invalid=True))
            return FP._finish(*FP._inf_result(s, fmt), -math.inf if s else math.inf, fmt)
        if c.is_inf:
            return FP._finish(*FP._inf_result(c.sign, fmt),
                              -math.inf if c.sign else math.inf, fmt)
        x = a.exact * b.exact + c.exact
        return FP._encode(x, fmt, FP._sum_zero_sign(s, c.sign, fmt))

    def sqrt(self) -> "FP":
        """Square root, correctly rounded (IEEE 754 5.4.1)."""
        fmt = self._format
        if self.is_nan:
            return FP._finish(*FP._nan_result(fmt, self), None, None, fmt)
        if self.is_zero:     # sqrt(-0) = -0; a DAZ subnormal reads as zero
            return FP._make(self._sign, 0, 0, fmt)
        if self._sign:
            return FP._finish(*FP._nan_result(fmt, invalid=True), None, None, fmt)
        if self.is_inf:
            return FP._make(*self._fields(), fmt)
        x = self.exact
        n, d = x.numerator, x.denominator
        # Enough bits below the result's last place that a sticky half-bit
        # decides rounding exactly (as hardware does with a sticky bit).
        e_s = _floor_log2(n, d) // 2
        p = max(0, fmt.mantissa_bits + fmt.sr_bits + 4 - min(e_s, fmt.emin))
        scaled, rem = divmod(n << (2 * p), d)
        r = math.isqrt(scaled)
        exact = rem == 0 and r * r == scaled
        y = Fraction(r, 1 << p) if exact else Fraction(2 * r + 1, 1 << (p + 1))
        result, flags, event = FP._round(y, fmt)
        if not exact:
            flags |= _F.INEXACT
        return FP._finish(result, flags, event, y if exact else None, fmt)

    # IEEE 754-2019 9.6: minimum/maximum and minimumNumber/maximumNumber
    def _minmax(self, other, pick_max: bool, number: bool) -> "FP":
        c = self._coerce(other)
        if c is None:
            raise TypeError(f"cannot compare FP with {type(other).__name__}")
        a, b, fmt = c
        snan = a.is_snan or b.is_snan
        if a.is_nan or b.is_nan:
            if number and not (a.is_nan and b.is_nan):
                r = b if a.is_nan else a
                flags = _F.INVALID if snan else _F(0)
                out = FP._make(r.sign, 0, 0, fmt) if r.is_zero else FP._make(*r._fields(), fmt)
                return FP._finish(out, flags, None, None, fmt)
            return FP._finish(*FP._nan_result(fmt, a, b), None, None, fmt)
        ka, kb = a._key(), b._key()
        if ka == kb:  # -0 is less than +0
            pick_a = (a.sign and not b.sign) != pick_max
        else:
            pick_a = (ka > kb) == pick_max
        r = a if pick_a else b
        if r.is_zero:        # a DAZ subnormal reads as zero
            return FP._make(r.sign, 0, 0, fmt)
        return FP._make(*r._fields(), fmt)

    def minimum(self, other) -> "FP":
        return self._minmax(other, False, False)

    def maximum(self, other) -> "FP":
        return self._minmax(other, True, False)

    def minimum_number(self, other) -> "FP":
        """minimumNumber: a NaN operand is ignored (RISC-V fmin)."""
        return self._minmax(other, False, True)

    def maximum_number(self, other) -> "FP":
        """maximumNumber: a NaN operand is ignored (RISC-V fmax)."""
        return self._minmax(other, True, True)

    # Integer conversions and rounding to integral
    _INT_INVALID = {
        # Integer returned for (NaN, positive overflow, negative overflow)
        NaNMode.CANONICAL: ("max", "max", "min"),   # RISC-V fcvt
        NaNMode.PROPAGATE: ("max", "max", "min"),
        NaNMode.X86: ("x86", "x86", "x86"),          # "integer indefinite"
        NaNMode.ARM: ("zero", "max", "min"),         # ARM VCVT
    }

    def to_int(self, bits: int = 32, signed: bool = True,
               rounding: Rounding | None = None, exact: bool = True):
        """Convert to a ``bits``-wide INT/UINT. Returns (value, flags).

        Out-of-range, infinite and NaN inputs raise INVALID and return the
        integer the format's nan_mode selects (RISC-V saturates, x86 returns
        the "integer indefinite", ARM saturates and maps NaN to 0).
        ``exact`` raises INEXACT when rounding changed the value."""
        rounding = rounding or self._format.rounding
        cls = INT if signed else UINT
        lo = -(1 << (bits - 1)) if signed else 0
        hi = (1 << (bits - (1 if signed else 0))) - 1

        def invalid(kind):
            which = self._INT_INVALID[self._format.nan_mode][kind]
            v = {"max": hi, "min": lo, "zero": 0,
                 "x86": lo if signed else (1 << bits) - 1}[which]
            return cls(v, bits), _F.INVALID

        if self.is_nan:
            return invalid(0)
        if self.is_inf:
            return invalid(2 if self._sign else 1)
        sr = _sr_state["source"](self.sr_bits) if rounding is Rounding.SR else 0
        q, inexact = _round_to_int(self.exact, rounding, sr, self.sr_bits)
        if q > hi:
            return invalid(1)
        if q < lo:
            return invalid(2)
        return cls(q, bits), (_F.INEXACT if inexact and exact else _F(0))

    def round_to_integral(self, rounding: Rounding | None = None,
                          exact: bool = False) -> "FP":
        """Round to an integral value in the same format (IEEE 754
        roundToIntegral; ``exact=True`` is roundToIntegralExact)."""
        fmt = self._format
        if self.is_nan:
            return FP._finish(*FP._nan_result(fmt, self), None, None, fmt)
        if self.is_inf:
            return FP._make(*self._fields(), fmt)
        if self.is_zero:     # a DAZ subnormal reads as zero
            return FP._make(self._sign, 0, 0, fmt)
        rounding = rounding or fmt.rounding
        sr = _sr_state["source"](fmt.sr_bits) if rounding is Rounding.SR else 0
        q, inexact = _round_to_int(self.exact, rounding, sr, fmt.sr_bits)
        # The integer is representable: it has no more significant bits
        # than the input (or it is 0 or a power of two).
        if q == 0 and not fmt.has_zero:
            return FP._finish(*FP._no_zero(fmt), None, fmt)
        result = FP._round(Fraction(q), fmt.replace(rounding=Rounding.RNE,
                                                    wrap=False, ftz=False),
                           self._sign)[0]
        result = FP._make(*result._fields(), fmt)
        flags = _F.INEXACT if inexact and exact else _F(0)
        return FP._finish(result, flags, None, Fraction(q), fmt)

    # Comparisons with IEEE flags
    def compare(self, other, signaling: bool = False):
        """IEEE comparison. Returns (relation, flags) with relation one of
        'lt', 'eq', 'gt', 'unordered'. Quiet comparisons raise INVALID only
        for signaling NaNs; signaling ones for any NaN."""
        c = self._coerce(other)
        if c is None:
            raise TypeError(f"cannot compare FP with {type(other).__name__}")
        a, b, _ = c
        if a.is_nan or b.is_nan:
            invalid = signaling or a.is_snan or b.is_snan
            return "unordered", (_F.INVALID if invalid else _F(0))
        ka, kb = a._key(), b._key()
        return ("lt" if ka < kb else "gt" if ka > kb else "eq"), _F(0)

    def eq(self, other, signaling: bool = False):
        rel, flags = self.compare(other, signaling)
        return rel == "eq", flags

    def lt(self, other, signaling: bool = True):
        rel, flags = self.compare(other, signaling)
        return rel == "lt", flags

    def le(self, other, signaling: bool = True):
        rel, flags = self.compare(other, signaling)
        return rel in ("lt", "eq"), flags

    def __neg__(self):
        if not self.signed:
            if self.is_nan:
                return FP._make(*self._fields(), self._format)
            if self.is_inf:
                return FP._finish(*FP._inf_result(True, self._format), -math.inf,
                                  self._format)
            return FP._encode(-self.exact, self._format)
        return FP._make(not self._sign, self._exp.val, self._mantissa.val,
                        self._format)

    def __pos__(self):
        return FP._make(*self._fields(), self._format)

    def __abs__(self):
        return FP._make(False, self._exp.val, self._mantissa.val, self._format)

    # Logical
    def _bitop(self, other, fn, reverse=False):
        c = self._coerce(other)
        if c is None:
            return NotImplemented
        a, b, fmt = c
        if reverse:
            a, b = b, a
        return FP._from_raw_fmt(fn(a.raw, b.raw), fmt)

    def __and__(self, o):   return self._bitop(o, lambda a, b: a & b)
    def __rand__(self, o):  return self._bitop(o, lambda a, b: a & b, True)
    def __or__(self, o):    return self._bitop(o, lambda a, b: a | b)
    def __ror__(self, o):   return self._bitop(o, lambda a, b: a | b, True)
    def __xor__(self, o):   return self._bitop(o, lambda a, b: a ^ b)
    def __rxor__(self, o):  return self._bitop(o, lambda a, b: a ^ b, True)
    def __invert__(self):
        return FP._from_raw_fmt(~self.raw & ((1 << self.size) - 1), self._format)

    # Comparison operators (IEEE: NaN is unordered, so only != is true)
    def _cmp(self, other, fn, if_nan=False):
        c = self._coerce(other)
        if c is None:
            return NotImplemented
        a, b, _ = c
        if a.is_nan or b.is_nan:
            return if_nan
        return fn(a._key(), b._key())

    def __eq__(self, o): return self._cmp(o, lambda a, b: a == b)
    def __ne__(self, o): return self._cmp(o, lambda a, b: a != b, True)
    def __lt__(self, o): return self._cmp(o, lambda a, b: a < b)
    def __le__(self, o): return self._cmp(o, lambda a, b: a <= b)
    def __gt__(self, o): return self._cmp(o, lambda a, b: a > b)
    def __ge__(self, o): return self._cmp(o, lambda a, b: a >= b)

    def __hash__(self):
        if self.is_nan:
            return object.__hash__(self)
        return hash(self._key())

    # Casting
    def __float__(self):
        if self.is_nan:
            return math.copysign(math.nan, -1.0 if self._sign else 1.0)
        if self.is_inf:
            return -math.inf if self._sign else math.inf
        if self.is_zero:
            return -0.0 if self._sign else 0.0
        try:
            return float(self.exact)
        except OverflowError:
            return -math.inf if self._sign else math.inf

    def __int__(self):
        if self.is_nan:
            raise ValueError("cannot convert NaN to integer")
        if self.is_inf:
            raise OverflowError("cannot convert infinity to integer")
        return int(self.exact)

    def __bool__(self):
        return not self.is_zero

    def to_bin(self, sep: str = " ") -> str:
        fields = [self._exp.to_bin(), self._mantissa.to_bin()]
        if self.signed:
            fields.insert(0, str(int(self._sign)))
        return sep.join(f for f in fields if f)

    def to_hex(self) -> str:
        return format(self.raw, f"0{(self.size + 3) // 4}x")

    def __repr__(self):
        extra = f", flags={self._flags.name}" if self._flags else ""
        return f"FP({float(self)!r}, {self._format}{extra})"

    def __str__(self):
        return str(float(self))
