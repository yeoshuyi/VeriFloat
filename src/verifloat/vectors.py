"""Test-vector export: (operands, result, flags) files for any FP format.

``generate`` runs the model on stimuli (random, edge cases, biased mixes or an
exhaustive sweep) and yields ``Vector`` records; ``write`` and ``read`` store
them as TestFloat text, CSV, JSON lines or ``$readmemh`` words; ``verify``
re-computes a file with the model and reports every difference. Every
expected value comes from the model itself, so a file is exactly as right as
the model, and ``verify`` against real ``testfloat_gen`` output is the check
on that. Also a CLI: ``python -m verifloat.vectors --help``.

Vectors the model cannot answer (it raises, e.g. ``x/0`` in a format without
NaN) are skipped and counted in ``VectorStream.stats``; they never appear in
a file. Stochastic rounding is supported by exporting the random bits as an
``sr`` field (after the operands) so a file replays without any shared random
source; the model's global SR source is swapped out during each call and
restored afterwards (not thread-safe).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import operator
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, NamedTuple

from . import __version__
from . import fp as _fp
from .fp import (BF16, E2M1, E2M3, E3M2, E4M3, E5M2, FP16, FP32, FP64, UE4M3, UE8M0,
                 FPFlags, FPFormat, Rounding, set_sr_source)

__all__ = ["FORMATS", "OPS", "STYLES", "Mismatch", "Stats", "Vector", "VectorFormatError", "VectorSpec",
           "VectorStream", "VerifyReport", "detect_format", "edge_codes", "generate", "main", "parse_format",
           "read", "read_meta", "systemverilog", "verify", "write"]

FORMATS = ("testfloat", "csv", "jsonl", "readmemh")
STYLES = ("random", "edges", "mixed", "exhaustive")
_META_VERSION = 1


class VectorFormatError(ValueError):
    """A vector file is malformed (wrong field count, bad hex, value too wide)."""


class Vector(NamedTuple):
    """One test vector: raw operand codes, raw result code and flag bits.
    ``sr`` holds the stochastic-rounding random bits when the spec needs them.
    Comparison results are 0/1 (``compare``: lt=0, eq=1, gt=2, unordered=3)."""
    operands: tuple[int, ...]
    result: int
    flags: int
    sr: int | None = None


# --------------------------------------------------------------- operations
_BINARY: dict[str, Callable] = {
    "add": operator.add, "sub": operator.sub, "mul": operator.mul, "div": operator.truediv,
    "minimum": lambda a, b: a.minimum(b), "maximum": lambda a, b: a.maximum(b),
    "minimum_number": lambda a, b: a.minimum_number(b),
    "maximum_number": lambda a, b: a.maximum_number(b),
    "remainder": lambda a, b: a.remainder(b), "fmod": lambda a, b: a.fmod(b),
    "copysign": lambda a, b: a.copysign(b),
}
_UNARY: dict[str, Callable] = {
    "sqrt": lambda a: a.sqrt(), "next_up": lambda a: a.next_up(), "next_down": lambda a: a.next_down(),
    "logb": lambda a: a.logb(),
}
# name -> (method, signaling). Same names and defaults as TestFloat.
_CMP = {"eq": ("eq", False), "eq_signaling": ("eq", True), "lt": ("lt", True),
        "lt_quiet": ("lt", False), "le": ("le", True), "le_quiet": ("le", False)}
# Operations that return an operand unrounded, so stochastic rounding is moot.
_NO_ROUNDING = {"minimum", "maximum", "minimum_number", "maximum_number", "copysign", "next_up", "next_down"}
_RELATION = {"lt": 0, "eq": 1, "gt": 2, "unordered": 3}
_ALIASES = {"min": "minimum", "max": "maximum", "minnum": "minimum_number",
            "maxnum": "maximum_number", "mulAdd": "fma", "roundToInt": "round_to_integral",
            "rem": "remainder", "sgnj": "copysign", "fsgnj": "copysign"}
_KIND = {**dict.fromkeys(_BINARY, "binary"), "fma": "ternary", **dict.fromkeys(_UNARY, "unary"),
         **dict.fromkeys(_CMP, "cmp"), "compare": "compare", "compare_signaling": "compare",
         "convert": "convert", "round_to_integral": "rti", "to_int": "to_int",
         "from_int": "from_int"}
OPS = tuple(_KIND)


def _hexw(width: int) -> int:
    return (width + 3) // 4


@dataclass(frozen=True)
class VectorSpec:
    """What a vector file contains: operation, formats and operation options.

    fmt        operand format (``from_int``: the result format).
    to         ``convert`` only: the result format.
    bits, signed
               ``to_int`` / ``from_int`` only: the integer's width (default
               32) and signedness (default True). Integer fields are the
               two's-complement bit pattern.
    rounding, exact
               ``to_int`` / ``round_to_integral`` only (the other operations
               round as ``fmt.rounding`` says). ``rounding`` defaults to
               ``fmt.rounding`` and may not be SR; ``exact`` defaults to the model's
               own default: True for ``to_int``, False for ``round_to_integral``.

    Options that an operation does not use raise TypeError, so a typo is not
    silently ignored. Names accept TestFloat's spellings (``mulAdd``,
    ``roundToInt``) and ``min``, ``max``, ``minnum``, ``maxnum``.
    """
    op: str
    fmt: FPFormat
    to: FPFormat | None = None
    bits: int | None = None
    signed: bool | None = None
    rounding: Rounding | None = None
    exact: bool | None = None

    def __post_init__(self):
        op = _ALIASES.get(self.op, self.op)
        if op not in _KIND:
            raise ValueError(f"unknown operation {self.op!r}; choose from {', '.join(OPS)}")
        object.__setattr__(self, "op", op)
        kind = _KIND[op]
        if not isinstance(self.fmt, FPFormat):
            raise TypeError("fmt must be an FPFormat (see parse_format for names)")

        def only(names, owners):
            for name in names:
                if getattr(self, name) is not None and kind not in owners:
                    raise TypeError(f"{name} does not apply to {op}")
        only(["to"], {"convert"})
        only(["bits", "signed"], {"to_int", "from_int"})
        only(["rounding", "exact"], {"to_int", "rti"})
        if kind == "convert":
            if not isinstance(self.to, FPFormat):
                raise TypeError("convert needs a target format: to=<FPFormat>")
        if kind in ("to_int", "from_int"):
            bits = 32 if self.bits is None else self.bits
            if not isinstance(bits, int) or isinstance(bits, bool) or bits < 1:
                raise ValueError("bits must be a positive int")
            object.__setattr__(self, "bits", bits)
            object.__setattr__(self, "signed", True if self.signed is None else bool(self.signed))
        if kind in ("to_int", "rti"):
            rounding = self.fmt.rounding if self.rounding is None else self.rounding
            if not isinstance(rounding, Rounding):
                raise TypeError("rounding must be a Rounding member")
            if rounding is Rounding.SR:
                raise ValueError(f"{op} with stochastic rounding is not exportable (no random "
                                 "bits are exported for it); pass rounding=<deterministic mode>")
            object.__setattr__(self, "rounding", rounding)
            object.__setattr__(self, "exact", kind == "to_int" if self.exact is None else bool(self.exact))

    # -- layout
    @property
    def kind(self) -> str:
        return _KIND[self.op]

    @property
    def arity(self) -> int:
        return {"ternary": 3, "binary": 2, "cmp": 2, "compare": 2}.get(self.kind, 1)

    @property
    def result_format(self) -> FPFormat | None:
        """Format of the result, or None when it is an integer or boolean."""
        if self.kind in ("cmp", "compare", "to_int"):
            return None
        return self.to if self.kind == "convert" else self.fmt

    @property
    def uses_sr(self) -> bool:
        """Whether the result is rounded by stochastic rounding (an ``sr`` field exists)."""
        rf = self.result_format
        return (rf is not None and rf.rounding is Rounding.SR
                and self.kind != "rti" and self.op not in _NO_ROUNDING)

    @property
    def operand_widths(self) -> tuple[int, ...]:
        w = self.bits if self.kind == "from_int" else self.fmt.size
        return (w,) * self.arity

    @property
    def result_width(self) -> int:
        if self.kind == "cmp":
            return 1
        if self.kind == "compare":
            return 2
        if self.kind == "to_int":
            return self.bits
        return self.result_format.size

    @property
    def fields(self) -> tuple[tuple[str, int], ...]:
        """(name, width in bits) of every field, in file order."""
        names = "abc"[:self.arity]
        out = [(n, w) for n, w in zip(names, self.operand_widths)]
        if self.uses_sr:
            out.append(("sr", self.result_format.sr_bits))
        return (*out, ("result", self.result_width), ("flags", 5))

    @property
    def word_width(self) -> int:
        return sum(w for _, w in self.fields)

    # -- model
    def compute(self, operands: Iterable[int], sr: int | None = None) -> Vector:
        """Run the model on raw operand codes. May raise what the model raises."""
        return self._compute(tuple(operands), sr)[0]

    def _compute(self, operands: tuple[int, ...], sr: int | None) -> tuple[Vector, bool]:
        if len(operands) != self.arity:
            raise ValueError(f"{self.op} takes {self.arity} operands, got {len(operands)}")
        for x, w in zip(operands, self.operand_widths):
            if not 0 <= x < 1 << w:
                raise ValueError(f"operand {x:#x} does not fit in {w} bits")
        if self.uses_sr:
            if sr is None or not 0 <= sr < 1 << self.result_format.sr_bits:
                raise ValueError("this spec needs sr= (sr_bits random bits)")
        elif sr is not None:
            raise ValueError("this spec does not use stochastic rounding")
        saved = _fp._sr_state["source"]
        if sr is not None:
            set_sr_source(lambda nbits: sr & ((1 << nbits) - 1))
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result, flags = self._model(operands)
        finally:
            set_sr_source(saved)
        return Vector(operands, result, flags, sr), bool(caught)

    def _model(self, operands: tuple[int, ...]) -> tuple[int, int]:
        f, op, kind = self.fmt, self.op, self.kind
        if kind == "from_int":
            v = operands[0]
            if self.signed and v >> (self.bits - 1):
                v -= 1 << self.bits
            r = f(v)
            return r.raw, int(r.flags)
        xs = [f.from_raw(x) for x in operands]
        if kind in ("binary", "ternary", "unary"):
            if kind == "binary":
                r = _BINARY[op](*xs)
            else:
                r = xs[0].fma(xs[1], xs[2]) if kind == "ternary" else _UNARY[op](xs[0])
            return r.raw, int(r.flags)
        if kind == "cmp":
            method, sig = _CMP[op]
            v, fl = getattr(xs[0], method)(xs[1], signaling=sig)
            return int(v), int(fl)
        if kind == "compare":
            rel, fl = xs[0].compare(xs[1], signaling=op == "compare_signaling")
            return _RELATION[rel], int(fl)
        if kind == "convert":
            r = xs[0].convert(self.to)
        elif kind == "rti":
            r = xs[0].round_to_integral(self.rounding, self.exact)
        else:
            v, fl = xs[0].to_int(self.bits, self.signed, self.rounding, self.exact)
            return int(v) & ((1 << self.bits) - 1), int(fl)
        return r.raw, int(r.flags)

    # -- description
    def to_meta(self) -> dict:
        d: dict = {"op": self.op, "fmt": str(self.fmt)}
        if self.to is not None:
            d["to"] = str(self.to)
        if self.bits is not None:
            d["bits"], d["signed"] = self.bits, self.signed
        if self.rounding is not None:
            d["rounding"], d["exact"] = self.rounding.value, self.exact
        return d

    @classmethod
    def from_meta(cls, meta: dict) -> "VectorSpec":
        return cls(meta["op"], FPFormat.parse(meta["fmt"]),
                   to=FPFormat.parse(meta["to"]) if "to" in meta else None,
                   bits=meta.get("bits"), signed=meta.get("signed"),
                   rounding=Rounding(meta["rounding"]) if "rounding" in meta else None,
                   exact=meta.get("exact"))

    def __str__(self) -> str:
        extra = ""
        if self.to is not None:
            extra += f" -> {self.to}"
        if self.bits is not None:
            extra += f" {'i' if self.signed else 'u'}{self.bits}"
        if self.rounding is not None:
            extra += f" {self.rounding.value}{' exact' if self.exact else ''}"
        return f"{self.op}({self.fmt}){extra}"




# --------------------------------------------------------------- stimuli
class _Rng:
    """random.Random restricted to getrandbits, so the stream does not depend
    on how a Python release implements choice() and friends."""

    def __init__(self, seed: int):
        self._r = random.Random(seed)

    def bits(self, k: int) -> int:
        return self._r.getrandbits(k) if k > 0 else 0

    def below(self, n: int) -> int:
        k = (n - 1).bit_length()
        while True:
            v = self.bits(k)
            if v < n:
                return v

    def choice(self, seq):
        return seq[self.below(len(seq))]


def _specials(fmt: FPFormat) -> list[int]:
    """Magnitude codes of inf and of quiet, signaling and extreme-payload NaNs."""
    if not fmt.has_nan:
        return []
    top, M = fmt._top, fmt.mantissa_bits
    if fmt.inf_nan == "fn":
        return [(top << M) | fmt._mask]
    quiet = 1 << (M - 1)
    payloads = {0, 1, quiet, quiet | 1, fmt._mask, quiet - 1, fmt._mask ^ 1}
    return sorted((top << M) | m for m in payloads if 0 <= m <= fmt._mask)


def _edge_mags(fmt: FPFormat) -> list[int]:
    M, bias = fmt.mantissa_bits, fmt.bias
    top_code = (1 << (fmt.exp_bits + M)) - 1
    mags: set[int] = set()
    if fmt.has_zero:
        mags.add(0)
        if M:
            mags.update((1, (1 << M) - 1))     # smallest and largest subnormal
    first_normal = fmt._min_field << M
    mags.update((first_normal, first_normal + 1))
    for field_ in (bias, bias + 1, bias - 1):  # around 1.0, 2.0, 0.5
        if 0 <= field_ <= fmt._max_field:
            code = field_ << M
            mags.update((code, code + 1, code - 1))
            if M:
                mags.add(code | 1 << (M - 1))  # 1.5 x 2^e
    mags.update((fmt._max_code, fmt._max_code - 1))
    mags.update(_specials(fmt))
    return sorted(c for c in mags if 0 <= c <= top_code)


def edge_codes(fmt: FPFormat) -> list[int]:
    """Edge encodings of a format, ascending: zeros (both signs), smallest and
    largest subnormal, smallest normal, values around 0.5, 1, 1.5 and 2, the
    largest finite value, infinities and quiet / signaling NaNs, as far as the
    format has them."""
    mags = _edge_mags(fmt)
    if not fmt.signed:
        return mags
    sign = 1 << (fmt.exp_bits + fmt.mantissa_bits)
    return sorted(mags + [m | sign for m in mags])


def _edge_ints(bits: int, signed: bool, mant: int) -> list[int]:
    """Raw patterns of integers worth converting: 0, +-1, extremes, powers of
    two and their neighbours, and rounding ties for a ``mant``-bit mantissa."""
    vals = {0, 1, 2, 3}
    lo, hi = (-(1 << bits - 1), (1 << bits - 1) - 1) if signed else (0, (1 << bits) - 1)
    vals.update((lo, hi, lo + 1, hi - 1, -1))
    for k in range(1, bits):
        for base in (1 << k, (1 << k) | (1 << (k - mant - 1)) if k - mant - 1 >= 0 else 1 << k):
            for d in (-1, 0, 1):
                vals.update((base + d, -(base + d)))
    return sorted((v & ((1 << bits) - 1)) for v in vals if lo <= v <= hi)


def _unbiased(fmt: FPFormat, mag: int) -> int:
    """Binary exponent of the leading bit of a magnitude code."""
    M = fmt.mantissa_bits
    field_ = mag >> M
    if field_ == 0 and fmt.has_zero:
        mant = mag & fmt._mask
        return fmt.emin - M + max(mant.bit_length(), 1) - 1
    return field_ - fmt.bias


def _code_for_exp(fmt: FPFormat, e: int, rng: _Rng) -> int:
    """Magnitude code of a value near 2**e, random mantissa; beyond the range
    this gives subnormals, zero or the inf/NaN region."""
    M = fmt.mantissa_bits
    if e >= fmt.emin:
        return (min(e + fmt.bias, fmt._top) << M) | rng.bits(M)
    if not fmt.has_zero:
        return rng.bits(M)
    k = e - fmt.emin + M
    if k < 0:
        return rng.below(2)
    return (1 << k) | rng.bits(k)


class _Stimulus:
    """Operand tuples for a spec: edges, uniform, and biased-random draws."""

    def __init__(self, spec: VectorSpec):
        self.spec = spec
        self.widths = spec.operand_widths
        self.is_int = spec.kind == "from_int"
        f = spec.fmt
        if self.is_int:
            self.edges = _edge_ints(spec.bits, spec.signed, f.mantissa_bits)
        else:
            self.edges = edge_codes(f)
        self.specials = _specials(f)
        self.sign_bit = (1 << (f.exp_bits + f.mantissa_bits)) if f.signed else 0
        self.max_code = (1 << (f.exp_bits + f.mantissa_bits)) - 1
        exps = {f.emin - f.mantissa_bits - 1, f.emin - f.mantissa_bits, f.emin - 1, f.emin,
                -1, 0, 1, f.emax - 1, f.emax, f.emax + 1}
        if spec.kind == "convert":
            t = spec.to
            exps.update((t.emax - 1, t.emax, t.emax + 1, t.emin - 1, t.emin,
                         t.emin - t.mantissa_bits - 1, t.emin - t.mantissa_bits,
                         t.emin - t.mantissa_bits + 1))
        elif spec.kind == "to_int":
            exps.update(range(-2, spec.bits + 2))
        elif spec.kind == "rti":
            exps.update(range(-2, f.mantissa_bits + 3))
        self.exps = sorted(exps)

    def sizes(self, style: str) -> list[int]:
        return [len(self.edges)] * self.spec.arity if style == "edges" \
            else [1 << w for w in self.widths]

    def at(self, style: str, index: int) -> tuple[int, ...]:
        """Operand tuple number ``index`` of the fixed set (first operand most significant)."""
        sizes = self.sizes(style)
        out = []
        for size in reversed(sizes):
            index, r = divmod(index, size)
            out.append(self.edges[r] if style == "edges" else r)
        return tuple(reversed(out))

    def uniform(self, rng: _Rng) -> tuple[int, ...]:
        return tuple(rng.bits(w) for w in self.widths)

    def mixed(self, rng: _Rng) -> tuple[int, ...]:
        ops: list[int] = []
        for slot in range(self.spec.arity):
            ops.append(self._int(rng) if self.is_int else self._fp(rng, slot, ops))
        return tuple(ops)

    def _int(self, rng: _Rng) -> int:
        bits, signed, M = self.spec.bits, self.spec.signed, self.spec.fmt.mantissa_bits
        r = rng.below(10)
        if r < 2:
            return rng.bits(bits)
        if r < 5:
            return rng.choice(self.edges)
        length = 1 + rng.below(bits)
        v = rng.bits(length) | 1 << (length - 1)
        if r >= 8 and length >= M + 3:       # a rounding tie, nudged by one
            shift = length - M - 1
            v = (v >> shift << shift) | 1 << (shift - 1)
            v += rng.below(3) - 1
        if signed and rng.bits(1):
            v = -v
        return v & ((1 << bits) - 1)

    def _fp(self, rng: _Rng, slot: int, prev: list[int]) -> int:
        f = self.spec.fmt
        sign = rng.bits(1) * self.sign_bit if f.signed else 0
        if self.spec.op == "sqrt" and rng.below(4):
            sign = 0                          # keep most sqrt operands valid
        if slot and rng.below(100) < 35:
            mag, sign = self._correlated(rng, slot, prev, sign)
            return mag | sign
        r = rng.below(100)
        if r < 20:
            return rng.bits(f.size)
        if r < 45:
            return rng.choice(self.edges)
        if r < 55 and self.specials:
            return rng.choice(self.specials) | sign
        if r < 70:
            mag = rng.choice(_edge_mags(f)) + rng.below(7) - 3
            return min(max(mag, 0), self.max_code) | sign
        e = rng.choice(self.exps) + rng.below(3) - 1
        return _code_for_exp(f, e, rng) | sign

    def _correlated(self, rng: _Rng, slot: int, prev: list[int], sign: int):
        """Second/third operand built from the first: cancellation, equal
        magnitudes, close exponents, reciprocals, exponent sums near a limit."""
        f, op = self.spec.fmt, self.spec.op
        sbit = self.sign_bit
        a_mag, a_sign = prev[0] & (self.max_code), prev[0] & sbit
        ea = _unbiased(f, a_mag)
        modes = ["same", "close"]
        if op in ("mul", "div", "fma") and slot == 1:
            modes += ["target", "target", "recip"]
        elif op == "fma":
            modes += ["fma_cancel", "fma_cancel", "fma_cancel"]
        else:
            modes += ["cancel", "cancel", "neighbor"]
        mode = rng.choice(modes)
        if mode == "same":
            return a_mag, sign
        if mode == "cancel":
            mag = a_mag + (0, 0, 0, 1, -1, 2, -2)[rng.below(7)]
            return min(max(mag, 0), self.max_code), (a_sign ^ sbit if f.signed else 0)
        if mode == "neighbor":
            return min(max(a_mag + rng.below(7) - 3, 0), self.max_code), sign
        if mode == "close":
            return _code_for_exp(f, ea + rng.below(5) - 2, rng), sign
        if mode == "recip":
            field_ = min(max(2 * f.bias - (a_mag >> f.mantissa_bits), 0), f._top)
            return (field_ << f.mantissa_bits) | rng.bits(f.mantissa_bits), sign
        if mode == "target":                  # make the product/quotient land near a limit
            t = rng.choice(self.exps) + rng.below(3) - 1
            e = t - ea if op != "div" else ea - t
            return _code_for_exp(f, e, rng), sign
        # fma_cancel: addend of about -(a*b)
        eb = _unbiased(f, prev[1] & self.max_code)
        psign = (a_sign ^ (prev[1] & sbit)) if f.signed else 0
        mag = _code_for_exp(f, ea + eb + rng.below(3) - 1, rng)
        return mag, (psign ^ sbit if f.signed else 0)


def _sample(rng: _Rng, total: int, k: int) -> Iterable[int]:
    """k distinct indices of range(total), ascending."""
    if k >= total:
        return range(total)
    if 3 * k >= total:                         # partial Fisher-Yates
        idx = list(range(total))
        for i in range(k):
            j = i + rng.below(total - i)
            idx[i], idx[j] = idx[j], idx[i]
        return sorted(idx[:k])
    chosen: set[int] = set()
    while len(chosen) < k:
        chosen.add(rng.below(total))
    return sorted(chosen)


@dataclass
class Stats:
    """What happened while generating. ``skipped_by`` counts the exceptions the
    model raised (those stimuli are not vectors); ``warned`` counts vectors
    whose computation emitted a VeriFloatWarning (kept: the result is defined)."""
    stimuli: int = 0
    generated: int = 0
    warned: int = 0
    skipped_by: dict[str, int] = field(default_factory=dict)

    @property
    def skipped(self) -> int:
        return sum(self.skipped_by.values())

    def __str__(self) -> str:
        s = f"{self.generated} vectors"
        if self.skipped:
            s += ", skipped " + ", ".join(f"{n} ({k})" for k, n in sorted(self.skipped_by.items()))
        if self.warned:
            s += f", {self.warned} with warnings"
        return s


class VectorStream:
    """Iterator of ``Vector``s with ``spec`` and live ``stats`` attributes."""

    def __init__(self, spec: VectorSpec, count: int | None, seed: int, style: str, limit: int):
        self.spec, self.count, self.seed, self.style = spec, count, seed, style
        self.stats = Stats()
        self._stim = _Stimulus(spec)
        self._total = 0
        if style in ("edges", "exhaustive"):
            total = 1
            for size in self._stim.sizes(style):
                total *= size
            if count is None and total > limit:
                raise ValueError(
                    f"{style} for {spec} has {total} stimuli (limit {limit}); give count= to "
                    f"sample {style} stimuli, or raise limit=")
            self._total = total
        self._it = self._run()

    def __iter__(self) -> "VectorStream":
        return self

    def __next__(self) -> Vector:
        return next(self._it)

    def _one(self, operands: tuple[int, ...], rng: _Rng) -> Vector | None:
        spec, st = self.spec, self.stats
        sr = rng.bits(spec.result_format.sr_bits) if spec.uses_sr else None
        st.stimuli += 1
        try:
            vec, warned = spec._compute(operands, sr)
        except (ArithmeticError, ValueError) as e:
            st.skipped_by[type(e).__name__] = st.skipped_by.get(type(e).__name__, 0) + 1
            return None
        st.generated += 1
        st.warned += warned
        return vec

    def _run(self) -> Iterator[Vector]:
        spec, count, style, stim = self.spec, self.count, self.style, self._stim
        rng = _Rng(self.seed)
        if style in ("random", "mixed"):
            draw = stim.uniform if style == "random" else stim.mixed
            while self.stats.generated < count:
                vec = self._one(draw(rng), rng)
                if vec is not None:
                    yield vec
                elif self.stats.skipped > 100 * count + 1000:
                    raise RuntimeError(
                        f"{spec} raised for {self.stats.skipped} of {self.stats.stimuli} "
                        f"stimuli ({self.stats}); no usable vectors can be drawn")
            return
        total = self._total
        picks = range(total) if count is None else _sample(rng, total, count)
        for i in picks:
            vec = self._one(stim.at(style, i), rng)
            if vec is not None:
                yield vec


def generate(op: str, fmt: FPFormat, count: int | None = None, *, seed: int = 0,
             style: str = "mixed", to: FPFormat | None = None, bits: int | None = None,
             signed: bool | None = None, rounding: Rounding | None = None,
             exact: bool | None = None, limit: int = 1 << 22) -> VectorStream:
    """Vectors for ``op`` on ``fmt`` (see VectorSpec for ``to``, ``bits``, ...).

    style    ``"random"``: uniform codes. ``"edges"``: every combination of the
             edge encodings (see ``edge_codes``). ``"mixed"`` (default): random
             codes biased toward edges, specials, code neighbours, exponents
             at format limits, and cancelling / equal / reciprocal operands.
             ``"exhaustive"``: every operand combination, first operand
             slowest.
    count    ``random``/``mixed``: the number of vectors (stimuli the model
             rejects are skipped and redrawn). ``edges``/``exhaustive``: None
             is the whole set; a smaller count takes that many stimuli of it
             at random (skipped ones are not replaced).
    limit    refuse ``edges``/``exhaustive`` sets larger than this unless
             ``count`` is given.

    The same seed gives the same vectors on every Python version. With
    stochastic rounding each vector carries its random bits (``Vector.sr``).
    """
    spec = VectorSpec(op, fmt, to, bits, signed, rounding, exact)
    if style not in STYLES:
        raise ValueError(f"style must be one of {', '.join(STYLES)}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int")
    if count is not None and (not isinstance(count, int) or count < 0):
        raise ValueError("count must be a non-negative int")
    if count is None and style in ("random", "mixed"):
        raise ValueError(f"style {style!r} needs count=")
    return VectorStream(spec, count, seed, style, limit)


# --------------------------------------------------------------- files
def _fields(spec: VectorSpec, v: Vector) -> list[int]:
    return [*v.operands, *([v.sr] if spec.uses_sr else []), v.result, v.flags]


def _vector(spec: VectorSpec, vals: list[int]) -> Vector:
    n = spec.arity
    sr = vals[n] if spec.uses_sr else None
    return Vector(tuple(vals[:n]), vals[-2], vals[-1], sr)


def _meta(spec: VectorSpec, source, extra: dict | None) -> dict:
    meta = {"verifloat_vectors": _META_VERSION, "version": __version__, **spec.to_meta()}
    if isinstance(source, VectorStream):
        meta.update(seed=source.seed, style=source.style, count=source.count)
    meta.update(extra or {})
    return meta


def _meta_comments(meta: dict, prefix: str) -> list[str]:
    return [f"{prefix} {k}: {v}" for k, v in meta.items() if k != "verifloat_vectors"]


def systemverilog(spec: VectorSpec, name: str = "vec") -> str:
    """SystemVerilog declarations for the ``readmemh`` word: a packed struct
    (fields in file order, first field at the MSB) and its slices."""
    total = spec.word_width
    lines = [f"// {spec}: {total}-bit words, {_hexw(total)} hex digits per line",
             f"localparam int {name.upper()}_W = {total};"]
    hi = total
    members = []
    for fname, w in spec.fields:
        lo = hi - w
        lines.append(f"localparam int {name.upper()}_{fname.upper()}_LSB = {lo};  "
                     f"// [{hi - 1}:{lo}] {fname}, {w} bits")
        members.append(f"  logic [{w - 1}:0] {fname};")
        hi = lo
    lines += ["typedef struct packed {", *members, f"}} {name}_t;",
              f"// {name}_t mem [N]; initial $readmemh(\"file.hex\", mem);"]
    return "\n".join(lines) + "\n"


@contextlib.contextmanager
def _open_out(dest):
    if dest is None or dest == "-":
        yield sys.stdout
    elif isinstance(dest, (str, os.PathLike)):
        with open(dest, "w", encoding="utf-8", newline="\n") as fh:
            yield fh
    else:
        yield dest


@contextlib.contextmanager
def _open_lines(source):
    if source == "-":
        yield sys.stdin
    elif isinstance(source, (str, os.PathLike)):
        with open(source, encoding="utf-8", newline=None) as fh:
            yield fh
    else:
        yield source


def detect_format(source) -> str:
    """File format from the extension: .csv, .jsonl/.ndjson, .hex/.mem/.memh
    (readmemh); anything else is testfloat."""
    if isinstance(source, (str, os.PathLike)):
        ext = os.path.splitext(os.fspath(source))[1].lower()
        return {".csv": "csv", ".jsonl": "jsonl", ".ndjson": "jsonl", ".hex": "readmemh",
                ".mem": "readmemh", ".memh": "readmemh"}.get(ext, "testfloat")
    return "testfloat"


def write(dest, vectors: Iterable[Vector], spec: VectorSpec | None = None, *,
          format: str | None = None, header: bool | None = None,
          meta: dict | None = None) -> int:
    """Write vectors to a path, ``"-"``/None (stdout) or a text file; returns
    the count. ``spec`` may be omitted for a ``VectorStream``. ``format``
    defaults to the path's extension (see ``detect_format``), else testfloat.

    testfloat  hex fields as ``testfloat_gen`` prints them (uppercase, zero
               padded; flags 2 digits; booleans 0/1). With a stochastic-rounding
               format the ``sr`` field follows the operands.
    csv        header row of field names, then the same hex fields.
    jsonl      a metadata object, then one object per vector with hex strings.
    readmemh   one word per line, fields concatenated first field at the MSB.

    ``header`` adds the metadata as comment lines (``#``; ``//`` for readmemh,
    which also lists the bit slices); it is off for testfloat and csv, whose
    consumers expect plain data, and on for readmemh. jsonl always has its
    metadata line. ``meta`` adds keys to the metadata."""
    format = format or detect_format(dest)
    if format not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    spec = spec or getattr(vectors, "spec", None)
    if spec is None:
        raise ValueError("spec= is required unless vectors is a VectorStream")
    if header is None:
        header = format == "readmemh"
    info = _meta(spec, vectors, meta)
    names = [n for n, _ in spec.fields]
    digits = [_hexw(w) for _, w in spec.fields]
    n = 0
    with _open_out(dest) as fh:
        out = fh.write
        rows = csv.writer(fh, lineterminator="\n")
        if format == "jsonl":
            out(json.dumps(info) + "\n")
        elif format == "csv":
            if header:
                out("".join(c + "\n" for c in _meta_comments(info, "#")))
            rows.writerow(names)
        elif format == "readmemh":
            if header:
                out("".join(c + "\n" for c in _meta_comments(info, "//")))
                out("".join(ln if ln.startswith("//") else "// " + ln
                            for ln in systemverilog(spec).splitlines(keepends=True)))
        elif header:
            out("".join(c + "\n" for c in _meta_comments(info, "#")))
        total_digits = _hexw(spec.word_width)
        widths = [w for _, w in spec.fields]
        for v in vectors:
            vals = _fields(spec, v)
            for x, w, nm in zip(vals, widths, names):
                if not 0 <= x < 1 << w:
                    raise ValueError(f"{nm} {x:#x} does not fit in {w} bits")
            hexes = [f"{x:0{d}X}" for x, d in zip(vals, digits)]
            if format == "jsonl":
                out(json.dumps(dict(zip(names, hexes))) + "\n")
            elif format == "csv":
                rows.writerow(hexes)
            elif format == "readmemh":
                word = 0
                for x, w in zip(vals, widths):
                    word = word << w | x
                out(f"{word:0{total_digits}X}\n")
            else:
                out(" ".join(hexes) + "\n")
            n += 1
    return n


def _hex(text: str, width: int, where: str) -> int:
    try:
        v = int(text.replace("_", ""), 16)
    except ValueError:
        raise VectorFormatError(f"{where}: not a hex number: {text!r}") from None
    if not 0 <= v < 1 << width:
        raise VectorFormatError(f"{where}: {text!r} does not fit in {width} bits")
    return v


def read_meta(source) -> dict | None:
    """The metadata line of a jsonl file, or None for other files."""
    if detect_format(source) != "jsonl":
        return None
    with _open_lines(source) as lines:
        for line in lines:
            if line.strip():
                return _parse_meta(line, 1)
    return None


def _parse_meta(line: str, lineno: int) -> dict:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise VectorFormatError(f"line {lineno}: not JSON ({e})") from None
    if not isinstance(obj, dict) or "verifloat_vectors" not in obj:
        raise VectorFormatError(f"line {lineno}: first jsonl line must be the metadata object")
    return obj


def _read_lines(lines: Iterable[str], spec: VectorSpec, format: str
                ) -> Iterator[tuple[int, Vector]]:
    names = [n for n, _ in spec.fields]
    widths = [w for _, w in spec.fields]
    header_seen = format != "csv"
    meta_seen = format != "jsonl"
    for lineno, raw in enumerate(lines, 1):
        text = raw.strip()
        if not text:
            continue
        where = f"line {lineno}"
        if format == "jsonl":
            if not meta_seen:
                _parse_meta(text, lineno)
                meta_seen = True
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as e:
                raise VectorFormatError(f"{where}: not JSON ({e})") from None
            if not isinstance(obj, dict) or set(obj) != set(names):
                raise VectorFormatError(f"{where}: expected keys {names}")
            vals = [_hex(str(obj[n]), w, f"{where} {n}") for n, w in zip(names, widths)]
        elif format == "readmemh":
            text = text.split("//")[0].strip()
            for tok in text.split():
                if tok.startswith("@"):
                    raise VectorFormatError(f"{where}: address directives are not supported")
                word = _hex(tok, spec.word_width, where)
                vals = []
                for w in reversed(widths):
                    vals.append(word & ((1 << w) - 1))
                    word >>= w
                yield lineno, _vector(spec, vals[::-1])
            continue
        else:
            if text.startswith("#"):
                continue
            if format == "csv":
                cells = [c.strip() for c in next(csv.reader([text]))]
                if not header_seen:
                    if cells != names:
                        raise VectorFormatError(f"{where}: header {cells} != expected {names}")
                    header_seen = True
                    continue
            else:
                cells = text.split()
            if len(cells) != len(names):
                raise VectorFormatError(f"{where}: expected {len(names)} fields "
                                        f"({' '.join(names)}), found {len(cells)}")
            vals = [_hex(c, w, f"{where} {n}") for c, n, w in zip(cells, names, widths)]
        yield lineno, _vector(spec, vals)


def read(source, spec: VectorSpec | None = None, format: str | None = None
         ) -> Iterator[Vector]:
    """Read vectors from a path (``"-"`` is stdin) or an iterable of lines.
    ``format`` defaults to the extension (testfloat for lines). jsonl files
    carry their spec, so ``spec`` may be omitted for them. Raises
    VectorFormatError, naming the line, on malformed input."""
    spec, format = _resolve_source(source, spec, format)
    with _open_lines(source) as lines:
        for _, vec in _read_lines(lines, spec, format):
            yield vec


def _resolve_source(source, spec, format):
    format = format or detect_format(source)
    if format not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    if spec is None:
        if format != "jsonl":
            raise ValueError("spec= is required (only jsonl files describe themselves)")
        meta = read_meta(source) if isinstance(source, (str, os.PathLike)) else None
        if meta is None:
            raise ValueError("cannot read the jsonl metadata from this source; pass spec=")
        spec = VectorSpec.from_meta(meta)
    return spec, format


# --------------------------------------------------------------- verify
def _flag_names(flags: int) -> str:
    names = [f.name for f in FPFlags if flags & f]
    rest = flags & ~sum(FPFlags)
    return "|".join(names + ([f"{rest:#x}"] if rest else [])) or "none"


@dataclass
class Mismatch:
    """A vector whose file contents differ from the model. ``expected`` is None
    when the model raised (``error`` then says what)."""
    line: int
    got: Vector
    expected: Vector | None
    error: str | None = None

    def describe(self, spec: VectorSpec) -> str:
        w = _hexw(spec.result_width)
        ins = ", ".join(f"{x:#0{_hexw(b) + 2}x}" for x, b in zip(self.got.operands,
                                                                  spec.operand_widths))
        sr = f" sr={self.got.sr:#x}" if self.got.sr is not None else ""
        head = f"line {self.line}: {spec.op}({ins}){sr}"
        have = f"file {self.got.result:#0{w + 2}x} [{_flag_names(self.got.flags)}]"
        if self.expected is None:
            return f"{head}: {have}; model raised {self.error}"
        return (f"{head}: {have}; model {self.expected.result:#0{w + 2}x} "
                f"[{_flag_names(self.expected.flags)}]")


@dataclass
class VerifyReport:
    """Result of ``verify``: ``checked`` vectors, ``mismatch_count`` in all, the
    first ``mismatches`` kept in full."""
    spec: VectorSpec
    checked: int = 0
    mismatch_count: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    warned: int = 0

    @property
    def ok(self) -> bool:
        return self.mismatch_count == 0

    def summary(self) -> str:
        return (f"{self.spec}: checked {self.checked} vectors, "
                f"{self.mismatch_count} mismatches")


def verify(source, op: str | VectorSpec | None = None, fmt: FPFormat | None = None, *,
           format: str | None = None, to: FPFormat | None = None, bits: int | None = None,
           signed: bool | None = None, rounding: Rounding | None = None,
           exact: bool | None = None, max_report: int = 20, out=None) -> VerifyReport:
    """Re-compute every vector of a file (or iterable of lines) with the model.

    ``op``/``fmt`` and the options are those of ``generate`` (or pass a
    VectorSpec as ``op``); a jsonl file supplies them itself, and given ones
    must agree with its metadata. A vector whose operands the model rejects
    (it raises) counts as a mismatch. Malformed files raise VectorFormatError.
    The first ``max_report`` mismatches are kept, and printed to ``out`` (a
    text stream) together with a summary line when given."""
    if isinstance(op, VectorSpec):
        spec = op
    elif op is None:
        spec = None
    else:
        if fmt is None:
            raise TypeError("fmt is required with an operation name")
        spec = VectorSpec(op, fmt, to, bits, signed, rounding, exact)
    fmt_name = format or detect_format(source)
    if fmt_name == "jsonl" and isinstance(source, (str, os.PathLike)):
        meta = read_meta(source)
        if meta is not None and "op" in meta:
            file_spec = VectorSpec.from_meta(meta)
            if spec is not None and spec != file_spec:
                raise ValueError(f"file was generated for {file_spec}, not {spec}")
            spec = file_spec
    if spec is None:
        raise TypeError("op and fmt are required (only jsonl files describe themselves)")
    spec, fmt_name = _resolve_source(source, spec, format)
    report = VerifyReport(spec)
    with _open_lines(source) as lines:
        for lineno, got in _read_lines(lines, spec, fmt_name):
            report.checked += 1
            try:
                want, warned = spec._compute(got.operands, got.sr)
                report.warned += warned
                bad = (want.result, want.flags) != (got.result, got.flags)
                err = None
            except (ArithmeticError, ValueError) as e:
                want, bad, err = None, True, f"{type(e).__name__}: {e}"
            if bad:
                report.mismatch_count += 1
                if len(report.mismatches) < max_report:
                    report.mismatches.append(Mismatch(lineno, got, want, err))
    if out is not None:
        for m in report.mismatches:
            print(m.describe(spec), file=out)
        if report.mismatch_count > len(report.mismatches):
            print(f"... and {report.mismatch_count - len(report.mismatches)} more", file=out)
        print(report.summary(), file=out)
    return report


# --------------------------------------------------------------- CLI
_PRESETS = {"FP16": FP16, "BF16": BF16, "FP32": FP32, "FP64": FP64, "E5M2": E5M2, "E4M3": E4M3,
            "E3M2": E3M2, "E2M3": E2M3, "E2M1": E2M1, "UE4M3": UE4M3, "UE8M0": UE8M0}


def parse_format(text: str) -> FPFormat:
    """An FPFormat from a name: ``"e4m3, fn, rtz"`` (as ``str(fmt)`` prints it) or
    an uppercase preset, optionally with tags: ``"FP16"``, ``"E4M3, rtz"``. Note
    that lowercase ``e4m3`` is the plain IEEE-style format, not the OCP preset."""
    base, sep, rest = text.strip().partition(",")
    preset = _PRESETS.get(base.strip())
    if preset is None:
        return FPFormat.parse(text)
    return FPFormat.parse(str(preset) + sep + rest) if sep else preset


def _fmt_arg(text: str) -> FPFormat:
    try:
        return parse_format(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


_EPILOG = """\
operations: %s
  (also min, max, minnum, maxnum, rem, sgnj, mulAdd, roundToInt)

formats: a name as str(fmt) prints it ("e5m10", "e4m3, fn, rtz", "ue4m3, bias=10, fn"),
or an uppercase preset with optional tags ("FP16", "BF16", "E4M3, rtz"). Rounding mode,
NaN convention (x86, arm, ...) and tininess=before are tags of the format.
Lowercase e4m3 is IEEE-style; uppercase E4M3 is the OCP preset (= "e4m3, fn").

file formats: testfloat (testfloat_gen text), csv, jsonl (self-describing), readmemh.
Fields in file order: operands a b c, [sr: stochastic-rounding bits], result, flags
(5 bits, RISC-V fflags order NX UF OF DZ NV). Comparisons: result 0/1 (compare: lt=0
eq=1 gt=2 unordered=3).
Stochastic rounding (a format tagged "sr"): the random bits are exported in the sr field
so files replay without a shared random source; to_int / round_to_integral refuse it.
Vectors the model cannot answer (it raises, e.g. x/0 without NaN) are skipped and counted.
""" % ", ".join(OPS)


def _add_spec_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--op", required=True, metavar="OP", help="operation (see below)")
    p.add_argument("--fmt", required=True, type=_fmt_arg, metavar="FMT",
                   help='operand format; for from_int the result format; e.g. "e4m3, fn"')
    p.add_argument("--to", type=_fmt_arg, metavar="FMT", help="convert: the result format")
    p.add_argument("--bits", type=int, help="to_int / from_int: integer width (default 32)")
    p.add_argument("--unsigned", action="store_true",
                   help="to_int / from_int: unsigned integer (default signed)")
    p.add_argument("--rounding", choices=[r.value for r in Rounding if r is not Rounding.SR],
                   help="to_int / round_to_integral: rounding mode (default: the format's)")
    p.add_argument("--exact", action=argparse.BooleanOptionalAction, default=None,
                   help="to_int / round_to_integral: raise INEXACT (default: to_int yes, round_to_integral no)")


def _spec_from_args(a: argparse.Namespace) -> VectorSpec:
    return VectorSpec(a.op, a.fmt, a.to, a.bits, False if a.unsigned else None,
                      Rounding(a.rounding) if a.rounding else None, a.exact)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m verifloat.vectors", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate and verify (operands, result, flags) test vectors for any "
                    "VeriFloat format. Expected values always come from the model.",
        epilog=_EPILOG)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write test vectors",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_EPILOG,
                       description="Write vectors for one operation and format.")
    _add_spec_args(g)
    g.add_argument("--count", type=int, help="number of vectors (required for random and "
                   "mixed; for edges/exhaustive the default is the whole set)")
    g.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    g.add_argument("--style", choices=STYLES, default="mixed",
                   help="stimulus: random, edges, mixed (default) or exhaustive")
    g.add_argument("--format", choices=FORMATS, default=None,
                   help="file format (default: from the -o extension, else testfloat)")
    g.add_argument("-o", "--output", default="-", help="output file (default: stdout)")
    g.add_argument("--header", action=argparse.BooleanOptionalAction, default=None,
                   help="metadata comment lines (default: only for readmemh)")
    g.add_argument("--meta", metavar="FILE", help="also write the metadata as a JSON sidecar")
    g.add_argument("--sv", metavar="FILE",
                   help="also write SystemVerilog declarations of the readmemh word")
    g.add_argument("--limit", type=int, default=1 << 22,
                   help="largest edges/exhaustive set generated without --count (default 2**22)")
    v = sub.add_parser("verify", help="re-compute a vector file with the model",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_EPILOG,
                       description="Check every vector of FILE against the model; exit status "
                                   "1 if any differ, 2 if the file or arguments are bad.")
    v.add_argument("file", help="vector file, or - for stdin")
    v.add_argument("--op", metavar="OP", help="operation (optional for jsonl)")
    v.add_argument("--fmt", type=_fmt_arg, metavar="FMT", help="format (optional for jsonl)")
    v.add_argument("--to", type=_fmt_arg, metavar="FMT")
    v.add_argument("--bits", type=int)
    v.add_argument("--unsigned", action="store_true")
    v.add_argument("--rounding", choices=[r.value for r in Rounding if r is not Rounding.SR])
    v.add_argument("--exact", action=argparse.BooleanOptionalAction, default=None)
    v.add_argument("--format", choices=FORMATS, default=None,
                   help="file format (default: from the extension, else testfloat)")
    v.add_argument("--max-report", type=int, default=20, help="mismatches to print (default 20)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "generate":
            return _cmd_generate(args)
        return _cmd_verify(args)
    except (ValueError, TypeError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _cmd_generate(a: argparse.Namespace) -> int:
    spec = _spec_from_args(a)
    stream = generate(spec.op, spec.fmt, a.count, seed=a.seed, style=a.style, to=spec.to,
                      bits=spec.bits, signed=spec.signed, rounding=spec.rounding,
                      exact=spec.exact, limit=a.limit)
    fmt_name = a.format or (detect_format(a.output) if a.output != "-" else "testfloat")
    write(a.output, stream, format=fmt_name, header=a.header)
    if a.meta:
        with open(a.meta, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(_meta(spec, stream, None), fh, indent=2)
            fh.write("\n")
    if a.sv:
        with open(a.sv, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(systemverilog(spec))
    print(f"{spec}: wrote {stream.stats}" + ("" if a.output == "-" else f" to {a.output}"),
          file=sys.stderr)
    return 0


def _cmd_verify(a: argparse.Namespace) -> int:
    spec = _spec_from_args(a) if a.op else None
    if spec is None and (a.fmt or a.to):
        raise ValueError("--fmt needs --op")
    report = verify(a.file, spec, format=a.format, max_report=a.max_report, out=sys.stdout)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
