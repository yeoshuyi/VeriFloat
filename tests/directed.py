"""Designed operands for IEEE-style formats: the cases where implementations
of floating point go wrong, built from the format's parameters instead of
drawn at random.

Nothing here computes an expected result. The cases are handed to references
VeriFloat did not write (Berkeley SoftFloat, the host FPU, GNU MPFR), and
``classify`` says, with exact rational arithmetic, which boundary a case's
exact result sits on, so that a test can require that the design reaches all
of them:

* rounding: exact results, exact ties, and results a hair above or below a
  tie, in the normal and in the subnormal range;
* overflow: results between the largest value and the overflow threshold,
  on the threshold and above it;
* underflow: results that round up to the smallest normal value, where
  detecting tininess before or after rounding gives different flags;
* cancellation, sticky bits far below the result, signed zeros;
* every kind of special operand: zeros, subnormals, infinities, quiet and
  signaling NaNs with assorted payloads, of both signs.

A format is any FPFormat with IEEE inf/NaN encodings (a standard format or
not); operands are raw codes.
"""

from __future__ import annotations

from fractions import Fraction

from verifloat import FPFormat


# ---------------------------------------------------------------- building codes

def code(f: FPFormat, sign: int, e: int, mant: int) -> int | None:
    """The code of (-1)**sign * 1.mant * 2**e. Below the normal range the
    significand is shifted into a subnormal (low bits dropped); None when e
    is beyond the format's range in either direction."""
    E, M = f.exp_bits, f.mantissa_bits
    if e > f.emax:
        return None
    if e >= f.emin:
        return (sign << (E + M)) | ((e + f.bias) << M) | mant
    shift = f.emin - e
    if shift > M:
        return None
    return (sign << (E + M)) | (((1 << M) | mant) >> shift)


def from_dyadic(f: FPFormat, sign: int, n: int, q: int) -> int | None:
    """The code of (-1)**sign * n * 2**q, or None if it is not a value of the format."""
    E, M = f.exp_bits, f.mantissa_bits
    if n == 0:
        return sign << (E + M)
    while n % 2 == 0:
        n >>= 1
        q += 1
    e = q + n.bit_length() - 1                  # exponent of the leading bit
    if e > f.emax or n.bit_length() > M + 1:
        return None
    if e >= f.emin:
        mant = (n << (M + 1 - n.bit_length())) & f._mask
        return (sign << (E + M)) | ((e + f.bias) << M) | mant
    lsb = f.emin - M                            # exponent of the smallest subnormal
    if q < lsb:
        return None
    return (sign << (E + M)) | (n << (q - lsb))


def mantissas(f: FPFormat, few: bool = False) -> list[int]:
    """Mantissa fields that expose rounding: all zeros and ones, single low
    and high bits, half patterns, alternating bits."""
    M, mask = f.mantissa_bits, f._mask
    half = M // 2
    alt = int("01" * (M // 2 + 1), 2) & mask
    pats = [0, 1, mask, mask - 1, 1 << (M - 1), alt, 1 << half]
    if not few:
        pats += [2, 3, (1 << (M - 1)) - 1, (1 << (M - 1)) + 1, (alt << 1) & mask, (1 << half) - 1,
                 mask ^ ((1 << half) - 1), 3 << (M - 2), mask - 2]
    out = []
    for m in pats:
        if 0 <= m <= mask and m not in out:
            out.append(m)
    return out


def edge_codes(f: FPFormat) -> list[int]:
    """The special and extreme encodings, of both signs: zero, the smallest
    and largest subnormals, the smallest normals, values around one, the
    largest values, infinity, and quiet and signaling NaNs with several
    payloads."""
    E, M, mask, top = f.exp_bits, f.mantissa_bits, f._mask, f._top
    one = f.bias << M
    quiet = 1 << (M - 1)
    mags = [0, 1, 2, mask, mask - 1, 1 << M, (1 << M) | 1, (2 << M) - 1,
            one - 1, one, one + 1, one | quiet, one + (1 << M),
            ((top - 1) << M) | mask, ((top - 1) << M) | (mask - 1), (top - 1) << M,
            top << M,                                                   # inf
            (top << M) | quiet, (top << M) | quiet | 1, (top << M) | mask,          # quiet NaNs
            (top << M) | 1, (top << M) | (quiet - 1), (top << M) | (quiet >> 1)]    # signaling NaNs
    out = []
    for m in mags:
        for s in (0, 1):
            c = (s << (E + M)) | m
            if c not in out:
                out.append(c)
    return out


def few_edges(f: FPFormat) -> list[int]:
    """A smaller set for triples: one of each kind."""
    E, M, mask, top = f.exp_bits, f.mantissa_bits, f._mask, f._top
    one, quiet, sb = f.bias << M, 1 << (M - 1), 1 << (E + M)
    return [0, sb, 1, sb | mask, 1 << M, one, sb | one, one + 1, ((top - 1) << M) | mask,
            sb | ((top - 1) << M) | mask, top << M, sb | (top << M), (top << M) | quiet,
            sb | (top << M) | quiet | 1, (top << M) | 1, sb | (top << M) | (quiet - 1)]


# ---------------------------------------------------------------- arithmetic

def _exps(f: FPFormat) -> list[int]:
    p = f.mantissa_bits + 1
    return sorted({f.emin - 2, f.emin, f.emin + 1, f.emin + p, -1, 0, 1, f.emax - p - 1, f.emax - 1, f.emax})


def _targets(f: FPFormat) -> list[int]:
    """Exponents of a result that put it on a boundary of the format."""
    p = f.mantissa_bits + 1
    return sorted({f.emin - p - 2, f.emin - p - 1, f.emin - p, f.emin - p + 1, f.emin - p // 2, f.emin - 2,
                   f.emin - 1, f.emin, f.emin + 1, -1, 0, 1, f.emax - 1, f.emax, f.emax + 1})


def add_cases(f: FPFormat) -> list[tuple[int, int]]:
    """Sums and differences: equal exponents (cancellation), operands one
    place apart, and an addend at, just above and just below half an ulp of
    the other (ties and sticky bits), at each end of the exponent range."""
    p = f.mantissa_bits + 1
    out = []
    for ea in _exps(f):
        for d in (0, 1, -1, 2, p - 1, p, p + 1, p + 2, -(p - 1), -p, -(p + 1), 2 * p + 1):
            for ma in mantissas(f):
                for mb in mantissas(f, few=True):
                    for sa, sb in ((0, 0), (0, 1), (1, 0), (1, 1)):
                        a, b = code(f, sa, ea, ma), code(f, sb, ea - d, mb)
                        if a is not None and b is not None:
                            out.append((a, b))
    return out


def _splits(f: FPFormat, t: int) -> list[tuple[int, int]]:
    """Ways to reach a product exponent t from two operand exponents."""
    out = []
    for ea in (-(-t // 2), f.emin, f.emax, -1):
        eb = t - ea
        if f.emin - f.mantissa_bits <= eb <= f.emax and (ea, eb) not in out:
            out.append((ea, eb))
    return out


def mul_cases(f: FPFormat) -> list[tuple[int, int]]:
    """Products whose exponent lands at each end of the range and in the
    subnormal range, from mantissas whose products are exact, ties or carry
    sticky bits."""
    out = []
    for t in _targets(f):
        for ea, eb in _splits(f, t):
            for ma in mantissas(f):
                for mb in mantissas(f):
                    for sa, sb in ((0, 0), (0, 1), (1, 1)):
                        a, b = code(f, sa, ea, ma), code(f, sb, eb, mb)
                        if a is not None and b is not None:
                            out.append((a, b))
    return out


def div_cases(f: FPFormat) -> list[tuple[int, int]]:
    """Quotients at the same exponents; a divisor that is a power of two
    makes exact quotients and, in the subnormal range, exact ties."""
    out = []
    for t in _targets(f):
        for ea in (-(-t // 2), f.emin, f.emax, 0):
            eb = ea - t
            if not f.emin - f.mantissa_bits <= eb <= f.emax:
                continue
            for ma in mantissas(f):
                for mb in mantissas(f):
                    for sa, sb in ((0, 0), (0, 1), (1, 1)):
                        a, b = code(f, sa, ea, ma), code(f, sb, eb, mb)
                        if a is not None and b is not None:
                            out.append((a, b))
    return out


def fma_cases(f: FPFormat) -> list[tuple[int, int, int]]:
    """a * b + c with the addend at, above and far below the product: ties
    made by the addend, sticky bits from a product's low half, and exact
    cancellation of the product."""
    p = f.mantissa_bits + 1
    out = []
    few = mantissas(f, few=True)
    for t in (f.emin - p - 1, f.emin - 1, f.emin, 0, f.emax - 1, f.emax, f.emax + 1):
        ea = -(-t // 2)
        eb = t - ea
        for d in (0, 1, -1, 2, p, p + 1, -p, -(p + 1), -2 * p, -(2 * p + 1), 2 * p + 2):
            for ma in few:
                for mb in few:
                    for mc in few:
                        for sa, sc in ((0, 0), (0, 1), (1, 0), (1, 1)):
                            a, b, c = code(f, sa, ea, ma), code(f, 0, eb, mb), code(f, sc, t + d, mc)
                            if a is not None and b is not None and c is not None:
                                out.append((a, b, c))
    return out


def sqrt_cases(f: FPFormat) -> list[int]:
    """Every mantissa pattern at even and odd exponents across the range,
    subnormals, and perfect squares with their neighbours."""
    M = f.mantissa_bits
    out = []
    for e in sorted({f.emin - M, f.emin - 2, f.emin - 1, f.emin, f.emin + 1, -2, -1, 0, 1, 2, f.emax - 1, f.emax}):
        for m in mantissas(f):
            c = code(f, 0, e, m)
            if c is not None:
                out.append(c)
    for s in [*range(1, 48), (1 << (M // 2)) - 1, (1 << (M // 2)) + 1, (1 << ((M + 1) // 2)) - 1]:
        c = from_dyadic(f, 0, s * s, 0)
        if c is not None:
            out += [c - 1, c, c + 1]
        c = from_dyadic(f, 0, s * s, -2 * (M // 2))
        if c is not None:
            out += [c - 1, c, c + 1]
    out += [c | (1 << (f.exp_bits + M)) for c in out[:8]]         # negative operands: invalid
    return list(dict.fromkeys(out))


def rem_cases(f: FPFormat) -> list[tuple[int, int]]:
    """Remainders with the operands' exponents far apart (long quotients),
    equal, and with x below y (where the result is x or x - y)."""
    p = f.mantissa_bits + 1
    out = []
    for ea in (f.emin - 2, f.emin, 0, 1, p + 3, f.emax):
        for d in (0, 1, -1, 2, 3, p, 2 * p + 1, 5 * p + 3, f.emax - f.emin):
            for ma in mantissas(f):
                for mb in mantissas(f, few=True):
                    for sa, sb in ((0, 0), (0, 1), (1, 0)):
                        a, b = code(f, sa, ea, ma), code(f, sb, ea - d, mb)
                        if a is not None and b is not None:
                            out.append((a, b))
    return out


# ---------------------------------------------------------------- conversions

def _guards(g: int) -> list[int]:
    """Values of g guard bits below a rounding position: zero, the smallest,
    just below half, half, just above half, the largest."""
    half = 1 << (g - 1)
    return sorted({0, 1, half - 1, half, half + 1, (1 << g) - 1})


def convert_cases(src: FPFormat, dst: FPFormat) -> list[int]:
    """Codes of ``src`` whose values sit on the rounding boundaries of
    ``dst``: every guard-bit pattern below dst's last place, in dst's normal
    range, at its overflow threshold and through its subnormal range."""
    out = list(edge_codes(src))
    Md, ps = dst.mantissa_bits, src.mantissa_bits + 1
    for e in sorted({dst.emin - Md - 2, dst.emin - Md - 1, dst.emin - Md, dst.emin - Md + 1, dst.emin - Md // 2,
                     dst.emin - 2, dst.emin - 1, dst.emin, dst.emin + 1, -1, 0, 1, dst.emax - 1, dst.emax,
                     dst.emax + 1, src.emin, src.emax}):
        # dst keeps `keep` significant bits of a value with exponent e
        keep = Md + 1 if e >= dst.emin else Md + 1 - (dst.emin - e)
        g = min(4, ps - max(keep, 0) - (1 if keep <= 0 else 0))
        if g < 1:                                   # src has no bits below dst's last place here
            g = 0
        for top in mantissas(dst):
            lead = ((1 << Md) | top) >> (Md + 1 - keep) if keep > 0 else 0
            for guard in (_guards(g) if g else [0]):
                if keep > 0:
                    n, q = (lead << g) | guard, e - (keep - 1) - g
                else:                               # the whole value lies below dst's last place
                    n, q = (1 << g) | guard, e - g
                for s in (0, 1):
                    c = from_dyadic(src, s, n, q)
                    if c is not None:
                        out.append(c)
    return list(dict.fromkeys(out))


def _near(f: FPFormat, n: int) -> list[int]:
    """Codes of the values of f at and around the integer n: n itself if it
    is one, else its two neighbours, and one more step each way."""
    M = f.mantissa_bits
    if n == 0:
        return [0, 1]
    bits = n.bit_length()
    if bits <= M + 1:
        c = from_dyadic(f, 0, n, 0)
    else:                                           # truncate to the format's precision
        c = from_dyadic(f, 0, n >> (bits - M - 1), bits - M - 1)
    if c is None:
        return []
    return [c - 1, c, c + 1, c + 2]


def to_int_cases(f: FPFormat, bits: int, signed: bool) -> list[int]:
    """Operands around the integer type's limits and around every half."""
    out = list(edge_codes(f))
    sb = 1 << (f.exp_bits + f.mantissa_bits)
    lim = bits - 1 if signed else bits
    for n in (0, 1, 2, 3, 4, 255, (1 << lim) - 2, (1 << lim) - 1, 1 << lim, (1 << lim) + 1, (1 << (lim + 1)),
              (1 << bits) - 1, 1 << bits, (1 << bits) + 1):
        for c in _near(f, n):
            out += [c, c | sb]
        for num in (1, 2, 3, 5, 6, 7):              # n + k/4 and n + k/8 where the format has the bits
            for den_log in (2, 3):
                c = from_dyadic(f, 0, (n << den_log) + num, -den_log)
                if c is not None:
                    out += [c, c | sb]
    return list(dict.fromkeys(out))


def from_int_cases(f: FPFormat, bits: int, signed: bool) -> list[int]:
    """Integers (as bit patterns of the type) at the type's limits and with
    every guard-bit pattern below the format's last place."""
    p = f.mantissa_bits + 1
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    vals = {0, 1, 2, 3, hi, hi - 1, hi - 2, lo, lo + 1, lo + 2, -1, -2}
    for width in range(p - 1, min(bits, p + 6) + 1):       # integers of p-1 .. p+6 bits
        for top in (1 << (width - 1), (1 << width) - 1, (1 << (width - 1)) | (1 << (width - 2)) if width > 1 else 1):
            g = max(width - p, 0)
            for guard in (_guards(g) if g else [0]):
                v = (top >> g << g) | guard
                vals |= {v, -v, v + 1, -(v + 1)}
    for shift in (bits - p - 3, bits - p - 2, bits - p - 1, bits - p):   # the same at the top of the type
        if shift > 0:
            g = min(shift, 4)
            for guard in _guards(g):
                v = (((1 << p) - 1) << shift) | (guard << (shift - g))
                vals |= {v, -v, v >> 1, -(v >> 1)}
    return sorted((v & ((1 << bits) - 1)) for v in vals if lo <= v <= hi)


def round_int_cases(f: FPFormat) -> list[int]:
    """Values at and around every kind of half-integer, small and at the
    point where the format's spacing reaches one."""
    M = f.mantissa_bits
    out = list(edge_codes(f))
    sb = 1 << (f.exp_bits + M)
    for n in (0, 1, 2, 3, 4, 5, (1 << (M - 2)) - 1, 1 << (M - 2), (1 << (M - 1)) - 1, 1 << (M - 1), (1 << M) - 1,
              1 << M, (1 << (M + 1)) - 1):
        for num, den_log in ((0, 0), (1, 1), (1, 2), (3, 2), (1, 3), (7, 3)):
            c = from_dyadic(f, 0, (n << den_log) + num, -den_log)
            if c is not None:
                out += [c - 1, c, c + 1, (c - 1) | sb, c | sb, (c + 1) | sb]
    return [c for c in dict.fromkeys(out) if c >= 0]


# ---------------------------------------------------------------- what a case reaches

def classify(f: FPFormat, x: Fraction) -> str:
    """Where the exact result x sits relative to the format's rounding and
    range boundaries."""
    if x == 0:
        return "zero"
    M = f.mantissa_bits
    mag = abs(x)
    big = Fraction(f.max)
    ulp_top = Fraction(2) ** (f.emax - M)
    if mag > big:
        thr = big + ulp_top / 2                     # where round-to-nearest starts to overflow
        return "over, below the threshold" if mag < thr else "over, on the threshold" if mag == thr else "over"
    small = Fraction(2) ** f.emin                   # smallest normal
    if mag < small:
        u = Fraction(2) ** (f.emin - M)
        r = mag / u
        frac = r - (r.numerator // r.denominator)
        kind = "exact" if frac == 0 else "tie" if frac == Fraction(1, 2) else "inexact"
        if mag >= small * (1 - Fraction(1, 1 << (M + 2))) and frac != 0:
            return "tiny, rounds to the smallest normal"         # tininess before / after differ
        return f"subnormal, {kind}"
    e = mag.numerator.bit_length() - mag.denominator.bit_length()
    if Fraction(2) ** e > mag:
        e -= 1
    r = mag / Fraction(2) ** (e - M)
    frac = r - (r.numerator // r.denominator)
    if frac == 0:
        return "exact"
    if frac == Fraction(1, 2):
        return "tie"
    near = Fraction(1, 1 << (M - 1))
    return "just off a tie" if abs(frac - Fraction(1, 2)) <= near else "inexact"


def exact(f: FPFormat, c: int) -> Fraction | None:
    """The value of a finite code (None for inf and NaN), from the encoding's definition."""
    E, M = f.exp_bits, f.mantissa_bits
    field, mant = (c >> M) & f._top, c & f._mask
    if field == f._top:
        return None
    sig = mant if field == 0 else (1 << M) | mant
    v = Fraction(sig) * Fraction(2) ** (max(field, 1) - f.bias - M)
    return -v if c >> (E + M) else v
