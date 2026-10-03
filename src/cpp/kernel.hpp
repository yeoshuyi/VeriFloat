// Encodings, decoding and the rounding kernel (FP._round in the Python
// reference), templated on the significand type U (u128 or BigInt).
#pragma once

#include "common.hpp"

#include <memory>

namespace vf {

inline uint64_t low64(u128 x) { return (uint64_t)x; }
inline uint64_t low64(const BigInt& x) { return static_cast<uint64_t>(x & BigInt(~uint64_t(0))); }

// One encoding: sign, exponent field, mantissa field. Mantissas of wide
// formats (M > 64) live in `wmant`.
struct Code {
    bool sign = false;
    uint64_t field = 0;
    uint64_t mant = 0;
    Rc<BigInt> wmant;
};

inline BigInt big_mask(int64_t M) { return (BigInt(1) << (unsigned)M) - 1; }

inline bool mant_zero(const Code& c, const Fmt& f) { return f.wide ? (!c.wmant || *c.wmant == 0) : c.mant == 0; }
inline bool mant_is_mask(const Code& c, const Fmt& f) {
    return f.wide ? (c.wmant && *c.wmant == big_mask(f.M)) : c.mant == f.mask();
}
inline bool mant_bit(const Code& c, const Fmt& f, int64_t k) {
    if (k < 0) return false;
    if (f.wide) return c.wmant && bit(*c.wmant, k);
    return k < 64 && ((c.mant >> k) & 1);
}
template <class U> inline U mant_as(const Code& c, const Fmt& f) {
    if (!f.wide) return U(c.mant);
    if constexpr (std::is_same_v<U, BigInt>) return c.wmant ? *c.wmant : BigInt(0);
    else return to_u128(c.wmant ? *c.wmant : BigInt(0));
}
inline BigInt mant_big(const Code& c, const Fmt& f) { return mant_as<BigInt>(c, f); }

// Store the low M bits of `m` as the mantissa field (UINT(m, M) wraps).
template <class U> inline void set_mant(Code& c, const Fmt& f, const U& m) {
    if (!f.wide) c.mant = low64(m) & f.mask();
    else c.wmant = Rc<BigInt>(to_big(m) & big_mask(f.M));
}
inline void set_mant_big(Code& c, const Fmt& f, const BigInt& m) { set_mant<BigInt>(c, f, m); }

inline Code make_code(bool sign, uint64_t field, uint64_t mant, const Fmt& f) {
    Code c;
    c.sign = sign;
    c.field = field & f.top;
    if (!f.wide) c.mant = mant & f.mask();
    else c.wmant = Rc<BigInt>(BigInt(mant) & big_mask(f.M));
    return c;
}

// Classification
inline bool is_nan(const Code& c, const Fmt& f) {
    if (!f.has_nan() || c.field != f.top) return false;
    return f.inf_nan == FN ? mant_is_mask(c, f) : !mant_zero(c, f);
}
inline bool is_inf(const Code& c, const Fmt& f) {
    return f.has_inf() && c.field == f.top && mant_zero(c, f);
}
inline bool is_finite(const Code& c, const Fmt& f) { return !is_nan(c, f) && !is_inf(c, f); }
inline bool is_snan(const Code& c, const Fmt& f) {
    return is_nan(c, f) && f.inf_nan == IEEE && !mant_bit(c, f, f.M - 1);
}
// Finite and exactly zero (a subnormal reads as zero under ftz).
inline bool is_zero(const Code& c, const Fmt& f) {
    if (!is_finite(c, f)) return false;
    return c.field == 0 && f.has_zero && (f.ftz || mant_zero(c, f));
}

// Exact value of a finite encoding.
template <class U> inline Dy<U> decode(const Code& c, const Fmt& f) {
    Dy<U> d;
    d.neg = c.sign;
    if (c.field == 0 && f.has_zero) {
        if (f.ftz) { d.sig = 0; d.exp = 0; return d; }
        d.sig = mant_as<U>(c, f);
        d.exp = 1 - f.bias - f.M;
    } else {
        d.sig = mant_as<U>(c, f) | pow2<U>(f.M);
        d.exp = (int64_t)c.field - f.bias - f.M;
    }
    return d;
}

// Python-format names (FPFormat.__str__) for internal formats.
std::string fmt_str(const Fmt& f);

// Special encodings
inline Code max_code(bool sign, const Fmt& f) {
    Code c;
    c.sign = sign;
    c.field = f.max_field;
    if (!f.wide) c.mant = f.max_mant_is_mask ? f.mask() : f.mask() - 1;
    else c.wmant = Rc<BigInt>(big_mask(f.M) - (f.max_mant_is_mask ? 0 : 1));
    return c;
}
inline Code nan_code(bool sign, const Fmt& f) {   // fmt.nan(sign): canonical
    Code c;
    c.sign = sign && f.is_signed;
    c.field = f.top;
    if (f.inf_nan == FN) {
        if (!f.wide) c.mant = f.mask();
        else c.wmant = Rc<BigInt>(big_mask(f.M));
    } else {
        if (!f.wide) c.mant = uint64_t(1) << (f.M - 1);
        else c.wmant = Rc<BigInt>(BigInt(1) << (unsigned)(f.M - 1));
    }
    return c;
}
inline Code inf_code(bool sign, const Fmt& f) { return make_code(sign, f.top, 0, f); }
inline Code zero_code(bool sign, const Fmt& f) { return make_code(sign, 0, 0, f); }

struct RoundOut {
    Code c;
    uint8_t flags = 0;
    Event ev = EV_NONE;
};

[[noreturn]] void raise_no_zero(const Fmt& f);
inline RoundOut no_zero(const Fmt& f) {
    if (!f.has_nan()) raise_no_zero(f);
    return {nan_code(false, f), INVALID, EV_NONE};
}

// IEEE 754 7.4 overflow result.
inline RoundOut overflow_result(bool sign, const Fmt& f) {
    bool to_inf;
    switch (f.rounding) {
        case RTZ: to_inf = false; break;
        case RUP: to_inf = !sign; break;
        case RDN: to_inf = sign; break;
        default: to_inf = true;
    }
    if (!f.has_nan() || f.saturate || !to_inf) return {max_code(sign, f), 0, EV_SATURATED};
    if (f.has_inf()) return {inf_code(sign, f), 0, EV_OVERFLOWED};
    return {nan_code(sign, f), 0, EV_OVERFLOWED};
}

// Random bits for stochastic rounding: drawn from the host when first
// needed, exactly as the reference does. `given` is the host's own handle
// of an explicit value (the Python binding keeps the sr_rand argument
// there); sr_draw reads it, or draws from the host's source, and stores the
// integer with set().
struct SRArg {
    void* given = nullptr;
    bool have = false;
    bool small = true;
    int64_t sv = 0;
    Rc<BigInt> bv;   // when the value does not fit int64

    void set(int64_t v) { small = true; sv = v; }
    void set(const BigInt& v) { small = false; bv = Rc<BigInt>(v); }
    inline void resolve(const Fmt& f);
    // q + sr >= 2**sb ?
    template <class U> bool carries(const U& q, int64_t sb) const {
        if (small && sb <= 62) {
            if constexpr (std::is_same_v<U, u128>) {
                i128 s = (i128)q + (i128)sv;
                return s >= ((i128)1 << sb);
            }
        }
        BigInt s = to_big(q);
        if (small) s += sv; else s += *bv;
        return s >= (BigInt(1) << (unsigned)sb);
    }
};

void sr_draw(SRArg& sr, const Fmt& f);   // supplied by the host
inline void SRArg::resolve(const Fmt& f) {
    if (have) return;
    have = true;
    sr_draw(*this, f);
}

template <class U> struct Rnd {
    U s;
    int64_t e;
    bool inexact;
};

// _round_sig's direction decision for an inexact quotient: s = sig >> k
// (k > 0) has discarded the nonzero low k bits of sig. `odd` overrides the
// parity used by ties-to-even (-1: use s).
template <class U>
inline bool decide_up(const U& sig, int64_t k, const U& s, bool neg, uint8_t rounding, int odd,
                      const SRArg& sr, int64_t sb) {
    switch (rounding) {
        case RNE: {
            bool half = bit(sig, k - 1);
            bool below = low_nonzero(sig, k - 1);
            bool is_odd = odd < 0 ? bit(s, 0) : odd;
            return half && (below || is_odd);
        }
        case RNA: return bit(sig, k - 1);
        case RTZ: return false;
        case RUP: return !neg;
        case RDN: return neg;
        default: {
            // Stochastic: the discarded fraction rounded to sb bits (ties
            // to even), plus the random bits; a carry out rounds away.
            U rem = k >= bitlen(sig) ? sig : U(sig & (pow2<U>(k) - 1));
            U q;
            int64_t j = k - sb;
            if (j <= 0) q = shl(rem, -j);
            else {
                q = shr(rem, j);
                if (bit(rem, j - 1) && (low_nonzero(rem, j - 1) || bit(q, 0))) q += 1;
            }
            return sr.carries(q, sb);
        }
    }
}

// _round_sig at exponent E: round the exact value sig * 2**exp to M+1
// significant bits at binade E (top = floor(log2(value))).
template <class U>
inline Rnd<U> rnd_at(const U& sig, int64_t exp, int64_t top, int64_t E, const Fmt& f, bool neg,
                     const SRArg& sr) {
    const int64_t M = f.M;
    // Zero-width mantissa: ties-to-even uses the code's parity.
    int odd = -1;
    if (M == 0) odd = (E == top) && (((E + f.bias) & 1) != 0);
    int64_t k = (E - M) - exp;
    Rnd<U> r;
    r.e = E;
    if (k <= 0) {
        r.s = shl(sig, -k);
        r.inexact = false;
    } else {
        r.s = shr(sig, k);
        r.inexact = low_nonzero(sig, k);
        if (r.inexact && decide_up(sig, k, r.s, neg, f.rounding, odd, sr, f.sr_bits)) r.s += 1;
    }
    if (shr(r.s, M + 1) != 0) {
        r.s = shr(r.s, 1);
        r.e = E + 1;
    }
    return r;
}

// FP._round: round a sticky dyadic into f. `sr` is resolved lazily.
template <class U>
RoundOut round_dy(Dy<U> x, const Fmt& f, bool zero_sign, SRArg& sr) {
    if (x.sticky) {
        x.sig = shl(x.sig, 1) | U(1);
        x.exp -= 1;
        x.sticky = false;
    }
    if (x.sig == 0) {
        if (!f.has_zero) return no_zero(f);
        return {zero_code(zero_sign && f.is_signed, f), 0, EV_NONE};
    }
    if (x.neg && !f.is_signed) {
        if (!f.has_zero) return no_zero(f);
        return {zero_code(false, f), UNDERFLOW | INEXACT, EV_CLAMPED};
    }
    if (f.rounding == SR) sr.resolve(f);
    const bool sign = x.neg;
    const int64_t M = f.M;
    int64_t e = x.exp + bitlen(x.sig) - 1;

    // Tininess: before rounding, or after rounding to M bits with an
    // unbounded exponent (IEEE 754-2019 7.5).
    bool tiny;
    if (f.tininess_before) tiny = e < f.emin;
    else tiny = e < f.emin && rnd_at(x.sig, x.exp, e, e, f, sign, sr).e < f.emin;

    const bool gradual = f.has_zero && !(f.wrap || f.ftz);
    Rnd<U> r = rnd_at(x.sig, x.exp, e, gradual ? std::max(e, f.emin) : e, f, sign, sr);
    e = r.e;

    if (f.ftz && tiny) return {zero_code(sign, f), UNDERFLOW | INEXACT, EV_FLUSHED};
    bool over = e > f.emax;
    if (!over && e == f.emax && !f.max_mant_is_mask && (shr(r.s, M) != 0 || !f.has_zero)) {
        // 'fn': the all-ones code of the top binade is NaN. (Only a normal
        // result has that code: in a format with a single exponent bit the
        // subnormals share this exponent, and their all-ones mantissa is an
        // ordinary value. Version 0.1 took that one for the NaN code too.)
        U m = r.s & (pow2<U>(M) - 1);
        over = m == pow2<U>(M) - 1;
    }
    if (over) {
        if (f.wrap) {
            Code c;
            c.sign = sign;
            c.field = (uint64_t)(e + f.bias) & f.top;
            set_mant(c, f, r.s);
            return {c, OVERFLOW | INEXACT, EV_WRAPPED_UP};
        }
        RoundOut o = overflow_result(sign, f);
        o.flags = OVERFLOW | INEXACT;
        return o;
    }
    if (e < f.emin) {
        if (f.wrap) {
            Code c;
            c.sign = sign;
            c.field = (uint64_t)(e + f.bias) & f.top;
            set_mant(c, f, r.s);
            return {c, UNDERFLOW | INEXACT, EV_WRAPPED_DOWN};
        }
        // No zero and no subnormals: the smallest value is the floor.
        return {make_code(sign, (uint64_t)f.min_field(), 0, f), UNDERFLOW | INEXACT, EV_UNDERFLOWED};
    }
    RoundOut o;
    o.c.sign = sign;
    o.c.field = (shr(r.s, M) != 0 || !f.has_zero) ? (uint64_t)(e + f.bias) & f.top : 0;
    set_mant(o.c, f, r.s);
    if (r.inexact) {
        o.flags |= INEXACT;
        if (tiny && (gradual || !f.has_zero)) {
            o.flags |= UNDERFLOW;
            o.ev = EV_UNDERFLOWED;
        }
    }
    return o;
}

// Does rounding a dyadic of `bits` significant bits into f fit the u128 tier?
inline bool fits_u128(int64_t bits) { return bits <= kU128Bits; }

// Reduce an exact dyadic to at most `keep` significant bits plus sticky.
template <class U> inline Dy<U> reduce(Dy<U> d, int64_t keep) {
    int64_t n = bitlen(d.sig);
    if (n > keep) {
        int64_t k = n - keep;
        d.sticky = d.sticky || low_nonzero(d.sig, k);
        d.sig = shr(d.sig, k);
        d.exp += k;
    }
    return d;
}

inline Dy<BigInt> to_big(const Dy<u128>& d) { return {d.neg, d.exp, to_big(d.sig), d.sticky}; }

// Round a dyadic whose significand is held as BigInt, using the u128 kernel
// when it fits after reduction to the format's precision.
inline RoundOut round_big(Dy<BigInt> x, const Fmt& f, bool zero_sign, SRArg& sr) {
    int64_t keep = f.prec + 2;
    if (keep + 2 <= kU128Bits && f.M <= 64) {
        x = reduce(x, keep);
        Dy<u128> y{x.neg, x.exp, to_u128(x.sig), x.sticky};
        return round_dy(y, f, zero_sign, sr);
    }
    return round_dy(x, f, zero_sign, sr);
}

// Round a u128 dyadic (any width), falling back to BigInt if the format is wide.
inline RoundOut round_u(Dy<u128> x, const Fmt& f, bool zero_sign, SRArg& sr) {
    int64_t keep = f.prec + 2;
    if (keep + 2 <= kU128Bits && f.M <= 64) return round_dy(reduce(x, keep), f, zero_sign, sr);
    return round_dy(to_big(x), f, zero_sign, sr);
}

}  // namespace vf
