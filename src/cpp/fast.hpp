// Fast arithmetic for the common case: finite operands and a normal, in-range
// result in a signed format with a zero. Every function here either gives
// exactly what the general kernel (kernel.hpp) gives or reports that it cannot
// (returns false); the caller then repeats the operation on the general path.
// Nothing here raises a warning: any result that could (overflow, underflow,
// clamping) is declined.
#pragma once

#include "kernel.hpp"
#include "simd_api.hpp"

// The kernels are small functions meant to be merged into their loops, and
// the loops are kept as functions of their own so that they are optimized in
// isolation.
#if defined(__GNUC__) || defined(__clang__)
#define VF_INLINE inline __attribute__((always_inline))
#define VF_NOINLINE __attribute__((noinline))
#else
#define VF_INLINE inline
#define VF_NOINLINE
#endif

namespace vf {

// Fast kernels enabled? (_core.set_fast; tests compare both paths.)
inline bool g_fast = true;
// Results produced by a fast kernel, and results it handed to the general
// path (_core.fast_stats; tests check that ordinary data stays on the kernels).
inline uint64_t g_fast_hits = 0, g_fast_misses = 0;

// A finite value sig * 2**exp. `neg` is also the sign of a zero.
template <class U> struct FV {
    U sig = 0;
    int64_t exp = 0;
    bool neg = false;
};

template <class U> constexpr int kBitsOf = (int)sizeof(U) * 8;

inline int64_t fbitlen(uint64_t x) { return 64 - __builtin_clzll(x); }   // x != 0
inline int64_t fbitlen(u128 x) {
    uint64_t hi = (uint64_t)(x >> 64);
    return hi ? 128 - __builtin_clzll(hi) : 64 - __builtin_clzll((uint64_t)x);
}

// Can fast_round target this format?
inline bool fast_target(const Fmt& f) { return f.fast; }

static_assert((int)kRNE == (int)RNE && (int)kRNA == (int)RNA && (int)kRTZ == (int)RTZ && (int)kRUP == (int)RUP &&
              (int)kRDN == (int)RDN);

// What fast_round needs of a target format (RT, simd_api.hpp).
inline RT rt_of(const Fmt& f) {
    return {f.M, f.emin, f.emax, f.max_mant_is_mask ? 0 : f.mask(), f.rounding};
}

// A format's encoding geometry for the element-wise SIMD kernels.
inline EwFmt ew_fmt(const Fmt& f) {
    return {f.M, f.bias, f.E + f.M, f.top, f.mask(), f.max_field, f.max_mant_is_mask ? 0 : f.mask(), f.has_nan(),
            f.ftz, f.rounding};
}
// May codes of this format be operands of the element-wise SIMD kernels?
inline bool ew_source(const Fmt& f) { return f.has_zero && f.M >= 1 && f.M <= 61 && f.size <= 64; }

// The SIMD kernels in use (nullptr: scalar kernels only). Chosen once from
// what the CPU supports; _core.set_simd overrides it for tests.
const SimdKernels* simd_active();

// Decode a finite encoding of a narrow format (M <= 61).
template <class U> inline FV<U> fv_of(const Code& c, const Fmt& f) {
    FV<U> v;
    v.neg = c.sign;
    if (c.field == 0 && f.has_zero) {
        if (!f.ftz) { v.sig = (U)c.mant; v.exp = 1 - f.bias - f.M; }
    } else {
        v.sig = (U)(c.mant | (uint64_t(1) << f.M));
        v.exp = (int64_t)c.field - f.bias - f.M;
    }
    return v;
}

// Exact x + y, or the sum with its low bits folded into `sticky` (the value
// is then strictly between sig and sig + 1 units). Operands must be below
// 2**(bits - 2). A zero result takes the IEEE sign for the rounding mode.
// Written with selects rather than branches: which operand is larger and
// whether the signs agree are unpredictable in a dot product.
template <class U>
VF_INLINE void fast_add(const FV<U>& x0, const FV<U>& y0, uint8_t rounding, FV<U>& out, bool& sticky) {
    sticky = false;
    if (x0.sig == 0 || y0.sig == 0) [[unlikely]] {
        if (x0.sig == 0 && y0.sig == 0) out = {0, 0, x0.neg == y0.neg ? x0.neg : rounding == RDN};
        else out = x0.sig != 0 ? x0 : y0;
        return;
    }
    const bool sw = x0.exp < y0.exp;
    const U xs = sw ? y0.sig : x0.sig, ys = sw ? x0.sig : y0.sig;
    const int64_t xe = sw ? y0.exp : x0.exp, ye = sw ? x0.exp : y0.exp;
    const bool xn = sw ? y0.neg : x0.neg, yn = sw ? x0.neg : y0.neg;
    const int64_t d = xe - ye;
    const int64_t room = (kBitsOf<U> - 2) - fbitlen(xs);
    const int64_t sh = d < room ? d : room;
    const int64_t r = d - sh;
    const U A = xs << sh;
    U B = ys;
    if (r != 0) [[unlikely]] {
        if (r >= kBitsOf<U>) { B = 0; sticky = true; }
        else {
            B = ys >> r;
            sticky = (ys & ((U(1) << r) - 1)) != 0;
        }
    }
    // With r != 0, A fills the word and B is below it, so A > B.
    const bool same = xn == yn, ge = A >= B;
    const U dif = ge ? A - B - (U)sticky : B - A;
    out.sig = same ? A + B : dif;
    out.exp = xe - sh;
    out.neg = (same || ge) ? xn : yn;
    if (out.sig == 0 && !sticky) [[unlikely]] out.neg = rounding == RDN;   // exact cancellation
}

// Round sig * 2**exp (plus sticky) into f. Succeeds only for zero and for
// results that are normal before and after rounding; `out` is then the
// rounded value with its leading bit at position M. ORs INEXACT into flags
// (an unsigned, not a byte: a byte store would alias everything).
// The rounding decision is branch-free (it is data dependent).
template <class U>
VF_INLINE bool fast_round(const FV<U>& v, bool sticky, const RT& f, FV<U>& out, unsigned& flags) {
    if (v.sig == 0) [[unlikely]] {
        if (sticky) return false;
        out = {0, 0, v.neg};
        return true;
    }
    const int64_t M = f.M;
    const int64_t n = fbitlen(v.sig);
    int64_t e = v.exp + n - 1;
    const int64_t k = n - (M + 1);
    if (e < f.emin || (sticky && k <= 0)) [[unlikely]] return false;   // subnormal or tiny; too few bits
    const int64_t kp = k > 0 ? k : 0, kn = k > 0 ? 0 : -k;
    U s = (v.sig >> kp) << kn;
    const U rem = v.sig & ((U(1) << kp) - 1), half = (U(1) << kp) >> 1;
    const bool inexact = (rem != 0) | sticky;
    bool up;
    switch (f.rounding) {
        case RNE: up = (rem > half) | ((rem == half) & (sticky | (bool)(s & 1))); break;
        case RNA: up = rem >= half; break;
        case RTZ: up = false; break;
        case RUP: up = !v.neg; break;
        default: up = v.neg;   // RDN
    }
    s += (U)(up & inexact);
    const int64_t carry = (int64_t)(s >> (M + 1));   // 0 or 1
    s >>= carry;
    e += carry;
    flags |= (unsigned)inexact;   // INEXACT == 1
    if (e >= f.emax) [[unlikely]] {
        if (e > f.emax) return false;
        // 'fn': the all-ones code of the top binade is NaN
        if (f.fn_mask && ((uint64_t)s & f.fn_mask) == f.fn_mask) return false;
    }
    out = {s, e - M, v.neg};
    return true;
}

// Round sig * 2**exp (plus sticky) into an encoding of f, as fast_round does,
// and also when the result is subnormal (formats with gradual underflow),
// with UNDERFLOW by the format's tininess convention. Used where a single
// result is produced (scalars, array elements); the dot-product kernels keep
// to fast_round. Still declines overflow, the 'fn' NaN code, and tiny results
// of formats that wrap or flush. f must be a fast_target (signed: an
// underflow raises no warning).
// The fields of an encoding of a narrow format, as plain data (a Code also
// carries the storage of wide mantissas, which costs in a tight loop).
struct FCode {
    uint64_t field = 0, mant = 0;
    bool sign = false;
    Code code() const {
        Code c;
        c.sign = sign;
        c.field = field;
        c.mant = mant;
        return c;
    }
    uint64_t raw(const Fmt& f) const { return ((uint64_t)sign << (f.E + f.M)) | (field << f.M) | mant; }
};

// The subnormal part of fast_round_code: v is nonzero and below the smallest
// normal value of f.
template <class U>
VF_NOINLINE bool fast_round_subnormal(const FV<U>& v, bool sticky, const Fmt& f, FCode& c, unsigned& flags) {
    if (f.wrap || f.ftz) return false;   // no gradual underflow
    const int64_t M = f.M;
    const int64_t n = fbitlen(v.sig);
    const int64_t e = v.exp + n - 1;
    const int64_t k = n - (M + 1);   // bits below a normal result
    if (sticky && k <= 0) return false;   // too few bits to place the sticky bit
    auto round_up = [&](const U& rem, const U& half, const U& kept) -> bool {
        switch (f.rounding) {
            case RNE: return (rem > half) | ((rem == half) & (sticky | (bool)(kept & 1)));
            case RNA: return rem >= half;
            case RTZ: return false;
            case RUP: return !v.neg;
            default: return v.neg;   // RDN
        }
    };
    // Tiny after rounding unless rounding to M + 1 bits reaches the smallest normal.
    bool carry = false;
    if (k > 0) {
        const U s = v.sig >> k, rem = v.sig & ((U(1) << k) - 1), half = U(1) << (k - 1);
        const bool inexact = (rem != 0) | sticky;
        carry = inexact && round_up(rem, half, s) && ((s + 1) >> (M + 1)) != 0;
    }
    const bool tiny = f.tininess_before || !(e == f.emin - 1 && carry);
    const int64_t cut = (f.emin - M) - v.exp;   // bits below the subnormal unit
    U q;
    bool lossy;
    if (cut <= 0) {
        if (sticky) return false;
        q = v.sig << -cut;
        lossy = false;
    } else {
        U r2, h2;
        if (cut > n) {           // entirely below half a unit
            q = 0;
            r2 = 1;
            h2 = 2;
        } else {
            q = cut < kBitsOf<U> ? U(v.sig >> cut) : U(0);
            r2 = cut < kBitsOf<U> ? U(v.sig & ((U(1) << cut) - 1)) : v.sig;
            h2 = U(1) << (cut - 1);
        }
        lossy = (r2 != 0) | sticky;
        q += (U)(lossy && round_up(r2, h2, q));
    }
    if (lossy) flags |= INEXACT | (tiny ? UNDERFLOW : 0);
    c.sign = v.neg;
    c.field = (uint64_t)(q >> M) != 0 ? 1 : 0;   // rounded up to the smallest normal?
    c.mant = (uint64_t)q & f.mask();
    return true;
}

template <class U>
VF_INLINE bool fast_round_code(const FV<U>& v, bool sticky, const Fmt& f, FCode& c, unsigned& flags) {
    FV<U> r;
    unsigned fl = 0;
    if (fast_round(v, sticky, rt_of(f), r, fl)) [[likely]] {   // zero, or normal and in range
        flags |= fl;
        c.sign = r.neg;
        c.field = r.sig != 0 ? (uint64_t)(r.exp + f.M + f.bias) : 0;
        c.mant = (uint64_t)r.sig & f.mask();
        return true;
    }
    if (v.sig == 0 || v.exp + fbitlen(v.sig) - 1 >= f.emin) return false;   // overflow, the 'fn' NaN code, ...
    return fast_round_subnormal(v, sticky, f, c, flags);
}

// a op b for finite operands of format f, on the fast kernel: the result's
// encoding and flags, or false to decline. One function per operation, so
// that a loop over one operation holds only that operation's code.
VF_INLINE bool fast_add_code(const FV<uint64_t>& x, const FV<uint64_t>& y, const Fmt& f, FCode& c, unsigned& fl) {
    FV<uint64_t> s;
    bool sticky;
    fast_add(x, y, f.rounding, s, sticky);
    return fast_round_code(s, sticky, f, c, fl);
}
VF_INLINE bool fast_sub_code(const FV<uint64_t>& x, FV<uint64_t> y, const Fmt& f, FCode& c, unsigned& fl) {
    y.neg = !y.neg;
    return fast_add_code(x, y, f, c, fl);
}
VF_INLINE bool fast_mul_code(const FV<uint64_t>& x, const FV<uint64_t>& y, const Fmt& f, FCode& c, unsigned& fl) {
    if (2 * (f.M + 1) <= 62)
        return fast_round_code(FV<uint64_t>{x.sig * y.sig, x.exp + y.exp, x.neg != y.neg}, false, f, c, fl);
    return fast_round_code(FV<u128>{(u128)x.sig * y.sig, x.exp + y.exp, x.neg != y.neg}, false, f, c, fl);
}
VF_INLINE bool fast_div_code(const FV<uint64_t>& x, const FV<uint64_t>& y, const Fmt& f, FCode& c, unsigned& fl) {
    if (y.sig == 0) return false;   // x / 0: DIVZERO or INVALID
    if (x.sig == 0) return fast_round_code(FV<uint64_t>{0, 0, x.neg != y.neg}, false, f, c, fl);
    // quotient with at least M + 3 bits, the remainder as sticky
    int64_t k = f.M + 4 + fbitlen(y.sig) - fbitlen(x.sig);
    if (k < 0) k = 0;
    if (fbitlen(x.sig) + k <= 64) [[likely]] {
        const uint64_t N = x.sig << k;
        return fast_round_code(FV<uint64_t>{N / y.sig, x.exp - y.exp - k, x.neg != y.neg}, N % y.sig != 0, f, c, fl);
    }
    const u128 N = (u128)x.sig << k;
    return fast_round_code(FV<u128>{N / y.sig, x.exp - y.exp - k, x.neg != y.neg}, N % y.sig != 0, f, c, fl);
}
inline bool fast_op_code(char op, const FV<uint64_t>& x, const FV<uint64_t>& y, const Fmt& f, FCode& c, unsigned& fl) {
    switch (op) {
        case '+': return fast_add_code(x, y, f, c, fl);
        case '-': return fast_sub_code(x, y, f, c, fl);
        case '*': return fast_mul_code(x, y, f, c, fl);
        default: return fast_div_code(x, y, f, c, fl);
    }
}

// Encoding of a fast_round result (or of zero).
template <class U> inline Code fv_code(const FV<U>& v, const Fmt& f) {
    Code c;
    c.sign = v.neg;
    if (v.sig != 0) {
        c.field = (uint64_t)(v.exp + f.M + f.bias);
        c.mant = (uint64_t)v.sig & f.mask();
    }
    return c;
}

// Raw codes of a fast_target format straight to and from values, without
// going through a Code (array loops).
struct RawFmt {
    int64_t M, bias, sign_shift;
    uint64_t top, mmask;
    bool ftz;
    explicit RawFmt(const Fmt& f)
        : M(f.M), bias(f.bias), sign_shift(f.E + f.M), top(f.top), mmask(f.mask()), ftz(f.ftz) {}
};
// Decode a finite code (the caller has excluded the all-ones exponent when it
// holds inf/NaN).
template <class U> VF_INLINE FV<U> fv_of_raw(uint64_t raw, const RawFmt& d) {
    const uint64_t field = (raw >> d.M) & d.top, mant = raw & d.mmask;
    const bool normal = field != 0;
    FV<U> v;
    v.neg = (raw >> d.sign_shift) & 1;
    v.sig = (U)(normal ? mant | (uint64_t(1) << d.M) : (d.ftz ? 0 : mant));
    v.exp = (int64_t)(normal ? field : 1) - d.bias - d.M;
    return v;
}
// Encode a fast_round result (or zero).
template <class U> VF_INLINE uint64_t raw_of_fv(const FV<U>& v, const RawFmt& d) {
    const uint64_t sign = (uint64_t)v.neg << d.sign_shift;
    if (v.sig == 0) return sign;
    return sign | ((uint64_t)(v.exp + d.M + d.bias) << d.M) | ((uint64_t)v.sig & d.mmask);
}

// Raw code <-> fields for formats of at most 64 bits.
inline uint64_t raw64_of(const Code& c, const Fmt& f) {
    uint64_t r = (c.field << f.M) | c.mant;
    if (c.sign) r |= uint64_t(1) << (f.E + f.M);
    return r;
}
inline Code code_of_raw64(uint64_t raw, const Fmt& f) {
    Code c;
    c.sign = f.is_signed && ((raw >> (f.E + f.M)) & 1);
    c.field = (raw >> f.M) & f.top;
    c.mant = raw & f.mask();
    return c;
}

}  // namespace vf
