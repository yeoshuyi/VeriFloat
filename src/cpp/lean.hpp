// The tight kernel behind sequential dot products of narrow formats (operand
// significands of at most 31 bits: FP8, FP16, BF16, FP32, ...).
//
// Operands are normalized to exactly 31 significant bits, so a product has
// its leading bit at one of two positions and every rounding happens at a
// fixed bit. Like fast.hpp, a result it cannot produce exactly as the general
// kernel would (overflow, a subnormal or tiny result) marks the lane bad, and
// the caller repeats that dot product on the general path.
//
// lean_term, lean_add and lean_add_zero are the scalar definition of one step.
// The SIMD kernels (simd_kernels.inc) do the same arithmetic on several lanes
// at once.
#pragma once

#include "fast.hpp"

#include <vector>

namespace vf {

constexpr size_t kLanes = 4;            // dot products interleaved by the scalar kernel
constexpr int64_t kLeanBits = 31;       // operand significand bits
constexpr int64_t kLeanMaxP = 58;       // widest product-format mantissa
constexpr int64_t kLeanMaxF = 60;       // widest accumulator mantissa
constexpr int64_t kLeanMaxExp = int64_t(1) << 40;   // operand exponents: no overflow in any sum

// Operands as one array per field, normalized to 31 significant bits (zero
// stays zero), with room to read a whole vector at the end.
struct LeanOperands {
    std::vector<uint64_t> sig, neg;
    std::vector<int64_t> exp;
    bool ok = true;   // false: an exponent is too large for this kernel

    void assign(const std::vector<FV<u128>>& v) {
        const size_t n = v.size(), pad = 16;
        sig.assign(n + pad, 0);
        neg.assign(n + pad, 0);
        exp.assign(n + pad, 0);
        for (size_t i = 0; i < n; ++i) {
            uint64_t s = (uint64_t)v[i].sig;
            int64_t e = v[i].exp;
            if (s != 0) {
                const int64_t sh = kLeanBits - fbitlen(s);
                s <<= sh;
                e -= sh;
                if (e > kLeanMaxExp || e < -kLeanMaxExp) ok = false;
            } else e = 0;
            sig[i] = s;
            exp[i] = e;
            neg[i] = v[i].neg;
        }
    }
    LeanVec at(size_t off) const { return {sig.data() + off, exp.data() + off, neg.data() + off}; }
};

// Round up? `low` is the discarded fraction, left-justified (bit 63 = half).
VF_INLINE bool lean_up(uint64_t low, uint64_t s, bool neg, uint8_t rounding) {
    constexpr uint64_t H = uint64_t(1) << 63;
    switch (rounding) {
        case RNE: return (low | (s & 1)) > H;
        case RNA: return low >= H;
        case RTZ: return false;
        case RUP: return !neg & (low != 0);
        default: return neg & (low != 0);   // RDN
    }
}

// Is a result outside what the kernel may return for this format? e0 is the
// exponent of the leading bit before rounding and e after it, s the rounded
// significand. A value below the smallest normal is declined even when it
// rounds up to it: the general kernel rounds that at the subnormal position.
VF_INLINE bool lean_out_of_range(int64_t e0, int64_t e, uint64_t s, const RT& f) {
    if (e0 >= f.emin && e < f.emax) [[likely]] return false;
    return e0 < f.emin || e > f.emax || (f.fn_mask && (s & f.fn_mask) == f.fn_mask);
}

// A product x * y of two normalized operands, rounded by P when there is one:
// T with its leading bit at 62, exponent te, sign tn. False for a zero
// product (then only tn is set). ORs INEXACT and "bad" into flags.
struct LeanTerm {
    uint64_t T = 0;
    int64_t te = 0;
    bool tn = false;
};
VF_INLINE bool lean_term(LeanTerm& t, uint64_t xs, int64_t xe, uint64_t xn, uint64_t ys, int64_t ye, uint64_t yn,
                         const LeanSpec& sp, uint64_t& inexact, uint64_t& bad) {
    const uint64_t p = xs * ys;   // 0, or leading bit at 60 or 61
    const int64_t pe = xe + ye;
    t.tn = xn != yn;
    if (p == 0) return false;
    const int64_t top = (int64_t)(p >> 61);
    if (sp.has_p) {
        const RT& P = sp.P;
        const int64_t k = 60 - P.M + top;
        uint64_t s = p >> k;
        const uint64_t low = p << (64 - k);
        s += lean_up(low, s, t.tn, P.rounding);
        const int64_t cy = (int64_t)(s >> (P.M + 1));
        s >>= cy;
        inexact |= (uint64_t)(low != 0);
        const int64_t e0 = pe + 60 + top, e = e0 + cy;
        if (lean_out_of_range(e0, e, s, P)) [[unlikely]] bad = 1;
        t.T = s << (62 - P.M);
        t.te = e - 62;
    } else {
        t.T = p << (2 - top);
        t.te = pe - (2 - top);
    }
    return true;
}

// st = round_F(st + T * 2**te) for a nonzero T with its leading bit at 62.
// A lane that leaves the kernel is marked bad and keeps running on stale
// values (every shift count stays in range whatever the state holds).
VF_INLINE void lean_add(LeanState& st, uint64_t T, int64_t te, bool tn, const RT& F) {
    const int64_t MF = F.M;
    const uint64_t A = st.A;
    const int64_t d = st.ae - te;
    const bool an = st.an != 0;
    // |acc| >= |term|? (never for a zero accumulator)
    const bool ge = (A != 0) & ((d > 0) | ((d == 0) & (A >= T)));
    const uint64_t big = ge ? A : T, small = ge ? T : A;
    const int64_t be = ge ? st.ae : te;
    const bool bn = ge ? an : tn;
    int64_t dd = d < 0 ? -d : d;
    dd = dd > 63 ? 63 : dd;
    const uint64_t sm = small >> dd;
    const uint64_t lost = (uint64_t)((sm << dd) != small);
    const uint64_t S = an == tn ? big + sm : big - sm - lost;
    if (S == 0) [[unlikely]] {   // exact cancellation
        st.A = 0;
        st.ae = 0;
        st.an = F.rounding == RDN;
        return;
    }
    const int64_t lz = (int)std::countl_zero((uint64_t)S);
    const uint64_t N = S << lz;   // leading bit at 63
    const int64_t kF = 63 - MF;
    uint64_t s = N >> kF;
    const uint64_t low = (N << (64 - kF)) | lost;
    s += lean_up(low, s, bn, F.rounding);
    const int64_t cy = (int64_t)(s >> (MF + 1));
    s >>= cy;
    st.inexact |= (uint64_t)(low != 0);
    const int64_t e0 = be + 63 - lz, e = e0 + cy;
    if (lean_out_of_range(e0, e, s, F)) [[unlikely]] st.bad = 1;
    st.A = s << (62 - MF);
    st.ae = e - 62;
    st.an = bn;
}

// The sign of a zero accumulator after adding a zero of sign tn.
VF_INLINE void lean_add_zero(LeanState& st, bool tn, const RT& F) {
    if (st.A == 0 && (st.an != 0) != tn) st.an = F.rounding == RDN;
}

// One step of a sequential accumulation: st = round_F(st + round_P(x * y)).
VF_INLINE void lean_step(LeanState& st, uint64_t xs, int64_t xe, uint64_t xn, uint64_t ys, int64_t ye, uint64_t yn,
                         const LeanSpec& sp) {
    LeanTerm t;
    if (lean_term(t, xs, xe, xn, ys, ye, yn, sp, st.inexact, st.bad)) [[likely]] lean_add(st, t.T, t.te, t.tn, sp.F);
    else lean_add_zero(st, t.tn, sp.F);
}

// The scalar kernels (SeqLeanFn, PairLeanFn, ExactLeanFn). In the sequential
// one each sum is one dependency chain, so up to kLanes independent chains
// are interleaved.
VF_NOINLINE inline void seq_lean_scalar(const LeanVec* a, size_t rows, const LeanVec& b, size_t stride, size_t k,
                                        size_t lanes, const LeanSpec& sp, LeanState* out) {
    for (size_t r = 0; r < rows; ++r) {
        LeanState st[kLanes];
        const LeanVec& x = a[r];
        for (size_t i = 0; i < k; ++i) {
            const uint64_t xs = x.sig[i], xn = x.neg[i];
            const int64_t xe = x.exp[i];
            const size_t o = i * stride;
            for (size_t c = 0; c < lanes; ++c)
                lean_step(st[c], xs, xe, xn, b.sig[o + c], b.exp[o + c], b.neg[o + c], sp);
        }
        for (size_t c = 0; c < lanes; ++c) out[r * kLanes + c] = st[c];
    }
}

VF_NOINLINE inline void pair_lean_scalar(const LeanVec& a, const LeanVec& b, size_t stride, size_t k, size_t lanes,
                                         const LeanSpec& sp, uint64_t* scratch, LeanState* out) {
    // scratch as 3 words per term: A, ae, an
    for (size_t c = 0; c < lanes; ++c) {
        LeanState acc;   // collects the flags
        scratch[0] = scratch[1] = scratch[2] = 0;
        for (size_t i = 0; i < k; ++i) {
            const size_t o = i * stride + c;
            LeanTerm t;
            LeanState term;
            if (lean_term(t, a.sig[i], a.exp[i], a.neg[i], b.sig[o], b.exp[o], b.neg[o], sp, acc.inexact, acc.bad))
                lean_add(term, t.T, t.te, t.tn, sp.F);
            else term.an = t.tn;
            acc.inexact |= term.inexact;
            acc.bad |= term.bad;
            scratch[3 * i] = term.A;
            scratch[3 * i + 1] = (uint64_t)term.ae;
            scratch[3 * i + 2] = term.an;
        }
        size_t len = k ? k : 1;
        while (len > 1) {
            size_t o = 0;
            for (size_t i = 0; i + 1 < len; i += 2, ++o) {
                LeanState x;
                x.A = scratch[3 * i];
                x.ae = (int64_t)scratch[3 * i + 1];
                x.an = scratch[3 * i + 2];
                const uint64_t T = scratch[3 * i + 3];
                if (T != 0) lean_add(x, T, (int64_t)scratch[3 * i + 4], scratch[3 * i + 5] != 0, sp.F);
                else lean_add_zero(x, scratch[3 * i + 5] != 0, sp.F);
                acc.inexact |= x.inexact;
                acc.bad |= x.bad;
                scratch[3 * o] = x.A;
                scratch[3 * o + 1] = (uint64_t)x.ae;
                scratch[3 * o + 2] = x.an;
            }
            if (len % 2) {
                for (int w = 0; w < 3; ++w) scratch[3 * o + w] = scratch[3 * (len - 1) + w];
                ++o;
            }
            len = o;
        }
        acc.A = scratch[0];
        acc.ae = (int64_t)scratch[1];
        acc.an = scratch[2];
        out[c] = acc;
    }
}

VF_NOINLINE inline void exact_lean_scalar(const LeanVec& a, const LeanVec& b, size_t stride, size_t k, size_t lanes,
                                          const LeanSpec& sp, const int64_t* base, ExactSum* out) {
    for (size_t c = 0; c < lanes; ++c) {
        i128 sum = 0;
        ExactSum r;
        for (size_t i = 0; i < k; ++i) {
            const size_t o = i * stride + c;
            const uint64_t p = a.sig[i] * b.sig[o];
            if (p == 0) continue;
            const int64_t pe = a.exp[i] + b.exp[o];
            const bool neg = a.neg[i] != b.neg[o];
            uint64_t v = p;
            int64_t sh = pe - base[c];
            if (sp.has_p) {
                const RT& P = sp.P;
                const int64_t top = (int64_t)(p >> 61);
                const int64_t kk = 60 - P.M + top;
                uint64_t s = p >> kk;
                const uint64_t low = p << (64 - kk);
                s += lean_up(low, s, neg, P.rounding);
                const int64_t cy = (int64_t)(s >> (P.M + 1));
                r.inexact |= (uint64_t)(low != 0);
                const int64_t e0 = pe + 60 + top;
                if (lean_out_of_range(e0, e0 + cy, s >> cy, P)) [[unlikely]] r.bad = 1;
                v = s;          // s * 2**(pe + 60 - P.M + top), carry included
                sh += top;
            }
            if (sh < 0 || sh > 64) [[unlikely]] { r.bad = 1; continue; }   // outside the caller's promise
            const i128 term = (i128)((u128)v << sh);
            sum += neg ? -term : term;
        }
        // as four signed limbs: sum = limb0 + limb1 * 2**32 + ... (limb3 carries the sign)
        const u128 u = (u128)sum;
        r.limb[0] = (int64_t)((uint64_t)u & 0xffffffff);
        r.limb[1] = (int64_t)((uint64_t)(u >> 32) & 0xffffffff);
        r.limb[2] = (int64_t)((uint64_t)(u >> 64) & 0xffffffff);
        r.limb[3] = (int64_t)(sum >> 96);
        out[c] = r;
    }
}

// The value of an ExactSum (see simd_api.hpp).
inline i128 exact_total(const ExactSum& s) {
    return (i128)s.limb[0] + ((i128)s.limb[1] << 32) + ((i128)s.limb[2] << 64) + ((i128)s.limb[3] << 96);
}

// The accumulator as a value with its leading bit at position M (or zero).
inline FV<uint64_t> lean_result(const LeanState& st, int64_t MF) {
    if (st.A == 0) return {0, 0, st.an != 0};
    return {st.A >> (62 - MF), st.ae + (62 - MF), st.an != 0};
}

}  // namespace vf
