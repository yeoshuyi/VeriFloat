// Plain-data interface between the core and its SIMD kernels.
//
// The SIMD kernels are compiled in their own translation units with their own
// instruction-set flags (simd_avx2.cpp, simd_avx512.cpp). Those units include
// nothing but this header and <immintrin.h>, and everything in them has
// internal linkage, so that no function built for a newer instruction set can
// be picked up by code that runs on an older CPU.
//
// Every kernel is integer-only and does exactly what the scalar kernels in
// fast.hpp / lean.hpp do, lane by lane. A lane it cannot finish (a rare case
// or a result outside the kernel's range) is handed back to scalar code.
#pragma once

#include <cstddef>
#include <cstdint>

namespace vf {

// Rounding modes, as in common.hpp (checked there with static_assert).
enum : uint8_t { kRNE = 0, kRNA = 1, kRTZ = 2, kRUP = 3, kRDN = 4 };

// What fast rounding needs of a target format, held by value so that an
// inner loop keeps it in registers.
struct RT {
    int64_t M, emin, emax;
    uint64_t fn_mask;   // 'fn' formats: the mantissa that is NaN in the top binade (else 0)
    uint8_t rounding;
};

// ---------------------------------------------------------------- dot products

// How products are rounded (P, optional) and what the sum is rounded into (F).
struct LeanSpec {
    RT F, P;
    bool has_p;
};

// One accumulator, or one rounded term. A is the significand with its leading
// bit at 62 (or 0 for zero), the value is A * 2**ae, `an` its sign (also the
// sign of a zero).
struct LeanState {
    uint64_t A = 0;
    int64_t ae = 0;
    uint64_t an = 0;
    uint64_t inexact = 0;
    uint64_t bad = 0;   // the lane left the kernel: redo the dot product on another path
};

// Operands, one array per field ("structure of arrays"), normalized to 31
// significant bits (lean.hpp): value = sig * 2**exp, neg is 0 or 1.
struct LeanVec {
    const uint64_t* sig;
    const int64_t* exp;
    const uint64_t* neg;
};

// All dot-product kernels compute `lanes` dot products (lanes <= width) of
// the vector a (k elements) with consecutive columns of b, where element t of
// column c is b[t * stride + c]. The b arrays must be readable up to `width`
// columns from the first one.

// Sequential: s = round_F(s + round_P(a[i] * b[i])). rows is 1 or 2 (two rows
// run side by side: each sum is one dependency chain, and independent chains
// overlap in the CPU); st[r * width + c] is row r, column c.
constexpr size_t kLeanRows = 2;
using SeqLeanFn = void (*)(const LeanVec* a, size_t rows, const LeanVec& b, size_t stride, size_t k, size_t lanes,
                           const LeanSpec& sp, LeanState* st);

// Pairwise: every term rounded into F (after P), then a balanced tree of
// rounded additions. scratch holds 3 * max(k, 1) * width words.
using PairLeanFn = void (*)(const LeanVec& a, const LeanVec& b, size_t stride, size_t k, size_t lanes,
                            const LeanSpec& sp, uint64_t* scratch, LeanState* st);

// Exact: the sum of the (P-rounded) terms, not yet rounded into F. Lane c
// returns sum_j limb[j] * 2**(32 j) in units of 2**(base[c] + (has_p ? 60 -
// P.M : 0)), where base[c] is a lower bound of every term's exponent
// (a[i].exp + b[i].exp) in that lane that is at most kExactMaxShift below any
// of them, and k < 2**kExactMaxLog2K: then no limb can overflow.
constexpr int64_t kExactMaxShift = 46;
constexpr int64_t kExactMaxLog2K = 14;
struct ExactSum {
    int64_t limb[4] = {0, 0, 0, 0};
    uint64_t inexact = 0;
    uint64_t bad = 0;
};
using ExactLeanFn = void (*)(const LeanVec& a, const LeanVec& b, size_t stride, size_t k, size_t lanes,
                             const LeanSpec& sp, const int64_t* base, ExactSum* out);

// ---------------------------------------------------------------- element-wise

// Encoding geometry of a signed format with a zero, for element-wise kernels.
struct EwFmt {
    int64_t M, bias, sign_shift;
    uint64_t top;          // all-ones exponent field
    uint64_t mmask;
    uint64_t max_field;    // exponent field of the largest finite binade
    uint64_t fn_mask;
    bool has_special;      // the all-ones exponent field holds inf/NaN ('fn': also finite values)
    bool ftz;              // subnormal operands read as zero
    uint8_t rounding;
};

// out[i] = a[i * sa] op b[i * sb] for i < n (sa, sb are 0 or 1), with
// flags[i] = INEXACT or 0. The kernel works in vectors of `width` elements;
// bit l of redo[v] marks element v * width + l as not computed (its out and
// flags are unspecified): the caller computes it in scalar code. op is '+',
// '-' or '*'. Sums need three bits below the rounding position (a sticky bit
// can sit two places above the last one after a cancellation), products need
// the two significands to fit 32 bits each.
constexpr int64_t kEwMaxAddM = 60;
constexpr int64_t kEwMaxMulM = 31;
using EwBinFn = void (*)(char op, const uint64_t* a, size_t sa, const uint64_t* b, size_t sb, size_t n,
                         const EwFmt& f, uint64_t* out, uint8_t* flags, uint8_t* redo);

// out[i] = src[i] (a code of format s) rounded into format f. Same contract.
using EwConvFn = void (*)(const uint64_t* src, size_t n, const EwFmt& s, const EwFmt& f, uint64_t* out,
                          uint8_t* flags, uint8_t* redo);

struct SimdKernels {
    const char* name;
    size_t width;          // lanes per vector
    SeqLeanFn seq_lean;
    PairLeanFn pair_lean;
    ExactLeanFn exact_lean;
    EwBinFn ew_bin;
    EwConvFn ew_conv;
};

// Kernel tables; nullptr where the build has none for that instruction set.
const SimdKernels* simd_kernels_avx2();
const SimdKernels* simd_kernels_avx512();

}  // namespace vf
