// AVX2 kernels: 4 lanes of 64 bits. Compiled with -mavx2 (see CMakeLists.txt);
// see simd_api.hpp for why this file includes nothing else of the core.
#include "simd_api.hpp"

#if defined(VF_SIMD_AVX2)
#include <immintrin.h>

namespace vf {
namespace {

constexpr unsigned W = 4;

struct V {
    __m256i v;
    static V set1(uint64_t x) { return {_mm256_set1_epi64x((long long)x)}; }
    static V load(const uint64_t* p) { return {_mm256_loadu_si256((const __m256i*)p)}; }
    void store(uint64_t* p) const { _mm256_storeu_si256((__m256i*)p, v); }
};
inline V operator+(V a, V b) { return {_mm256_add_epi64(a.v, b.v)}; }
inline V operator-(V a, V b) { return {_mm256_sub_epi64(a.v, b.v)}; }
inline V operator&(V a, V b) { return {_mm256_and_si256(a.v, b.v)}; }
inline V operator|(V a, V b) { return {_mm256_or_si256(a.v, b.v)}; }
inline V operator^(V a, V b) { return {_mm256_xor_si256(a.v, b.v)}; }
inline V shlv(V a, V n) { return {_mm256_sllv_epi64(a.v, n.v)}; }
inline V shrv(V a, V n) { return {_mm256_srlv_epi64(a.v, n.v)}; }
inline V mul32(V a, V b) { return {_mm256_mul_epu32(a.v, b.v)}; }
inline V abs64(V a) {
    const __m256i m = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a.v);
    return {_mm256_sub_epi64(_mm256_xor_si256(a.v, m), m)};
}

// A mask is a vector with all bits of a lane set or clear.
struct M {
    __m256i v;
    static M none() { return {_mm256_setzero_si256()}; }
    static M all() { return {_mm256_set1_epi64x(-1)}; }
    static M from_bits(unsigned b) {
        const __m256i lane = _mm256_set_epi64x(8, 4, 2, 1);
        return {_mm256_cmpeq_epi64(_mm256_and_si256(_mm256_set1_epi64x((long long)b), lane), lane)};
    }
};
inline M kand(M a, M b) { return {_mm256_and_si256(a.v, b.v)}; }
inline M kor(M a, M b) { return {_mm256_or_si256(a.v, b.v)}; }
inline M kxor(M a, M b) { return {_mm256_xor_si256(a.v, b.v)}; }
inline M kandn(M a, M b) { return {_mm256_andnot_si256(a.v, b.v)}; }   // ~a & b
inline M knot(M a) { return {_mm256_xor_si256(a.v, _mm256_set1_epi64x(-1))}; }
inline M ksel(M c, M a, M b) { return {_mm256_blendv_epi8(b.v, a.v, c.v)}; }
inline M flip_if(M m, uint64_t bit) { return {_mm256_xor_si256(m.v, _mm256_set1_epi64x(-(long long)bit))}; }
inline M eq(V a, V b) { return {_mm256_cmpeq_epi64(a.v, b.v)}; }
inline M gt(V a, V b) { return {_mm256_cmpgt_epi64(a.v, b.v)}; }   // signed
inline M ge(V a, V b) { return knot(gt(b, a)); }                    // signed
inline M gt_half(V a, V half) { return {_mm256_cmpgt_epi64(_mm256_xor_si256(a.v, half.v), _mm256_setzero_si256())}; }
inline M eq0(V a) { return {_mm256_cmpeq_epi64(a.v, _mm256_setzero_si256())}; }
inline M nz(V a) { return knot(eq0(a)); }
inline V select(M m, V a, V b) { return {_mm256_blendv_epi8(b.v, a.v, m.v)}; }
inline V one_if(M m) { return {_mm256_srli_epi64(m.v, 63)}; }
inline V inc_if(V a, M m) { return {_mm256_sub_epi64(a.v, m.v)}; }   // a set lane is -1
inline V addsub(V a, V x, M neg) {   // (x ^ m) - m negates x where m is set
    return {_mm256_add_epi64(a.v, _mm256_sub_epi64(_mm256_xor_si256(x.v, neg.v), neg.v))};
}
inline unsigned bits(M m) { return (unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(m.v)); }

// s shifted left until its leading bit is at 63, and the shift count (64 and
// 0 for s == 0). AVX2 has no leading-zero count; the count of each 32-bit
// half comes from the exponent of its conversion to float. Clearing the bits
// of x that have a set bit 8 places above them keeps the leading bit and
// makes sure the conversion cannot round up to the next power of two.
inline V lznorm(V s, V& lz) {
    const __m256i x = s.v;
    const __m256i t = _mm256_andnot_si256(_mm256_srli_epi32(x, 8), x);
    const __m256i e = _mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(t)), 23);
    __m256i n32 = _mm256_sub_epi32(_mm256_set1_epi32(158), e);        // 31 - floor(log2), 158 for 0
    n32 = _mm256_andnot_si256(_mm256_srai_epi32(x, 31), n32);         // bit 31 set: 0 (the float was negative)
    n32 = _mm256_min_epu32(n32, _mm256_set1_epi32(32));
    const __m256i hi = _mm256_srli_epi64(n32, 32);                    // count of the high half
    const __m256i lo = _mm256_and_si256(n32, _mm256_set1_epi64x(0xffffffff));
    const __m256i hi_empty = _mm256_cmpeq_epi64(hi, _mm256_set1_epi64x(32));
    const __m256i n = _mm256_add_epi64(hi, _mm256_and_si256(hi_empty, lo));
    lz = V{n};
    return {_mm256_sllv_epi64(x, n)};
}

#define VF_SIMD_INLINE static inline __attribute__((always_inline))
#include "simd_kernels.inc"

const SimdKernels kTable = {"avx2", W, seq_lean_simd, pair_lean_simd, exact_lean_simd, ew_bin_simd, ew_conv_simd};

}  // namespace

const SimdKernels* simd_kernels_avx2() { return &kTable; }

}  // namespace vf
#else
namespace vf {
const SimdKernels* simd_kernels_avx2() { return nullptr; }
}  // namespace vf
#endif
