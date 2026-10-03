// AVX-512 kernels (F + CD + DQ): 8 lanes of 64 bits. Compiled with -mavx512f
// -mavx512cd -mavx512dq (see CMakeLists.txt); see simd_api.hpp for why this file
// includes nothing else of the core.
#include "simd_api.hpp"

#if defined(VF_SIMD_AVX512)
#include <immintrin.h>

namespace vf {
namespace {

constexpr unsigned W = 8;

struct V {
    __m512i v;
    static V set1(uint64_t x) { return {_mm512_set1_epi64((long long)x)}; }
    static V load(const uint64_t* p) { return {_mm512_loadu_si512((const void*)p)}; }
    void store(uint64_t* p) const { _mm512_storeu_si512((void*)p, v); }
};
inline V operator+(V a, V b) { return {_mm512_add_epi64(a.v, b.v)}; }
inline V operator-(V a, V b) { return {_mm512_sub_epi64(a.v, b.v)}; }
inline V operator&(V a, V b) { return {_mm512_and_si512(a.v, b.v)}; }
inline V operator|(V a, V b) { return {_mm512_or_si512(a.v, b.v)}; }
inline V operator^(V a, V b) { return {_mm512_xor_si512(a.v, b.v)}; }
inline V shlv(V a, V n) { return {_mm512_sllv_epi64(a.v, n.v)}; }
inline V shrv(V a, V n) { return {_mm512_srlv_epi64(a.v, n.v)}; }
inline V mul32(V a, V b) { return {_mm512_mul_epu32(a.v, b.v)}; }
inline V abs64(V a) { return {_mm512_abs_epi64(a.v)}; }

// A mask is an 8-bit mask register (the 8-bit mask instructions are AVX512DQ).
struct M {
    __mmask8 v;
    static M none() { return {_cvtu32_mask8(0)}; }
    static M all() { return {_cvtu32_mask8(0xff)}; }
    static M from_bits(unsigned b) { return {_cvtu32_mask8(b)}; }
};
inline M kand(M a, M b) { return {_kand_mask8(a.v, b.v)}; }
inline M kor(M a, M b) { return {_kor_mask8(a.v, b.v)}; }
inline M kxor(M a, M b) { return {_kxor_mask8(a.v, b.v)}; }
inline M kandn(M a, M b) { return {_kandn_mask8(a.v, b.v)}; }   // ~a & b
inline M knot(M a) { return {_knot_mask8(a.v)}; }
inline M ksel(M c, M a, M b) { return {_kor_mask8(_kand_mask8(c.v, a.v), _kandn_mask8(c.v, b.v))}; }
inline M flip_if(M m, uint64_t bit) { return {_kxor_mask8(m.v, _cvtu32_mask8((unsigned)(0 - bit) & 0xff))}; }
inline M eq(V a, V b) { return {_mm512_cmpeq_epi64_mask(a.v, b.v)}; }
inline M gt(V a, V b) { return {_mm512_cmpgt_epi64_mask(a.v, b.v)}; }   // signed
inline M ge(V a, V b) { return {_mm512_cmpge_epi64_mask(a.v, b.v)}; }   // signed
inline M gt_half(V a, V half) { return {_mm512_cmpgt_epu64_mask(a.v, half.v)}; }
inline M eq0(V a) { return {_mm512_testn_epi64_mask(a.v, a.v)}; }
inline M nz(V a) { return {_mm512_test_epi64_mask(a.v, a.v)}; }
inline V select(M m, V a, V b) { return {_mm512_mask_blend_epi64(m.v, b.v, a.v)}; }
inline V one_if(M m) { return {_mm512_maskz_set1_epi64(m.v, 1)}; }
inline V inc_if(V a, M m) { return {_mm512_mask_sub_epi64(a.v, m.v, a.v, _mm512_set1_epi64(-1))}; }
inline V addsub(V a, V x, M neg) { return {_mm512_mask_sub_epi64(_mm512_add_epi64(a.v, x.v), neg.v, a.v, x.v)}; }
inline unsigned bits(M m) { return _cvtmask8_u32(m.v); }

// s shifted left until its leading bit is at 63, and the shift count.
inline V lznorm(V s, V& lz) {
    lz = V{_mm512_lzcnt_epi64(s.v)};
    return shlv(s, lz);
}

#define VF_SIMD_INLINE static inline __attribute__((always_inline))
#include "simd_kernels.inc"

const SimdKernels kTable = {"avx512", W, seq_lean_simd, pair_lean_simd, exact_lean_simd, ew_bin_simd, ew_conv_simd};

}  // namespace

const SimdKernels* simd_kernels_avx512() { return &kTable; }

}  // namespace vf
#else
namespace vf {
const SimdKernels* simd_kernels_avx512() { return nullptr; }
}  // namespace vf
#endif
