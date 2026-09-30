/* One SSE/FMA operation under a given MXCSR (rounding, FTZ, DAZ); returns
   the result bits and the MXCSR exception flags. Built at test time. */
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

enum { OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_SQRT, OP_FMA };

static unsigned run_setup(unsigned rc, int ftz, int daz) {
    unsigned csr = 0x1F80u;                 /* all exceptions masked */
    csr |= (rc & 3u) << 13;                 /* RC: 0 RNE, 1 down, 2 up, 3 RTZ */
    if (ftz) csr |= 1u << 15;
    if (daz) csr |= 1u << 6;
    return csr;
}

__attribute__((target("fma")))
uint32_t op32(int op, uint32_t a, uint32_t b, uint32_t c, unsigned rc,
              int ftz, int daz, unsigned *flags) {
    float fa, fb, fc; memcpy(&fa, &a, 4); memcpy(&fb, &b, 4); memcpy(&fc, &c, 4);
    unsigned old = _mm_getcsr();
    _mm_setcsr(run_setup(rc, ftz, daz));
    __m128 va = _mm_set_ss(fa), vb = _mm_set_ss(fb), vc = _mm_set_ss(fc), r;
    switch (op) {
        case OP_ADD: r = _mm_add_ss(va, vb); break;
        case OP_SUB: r = _mm_sub_ss(va, vb); break;
        case OP_MUL: r = _mm_mul_ss(va, vb); break;
        case OP_DIV: r = _mm_div_ss(va, vb); break;
        case OP_SQRT: r = _mm_sqrt_ss(va); break;
        default: r = _mm_fmadd_ss(va, vb, vc); break;
    }
    *flags = _mm_getcsr() & 0x3Fu;
    _mm_setcsr(old);
    float fr = _mm_cvtss_f32(r); uint32_t out; memcpy(&out, &fr, 4);
    return out;
}

__attribute__((target("fma")))
uint64_t op64(int op, uint64_t a, uint64_t b, uint64_t c, unsigned rc,
              int ftz, int daz, unsigned *flags) {
    double fa, fb, fc; memcpy(&fa, &a, 8); memcpy(&fb, &b, 8); memcpy(&fc, &c, 8);
    unsigned old = _mm_getcsr();
    _mm_setcsr(run_setup(rc, ftz, daz));
    __m128d va = _mm_set_sd(fa), vb = _mm_set_sd(fb), vc = _mm_set_sd(fc), r;
    switch (op) {
        case OP_ADD: r = _mm_add_sd(va, vb); break;
        case OP_SUB: r = _mm_sub_sd(va, vb); break;
        case OP_MUL: r = _mm_mul_sd(va, vb); break;
        case OP_DIV: r = _mm_div_sd(va, vb); break;
        case OP_SQRT: r = _mm_sqrt_sd(va, va); break;
        default: r = _mm_fmadd_sd(va, vb, vc); break;
    }
    *flags = _mm_getcsr() & 0x3Fu;
    _mm_setcsr(old);
    double fr = _mm_cvtsd_f64(r); uint64_t out; memcpy(&out, &fr, 8);
    return out;
}

int has_fma(void) { return __builtin_cpu_supports("fma"); }
