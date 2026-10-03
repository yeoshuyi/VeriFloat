/* VeriFloat: the bit-accurate floating-point model as a C library.
 *
 * The same core as the Python package (same source, general kernel): every
 * function returns the result's encoding and its IEEE status flags. Meant
 * for SystemVerilog testbenches through DPI-C (sv/verifloat_pkg.sv) and for
 * C / C++ models.
 *
 * A format is named as the Python package prints it:
 *     "e8m23"                 IEEE binary32 (8 exponent bits, 23 mantissa bits)
 *     "e5m10, rtz"            binary16, rounding toward zero
 *     "e4m3, fn"              OCP E4M3 (finite + NaN)
 *     "ue4m3, bias=10, saturate, tininess=before"
 * Tags: bias=N, finite | fn, no-zero, saturate, wrap, ftz,
 *       rne | rna | rtz | rup | rdn | sr, sr_bits=N,
 *       canonical | propagate | x86 | arm, tininess=before.
 *
 * Encodings are raw codes [sign | exponent | mantissa] in the low bits of a
 * uint64_t; formats wider than 64 bits use the *128 functions (four 32-bit
 * words, least significant first: a SystemVerilog bit [127:0]) or, for any
 * width, the *_w functions.
 *
 * When the model has no answer (an invalid operation in a format without
 * NaN, an unknown format), the result is 0, `flags` is VF_ERROR, and
 * vf_last_error() has the reason. `flags` may be NULL.
 */
#ifndef VERIFLOAT_H
#define VERIFLOAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define VF_API __declspec(dllexport)
#else
#  define VF_API __attribute__((visibility("default")))
#endif

typedef struct vf_fmt vf_fmt;   /* owned by the library; valid until exit */

/* Status flags, in the bit order of RISC-V fflags. */
enum { VF_INEXACT = 1, VF_UNDERFLOW = 2, VF_OVERFLOW = 4, VF_DIVZERO = 8, VF_INVALID = 16, VF_ERROR = 256 };

/* Rounding modes for vf_to_int and vf_round_to_integral. */
enum { VF_ROUND_FORMAT = -1, VF_RNE = 0, VF_RNA = 1, VF_RTZ = 2, VF_RUP = 3, VF_RDN = 4, VF_SR = 5 };

/* vf_compare results. */
enum { VF_LT = 0, VF_EQ = 1, VF_GT = 2, VF_UNORDERED = 3 };

/* vf_fclass bits (RISC-V fclass). */
enum {
    VF_CLASS_NEG_INF = 1 << 0, VF_CLASS_NEG_NORMAL = 1 << 1, VF_CLASS_NEG_SUBNORMAL = 1 << 2,
    VF_CLASS_NEG_ZERO = 1 << 3, VF_CLASS_POS_ZERO = 1 << 4, VF_CLASS_POS_SUBNORMAL = 1 << 5,
    VF_CLASS_POS_NORMAL = 1 << 6, VF_CLASS_POS_INF = 1 << 7, VF_CLASS_SNAN = 1 << 8, VF_CLASS_QNAN = 1 << 9
};

/* Operations of vf_op and vf_op_w. */
enum {
    VF_OP_ADD, VF_OP_SUB, VF_OP_MUL, VF_OP_DIV, VF_OP_REM, VF_OP_FMOD,
    VF_OP_MIN, VF_OP_MAX, VF_OP_MINNUM, VF_OP_MAXNUM,
    VF_OP_SGNJ, VF_OP_SGNJN, VF_OP_SGNJX,
    VF_OP_FMA,
    VF_OP_SQRT, VF_OP_NEG, VF_OP_ABS, VF_OP_NEXT_UP, VF_OP_NEXT_DOWN, VF_OP_LOGB
};

/* ---- formats ---- */
VF_API const vf_fmt* vf_format(const char* name);      /* NULL on error */
VF_API int vf_format_size(const vf_fmt* f);            /* width in bits */
VF_API const char* vf_format_name(const vf_fmt* f);    /* canonical name */
VF_API const char* vf_last_error(void);                /* of this thread; "" if none */
VF_API const char* vf_version(void);

/* ---- arithmetic (formats of at most 64 bits) ---- */
VF_API uint64_t vf_add(const vf_fmt* f, uint64_t a, uint64_t b, int* flags);
VF_API uint64_t vf_sub(const vf_fmt* f, uint64_t a, uint64_t b, int* flags);
VF_API uint64_t vf_mul(const vf_fmt* f, uint64_t a, uint64_t b, int* flags);
VF_API uint64_t vf_div(const vf_fmt* f, uint64_t a, uint64_t b, int* flags);
VF_API uint64_t vf_fma(const vf_fmt* f, uint64_t a, uint64_t b, uint64_t c, int* flags);   /* a * b + c */
VF_API uint64_t vf_sqrt(const vf_fmt* f, uint64_t a, int* flags);
VF_API uint64_t vf_rem(const vf_fmt* f, uint64_t a, uint64_t b, int* flags);    /* IEEE remainder */
VF_API uint64_t vf_fmod(const vf_fmt* f, uint64_t a, uint64_t b, int* flags);   /* C fmod */

/* IEEE 754-2019 minimum / maximum, and minimumNumber / maximumNumber
 * (RISC-V fmin / fmax). */
VF_API uint64_t vf_min(const vf_fmt* f, uint64_t a, uint64_t b, int* flags);
VF_API uint64_t vf_max(const vf_fmt* f, uint64_t a, uint64_t b, int* flags);
VF_API uint64_t vf_minnum(const vf_fmt* f, uint64_t a, uint64_t b, int* flags);
VF_API uint64_t vf_maxnum(const vf_fmt* f, uint64_t a, uint64_t b, int* flags);

/* Sign operations: bits only, no flags raised. */
VF_API uint64_t vf_neg(const vf_fmt* f, uint64_t a, int* flags);
VF_API uint64_t vf_abs(const vf_fmt* f, uint64_t a, int* flags);
VF_API uint64_t vf_sgnj(const vf_fmt* f, uint64_t a, uint64_t b, int* flags);    /* a with b's sign */
VF_API uint64_t vf_sgnjn(const vf_fmt* f, uint64_t a, uint64_t b, int* flags);   /* a with the opposite of b's sign */
VF_API uint64_t vf_sgnjx(const vf_fmt* f, uint64_t a, uint64_t b, int* flags);   /* a's sign xor b's sign */

VF_API uint64_t vf_next_up(const vf_fmt* f, uint64_t a, int* flags);
VF_API uint64_t vf_next_down(const vf_fmt* f, uint64_t a, int* flags);
VF_API uint64_t vf_scaleb(const vf_fmt* f, uint64_t a, int64_t n, int* flags);   /* a * 2**n, |n| <= 2**62 */
VF_API uint64_t vf_logb(const vf_fmt* f, uint64_t a, int* flags);
VF_API int vf_fclass(const vf_fmt* f, uint64_t a);

/* Any of the operations above by number (VF_OP_*); unused operands are ignored. */
VF_API uint64_t vf_op(const vf_fmt* f, int op, uint64_t a, uint64_t b, uint64_t c, int* flags);

/* ---- comparisons ---- */
/* `signaling`: a quiet NaN operand raises VF_INVALID too. IEEE's usual
 * predicates are a quiet vf_eq and signaling vf_lt / vf_le. */
VF_API int vf_compare(const vf_fmt* f, uint64_t a, uint64_t b, int signaling, int* flags);   /* VF_LT ... */
VF_API int vf_eq(const vf_fmt* f, uint64_t a, uint64_t b, int signaling, int* flags);
VF_API int vf_lt(const vf_fmt* f, uint64_t a, uint64_t b, int signaling, int* flags);
VF_API int vf_le(const vf_fmt* f, uint64_t a, uint64_t b, int signaling, int* flags);

/* ---- conversions ---- */
VF_API uint64_t vf_convert(const vf_fmt* to, const vf_fmt* from, uint64_t a, int* flags);
/* rounding: VF_ROUND_FORMAT or a mode; exact: raise VF_INEXACT when the value changes. */
VF_API uint64_t vf_round_to_integral(const vf_fmt* f, uint64_t a, int rounding, int exact, int* flags);
/* The integer in the low `bits` bits (two's complement), 1 <= bits <= 64. */
VF_API uint64_t vf_to_int(const vf_fmt* f, uint64_t a, int bits, int is_signed, int rounding, int exact, int* flags);
VF_API uint64_t vf_from_int(const vf_fmt* f, int64_t value, int* flags);
VF_API uint64_t vf_from_uint(const vf_fmt* f, uint64_t value, int* flags);
VF_API uint64_t vf_from_double(const vf_fmt* f, double value, int* flags);
VF_API double vf_to_double(const vf_fmt* f, uint64_t a);   /* correctly rounded, ties to even */

/* ---- stochastic rounding ---- */
/* The random bits (sr_bits of them) each rounding of an "sr" format uses:
 * a fixed value until changed, or a callback. */
VF_API void vf_set_sr(uint64_t bits);
VF_API void vf_set_sr_source(uint64_t (*source)(int bits, void* ctx), void* ctx);

/* ---- any width: codes as 32-bit words, least significant first ---- */
/* Operands and results have as many words as the format needs,
 * (vf_format_size + 31) / 32. On an error the result is not written. */
VF_API void vf_op_w(const vf_fmt* f, int op, const uint32_t* a, const uint32_t* b, const uint32_t* c,
                    uint32_t* result, int* flags);
VF_API void vf_scaleb_w(const vf_fmt* f, const uint32_t* a, int64_t n, uint32_t* result, int* flags);
VF_API int vf_fclass_w(const vf_fmt* f, const uint32_t* a);
VF_API int vf_compare_w(const vf_fmt* f, const uint32_t* a, const uint32_t* b, int signaling, int* flags);
VF_API void vf_convert_w(const vf_fmt* to, const vf_fmt* from, const uint32_t* a, uint32_t* result, int* flags);
VF_API void vf_round_to_integral_w(const vf_fmt* f, const uint32_t* a, int rounding, int exact, uint32_t* result,
                                   int* flags);

/* ---- formats of at most 128 bits, as four words (SystemVerilog bit [127:0]) ---- */
/* The same operations with a fixed vector width: operands are read from the
 * format's low words, and all four result words are written, zero above the
 * format's width and zero on an error. */
VF_API void vf_op128(const vf_fmt* f, int op, const uint32_t* a, const uint32_t* b, const uint32_t* c,
                     uint32_t* result, int* flags);
VF_API void vf_scaleb128(const vf_fmt* f, const uint32_t* a, int64_t n, uint32_t* result, int* flags);
VF_API int vf_fclass128(const vf_fmt* f, const uint32_t* a);
VF_API int vf_compare128(const vf_fmt* f, const uint32_t* a, const uint32_t* b, int signaling, int* flags);
VF_API void vf_convert128(const vf_fmt* to, const vf_fmt* from, const uint32_t* a, uint32_t* result, int* flags);
VF_API void vf_round_to_integral128(const vf_fmt* f, const uint32_t* a, int rounding, int exact, uint32_t* result,
                                    int* flags);

#ifdef __cplusplus
}
#endif

#endif /* VERIFLOAT_H */
