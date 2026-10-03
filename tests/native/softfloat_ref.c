/* Berkeley SoftFloat on operands of our choosing: one operation, rounding
   mode and tininess convention per run; operands (hex) on stdin, one case a
   line, "result flags" (hex) on stdout. Built at test time against the
   SoftFloat library that testfloat_build.py builds.

       softfloat_ref <width> <op> <rounding> <tininess> <exact>  < operands

   width     16, 32, 64 or 128
   op        add sub mul div rem sqrt mulAdd roundToInt
             eq le lt eq_signaling le_quiet lt_quiet
             to_i32 to_ui32 to_i64 to_ui64 (float to integer)
             from_i32 from_ui32 from_i64 from_ui64 (integer to this width)
             to_f16 to_f32 to_f64 to_f128 (float to float)
   rounding  0 near_even, 1 minMag, 2 min, 3 max, 4 near_maxMag
   tininess  0 before rounding, 1 after rounding
   exact     roundToInt and to_<int>: raise inexact */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "softfloat.h"

typedef struct { uint64_t lo, hi; } V;

static V parse(const char *s) {
    V v = {0, 0};
    size_t n = strlen(s);
    if (n > 16) {
        char buf[32];
        memcpy(buf, s, n - 16);
        buf[n - 16] = 0;
        v.hi = strtoull(buf, NULL, 16);
        v.lo = strtoull(s + n - 16, NULL, 16);
    } else {
        v.lo = strtoull(s, NULL, 16);
    }
    return v;
}

static float16_t in16(V x) { float16_t r; r.v = (uint16_t)x.lo; return r; }
static float32_t in32(V x) { float32_t r; r.v = (uint32_t)x.lo; return r; }
static float64_t in64(V x) { float64_t r; r.v = x.lo; return r; }
/* SoftFloat keeps the two words of a float128_t in the host's order. */
static int big_endian(void) { const uint16_t probe = 1; return *(const unsigned char *)&probe != 1; }
static float128_t in128(V x) {
    float128_t r;
    r.v[big_endian() ? 1 : 0] = x.lo;
    r.v[big_endian() ? 0 : 1] = x.hi;
    return r;
}
static V out16(float16_t r) { V v = {r.v, 0}; return v; }
static V out32(float32_t r) { V v = {r.v, 0}; return v; }
static V out64(float64_t r) { V v = {r.v, 0}; return v; }
static V out128(float128_t r) { V v = {r.v[big_endian() ? 1 : 0], r.v[big_endian() ? 0 : 1]}; return v; }
static V outi(uint64_t x) { V v = {x, 0}; return v; }

#define IS(name) (!strcmp(op, name))
#define DEFINE(W)                                                                              \
    static bool run##W(const char *op, V a, V b, V c, bool exact, V *out) {                    \
        float##W##_t x = in##W(a), y = in##W(b), z = in##W(c);                                 \
        uint_fast8_t rm = softfloat_roundingMode;                                              \
        if (IS("add")) *out = out##W(f##W##_add(x, y));                                        \
        else if (IS("sub")) *out = out##W(f##W##_sub(x, y));                                   \
        else if (IS("mul")) *out = out##W(f##W##_mul(x, y));                                   \
        else if (IS("div")) *out = out##W(f##W##_div(x, y));                                   \
        else if (IS("rem")) *out = out##W(f##W##_rem(x, y));                                   \
        else if (IS("sqrt")) *out = out##W(f##W##_sqrt(x));                                    \
        else if (IS("mulAdd")) *out = out##W(f##W##_mulAdd(x, y, z));                          \
        else if (IS("roundToInt")) *out = out##W(f##W##_roundToInt(x, rm, exact));             \
        else if (IS("eq")) *out = outi(f##W##_eq(x, y));                                       \
        else if (IS("le")) *out = outi(f##W##_le(x, y));                                       \
        else if (IS("lt")) *out = outi(f##W##_lt(x, y));                                       \
        else if (IS("eq_signaling")) *out = outi(f##W##_eq_signaling(x, y));                   \
        else if (IS("le_quiet")) *out = outi(f##W##_le_quiet(x, y));                           \
        else if (IS("lt_quiet")) *out = outi(f##W##_lt_quiet(x, y));                           \
        else if (IS("to_i32")) *out = outi((uint32_t)f##W##_to_i32(x, rm, exact));             \
        else if (IS("to_ui32")) *out = outi((uint32_t)f##W##_to_ui32(x, rm, exact));           \
        else if (IS("to_i64")) *out = outi((uint64_t)f##W##_to_i64(x, rm, exact));             \
        else if (IS("to_ui64")) *out = outi((uint64_t)f##W##_to_ui64(x, rm, exact));           \
        else if (IS("from_i32")) *out = out##W(i32_to_f##W((int32_t)a.lo));                    \
        else if (IS("from_ui32")) *out = out##W(ui32_to_f##W((uint32_t)a.lo));                 \
        else if (IS("from_i64")) *out = out##W(i64_to_f##W((int64_t)a.lo));                    \
        else if (IS("from_ui64")) *out = out##W(ui64_to_f##W(a.lo));                           \
        else return false;                                                                     \
        return true;                                                                           \
    }
DEFINE(16)
DEFINE(32)
DEFINE(64)
DEFINE(128)

static bool convert(int w, const char *op, V a, V *out) {
    int to = atoi(op + 4);
    if (strncmp(op, "to_f", 4) || to == w) return false;
#define CONV(S, D) if (w == S && to == D) { *out = out##D(f##S##_to_f##D(in##S(a))); return true; }
    CONV(16, 32) CONV(16, 64) CONV(16, 128) CONV(32, 16) CONV(32, 64) CONV(32, 128)
    CONV(64, 16) CONV(64, 32) CONV(64, 128) CONV(128, 16) CONV(128, 32) CONV(128, 64)
    return false;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: softfloat_ref width op rounding tininess exact\n");
        return 2;
    }
    int w = atoi(argv[1]);
    const char *op = argv[2];
    softfloat_roundingMode = (uint_fast8_t)atoi(argv[3]);
    softfloat_detectTininess = (uint_fast8_t)atoi(argv[4]);
    bool exact = atoi(argv[5]) != 0;
    char sa[64], sb[64], sc[64];
    while (scanf("%63s %63s %63s", sa, sb, sc) == 3) {
        V a = parse(sa), b = parse(sb), c = parse(sc), r = {0, 0};
        softfloat_exceptionFlags = 0;
        bool ok = convert(w, op, a, &r);
        if (!ok) {
            switch (w) {
                case 16: ok = run16(op, a, b, c, exact, &r); break;
                case 32: ok = run32(op, a, b, c, exact, &r); break;
                case 64: ok = run64(op, a, b, c, exact, &r); break;
                case 128: ok = run128(op, a, b, c, exact, &r); break;
            }
        }
        if (!ok) {
            fprintf(stderr, "softfloat_ref: no operation %s for width %d\n", op, w);
            return 2;
        }
        if (r.hi) printf("%llx%016llx %x\n", (unsigned long long)r.hi, (unsigned long long)r.lo,
                         (unsigned)softfloat_exceptionFlags);
        else printf("%llx %x\n", (unsigned long long)r.lo, (unsigned)softfloat_exceptionFlags);
    }
    return 0;
}
