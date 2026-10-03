/* Puts the processor's floating-point state where other code in a process
   can leave it, for tests/test_fpstate.py. x86-64 only. Built at test time. */
#include <immintrin.h>

unsigned get_mxcsr(void) { return _mm_getcsr(); }
void set_mxcsr(unsigned v) { _mm_setcsr(v); }

unsigned short get_x87_cw(void) {
    unsigned short cw;
    __asm__("fnstcw %0" : "=m"(cw));
    return cw;
}
void set_x87_cw(unsigned short cw) { __asm__("fldcw %0" : : "m"(cw)); }

/* An MMX instruction without EMMS after it: every x87 register reads as in
   use, and the next x87 load gives NaN. */
void dirty_mmx(void) { __asm__("pxor %mm0, %mm0"); }
void clean_mmx(void) { __asm__("emms"); }
