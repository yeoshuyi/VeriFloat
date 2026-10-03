// The operations of the FP type on plain encodings: codes and formats in,
// a code with its flags, rounding event and exact (unrounded) value out.
// No Python here: the extension wraps these results in FP objects and
// raises the warnings, the C library hands them to a simulator.
#pragma once

#include "kernel.hpp"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vf {

// An operand during an operation: an encoding in some format (a format the
// host owns, or an internal copy such as a widened operand's format).
struct Opnd {
    Code c;
    const Fmt* f = nullptr;
    void* fmt_obj = nullptr;   // the host's handle of the format, if any (borrowed)
};

// Exact value of a finite operand (u128: format must not be wide).
template <class U> inline Dy<U> exact_of(const Opnd& x) { return decode<U>(x.c, *x.f); }

// The infinitely precise result before rounding, kept as the exact dyadic
// operands of the operation (the host turns them into a rational on demand).
struct Term {   // only meaningful for the kinds that use it
    u128 sig = 0;
    int64_t exp = 0;
    bool neg = false;
};
struct WideTerms {
    Dy<BigInt> t[3];
};
struct UnrD {
    // OBJ belongs to the host (a number it holds itself).
    enum Kind : uint8_t { NONE, OBJ, POSINF, NEGINF, DY, ADD, MUL, FMA, DIV, DYB };
    uint8_t kind = NONE;
    Term t[3];              // DY: t[0]; ADD/MUL/DIV: t[0] op t[1]; FMA: t[0] * t[1] + t[2]
    Rc<Dy<BigInt>> big;     // DYB: a wide exact dyadic
    Rc<WideTerms> wide;     // DY..DIV whose terms do not fit 128 bits (then `t` is unused)
};

struct OpOut {
    RoundOut r;
    UnrD u;
};

// ---- format names and raw codes

// A format from its name, the inverse of fmt_str: "e8m23", "ue4m3, bias=10,
// fn, saturate, rtz", ... (FPFormat.parse; spaces around commas are free).
Fmt fmt_parse(const std::string& name);
// An encoding from / as the integer [sign | exponent | mantissa]; bits above
// the format's width are ignored.
Code code_from_raw(const BigInt& raw, const Fmt& f);
BigInt raw_of_code(const Code& c, const Fmt& f);

// ---- numbers

// A finite number as a sticky dyadic.
struct NumD {
    bool big = false;
    Dy<u128> s;
    Dy<BigInt> b;
    bool neg() const { return big ? b.neg : s.neg; }
    bool zero() const { return big ? b.is_zero() : s.is_zero(); }
};
void num_from_double(double d, NumD& n);
// num / den (den > 0, not a power of two) with `keep` bits plus sticky.
void num_from_ratio(const BigInt& num, const BigInt& den, int64_t keep, NumD& n);
RoundOut round_num(NumD& n, const Fmt& f, bool zero_sign, SRArg& sr);
// Round num / den (den > 0) into f: FP._round on an exact rational.
RoundOut round_ratio(const BigInt& num, const BigInt& den, const Fmt& f, bool zero_sign, SRArg& sr);

const Fmt& fp64_fmt();
// The value as a double (correctly rounded, ties to even).
double fp_to_double(const Code& c, const Fmt& f);

// ---- special values

// IEEE 754 6.3: the sign of an exact zero sum.
bool sum_zero_sign(bool sa, bool sb, const Fmt& f);
// This NaN re-encoded in T (canonical, or payload kept and quieted), and
// INVALID if it was signaling.
std::pair<Code, uint8_t> nan_to(const Opnd& x, const Fmt& T);
// The NaN an operation returns: propagated from `ops` by f's nan_mode.
RoundOut nan_result(const Fmt& f, const std::vector<const Opnd*>& ops, bool invalid);
RoundOut default_nan(const Fmt& f, uint8_t flags);
RoundOut inf_result(bool sign, const Fmt& f);
// A double inf/NaN into f. Sets `unr` to the infinity (NaN: none).
RoundOut special_in(double v, const Fmt& f, UnrD& unr);
// `c` with this sign; an error if the format is unsigned and the sign is set.
Code checked_code(bool sign, const Code& c, const Fmt& F);

// Re-encode exactly into a (wider) common format, keeping the sign. The
// copy uses gradual underflow and no wrap/ftz, so no operand is altered;
// the op result is still rounded with F's modes.
Opnd widen(const Opnd& x, const Fmt& F, std::optional<Fmt>& store);

// ---- operations
// Operands of op_arith, op_fma, op_minmax and the comparisons are exact in
// F: of format F itself, or widened into it.

// a op b ('+', '-', '*', '/') rounded into F, following IEEE 754 for
// inf/NaN operands.
OpOut op_arith(char op, const Opnd& a, const Opnd& b, const Fmt& F);
// a * b + c with one rounding.
OpOut op_fma(const Opnd& a, const Opnd& b, const Opnd& c, const Fmt& F);
// Square root in the operand's own format.
OpOut op_sqrt(const Opnd& x);
// -x in the operand's own format (an unsigned format rounds the negated value).
OpOut op_neg(const Opnd& x);
// IEEE 754-2019 9.6: minimum/maximum, or minimumNumber/maximumNumber.
OpOut op_minmax(const Opnd& a, const Opnd& b, const Fmt& F, bool pick_max, bool number);
// floor(sqrt(n)) for n >= 0, with integers only: the 64-bit, the 128-bit or
// the big-integer routine by the size of n, as the square root uses them.
BigInt isqrt_big(const BigInt& n);
// x - n*y for exact dyadics (y nonzero), where n is x/y rounded to an
// integer: to nearest with ties to even, or toward zero (`truncate`). The
// result is exact; a zero result keeps x's sign.
Dy<BigInt> rem_dy(const Dy<BigInt>& x, const Dy<BigInt>& y, bool truncate);
// IEEE 754 remainder x - n*y, n = x/y rounded to nearest (ties to even);
// with `truncate`, n is rounded toward zero (C fmod). Exact in a format
// with gradual underflow; otherwise rounded into F like any result.
OpOut op_rem(const Opnd& a, const Opnd& b, const Fmt& F, bool truncate);
// IEEE 754 nextUp / nextDown: the neighbouring value of the format. No
// flags (a signaling NaN: INVALID). Where the format has nothing further
// (no infinity, or below zero when unsigned), x itself.
OpOut op_next(const Opnd& x, bool up);
// IEEE 754 scaleB: x * 2**n rounded into the operand's format.
OpOut op_scaleb(const Opnd& x, int64_t n);
// IEEE 754 logB: floor(log2(|x|)) as a value of the operand's format;
// -inf with DIVZERO for zero, +inf for an infinity.
OpOut op_logb(const Opnd& x);
// RISC-V fclass: one bit set of -inf, negative normal, negative subnormal,
// -0, +0, positive subnormal, positive normal, +inf, signaling NaN, quiet
// NaN (bits 0 to 9). A subnormal that reads as zero (ftz) is a zero.
unsigned op_fclass(const Code& c, const Fmt& f);
// Sign injection (RISC-V fsgnj, fsgnjn, fsgnjx): x with the sign `sign`
// (mode 0), its opposite (1), or x's sign xor `sign` (2). Bits only: no
// flags, NaNs are not quieted.
Code op_sgnj(const Opnd& x, bool sign, int mode);
// Ordering of two non-NaN operands: -1, 0, 1.
int cmp_keys(const Opnd& a, const Opnd& b);
// IEEE comparison: (0 lt, 1 eq, 2 gt, 3 unordered; flags).
std::pair<int, uint8_t> op_compare(const Opnd& a, const Opnd& b, bool signaling);
// x converted into F (sr: random bits when F rounds stochastically).
OpOut op_convert(const Opnd& x, const Fmt& F, SRArg& sr);
// A double into F.
OpOut op_from_double(double v, const Fmt& F, SRArg& sr);

// The exact value of a finite x rounded to an integer: (q, inexact).
std::pair<BigInt, bool> round_to_int(const Code& c, const Fmt& f, uint8_t rounding, const SRArg& sr);
// x as an integer of `bits` bits: the value (as invalid results the mode's
// substitute) and INVALID, or INEXACT when asked for with want_exact.
struct IntOut {
    BigInt v;
    uint8_t flags = 0;
};
IntOut op_to_int(const Opnd& x, int64_t bits, bool is_signed, uint8_t rounding, bool want_exact, SRArg& sr);
// IEEE roundToIntegral in the operand's own format.
OpOut op_round_to_integral(const Opnd& x, uint8_t rounding, bool want_exact, SRArg& sr);

}  // namespace vf
