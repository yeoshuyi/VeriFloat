// Shared types for the VeriFloat C++ core.
//
// Every finite value the library handles is a dyadic rational sig * 2**exp.
// Arithmetic produces a "sticky dyadic": a significand with enough bits for
// correct rounding plus a sticky flag for anything nonzero below it. The
// rounding kernel and the operations are templated on the significand type:
// a 128-bit integer for the common case, cpp_int when a format is too wide.
#pragma once

#include <boost/multiprecision/cpp_int.hpp>

// The standard headers every file of the core relies on are included here,
// not left to what another header happens to pull in (that differs between
// standard libraries and their versions).
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// MinGW's math.h defines the old System V matherr codes as macros unless the
// compiler is in strict ISO mode (-std=c++20 rather than gnu++20). Two of them
// collide with the names below; math.h is include-guarded, so removing them
// here, after <cmath>, keeps them out for good.
#undef OVERFLOW
#undef UNDERFLOW

#include "int128.hpp"

namespace vf {

// The core (this header, kernel.hpp, ops.hpp, the fast kernels) is plain
// C++ with no Python in it. Its host - the Python extension, or the C
// library - supplies the two things the core cannot do alone: reporting an
// error to the caller, and drawing random bits for stochastic rounding
// (sr_draw, kernel.hpp).
enum class Err : uint8_t { VALUE, ZERO_DIVISION, OVERFLOW, TYPE, RUNTIME };
[[noreturn]] void fail(Err kind, const std::string& msg);

// The compiler's 128-bit integers where it has them (GCC, Clang), the
// portable ones of int128.hpp elsewhere (MSVC) or when VF_PORTABLE_INT128 asks
// for them, to test them. Code that converts between u128, i128 and narrower
// integers casts explicitly, so that it compiles with either.
#if defined(__SIZEOF_INT128__) && !defined(VF_PORTABLE_INT128)
using u128 = unsigned __int128;
using i128 = __int128;
#else
using u128 = U128;
using i128 = I128;
#endif
using BigInt = boost::multiprecision::cpp_int;

// Widest integer and fixed-point formats (bits), as mantissa_bits is capped
// for FP formats: wider ones would take memory without bound.
constexpr int64_t kMaxIntBits = int64_t(1) << 24;

// Widest intermediate the u128 tier may hold.
constexpr int64_t kU128Bits = 126;

// Bit length (0 for 0).
inline int64_t bitlen(uint64_t x) { return 64 - std::countl_zero(x); }
inline int64_t bitlen(u128 x) {
    uint64_t hi = (uint64_t)(x >> 64);
    return hi ? 128 - std::countl_zero(hi) : bitlen((uint64_t)x);
}
inline int64_t bitlen(const BigInt& x) {
    return x == 0 ? 0 : (int64_t)boost::multiprecision::msb(x) + 1;
}

// ---- doubles, by their bits
// The core reads and writes doubles through their encoding, never with a
// floating-point instruction, so that the processor's floating-point state
// cannot change a result: not the rounding mode, not a disturbed x87 state,
// and not the flush-to-zero / denormals-are-zero bits, which a library built
// with -ffast-math switches on for the whole process when it is loaded.
inline uint64_t double_bits(double d) {
    uint64_t b;
    std::memcpy(&b, &d, 8);
    return b;
}
inline double double_of_bits(uint64_t b) {
    double d;
    std::memcpy(&d, &b, 8);
    return d;
}
inline bool double_finite(double d) { return ((double_bits(d) >> 52) & 0x7ff) != 0x7ff; }
inline bool double_neg(double d) { return double_bits(d) >> 63; }
// A finite double as sig * 2**exp with bit 52 of sig set; zero gives 0 * 2**0.
inline void double_parts(double d, uint64_t& sig, int64_t& exp) {
    const uint64_t b = double_bits(d), field = (b >> 52) & 0x7ff, mant = b & ((uint64_t(1) << 52) - 1);
    if (field) {
        sig = mant | (uint64_t(1) << 52);
        exp = (int64_t)field - 1075;
    } else if (mant) {                      // subnormal: normalized like the others
        const int up = 53 - (int)bitlen(mant);
        sig = mant << up;
        exp = -1074 - up;
    } else {
        sig = 0;
        exp = 0;
    }
}

// Shifts that tolerate any count.
inline u128 shl(u128 x, int64_t k) { return k >= 128 ? 0 : x << k; }
inline u128 shr(u128 x, int64_t k) { return k >= 128 ? 0 : x >> k; }
inline BigInt shl(const BigInt& x, int64_t k) { return x << (unsigned)k; }
inline BigInt shr(const BigInt& x, int64_t k) { return x >> (unsigned)k; }

// Low k bits nonzero?
inline bool low_nonzero(u128 x, int64_t k) {
    if (k <= 0) return false;
    if (k >= 128) return x != 0;
    return (x & ((u128(1) << k) - 1)) != 0;
}
inline bool low_nonzero(const BigInt& x, int64_t k) {
    if (k <= 0 || x == 0) return false;
    return (int64_t)boost::multiprecision::lsb(x) < k;
}
inline bool bit(u128 x, int64_t k) { return k < 128 && ((x >> k) & 1); }
inline bool bit(const BigInt& x, int64_t k) { return boost::multiprecision::bit_test(x, (unsigned)k); }

template <class U> inline U pow2(int64_t k) { return shl(U(1), k); }
inline uint64_t mask64(int64_t bits) { return bits >= 64 ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1); }

inline BigInt to_big(u128 x) {
    BigInt r = (uint64_t)(x >> 64);
    r <<= 64;
    r |= (uint64_t)x;
    return r;
}
inline const BigInt& to_big(const BigInt& x) { return x; }
inline u128 to_u128(const BigInt& x) {
    return (u128(static_cast<uint64_t>(x >> 64)) << 64) | static_cast<uint64_t>(x & BigInt(~uint64_t(0)));
}

template <class U> U from_u64(uint64_t v) { return U(v); }

// A shared, immutable heap value with an inline, non-atomic reference count
// (all access happens under the GIL). Copying a null Rc costs nothing, which
// keeps the common case (no wide mantissa) free of library calls.
template <class T> class Rc {
    struct Node {
        T v;
        long n;
    };
    Node* p = nullptr;

public:
    Rc() = default;
    explicit Rc(T v) : p(new Node{std::move(v), 1}) {}
    Rc(const Rc& o) : p(o.p) { if (p) ++p->n; }
    Rc(Rc&& o) noexcept : p(o.p) { o.p = nullptr; }
    Rc& operator=(Rc o) noexcept { std::swap(p, o.p); return *this; }
    ~Rc() { if (p && --p->n == 0) delete p; }
    explicit operator bool() const { return p != nullptr; }
    const T& operator*() const { return p->v; }
    const T* operator->() const { return &p->v; }
};

// Format enums (values match the Python enum order).
enum Round : uint8_t { RNE, RNA, RTZ, RUP, RDN, SR };
enum NanMode : uint8_t { CANONICAL, PROPAGATE, X86, ARM };
enum InfNan : uint8_t { FINITE = 0, IEEE = 1, FN = 2 };

enum Flag : uint8_t { INEXACT = 1, UNDERFLOW = 2, OVERFLOW = 4, DIVZERO = 8, INVALID = 16 };

// Rounding events that may raise a warning (see fp.py _EVENTS).
enum Event : uint8_t {
    EV_NONE, EV_SATURATED, EV_OVERFLOWED, EV_WRAPPED_UP, EV_WRAPPED_DOWN,
    EV_CLAMPED, EV_UNDERFLOWED, EV_FLUSHED
};

// A complete FP format with precomputed geometry. It is the native base of
// the Python FPFormat dataclass, and is also used for internal formats that
// never reach Python (widening copies, cast targets).
struct Fmt {
    int64_t E = 2, M = 1, bias = 0;
    bool is_signed = true, has_zero = true, saturate = false, wrap = false, ftz = false;
    uint8_t inf_nan = IEEE;
    uint8_t rounding = RNE;
    int64_t sr_bits = 8;
    uint8_t nan_mode = CANONICAL;
    bool tininess_before = false;

    // Derived
    int64_t id = -1;          // interned id of the Python format (-1: internal)
    bool wide = false;        // M > 64: mantissas are stored as BigInt
    uint64_t top = 0;         // all-ones exponent field
    uint64_t max_field = 0;
    bool max_mant_is_mask = true;   // else mask - 1 ('fn' formats with M > 0)
    int64_t emin = 0, emax = 0;
    int64_t size = 0;
    int64_t prec = 0;         // significand bits rounding needs (incl. SR bits)
    bool fast = false;        // a target of the fast kernels (fast.hpp)

    bool has_inf() const { return inf_nan == IEEE; }
    bool has_nan() const { return inf_nan != FINITE; }
    int64_t min_field() const { return has_zero ? 1 : 0; }
    uint64_t mask() const { return mask64(M); }

    void derive() {
        wide = M > 64;
        top = mask64(E);
        // max code = all ones minus the reserved inf/NaN codes
        if (inf_nan == IEEE) { max_field = top - 1; max_mant_is_mask = true; }
        else if (inf_nan == FN) {
            if (M > 0) { max_field = top; max_mant_is_mask = false; }
            else { max_field = top - 1; max_mant_is_mask = true; }
        } else { max_field = top; max_mant_is_mask = true; }
        emin = min_field() - bias;
        emax = (int64_t)max_field - bias;
        size = (is_signed ? 1 : 0) + E + M;
        prec = M + 4 + (rounding == SR ? sr_bits : 0);
        fast = is_signed && has_zero && rounding != SR && M >= 1 && M <= 61 && size <= 64;
    }
};

// Sticky dyadic: value is (-1)**neg * (sig + (sticky ? half-open (0,1) : 0)) * 2**exp.
template <class U> struct Dy {
    bool neg = false;
    int64_t exp = 0;
    U sig = 0;
    bool sticky = false;
    bool is_zero() const { return sig == 0 && !sticky; }
};

}  // namespace vf
