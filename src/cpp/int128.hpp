// 128-bit integers for compilers without __int128 (MSVC).
//
// U128 and I128 behave like unsigned __int128 and __int128: arithmetic wraps
// modulo 2**128, I128 is two's complement, >> on I128 is arithmetic, and
// division truncates toward zero. They are written in plain 64-bit integer
// C++ with no intrinsics, so the same code runs on every compiler; building
// with VF_PORTABLE_INT128 makes GCC and Clang use it too, which is how it is
// tested (the whole suite and the golden corpus run against it on Linux).
//
// Conversions that would narrow or change signedness are explicit, unlike
// with the built-in types; the core spells those out with casts.
#pragma once

#include <bit>
#include <cstdint>
#include <type_traits>

namespace vf {

template <bool Signed> class Int128 {
public:
    uint64_t lo = 0, hi = 0;

    constexpr Int128() = default;
    // From any integer type: sign-extended if that type is signed.
    template <class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    constexpr Int128(T v)
        : lo((uint64_t)v), hi(std::is_signed_v<T> && v < 0 ? ~uint64_t(0) : 0) {}
    // Between the signed and the unsigned type: the same 128 bits.
    constexpr explicit Int128(const Int128<!Signed>& o) : lo(o.lo), hi(o.hi) {}
    static constexpr Int128 make(uint64_t h, uint64_t l) {
        Int128 r;
        r.hi = h;
        r.lo = l;
        return r;
    }

    // To an integer type: the low bits, as a cast of the built-in type does.
    template <class T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int> = 0>
    constexpr explicit operator T() const { return (T)lo; }
    constexpr explicit operator bool() const { return (lo | hi) != 0; }

    constexpr bool negative() const { return Signed && (hi >> 63); }

    // ---- bitwise
    friend constexpr Int128 operator~(Int128 a) { return make(~a.hi, ~a.lo); }
    friend constexpr Int128 operator&(Int128 a, Int128 b) { return make(a.hi & b.hi, a.lo & b.lo); }
    friend constexpr Int128 operator|(Int128 a, Int128 b) { return make(a.hi | b.hi, a.lo | b.lo); }
    friend constexpr Int128 operator^(Int128 a, Int128 b) { return make(a.hi ^ b.hi, a.lo ^ b.lo); }

    template <class K, std::enable_if_t<std::is_integral_v<K>, int> = 0>
    friend constexpr Int128 operator<<(Int128 a, K k) {
        const unsigned n = (unsigned)k & 127;
        if (n == 0) return a;
        if (n >= 64) return make(a.lo << (n - 64), 0);
        return make((a.hi << n) | (a.lo >> (64 - n)), a.lo << n);
    }
    template <class K, std::enable_if_t<std::is_integral_v<K>, int> = 0>
    friend constexpr Int128 operator>>(Int128 a, K k) {
        const unsigned n = (unsigned)k & 127;
        const uint64_t fill = a.negative() ? ~uint64_t(0) : 0;
        if (n == 0) return a;
        if (n >= 64) {
            const unsigned m = n - 64;
            return make(fill, m ? (a.hi >> m) | (fill << (64 - m)) : a.hi);
        }
        return make((a.hi >> n) | (fill << (64 - n)), (a.lo >> n) | (a.hi << (64 - n)));
    }

    // ---- arithmetic (modulo 2**128)
    friend constexpr Int128 operator+(Int128 a, Int128 b) {
        const uint64_t lo = a.lo + b.lo;
        return make(a.hi + b.hi + (lo < a.lo), lo);
    }
    friend constexpr Int128 operator-(Int128 a, Int128 b) {
        return make(a.hi - b.hi - (a.lo < b.lo), a.lo - b.lo);
    }
    friend constexpr Int128 operator-(Int128 a) { return Int128(0) - a; }
    friend constexpr Int128 operator+(Int128 a) { return a; }
    friend constexpr Int128 operator*(Int128 a, Int128 b) {
        Int128 r = mul64(a.lo, b.lo);
        r.hi += a.lo * b.hi + a.hi * b.lo;
        return r;
    }
    friend constexpr Int128 operator/(Int128 a, Int128 b) { return divmod(a, b, false); }
    friend constexpr Int128 operator%(Int128 a, Int128 b) { return divmod(a, b, true); }

    // ---- comparison
    friend constexpr bool operator==(Int128 a, Int128 b) { return a.lo == b.lo && a.hi == b.hi; }
    friend constexpr bool operator<(Int128 a, Int128 b) {
        const uint64_t flip = Signed ? uint64_t(1) << 63 : 0;
        return (a.hi ^ flip) != (b.hi ^ flip) ? (a.hi ^ flip) < (b.hi ^ flip) : a.lo < b.lo;
    }
    friend constexpr bool operator!=(Int128 a, Int128 b) { return !(a == b); }
    friend constexpr bool operator>(Int128 a, Int128 b) { return b < a; }
    friend constexpr bool operator<=(Int128 a, Int128 b) { return !(b < a); }
    friend constexpr bool operator>=(Int128 a, Int128 b) { return !(a < b); }

    // ---- compound assignment
    template <class T> constexpr Int128& operator+=(const T& b) { return *this = *this + Int128(b); }
    template <class T> constexpr Int128& operator-=(const T& b) { return *this = *this - Int128(b); }
    template <class T> constexpr Int128& operator*=(const T& b) { return *this = *this * Int128(b); }
    template <class T> constexpr Int128& operator/=(const T& b) { return *this = *this / Int128(b); }
    template <class T> constexpr Int128& operator%=(const T& b) { return *this = *this % Int128(b); }
    template <class T> constexpr Int128& operator&=(const T& b) { return *this = *this & Int128(b); }
    template <class T> constexpr Int128& operator|=(const T& b) { return *this = *this | Int128(b); }
    template <class T> constexpr Int128& operator^=(const T& b) { return *this = *this ^ Int128(b); }
    template <class K> constexpr Int128& operator<<=(K k) { return *this = *this << k; }
    template <class K> constexpr Int128& operator>>=(K k) { return *this = *this >> k; }
    constexpr Int128& operator++() { return *this += 1; }
    constexpr Int128& operator--() { return *this -= 1; }
    constexpr Int128 operator++(int) { Int128 t = *this; *this += 1; return t; }
    constexpr Int128 operator--(int) { Int128 t = *this; *this -= 1; return t; }

    // The full product of two 64-bit integers.
    static constexpr Int128 mul64(uint64_t a, uint64_t b) {
        const uint64_t a0 = (uint32_t)a, a1 = a >> 32, b0 = (uint32_t)b, b1 = b >> 32;
        const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
        const uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
        return make(p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32), (mid << 32) | (uint32_t)p00);
    }

private:
    using U = Int128<false>;
    static constexpr int bitlen_u(U x) {
        return x.hi ? 128 - std::countl_zero(x.hi) : 64 - std::countl_zero(x.lo);
    }
    // Unsigned quotient or remainder, by shift and subtract over the quotient's bits.
    static constexpr U udivmod(U n, U d, bool rem) {
        if (!d.hi && !n.hi) return rem ? U(n.lo % d.lo) : U(n.lo / d.lo);
        U q = 0;
        const int shift = bitlen_u(n) - bitlen_u(d);
        if (shift < 0) return rem ? n : q;
        U ds = d << shift;
        for (int i = shift; i >= 0; --i) {
            if (!(n < ds)) {
                n = n - ds;
                q.lo |= i < 64 ? uint64_t(1) << i : 0;
                q.hi |= i >= 64 ? uint64_t(1) << (i - 64) : 0;
            }
            ds = ds >> 1;
        }
        return rem ? n : q;
    }
    static constexpr Int128 divmod(Int128 a, Int128 b, bool rem) {
        if constexpr (!Signed) {
            return udivmod(a, b, rem);
        } else {
            const bool na = a.negative(), nb = b.negative();
            const U ua = U(na ? -a : a), ub = U(nb ? -b : b);
            const Int128 r = Int128(udivmod(ua, ub, rem));
            // The remainder takes the dividend's sign, the quotient the product of both.
            return (rem ? na : na != nb) ? -r : r;
        }
    }
};

// Mixed operands: an integer of another type converts to the 128-bit one, as
// the usual arithmetic conversions do with the built-in types.
#define VF_INT128_MIXED(op)                                                                       \
    template <bool S, class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>                  \
    constexpr auto operator op(Int128<S> a, T b) { return a op Int128<S>(b); }                    \
    template <bool S, class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>                  \
    constexpr auto operator op(T a, Int128<S> b) { return Int128<S>(a) op b; }
VF_INT128_MIXED(+)
VF_INT128_MIXED(-)
VF_INT128_MIXED(*)
VF_INT128_MIXED(/)
VF_INT128_MIXED(%)
VF_INT128_MIXED(&)
VF_INT128_MIXED(|)
VF_INT128_MIXED(^)
VF_INT128_MIXED(==)
VF_INT128_MIXED(!=)
VF_INT128_MIXED(<)
VF_INT128_MIXED(>)
VF_INT128_MIXED(<=)
VF_INT128_MIXED(>=)
#undef VF_INT128_MIXED

using U128 = Int128<false>;
using I128 = Int128<true>;

}  // namespace vf
