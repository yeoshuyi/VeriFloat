// The portable 128-bit integers (src/cpp/int128.hpp), which MSVC builds use,
// against the compiler's own unsigned __int128 and __int128: every operator,
// on boundary values in all pairs and on random operands. Prints the number
// of checks and exits 0, or prints the first difference and exits 1.
//
//   c++ -std=c++20 -O2 -Isrc/cpp tests/native/int128_check.cpp && ./a.out [random pairs]
#include "int128.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using vf::I128;
using vf::U128;
using nu = unsigned __int128;
using ns = __int128;

static unsigned long long checks = 0;

static nu native(U128 x) { return ((nu)x.hi << 64) | x.lo; }
static ns native(I128 x) { return (ns)(((nu)x.hi << 64) | x.lo); }
static U128 port(nu x) { return U128::make((uint64_t)(x >> 64), (uint64_t)x); }
static I128 port_s(ns x) { return I128(port((nu)x)); }

[[noreturn]] static void fail(const char* what, nu a, nu b) {
    std::printf("MISMATCH %s a=%016llx%016llx b=%016llx%016llx\n", what, (unsigned long long)(a >> 64),
                (unsigned long long)a, (unsigned long long)(b >> 64), (unsigned long long)b);
    std::exit(1);
}
#define SAME(what, got, want, a, b)                    \
    do {                                               \
        ++checks;                                      \
        if ((got) != (want)) fail(what, (nu)(a), (nu)(b)); \
    } while (0)

static void pair(nu a, nu b) {
    const U128 pa = port(a), pb = port(b);
    const ns sa = (ns)a, sb = (ns)b;
    const I128 qa = port_s(sa), qb = port_s(sb);
    SAME("u+", native(pa + pb), a + b, a, b);
    SAME("u-", native(pa - pb), a - b, a, b);
    SAME("u*", native(pa * pb), a * b, a, b);
    SAME("u&", native(pa & pb), a & b, a, b);
    SAME("u|", native(pa | pb), a | b, a, b);
    SAME("u^", native(pa ^ pb), a ^ b, a, b);
    SAME("u~", native(~pa), ~a, a, b);
    SAME("u-neg", native(-pa), -a, a, b);
    SAME("u==", pa == pb, a == b, a, b);
    SAME("u<", pa < pb, a < b, a, b);
    SAME("u<=", pa <= pb, a <= b, a, b);
    SAME("u>", pa > pb, a > b, a, b);
    if (b) {
        SAME("u/", native(pa / pb), a / b, a, b);
        SAME("u%", native(pa % pb), a % b, a, b);
    }
    // Signed: wrapping arithmetic is computed on the unsigned type, where it is defined.
    SAME("s+", native(qa + qb), (ns)(a + b), a, b);
    SAME("s-", native(qa - qb), (ns)(a - b), a, b);
    SAME("s*", native(qa * qb), (ns)(a * b), a, b);
    SAME("s<", qa < qb, sa < sb, a, b);
    SAME("s>=", qa >= qb, sa >= sb, a, b);
    const ns smin = (ns)((nu)1 << 127);
    if (sb != 0 && !(sa == smin && sb == -1)) {
        SAME("s/", native(qa / qb), sa / sb, a, b);
        SAME("s%", native(qa % qb), sa % sb, a, b);
    }
    const int k = (int)(b & 127);
    SAME("u<<", native(pa << k), a << k, a, k);
    SAME("u>>", native(pa >> k), a >> k, a, k);
    SAME("s>>", native(qa >> k), sa >> k, a, k);
    // Mixed with 64-bit integers, as the core writes it.
    const uint64_t u = (uint64_t)b;
    const int64_t s = (int64_t)b;
    SAME("u+u64", native(pa + u), a + u, a, b);
    SAME("u*u64", native(pa * u), a * u, a, b);
    SAME("u-i64", native(pa - s), a - s, a, b);
    SAME("u<u64", pa < u, a < u, a, b);
    SAME("s*i64", native(qa * s), (ns)(a * (nu)(ns)s), a, b);
    SAME("s<i64", qa < s, sa < s, a, b);
    SAME("s+u64", native(qa + u), (ns)(a + u), a, b);
    if (u) SAME("u/u64", native(pa / u), a / u, a, b);
    if (s && !(sa == smin && s == -1)) SAME("s/i64", native(qa / s), sa / s, a, b);
    SAME("full64", native(U128::mul64((uint64_t)a, u)), (nu)(uint64_t)a * u, a, b);
    // Conversions.
    SAME("to u64", (uint64_t)pa, (uint64_t)a, a, b);
    SAME("to i64", (int64_t)qa, (int64_t)sa, a, b);
    SAME("from i64", native(U128(s)), (nu)s, a, b);
    SAME("from i64 signed", native(I128(s)), (ns)s, a, b);
    SAME("bool", (bool)pa, a != 0, a, b);
}

int main(int argc, char** argv) {
    std::vector<nu> edges;
    for (int i = 0; i < 128; ++i) {
        const nu p = (nu)1 << i;
        edges.insert(edges.end(), {p, p - 1, p + 1, ~p, -p});
    }
    edges.insert(edges.end(), {0, 3, 10, ~(nu)0, ((nu)1 << 64) * 0xFFFFFFFFull, (nu)0x123456789ABCDEFull << 40});
    for (nu a : edges)
        for (nu b : edges) pair(a, b);
    // Random operands of random widths, so that quotients of every length occur.
    const long n = argc > 1 ? std::atol(argv[1]) : 2000000;
    uint64_t st = 0x9E3779B97F4A7C15ull;
    auto next = [&] {
        st ^= st << 13, st ^= st >> 7, st ^= st << 17;
        return st;
    };
    for (long i = 0; i < n; ++i) {
        nu a = ((nu)next() << 64) | next(), b = ((nu)next() << 64) | next();
        a >>= next() & 127;
        b >>= next() & 127;
        if (next() & 1) a = -a;
        if (next() & 1) b = -b;
        pair(a, b);
    }
    std::printf("%llu checks, no difference\n", checks);
    return 0;
}
