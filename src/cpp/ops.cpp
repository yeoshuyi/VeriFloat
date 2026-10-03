// The FP operations on plain encodings (ops.hpp): a port of the arithmetic
// of verifloat_py/fp.py. Everything here is exact integer arithmetic on
// dyadics followed by one rounding in the kernel (kernel.hpp).
#include "ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <string>

namespace vf {

// ---------------------------------------------------------------- names

static const char* kRoundingName[] = {"rne", "rna", "rtz", "rup", "rdn", "sr"};
static const char* kNanModeName[] = {"canonical", "propagate", "x86", "arm"};

std::string fmt_str(const Fmt& f) {
    std::string s = (f.is_signed ? "e" : "ue") + std::to_string(f.E) + "m" + std::to_string(f.M);
    int64_t default_bias = (int64_t(1) << (f.E - 1)) - 1;
    if (f.bias != default_bias) s += ", bias=" + std::to_string(f.bias);
    if (f.inf_nan != IEEE) s += f.inf_nan == FN ? ", fn" : ", finite";
    if (!f.has_zero) s += ", no-zero";
    if (f.saturate) s += ", saturate";
    if (f.wrap) s += ", wrap";
    if (f.ftz) s += ", ftz";
    if (f.rounding != RNE) s += std::string(", ") + kRoundingName[f.rounding];
    if (f.sr_bits != 8) s += ", sr_bits=" + std::to_string(f.sr_bits);
    if (f.nan_mode != CANONICAL) s += std::string(", ") + kNanModeName[f.nan_mode];
    if (f.tininess_before) s += ", tininess=before";
    return s;
}

void raise_no_zero(const Fmt& f) {
    fail(Err::VALUE, fmt_str(f) + " has no zero and no NaN to return instead");
}

const Fmt& fp64_fmt() {
    static Fmt f = [] {
        Fmt g;
        g.E = 11; g.M = 52; g.bias = 1023;
        g.derive();
        return g;
    }();
    return f;
}

Code checked_code(bool sign, const Code& c, const Fmt& F) {
    if (sign && !F.is_signed) fail(Err::VALUE, "unsigned FP cannot have its sign set");
    Code r = c;
    r.sign = sign;
    return r;
}

// ---------------------------------------------------------------- format names and raw codes

static std::string trimmed(const std::string& s) {
    size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

// Decimal digits at s[pos...] (at most 18 of them).
static bool digits(const std::string& s, size_t& pos, int64_t& out) {
    const size_t start = pos;
    int64_t v = 0;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') {
        if (pos - start >= 18) return false;
        v = v * 10 + (s[pos] - '0');
        ++pos;
    }
    out = v;
    return pos > start;
}

Fmt fmt_parse(const std::string& name) {
    auto bad = [&](const std::string& why) {
        fail(Err::VALUE, why.empty() ? "not an FP format name: '" + name + "'" : why);
    };
    std::vector<std::string> parts;
    for (size_t start = 0;;) {
        const size_t k = name.find(',', start);
        parts.push_back(trimmed(name.substr(start, k == std::string::npos ? k : k - start)));
        if (k == std::string::npos) break;
        start = k + 1;
    }
    Fmt f;
    const std::string& head = parts[0];
    size_t i = 0;
    if (i < head.size() && head[i] == 'u') { f.is_signed = false; ++i; }
    if (i >= head.size() || head[i] != 'e' || !digits(head, ++i, f.E)) bad("");
    if (i >= head.size() || head[i] != 'm' || !digits(head, ++i, f.M) || i != head.size()) bad("");
    bool have_bias = false;
    int64_t bias = 0;
    for (size_t k = 1; k < parts.size(); ++k) {
        const std::string& t = parts[k];
        // key=N, N a decimal integer
        auto number = [&](const char* key, int64_t& out) {
            const std::string pre = std::string(key) + "=";
            if (t.compare(0, pre.size(), pre) != 0) return false;
            size_t p = pre.size();
            const bool neg = p < t.size() && t[p] == '-';
            if (neg) ++p;
            int64_t v;
            if (!digits(t, p, v) || p != t.size()) bad("");
            out = neg ? -v : v;
            return true;
        };
        int64_t v;
        bool known = true;
        if (number("bias", v)) { have_bias = true; bias = v; }
        else if (number("sr_bits", v)) f.sr_bits = v;
        else if (t == "tininess=before") f.tininess_before = true;
        else if (t == "tininess=after") f.tininess_before = false;
        else if (t == "finite") f.inf_nan = FINITE;
        else if (t == "fn") f.inf_nan = FN;
        else if (t == "no-zero") f.has_zero = false;
        else if (t == "saturate") f.saturate = true;
        else if (t == "wrap") f.wrap = true;
        else if (t == "ftz") f.ftz = true;
        else {
            known = false;
            for (int r = 0; r < 6 && !known; ++r)
                if (t == kRoundingName[r]) { f.rounding = (uint8_t)r; known = true; }
            for (int m = 0; m < 4 && !known; ++m)
                if (t == kNanModeName[m]) { f.nan_mode = (uint8_t)m; known = true; }
        }
        if (!known) bad("unknown format tag '" + t + "' in '" + name + "'");
    }
    // The checks of FPFormat (fp.py).
    if (f.E < 1) bad("need at least 1 exponent bit");
    if (f.sr_bits < 1) bad("sr_bits must be >= 1");
    if (f.inf_nan == IEEE && f.M == 0) bad("IEEE inf/NaN needs a mantissa bit; use 'fn'");
    if (f.inf_nan == IEEE && f.E < 2) bad("IEEE inf/NaN needs at least 2 exponent bits");
    const int64_t lim = int64_t(1) << 60;
    if (f.E > 60 || bias > lim || bias < -lim || f.M > (int64_t(1) << 24) || f.sr_bits > (int64_t(1) << 20))
        bad("format too large: needs exp_bits <= 60, |bias| <= 2**60, mantissa_bits <= 2**24, "
            "sr_bits <= 2**20");
    f.bias = have_bias ? bias : (int64_t(1) << (f.E - 1)) - 1;
    // 'fn' with a single code bit: that code is the NaN, nothing is left above zero.
    if (f.inf_nan == FN && f.E + f.M == 1 && f.has_zero) bad("format " + fmt_str(f) + " has no positive finite value");
    f.derive();
    return f;
}

Code code_from_raw(const BigInt& raw, const Fmt& f) {
    Code c;
    c.sign = f.is_signed && bit(raw, f.E + f.M);
    c.field = static_cast<uint64_t>((raw >> (unsigned)f.M) & BigInt(f.top));
    set_mant_big(c, f, raw);
    return c;
}

BigInt raw_of_code(const Code& c, const Fmt& f) {
    BigInt r = BigInt(c.sign && f.is_signed ? 1 : 0) << (unsigned)(f.E + f.M);
    r |= BigInt(c.field) << (unsigned)f.M;
    r |= mant_big(c, f);
    return r;
}

// ---------------------------------------------------------------- unrounded

static UnrD unr_inf(bool neg) {
    UnrD u;
    u.kind = neg ? UnrD::NEGINF : UnrD::POSINF;
    return u;
}

static bool term_of(const Dy<u128>& d, Term& t) {
    t = {d.sig, d.exp, d.neg};
    return true;
}
static bool term_of(const Dy<BigInt>& d, Term& t) {
    if (bitlen(d.sig) > 128) return false;
    t = {to_u128(d.sig), d.exp, d.neg};
    return true;
}
static inline Dy<BigInt> big_dy(const Dy<u128>& d) { return to_big(d); }
static inline const Dy<BigInt>& big_dy(const Dy<BigInt>& d) { return d; }

// Unrounded value of an operation on exact dyadic terms.
template <class U> static UnrD unr_op(uint8_t kind, std::initializer_list<const Dy<U>*> terms) {
    UnrD u;
    u.kind = kind;
    int i = 0;
    bool ok = true;
    for (const Dy<U>* t : terms) ok = ok && term_of(*t, u.t[i++]);
    if (ok) return u;
    WideTerms w;
    i = 0;
    for (const Dy<U>* t : terms) w.t[i++] = big_dy(*t);
    u.wide = Rc<WideTerms>(std::move(w));
    return u;
}

// ---------------------------------------------------------------- numbers

void num_from_double(double d, NumD& n) {
    n.big = false;
    n.s.neg = double_neg(d);
    uint64_t sig;
    int64_t exp;
    double_parts(d, sig, exp);
    n.s.sig = sig;
    n.s.exp = exp;
}

void num_from_ratio(const BigInt& num, const BigInt& den, int64_t keep, NumD& n) {
    bool neg = num < 0;
    BigInt a = abs(num);
    int64_t k = std::max<int64_t>(0, keep + bitlen(den) - bitlen(a) + 1);
    if (bitlen(a) + k <= 127 && bitlen(den) <= 64) {
        u128 N = to_u128(a) << k, D = to_u128(den);
        n.big = false;
        n.s = {neg, -k, N / D, N % D != 0};
        return;
    }
    BigInt N = a << (unsigned)k, q, r;
    boost::multiprecision::divide_qr(N, den, q, r);
    n.big = true;
    n.b = {neg, -k, q, r != 0};
}

RoundOut round_num(NumD& n, const Fmt& f, bool zero_sign, SRArg& sr) {
    return n.big ? round_big(n.b, f, zero_sign, sr) : round_u(n.s, f, zero_sign, sr);
}

// Round num/den (den > 0) into f: FP._round on an exact rational.
RoundOut round_ratio(const BigInt& num, const BigInt& den, const Fmt& f, bool zero_sign, SRArg& sr) {
    NumD n;
    if ((den & (den - 1)) == 0) {
        n.big = true;
        n.b = {num < 0, -(bitlen(den) - 1), abs(num), false};
    } else num_from_ratio(num, den, f.prec + 2, n);
    return round_num(n, f, zero_sign, sr);
}

// ---------------------------------------------------------------- decode

// A double from the encoding of a finite, correctly rounded binary64 result.
static double double_of(const RoundOut& r) {
    return double_of_bits(((uint64_t)r.c.sign << 63) | (r.c.field << 52) | r.c.mant);
}

double fp_to_double(const Code& c, const Fmt& f) {
    const uint64_t sign = (uint64_t)c.sign << 63;
    if (is_nan(c, f)) return double_of_bits(sign | 0x7ff8000000000000u);
    if (is_inf(c, f)) return double_of_bits(sign | 0x7ff0000000000000u);
    if (is_zero(c, f)) return double_of_bits(sign);
    SRArg sr;
    if (!f.wide) {
        Dy<u128> d = decode<u128>(c, f);
        const int64_t n = bitlen(d.sig);
        if (n <= 53 && d.exp >= -1074 && d.exp + n <= 1024) {
            // Exactly a double: its bits are put together here.
            const uint64_t sig = (uint64_t)d.sig, neg = (uint64_t)d.neg << 63;
            const int64_t e = d.exp + n - 1;                    // exponent of the leading bit
            if (e >= -1022)
                return double_of_bits(neg | ((uint64_t)(e + 1023) << 52) |
                                      ((sig << (53 - n)) & ((uint64_t(1) << 52) - 1)));
            return double_of_bits(neg | (sig << (d.exp + 1074)));   // subnormal
        }
        return double_of(round_u(d, fp64_fmt(), d.neg, sr));
    }
    Dy<BigInt> d = decode<BigInt>(c, f);
    return double_of(round_big(d, fp64_fmt(), d.neg, sr));
}

// ---------------------------------------------------------------- arithmetic on dyadics

// Exact a + b reduced to a sticky dyadic with at least prec + 3 bits below
// the result's leading bit. Only the smaller operand is ever truncated, and
// only when cancellation cannot reach the truncated bits.
template <class U> static Dy<U> add_dy(const Dy<U>& a, const Dy<U>& b, int64_t prec) {
    if (a.sig == 0) return b;
    if (b.sig == 0) return a;
    int64_t ta = a.exp + bitlen(a.sig), tb = b.exp + bitlen(b.sig);
    const Dy<U>& X = ta >= tb ? a : b;
    const Dy<U>& Y = ta >= tb ? b : a;
    int64_t tx = std::max(ta, tb), g = tx - std::min(ta, tb);
    int64_t L = g <= 2 ? std::min(X.exp, Y.exp) : std::min(X.exp, tx - prec - 4);
    U xs = shl(X.sig, X.exp - L), ys;
    bool st = false;
    if (Y.exp >= L) ys = shl(Y.sig, Y.exp - L);
    else {
        int64_t k = L - Y.exp;
        ys = shr(Y.sig, k);
        st = low_nonzero(Y.sig, k);
    }
    Dy<U> r;
    r.exp = L;
    if (X.neg == Y.neg) { r.neg = X.neg; r.sig = xs + ys; r.sticky = st; }
    else if (!st) {
        if (xs >= ys) { r.neg = X.neg; r.sig = xs - ys; }
        else { r.neg = Y.neg; r.sig = ys - xs; }
    } else { r.neg = X.neg; r.sig = xs - ys - 1; r.sticky = true; }
    return r;
}

template <class U> static Dy<U> mul_dy(const Dy<U>& a, const Dy<U>& b) {
    return {a.neg != b.neg, a.exp + b.exp, a.sig * b.sig, false};
}

template <class U> static Dy<U> div_dy(const Dy<U>& a, const Dy<U>& b, int64_t prec) {
    Dy<U> r;
    r.neg = a.neg != b.neg;
    if (a.sig == 0) return r;
    int64_t k = std::max<int64_t>(0, prec + 3 + bitlen(b.sig) - bitlen(a.sig) + 1);
    U num = shl(a.sig, k);
    r.sig = num / b.sig;
    r.sticky = (num % b.sig) != 0;
    r.exp = a.exp - b.exp - k;
    return r;
}

// floor(sqrt(n)), with integers only (a floating-point estimate would make
// the result depend on the processor's floating-point state). Newton's
// iteration from a start at or above the root decreases to it and stops
// there.
static uint64_t isqrt64(uint64_t n) {
    if (n == 0) return 0;
    uint64_t r = uint64_t(1) << ((bitlen(n) + 1) / 2);
    for (;;) {
        const uint64_t q = (r + n / r) >> 1;
        if (q >= r) return r;
        r = q;
    }
}
static u128 isqrt(u128 n) {
    if ((uint64_t)(n >> 64) == 0) return isqrt64((uint64_t)n);
    // The root of the top 64 bits, rounded up, gives 32 good bits to start from.
    const int shift = (int)((bitlen(n) - 64 + 1) / 2 * 2);
    u128 r = ((u128)isqrt64((uint64_t)(n >> shift)) + 1) << (shift / 2);
    for (;;) {
        const u128 q = (r + n / r) >> 1;
        if (q >= r) return r;
        r = q;
    }
}
static BigInt isqrt(const BigInt& n) {
    const int64_t bits = bitlen(n);
    if (bits <= 64) return BigInt(isqrt64(static_cast<uint64_t>(n)));
    if (bits <= 128) return to_big(isqrt(to_u128(n)));
    // The same iteration, started from the root of the top 126 bits.
    const unsigned shift = (unsigned)((bits - 126 + 1) / 2 * 2);
    BigInt r = (to_big(isqrt(to_u128(BigInt(n >> shift)))) + 1) << (shift / 2);
    for (;;) {
        BigInt q = (r + n / r) >> 1;
        if (q >= r) return r;
        r = std::move(q);
    }
}

BigInt isqrt_big(const BigInt& n) { return isqrt(n); }

// sqrt(x) for x > 0 with prec + 3 bits and a sticky bit; `exact` reports
// whether the root is exact.
template <class U> static Dy<U> sqrt_dy(Dy<U> x, int64_t prec, bool& exact) {
    if (x.exp & 1) { x.sig = shl(x.sig, 1); x.exp -= 1; }
    int64_t want = prec + 3;
    int64_t k = std::max<int64_t>(0, want - bitlen(x.sig) / 2 + 1);
    U N = shl(x.sig, 2 * k);
    U r = isqrt(N);
    exact = r * r == N;
    return {false, x.exp / 2 - k, r, !exact};
}

static RoundOut round_any(const Dy<u128>& d, const Fmt& f, bool zs, SRArg& sr) { return round_u(d, f, zs, sr); }
static RoundOut round_any(const Dy<BigInt>& d, const Fmt& f, bool zs, SRArg& sr) { return round_big(d, f, zs, sr); }

// Compare finite exact values: -1, 0, 1.
template <class U> static int cmp_dy(const Dy<U>& a, const Dy<U>& b) {
    bool az = a.sig == 0, bz = b.sig == 0;
    if (az && bz) return 0;
    if (az) return b.neg ? 1 : -1;
    if (bz) return a.neg ? -1 : 1;
    if (a.neg != b.neg) return a.neg ? -1 : 1;
    int64_t ta = a.exp + bitlen(a.sig), tb = b.exp + bitlen(b.sig);
    int mag;
    if (ta != tb) mag = ta < tb ? -1 : 1;
    else {
        int64_t e = std::min(a.exp, b.exp);
        U x = shl(a.sig, a.exp - e), y = shl(b.sig, b.exp - e);
        mag = x < y ? -1 : x > y ? 1 : 0;
    }
    return a.neg ? -mag : mag;
}

// ---------------------------------------------------------------- specials

bool sum_zero_sign(bool sa, bool sb, const Fmt& f) {
    // IEEE 754 6.3: an exact zero sum of opposite-signed operands is +0,
    // except -0 when rounding toward negative.
    return sa == sb ? sa : f.rounding == RDN;
}

// This NaN re-encoded in T: canonical, or payload kept and quieted.
std::pair<Code, uint8_t> nan_to(const Opnd& x, const Fmt& T) {
    uint8_t inv = is_snan(x.c, *x.f) ? INVALID : 0;
    if (!T.has_nan()) fail(Err::VALUE, "NaN is not representable in " + fmt_str(T));
    bool keep = T.nan_mode != CANONICAL;
    if (!keep || T.inf_nan == FN) return {nan_code(x.c.sign && keep, T), inv};
    const Fmt& sf = *x.f;
    BigInt payload = (sf.inf_nan == IEEE && sf.M) ? mant_big(x.c, sf)
                                                  : BigInt(1) << (unsigned)std::max<int64_t>(sf.M - 1, 0);
    int64_t shift = T.M - std::max<int64_t>(sf.M, 1);
    payload = shift >= 0 ? BigInt(payload << (unsigned)shift) : BigInt(payload >> (unsigned)-shift);
    payload |= BigInt(1) << (unsigned)(T.M - 1);
    Code c;
    c.sign = x.c.sign && T.is_signed;
    c.field = T.top;
    set_mant_big(c, T, payload);
    return {c, inv};
}

RoundOut nan_result(const Fmt& f, const std::vector<const Opnd*>& ops, bool invalid) {
    bool snan = false;
    for (const Opnd* o : ops) snan = snan || is_snan(o->c, *o->f);
    uint8_t flags = (invalid || snan) ? INVALID : 0;
    if (!f.has_nan()) fail(Err::VALUE, "invalid operation: NaN is not representable in " + fmt_str(f));
    if (f.nan_mode != CANONICAL) {
        const Opnd* src = nullptr;
        for (const Opnd* o : ops)
            if (is_nan(o->c, *o->f)) { src = o; break; }
        if (src) {
            if (f.nan_mode == ARM)
                for (const Opnd* o : ops)
                    if (is_snan(o->c, *o->f)) { src = o; break; }
            return {nan_to(*src, f).first, flags, EV_NONE};
        }
    }
    return {nan_code(f.nan_mode == X86, f), flags, EV_NONE};
}

RoundOut default_nan(const Fmt& f, uint8_t flags) { return {nan_code(f.nan_mode == X86, f), flags, EV_NONE}; }

RoundOut inf_result(bool sign, const Fmt& f) {
    if (sign && !f.is_signed) {
        if (!f.has_zero) return no_zero(f);
        return {zero_code(false, f), UNDERFLOW | INEXACT, EV_CLAMPED};
    }
    if (f.has_inf() && !f.saturate) return {inf_code(sign, f), 0, EV_NONE};
    if (f.has_nan() && !f.saturate) return {nan_code(sign, f), INVALID, EV_NONE};   // 'fn': no infinity
    return {max_code(sign, f), OVERFLOW | INEXACT, EV_SATURATED};
}

RoundOut special_in(double v, const Fmt& f, UnrD& unr) {
    if (std::isnan(v)) {
        if (!f.has_nan()) fail(Err::VALUE, "NaN is not representable in " + fmt_str(f));
        bool keep = f.nan_mode != CANONICAL;
        return {nan_code(std::signbit(v) && keep, f), 0, EV_NONE};
    }
    unr = unr_inf(v < 0);
    return inf_result(v < 0, f);
}

// ---------------------------------------------------------------- formats

Opnd widen(const Opnd& x, const Fmt& F, std::optional<Fmt>& store) {
    if (x.f->id == F.id && F.id >= 0) return x;
    Fmt& plain = store.emplace(F);
    plain.wrap = plain.ftz = plain.saturate = false;
    plain.has_zero = true;
    plain.rounding = RNE;
    plain.id = -1;
    plain.derive();
    Opnd r;
    r.f = &plain;
    const Fmt& sf = *x.f;
    if (is_nan(x.c, sf)) {
        // Keep the payload (and signaling state) so the operation itself
        // raises INVALID; widening never shrinks the mantissa.
        if (plain.inf_nan == FN || sf.inf_nan == FN || !sf.M) r.c = nan_code(x.c.sign, plain);
        else {
            r.c.sign = x.c.sign && plain.is_signed;
            r.c.field = plain.top;
            set_mant_big(r.c, plain, BigInt(mant_big(x.c, sf) << (unsigned)(plain.M - sf.M)));
        }
        return r;
    }
    if (is_inf(x.c, sf)) {
        r.c = checked_code(x.c.sign, inf_code(false, plain), plain);
        return r;
    }
    SRArg sr;
    r.c = sf.wide ? round_big(decode<BigInt>(x.c, sf), plain, x.c.sign, sr).c
                  : round_u(decode<u128>(x.c, sf), plain, x.c.sign, sr).c;
    return r;
}

// ---------------------------------------------------------------- operations

static int64_t need_bits(char op, const Fmt& F) {
    switch (op) {
        case '+': case '-': return std::max(F.M + 3, F.prec + 4) + 2;
        case '*': return 2 * (F.M + 1) + 1;
        case 'f': return std::max(2 * (F.M + 1) + 2, F.prec + 4) + 2;
        case 'q': return 2 * (F.prec + 3) + 4;
        default: return F.prec + F.M + 6;
    }
}

template <class U>
static OpOut finite_op_t(char op, const Opnd& A, const Opnd& B, const Fmt& F, bool zs) {
    Dy<U> a = exact_of<U>(A), b = exact_of<U>(B);
    Dy<U> x;
    uint8_t kind;
    switch (op) {
        case '+': x = add_dy(a, b, F.prec); kind = UnrD::ADD; break;
        case '-': b.neg = !b.neg; x = add_dy(a, b, F.prec); kind = UnrD::ADD; break;
        case '*': x = mul_dy(a, b); kind = UnrD::MUL; break;
        default: x = div_dy(a, b, F.prec); kind = UnrD::DIV; break;
    }
    SRArg sr;
    RoundOut r = round_any(x, F, zs, sr);
    return {r, unr_op<U>(kind, {&a, &b})};
}

static OpOut finite_op(char op, const Opnd& A, const Opnd& B, const Fmt& F, bool zs) {
    if (!F.wide && need_bits(op, F) <= kU128Bits) return finite_op_t<u128>(op, A, B, F, zs);
    return finite_op_t<BigInt>(op, A, B, F, zs);
}

static OpOut plain(const Code& c) { return {{c, 0, EV_NONE}, UnrD()}; }
static OpOut nan_out(const Fmt& F, const std::vector<const Opnd*>& ops, bool invalid) {
    return {nan_result(F, ops, invalid), UnrD()};
}
static OpOut inf_out(const Fmt& F, bool s, uint8_t extra = 0) {
    RoundOut r = inf_result(s, F);
    r.flags |= extra;
    return {r, unr_inf(s)};
}

OpOut op_arith(char op, const Opnd& a, const Opnd& b, const Fmt& F) {
    if (is_nan(a.c, *a.f) || is_nan(b.c, *b.f)) return nan_out(F, {&a, &b}, false);
    bool sa = a.c.sign, sb = b.c.sign;
    bool ai = is_inf(a.c, *a.f), bi = is_inf(b.c, *b.f);
    if (op == '+' || op == '-') {
        if (op == '-') sb = !sb;
        if (ai && bi && sa != sb) return nan_out(F, {}, true);   // inf - inf
        if (ai || bi) return inf_out(F, ai ? sa : sb);
        return finite_op(op, a, b, F, sum_zero_sign(sa, sb, F));
    }
    bool s = sa != sb;
    if (op == '*') {
        if (ai || bi) {
            if (is_zero(a.c, *a.f) || is_zero(b.c, *b.f)) return nan_out(F, {}, true);   // 0 * inf
            return inf_out(F, s);
        }
        return finite_op('*', a, b, F, s);
    }
    if (ai && bi) return nan_out(F, {}, true);   // inf / inf
    if (ai) return inf_out(F, s);
    if (bi) {
        SRArg sr;
        Dy<u128> zero;
        return {round_u(zero, F, s, sr), unr_op<u128>(UnrD::DY, {&zero})};
    }
    if (is_zero(b.c, *b.f)) {
        if (!F.has_nan()) fail(Err::ZERO_DIVISION, "FP division by zero in " + fmt_str(F));
        if (is_zero(a.c, *a.f)) return nan_out(F, {}, true);   // 0 / 0
        return inf_out(F, s, DIVZERO);
    }
    return finite_op('/', a, b, F, s);
}

template <class U>
static OpOut fma_finite(const Opnd& A, const Opnd& B, const Opnd& C, const Fmt& F, bool zs) {
    Dy<U> a = exact_of<U>(A), b = exact_of<U>(B), c = exact_of<U>(C);
    Dy<U> x = add_dy(mul_dy(a, b), c, F.prec);
    SRArg sr;
    RoundOut r = round_any(x, F, zs, sr);
    return {r, unr_op<U>(UnrD::FMA, {&a, &b, &c})};
}

OpOut op_fma(const Opnd& a, const Opnd& b, const Opnd& c, const Fmt& F) {
    if (is_nan(a.c, *a.f) || is_nan(b.c, *b.f)) {
        if (F.nan_mode == ARM && is_snan(c.c, *c.f)) {
            // ARM picks among (a, b) first, then against c, so a
            // signaling c wins over the (by then quiet) a/b NaN.
            return {{nan_to(c, F).first, INVALID, EV_NONE}, UnrD()};
        }
        return nan_out(F, {&a, &b, &c}, false);
    }
    bool s = a.c.sign != b.c.sign;
    bool ai = is_inf(a.c, *a.f), bi = is_inf(b.c, *b.f), ci = is_inf(c.c, *c.f);
    if ((ai && is_zero(b.c, *b.f)) || (is_zero(a.c, *a.f) && bi)) {
        // IEEE 754-2019 7.2: 0 * inf + c is invalid; when c is a quiet
        // NaN, signaling is implementation-defined. RISC-V and ARM signal
        // (ARM returns a signaling c, quieted); x86 hardware does not and
        // returns c.
        if (F.nan_mode == X86 && is_nan(c.c, *c.f) && !is_snan(c.c, *c.f))
            return {{nan_to(c, F).first, 0, EV_NONE}, UnrD()};
        if (F.nan_mode == ARM && is_snan(c.c, *c.f))
            return {{nan_to(c, F).first, INVALID, EV_NONE}, UnrD()};
        return {default_nan(F, INVALID), UnrD()};
    }
    if (is_nan(c.c, *c.f)) return nan_out(F, {&c}, false);
    if (ai || bi) {
        if (ci && c.c.sign != s) return nan_out(F, {}, true);
        return inf_out(F, s);
    }
    if (ci) return inf_out(F, c.c.sign);
    bool zs = sum_zero_sign(s, c.c.sign, F);
    if (!F.wide && need_bits('f', F) <= kU128Bits) return fma_finite<u128>(a, b, c, F, zs);
    return fma_finite<BigInt>(a, b, c, F, zs);
}

template <class U> static OpOut sqrt_finite(const Opnd& X) {
    const Fmt& F = *X.f;
    Dy<U> x = decode<U>(X.c, F);
    bool exact;
    Dy<U> y = sqrt_dy(x, F.prec, exact);
    SRArg sr;
    RoundOut r = round_any(y, F, false, sr);
    if (!exact) r.flags |= INEXACT;
    UnrD u;
    if (exact) u = unr_op<U>(UnrD::DY, {&y});
    return {r, u};
}

OpOut op_sqrt(const Opnd& x) {
    const Fmt& F = *x.f;
    if (is_nan(x.c, F)) return nan_out(F, {&x}, false);
    if (is_zero(x.c, F)) return plain(zero_code(x.c.sign, F));   // sqrt(-0) = -0
    if (x.c.sign) return nan_out(F, {}, true);
    if (is_inf(x.c, F)) return plain(x.c);
    if (!F.wide && need_bits('q', F) <= kU128Bits) return sqrt_finite<u128>(x);
    return sqrt_finite<BigInt>(x);
}

OpOut op_neg(const Opnd& x) {
    const Fmt& f = *x.f;
    if (!f.is_signed) {
        if (is_nan(x.c, f)) return plain(x.c);
        if (is_inf(x.c, f)) return inf_out(f, true);
        SRArg sr;
        if (!f.wide) {
            Dy<u128> d = decode<u128>(x.c, f);
            d.neg = !d.neg;
            return {round_u(d, f, false, sr), unr_op<u128>(UnrD::DY, {&d})};
        }
        Dy<BigInt> d = decode<BigInt>(x.c, f);
        d.neg = !d.neg;
        return {round_big(d, f, false, sr), unr_op<BigInt>(UnrD::DY, {&d})};
    }
    Code c = x.c;
    c.sign = !c.sign;
    return plain(c);
}

int cmp_keys(const Opnd& a, const Opnd& b) {
    bool ai = is_inf(a.c, *a.f), bi = is_inf(b.c, *b.f);
    if (ai || bi) {
        int ka = ai ? (a.c.sign ? -2 : 2) : 0, kb = bi ? (b.c.sign ? -2 : 2) : 0;
        if (ai && bi) return ka < kb ? -1 : ka > kb ? 1 : 0;
        if (ai) return a.c.sign ? -1 : 1;
        return b.c.sign ? 1 : -1;
    }
    if (!a.f->wide && !b.f->wide) return cmp_dy(exact_of<u128>(a), exact_of<u128>(b));
    return cmp_dy(exact_of<BigInt>(a), exact_of<BigInt>(b));
}

OpOut op_minmax(const Opnd& a, const Opnd& b, const Fmt& F, bool pick_max, bool number) {
    bool snan = is_snan(a.c, *a.f) || is_snan(b.c, *b.f);
    bool an = is_nan(a.c, *a.f), bn = is_nan(b.c, *b.f);
    if (an || bn) {
        if (number && !(an && bn)) {
            const Opnd& r = an ? b : a;
            Code out = is_zero(r.c, *r.f) ? checked_code(r.c.sign, zero_code(false, F), F)
                                          : checked_code(r.c.sign, r.c, F);
            return {{out, (uint8_t)(snan ? INVALID : 0), EV_NONE}, UnrD()};
        }
        return nan_out(F, {&a, &b}, false);
    }
    int k = cmp_keys(a, b);
    bool pick_a = k == 0 ? ((a.c.sign && !b.c.sign) != pick_max) : ((k > 0) == pick_max);
    const Opnd& r = pick_a ? a : b;
    if (is_zero(r.c, *r.f)) return plain(checked_code(r.c.sign, zero_code(false, F), F));
    return plain(checked_code(r.c.sign, r.c, F));
}

std::pair<int, uint8_t> op_compare(const Opnd& a, const Opnd& b, bool signaling) {
    if (is_nan(a.c, *a.f) || is_nan(b.c, *b.f)) {
        bool invalid = signaling || is_snan(a.c, *a.f) || is_snan(b.c, *b.f);
        return {3, (uint8_t)(invalid ? INVALID : 0)};
    }
    return {cmp_keys(a, b) + 1, 0};
}

// ---------------------------------------------------------------- remainder

Dy<BigInt> rem_dy(const Dy<BigInt>& x, const Dy<BigInt>& y, bool truncate) {
    if (x.sig == 0) return x;
    BigInt r, Y;         // |x| mod |y| and |y|, in units of 2**e
    int64_t e;
    bool odd;            // parity of floor(|x| / |y|)
    const int64_t d = x.exp - y.exp;
    if (d >= 0) {
        // |x| = sx * 2**d units of 2**y.exp. Reducing modulo 2*sy instead
        // of sy keeps the parity of the quotient, whatever the size of d.
        e = y.exp;
        Y = y.sig;
        const BigInt m = Y << 1;
        const BigInt p = boost::multiprecision::powm(BigInt(2), BigInt(d), m);
        const BigInt t = ((x.sig % m) * p) % m;
        odd = t >= Y;
        r = odd ? BigInt(t - Y) : t;
    } else {
        const int64_t k = -d;
        // |y| >= 2**k units of 2**x.exp: beyond this |x| < |y| / 4, n = 0.
        if (k > bitlen(x.sig) + 1) return x;
        e = x.exp;
        Y = y.sig << (unsigned)k;
        BigInt q;
        boost::multiprecision::divide_qr(x.sig, Y, q, r);
        odd = bit(q, 0);
    }
    bool neg = x.neg;
    if (!truncate) {
        const BigInt twice = r << 1;
        if (twice > Y || (twice == Y && odd)) {   // n rounds up: the remainder changes sign
            r = Y - r;
            neg = !neg;
        }
    }
    return {neg, e, r, false};
}

OpOut op_rem(const Opnd& a, const Opnd& b, const Fmt& F, bool truncate) {
    if (is_nan(a.c, *a.f) || is_nan(b.c, *b.f)) return nan_out(F, {&a, &b}, false);
    if (is_inf(a.c, *a.f) || is_zero(b.c, *b.f)) return nan_out(F, {}, true);   // inf rem y, x rem 0
    const Dy<BigInt> x = exact_of<BigInt>(a);
    // x rem inf = x
    const Dy<BigInt> r = is_inf(b.c, *b.f) ? x : rem_dy(x, exact_of<BigInt>(b), truncate);
    SRArg sr;
    // A zero remainder has the sign of x.
    return {round_big(r, F, a.c.sign, sr), unr_op<BigInt>(UnrD::DY, {&r})};
}

// ---------------------------------------------------------------- neighbours, scaling, classes

// The smallest nonzero magnitude of a format with a zero.
static Code min_code(bool sign, const Fmt& f) {
    if (!f.ftz && f.M > 0) {   // the smallest subnormal
        Code c = zero_code(sign, f);
        set_mant_big(c, f, BigInt(1));
        return c;
    }
    return make_code(sign, 1, 0, f);
}

static bool same_code(const Code& a, const Code& b, const Fmt& f) {
    return a.sign == b.sign && a.field == b.field && mant_big(a, f) == mant_big(b, f);
}

OpOut op_next(const Opnd& x, bool up) {
    const Fmt& f = *x.f;
    const Code& c = x.c;
    if (is_nan(c, f)) return nan_out(f, {&x}, false);
    const bool neg = c.sign;
    const bool away = up != neg;   // the magnitude grows
    if (is_inf(c, f)) return plain(away ? c : max_code(neg, f));
    if (is_zero(c, f)) {
        // Either zero steps to the smallest value of the direction's sign.
        if (!up && !f.is_signed) return plain(zero_code(false, f));   // nothing below zero
        return plain(min_code(!up, f));
    }
    if (away) {
        if (same_code(c, max_code(neg, f), f)) {
            if (f.has_inf() && !f.saturate) return plain(inf_code(neg, f));
            return plain(c);   // no infinity to step to
        }
        Code r = c;
        if (mant_is_mask(c, f) || f.M == 0) {
            r.field = c.field + 1;
            set_mant_big(r, f, BigInt(0));
        } else set_mant_big(r, f, BigInt(mant_big(c, f) + 1));
        return plain(r);
    }
    Code r = c;
    if (!mant_zero(c, f)) {
        set_mant_big(r, f, BigInt(mant_big(c, f) - 1));   // down to the zero code keeps the sign
        return plain(r);
    }
    if (c.field == 0) {
        // The smallest magnitude of a format without a zero: the next value
        // is across the gap, on the other side.
        if (!f.is_signed) return plain(c);
        r.sign = !neg;
        return plain(r);
    }
    r.field = c.field - 1;
    if (r.field == 0 && f.has_zero && f.ftz) return plain(zero_code(neg, f));   // no subnormals
    set_mant_big(r, f, big_mask(f.M));
    return plain(r);
}

OpOut op_scaleb(const Opnd& x, int64_t n) {
    const Fmt& f = *x.f;
    if (is_nan(x.c, f)) return nan_out(f, {&x}, false);
    if (is_inf(x.c, f)) return plain(x.c);
    if (is_zero(x.c, f)) return plain(zero_code(x.c.sign, f));
    Dy<BigInt> d = decode<BigInt>(x.c, f);
    Dy<BigInt> exact = d;
    exact.exp += n;
    // Past `lim` the result has left the format's range and no longer
    // depends on n, except for the exponent field of a wrapping format,
    // which keeps n modulo 2**E.
    const int64_t lim = (f.emax - f.emin) + 2 * f.M + f.sr_bits + 8;
    const int64_t period = int64_t(1) << f.E;
    if (n > lim) n = lim + (n - lim) % period;
    else if (n < -lim) n = -(lim + (-n - lim) % period);
    d.exp += n;
    SRArg sr;
    return {round_big(d, f, x.c.sign, sr), unr_op<BigInt>(UnrD::DY, {&exact})};
}

OpOut op_logb(const Opnd& x) {
    const Fmt& f = *x.f;
    if (is_nan(x.c, f)) return nan_out(f, {&x}, false);
    if (is_inf(x.c, f)) return plain(inf_code(false, f));
    if (is_zero(x.c, f)) return inf_out(f, true, DIVZERO);
    const Dy<BigInt> d = decode<BigInt>(x.c, f);
    const int64_t e = d.exp + bitlen(d.sig) - 1;
    const Dy<BigInt> v{e < 0, 0, BigInt(e < 0 ? -e : e), false};
    SRArg sr;
    return {round_big(v, f, false, sr), unr_op<BigInt>(UnrD::DY, {&v})};
}

unsigned op_fclass(const Code& c, const Fmt& f) {
    if (is_nan(c, f)) return is_snan(c, f) ? 1u << 8 : 1u << 9;
    if (is_inf(c, f)) return c.sign ? 1u << 0 : 1u << 7;
    if (is_zero(c, f)) return c.sign ? 1u << 3 : 1u << 4;
    if (f.has_zero && c.field == 0) return c.sign ? 1u << 2 : 1u << 5;
    return c.sign ? 1u << 1 : 1u << 6;
}

Code op_sgnj(const Opnd& x, bool sign, int mode) {
    const bool s = mode == 0 ? sign : mode == 1 ? !sign : (x.c.sign != sign);
    return checked_code(s, x.c, *x.f);
}

// ---------------------------------------------------------------- conversion

OpOut op_convert(const Opnd& x, const Fmt& F, SRArg& sr) {
    if (is_nan(x.c, *x.f)) {
        auto [c, flags] = nan_to(x, F);
        return {{c, flags, EV_NONE}, UnrD()};
    }
    if (is_inf(x.c, *x.f)) return inf_out(F, x.c.sign);
    if (!x.f->wide) {
        Dy<u128> d = decode<u128>(x.c, *x.f);
        return {round_u(d, F, x.c.sign, sr), unr_op<u128>(UnrD::DY, {&d})};
    }
    Dy<BigInt> d = decode<BigInt>(x.c, *x.f);
    return {round_big(d, F, x.c.sign, sr), unr_op<BigInt>(UnrD::DY, {&d})};
}

OpOut op_from_double(double v, const Fmt& F, SRArg& sr) {
    OpOut o;
    if (!double_finite(v)) {
        o.r = special_in(v, F, o.u);
        return o;
    }
    NumD n;
    num_from_double(v, n);
    o.r = round_num(n, F, std::signbit(v), sr);
    o.u = unr_op<u128>(UnrD::DY, {&n.s});
    return o;
}

// ---------------------------------------------------------------- integer rounding

// _round_to_int(exact, rounding, sr, sr_bits): (q, inexact)
template <class U>
static std::pair<BigInt, bool> round_to_int_t(const Dy<U>& d, uint8_t rounding, const SRArg& sr, int64_t sb) {
    if (d.sig == 0) return {BigInt(0), false};
    if (d.exp >= 0) {
        BigInt q = to_big(d.sig) << (unsigned)d.exp;
        return {d.neg ? BigInt(-q) : q, false};
    }
    int64_t k = -d.exp;
    U s = shr(d.sig, k);
    bool inexact = low_nonzero(d.sig, k);
    if (inexact && decide_up(d.sig, k, s, d.neg, rounding, -1, sr, sb)) s += 1;
    BigInt q = to_big(s);
    return {d.neg ? BigInt(-q) : q, inexact};
}

std::pair<BigInt, bool> round_to_int(const Code& c, const Fmt& f, uint8_t rounding, const SRArg& sr) {
    if (!f.wide) return round_to_int_t(decode<u128>(c, f), rounding, sr, f.sr_bits);
    return round_to_int_t(decode<BigInt>(c, f), rounding, sr, f.sr_bits);
}

IntOut op_to_int(const Opnd& x, int64_t bits, bool is_signed, uint8_t rounding, bool want_exact, SRArg& sr) {
    const Fmt& f = *x.f;
    BigInt lo, hi;
    if (is_signed) {
        BigInt p = BigInt(1) << (unsigned)(bits - 1);
        lo = -p;
        hi = p - 1;
    } else {
        lo = 0;
        hi = (BigInt(1) << (unsigned)bits) - 1;
    }
    // What an invalid conversion returns (NaN, too large, too small), by
    // nan_mode: the largest value, the smallest, zero, or x86's "integer
    // indefinite" (the smallest signed value, or all ones when unsigned).
    auto invalid = [&](int kind) -> IntOut {
        static const char table[4][3] = {{'M', 'M', 'm'}, {'M', 'M', 'm'}, {'x', 'x', 'x'}, {'0', 'M', 'm'}};
        char which = table[f.nan_mode][kind];
        if (which == 'M') return {hi, INVALID};
        if (which == 'm') return {lo, INVALID};
        if (which == '0') return {BigInt(0), INVALID};
        return {is_signed ? lo : hi, INVALID};
    };
    if (is_nan(x.c, f)) return invalid(0);
    if (is_inf(x.c, f)) return invalid(x.c.sign ? 2 : 1);
    if (rounding == SR) sr.resolve(f);
    auto [q, inexact] = round_to_int(x.c, f, rounding, sr);
    if (q > hi) return invalid(1);
    if (q < lo) return invalid(2);
    return {q, (uint8_t)(inexact && want_exact ? INEXACT : 0)};
}

OpOut op_round_to_integral(const Opnd& x, uint8_t rounding, bool want_exact, SRArg& sr) {
    const Fmt& f = *x.f;
    if (is_nan(x.c, f)) return nan_out(f, {&x}, false);
    if (is_inf(x.c, f)) return plain(x.c);
    if (is_zero(x.c, f)) return plain(zero_code(x.c.sign, f));   // DAZ subnormal reads as zero
    if (rounding == SR) sr.resolve(f);
    auto [q, inexact] = round_to_int(x.c, f, rounding, sr);
    if (q == 0 && !f.has_zero) return {no_zero(f), UnrD()};
    // The integer is representable: it has no more significant bits than
    // the input (or it is 0 or a power of two).
    Fmt g = f;
    g.rounding = RNE;
    g.wrap = g.ftz = false;
    g.id = -1;
    g.derive();
    SRArg none;
    Dy<BigInt> d{q < 0, 0, abs(q), false};
    RoundOut r = round_big(d, g, x.c.sign, none);
    r.flags = inexact && want_exact ? INEXACT : 0;
    r.ev = EV_NONE;
    return {r, unr_op<BigInt>(UnrD::DY, {&d})};
}

}  // namespace vf
