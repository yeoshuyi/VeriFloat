// _core.kernel_selfcheck(seed, iters): every fast kernel against exact
// arithmetic, in C++ so that millions of cases run in seconds.
//
// The reference is the general kernel on arbitrary-precision integers
// (round_big). A kernel may decline a case (it is then redone on the general
// path); whatever it does return must equal the reference bit for bit, flags
// included, and the reference must not have raised a rounding event, because
// the kernels never warn. The check runs on whichever kernels are active
// (scalar, or the SIMD level set with _core.set_simd).
#include "accum.hpp"
#include "lean.hpp"

#include <random>
#include <string>
#include <vector>

namespace vf {

namespace {

using Rng = std::mt19937_64;

Fmt random_format(Rng& rng, int64_t max_m, bool narrow_exp) {
    Fmt f;
    for (;;) {
        f.E = 2 + (int64_t)(rng() % (narrow_exp ? 4 : 10));
        const int kind = (int)(rng() % 4);
        static const int64_t common[7] = {2, 3, 7, 10, 23, 30, 52};
        f.M = kind == 0 ? 1 + (int64_t)(rng() % 4) : kind == 1 ? 1 + (int64_t)(rng() % max_m)
              : std::min(max_m, common[rng() % 7]);
        if (f.E + f.M > 63) continue;
        f.bias = (int64_t(1) << (f.E - 1)) - 1 + (int64_t)(rng() % 5) - 2;
        f.rounding = (uint8_t)(rng() % 5);
        f.inf_nan = (uint8_t)(rng() % 3);
        if (f.inf_nan == IEEE && f.E < 2) continue;
        f.tininess_before = rng() & 1;
        f.saturate = rng() % 4 == 0;
        f.ftz = rng() % 8 == 0;
        f.wrap = rng() % 16 == 0;
        f.is_signed = true;
        f.has_zero = true;
        f.derive();
        if (f.emax < f.emin) continue;
        return f;
    }
}

uint64_t random_code(Rng& rng, const Fmt& f, int style) {
    const uint64_t sign = (rng() & 1) << (f.E + f.M);
    uint64_t field;
    const int r = (int)(rng() % 16);
    if (style == 0) field = (uint64_t)std::clamp<int64_t>(f.bias + (int64_t)(rng() % 7) - 3, 0, (int64_t)f.top);
    else if (style == 1) field = rng() & f.top;
    else if (style == 2) field = rng() % 3;
    else field = f.top - std::min<uint64_t>(f.top, rng() % 3);
    if (r == 0) field = 0;
    uint64_t mant = rng() & f.mask();
    if (r == 1) mant = 0;
    if (r == 2) mant = f.mask();
    if (r == 3) mant = 1;
    if (r == 4) return sign;   // zero
    return sign | (field << f.M) | mant;
}

DB exact_add(const DB& x, const DB& y) {
    if (x.sig == 0) return y;
    if (y.sig == 0) return x;
    int64_t e = std::min(x.exp, y.exp);
    BigInt X = x.sig << (unsigned)(x.exp - e), Y = y.sig << (unsigned)(y.exp - e);
    if (x.neg == y.neg) return {x.neg, e, X + Y, false};
    if (X >= Y) return {x.neg, e, X - Y, false};
    return {y.neg, e, Y - X, false};
}

[[noreturn]] void fail(const std::string& what) { raise(PyExc_AssertionError, "kernel self-check: " + what); }

std::string show(const Fmt& f) { return fmt_str(f); }

// ---------------------------------------------------------------- dot products

struct RefDot {
    Code c;
    uint8_t flags = 0;
    bool clean = true;   // no rounding event and no non-finite intermediate
    DB total;            // order 0: the exact sum
};

// sum_terms (accum.cpp) for group = 1 and no alignment, on exact integers.
RefDot reference_dot(const std::vector<FV<u128>>& a, const std::vector<FV<u128>>& b, size_t col, size_t stride,
                     const Fmt& F, const Fmt* P, int order) {
    RefDot R;
    SRArg sr;
    struct SVx {
        DB v;
        bool sign;
    };
    auto round_to = [&](const DB& v, bool sign, const Fmt& f, Code& out) {
        RoundOut r = round_big(v, f, sign, sr);
        R.flags |= r.flags;
        if (r.ev != EV_NONE || !is_finite(r.c, f)) R.clean = false;
        out = r.c;
        return is_finite(r.c, f);
    };
    auto add = [&](const SVx& x, const SVx& y) {
        DB v = exact_add(x.v, y.v);
        return SVx{v, v.sig != 0 ? v.neg : sum_zero_sign(x.sign, y.sign, F)};
    };
    std::vector<SVx> terms;
    for (size_t i = 0; i < a.size(); ++i) {
        const FV<u128>& x = a[i];
        const FV<u128>& y = b[i * stride + col];
        SVx t{DB{x.neg != y.neg, x.exp + y.exp, to_big(x.sig) * to_big(y.sig), false}, x.neg != y.neg};
        if (P) {
            Code c;
            if (!round_to(t.v, t.sign, *P, c)) return R;
            t = {decode<BigInt>(c, *P), c.sign};
        }
        terms.push_back(t);
    }
    if (order == 0) {
        SVx total{DB(), false};
        for (const SVx& t : terms) total = add(total, t);
        R.total = total.v;
        round_to(total.v, total.sign, F, R.c);
        return R;
    }
    if (order == 1) {
        R.c = zero_code(false, F);
        for (const SVx& t : terms) {
            SVx v = add(SVx{decode<BigInt>(R.c, F), R.c.sign}, t);
            if (!round_to(v.v, v.sign, F, R.c)) return R;
        }
        return R;
    }
    std::vector<Code> level;
    for (const SVx& t : terms) {
        Code c;
        if (!round_to(t.v, t.sign, F, c)) return R;
        level.push_back(c);
    }
    if (level.empty()) level.push_back(zero_code(false, F));
    while (level.size() > 1) {
        std::vector<Code> next;
        for (size_t i = 0; i + 1 < level.size(); i += 2) {
            SVx v = add(SVx{decode<BigInt>(level[i], F), level[i].sign},
                        SVx{decode<BigInt>(level[i + 1], F), level[i + 1].sign});
            Code c;
            if (!round_to(v.v, v.sign, F, c)) return R;
            next.push_back(c);
        }
        if (level.size() % 2) next.push_back(level.back());
        level = next;
    }
    R.c = level[0];
    return R;
}

struct Counts {
    uint64_t dots = 0, dots_taken = 0, elems = 0, elems_taken = 0;
};

void check_dots(Rng& rng, Counts& n) {
    const Fmt A = random_format(rng, 30, rng() & 1), B = rng() & 1 ? A : random_format(rng, 30, rng() & 1);
    const Fmt F = random_format(rng, rng() & 1 ? 60 : 24, rng() & 1);
    const Fmt P = random_format(rng, rng() & 1 ? 58 : 24, rng() & 1);
    const bool use_p = rng() & 1;
    const size_t m = 1 + rng() % 3, k = rng() % 15, cols = 1 + rng() % 9;
    const int style = (int)(rng() % 4);
    std::vector<FV<u128>> a(m * k), b(k * cols);
    auto value = [&](const Fmt& f) {
        for (;;) {
            const Code c = code_of_raw64(random_code(rng, f, style), f);
            if (is_finite(c, f)) return fv_of<u128>(c, f);
        }
    };
    for (auto& v : a) v = value(A);
    for (auto& v : b) v = value(B);
    if (cols > 1 && rng() % 3 == 0)   // exact cancellation between terms
        for (size_t i = 0; i + 1 < k; i += 2) {
            for (size_t r = 0; r < m; ++r) a[r * k + i + 1] = a[r * k + i];
            for (size_t c = 0; c < cols; ++c) {
                b[(i + 1) * cols + c] = b[i * cols + c];
                b[(i + 1) * cols + c].neg = !b[(i + 1) * cols + c].neg;
            }
        }
    if (rng() % 4 == 0)   // sparse: zeros of both signs
        for (auto& v : b)
            if (rng() & 1) v = FV<u128>{0, 0, (bool)(rng() & 1)};
    const MatDims d{1, (int64_t)m, (int64_t)k, (int64_t)cols, false, false};
    for (int order = 0; order < 3; ++order) {
        std::vector<uint64_t> raw(m * cols);
        std::vector<uint8_t> flags(m * cols), ok(m * cols);
        std::vector<DB> unr(m * cols);
        if (!lean_matmul_check(a, b, d, F, use_p ? &P : nullptr, order, raw, flags, ok, &unr)) return;
        for (size_t r = 0; r < m; ++r) {
            const std::vector<FV<u128>> row(a.begin() + (long)(r * k), a.begin() + (long)((r + 1) * k));
            for (size_t j = 0; j < cols; ++j) {
                const size_t idx = r * cols + j;
                ++n.dots;
                if (!ok[idx]) continue;
                ++n.dots_taken;
                const RefDot R = reference_dot(row, b, j, cols, F, use_p ? &P : nullptr, order);
                bool good = R.clean && raw[idx] == raw64_of(R.c, F) && flags[idx] == R.flags;
                if (good && order == 0) {   // the unrounded sum must be the same value
                    const DB& u = unr[idx];
                    DB neg_ref = R.total;
                    neg_ref.neg = !neg_ref.neg;
                    good = exact_add(u, neg_ref).sig == 0;
                }
                if (!good)
                    fail(std::string(simd_active() ? simd_active()->name : "scalar") + " dot kernel (order " +
                         std::to_string(order) + ") differs from exact arithmetic: acc " + show(F) +
                         (use_p ? ", product " + show(P) : "") + ", operands " + show(A) + " x " + show(B) + ", k=" +
                         std::to_string(k) + ", row " + std::to_string(r) + ", column " + std::to_string(j) + ": got " +
                         std::to_string(raw[idx]) + " flags " + std::to_string(flags[idx]) + ", exact " +
                         std::to_string(raw64_of(R.c, F)) + " flags " + std::to_string(R.flags) +
                         (R.clean ? "" : " (with a rounding event)"));
            }
        }
    }
}

// ---------------------------------------------------------------- element-wise

void check_elementwise(Rng& rng, const SimdKernels* kn, Counts& n) {
    const Fmt F = random_format(rng, 61, rng() & 1);
    const Fmt S = random_format(rng, 61, rng() & 1);
    const size_t len = 1 + rng() % 40;
    const int style = (int)(rng() % 4);
    std::vector<uint64_t> a(len), b(len), src(len), out(len);
    std::vector<uint8_t> flags(len), redo(len);
    for (size_t i = 0; i < len; ++i) {
        a[i] = random_code(rng, F, style);
        b[i] = rng() % 5 == 0 ? a[i] ^ (rng() & 3) ^ ((rng() & 1) << (F.E + F.M)) : random_code(rng, F, style);
        src[i] = random_code(rng, S, style);
    }
    const EwFmt ef = ew_fmt(F), es = ew_fmt(S);
    const size_t W = kn->width;
    auto redone = [&](size_t i) { return (redo[i / W] >> (i % W)) & 1; };
    SRArg sr;
    for (char op : {'+', '-', '*'}) {
        if (F.M > (op == '*' ? kEwMaxMulM : kEwMaxAddM)) continue;
        for (int scalar_b = 0; scalar_b < 2; ++scalar_b) {
            kn->ew_bin(op, a.data(), 1, b.data(), scalar_b ? 0 : 1, len, ef, out.data(), flags.data(), redo.data());
            for (size_t i = 0; i < len; ++i) {
                ++n.elems;
                if (redone(i)) continue;
                ++n.elems_taken;
                const Code ca = code_of_raw64(a[i], F), cb = code_of_raw64(b[scalar_b ? 0 : i], F);
                bool good = is_finite(ca, F) && is_finite(cb, F);
                if (good) {
                    DB x = decode<BigInt>(ca, F), y = decode<BigInt>(cb, F);
                    bool sb = cb.sign;
                    if (op == '-') { y.neg = !y.neg; sb = !sb; }
                    DB v;
                    bool zs;
                    if (op == '*') {
                        v = {x.neg != y.neg, x.exp + y.exp, x.sig * y.sig, false};
                        zs = ca.sign != cb.sign;
                    } else {
                        v = exact_add(x, y);
                        zs = sum_zero_sign(ca.sign, sb, F);
                    }
                    const RoundOut r = round_big(v, F, zs, sr);
                    good = r.ev == EV_NONE && is_finite(r.c, F) && raw64_of(r.c, F) == out[i] && r.flags == flags[i];
                }
                if (!good)
                    fail(std::string(kn->name) + " element-wise '" + op + "' differs from exact arithmetic in " +
                         show(F) + ": " + std::to_string(a[i]) + ", " + std::to_string(b[scalar_b ? 0 : i]) + " -> " +
                         std::to_string(out[i]) + " flags " + std::to_string(flags[i]));
            }
        }
    }
    kn->ew_conv(src.data(), len, es, ef, out.data(), flags.data(), redo.data());
    for (size_t i = 0; i < len; ++i) {
        ++n.elems;
        if (redone(i)) continue;
        ++n.elems_taken;
        const Code c = code_of_raw64(src[i], S);
        bool good = is_finite(c, S);
        if (good) {
            const RoundOut r = round_big(decode<BigInt>(c, S), F, c.sign, sr);
            good = r.ev == EV_NONE && is_finite(r.c, F) && raw64_of(r.c, F) == out[i] && r.flags == flags[i];
        }
        if (!good)
            fail(std::string(kn->name) + " conversion differs from exact arithmetic: " + show(S) + " -> " + show(F) +
                 ": " + std::to_string(src[i]) + " -> " + std::to_string(out[i]) + " flags " + std::to_string(flags[i]));
    }
}

nb::object kernel_selfcheck(uint64_t seed, uint64_t iters) {
    Rng rng(seed);
    const SimdKernels* simd = simd_active();
    Counts n;
    for (uint64_t i = 0; i < iters; ++i) {
        check_dots(rng, n);
        if (simd) check_elementwise(rng, simd, n);
    }
    nb::dict r;
    r["dots"] = n.dots;
    r["dots_on_kernel"] = n.dots_taken;
    r["elements"] = n.elems;
    r["elements_on_kernel"] = n.elems_taken;
    return r;
}

}  // namespace

void register_selfcheck(nb::module_& m) {
    m.def("kernel_selfcheck", &kernel_selfcheck, nb::arg("seed") = 0, nb::arg("iters") = 1000);
}

}  // namespace vf
