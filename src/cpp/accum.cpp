// Dot-product accumulation (verifloat_py/accum.py) and the inner loops of
// dot/matmul (blockscale.py _reduce). Inputs that are not dyadic (e.g. a
// Fraction 1/3) return None so the Python implementation handles them.
#include "accum.hpp"
#include "lean.hpp"

#include <nanobind/stl/string.h>

#include <climits>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace vf {

nb::object arith(char op, const Opnd& a, const Opnd& b, PyObject* fmt);
nb::object convert_to(PyFP* self, PyObject* fmt, PyObject* sr_rand);

// ---------------------------------------------------------------- exact dyadics

static DB dmul(const DB& a, const DB& b) { return {a.neg != b.neg, a.exp + b.exp, a.sig * b.sig, false}; }

static DB dadd(const DB& a, const DB& b) {
    if (a.sig == 0) return b;
    if (b.sig == 0) return a;
    int64_t e = std::min(a.exp, b.exp);
    BigInt x = a.sig << (unsigned)(a.exp - e), y = b.sig << (unsigned)(b.exp - e);
    if (a.neg == b.neg) return {a.neg, e, x + y, false};
    if (x >= y) return {a.neg, e, x - y, false};
    return {b.neg, e, y - x, false};
}

static SV sv_add(const SV& a, const SV& b, const Fmt& f) {
    DB v = dadd(a.v, b.v);
    return {v, v.sig != 0 ? v.neg : sum_zero_sign(a.sign, b.sign, f)};
}

// ---------------------------------------------------------------- inputs

static bool pow2_den(PyObject* d, int64_t& sh) {
    int overflow = 0;
    long long x = PyLong_AsLongLongAndOverflow(d, &overflow);
    if (x == -1 && PyErr_Occurred()) raise_current();
    if (!overflow) {
        if (x <= 0 || (x & (x - 1))) return false;
        sh = bitlen((uint64_t)x) - 1;
        return true;
    }
    BigInt b = big_from_long(d);
    if (b <= 0 || (b & (b - 1)) != 0) return false;
    sh = bitlen(b) - 1;
    return true;
}

static void set_from_long(PyObject* o, DB& v) {
    BigInt b = big_from_long(o);
    v.neg = b < 0;
    v.sig = abs(b);
    v.exp = 0;
}

// Parse a dot-product operand; false if the Python path must handle it.
static bool parse_in(PyObject* o, In& r) {
    if (is_fp(o)) { r.fp = (PyFP*)o; return true; }
    DB& v = r.n.v;
    if (PyLong_Check(o)) set_from_long(o, v);
    else if (PyFloat_Check(o)) {
        double d = PyFloat_AS_DOUBLE(o);
        if (!double_finite(d)) return false;
        uint64_t sig;
        int64_t exp;
        double_parts(d, sig, exp);
        v.neg = sig != 0 && double_neg(d);
        v.sig = BigInt(sig);
        v.exp = exp;
    } else if (is_fraction(o)) {
        static PyObject* sn = PyUnicode_InternFromString("_numerator");
        static PyObject* sd = PyUnicode_InternFromString("_denominator");
        nb::object num = steal_checked(PyObject_GetAttr(o, sn));
        nb::object den = steal_checked(PyObject_GetAttr(o, sd));
        int64_t sh;
        if (!pow2_den(den.ptr(), sh)) return false;
        set_from_long(num.ptr(), v);
        v.exp = -sh;
    } else return false;
    r.n.sign = v.sig != 0 && v.neg;
    return true;
}

static bool fp_finite(PyFP* p) { return is_finite(p->c, *p->f); }

static SV signed_of(const In& x) {   // accum.py _signed
    if (!x.fp) return x.n;
    return {decode<BigInt>(x.fp->c, *x.fp->f), x.fp->c.sign};
}

// ---------------------------------------------------------------- accumulation

// A term or a partial sum: an FP encoding in some format, or an exact value.
struct T {
    bool is_fp = false;
    nb::object obj;       // FP object when it came from Python arithmetic
    Code c;
    const Fmt* f = nullptr;
    uint8_t flags = 0;
    SV x;                 // exact value (non-FP terms)
};

static T t_from_fp(nb::object o) {
    PyFP* p = (PyFP*)o.ptr();
    T t;
    t.is_fp = true;
    t.c = p->c;
    t.f = p->f;
    t.flags = p->flags;
    t.obj = std::move(o);
    return t;
}

static Unr unr_big(const DB& v) {
    Unr u;
    u.kind = Unr::DYB;
    u.big = Rc<DB>(v);
    return u;
}

// FP._encode(v, fmt, sign): round, warning if the format asks for it.
static T encode(const DB& v, bool sign, PyObject* fmt) {
    const Fmt& F = fmt_of(fmt);
    SRArg sr;
    RoundOut r = round_big(v, F, sign, sr);
    if (r.ev != EV_NONE) fp_finish(fmt, r, unr_big(v));
    T t;
    t.is_fp = true;
    t.c = r.c;
    t.f = &F;
    t.flags = r.flags;
    return t;
}

static SV t_signed(const T& t) {
    if (!t.is_fp) return t.x;
    return {decode<BigInt>(t.c, *t.f), t.c.sign};
}

// accum.py _mul
static T mul_term(const In& x, const In& y, PyObject* product) {
    if (x.fp && y.fp && !(fp_finite(x.fp) && fp_finite(y.fp))) {
        nb::object r = steal_checked(PyNumber_Multiply((PyObject*)x.fp, (PyObject*)y.fp));
        if (product) r = convert_to((PyFP*)r.ptr(), product, Py_None);
        return t_from_fp(std::move(r));
    }
    SV a = signed_of(x), b = signed_of(y);
    DB p = dmul(a.v, b.v);
    bool s = a.sign != b.sign;
    if (!product) {
        T t;
        t.x = {p, s};
        return t;
    }
    return encode(p, s, product);
}

static nb::object finish_plain(PyObject* fmt, const Code& c, uint8_t flags, Unr u = Unr()) {
    return fp_finish(fmt, {c, flags, EV_NONE}, std::move(u));
}

// accum.py _aligned_sum
static SV aligned_sum(std::vector<SV>& vals, size_t lo, size_t hi, const AccSpec& a, const Fmt& F, bool& inexact) {
    if (a.aligned) {
        bool any = false;
        int64_t top = 0;
        for (size_t i = lo; i < hi; ++i) {
            const DB& v = vals[i].v;
            if (v.sig == 0) continue;
            int64_t t = v.exp + bitlen(v.sig) - 1;
            top = any ? std::max(top, t) : t;
            any = true;
        }
        if (any) {
            int64_t p = top - a.align_bits;
            for (size_t i = lo; i < hi; ++i) {
                DB& v = vals[i].v;
                if (v.sig == 0 || v.exp >= p) continue;
                int64_t k = p - v.exp;
                if (low_nonzero(v.sig, k)) inexact = true;
                v.sig = v.sig >> (unsigned)k;
                v.exp = p;
            }
        }
    }
    if (lo == hi) return SV();
    SV total = vals[lo];
    for (size_t i = lo + 1; i < hi; ++i) total = sv_add(total, vals[i], F);
    return total;
}

static nb::object sum_terms(std::vector<T>& terms, const In* init, const AccSpec& a) {
    PyObject* fmt = a.fmt;
    const Fmt& F = fmt_of(fmt);
    uint8_t flags = 0;
    for (const T& t : terms)
        if (t.is_fp) flags |= t.flags;
    // IEEE result when any term (or an FP init) is inf or NaN
    std::vector<Opnd> specials;
    for (const T& t : terms)
        if (t.is_fp && !is_finite(t.c, *t.f)) specials.push_back({t.c, t.f, nullptr});
    if (init && init->fp && !fp_finite(init->fp)) specials.push_back(opnd_of(init->fp));
    if (!specials.empty()) {
        std::vector<const Opnd*> nans;
        bool pos = false, neg = false;
        for (const Opnd& o : specials) {
            if (is_nan(o.c, *o.f)) nans.push_back(&o);
            else (o.c.sign ? neg : pos) = true;
        }
        RoundOut r;
        if (!nans.empty()) r = nan_result(F, nans, false);
        else if (pos && neg) r = nan_result(F, {}, true);
        else r = inf_result(neg, F);
        return finish_plain(fmt, r.c, r.flags | flags);
    }
    std::vector<SV> vals;
    vals.reserve(terms.size());
    for (const T& t : terms) vals.push_back(t_signed(t));
    std::vector<SV> adds;
    for (size_t i = 0; i < vals.size(); i += (size_t)a.group) {
        bool inexact = false;
        adds.push_back(aligned_sum(vals, i, std::min(vals.size(), i + (size_t)a.group), a, F, inexact));
        if (inexact) flags |= INEXACT;
    }
    bool has_start = init != nullptr;
    SV start = has_start ? signed_of(*init) : SV();
    if (a.order == 0) {   // exact: one rounding
        SV total = start;
        for (const SV& t : adds) total = sv_add(total, t, F);
        T r = encode(total.v, total.sign, fmt);
        return finish_plain(fmt, r.c, r.flags | flags, unr_big(total.v));
    }
    if (a.order == 1) {   // sequential
        T acc = encode(start.v, start.sign, fmt);
        flags |= acc.flags;
        for (const SV& t : adds) {
            if (!is_finite(acc.c, F)) break;   // inf + finite stays inf (IEEE)
            SV v = sv_add(t_signed(acc), t, F);
            acc = encode(v.v, v.sign, fmt);
            flags |= acc.flags;
        }
        return finish_plain(fmt, acc.c, flags);
    }
    // pairwise tree
    std::vector<T> level;
    for (const SV& t : adds) level.push_back(encode(t.v, t.sign, fmt));
    if (has_start) level.insert(level.begin(), encode(start.v, start.sign, fmt));
    if (level.empty()) level.push_back(encode(DB(), false, fmt));
    for (const T& v : level) flags |= v.flags;
    while (level.size() > 1) {
        std::vector<T> nxt;
        for (size_t i = 0; i + 1 < level.size(); i += 2) {
            const T &x = level[i], &y = level[i + 1];
            T r;
            if (is_finite(x.c, F) && is_finite(y.c, F)) {
                SV v = sv_add(t_signed(x), t_signed(y), F);
                r = encode(v.v, v.sign, fmt);
            } else {   // an overflowed node: IEEE rules
                r = t_from_fp(arith('+', {x.c, &F, fmt}, {y.c, &F, fmt}, fmt));
            }
            flags |= r.flags;
            nxt.push_back(std::move(r));
        }
        if (level.size() % 2) nxt.push_back(std::move(level.back()));
        level = std::move(nxt);
    }
    return finish_plain(fmt, level[0].c, flags);
}

nb::object sum_products(const In* a, const In* b, size_t n, size_t sa, size_t sb, const In* init,
                               const AccSpec& spec) {
    std::vector<T> terms;
    terms.reserve(n);
    for (size_t i = 0; i < n; ++i) terms.push_back(mul_term(a[i * sa], b[i * sb], spec.product));
    return sum_terms(terms, init, spec);
}

// Inputs the native path cannot reproduce exactly: non-dyadic or non-finite
// numbers, and non-finite FP values next to numbers (the reference raises
// part way through).
bool parse_all(nb::handle seq, std::vector<In>& out, bool& nonfinite_fp, bool& numbers) {
    nb::object list = steal_checked(PySequence_Fast(seq.ptr(), "expected a sequence"));
    Py_ssize_t n = PySequence_Fast_GET_SIZE(list.ptr());
    PyObject** items = PySequence_Fast_ITEMS(list.ptr());
    out.resize((size_t)n);
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (!parse_in(items[i], out[(size_t)i])) return false;
        if (out[(size_t)i].fp) nonfinite_fp |= !fp_finite(out[(size_t)i].fp);
        else numbers = true;
    }
    return true;
}

bool spec_from(nb::handle acc, AccSpec& s) {
    // acc = (fmt, order, product, group, align_bits)
    nb::tuple t = nb::borrow<nb::tuple>(acc);
    s.fmt = t[0].ptr();
    std::string order = nb::cast<std::string>(nb::str(t[1]));
    s.order = order == "exact" ? 0 : order == "sequential" ? 1 : 2;
    s.product = t[2].is_none() ? nullptr : t[2].ptr();
    s.group = nb::cast<int64_t>(t[3]);
    s.aligned = !t[4].is_none();
    s.align_bits = s.aligned ? nb::cast<int64_t>(t[4]) : 0;
    return is_format(s.fmt) && (!s.product || is_format(s.product));
}

static nb::object exact_dot(const In* a, const In* b, size_t n, size_t sa, size_t sb) {
    DB total;
    for (size_t i = 0; i < n; ++i) total = dadd(total, dmul(signed_of(a[i * sa]).v, signed_of(b[i * sb]).v));
    return fraction_dy<BigInt>(total.neg, total.sig, total.exp);
}

// ---------------------------------------------------------------- fast kernel

// One dot product on the fast kernel (see fast.hpp); false means "use the
// general path". Mirrors sum_terms for group = 1 and no alignment.
template <class U>
VF_NOINLINE static bool fast_dot(const FV<U>* a, const FV<U>* b, size_t n, const Fmt& Ff, const Fmt* Pf, int order,
                     std::vector<FV<U>>& tmp, FV<u128>& res, uint8_t& flags_out, bool want_unr, DB* unr) {
    const RT F = rt_of(Ff), PT = rt_of(Pf ? *Pf : Ff);
    const RT* P = Pf ? &PT : nullptr;
    unsigned fl = 0;
    bool st;
    if (order == 1) {   // sequential
        FV<U> acc;
        for (size_t i = 0; i < n; ++i) {
            FV<U> t{a[i].sig * b[i].sig, a[i].exp + b[i].exp, a[i].neg != b[i].neg};
            if (P && !fast_round(t, false, *P, t, fl)) return false;
            FV<U> s;
            fast_add(acc, t, F.rounding, s, st);
            if (!fast_round(s, st, F, acc, fl)) return false;
        }
        res = {(u128)acc.sig, acc.exp, acc.neg};
        flags_out = (uint8_t)fl;
        return true;
    }
    if (order == 2) {   // pairwise tree
        tmp.resize(n ? n : 1);
        tmp[0] = FV<U>();
        for (size_t i = 0; i < n; ++i) {
            FV<U> t{a[i].sig * b[i].sig, a[i].exp + b[i].exp, a[i].neg != b[i].neg};
            if (P && !fast_round(t, false, *P, t, fl)) return false;
            if (!fast_round(t, false, F, tmp[i], fl)) return false;
        }
        size_t len = tmp.size();
        while (len > 1) {
            size_t o = 0;
            for (size_t i = 0; i + 1 < len; i += 2) {
                FV<U> s;
                fast_add(tmp[i], tmp[i + 1], F.rounding, s, st);
                if (!fast_round(s, st, F, tmp[o++], fl)) return false;
            }
            if (len % 2) tmp[o++] = tmp[len - 1];
            len = o;
        }
        res = {(u128)tmp[0].sig, tmp[0].exp, tmp[0].neg};
        flags_out = (uint8_t)fl;
        return true;
    }
    // exact: every term on one grid, summed in 128 bits, rounded once
    tmp.resize(n);
    bool any = false;
    int64_t lo = 0, hi = 0;
    for (size_t i = 0; i < n; ++i) {
        FV<U> t{a[i].sig * b[i].sig, a[i].exp + b[i].exp, a[i].neg != b[i].neg};
        if (P && !fast_round(t, false, *P, t, fl)) return false;
        tmp[i] = t;
        if (t.sig == 0) continue;
        int64_t top = t.exp + fbitlen(t.sig);
        lo = any ? std::min(lo, t.exp) : t.exp;
        hi = any ? std::max(hi, top) : top;
        any = true;
    }
    if (hi - lo + bitlen((uint64_t)n) + 1 > kU128Bits) return false;
    i128 sum = 0;
    bool zs = false;   // sign of the running total while it is zero
    for (size_t i = 0; i < n; ++i) {
        const FV<U>& t = tmp[i];
        if (t.sig != 0) {
            i128 v = (i128)((u128)t.sig << (t.exp - lo));
            sum += t.neg ? -v : v;
        }
        zs = sum != 0 ? sum < 0 : sum_zero_sign(zs, t.neg, Ff);
    }
    FV<u128> total{sum < 0 ? (u128)(-sum) : (u128)sum, any ? lo : 0, zs};
    if (!fast_round(total, false, F, res, fl)) return false;
    if (want_unr) *unr = DB{sum < 0, total.exp, to_big(total.sig), false};
    flags_out = (uint8_t)fl;
    return true;
}

// Sequential accumulation of up to kLanes dot products that share the row
// `a`, side by side. Each sum is one long dependency chain, so interleaving
// independent chains is what lets the CPU overlap them. A lane that leaves
// the fast kernel is marked not ok and keeps running on stale values.
// (lean.hpp has the tighter version for narrow formats.)

template <class U>
VF_NOINLINE static void seq_lanes(const FV<U>* a, const FV<U>* const* b, size_t lanes, size_t n, const Fmt& Ff, const Fmt* Pf,
                      FV<U>* acc_out, uint8_t* fl_out, bool* good_out) {
    const RT F = rt_of(Ff), PT = rt_of(Pf ? *Pf : Ff);
    const bool P = Pf != nullptr;
    const uint8_t rounding = F.rounding;
    FV<U> acc[kLanes];
    unsigned fl[kLanes] = {};
    bool good[kLanes];
    for (size_t c = 0; c < lanes; ++c) good[c] = true;
    for (size_t i = 0; i < n; ++i) {
        const FV<U> x = a[i];
        for (size_t c = 0; c < lanes; ++c) {
            const FV<U>& y = b[c][i];
            FV<U> t{x.sig * y.sig, x.exp + y.exp, x.neg != y.neg};
            bool g = true;
            if (P) g = fast_round(t, false, PT, t, fl[c]);
            FV<U> s;
            bool st;
            fast_add(acc[c], t, rounding, s, st);
            g &= fast_round(s, st, F, acc[c], fl[c]);
            good[c] &= g;
        }
    }
    for (size_t c = 0; c < lanes; ++c) { acc_out[c] = acc[c]; fl_out[c] = (uint8_t)fl[c]; good_out[c] = good[c]; }
}

template <class U>
static void fast_matmul_t(const std::vector<FV<u128>>& a, const std::vector<FV<u128>>& b, const MatDims& d,
                          const Fmt& F, const Fmt* P, int order, std::vector<uint64_t>& raw,
                          std::vector<uint8_t>& flags, std::vector<uint8_t>& ok, std::vector<DB>* unr) {
    const size_t m = (size_t)d.m, k = (size_t)d.k, n = (size_t)d.n;
    std::vector<FV<U>> A(a.size()), Bt(b.size()), tmp;
    auto conv = [](const FV<u128>& v) -> FV<U> { return {(U)v.sig, v.exp, v.neg}; };
    for (size_t i = 0; i < a.size(); ++i) A[i] = conv(a[i]);
    // b transposed, so that each column is contiguous
    const size_t nb_b = d.batch_b ? (size_t)d.nbatch : 1;
    for (size_t bi = 0; bi < nb_b; ++bi)
        for (size_t t = 0; t < k; ++t)
            for (size_t j = 0; j < n; ++j)
                Bt[bi * k * n + j * k + t] = conv(b[bi * k * n + t * n + j]);
    size_t idx = 0;
    for (size_t bi = 0; bi < (size_t)d.nbatch; ++bi) {
        const FV<U>* pa = A.data() + (d.batch_a ? bi * m * k : 0);
        const FV<U>* pb = Bt.data() + (d.batch_b ? bi * k * n : 0);
        if (order == 1) {
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < n; j += kLanes) {
                    const size_t lanes = std::min(kLanes, n - j);
                    const FV<U>* cols[kLanes];
                    FV<U> acc[kLanes];
                    uint8_t fl[kLanes];
                    bool good[kLanes];
                    for (size_t c = 0; c < lanes; ++c) cols[c] = pb + (j + c) * k;
                    seq_lanes<U>(pa + i * k, cols, lanes, k, F, P, acc, fl, good);
                    for (size_t c = 0; c < lanes; ++c, ++idx) {
                        if (!good[c]) continue;
                        raw[idx] = raw64_of(fv_code(acc[c], F), F);
                        flags[idx] = fl[c];
                        ok[idx] = 1;
                    }
                }
            continue;
        }
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j, ++idx) {
                if (ok[idx]) continue;   // done by a tighter kernel
                FV<u128> r;
                uint8_t fl = 0;
                DB* u = unr ? &(*unr)[idx] : nullptr;
                if (fast_dot<U>(pa + i * k, pb + j * k, k, F, P, order, tmp, r, fl, u != nullptr, u)) {
                    raw[idx] = raw64_of(fv_code(r, F), F);
                    flags[idx] = fl;
                    ok[idx] = 1;
                }
            }
    }
}

// ---------------------------------------------------------------- kernel selection

static const SimdKernels* simd_by_name(const std::string& name) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    // Ask the CPU before touching anything built for an instruction set.
    if (name == "avx512" && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512cd") &&
        __builtin_cpu_supports("avx512dq"))
        return simd_kernels_avx512();
    if (name == "avx2" && __builtin_cpu_supports("avx2")) return simd_kernels_avx2();
#else
    (void)name;
#endif
    return nullptr;
}

static const SimdKernels* g_simd = [] {
    const SimdKernels* k = simd_by_name("avx512");
    return k ? k : simd_by_name("avx2");
}();

const SimdKernels* simd_active() { return g_fast ? g_simd : nullptr; }

// The exact sum of the (P-rounded) products of every row of a with every
// column of b: total[idx] * 2**unit[idx], with got[idx] = 0 where the kernel
// declined. The grid of each sum is set by the smallest exponents of its row
// and column; a pair whose exponents spread further than the kernel's limbs
// hold is declined.
static void lean_exact_sums(const LeanOperands& A, const LeanOperands& B, const MatDims& d, const LeanSpec& sp,
                            std::vector<i128>& total, std::vector<int64_t>& unit, std::vector<uint8_t>& inexact,
                            std::vector<uint8_t>& got) {
    struct Range {
        int64_t lo = 0, hi = 0;
        bool any = false;
        void add(uint64_t sig, int64_t e) {
            if (sig == 0) return;
            lo = any ? std::min(lo, e) : e;
            hi = any ? std::max(hi, e) : e;
            any = true;
        }
    };
    const size_t m = (size_t)d.m, k = (size_t)d.k, n = (size_t)d.n, outs = d.outputs();
    total.assign(outs, 0);
    unit.assign(outs, 0);
    inexact.assign(outs, 0);
    got.assign(outs, 0);
    const SimdKernels* simd = simd_active();
    const size_t width = simd ? simd->width : kLanes;
    const ExactLeanFn kernel = simd ? simd->exact_lean : exact_lean_scalar;
    const int64_t unit0 = sp.has_p ? 60 - sp.P.M : 0;
    std::vector<Range> rrow(m), rcol(n);
    ExactSum sums[16];
    int64_t base[16];
    size_t idx = 0;
    for (size_t bi = 0; bi < (size_t)d.nbatch; ++bi) {
        const size_t oa = d.batch_a ? bi * m * k : 0, ob = d.batch_b ? bi * k * n : 0;
        if (bi == 0 || d.batch_a) {
            for (size_t i = 0; i < m; ++i) {
                rrow[i] = Range();
                for (size_t t = 0; t < k; ++t) rrow[i].add(A.sig[oa + i * k + t], A.exp[oa + i * k + t]);
            }
        }
        if (bi == 0 || d.batch_b) {
            for (size_t j = 0; j < n; ++j) rcol[j] = Range();
            for (size_t t = 0; t < k; ++t)
                for (size_t j = 0; j < n; ++j) rcol[j].add(B.sig[ob + t * n + j], B.exp[ob + t * n + j]);
        }
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; j += width) {
                const size_t lanes = std::min(width, n - j);
                bool valid[16];
                for (size_t c = 0; c < lanes; ++c) {
                    const Range &x = rrow[i], &y = rcol[j + c];
                    base[c] = x.any && y.any ? x.lo + y.lo : 0;
                    // + 1: a rounded product can sit one binade higher
                    valid[c] = !(x.any && y.any) || x.hi + y.hi - base[c] + 1 <= kExactMaxShift;
                }
                kernel(A.at(oa + i * k), B.at(ob + j), n, k, lanes, sp, base, sums);
                for (size_t c = 0; c < lanes; ++c, ++idx) {
                    if (!valid[c] || sums[c].bad) continue;
                    total[idx] = exact_total(sums[c]);
                    unit[idx] = base[c] + unit0;
                    inexact[idx] = (uint8_t)sums[c].inexact;
                    got[idx] = 1;
                }
            }
    }
}

// Matmul of narrow formats on the tight kernels (lean.hpp), SIMD when the CPU
// has it. False: an operand exponent or the length is out of their range.
// Outputs the kernels decline are left with ok = 0.
static bool lean_matmul(const std::vector<FV<u128>>& a, const std::vector<FV<u128>>& b, const MatDims& d,
                        const Fmt& F, const Fmt* P, int order, std::vector<uint64_t>& raw,
                        std::vector<uint8_t>& flags, std::vector<uint8_t>& ok, std::vector<DB>* unr) {
    const size_t m = (size_t)d.m, k = (size_t)d.k, n = (size_t)d.n;
    if (order == 0 && k >= (size_t(1) << kExactMaxLog2K)) return false;
    LeanOperands A, B;
    A.assign(a);
    B.assign(b);
    if (!A.ok || !B.ok) return false;
    const LeanSpec sp{rt_of(F), rt_of(P ? *P : F), P != nullptr};
    const SimdKernels* simd = simd_active();
    const size_t width = simd ? simd->width : kLanes;
    const RawFmt rf(F);
    LeanState st[kLeanRows * 16];
    if (order == 1) {
        const SeqLeanFn kernel = simd ? simd->seq_lean : seq_lean_scalar;
        for (size_t bi = 0; bi < (size_t)d.nbatch; ++bi) {
            const size_t oa = d.batch_a ? bi * m * k : 0, ob = d.batch_b ? bi * k * n : 0;
            for (size_t i = 0; i < m; i += kLeanRows) {
                const size_t rows = std::min(kLeanRows, m - i);
                LeanVec row[kLeanRows];
                for (size_t r = 0; r < rows; ++r) row[r] = A.at(oa + (i + r) * k);
                for (size_t j = 0; j < n; j += width) {
                    const size_t lanes = std::min(width, n - j);
                    kernel(row, rows, B.at(ob + j), n, k, lanes, sp, st);
                    for (size_t r = 0; r < rows; ++r)
                        for (size_t c = 0; c < lanes; ++c) {
                            const LeanState& x = st[r * width + c];
                            if (x.bad) continue;
                            const size_t idx = (bi * m + i + r) * n + j + c;
                            raw[idx] = raw_of_fv(lean_result(x, F.M), rf);
                            flags[idx] = (uint8_t)x.inexact;
                            ok[idx] = 1;
                        }
                }
            }
        }
        return true;
    }
    if (order == 2) {
        const PairLeanFn kernel = simd ? simd->pair_lean : pair_lean_scalar;
        std::vector<uint64_t> scratch(3 * std::max<size_t>(k, 1) * width);
        size_t idx = 0;
        for (size_t bi = 0; bi < (size_t)d.nbatch; ++bi) {
            const size_t oa = d.batch_a ? bi * m * k : 0, ob = d.batch_b ? bi * k * n : 0;
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < n; j += width) {
                    const size_t lanes = std::min(width, n - j);
                    kernel(A.at(oa + i * k), B.at(ob + j), n, k, lanes, sp, scratch.data(), st);
                    for (size_t c = 0; c < lanes; ++c, ++idx) {
                        if (st[c].bad) continue;
                        raw[idx] = raw_of_fv(lean_result(st[c], F.M), rf);
                        flags[idx] = (uint8_t)st[c].inexact;
                        ok[idx] = 1;
                    }
                }
        }
        return true;
    }
    // exact: the sum on one grid per output, then one rounding
    const RT rtF = rt_of(F);
    std::vector<i128> total;
    std::vector<int64_t> unit;
    std::vector<uint8_t> inexact, got;
    lean_exact_sums(A, B, d, sp, total, unit, inexact, got);
    for (size_t idx = 0; idx < total.size(); ++idx) {
        if (!got[idx]) continue;
        unsigned fl = inexact[idx];
        if (total[idx] == 0) {
            // The sign of a zero total depends on the order of the terms only
            // when rounding toward negative.
            if (F.rounding == RDN) continue;
            raw[idx] = 0;
            if (unr) (*unr)[idx] = DB();
        } else {
            const bool neg = total[idx] < 0;
            const u128 mag = neg ? (u128)(-total[idx]) : (u128)total[idx];
            FV<u128> r;
            if (!fast_round(FV<u128>{mag, unit[idx], neg}, false, rtF, r, fl)) continue;
            raw[idx] = raw_of_fv(r, rf);
            if (unr) (*unr)[idx] = DB{neg, unit[idx], to_big(mag), false};
        }
        flags[idx] = (uint8_t)fl;
        ok[idx] = 1;
    }
    return true;
}

// ---------------------------------------------------------------- grouped sums

// Exact a + b for values of up to 126 bits; false if the sum would not fit.
// A zero result takes the IEEE sign for the rounding mode (sv_add).
static bool wide_add(FV<u128>& a, const FV<u128>& b, uint8_t rounding) {
    if (b.sig == 0) {
        if (a.sig == 0 && a.neg != b.neg) a.neg = rounding == RDN;
        return true;
    }
    if (a.sig == 0) {
        a = b;
        return true;
    }
    const int64_t e = std::min(a.exp, b.exp);
    const int64_t sa = a.exp - e, sb = b.exp - e;
    if (fbitlen(a.sig) + sa > 124 || fbitlen(b.sig) + sb > 124) return false;
    const u128 x = a.sig << sa, y = b.sig << sb;
    if (a.neg == b.neg) a.sig = x + y;
    else if (x >= y) a.sig = x - y;
    else {
        a.sig = y - x;
        a.neg = b.neg;
    }
    a.exp = e;
    if (a.sig == 0) {
        a.exp = 0;
        a.neg = rounding == RDN;
    }
    return true;
}

// One dot product with products summed `group` at a time, optionally after
// aligning each group to its largest exponent (sum_terms and aligned_sum
// above, on machine integers). False means "use the general path".
static bool group_dot(const LeanVec& a, const LeanVec& b, size_t stride, size_t k, const AccSpec& spec,
                      const Fmt& Ff, const Fmt* Pf, std::vector<FV<uint64_t>>& terms, std::vector<FV<u128>>& adds,
                      FV<u128>& res, uint8_t& flags_out, DB* unr) {
    const RT F = rt_of(Ff), P = rt_of(Pf ? *Pf : Ff);
    unsigned fl = 0;
    terms.resize(k);
    for (size_t i = 0; i < k; ++i) {
        const size_t o = i * stride;
        FV<uint64_t> t{a.sig[i] * b.sig[o], a.exp[i] + b.exp[o], a.neg[i] != b.neg[o]};
        if (t.sig == 0) t.exp = 0;
        if (Pf && !fast_round(t, false, P, t, fl)) return false;
        terms[i] = t;
    }
    adds.clear();
    const size_t group = (size_t)spec.group;
    for (size_t lo = 0; lo < k; lo += group) {
        const size_t hi = std::min(k, lo + group);
        if (spec.aligned) {
            bool any = false;
            int64_t top = 0;
            for (size_t i = lo; i < hi; ++i) {
                if (terms[i].sig == 0) continue;
                const int64_t t = terms[i].exp + fbitlen(terms[i].sig) - 1;
                top = any ? std::max(top, t) : t;
                any = true;
            }
            if (any) {
                const int64_t p = top - spec.align_bits;
                for (size_t i = lo; i < hi; ++i) {
                    FV<uint64_t>& v = terms[i];
                    if (v.sig == 0 || v.exp >= p) continue;
                    const int64_t cut = p - v.exp;   // truncate toward zero below 2**p
                    if (cut >= 64) {
                        fl |= INEXACT;
                        v.sig = 0;
                    } else {
                        if (v.sig & ((uint64_t(1) << cut) - 1)) fl |= INEXACT;
                        v.sig >>= cut;
                    }
                    v.exp = p;
                }
            }
        }
        FV<u128> total{(u128)terms[lo].sig, terms[lo].sig ? terms[lo].exp : 0, terms[lo].neg};
        for (size_t i = lo + 1; i < hi; ++i) {
            const FV<u128> t{(u128)terms[i].sig, terms[i].sig ? terms[i].exp : 0, terms[i].neg};
            if (!wide_add(total, t, Ff.rounding)) return false;
        }
        adds.push_back(total);
    }
    bool st;
    if (spec.order == 0) {   // exact: one rounding
        FV<u128> total;
        for (const FV<u128>& t : adds)
            if (!wide_add(total, t, Ff.rounding)) return false;
        if (!fast_round(total, false, F, res, fl)) return false;
        if (unr) *unr = DB{total.sig != 0 && total.neg, total.exp, to_big(total.sig), false};
    } else if (spec.order == 1) {   // sequential
        FV<u128> acc;
        for (const FV<u128>& t : adds) {
            FV<u128> s;
            fast_add(acc, t, Ff.rounding, s, st);
            if (!fast_round(s, st, F, acc, fl)) return false;
        }
        res = acc;
    } else {   // pairwise tree
        for (FV<u128>& t : adds)
            if (!fast_round(t, false, F, t, fl)) return false;
        if (adds.empty()) adds.push_back(FV<u128>());
        size_t len = adds.size();
        while (len > 1) {
            size_t o = 0;
            for (size_t i = 0; i + 1 < len; i += 2) {
                FV<u128> s;
                fast_add(adds[i], adds[i + 1], Ff.rounding, s, st);
                if (!fast_round(s, st, F, adds[o++], fl)) return false;
            }
            if (len % 2) adds[o++] = adds[len - 1];
            len = o;
        }
        res = adds[0];
    }
    flags_out = (uint8_t)fl;
    return true;
}

static bool group_matmul(const std::vector<FV<u128>>& a, const std::vector<FV<u128>>& b, const MatDims& d,
                         const AccSpec& spec, const Fmt& F, const Fmt* P, std::vector<uint64_t>& raw,
                         std::vector<uint8_t>& flags, std::vector<uint8_t>& ok, std::vector<DB>* unr) {
    LeanOperands A, B;
    A.assign(a);
    B.assign(b);
    if (!A.ok || !B.ok) return false;
    const size_t m = (size_t)d.m, k = (size_t)d.k, n = (size_t)d.n;
    const RawFmt rf(F);
    std::vector<FV<uint64_t>> terms;
    std::vector<FV<u128>> adds;
    size_t idx = 0;
    for (size_t bi = 0; bi < (size_t)d.nbatch; ++bi) {
        const size_t oa = d.batch_a ? bi * m * k : 0, ob = d.batch_b ? bi * k * n : 0;
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j, ++idx) {
                FV<u128> r;
                uint8_t fl = 0;
                DB* u = unr ? &(*unr)[idx] : nullptr;
                if (group_dot(A.at(oa + i * k), B.at(ob + j), n, k, spec, F, P, terms, adds, r, fl, u)) {
                    raw[idx] = raw_of_fv(r, rf);
                    flags[idx] = fl;
                    ok[idx] = 1;
                }
            }
    }
    return true;
}

bool lean_matmul_check(const std::vector<FV<u128>>& a, const std::vector<FV<u128>>& b, const MatDims& d, const Fmt& F,
                       const Fmt* P, int order, std::vector<uint64_t>& raw, std::vector<uint8_t>& flags,
                       std::vector<uint8_t>& ok, std::vector<DB>* unr) {
    if (!fast_target(F) || (P && !fast_target(*P)) || F.M > kLeanMaxF || (P && P->M > kLeanMaxP)) return false;
    return lean_matmul(a, b, d, F, P, order, raw, flags, ok, order == 0 ? unr : nullptr);
}

bool fast_matmul(const std::vector<FV<u128>>& a, int64_t bits_a, const std::vector<FV<u128>>& b, int64_t bits_b,
                 const MatDims& d, const AccSpec& spec, std::vector<uint64_t>& raw, std::vector<uint8_t>& flags,
                 std::vector<uint8_t>& ok, std::vector<DB>* unr) {
    if (!g_fast) return false;
    const Fmt& F = fmt_of(spec.fmt);
    const Fmt* P = spec.product ? &fmt_of(spec.product) : nullptr;
    if (!fast_target(F) || (P && !fast_target(*P))) return false;
    if (spec.group != 1 || spec.aligned) {
        // grouped sums: narrow operands only (their products fit 64 bits)
        if (bits_a > kLeanBits || bits_b > kLeanBits) return false;
        const size_t outs = d.outputs();
        raw.assign(outs, 0);
        flags.assign(outs, 0);
        ok.assign(outs, 0);
        if (unr) unr->assign(spec.order == 0 ? outs : 0, DB());
        if (!group_matmul(a, b, d, spec, F, P, raw, flags, ok, spec.order == 0 ? unr : nullptr)) return false;
        for (uint8_t o : ok) ++(o ? g_fast_hits : g_fast_misses);
        return true;
    }
    // Widest significand the adder sees: a term, or the accumulator.
    int64_t prod = bits_a + bits_b;
    int64_t term = P ? P->M + 1 : prod;
    int64_t need = std::max({prod, term, F.M + 1});
    size_t outs = d.outputs();
    raw.assign(outs, 0);
    flags.assign(outs, 0);
    ok.assign(outs, 0);
    if (unr) unr->assign(spec.order == 0 ? outs : 0, DB());
    std::vector<DB>* u = spec.order == 0 ? unr : nullptr;
    // The tight kernels first; then, for what they leave (and for wider
    // operands), the general fast kernel on 64 or 128 bits.
    const bool lean = bits_a <= kLeanBits && bits_b <= kLeanBits && F.M <= kLeanMaxF && (!P || P->M <= kLeanMaxP) &&
                      lean_matmul(a, b, d, F, P, spec.order, raw, flags, ok, u);
    if (lean && spec.order == 1) {
        // the sequential tight kernel declines the same cases as the general fast one
    } else if (need <= 62 && spec.order != 0) fast_matmul_t<uint64_t>(a, b, d, F, P, spec.order, raw, flags, ok, u);
    else if (need <= 126) fast_matmul_t<u128>(a, b, d, F, P, spec.order, raw, flags, ok, u);
    else if (!lean) return false;
    for (uint8_t o : ok) ++(o ? g_fast_hits : g_fast_misses);
    return true;
}

// An exact Python number (int, finite float, Fraction with a power-of-two
// denominator) as sig * 2**exp with sig odd and below 2**62; false if it is
// something else or too wide. As an operand a number has no negative zero.
bool number_fv(PyObject* o, FV<u128>& v) {
    uint64_t mag;
    int64_t exp = 0;
    bool neg;
    if (PyFloat_Check(o)) {
        const double d = PyFloat_AS_DOUBLE(o);
        if (!double_finite(d)) return false;
        double_parts(d, mag, exp);
        neg = mag != 0 && double_neg(d);
    } else {
        PyObject* num = o;
        nb::object hn, hd;
        if (!PyLong_Check(o)) {
            if (!is_fraction(o)) return false;
            static PyObject* sn = PyUnicode_InternFromString("_numerator");
            static PyObject* sd = PyUnicode_InternFromString("_denominator");
            hn = steal_checked(PyObject_GetAttr(o, sn));
            hd = steal_checked(PyObject_GetAttr(o, sd));
            int overflow = 0;
            const long long den = PyLong_AsLongLongAndOverflow(hd.ptr(), &overflow);
            if (den == -1 && PyErr_Occurred()) raise_current();
            if (overflow || den <= 0 || (den & (den - 1))) return false;
            exp = -(bitlen((uint64_t)den) - 1);
            num = hn.ptr();
        }
        int overflow = 0;
        const long long x = PyLong_AsLongLongAndOverflow(num, &overflow);
        if (x == -1 && PyErr_Occurred()) raise_current();
        if (overflow || x == LLONG_MIN) return false;
        neg = x < 0;
        mag = (uint64_t)(neg ? -x : x);
    }
    if (mag == 0) {
        v = FV<u128>();
        return true;
    }
    const int tz = (int)std::countr_zero((uint64_t)mag);
    mag >>= tz;
    if (mag >> 62) return false;
    v = {mag, exp + tz, neg};
    return true;
}

// Operands for the fast kernels: finite FP values of narrow formats and
// exact numbers. `bits` bounds their significand widths. False if any item
// is something else.
static bool parse_fast(nb::handle seq, std::vector<FV<u128>>& out, int64_t& bits) {
    if (!PyList_Check(seq.ptr())) return false;
    Py_ssize_t n = PyList_GET_SIZE(seq.ptr());
    out.resize((size_t)n);
    bits = 1;
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* o = PyList_GET_ITEM(seq.ptr(), i);
        if (is_fp(o)) [[likely]] {
            PyFP* p = (PyFP*)o;
            const Fmt& f = *p->f;
            if (f.M > 61 || !is_finite(p->c, f)) return false;
            out[(size_t)i] = fv_of<u128>(p->c, f);
            bits = std::max(bits, f.M + 1);
        } else {
            if (!number_fv(o, out[(size_t)i])) return false;
            if (out[(size_t)i].sig != 0) bits = std::max(bits, fbitlen((uint64_t)out[(size_t)i].sig));
        }
    }
    return true;
}

// Accumulator.sum_products(a, b, init); None means "use the Python path".
static nb::object py_sum_products(nb::handle a, nb::handle b, nb::handle init, nb::handle acc) {
    AccSpec spec;
    if (!spec_from(acc, spec)) return nb::none();
    bool has_init = !init.is_none();
    if (!has_init) {
        std::vector<FV<u128>> qa, qb;
        int64_t ba, bb;
        std::vector<uint64_t> raw;
        std::vector<uint8_t> flags, ok;
        std::vector<DB> unr;
        if (parse_fast(a, qa, ba) && parse_fast(b, qb, bb) && qa.size() == qb.size() &&
            fast_matmul(qa, ba, qb, bb, {1, 1, (int64_t)qa.size(), 1, false, false}, spec, raw, flags, ok, &unr) &&
            ok[0]) {
            Unr u;
            if (spec.order == 0) u = unr_big(unr[0]);
            return finish_plain(spec.fmt, code_of_raw64(raw[0], fmt_of(spec.fmt)), flags[0], std::move(u));
        }
    }
    std::vector<In> va, vb;
    bool nf = false, nums = false;
    if (!parse_all(a, va, nf, nums) || !parse_all(b, vb, nf, nums)) return nb::none();
    In vi;
    if (has_init) {
        if (!parse_in(init.ptr(), vi)) return nb::none();
        if (vi.fp) nf |= !fp_finite(vi.fp);
        else nums = true;
    }
    if ((nf && nums) || va.size() != vb.size()) return nb::none();
    return sum_products(va.data(), vb.data(), va.size(), 1, 1, has_init ? &vi : nullptr, spec);
}

// blockscale._reduce over a whole matmul: results in C order, or None.
static nb::object py_matmul(nb::handle fa, nb::handle fb, int64_t nbatch, int64_t m, int64_t k, int64_t n,
                            bool batch_a, bool batch_b, nb::handle acc) {
    AccSpec spec;
    bool exact = acc.is_none();
    if (!exact && !spec_from(acc, spec)) return nb::none();
    MatDims d{nbatch, m, k, n, batch_a, batch_b};
    std::vector<uint64_t> raw;
    std::vector<uint8_t> flags, ok;
    std::vector<DB> unr;
    bool fast = false;
    if (!exact) {
        std::vector<FV<u128>> qa, qb;
        int64_t ba, bb;
        fast = parse_fast(fa, qa, ba) && parse_fast(fb, qb, bb) &&
               qa.size() == (size_t)((batch_a ? nbatch : 1) * m * k) &&
               qb.size() == (size_t)((batch_b ? nbatch : 1) * k * n) &&
               fast_matmul(qa, ba, qb, bb, d, spec, raw, flags, ok, &unr);
    }
    // Exact results (Fractions): the same sums, not rounded.
    std::vector<i128> xtotal;
    std::vector<int64_t> xunit;
    std::vector<uint8_t> xgot;
    if (exact && g_fast) {
        std::vector<FV<u128>> qa, qb;
        int64_t ba, bb;
        if (parse_fast(fa, qa, ba) && parse_fast(fb, qb, bb) && ba <= kLeanBits && bb <= kLeanBits &&
            qa.size() == (size_t)((batch_a ? nbatch : 1) * m * k) &&
            qb.size() == (size_t)((batch_b ? nbatch : 1) * k * n) && (size_t)k < (size_t(1) << kExactMaxLog2K)) {
            LeanOperands A, B;
            A.assign(qa);
            B.assign(qb);
            if (A.ok && B.ok) {
                std::vector<uint8_t> inexact;
                lean_exact_sums(A, B, d, LeanSpec{RT(), RT(), false}, xtotal, xunit, inexact, xgot);
                for (uint8_t g : xgot) ++(g ? g_fast_hits : g_fast_misses);
            }
        }
    }
    std::vector<In> va, vb;
    bool parsed = false;
    auto general = [&]() {   // false: the Python path must handle this input
        if (parsed) return true;
        bool nf = false, nums = false;
        if (!parse_all(fa, va, nf, nums) || !parse_all(fb, vb, nf, nums)) return false;
        if (exact ? nf : (nf && nums)) return false;   // the reference raises its own error
        parsed = true;
        return true;
    };
    if (!fast && xgot.empty() && !general()) return nb::none();
    nb::list out;
    size_t idx = 0;
    for (int64_t bi = 0; bi < nbatch; ++bi) {
        int64_t oa = batch_a ? bi * m * k : 0, ob = batch_b ? bi * k * n : 0;
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < n; ++j, ++idx) {
                if (fast && ok[idx]) {
                    const Fmt& F = fmt_of(spec.fmt);
                    Unr u;
                    if (spec.order == 0) u = unr_big(unr[idx]);
                    out.append(finish_plain(spec.fmt, code_of_raw64(raw[idx], F), flags[idx], std::move(u)));
                    continue;
                }
                if (!xgot.empty() && xgot[idx]) {
                    const i128 t = xtotal[idx];
                    out.append(fraction_dy<u128>(t < 0, t < 0 ? (u128)(-t) : (u128)t, xunit[idx]));
                    continue;
                }
                if (!general()) return nb::none();
                const In* row = va.data() + oa + i * k;
                const In* col = vb.data() + ob + j;
                out.append(exact ? exact_dot(row, col, (size_t)k, 1, (size_t)n)
                                 : sum_products(row, col, (size_t)k, 1, (size_t)n, nullptr, spec));
            }
    }
    return out;
}

// Exact dot of two equal-length vectors (no inf/NaN), or None.
static nb::object py_exact_dot(nb::handle a, nb::handle b) {
    std::vector<In> va, vb;
    bool nf = false, nums = false;
    if (!parse_all(a, va, nf, nums) || !parse_all(b, vb, nf, nums) || nf || va.size() != vb.size())
        return nb::none();
    return exact_dot(va.data(), vb.data(), va.size(), 1, 1);
}

void register_accum(nb::module_& m) {
    // Tests switch the fast kernels off to compare them with the general path.
    m.def("set_fast", [](bool on) { bool was = g_fast; g_fast = on; return was; });
    m.def("set_simd", [](const std::string& name) {
        // "avx512", "avx2", or "none"; an unsupported name selects the scalar kernels
        std::string was = g_simd ? g_simd->name : "none";
        g_simd = simd_by_name(name);
        return was;
    });
    m.def("simd", []() { return std::string(g_simd ? g_simd->name : "none"); });
    m.def("simd_available", []() {
        nb::list r;
        for (const char* nm : {"avx2", "avx512"})
            if (simd_by_name(nm)) r.append(nm);
        return r;
    });
    m.def("fast_stats", [](bool reset) {
        nb::object r = nb::make_tuple(g_fast_hits, g_fast_misses);
        if (reset) g_fast_hits = g_fast_misses = 0;
        return r;
    }, nb::arg("reset") = false);
    m.def("acc_sum_products", &py_sum_products, nb::arg(), nb::arg(), nb::arg().none(), nb::arg());
    m.def("matmul_reduce", &py_matmul, nb::arg(), nb::arg(), nb::arg(), nb::arg(), nb::arg(), nb::arg(),
          nb::arg(), nb::arg(), nb::arg().none());
    m.def("exact_dot", &py_exact_dot);
}

}  // namespace vf
