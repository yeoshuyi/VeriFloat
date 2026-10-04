// Block quantization (BlockTensor.quantize in verifloat_py/blockscale.py)
// and dequantization, on exact rationals. Same operation order as the
// reference, so stochastic-rounding draws and errors match.
#include "accum.hpp"
#include "fp.hpp"
#include "uint.hpp"

#include <vector>

namespace vf {

// ---------------------------------------------------------------- rationals

struct Q {
    BigInt n = 0;
    BigInt d = 1;   // > 0
};

static void norm(Q& q) {
    if (q.d == 1) return;
    if (q.n == 0) { q.d = 1; return; }
    BigInt g = boost::multiprecision::gcd(abs(q.n), q.d);
    if (g != 1) { q.n /= g; q.d /= g; }
}
static Q qmul(const Q& a, const Q& b) { Q r{a.n * b.n, a.d * b.d}; norm(r); return r; }
static nb::object q_to_fraction(const Q& q);
static Q qdiv(const Q& a, const Q& b) {
    if (b.n == 0) {
        // The reference divides Fractions: raise what that division raises.
        nb::object fa = q_to_fraction(a), fb = q_to_fraction(b);
        steal_checked(PyNumber_TrueDivide(fa.ptr(), fb.ptr()));
        raise(PyExc_ZeroDivisionError, "division by zero");
    }
    Q r{a.n * b.d, a.d * b.n};
    if (r.d < 0) { r.n = -r.n; r.d = -r.d; }
    norm(r);
    return r;
}
static Q qsub(const Q& a, const Q& b) { Q r{a.n * b.d - b.n * a.d, a.d * b.d}; norm(r); return r; }
static Q qadd(const Q& a, const Q& b) { Q r{a.n * b.d + b.n * a.d, a.d * b.d}; norm(r); return r; }
static Q qabs(const Q& a) { return {abs(a.n), a.d}; }
static int qcmp(const Q& a, const Q& b) {
    BigInt l = a.n * b.d, r = b.n * a.d;
    return l < r ? -1 : l > r ? 1 : 0;
}
static bool qzero(const Q& a) { return a.n == 0; }
static Q qint(int64_t v) { return {BigInt(v), 1}; }
static Q qpow2(int64_t k) { return k >= 0 ? Q{BigInt(1) << (unsigned)k, 1} : Q{1, BigInt(1) << (unsigned)-k}; }

static Q q_from_dy(const Dy<BigInt>& v) {
    BigInt s = v.neg ? BigInt(-v.sig) : v.sig;
    Q q = v.exp >= 0 ? Q{s << (unsigned)v.exp, 1} : Q{s, BigInt(1) << (unsigned)-v.exp};
    norm(q);
    return q;
}

static int64_t floor_log2(const Q& x) {   // x > 0
    int64_t e = bitlen(x.n) - bitlen(x.d);
    BigInt l = x.n << (unsigned)std::max<int64_t>(0, -e), r = x.d << (unsigned)std::max<int64_t>(0, e);
    return l < r ? e - 1 : e;
}

static Q q_from_fraction(PyObject* f) {
    static PyObject* sn = PyUnicode_InternFromString("_numerator");
    static PyObject* sd = PyUnicode_InternFromString("_denominator");
    nb::object n = steal_checked(PyObject_GetAttr(f, sn)), d = steal_checked(PyObject_GetAttr(f, sd));
    return {big_from_long(n.ptr()), big_from_long(d.ptr())};
}

static nb::object q_to_fraction(const Q& q) {
    return fraction_nd(long_from_big(q.n), long_from_big(q.d));
}

// blockscale._to_fraction, natively for the common types.
static Q parse_value(PyObject* v, PyObject* to_fraction) {
    if (is_fp(v)) {
        PyFP* x = (PyFP*)v;
        if (is_finite(x->c, *x->f)) return q_from_dy(decode<BigInt>(x->c, *x->f));
    } else if (PyLong_Check(v)) {
        return {big_from_long(v), 1};
    } else if (PyFloat_Check(v)) {
        double d = PyFloat_AS_DOUBLE(v);
        if (double_finite(d)) {
            uint64_t sig;
            int64_t exp;
            double_parts(d, sig, exp);
            if (sig == 0) return Q{};
            return q_from_dy({double_neg(d), exp, BigInt(sig), false});
        }
    } else if (is_fraction(v)) {
        return q_from_fraction(v);
    } else if (is_uint(v)) {
        nb::object i = uint_val(v);
        return {big_from_long(i.ptr()), 1};
    }
    // Everything else (and the error cases) through the reference conversion.
    nb::object f = steal_checked(call_one(to_fraction, v));
    return q_from_fraction(f.ptr());
}

// ---------------------------------------------------------------- dyadic fast path
// Most tensors hold floats (or FP values), and most scales are dyadic too.
// Then |x| comparisons and x / step need no arbitrary-precision rationals: a
// value is kept as sig * 2**exp (FV, sig odd and below 2**62) and turned into
// a Q only where the general code needs one.

// A finite value as an FV with an odd significand; false if it is not dyadic
// or too wide. A zero has no sign (as in Q).
static bool value_fv(PyObject* v, FV<u128>& out) {
    if (is_fp(v)) {
        const PyFP* x = (const PyFP*)v;
        const Fmt& f = *x->f;
        if (f.M > 61 || !is_finite(x->c, f)) return false;
        FV<uint64_t> d = fv_of<uint64_t>(x->c, f);
        if (d.sig == 0) {
            out = FV<u128>();
            return true;
        }
        const int tz = (int)std::countr_zero((uint64_t)d.sig);
        out = {(u128)(d.sig >> tz), d.exp + tz, d.neg};
        return true;
    }
    return number_fv(v, out);
}

static Q q_from_fv(const FV<u128>& v) {   // already in lowest terms: sig is odd
    BigInt s = to_big(v.sig);
    if (v.neg) s = -s;
    return v.exp >= 0 ? Q{s << (unsigned)v.exp, 1} : Q{s, BigInt(1) << (unsigned)-v.exp};
}

static bool fv_from_q(const Q& q, FV<u128>& out) {
    if (q.n == 0) {
        out = FV<u128>();
        return true;
    }
    if ((q.d & (q.d - 1)) != 0) return false;
    BigInt a = abs(q.n);
    const int64_t tz = (int64_t)boost::multiprecision::lsb(a);
    a >>= (unsigned)tz;
    if (bitlen(a) > 62) return false;
    out = {to_u128(a), tz - (bitlen(q.d) - 1), q.n < 0};
    return true;
}

// |a| > |b| for dyadic values.
static bool fv_abs_gt(const FV<u128>& a, const FV<u128>& b) {
    if (a.sig == 0) return false;
    if (b.sig == 0) return true;
    const int64_t ta = a.exp + fbitlen((uint64_t)a.sig), tb = b.exp + fbitlen((uint64_t)b.sig);
    if (ta != tb) return ta > tb;
    // same binade: compare the significands aligned at their leading bits
    const int la = (int)fbitlen((uint64_t)a.sig), lb = (int)fbitlen((uint64_t)b.sig);
    return ((uint64_t)a.sig << (64 - la)) > ((uint64_t)b.sig << (64 - lb));
}

// a < b (signed) for dyadic values.
static bool fv_lt(const FV<u128>& a, const FV<u128>& b) {
    const bool an = a.sig != 0 && a.neg, bn = b.sig != 0 && b.neg;
    if (an != bn) return an;
    return an ? fv_abs_gt(a, b) : fv_abs_gt(b, a);
}

// The tensor's values: dyadic where possible, exact rationals on demand.
struct Values {
    std::vector<FV<u128>> fv;
    std::vector<uint8_t> has_fv;
    mutable std::vector<Q> q;
    mutable std::vector<uint8_t> has_q;
    bool all_fv = true;

    const Q& at(size_t i) const {
        if (!has_q[i]) {
            q[i] = q_from_fv(fv[i]);
            has_q[i] = 1;
        }
        return q[i];
    }
};

// x / step rounded into f (a signed format with a zero), for dyadic x and
// step != 0: the code and flags, or false to decline. A zero result of a zero
// x is +0, as round_ratio(..., zero_sign = false) gives.
static bool fast_quotient(const FV<u128>& x, const FV<u128>& step, const Fmt& f, FCode& c, unsigned& fl) {
    if (x.sig == 0) return fast_round_code(FV<uint64_t>(), false, f, c, fl);
    const uint64_t xs = (uint64_t)x.sig, ss = (uint64_t)step.sig;
    const bool neg = x.neg != step.neg;
    // quotient with at least M + 3 bits, the remainder as sticky
    int64_t k = f.M + 4 + fbitlen(ss) - fbitlen(xs);
    if (k < 0) k = 0;
    if (fbitlen(xs) + k <= 64) {
        const uint64_t N = xs << k;
        return fast_round_code(FV<uint64_t>{N / ss, x.exp - step.exp - k, neg}, N % ss != 0, f, c, fl);
    }
    if (fbitlen(xs) + k > 126) return false;
    const u128 N = (u128)xs << k;
    return fast_round_code(FV<u128>{N / ss, x.exp - step.exp - k, neg}, N % ss != 0, f, c, fl);
}

// ---------------------------------------------------------------- formats

struct IntF {   // IntFormat
    int64_t bits = 8;
    bool is_signed = true;
    uint8_t rounding = RNE;
    int64_t frac = 0;
    BigInt lo, hi;
    PyTypeObject* cls = nullptr;
};

static IntF int_spec(nb::handle t) {
    IntF f;
    f.bits = nb::cast<int64_t>(t[0]);
    f.is_signed = nb::cast<bool>(t[1]);
    f.rounding = (uint8_t)nb::cast<int>(t[2]);
    f.frac = nb::cast<int64_t>(t[3]);
    f.lo = f.is_signed ? BigInt(-(BigInt(1) << (unsigned)(f.bits - 1))) : BigInt(0);
    f.hi = (BigInt(1) << (unsigned)(f.bits - (f.is_signed ? 1 : 0))) - 1;
    f.cls = f.is_signed ? S.INT : S.UINT;
    return f;
}

struct Spec {
    PyObject* elem_fp = nullptr;   // FPFormat elements, or
    IntF elem_int;                 // IntFormat elements
    Fmt elem_sat;                  // elem.replace(saturate=True, wrap=False)
    Q elem_max;
    int64_t block = 16;
    int scale_kind = 0;            // 0 none, 1 FP, 2 Pow2
    PyObject* scale_fp = nullptr;
    int64_t p2_bits = 8, p2_bias = 127, p2_max = 254;
    Q scale_cap;
    bool has_zp = false;
    IntF zp;
    PyObject* tscale = nullptr;
    PyObject* compute = nullptr;
    bool has_smin = false, has_zscale = false;
    Q scale_min, zero_scale;
    PyObject* to_fraction = nullptr;
};

static Q q_of(nb::handle f) { return q_from_fraction(f.ptr()); }

static Spec spec_from(nb::handle s) {
    // (elem_fp, elem_int, elem_max, block, scale_kind, scale_fp, pow2, scale_cap,
    //  zero_point, tensor_scale, compute, scale_min, zero_scale, to_fraction)
    Spec r;
    if (!s[0].is_none()) {
        r.elem_fp = s[0].ptr();
        r.elem_sat = fmt_of(r.elem_fp);
        r.elem_sat.saturate = true;
        r.elem_sat.wrap = false;
        r.elem_sat.id = -1;
        r.elem_sat.derive();
    } else r.elem_int = int_spec(s[1]);
    r.elem_max = q_of(s[2]);
    r.block = nb::cast<int64_t>(s[3]);
    r.scale_kind = nb::cast<int>(s[4]);
    if (!s[5].is_none()) r.scale_fp = s[5].ptr();
    if (!s[6].is_none()) {
        r.p2_bits = nb::cast<int64_t>(s[6][0]);
        r.p2_bias = nb::cast<int64_t>(s[6][1]);
        r.p2_max = nb::cast<int64_t>(s[6][2]);
    }
    r.scale_cap = q_of(s[7]);
    if (!s[8].is_none()) { r.has_zp = true; r.zp = int_spec(s[8]); }
    if (!s[9].is_none()) r.tscale = s[9].ptr();
    if (!s[10].is_none()) r.compute = s[10].ptr();
    if (!s[11].is_none()) { r.has_smin = true; r.scale_min = q_of(s[11]); }
    if (!s[12].is_none()) { r.has_zscale = true; r.zero_scale = q_of(s[12]); }
    r.to_fraction = s[13].ptr();
    return r;
}

// Round an exact rational into a Python format: (code, flags).
static RoundOut round_q(const Q& x, const Fmt& f, SRArg& sr) { return round_ratio(x.n, x.d, f, false, sr); }

// FP(...).exact of a rounding result; non-finite results raise as FP.exact does.
static Q exact_or_raise(PyObject* fmt, const Code& c) {
    const Fmt& f = fmt_of(fmt);
    if (!is_finite(c, f)) {
        nb::object o = fp_plain_obj(fmt, c);
        steal_checked(PyObject_GetAttrString(o.ptr(), "exact"));
    }
    return q_from_dy(decode<BigInt>(c, f));
}

// BlockFormat._c: one step of the recipe's intermediate arithmetic.
static Q cstep(const Spec& s, const Q& x) {
    if (!s.compute || qzero(x)) return x;
    SRArg sr;
    RoundOut r = round_q(x, fmt_of(s.compute), sr);
    return exact_or_raise(s.compute, r.c);
}

// IntFormat.quantize(x): round to an integer code and saturate.
static std::pair<nb::object, uint8_t> int_quantize(const IntF& f, const Q& x) {
    bool neg = x.n < 0;
    BigInt a = abs(x.n), q, r;
    boost::multiprecision::divide_qr(a, x.d, q, r);
    bool inexact = r != 0;
    if (inexact) {
        bool up;
        BigInt r2 = r * 2;
        switch (f.rounding) {
            case RNE: up = r2 > x.d || (r2 == x.d && bit(q, 0)); break;
            case RNA: up = r2 >= x.d; break;
            case RTZ: up = false; break;
            case RUP: up = !neg; break;
            case RDN: up = neg; break;
            default: {   // SR with no random bits (sr=0, sr_bits=8)
                BigInt q2, rr;
                boost::multiprecision::divide_qr(BigInt(r << 8), x.d, q2, rr);
                BigInt rr2 = rr * 2;
                if (rr2 > x.d || (rr2 == x.d && bit(q2, 0))) q2 += 1;
                up = q2 >= 256;
            }
        }
        if (up) q += 1;
    }
    if (neg) q = -q;
    uint8_t flags = inexact ? INEXACT : 0;
    if (q > f.hi) { q = f.hi; flags = OVERFLOW | INEXACT; }
    else if (q < f.lo) {
        bool under = neg && !f.is_signed;
        q = f.lo;
        flags = (under ? UNDERFLOW : OVERFLOW) | INEXACT;
    }
    nb::object v = long_from_big(q);
    return {uint_make(f.cls, v.ptr(), f.bits, false), flags};
}

// ---------------------------------------------------------------- quantize

static nb::object py_quantize(nb::handle flat_seq, nb::handle shape_t, int64_t axis, nb::handle spec_t) {
    Spec s = spec_from(spec_t);
    nb::object flat = steal_checked(PySequence_Fast(flat_seq.ptr(), "expected a list"));
    Py_ssize_t N = PySequence_Fast_GET_SIZE(flat.ptr());
    PyObject** items = PySequence_Fast_ITEMS(flat.ptr());
    Values vals;
    vals.fv.resize((size_t)N);
    vals.has_fv.assign((size_t)N, 0);
    vals.q.resize((size_t)N);
    vals.has_q.assign((size_t)N, 0);
    for (Py_ssize_t i = 0; i < N; ++i) {
        if (g_fast && value_fv(items[i], vals.fv[(size_t)i])) vals.has_fv[(size_t)i] = 1;
        else {
            vals.q[(size_t)i] = parse_value(items[i], s.to_fraction);
            vals.has_q[(size_t)i] = 1;
            vals.all_fv = false;
        }
    }

    // Rows run along `axis`, ordered over the other axes in C order.
    std::vector<int64_t> shape;
    for (nb::handle h : shape_t) shape.push_back(nb::cast<int64_t>(h));
    int64_t nd = (int64_t)shape.size(), n = shape[(size_t)axis];
    std::vector<int64_t> st((size_t)nd);
    int64_t acc = 1;
    for (int64_t i = nd - 1; i >= 0; --i) { st[(size_t)i] = acc; acc *= shape[(size_t)i]; }
    std::vector<int64_t> others;
    for (int64_t i = 0; i < nd; ++i)
        if (i != axis) others.push_back(i);
    int64_t nrows = n ? N / n : 0;
    std::vector<int64_t> base((size_t)nrows);
    {
        std::vector<int64_t> idx(others.size(), 0);
        for (int64_t r = 0; r < nrows; ++r) {
            int64_t off = 0;
            for (size_t k = 0; k < others.size(); ++k) off += idx[k] * st[(size_t)others[k]];
            base[(size_t)r] = off;
            for (int64_t k = (int64_t)others.size() - 1; k >= 0; --k) {
                if (++idx[(size_t)k] < shape[(size_t)others[(size_t)k]]) break;
                idx[(size_t)k] = 0;
            }
        }
    }
    int64_t step_ax = st[(size_t)axis];
    auto pos = [&](int64_t r, int64_t c) { return (size_t)(base[(size_t)r] + c * step_ax); };
    auto at = [&](int64_t r, int64_t c) -> const Q& { return vals.at(pos(r, c)); };
    // max |x| over the columns [start, stop) of a row
    auto row_amax = [&](int64_t r, int64_t start, int64_t stop) {
        if (vals.all_fv) {
            size_t best = pos(r, start);
            for (int64_t c = start + 1; c < stop; ++c)
                if (fv_abs_gt(vals.fv[pos(r, c)], vals.fv[best])) best = pos(r, c);
            return qabs(vals.at(best));
        }
        Q amax;
        for (int64_t c = start; c < stop; ++c)
            if (qcmp(qabs(at(r, c)), amax) > 0) amax = qabs(at(r, c));
        return amax;
    };
    // The element formats the fast path can round into.
    const bool fast_elems = g_fast && s.elem_fp && s.elem_sat.fast && vals.all_fv &&
                            (!s.compute || fmt_of(s.compute).fast);

    uint8_t flags = 0;
    // Tensor scale
    nb::object tscale = nb::none();
    Q s_t = qint(1);
    if (s.tscale) {
        Q amax;
        if (vals.all_fv && N > 0) {
            size_t best = 0;
            for (size_t i = 1; i < (size_t)N; ++i)
                if (fv_abs_gt(vals.fv[i], vals.fv[best])) best = i;
            amax = qabs(vals.at(best));
        } else {
            for (size_t i = 0; i < (size_t)N; ++i)
                if (qcmp(qabs(vals.at(i)), amax) > 0) amax = qabs(vals.at(i));
        }
        Q target = qzero(amax) ? qint(1) : cstep(s, qdiv(amax, cstep(s, qmul(s.elem_max, s.scale_cap))));
        SRArg sr;
        RoundOut r = round_q(target, fmt_of(s.tscale), sr);
        tscale = fp_plain_obj(s.tscale, r.c);
        Q ex = exact_or_raise(s.tscale, r.c);
        if (ex.n <= 0) {
            nb::object tf = q_to_fraction(target);
            nb::object fl = steal_checked(PyNumber_Float(tf.ptr()));
            raise(PyExc_ValueError, "tensor scale " + py_repr(fl.ptr()) + " underflowed " + py_str(s.tscale));
        }
        s_t = ex;
    }

    nb::list elems, scales, zeros;
    for (int64_t r = 0; r < nrows; ++r) {
        nb::list er, sr_, zr;
        for (int64_t start = 0; start < n; start += s.block) {
            int64_t stop = std::min(n, start + s.block);
            // Block scale
            nb::object scale_obj = nb::none(), zero_obj = nb::none();
            Q s_b = qint(1), z;
            if (s.scale_kind == 2) {
                Q amax = row_amax(r, start, stop);
                amax = qdiv(amax, s_t);
                int64_t code = 0;   // smallest scale, as OCP reference code (gfloat) does
                if (!qzero(amax)) {
                    int64_t shared = floor_log2(amax) - floor_log2(s.elem_max);
                    code = std::min(std::max<int64_t>(shared + s.p2_bias, 0), s.p2_max);
                }
                scale_obj = nb::steal(uint_from_raw64(S.UINT, (uint64_t)code, s.p2_bits, false));
                s_b = qpow2(code - s.p2_bias);
            } else if (s.scale_kind == 1) {
                Q span, lo, hi;
                if (!s.has_zp) {
                    const Q amax = row_amax(r, start, stop);
                    span = cstep(s, qdiv(amax, cstep(s, qmul(s.elem_max, s_t))));
                } else {
                    const IntF& e = s.elem_int;
                    if (vals.all_fv) {   // min(x, 0) and max(x, 0), as the loop below gives
                        size_t ilo = pos(r, start), ihi = ilo;
                        for (int64_t c = start + 1; c < stop; ++c) {
                            if (fv_lt(vals.fv[pos(r, c)], vals.fv[ilo])) ilo = pos(r, c);
                            if (fv_lt(vals.fv[ihi], vals.fv[pos(r, c)])) ihi = pos(r, c);
                        }
                        if (qcmp(vals.at(ilo), lo) < 0) lo = vals.at(ilo);
                        if (qcmp(vals.at(ihi), hi) > 0) hi = vals.at(ihi);
                    } else
                        for (int64_t c = start; c < stop; ++c) {
                            if (qcmp(at(r, c), lo) < 0) lo = at(r, c);
                            if (qcmp(at(r, c), hi) > 0) hi = at(r, c);
                        }
                    Q range{e.hi - e.lo, 1};
                    span = cstep(s, qdiv(qsub(hi, lo), cstep(s, qmul(qmul(range, qpow2(-e.frac)), s_t))));
                }
                if (qzero(span) && s.has_zscale) span = s.zero_scale;
                else if (!qzero(span) && s.has_smin && qcmp(span, s.scale_min) < 0) span = s.scale_min;
                // Clamp before rounding so a scale never overflows (to inf/NaN).
                Q capped = qcmp(span, s.scale_cap) <= 0 ? span : s.scale_cap;
                SRArg srr;
                RoundOut ro = round_q(capped, fmt_of(s.scale_fp), srr);
                scale_obj = fp_plain_obj(s.scale_fp, ro.c);
                s_b = exact_or_raise(s.scale_fp, ro.c);
                if (s.has_zp) {
                    if (qzero(s_b)) {
                        nb::object zero = nb::int_(0);
                        zero_obj = uint_make(s.zp.cls, zero.ptr(), s.zp.bits, false);
                        z = Q{};
                    } else {
                        const IntF& e = s.elem_int;
                        Q denom = qmul(qmul(s_b, s_t), qpow2(-e.frac));
                        auto [zo, zf] = int_quantize(s.zp, qsub(Q{e.lo, 1}, qdiv(lo, denom)));
                        (void)zf;
                        zero_obj = zo;
                        nb::object zv = uint_val(zo.ptr());
                        z = Q{big_from_long(zv.ptr()), 1};
                    }
                }
            }
            sr_.append(scale_obj);
            zr.append(zero_obj);
            Q step = cstep(s, qmul(s_b, s_t));
            FV<u128> fstep;
            const bool fast_block = fast_elems && !qzero(step) && fv_from_q(step, fstep);
            for (int64_t c = start; c < stop; ++c) {
                if (fast_block) {
                    // Dyadic x and step: x / step on machine integers, rounded
                    // (through the compute format, if any) into the element.
                    const FV<u128>& x = vals.fv[pos(r, c)];
                    FCode code;
                    unsigned fl = 0, ignored = 0;
                    bool done;
                    if (!s.compute) done = fast_quotient(x, fstep, s.elem_sat, code, fl);
                    else {
                        const Fmt& C = fmt_of(s.compute);
                        FCode mid;
                        done = fast_quotient(x, fstep, C, mid, ignored);
                        if (done) {
                            FV<uint64_t> v = fv_of<uint64_t>(mid.code(), C);
                            if (v.sig == 0) v.neg = false;   // a rational zero has no sign
                            done = fast_round_code(v, false, s.elem_sat, code, fl);
                        }
                    }
                    if (done) {
                        er.append(fp_plain_obj(s.elem_fp, code.code()));
                        flags |= (uint8_t)fl;
                        continue;
                    }
                }
                if (s.elem_fp && !s.compute) {
                    // x / step rounded directly (no normalization needed)
                    SRArg sre;
                    const Q& x = at(r, c);
                    RoundOut ro = qzero(step) ? round_ratio(BigInt(0), BigInt(1), s.elem_sat, false, sre)
                                              : round_ratio(x.n * step.d, x.d * step.n, s.elem_sat, false, sre);
                    er.append(fp_plain_obj(s.elem_fp, ro.c));
                    flags |= ro.flags;
                    continue;
                }
                Q v = qzero(step) ? Q{} : cstep(s, qdiv(at(r, c), step));   // zero scale: zero block
                if (s.elem_fp) {
                    // Quantizers saturate elements: no inf/NaN or wrapped codes.
                    SRArg sre;
                    RoundOut ro = round_q(v, s.elem_sat, sre);
                    er.append(fp_plain_obj(s.elem_fp, ro.c));
                    flags |= ro.flags;
                } else {
                    auto [o, fl] = int_quantize(s.elem_int, qadd(qmul(v, qpow2(s.elem_int.frac)), z));
                    er.append(o);
                    flags |= fl;
                }
            }
        }
        elems.append(er);
        scales.append(sr_);
        if (s.has_zp) zeros.append(zr);
    }
    return nb::make_tuple(tscale, elems, scales, s.has_zp ? nb::object(zeros) : nb::none(), (int)flags);
}

// ---------------------------------------------------------------- dequantize

// Exact values of the rows (row order), or None if any code is inf/NaN
// (the reference raises from FP.exact / Pow2Format.value).
static nb::object py_values(nb::handle elems, nb::handle scales, nb::handle zeros, nb::handle tscale,
                            nb::handle spec_t) {
    Spec s = spec_from(spec_t);
    Q s_t = qint(1);
    if (!tscale.is_none()) {
        PyFP* t = (PyFP*)tscale.ptr();
        if (!is_finite(t->c, *t->f)) return nb::none();
        s_t = q_from_dy(decode<BigInt>(t->c, *t->f));
    }
    nb::list out;
    size_t r = 0;
    for (nb::handle row : elems) {
        nb::handle srow = scales[r];
        size_t c = 0;
        Q scale;
        int64_t cur_block = -1;
        Q z;
        for (nb::handle e : row) {
            int64_t b = (int64_t)c / s.block;
            if (b != cur_block) {
                cur_block = b;
                nb::handle so = srow[(size_t)b];
                if (so.is_none()) scale = qint(1);
                else if (s.scale_kind == 2) {
                    nb::object code = uint_val(so.ptr());
                    int64_t k = PyLong_AsLongLong(code.ptr());
                    if (k > s.p2_max) return nb::none();
                    scale = qpow2(k - s.p2_bias);
                } else {
                    PyFP* x = (PyFP*)so.ptr();
                    if (!is_finite(x->c, *x->f)) return nb::none();
                    scale = q_from_dy(decode<BigInt>(x->c, *x->f));
                }
                if (!zeros.is_none()) {
                    nb::object zv = uint_val(zeros[r][(size_t)b].ptr());
                    z = Q{big_from_long(zv.ptr()), 1};
                }
            }
            Q q;
            if (s.elem_fp) {
                PyFP* x = (PyFP*)e.ptr();
                if (!is_finite(x->c, *x->f)) return nb::none();
                q = q_from_dy(decode<BigInt>(x->c, *x->f));
            } else {
                nb::object ev = uint_val(e.ptr());
                q = qmul(qsub(Q{big_from_long(ev.ptr()), 1}, z), qpow2(-s.elem_int.frac));
            }
            out.append(q_to_fraction(qmul(qmul(s_t, scale), q)));
            ++c;
        }
        ++r;
    }
    return out;
}

// _permute(flat, shape, perm): C-order data of `shape` transposed by `perm`.
static nb::object py_permute(nb::handle flat_seq, nb::handle shape_t, nb::handle perm_t) {
    nb::object flat = steal_checked(PySequence_Fast(flat_seq.ptr(), "expected a list"));
    PyObject** items = PySequence_Fast_ITEMS(flat.ptr());
    std::vector<int64_t> shape, perm;
    for (nb::handle h : shape_t) shape.push_back(nb::cast<int64_t>(h));
    for (nb::handle h : perm_t) perm.push_back(nb::cast<int64_t>(h));
    size_t nd = shape.size();
    std::vector<int64_t> st(nd);
    int64_t acc = 1;
    for (int64_t i = (int64_t)nd - 1; i >= 0; --i) { st[(size_t)i] = acc; acc *= shape[(size_t)i]; }
    std::vector<int64_t> nshape(nd), nst(nd);
    for (size_t i = 0; i < nd; ++i) { nshape[i] = shape[(size_t)perm[i]]; nst[i] = st[(size_t)perm[i]]; }
    Py_ssize_t total = PySequence_Fast_GET_SIZE(flat.ptr());
    nb::object out = steal_checked(PyList_New(total));
    std::vector<int64_t> idx(nd, 0);
    int64_t off = 0;
    for (Py_ssize_t i = 0; i < total; ++i) {
        PyList_SET_ITEM(out.ptr(), i, Py_NewRef(items[off]));
        for (int64_t k = (int64_t)nd - 1; k >= 0; --k) {
            off += nst[(size_t)k];
            if (++idx[(size_t)k] < nshape[(size_t)k]) break;
            off -= nst[(size_t)k] * nshape[(size_t)k];
            idx[(size_t)k] = 0;
        }
    }
    return out;
}

// blockscale._shape + _flatten in one pass; None if a BlockTensor is nested.
static bool shape_walk(PyObject* v, PyObject* bt, std::vector<int64_t>& shape, PyObject* flat) {
    if (PyObject_TypeCheck(v, (PyTypeObject*)bt)) return false;
    shape.clear();
    if (!PyList_Check(v) && !PyTuple_Check(v)) {
        if (PyList_Append(flat, v) < 0) raise_current();
        return true;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(v);
    if (n == 0) raise(PyExc_ValueError, "empty tensor");
    PyObject** items = PySequence_Fast_ITEMS(v);
    std::vector<int64_t> first, cur;
    bool ragged = false, first_leaf = false;
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* x = items[i];
        if (!PyList_Check(x) && !PyTuple_Check(x)) {   // a leaf: its shape is ()
            if (PyObject_TypeCheck(x, (PyTypeObject*)bt)) return false;
            if (PyList_Append(flat, x) < 0) raise_current();
            if (i == 0) first_leaf = true;
            else if (!first_leaf) ragged = true;
            continue;
        }
        if (!shape_walk(x, bt, i ? cur : first, flat)) return false;
        if (i && (first_leaf || cur != first)) ragged = true;
    }
    if (ragged) raise(PyExc_ValueError, "ragged nested lists: all rows must have the same shape");
    shape.push_back(n);
    shape.insert(shape.end(), first.begin(), first.end());
    return true;
}

static nb::object py_shape_flatten(nb::handle values, nb::handle bt) {
    std::vector<int64_t> shape;
    nb::object flat = steal_checked(PyList_New(0));
    if (!shape_walk(values.ptr(), bt.ptr(), shape, flat.ptr())) return nb::none();
    nb::object t = steal_checked(PyTuple_New((Py_ssize_t)shape.size()));
    for (size_t i = 0; i < shape.size(); ++i) PyTuple_SET_ITEM(t.ptr(), (Py_ssize_t)i, PyLong_FromLongLong(shape[i]));
    return nb::make_tuple(t, flat);
}

// Every element an FP, an int, a finite float or a Fraction?
static bool py_plain_numbers(nb::handle seq) {
    nb::object f = steal_checked(PySequence_Fast(seq.ptr(), "expected a sequence"));
    Py_ssize_t n = PySequence_Fast_GET_SIZE(f.ptr());
    PyObject** items = PySequence_Fast_ITEMS(f.ptr());
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* x = items[i];
        if (is_fp(x) || PyLong_Check(x) || is_fraction(x)) continue;
        if (PyFloat_Check(x) && double_finite(PyFloat_AS_DOUBLE(x))) continue;
        return false;
    }
    return true;
}

void register_block(nb::module_& m) {
    m.def("block_quantize", &py_quantize);
    m.def("block_values", &py_values, nb::arg(), nb::arg(), nb::arg().none(), nb::arg().none(), nb::arg());
    m.def("permute", &py_permute);
    m.def("shape_flatten", &py_shape_flatten);
    m.def("plain_numbers", &py_plain_numbers);
}

}  // namespace vf
