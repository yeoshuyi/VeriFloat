// FP: bit-accurate floating point with IEEE status flags. A port of
// verifloat_py/fp.py (FP class) to C++, exposed as a CPython heap type.
#include "fp.hpp"
#include "args.hpp"
#include "fast.hpp"
#include "ops.hpp"
#include "uint.hpp"

#include <nanobind/stl/string.h>

#include <cmath>
#include <cstring>
#include <optional>
#include <unordered_map>

namespace vf {

PyState S;

// ---------------------------------------------------------------- objects

PyFP* fp_new(PyObject* fmt, const Code& c, uint8_t flags, Unr&& u) {
    PyTypeObject* t = S.FP;
    PyFP* o = (PyFP*)PyObject_Malloc(t->tp_basicsize);
    if (!o) { PyErr_NoMemory(); raise_current(); }
    PyObject_Init((PyObject*)o, t);
    Py_INCREF(fmt);
    o->fmt = fmt;
    o->f = &fmt_of(fmt);
    new (&o->c) Code(c);
    o->flags = flags;
    new (&o->u) Unr(std::move(u));
    return o;
}

PyFP* fp_new(PyObject* fmt, const Code& c, uint8_t flags) { return fp_new(fmt, c, flags, Unr()); }

static bool should_warn(Event ev, const Fmt& f) {
    // Unsigned formats warn on every event; signed formats follow standard
    // FPU behaviour silently, except for the opt-in exponent wrap.
    return ev != EV_NONE && (!f.is_signed || ev == EV_WRAPPED_UP || ev == EV_WRAPPED_DOWN);
}

// An exact value whose exponent is so large that writing it out as a
// Fraction would take megabytes or more (formats with very wide exponent
// fields, scaleb by a huge n). A warning does not build such a value.
static bool unr_too_large(const Unr& u) {
    constexpr int64_t kLimit = int64_t(1) << 24;
    auto big = [](int64_t e) { return e > kLimit || e < -kLimit; };
    if (u.kind == Unr::DYB) return big(u.big->exp);
    if (u.kind < Unr::DY || u.kind > Unr::DIV) return false;
    const int n = u.kind == Unr::DY ? 1 : u.kind == Unr::FMA ? 3 : 2;
    for (int i = 0; i < n; ++i)
        if (big(u.wide ? u.wide->t[i].exp : u.t[i].exp)) return true;
    return false;
}

nb::object fp_finish(PyObject* fmt, const RoundOut& r, Unr&& u) {
    PyFP* o = fp_new(fmt, r.c, r.flags, std::move(u));
    nb::object res = nb::steal((PyObject*)o);
    if (should_warn(r.ev, *o->f)) {
        nb::object un = unr_too_large(o->u) ? nb::borrow(Py_Ellipsis) : unrounded_of(o);
        steal_checked(PyObject_CallFunctionObjArgs(S.report, S.event_names[r.ev], un.ptr(), fmt,
                                                   res.ptr(), nullptr));
    }
    return res;
}

static nb::object fp_plain(PyObject* fmt, const Code& c) {   // FP._make
    return nb::steal((PyObject*)fp_new(fmt, c, 0));
}
nb::object fp_plain_obj(PyObject* fmt, const Code& c) { return fp_plain(fmt, c); }

// ---------------------------------------------------------------- unrounded

template <class U> static nb::object fraction_of(const Dy<U>& d) { return fraction_dy<U>(d.neg, d.sig, d.exp); }

static nb::object combine(uint8_t kind, const nb::object* f) {
    switch (kind) {
        case Unr::ADD: return steal_checked(PyNumber_Add(f[0].ptr(), f[1].ptr()));
        case Unr::MUL: return steal_checked(PyNumber_Multiply(f[0].ptr(), f[1].ptr()));
        case Unr::FMA: {
            nb::object p = steal_checked(PyNumber_Multiply(f[0].ptr(), f[1].ptr()));
            return steal_checked(PyNumber_Add(p.ptr(), f[2].ptr()));
        }
        case Unr::DIV: return steal_checked(PyNumber_TrueDivide(f[0].ptr(), f[1].ptr()));
    }
    return f[0];
}

nb::object fraction_of_term(const Term& t) { return fraction_dy<u128>(t.neg, t.sig, t.exp); }

nb::object unrounded_of(PyFP* x) {
    Unr& u = x->u;
    switch (u.kind) {
        case Unr::NONE: return nb::none();
        case Unr::POSINF: return nb::borrow(S.inf);
        case Unr::NEGINF: return nb::borrow(S.ninf);
        case Unr::OBJ:
            if (!is_fraction(u.obj.ptr()))
                u.obj = steal_checked(call_one(S.Fraction, u.obj.ptr()));
            return u.obj;
        case Unr::DYB:
            u.obj = fraction_of(*u.big);
            u.kind = Unr::OBJ;
            u.big = Rc<Dy<BigInt>>();
            return u.obj;
        default: break;
    }
    nb::object f[3];
    int n = u.kind == Unr::DY ? 1 : u.kind == Unr::FMA ? 3 : 2;
    for (int i = 0; i < n; ++i) f[i] = u.wide ? fraction_of(u.wide->t[i]) : fraction_of_term(u.t[i]);
    u.obj = u.kind == Unr::DY ? f[0] : combine(u.kind, f);
    u.kind = Unr::OBJ;
    u.wide = Rc<WideTerms>();
    return u.obj;
}

// ---------------------------------------------------------------- numbers

// A Python number as a sticky dyadic with at least `keep` significant bits.
struct Num : NumD {
    nb::object unr;      // what Fraction(value) is built from
};

static PyObject* str_numerator() {
    static PyObject* s = PyUnicode_InternFromString("_numerator");
    return s;
}
static PyObject* str_denominator() {
    static PyObject* s = PyUnicode_InternFromString("_denominator");
    return s;
}

static void num_from_long(PyObject* v, Num& n) {
    int overflow = 0;
    long long x = PyLong_AsLongLongAndOverflow(v, &overflow);
    if (x == -1 && PyErr_Occurred()) raise_current();
    if (!overflow) {
        n.big = false;
        n.s.neg = x < 0;
        n.s.sig = x < 0 ? (u128)(-(i128)x) : (u128)x;
        n.s.exp = 0;
        return;
    }
    BigInt b = big_from_long(v);
    n.big = true;
    n.b.neg = b < 0;
    n.b.sig = abs(b);
    n.b.exp = 0;
}

static void num_from_fraction(PyObject* frac, int64_t keep, Num& n) {
    nb::object num = steal_checked(PyObject_GetAttr(frac, str_numerator()));
    nb::object den = steal_checked(PyObject_GetAttr(frac, str_denominator()));
    int overflow = 0;
    long long d = PyLong_AsLongLongAndOverflow(den.ptr(), &overflow);
    if (d == -1 && PyErr_Occurred()) raise_current();
    if (!overflow && d == 1) { num_from_long(num.ptr(), n); return; }
    if (!overflow && (d & (d - 1)) == 0) {
        num_from_long(num.ptr(), n);
        int64_t sh = bitlen((uint64_t)d) - 1;
        if (n.big) n.b.exp -= sh; else n.s.exp -= sh;
        return;
    }
    BigInt bd = big_from_long(den.ptr());
    if (bd > 0 && (bd & (bd - 1)) == 0) {
        num_from_long(num.ptr(), n);
        int64_t sh = bitlen(bd) - 1;
        if (n.big) n.b.exp -= sh; else n.s.exp -= sh;
        return;
    }
    num_from_ratio(big_from_long(num.ptr()), bd, keep, n);
}

// Fraction(v) for a finite number v, as a sticky dyadic.
static Num parse_number(PyObject* v, int64_t keep) {
    Num n;
    if (PyLong_Check(v)) {
        num_from_long(v, n);
        n.unr = nb::borrow(v);
    } else if (PyFloat_Check(v)) {
        num_from_double(PyFloat_AS_DOUBLE(v), n);
        n.unr = nb::borrow(v);
    } else {
        nb::object frac = is_fraction(v) ? nb::borrow(v) : steal_checked(call_one(S.Fraction, v));
        num_from_fraction(frac.ptr(), keep, n);
        n.unr = frac;
    }
    return n;
}

RoundOut round_number(PyObject* value, const Fmt& f, bool zero_sign, SRArg& sr, nb::object* unr_obj) {
    Num n = parse_number(value, f.prec + 2);
    if (unr_obj) *unr_obj = n.unr;
    return round_num(n, f, zero_sign, sr);
}

// ---------------------------------------------------------------- decode

nb::object fp_exact(const Code& c, const Fmt& f) {
    if (!f.wide) return fraction_of(decode<u128>(c, f));
    return fraction_of(decode<BigInt>(c, f));
}

static constexpr uint64_t kHashP = (uint64_t(1) << 61) - 1;
static uint64_t mod_p(u128 x) {
    while (x >> 61) x = (x & kHashP) + (x >> 61);
    return x == kHashP ? 0 : (uint64_t)x;
}
static uint64_t mod_p(const BigInt& x) { return static_cast<uint64_t>(x % kHashP); }

// hash(Fraction(value)), matching Python's numeric hash.
template <class U> static Py_hash_t hash_dy(const Dy<U>& d) {
    if (d.sig == 0) return 0;
    uint64_t m = mod_p(d.sig);
    int64_t e = d.exp % 61;
    if (e < 0) e += 61;
    uint64_t h = e ? (((m << e) & kHashP) | (m >> (61 - e))) : m;
    int64_t r = d.neg ? -(int64_t)h : (int64_t)h;
    return r == -1 ? -2 : (Py_hash_t)r;
}

// ---------------------------------------------------------------- formats

static std::unordered_map<uint64_t, PyObject*> g_common;

// FP._common_of(fa, fb) (cached; the search runs in Python once per pair).
static PyObject* common_of(PyObject* fa, PyObject* fb) {
    int64_t ia = fmt_of(fa).id, ib = fmt_of(fb).id;
    if (ia == ib) return fa;
    uint64_t key = ((uint64_t)ia << 32) | (uint64_t)(uint32_t)ib;
    auto it = g_common.find(key);
    if (it != g_common.end()) return it->second;
    PyObject* r = check(PyObject_CallFunctionObjArgs(S.common_of, fa, fb, nullptr));
    g_common.emplace(key, r);
    return r;
}

static void warn_mismatch(PyObject* fa, PyObject* fb, PyObject* fmt) {
    steal_checked(PyObject_CallFunctionObjArgs(S.warn_mismatch, fa, fb, fmt, nullptr));
}

static inline bool is_scalar(PyObject* o) {
    return (PyLong_Check(o) && !PyBool_Check(o)) || PyFloat_Check(o) || is_fraction(o);
}

// FP._cast_scalar: a Python number cast into self's format (a signed copy
// for negative literals into unsigned formats). Warns if the cast is lossy.
static Opnd cast_scalar(PyFP* self, PyObject* other, nb::object& hold) {
    const Fmt& F = *self->f;
    Opnd b;
    RoundOut r;
    PyObject* target = self->fmt;
    if (PyFloat_Check(other) && !double_finite(PyFloat_AS_DOUBLE(other))) {
        double v = PyFloat_AS_DOUBLE(other);
        if (!F.is_signed && v < 0) {
            hold = nb::handle(self->fmt).attr("replace")(nb::arg("signed") = true);
            target = hold.ptr();
        }
        Unr ignore;
        r = special_in(v, fmt_of(target), ignore);
    } else {
        bool zero_sign = PyFloat_Check(other) && std::signbit(PyFloat_AS_DOUBLE(other));
        Num n = parse_number(other, F.prec + 2);
        // A negative literal keeps its sign even against an unsigned
        // format, so that e.g. unsigned - 1 is not clamped to unsigned - 0.
        if (!F.is_signed && n.neg() && !n.zero()) {
            hold = nb::handle(self->fmt).attr("replace")(nb::arg("signed") = true);
            target = hold.ptr();
        }
        if (n.zero() && !fmt_of(target).has_zero) {
            hold = nb::handle(target).attr("replace")(nb::arg("has_zero") = true);
            target = hold.ptr();
        }
        SRArg sr;
        r = round_num(n, fmt_of(target), zero_sign, sr);
    }
    b.c = r.c;
    b.f = &fmt_of(target);
    b.fmt_obj = target;
    if (r.flags) {
        nb::object bf = steal_checked(PyFloat_FromDouble(fp_to_double(b.c, *b.f)));
        steal_checked(PyObject_CallFunctionObjArgs(S.cast_warn, other, self->fmt, bf.ptr(),
                                                   S.flags[r.flags & 31], nullptr));
    }
    return b;
}

struct Coerced {
    Opnd a, b;
    PyObject* fmt = nullptr;
    std::optional<Fmt> pa, pb;   // widened operands' formats
    nb::object hold;
};

// FP._coerce: (self, other, fmt) with both exact in fmt; false if `other`
// is not an FP or a Python number.
static bool coerce(PyFP* self, PyObject* other, Coerced& out) {
    if (is_fp(other)) {
        PyFP* o = (PyFP*)other;
        if (self->f->id == o->f->id) {
            out.a = opnd_of(self);
            out.b = opnd_of(o);
            out.fmt = self->fmt;
            return true;
        }
        PyObject* fmt = common_of(self->fmt, o->fmt);
        warn_mismatch(self->fmt, o->fmt, fmt);
        out.fmt = fmt;
        out.a = widen(opnd_of(self), fmt_of(fmt), out.pa);
        out.b = widen(opnd_of(o), fmt_of(fmt), out.pb);
        return true;
    }
    if (is_scalar(other)) {
        out.a = opnd_of(self);
        out.b = cast_scalar(self, other, out.hold);
        out.fmt = self->fmt;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------- operations

// a op b rounded into fmt (operands exact in fmt), finished as an FP.
nb::object arith(char op, const Opnd& a, const Opnd& b, PyObject* fmt) {
    return fp_finish(fmt, op_arith(op, a, b, fmt_of(fmt)));
}

// ---------------------------------------------------------------- scalar fast paths
// The common case of each operation on the kernels of fast.hpp: operands of
// one format (a signed format with a zero, not stochastic), all finite, and
// a normal result in range. Each returns nullptr when it declines; the
// general code below then computes the same operation.

static inline Term term_of_fv(const FV<uint64_t>& v) { return {(u128)v.sig, v.exp, v.neg}; }

static inline bool fast_operands(const PyFP* a, const PyFP* b) {
    const Fmt& F = *a->f;
    if (!F.fast || F.id != b->f->id || !g_fast) return false;
    return !(F.has_nan() && (a->c.field == F.top || b->c.field == F.top));
}

static PyObject* fast_binop(char op, PyFP* a, PyFP* b) {
    if (!fast_operands(a, b)) return nullptr;
    const Fmt& F = *a->f;
    const FV<uint64_t> x = fv_of<uint64_t>(a->c, F);
    FV<uint64_t> y = fv_of<uint64_t>(b->c, F);
    FCode c;
    unsigned fl = 0;
    if (!fast_op_code(op, x, y, F, c, fl)) return nullptr;
    Unr u;
    u.kind = op == '*' ? Unr::MUL : op == '/' ? Unr::DIV : Unr::ADD;
    if (op == '-') y.neg = !y.neg;
    u.t[0] = term_of_fv(x);
    u.t[1] = term_of_fv(y);
    return (PyObject*)fp_new(a->fmt, c.code(), (uint8_t)fl, std::move(u));
}

static PyObject* fast_fma(PyFP* a, PyFP* b, PyFP* c) {
    if (!fast_operands(a, b) || !fast_operands(a, c)) return nullptr;
    const Fmt& F = *a->f;
    const FV<uint64_t> x = fv_of<uint64_t>(a->c, F), y = fv_of<uint64_t>(b->c, F), z = fv_of<uint64_t>(c->c, F);
    const FV<u128> p{(u128)x.sig * y.sig, x.exp + y.exp, x.neg != y.neg};
    FV<u128> s;
    bool sticky;
    unsigned fl = 0;
    FCode code;
    fast_add(p, FV<u128>{(u128)z.sig, z.exp, z.neg}, F.rounding, s, sticky);
    if (!fast_round_code(s, sticky, F, code, fl)) return nullptr;
    Unr u;
    u.kind = Unr::FMA;
    u.t[0] = term_of_fv(x);
    u.t[1] = term_of_fv(y);
    u.t[2] = term_of_fv(z);
    return (PyObject*)fp_new(a->fmt, code.code(), (uint8_t)fl, std::move(u));
}

// The value of a finite Python float.
static inline FV<uint64_t> fv_of_double(double d) {
    uint64_t bits;
    std::memcpy(&bits, &d, 8);
    const uint64_t field = (bits >> 52) & 0x7ff, mant = bits & ((uint64_t(1) << 52) - 1);
    FV<uint64_t> x;
    x.neg = bits >> 63;
    x.sig = field ? mant | (uint64_t(1) << 52) : mant;
    x.exp = (int64_t)(field ? field : 1) - 1075;
    return x;
}

// fmt(float): a finite Python float into a fast format.
static PyObject* fast_from_float(PyObject* value, PyObject* fmt) {
    const Fmt& F = fmt_of(fmt);
    if (!F.fast || !g_fast) return nullptr;
    const double d = PyFloat_AS_DOUBLE(value);
    if (!double_finite(d)) return nullptr;
    FCode c;
    unsigned fl = 0;
    if (!fast_round_code(fv_of_double(d), false, F, c, fl)) return nullptr;
    Unr u;
    u.kind = Unr::OBJ;
    u.obj = nb::borrow(value);
    return (PyObject*)fp_new(fmt, c.code(), (uint8_t)fl, std::move(u));
}

// x.convert(fmt): a finite value of a narrow format into a fast format.
static PyObject* fast_convert(PyFP* x, PyObject* fmt) {
    const Fmt &G = *x->f, &F = fmt_of(fmt);
    if (!F.fast || !g_fast || G.M > 61 || G.wide || !is_finite(x->c, G)) return nullptr;
    const FV<uint64_t> v = fv_of<uint64_t>(x->c, G);
    FCode c;
    unsigned fl = 0;
    if (!fast_round_code(v, false, F, c, fl)) return nullptr;
    Unr u;
    u.kind = Unr::DY;
    u.t[0] = term_of_fv(v);
    return (PyObject*)fp_new(fmt, c.code(), (uint8_t)fl, std::move(u));
}

static PyObject* binop(PyObject* x, PyObject* y, char op) {
    VF_TRY
    if (is_fp(x) && is_fp(y)) {
        if (PyObject* r = fast_binop(op, (PyFP*)x, (PyFP*)y)) return r;
    }
    bool reverse = !is_fp(x);
    PyFP* self = (PyFP*)(reverse ? y : x);
    PyObject* other = reverse ? x : y;
    Coerced c;
    if (!coerce(self, other, c)) Py_RETURN_NOTIMPLEMENTED;
    const Opnd& A = reverse ? c.b : c.a;
    const Opnd& B = reverse ? c.a : c.b;
    return arith(op, A, B, c.fmt).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* fp_add(PyObject* a, PyObject* b) { return binop(a, b, '+'); }
static PyObject* fp_sub(PyObject* a, PyObject* b) { return binop(a, b, '-'); }
static PyObject* fp_mul(PyObject* a, PyObject* b) { return binop(a, b, '*'); }
static PyObject* fp_div(PyObject* a, PyObject* b) { return binop(a, b, '/'); }

// Common format of three operands (FP._coerce3).
struct Coerced3 {
    Opnd a, b, c;
    PyObject* fmt = nullptr;
    std::optional<Fmt> pa, pb, pc;
    nb::object hb, hc, hold_ab;
    Opnd rb, rc;
};

static nb::object fp_fma_impl(PyFP* self, PyObject* bo, PyObject* co) {
    if (is_fp(bo) && is_fp(co)) {
        if (PyObject* r = fast_fma(self, (PyFP*)bo, (PyFP*)co)) return nb::steal(r);
    }
    Coerced3 k;
    Opnd sb, sc;
    if (is_fp(bo)) sb = opnd_of((PyFP*)bo);
    else sb = cast_scalar(self, bo, k.hb);
    if (is_fp(co)) sc = opnd_of((PyFP*)co);
    else sc = cast_scalar(self, co, k.hc);
    PyObject *fb = (PyObject*)sb.fmt_obj, *fc = (PyObject*)sc.fmt_obj;
    PyObject* f_ab = common_of(self->fmt, fb);
    if (self->f->id != sb.f->id) warn_mismatch(self->fmt, fb, f_ab);
    PyObject* fmt = common_of(f_ab, fc);
    if (fmt_of(f_ab).id != sc.f->id) warn_mismatch(f_ab, fc, fmt);
    const Fmt& F = fmt_of(fmt);
    Opnd a = widen(opnd_of(self), F, k.pa), b = widen(sb, F, k.pb), c = widen(sc, F, k.pc);
    return fp_finish(fmt, op_fma(a, b, c, F));
}

static nb::object fp_sqrt_impl(PyFP* self) { return fp_finish(self->fmt, op_sqrt(opnd_of(self))); }

static std::string short_type_name(PyObject* o) {
    nb::object n = steal_checked(PyObject_GetAttrString((PyObject*)Py_TYPE(o), "__name__"));
    return py_str(n.ptr());
}

// IEEE 754-2019 9.6: minimum/maximum and minimumNumber/maximumNumber
static nb::object minmax(PyFP* self, PyObject* other, bool pick_max, bool number) {
    Coerced c;
    if (!coerce(self, other, c))
        raise(PyExc_TypeError, "cannot compare FP with " + short_type_name(other));
    return fp_finish(c.fmt, op_minmax(c.a, c.b, fmt_of(c.fmt), pick_max, number));
}

// Raw encoding of an operand in its own format.
static nb::object raw_of(const Code& c, const Fmt& f) {
    if (f.size <= 64) {   // then M < 64 and, if signed, E + M < 64
        uint64_t r = (c.field << f.M) | c.mant;
        if (c.sign) r |= uint64_t(1) << (f.E + f.M);
        return steal_checked(PyLong_FromUnsignedLongLong(r));
    }
    BigInt r = BigInt(c.sign ? 1 : 0) << (unsigned)(f.E + f.M);
    r |= BigInt(c.field) << (unsigned)f.M;
    r |= mant_big(c, f);
    return long_from_big(r);
}

// FP._from_raw_fmt(raw, fmt)
static Code code_from_raw(PyObject* raw, const Fmt& f) {
    int overflow = 0;
    long long x = PyLong_AsLongLongAndOverflow(raw, &overflow);
    if (x == -1 && PyErr_Occurred()) raise_current();
    Code c;
    if (!overflow && f.size < 63) {
        c.sign = f.is_signed && ((x >> (f.E + f.M)) & 1);
        c.field = (uint64_t)(x >> f.M) & f.top;
        c.mant = (uint64_t)x & f.mask();
        return c;
    }
    // Arbitrary ints: Python floor shifts on two's complement
    BigInt b = big_from_long(raw);
    BigInt mod = BigInt(1) << (unsigned)(f.size + 1);
    b = ((b % mod) + mod) % mod;   // the low size+1 bits, two's complement
    c.sign = f.is_signed && bit(b, f.E + f.M);
    c.field = static_cast<uint64_t>((b >> (unsigned)f.M) & BigInt(f.top));
    set_mant_big(c, f, b);
    return c;
}

static PyObject* bitop(PyObject* x, PyObject* y, char op) {
    VF_TRY
    bool reverse = !is_fp(x);
    PyFP* self = (PyFP*)(reverse ? y : x);
    PyObject* other = reverse ? x : y;
    Coerced c;
    if (!coerce(self, other, c)) Py_RETURN_NOTIMPLEMENTED;
    nb::object ra = raw_of(c.a.c, *c.a.f), rb = raw_of(c.b.c, *c.b.f);
    if (reverse) std::swap(ra, rb);
    PyObject* r = op == '&' ? PyNumber_And(ra.ptr(), rb.ptr())
                : op == '|' ? PyNumber_Or(ra.ptr(), rb.ptr())
                            : PyNumber_Xor(ra.ptr(), rb.ptr());
    nb::object raw = steal_checked(r);
    return fp_plain(c.fmt, code_from_raw(raw.ptr(), fmt_of(c.fmt))).release().ptr();
    VF_CATCH(nullptr)
}
static PyObject* fp_and(PyObject* a, PyObject* b) { return bitop(a, b, '&'); }
static PyObject* fp_or(PyObject* a, PyObject* b) { return bitop(a, b, '|'); }
static PyObject* fp_xor(PyObject* a, PyObject* b) { return bitop(a, b, '^'); }

static PyObject* fp_invert(PyObject* s) {
    VF_TRY
    PyFP* self = (PyFP*)s;
    const Fmt& f = *self->f;
    nb::object raw = raw_of(self->c, f);
    nb::object m = steal_checked(PyNumber_Lshift(nb::int_(1).ptr(), nb::int_(f.size).ptr()));
    m = steal_checked(PyNumber_Subtract(m.ptr(), nb::int_(1).ptr()));
    nb::object inv = steal_checked(PyNumber_Invert(raw.ptr()));
    inv = steal_checked(PyNumber_And(inv.ptr(), m.ptr()));
    return fp_plain(self->fmt, code_from_raw(inv.ptr(), f)).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* fp_neg(PyObject* s) {
    VF_TRY
    PyFP* self = (PyFP*)s;
    return fp_finish(self->fmt, op_neg(opnd_of(self))).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* fp_pos(PyObject* s) {
    VF_TRY
    PyFP* self = (PyFP*)s;
    return fp_plain(self->fmt, self->c).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* fp_abs(PyObject* s) {
    VF_TRY
    PyFP* self = (PyFP*)s;
    Code c = self->c;
    c.sign = false;
    return fp_plain(self->fmt, c).release().ptr();
    VF_CATCH(nullptr)
}

static int fp_bool(PyObject* s) {
    PyFP* self = (PyFP*)s;
    return !is_zero(self->c, *self->f);
}

static PyObject* fp_float(PyObject* s) {
    PyFP* self = (PyFP*)s;
    VF_TRY
    return PyFloat_FromDouble(fp_to_double(self->c, *self->f));
    VF_CATCH(nullptr)
}

// int(exact): truncation toward zero.
static PyObject* fp_int(PyObject* s) {
    VF_TRY
    PyFP* self = (PyFP*)s;
    const Fmt& f = *self->f;
    if (is_nan(self->c, f)) raise(PyExc_ValueError, "cannot convert NaN to integer");
    if (is_inf(self->c, f)) raise(PyExc_OverflowError, "cannot convert infinity to integer");
    Dy<BigInt> d = decode<BigInt>(self->c, f);
    BigInt v = d.exp >= 0 ? BigInt(d.sig << (unsigned)d.exp) : BigInt(d.sig >> (unsigned)-d.exp);
    if (d.neg) v = -v;
    return long_from_big(v).release().ptr();
    VF_CATCH(nullptr)
}

static Py_hash_t fp_hash(PyObject* s) {
    PyFP* self = (PyFP*)s;
    const Fmt& f = *self->f;
    if (is_nan(self->c, f)) return PyBaseObject_Type.tp_hash(s);
    if (is_inf(self->c, f)) return self->c.sign ? -314159 : 314159;
    if (!f.wide) return hash_dy(decode<u128>(self->c, f));
    return hash_dy(decode<BigInt>(self->c, f));
}

// Comparison operators (IEEE: NaN is unordered, so only != is true)
static PyObject* fp_richcompare(PyObject* x, PyObject* y, int op) {
    VF_TRY
    // CPython passes the FP first, swapping `op` for reflected comparisons.
    PyFP* self = (PyFP*)x;
    PyObject* other = y;
    Coerced c;
    if (!coerce(self, other, c)) Py_RETURN_NOTIMPLEMENTED;
    if (is_nan(c.a.c, *c.a.f) || is_nan(c.b.c, *c.b.f)) return PyBool_FromLong(op == Py_NE);
    int k = cmp_keys(c.a, c.b);
    bool r;
    switch (op) {
        case Py_LT: r = k < 0; break;
        case Py_LE: r = k <= 0; break;
        case Py_EQ: r = k == 0; break;
        case Py_NE: r = k != 0; break;
        case Py_GT: r = k > 0; break;
        default: r = k >= 0;
    }
    return PyBool_FromLong(r);
    VF_CATCH(nullptr)
}

// ---------------------------------------------------------------- integer rounding

static uint8_t rounding_index(PyObject* r, const Fmt& f) {
    if (r == Py_None) return f.rounding;
    int t = PyObject_IsTrue(r);
    if (t < 0) raise_current();
    if (!t) return f.rounding;           // `rounding or fmt.rounding`
    for (int i = 0; i < 6; ++i)
        if (r == S.rounding_enum[i]) return (uint8_t)i;
    raise(PyExc_TypeError, "rounding must be a Rounding member");
}

static nb::object to_int_impl(PyFP* self, PyObject* bits, PyObject* signed_o, PyObject* rounding_o, PyObject* exact_o) {
    const Fmt& f = *self->f;
    uint8_t rounding = rounding_index(rounding_o, f);
    bool is_signed = PyObject_IsTrue(signed_o);
    bool want_exact = PyObject_IsTrue(exact_o);
    PyObject* cls = (PyObject*)(is_signed ? S.INT : S.UINT);
    if (PyLong_CheckExact(bits)) {
        int overflow = 0;
        long long n = PyLong_AsLongLongAndOverflow(bits, &overflow);
        if (!overflow && n >= 1 && n <= 65536) {
            SRArg sr;
            IntOut o = op_to_int(opnd_of(self), n, is_signed, rounding, want_exact, sr);
            nb::object v = long_from_big(o.v);
            nb::object r = steal_checked(PyObject_CallFunctionObjArgs(cls, v.ptr(), bits, nullptr));
            return nb::make_tuple(r, nb::handle(S.flags[o.flags]));
        }
    }
    // Any other `bits` (not a plain int, not positive, huge): the same steps
    // on Python ints, so that the errors are the reference's.
    nb::object one = nb::int_(1);
    nb::object lo, hi;
    if (is_signed) {
        nb::object b1 = steal_checked(PyNumber_Subtract(bits, one.ptr()));
        nb::object p = steal_checked(PyNumber_Lshift(one.ptr(), b1.ptr()));
        lo = steal_checked(PyNumber_Negative(p.ptr()));
        hi = steal_checked(PyNumber_Subtract(p.ptr(), one.ptr()));
    } else {
        lo = nb::int_(0);
        nb::object p = steal_checked(PyNumber_Lshift(one.ptr(), bits));
        hi = steal_checked(PyNumber_Subtract(p.ptr(), one.ptr()));
    }
    auto invalid = [&](int kind) {
        static const char table[4][3] = {{'M', 'M', 'm'}, {'M', 'M', 'm'}, {'x', 'x', 'x'}, {'0', 'M', 'm'}};
        char which = table[f.nan_mode][kind];
        nb::object v;
        if (which == 'M') v = hi;
        else if (which == 'm') v = lo;
        else if (which == '0') v = nb::int_(0);
        else if (is_signed) v = lo;
        else {
            nb::object p = steal_checked(PyNumber_Lshift(one.ptr(), bits));
            v = steal_checked(PyNumber_Subtract(p.ptr(), one.ptr()));
        }
        nb::object r = steal_checked(PyObject_CallFunctionObjArgs(cls, v.ptr(), bits, nullptr));
        return nb::make_tuple(r, nb::handle(S.flags[INVALID]));
    };
    if (is_nan(self->c, f)) return invalid(0);
    if (is_inf(self->c, f)) return invalid(self->c.sign ? 2 : 1);
    SRArg sr;
    if (rounding == SR) sr.resolve(f);
    auto [qb, inexact] = round_to_int(self->c, f, rounding, sr);
    nb::object q = long_from_big(qb);
    if (PyObject_RichCompareBool(q.ptr(), hi.ptr(), Py_GT)) return invalid(1);
    if (PyObject_RichCompareBool(q.ptr(), lo.ptr(), Py_LT)) return invalid(2);
    nb::object r = steal_checked(PyObject_CallFunctionObjArgs(cls, q.ptr(), bits, nullptr));
    return nb::make_tuple(r, nb::handle(S.flags[inexact && want_exact ? INEXACT : 0]));
}

static nb::object round_to_integral_impl(PyFP* self, PyObject* rounding_o, PyObject* exact_o) {
    const Fmt& f = *self->f;
    Opnd x = opnd_of(self);
    // NaN, infinity and zero come back before the arguments are looked at.
    const bool special = !is_finite(x.c, f) || is_zero(x.c, f);
    uint8_t rounding = special ? f.rounding : rounding_index(rounding_o, f);
    bool want_exact = !special && PyObject_IsTrue(exact_o);
    SRArg sr;
    return fp_finish(self->fmt, op_round_to_integral(x, rounding, want_exact, sr));
}

// ---------------------------------------------------------------- conversion

nb::object convert_to(PyFP* self, PyObject* fmt, PyObject* sr_rand) {
    if (PyObject* r = fast_convert(self, fmt)) return nb::steal(r);
    SRArg sr;
    sr.given = sr_rand;
    return fp_finish(fmt, op_convert(opnd_of(self), fmt_of(fmt), sr));
}

// FP.from_value(value, fmt, sr_rand) once the format is resolved.
nb::object fp_from_number(PyObject* value, PyObject* fmt, PyObject* sr_rand) {
    if (PyFloat_CheckExact(value)) {
        if (PyObject* r = fast_from_float(value, fmt)) return nb::steal(r);
    }
    if (is_fp(value)) return convert_to((PyFP*)value, fmt, sr_rand);
    const Fmt& F = fmt_of(fmt);
    nb::object v = nb::borrow(value);
    if (is_uint(value)) v = uint_val(value);
    else if (!PyLong_Check(value) && !PyFloat_Check(value) && !is_fraction(value)) {
        static PyObject* s_exact = PyUnicode_InternFromString("exact");
        if (PyObject_HasAttr(value, s_exact)) v = steal_checked(PyObject_GetAttr(value, s_exact));
    }
    if (PyFloat_Check(v.ptr()) && !double_finite(PyFloat_AS_DOUBLE(v.ptr()))) {
        Unr u;
        RoundOut r = special_in(PyFloat_AS_DOUBLE(v.ptr()), F, u);
        return fp_finish(fmt, r, std::move(u));
    }
    bool zero_sign = PyFloat_Check(v.ptr()) && std::signbit(PyFloat_AS_DOUBLE(v.ptr()));
    SRArg sr;
    sr.given = sr_rand;
    Num n = parse_number(v.ptr(), F.prec + 2);
    RoundOut r = round_num(n, F, zero_sign, sr);
    Unr u;
    u.kind = Unr::OBJ;
    u.obj = n.unr;
    return fp_finish(fmt, r, std::move(u));
}

// ---------------------------------------------------------------- Python methods

static PyObject* m_fma(PyObject* s, PyObject* const* args, Py_ssize_t nargs, PyObject* kw) {
    VF_TRY
    PyObject* a[2];
    parse_args("fma", args, nargs, kw, {"b", "c"}, a, 2);
    return fp_fma_impl((PyFP*)s, a[0], a[1]).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* m_sqrt(PyObject* s, PyObject*) {
    VF_TRY
    return fp_sqrt_impl((PyFP*)s).release().ptr();
    VF_CATCH(nullptr)
}

#define VF_MINMAX(name, pick_max, number)                                                        \
    static PyObject* m_##name(PyObject* s, PyObject* const* args, Py_ssize_t nargs, PyObject* kw) { \
        VF_TRY                                                                                   \
        PyObject* a[1];                                                                          \
        parse_args(#name, args, nargs, kw, {"other"}, a, 1);                                     \
        return minmax((PyFP*)s, a[0], pick_max, number).release().ptr();                         \
        VF_CATCH(nullptr)                                                                        \
    }
VF_MINMAX(minimum, false, false)
VF_MINMAX(maximum, true, false)
VF_MINMAX(minimum_number, false, true)
VF_MINMAX(maximum_number, true, true)

// remainder / fmod: operands coerced like any binary operation.
static nb::object rem_impl(PyFP* self, PyObject* other, bool truncate, const char* name) {
    Coerced c;
    if (!coerce(self, other, c))
        raise(PyExc_TypeError, std::string(name) + "() needs an FP or a number, not " + short_type_name(other));
    return fp_finish(c.fmt, op_rem(c.a, c.b, fmt_of(c.fmt), truncate));
}
#define VF_REM(name, truncate)                                                                   \
    static PyObject* m_##name(PyObject* s, PyObject* const* args, Py_ssize_t nargs, PyObject* kw) { \
        VF_TRY                                                                                   \
        PyObject* a[1];                                                                          \
        parse_args(#name, args, nargs, kw, {"other"}, a, 1);                                     \
        return rem_impl((PyFP*)s, a[0], truncate, #name).release().ptr();                        \
        VF_CATCH(nullptr)                                                                        \
    }
VF_REM(remainder, false)
VF_REM(fmod, true)

static PyObject* m_next_up(PyObject* s, PyObject*) {
    VF_TRY
    return fp_finish(((PyFP*)s)->fmt, op_next(opnd_of((PyFP*)s), true)).release().ptr();
    VF_CATCH(nullptr)
}
static PyObject* m_next_down(PyObject* s, PyObject*) {
    VF_TRY
    return fp_finish(((PyFP*)s)->fmt, op_next(opnd_of((PyFP*)s), false)).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* m_scaleb(PyObject* s, PyObject* const* args, Py_ssize_t nargs, PyObject* kw) {
    VF_TRY
    PyObject* a[1];
    parse_args("scaleb", args, nargs, kw, {"n"}, a, 1);
    if (!PyLong_Check(a[0]) || PyBool_Check(a[0])) raise(PyExc_TypeError, "scaleb() needs an int exponent");
    int overflow = 0;
    long long n = PyLong_AsLongLongAndOverflow(a[0], &overflow);
    if (n == -1 && PyErr_Occurred()) raise_current();
    if (overflow || n > (1LL << 62) || n < -(1LL << 62))
        raise(PyExc_OverflowError, "scaleb() exponent beyond 2**62");
    return fp_finish(((PyFP*)s)->fmt, op_scaleb(opnd_of((PyFP*)s), n)).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* m_logb(PyObject* s, PyObject*) {
    VF_TRY
    return fp_finish(((PyFP*)s)->fmt, op_logb(opnd_of((PyFP*)s))).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* m_fclass(PyObject* s, PyObject*) {
    PyFP* x = (PyFP*)s;
    return PyLong_FromUnsignedLong(op_fclass(x->c, *x->f));
}

// The sign a sign-injection takes from `other`: an FP of any format, or a
// Python number (a float's sign bit, so -0.0 counts as negative).
static bool sign_of(PyObject* other, const char* name) {
    if (is_fp(other)) return ((PyFP*)other)->c.sign;
    if (PyFloat_Check(other)) return std::signbit(PyFloat_AS_DOUBLE(other));
    if (PyLong_Check(other) || is_fraction(other)) {
        nb::object zero = nb::int_(0);
        int r = PyObject_RichCompareBool(other, zero.ptr(), Py_LT);
        if (r < 0) raise_current();
        return r;
    }
    raise(PyExc_TypeError, std::string(name) + "() needs an FP or a number, not " + short_type_name(other));
}
#define VF_SGNJ(name, mode)                                                                      \
    static PyObject* m_##name(PyObject* s, PyObject* const* args, Py_ssize_t nargs, PyObject* kw) { \
        VF_TRY                                                                                   \
        PyObject* a[1];                                                                          \
        parse_args(#name, args, nargs, kw, {"other"}, a, 1);                                     \
        PyFP* x = (PyFP*)s;                                                                      \
        return fp_plain(x->fmt, op_sgnj(opnd_of(x), sign_of(a[0], #name), mode)).release().ptr(); \
        VF_CATCH(nullptr)                                                                        \
    }
VF_SGNJ(copysign, 0)
VF_SGNJ(fsgnj, 0)
VF_SGNJ(fsgnjn, 1)
VF_SGNJ(fsgnjx, 2)

static PyObject* m_to_int(PyObject* s, PyObject* const* args, Py_ssize_t nargs, PyObject* kw) {
    VF_TRY
    nb::object d_bits = nb::int_(32);
    PyObject* a[4] = {d_bits.ptr(), Py_True, Py_None, Py_True};
    parse_args("to_int", args, nargs, kw, {"bits", "signed", "rounding", "exact"}, a, 0);
    return to_int_impl((PyFP*)s, a[0], a[1], a[2], a[3]).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* m_round_to_integral(PyObject* s, PyObject* const* args, Py_ssize_t nargs, PyObject* kw) {
    VF_TRY
    PyObject* a[2] = {Py_None, Py_False};
    parse_args("round_to_integral", args, nargs, kw, {"rounding", "exact"}, a, 0);
    return round_to_integral_impl((PyFP*)s, a[0], a[1]).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* rel_str(int k) {
    static PyObject* names[4] = {PyUnicode_InternFromString("lt"), PyUnicode_InternFromString("eq"),
                                 PyUnicode_InternFromString("gt"), PyUnicode_InternFromString("unordered")};
    return names[k];
}

// IEEE comparison: (relation index 0 lt, 1 eq, 2 gt, 3 unordered, flags)
static std::pair<int, uint8_t> compare_impl(PyFP* self, PyObject* other, bool signaling) {
    Coerced c;
    if (!coerce(self, other, c))
        raise(PyExc_TypeError, "cannot compare FP with " + short_type_name(other));
    return op_compare(c.a, c.b, signaling);
}

static PyObject* cmp_method(PyObject* s, PyObject* const* args, Py_ssize_t nargs, PyObject* kw, const char* name,
                            bool def_signaling, int mode) {
    VF_TRY
    PyObject* a[2] = {nullptr, def_signaling ? Py_True : Py_False};
    parse_args(name, args, nargs, kw, {"other", "signaling"}, a, 1);
    int t = PyObject_IsTrue(a[1]);
    if (t < 0) raise_current();
    auto [rel, flags] = compare_impl((PyFP*)s, a[0], t);
    PyObject* fl = S.flags[flags];
    if (mode == 0) return Py_BuildValue("(OO)", rel_str(rel), fl);
    bool r = mode == 1 ? rel == 1 : mode == 2 ? rel == 0 : (rel == 0 || rel == 1);
    return Py_BuildValue("(OO)", r ? Py_True : Py_False, fl);
    VF_CATCH(nullptr)
}
static PyObject* m_compare(PyObject* s, PyObject* const* a, Py_ssize_t n, PyObject* k) { return cmp_method(s, a, n, k, "compare", false, 0); }
static PyObject* m_eq(PyObject* s, PyObject* const* a, Py_ssize_t n, PyObject* k) { return cmp_method(s, a, n, k, "eq", false, 1); }
static PyObject* m_lt(PyObject* s, PyObject* const* a, Py_ssize_t n, PyObject* k) { return cmp_method(s, a, n, k, "lt", true, 2); }
static PyObject* m_le(PyObject* s, PyObject* const* a, Py_ssize_t n, PyObject* k) { return cmp_method(s, a, n, k, "le", true, 3); }

static std::string bin_str(const BigInt& v, int64_t bits) {
    std::string s((size_t)bits, '0');
    for (int64_t i = 0; i < bits; ++i)
        if (bit(v, i)) s[(size_t)(bits - 1 - i)] = '1';
    return s;
}
static std::string bin_str(uint64_t v, int64_t bits) {
    std::string s((size_t)bits, '0');
    for (int64_t i = 0; i < bits && i < 64; ++i)
        if ((v >> i) & 1) s[(size_t)(bits - 1 - i)] = '1';
    return s;
}

static PyObject* m_to_bin(PyObject* s, PyObject* const* args, Py_ssize_t nargs, PyObject* kw) {
    VF_TRY
    nb::object space = nb::str(" ");
    PyObject* a[1] = {space.ptr()};
    parse_args("to_bin", args, nargs, kw, {"sep"}, a, 0);
    PyFP* self = (PyFP*)s;
    const Fmt& f = *self->f;
    nb::list fields;
    if (f.is_signed) fields.append(nb::str(self->c.sign ? "1" : "0"));
    fields.append(nb::str(bin_str(self->c.field, f.E).c_str()));
    if (f.M) fields.append(nb::str((f.wide ? bin_str(mant_big(self->c, f), f.M) : bin_str(self->c.mant, f.M)).c_str()));
    return PyUnicode_Join(a[0], fields.ptr());
    VF_CATCH(nullptr)
}

static PyObject* m_to_hex(PyObject* s, PyObject*) {
    VF_TRY
    PyFP* self = (PyFP*)s;
    nb::object raw = raw_of(self->c, *self->f);
    std::string spec = "0" + std::to_string((self->f->size + 3) / 4) + "x";
    nb::object sp = nb::str(spec.c_str());
    return PyObject_Format(raw.ptr(), sp.ptr());
    VF_CATCH(nullptr)
}

static PyObject* m_convert_to(PyObject* s, PyObject* const* args, Py_ssize_t nargs, PyObject* kw) {
    VF_TRY
    PyObject* a[2] = {nullptr, Py_None};
    parse_args("_convert_to", args, nargs, kw, {"fmt", "sr_rand"}, a, 1);
    if (!is_format(a[0])) raise(PyExc_TypeError, "fmt must be an FPFormat");
    return convert_to((PyFP*)s, a[0], a[1]).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* g_convert_slow = nullptr;
static PyObject* g_from_value_slow = nullptr;

// Only `sr_rand=` among the keywords? Returns it (or None) in *sr.
static bool only_sr_kw(PyObject* const* args, Py_ssize_t nargs, PyObject* kw, PyObject** sr) {
    *sr = Py_None;
    if (!kw) return true;
    Py_ssize_t nk = PyTuple_GET_SIZE(kw);
    for (Py_ssize_t j = 0; j < nk; ++j) {
        if (PyUnicode_CompareWithASCIIString(PyTuple_GET_ITEM(kw, j), "sr_rand") != 0) return false;
        *sr = args[nargs + j];
    }
    return true;
}

// convert(fmt, *, sr_rand=None); other signatures go through FP.convert's resolver.
static PyObject* m_convert(PyObject* s, PyObject* const* args, Py_ssize_t nargs, PyObject* kw) {
    VF_TRY
    PyObject* sr;
    if (nargs == 1 && is_format(args[0]) && only_sr_kw(args, nargs, kw, &sr))
        return convert_to((PyFP*)s, args[0], sr).release().ptr();
    std::vector<PyObject*> a(args, args + nargs + (kw ? PyTuple_GET_SIZE(kw) : 0));
    a.insert(a.begin(), s);
    return PyObject_Vectorcall(g_convert_slow, a.data(), (size_t)nargs + 1, kw);
    VF_CATCH(nullptr)
}

// FP.from_value(value, fmt, *, sr_rand=None) fast path (classmethod).
static PyObject* m_from_value(PyObject* cls, PyObject* const* args, Py_ssize_t nargs, PyObject* kw) {
    VF_TRY
    PyObject* sr;
    if (nargs == 2 && is_format(args[1]) && only_sr_kw(args, nargs, kw, &sr))
        return fp_from_number(args[0], args[1], sr).release().ptr();
    std::vector<PyObject*> a(args, args + nargs + (kw ? PyTuple_GET_SIZE(kw) : 0));
    a.insert(a.begin(), cls);
    return PyObject_Vectorcall(g_from_value_slow, a.data(), (size_t)nargs + 1, kw);
    VF_CATCH(nullptr)
}

static PyObject* m_reduce(PyObject* s, PyObject*) {
    VF_TRY
    PyFP* self = (PyFP*)s;
    nb::object restore = steal_checked(PyObject_GetAttrString((PyObject*)S.FP, "_restore"));
    nb::object raw = raw_of(self->c, *self->f);
    nb::object un = unrounded_of(self);
    return Py_BuildValue("(O(OOiO))", restore.ptr(), self->fmt, raw.ptr(), (int)self->flags, un.ptr());
    VF_CATCH(nullptr)
}

// ---------------------------------------------------------------- getters

static PyObject* g_format(PyObject* s, void*) { return Py_NewRef(((PyFP*)s)->fmt); }
static PyObject* g_sign(PyObject* s, void*) { return PyBool_FromLong(((PyFP*)s)->c.sign); }
static PyObject* g_exp(PyObject* s, void*) {
    VF_TRY
    PyFP* x = (PyFP*)s;
    return uint_from_raw64(S.UINT, x->c.field, x->f->E, false);
    VF_CATCH(nullptr)
}
static PyObject* g_mantissa(PyObject* s, void*) {
    VF_TRY
    PyFP* x = (PyFP*)s;
    if (!x->f->wide) return uint_from_raw64(S.UINT, x->c.mant, x->f->M, false);
    nb::object v = long_from_big(mant_big(x->c, *x->f));
    return uint_make(S.UINT, v.ptr(), x->f->M, false).release().ptr();
    VF_CATCH(nullptr)
}
static PyObject* g_size(PyObject* s, void*) { return PyLong_FromLongLong(((PyFP*)s)->f->size); }
static PyObject* g_raw(PyObject* s, void*) {
    VF_TRY
    PyFP* x = (PyFP*)s;
    return raw_of(x->c, *x->f).release().ptr();
    VF_CATCH(nullptr)
}
#define VF_PRED(name, expr)                                                     \
    static PyObject* g_##name(PyObject* s, void*) {                             \
        PyFP* x = (PyFP*)s;                                                     \
        return PyBool_FromLong(expr);                                           \
    }
VF_PRED(is_nan, is_nan(x->c, *x->f))
VF_PRED(is_inf, is_inf(x->c, *x->f))
VF_PRED(is_finite, is_finite(x->c, *x->f))
VF_PRED(is_snan, is_snan(x->c, *x->f))
VF_PRED(is_zero, is_zero(x->c, *x->f))
VF_PRED(is_subnormal, x->f->has_zero && x->c.field == 0 && !mant_zero(x->c, *x->f))
VF_PRED(is_normal, (op_fclass(x->c, *x->f) & 0x42) != 0)

static PyObject* g_flags(PyObject* s, void*) { return Py_NewRef(S.flags[((PyFP*)s)->flags & 31]); }
static PyObject* g_unrounded(PyObject* s, void*) {
    VF_TRY
    return unrounded_of((PyFP*)s).release().ptr();
    VF_CATCH(nullptr)
}
static PyObject* fp_repr(PyObject* s);
static PyObject* g_exact(PyObject* s, void*) {
    VF_TRY
    PyFP* x = (PyFP*)s;
    if (!is_finite(x->c, *x->f)) raise(PyExc_ValueError, py_repr(s) + " is not finite");
    return fp_exact(x->c, *x->f).release().ptr();
    VF_CATCH(nullptr)
}
// Format fields read through to the format.
static PyObject* g_fmt_field(PyObject* s, void* name) {
    return PyObject_GetAttrString(((PyFP*)s)->fmt, (const char*)name);
}

static PyObject* fp_repr(PyObject* s) {
    VF_TRY
    PyFP* x = (PyFP*)s;
    nb::object fl = steal_checked(PyFloat_FromDouble(fp_to_double(x->c, *x->f)));
    std::string r = "FP(" + py_repr(fl.ptr()) + ", " + py_str(x->fmt);
    if (x->flags) {
        nb::object name = steal_checked(PyObject_GetAttrString(S.flags[x->flags & 31], "name"));
        r += ", flags=" + py_str(name.ptr());
    }
    r += ")";
    return PyUnicode_FromStringAndSize(r.data(), (Py_ssize_t)r.size());
    VF_CATCH(nullptr)
}

static PyObject* fp_str(PyObject* s) {
    VF_TRY
    PyFP* x = (PyFP*)s;
    nb::object fl = steal_checked(PyFloat_FromDouble(fp_to_double(x->c, *x->f)));
    return PyObject_Str(fl.ptr());
    VF_CATCH(nullptr)
}

static PyObject* fp_getattro(PyObject* s, PyObject* name) {
    PyObject* r = PyObject_GenericGetAttr(s, name);
    if (!r && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
        PyErr_SetObject(PyExc_AttributeError, name);
    }
    return r;
}

static PyObject* fp_tp_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    VF_TRY
    // FP(sign, exp: UINT, mantissa: UINT, bias=None, signed=True, **modes):
    // the five named parameters by position or by name, the rest are modes.
    Py_ssize_t n = PyTuple_GET_SIZE(args);
    if (n > 5) raise(PyExc_TypeError, "FP() takes sign, exp, mantissa[, bias[, signed]]");
    nb::dict kw = kwargs ? nb::steal<nb::dict>(check(PyDict_Copy(kwargs))) : nb::dict();
    static const char* names[5] = {"sign", "exp", "mantissa", "bias", "signed"};
    nb::object given[5];
    for (Py_ssize_t i = 0; i < 5; ++i) {
        if (i < n) {
            if (kw.contains(names[i]))
                raise(PyExc_TypeError, std::string("FP() got multiple values for argument '") + names[i] + "'");
            given[i] = nb::borrow(PyTuple_GET_ITEM(args, i));
        } else if (kw.contains(names[i])) {
            given[i] = kw[names[i]];
            PyDict_DelItemString(kw.ptr(), names[i]);
        } else if (i < 3)
            raise(PyExc_TypeError, std::string("FP() missing required argument '") + names[i] + "'");
    }
    PyObject* sign = given[0].ptr();
    PyObject* exp = given[1].ptr();
    PyObject* mant = given[2].ptr();
    if (!is_uint(exp) || !is_uint(mant)) raise(PyExc_TypeError, "exp and mantissa must be UINT");
    nb::object bias = given[3].is_valid() ? given[3] : nb::none();
    nb::object sgn = given[4].is_valid() ? given[4] : nb::borrow(Py_True);
    nb::object ebits = steal_checked(PyLong_FromLongLong(((PyUInt*)exp)->bits));
    nb::object mbits = steal_checked(PyLong_FromLongLong(((PyUInt*)mant)->bits));
    nb::object fargs = nb::make_tuple(ebits, mbits, bias, sgn);
    nb::object fmt = steal_checked(PyObject_Call(S.FPFormat, fargs.ptr(), kw.ptr()));
    int sg = PyObject_IsTrue(sign);
    if (sg < 0) raise_current();
    const Fmt& f = fmt_of(fmt.ptr());
    if (sg && !f.is_signed) raise(PyExc_ValueError, "unsigned FP cannot have its sign set");
    nb::object ev = uint_val(exp), mv = uint_val(mant);
    Code c;
    c.sign = sg;
    c.field = PyLong_AsUnsignedLongLongMask(ev.ptr()) & f.top;
    if (!f.wide) c.mant = PyLong_AsUnsignedLongLongMask(mv.ptr()) & f.mask();
    else set_mant_big(c, f, big_from_long(mv.ptr()));
    if (PyErr_Occurred()) raise_current();
    PyFP* o = (PyFP*)type->tp_alloc(type, 0);
    if (!o) raise_current();
    o->fmt = fmt.release().ptr();
    o->f = &f;
    new (&o->c) Code(c);
    o->flags = 0;
    new (&o->u) Unr();
    return (PyObject*)o;
    VF_CATCH(nullptr)
}

static void fp_dealloc(PyObject* s) {
    PyFP* o = (PyFP*)s;
    o->c.~Code();
    o->u.~Unr();
    Py_XDECREF(o->fmt);
    PyTypeObject* tp = Py_TYPE(s);
    tp->tp_free(s);
    Py_DECREF(tp);
}

#define FASTKW (METH_FASTCALL | METH_KEYWORDS)
static PyMethodDef fp_methods[] = {
    {"fma", (PyCFunction)(void (*)(void))m_fma, FASTKW,
     "fma($self, /, b, c)\n--\n\nFused multiply-add: self * b + c with a single rounding."},
    {"sqrt", m_sqrt, METH_NOARGS, "sqrt($self, /)\n--\n\nSquare root, correctly rounded (IEEE 754 5.4.1)."},
    {"minimum", (PyCFunction)(void (*)(void))m_minimum, FASTKW, "minimum($self, /, other)\n--\n\n"},
    {"maximum", (PyCFunction)(void (*)(void))m_maximum, FASTKW, "maximum($self, /, other)\n--\n\n"},
    {"minimum_number", (PyCFunction)(void (*)(void))m_minimum_number, FASTKW,
     "minimum_number($self, /, other)\n--\n\nminimumNumber: a NaN operand is ignored (RISC-V fmin)."},
    {"maximum_number", (PyCFunction)(void (*)(void))m_maximum_number, FASTKW,
     "maximum_number($self, /, other)\n--\n\nmaximumNumber: a NaN operand is ignored (RISC-V fmax)."},
    {"remainder", (PyCFunction)(void (*)(void))m_remainder, FASTKW,
     "remainder($self, /, other)\n--\n\nIEEE 754 remainder: self - n * other, n = self / other rounded to the "
     "nearest integer (ties to even)."},
    {"fmod", (PyCFunction)(void (*)(void))m_fmod, FASTKW,
     "fmod($self, /, other)\n--\n\nC fmod: self - n * other, n = self / other truncated toward zero."},
    {"next_up", m_next_up, METH_NOARGS,
     "next_up($self, /)\n--\n\nIEEE 754 nextUp: the least value of the format above this one."},
    {"next_down", m_next_down, METH_NOARGS,
     "next_down($self, /)\n--\n\nIEEE 754 nextDown: the greatest value of the format below this one."},
    {"scaleb", (PyCFunction)(void (*)(void))m_scaleb, FASTKW,
     "scaleb($self, /, n)\n--\n\nIEEE 754 scaleB: self * 2**n, rounded into the format."},
    {"logb", m_logb, METH_NOARGS,
     "logb($self, /)\n--\n\nIEEE 754 logB: floor(log2(|self|)) as a value of the format."},
    {"fclass", m_fclass, METH_NOARGS,
     "fclass($self, /)\n--\n\nRISC-V fclass: a 10-bit mask with one bit set (bit 0 -inf, 1 negative normal, "
     "2 negative subnormal, 3 -0, 4 +0, 5 positive subnormal, 6 positive normal, 7 +inf, 8 signaling NaN, "
     "9 quiet NaN)."},
    {"copysign", (PyCFunction)(void (*)(void))m_copysign, FASTKW,
     "copysign($self, /, other)\n--\n\nIEEE 754 copySign: this value with the sign of ``other``. No flags."},
    {"fsgnj", (PyCFunction)(void (*)(void))m_fsgnj, FASTKW,
     "fsgnj($self, /, other)\n--\n\nRISC-V fsgnj: the same as copysign."},
    {"fsgnjn", (PyCFunction)(void (*)(void))m_fsgnjn, FASTKW,
     "fsgnjn($self, /, other)\n--\n\nRISC-V fsgnjn: this value with the opposite of the sign of ``other``."},
    {"fsgnjx", (PyCFunction)(void (*)(void))m_fsgnjx, FASTKW,
     "fsgnjx($self, /, other)\n--\n\nRISC-V fsgnjx: this value with its sign xor the sign of ``other``."},
    {"to_int", (PyCFunction)(void (*)(void))m_to_int, FASTKW,
     "to_int($self, /, bits=32, signed=True, rounding=None, exact=True)\n--\n\n"
     "Convert to a ``bits``-wide INT/UINT. Returns (value, flags)."},
    {"round_to_integral", (PyCFunction)(void (*)(void))m_round_to_integral, FASTKW,
     "round_to_integral($self, /, rounding=None, exact=False)\n--\n\n"
     "Round to an integral value in the same format (IEEE 754 roundToIntegral)."},
    {"compare", (PyCFunction)(void (*)(void))m_compare, FASTKW,
     "compare($self, /, other, signaling=False)\n--\n\n"
     "IEEE comparison. Returns (relation, flags), relation one of 'lt', 'eq', 'gt', 'unordered'."},
    {"eq", (PyCFunction)(void (*)(void))m_eq, FASTKW, "eq($self, /, other, signaling=False)\n--\n\n"},
    {"lt", (PyCFunction)(void (*)(void))m_lt, FASTKW, "lt($self, /, other, signaling=True)\n--\n\n"},
    {"le", (PyCFunction)(void (*)(void))m_le, FASTKW, "le($self, /, other, signaling=True)\n--\n\n"},
    {"to_bin", (PyCFunction)(void (*)(void))m_to_bin, FASTKW, "to_bin($self, /, sep=' ')\n--\n\n"},
    {"to_hex", m_to_hex, METH_NOARGS, "to_hex($self, /)\n--\n\n"},
    {"_convert_to", (PyCFunction)(void (*)(void))m_convert_to, FASTKW, nullptr},
    {"convert", (PyCFunction)(void (*)(void))m_convert, FASTKW,
     "convert($self, /, exp_bits=2, mantissa_bits=1, bias=None, signed=True, *, fmt=None, sr_rand=None, **modes)\n"
     "--\n\nRound into another format: an FPFormat, or format fields."},
    {"from_value", (PyCFunction)(void (*)(void))m_from_value, FASTKW | METH_CLASS,
     "from_value($type, /, value, exp_bits=2, mantissa_bits=1, bias=None, signed=True, *, fmt=None, sr_rand=None, "
     "**modes)\n--\n\nRound a number (int, float, Fraction, FP, UINT/INT) into a format."},
    {"__reduce__", m_reduce, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static PyGetSetDef fp_getset[] = {
    {"format", g_format, nullptr, nullptr, nullptr},
    {"sign", g_sign, nullptr, nullptr, nullptr},
    {"exp", g_exp, nullptr, nullptr, nullptr},
    {"mantissa", g_mantissa, nullptr, nullptr, nullptr},
    {"size", g_size, nullptr, "Storage width in bits: [sign +] exponent + mantissa.", nullptr},
    {"raw", g_raw, nullptr, nullptr, nullptr},
    {"is_nan", g_is_nan, nullptr, nullptr, nullptr},
    {"is_inf", g_is_inf, nullptr, nullptr, nullptr},
    {"is_finite", g_is_finite, nullptr, nullptr, nullptr},
    {"is_snan", g_is_snan, nullptr, "Signaling NaN: IEEE formats, quiet bit (mantissa MSB) clear.", nullptr},
    {"is_zero", g_is_zero, nullptr, nullptr, nullptr},
    {"is_subnormal", g_is_subnormal, nullptr, nullptr, nullptr},
    {"is_normal", g_is_normal, nullptr, "Finite, nonzero and not subnormal.", nullptr},
    {"flags", g_flags, nullptr, "Status flags raised by the operation that produced this value.", nullptr},
    {"unrounded", g_unrounded, nullptr,
     "Infinitely precise result before rounding: a Fraction, +-inf as a float, or None.", nullptr},
    {"exact", g_exact, nullptr, "The exact value this encoding represents (finite values only).", nullptr},
    {"exp_bits", g_fmt_field, nullptr, nullptr, (void*)"exp_bits"},
    {"mantissa_bits", g_fmt_field, nullptr, nullptr, (void*)"mantissa_bits"},
    {"bias", g_fmt_field, nullptr, nullptr, (void*)"bias"},
    {"signed", g_fmt_field, nullptr, nullptr, (void*)"signed"},
    {"inf_nan", g_fmt_field, nullptr, nullptr, (void*)"inf_nan"},
    {"has_zero", g_fmt_field, nullptr, nullptr, (void*)"has_zero"},
    {"saturate", g_fmt_field, nullptr, nullptr, (void*)"saturate"},
    {"wrap", g_fmt_field, nullptr, nullptr, (void*)"wrap"},
    {"ftz", g_fmt_field, nullptr, nullptr, (void*)"ftz"},
    {"rounding", g_fmt_field, nullptr, nullptr, (void*)"rounding"},
    {"sr_bits", g_fmt_field, nullptr, nullptr, (void*)"sr_bits"},
    {"nan_mode", g_fmt_field, nullptr, nullptr, (void*)"nan_mode"},
    {"tininess", g_fmt_field, nullptr, nullptr, (void*)"tininess"},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

// ---------------------------------------------------------------- FormatBase

static PyObject* fmtbase_new(PyTypeObject* type, PyObject*, PyObject*) {
    PyFormat* o = (PyFormat*)type->tp_alloc(type, 0);
    if (!o) return nullptr;
    new (&o->f) Fmt();
    return (PyObject*)o;
}
static void fmtbase_dealloc(PyObject* s) {
    PyTypeObject* tp = Py_TYPE(s);
    if (PyType_IS_GC(tp)) PyObject_GC_UnTrack(s);
    ((PyFormat*)s)->f.~Fmt();
    tp->tp_free(s);
    Py_DECREF(tp);
}

// FormatBase._setup(exp_bits, mantissa_bits, bias, signed, inf_nan(0/1/2),
//   has_zero, saturate, wrap, ftz, rounding_idx, sr_bits, nan_mode_idx,
//   tininess_before, intern_id)
static PyObject* fmtbase_setup(PyObject* s, PyObject* args) {
    VF_TRY
    long long E, M, bias, sr_bits, id;
    int sg, inf_nan, hz, sat, wrap, ftz, rnd, nm, tb;
    if (!PyArg_ParseTuple(args, "LLLpippppiLipL", &E, &M, &bias, &sg, &inf_nan, &hz, &sat, &wrap, &ftz, &rnd,
                          &sr_bits, &nm, &tb, &id))
        return nullptr;
    if (E > 60) raise(PyExc_ValueError, "exp_bits above 60 is not supported");
    if (bias > (1LL << 60) || bias < -(1LL << 60)) raise(PyExc_ValueError, "|bias| above 2**60 is not supported");
    if (sr_bits > (1LL << 20)) raise(PyExc_ValueError, "sr_bits above 2**20 is not supported");
    if (M > (1LL << 24)) raise(PyExc_ValueError, "mantissa_bits above 2**24 is not supported");
    Fmt& f = ((PyFormat*)s)->f;
    f.E = E; f.M = M; f.bias = bias; f.is_signed = sg;
    f.inf_nan = (uint8_t)inf_nan; f.has_zero = hz; f.saturate = sat; f.wrap = wrap; f.ftz = ftz;
    f.rounding = (uint8_t)rnd; f.sr_bits = sr_bits; f.nan_mode = (uint8_t)nm; f.tininess_before = tb;
    f.id = id;
    f.derive();
    Py_RETURN_NONE;
    VF_CATCH(nullptr)
}

// fmt(value, *, sr_rand=None): round a value into this format.
static PyObject* fmtbase_call(PyObject* s, PyObject* args, PyObject* kwargs) {
    VF_TRY
    Py_ssize_t n = PyTuple_GET_SIZE(args);
    if (n > 1) raise(PyExc_TypeError, "FPFormat() takes exactly one value");
    PyObject* value = n == 1 ? PyTuple_GET_ITEM(args, 0) : nullptr;
    PyObject* sr = Py_None;
    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *k, *v;
        while (PyDict_Next(kwargs, &pos, &k, &v)) {
            if (PyUnicode_CompareWithASCIIString(k, "sr_rand") == 0) sr = v;
            else if (PyUnicode_CompareWithASCIIString(k, "value") == 0 && !value) value = v;
            else raise(PyExc_TypeError, "unexpected keyword argument " + py_repr(k));
        }
    }
    if (!value) raise(PyExc_TypeError, "FPFormat() takes exactly one value");
    return fp_from_number(value, s, sr).release().ptr();
    VF_CATCH(nullptr)
}

static PyMethodDef fmtbase_methods[] = {
    {"_setup", fmtbase_setup, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

// ---------------------------------------------------------------- module functions

// _core.fp_make(fmt, sign, field, mant): FP._make (sign checked)
static nb::object fp_make(nb::handle fmt, bool sign, nb::handle field, nb::handle mant) {
    if (!is_format(fmt.ptr())) raise(PyExc_TypeError, "expected an FPFormat");
    const Fmt& f = fmt_of(fmt.ptr());
    if (sign && !f.is_signed) raise(PyExc_ValueError, "unsigned FP cannot have its sign set");
    Code c;
    c.sign = sign;
    c.field = PyLong_AsUnsignedLongLongMask(field.ptr()) & f.top;
    if (!f.wide) c.mant = PyLong_AsUnsignedLongLongMask(mant.ptr()) & f.mask();
    else set_mant_big(c, f, big_from_long(mant.ptr()));
    if (PyErr_Occurred()) raise_current();
    return fp_plain(fmt.ptr(), c);
}

static nb::object fp_from_raw(nb::handle fmt, nb::handle raw) {
    if (!is_format(fmt.ptr())) raise(PyExc_TypeError, "expected an FPFormat");
    return fp_plain(fmt.ptr(), code_from_raw(raw.ptr(), fmt_of(fmt.ptr())));
}

// Restore a pickled FP: from_raw plus flags and unrounded.
static nb::object fp_restore(nb::handle fmt, nb::handle raw, int flags, nb::handle unrounded) {
    nb::object r = fp_from_raw(fmt, raw);
    PyFP* o = (PyFP*)r.ptr();
    o->flags = (uint8_t)flags;
    if (unrounded.is_none()) o->u.kind = Unr::NONE;
    else if (PyFloat_Check(unrounded.ptr()) && std::isinf(PyFloat_AS_DOUBLE(unrounded.ptr())))
        o->u.kind = PyFloat_AS_DOUBLE(unrounded.ptr()) < 0 ? Unr::NEGINF : Unr::POSINF;
    else { o->u.kind = Unr::OBJ; o->u.obj = nb::borrow(unrounded); }
    return r;
}

// FP._round(x, fmt, zero_sign): (FP, flags int, event str | None), no finish.
static nb::object py_round(nb::handle x, nb::handle fmt, bool zero_sign) {
    SRArg sr;
    RoundOut r = round_number(x.ptr(), fmt_of(fmt.ptr()), zero_sign, sr, nullptr);
    nb::object v = fp_plain(fmt.ptr(), r.c);
    return nb::make_tuple(v, (int)r.flags, r.ev == EV_NONE ? nb::none() : nb::borrow(S.event_names[r.ev]));
}

// FP._encode(x, fmt, zero_sign, sr): round and finish (warnings).
static nb::object py_encode(nb::handle x, nb::handle fmt, bool zero_sign, nb::handle sr_rand) {
    SRArg sr;
    sr.given = sr_rand.ptr();
    nb::object unr;
    RoundOut r = round_number(x.ptr(), fmt_of(fmt.ptr()), zero_sign, sr, &unr);
    Unr u;
    u.kind = Unr::OBJ;
    u.obj = unr;
    return fp_finish(fmt.ptr(), r, std::move(u));
}

static nb::object py_from_value(nb::handle value, nb::handle fmt, nb::handle sr_rand) {
    if (!is_format(fmt.ptr())) raise(PyExc_TypeError, "expected an FPFormat");
    return fp_from_number(value.ptr(), fmt.ptr(), sr_rand.ptr());
}

static Unr unr_from_py(nb::handle u) {
    Unr r;
    if (u.is_none()) return r;
    if (PyFloat_Check(u.ptr()) && std::isinf(PyFloat_AS_DOUBLE(u.ptr()))) {
        r.kind = PyFloat_AS_DOUBLE(u.ptr()) < 0 ? Unr::NEGINF : Unr::POSINF;
        return r;
    }
    r.kind = Unr::OBJ;
    r.obj = nb::borrow(u);
    return r;
}

static Event event_from(nb::handle ev) {
    if (ev.is_none()) return EV_NONE;
    for (int i = 1; i < 8; ++i)
        if (PyUnicode_Compare(ev.ptr(), S.event_names[i]) == 0) return (Event)i;
    raise(PyExc_ValueError, "unknown rounding event " + py_repr(ev.ptr()));
}

static PyFP* as_fp(nb::handle h) {
    if (!is_fp(h.ptr())) raise(PyExc_TypeError, "expected an FP");
    return (PyFP*)h.ptr();
}

// FP._finish(result, flags, event, unrounded, fmt)
static nb::object py_finish(nb::handle result, int flags, nb::handle event, nb::handle unrounded, nb::handle fmt) {
    RoundOut o{as_fp(result)->c, (uint8_t)flags, event_from(event)};
    return fp_finish(fmt.ptr(), o, unr_from_py(unrounded));
}

// FP._arith(op, a, b, fmt) for a, b already in fmt
static nb::object py_arith(nb::handle op, nb::handle a, nb::handle b, nb::handle fmt) {
    const char* s = PyUnicode_AsUTF8(op.ptr());
    if (!s) raise_current();
    return arith(s[0], opnd_of(as_fp(a)), opnd_of(as_fp(b)), fmt.ptr());
}

// FP._nan_result(fmt, *operands, invalid=False) -> (FP, flags)
static nb::object py_nan_result(nb::handle fmt, nb::list operands, bool invalid) {
    std::vector<Opnd> ops;
    for (nb::handle o : operands) ops.push_back(opnd_of(as_fp(o)));
    std::vector<const Opnd*> ptrs;
    for (auto& o : ops) ptrs.push_back(&o);
    RoundOut r = nan_result(fmt_of(fmt.ptr()), ptrs, invalid);
    return nb::make_tuple(fp_plain(fmt.ptr(), r.c), nb::handle(S.flags[r.flags]));
}

// FP._inf_result(sign, fmt) -> (FP, flags, event)
static nb::object py_inf_result(bool sign, nb::handle fmt) {
    RoundOut r = inf_result(sign, fmt_of(fmt.ptr()));
    return nb::make_tuple(fp_plain(fmt.ptr(), r.c), nb::handle(S.flags[r.flags]),
                          nb::handle(S.event_names[r.ev]));
}

void register_fp(nb::module_& m) {
    static PyType_Slot fmt_slots[] = {
        {Py_tp_new, (void*)fmtbase_new},
        {Py_tp_dealloc, (void*)fmtbase_dealloc},
        {Py_tp_call, (void*)fmtbase_call},
        {Py_tp_methods, fmtbase_methods},
        {Py_tp_doc, (void*)"Native part of FPFormat (precomputed geometry)."},
        {0, nullptr}};
    static PyType_Spec fmt_spec = {"verifloat._core.FormatBase", sizeof(PyFormat), 0,
                                   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, fmt_slots};
    S.FormatBase = (PyTypeObject*)check(PyType_FromSpec(&fmt_spec));
    S.FormatBase->tp_name = "FormatBase";
    m.attr("FormatBase") = nb::handle((PyObject*)S.FormatBase);

    static PyType_Slot fp_slots[] = {
        {Py_tp_new, (void*)fp_tp_new},
        {Py_tp_dealloc, (void*)fp_dealloc},
        {Py_tp_repr, (void*)fp_repr},
        {Py_tp_str, (void*)fp_str},
        {Py_tp_hash, (void*)fp_hash},
        {Py_tp_richcompare, (void*)fp_richcompare},
        {Py_tp_getattro, (void*)fp_getattro},
        {Py_tp_methods, fp_methods},
        {Py_tp_getset, fp_getset},
        {Py_nb_add, (void*)fp_add},
        {Py_nb_subtract, (void*)fp_sub},
        {Py_nb_multiply, (void*)fp_mul},
        {Py_nb_true_divide, (void*)fp_div},
        {Py_nb_negative, (void*)fp_neg},
        {Py_nb_positive, (void*)fp_pos},
        {Py_nb_absolute, (void*)fp_abs},
        {Py_nb_bool, (void*)fp_bool},
        {Py_nb_and, (void*)fp_and},
        {Py_nb_or, (void*)fp_or},
        {Py_nb_xor, (void*)fp_xor},
        {Py_nb_invert, (void*)fp_invert},
        {Py_nb_float, (void*)fp_float},
        {Py_nb_int, (void*)fp_int},
        {Py_tp_doc, (void*)"FP(sign, exp, mantissa, bias=None, signed=True, **modes)\n--\n\n"
                           "A bit-accurate floating-point value."},
        {0, nullptr}};
    static PyType_Spec fp_spec = {"verifloat.fp.FP", sizeof(PyFP), 0, Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                                  fp_slots};
    S.FP = (PyTypeObject*)check(PyType_FromSpec(&fp_spec));
    // Bare tp_name, as for a Python class (error messages say 'FP');
    // __module__ comes from the dotted spec name.
    S.FP->tp_name = "FP";
    m.attr("FP") = nb::handle((PyObject*)S.FP);

    m.def("fp_make", &fp_make);
    m.def("fp_from_raw", &fp_from_raw);
    m.def("fp_restore", &fp_restore, nb::arg(), nb::arg(), nb::arg(), nb::arg().none());
    m.def("fp_round", &py_round);
    m.def("fp_encode", &py_encode, nb::arg(), nb::arg(), nb::arg(), nb::arg().none());
    m.def("fp_from_value", &py_from_value, nb::arg(), nb::arg(), nb::arg().none());
    m.def("fp_finish", &py_finish, nb::arg(), nb::arg(), nb::arg().none(), nb::arg().none(), nb::arg());
    m.def("fp_arith", &py_arith);
    m.def("fp_nan_result", &py_nan_result);
    m.def("fp_inf_result", &py_inf_result);
    m.def("_init_slow", [](nb::handle from_value, nb::handle convert) {
        g_from_value_slow = Py_NewRef(from_value.ptr());
        g_convert_slow = Py_NewRef(convert.ptr());
    });
    m.def("fmt_str", [](nb::handle fmt) { return fmt_str(fmt_of(fmt.ptr())); });
    // The exact remainder of two dyadics xs * 2**xe and ys * 2**ye (signed
    // integers xs, ys): (rs, re). For the tests: FP operands of one format
    // reach only part of rem_dy.
    // floor(sqrt(n)) by the core's integer square roots. For the tests: the
    // operands a square root gives them are a small part of what they accept.
    m.def("_isqrt", [](nb::handle n) {
        const BigInt v = big_from_long(n.ptr());
        if (v < 0) raise(PyExc_ValueError, "isqrt of a negative number");
        return long_from_big(isqrt_big(v));
    });
    m.def("_rem_exact", [](nb::handle xs, long long xe, nb::handle ys, long long ye, bool truncate) {
        const BigInt a = big_from_long(xs.ptr()), b = big_from_long(ys.ptr());
        if (b == 0) raise(PyExc_ZeroDivisionError, "remainder by zero");
        const Dy<BigInt> x{a < 0, xe, abs(a), false}, y{b < 0, ye, abs(b), false};
        const Dy<BigInt> r = rem_dy(x, y, truncate);
        return nb::make_tuple(long_from_big(r.neg ? BigInt(-r.sig) : r.sig), (long long)r.exp);
    });
}

}  // namespace vf
