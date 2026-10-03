// FIXED: fixed-point values with exact bit-growing arithmetic (a port of
// verifloat_py/fixed.py). value = val * 2**-frac_bits.
#include "fp.hpp"
#include "args.hpp"
#include "uint.hpp"

#include <cmath>
#include <map>
#include <tuple>

namespace vf {

nb::object fp_exact(const Code& c, const Fmt& f);

struct FixFmt {
    int64_t ib = 1, fb = 0;
    bool is_signed = true;
    uint8_t rounding = RNE;
    bool sat = false;
    int64_t bits() const { return ib + fb; }
};

struct PyFixFmt {
    PyObject_HEAD
    FixFmt f;
};

struct PyFixed {
    PyObject_HEAD
    PyObject* fmt;   // Python FixedFormat (strong)
    const FixFmt* f;
    BigInt val;
    uint8_t flags;
};

static PyTypeObject* FixBase = nullptr;
static PyTypeObject* FixedT = nullptr;
static PyObject* FixedFormatCls = nullptr;
static PyObject* CastWarningCls = nullptr;

static bool is_fixfmt(PyObject* o) { return PyObject_TypeCheck(o, FixBase); }
static bool is_fixed(PyObject* o) { return PyObject_TypeCheck(o, FixedT); }
static const FixFmt& ffmt(PyObject* o) { return ((PyFixFmt*)o)->f; }

static BigInt min_raw(const FixFmt& f) { return f.is_signed ? BigInt(-(BigInt(1) << (unsigned)(f.bits() - 1))) : BigInt(0); }
static BigInt max_raw(const FixFmt& f) { return (BigInt(1) << (unsigned)(f.bits() - (f.is_signed ? 1 : 0))) - 1; }

// _wrap(q, fmt): two's complement into the format's width
static BigInt wrap(const BigInt& q, const FixFmt& f) {
    BigInt m = BigInt(1) << (unsigned)f.bits();
    BigInt r = q % m;
    if (r < 0) r += m;
    if (f.is_signed && bit(r, f.bits() - 1)) r -= m;
    return r;
}

static PyObject* fixed_new(PyObject* fmt, BigInt val, uint8_t flags) {
    PyFixed* o = (PyFixed*)PyObject_Malloc(FixedT->tp_basicsize);
    if (!o) { PyErr_NoMemory(); raise_current(); }
    PyObject_Init((PyObject*)o, FixedT);
    Py_INCREF(fmt);
    o->fmt = fmt;
    o->f = &ffmt(fmt);
    new (&o->val) BigInt(std::move(val));
    o->flags = flags;
    return (PyObject*)o;
}

// Formats produced by bit growth, created once per distinct shape.
static std::map<std::tuple<int64_t, int64_t, bool, uint8_t, bool>, PyObject*> g_formats;

static PyObject* format_for(const FixFmt& f) {
    auto key = std::make_tuple(f.ib, f.fb, f.is_signed, f.rounding, f.sat);
    auto it = g_formats.find(key);
    if (it != g_formats.end()) return it->second;
    nb::dict kw;
    kw["rounding"] = nb::handle(S.rounding_enum[f.rounding]);
    kw["overflow"] = nb::str(f.sat ? "saturate" : "wrap");
    nb::object args = nb::make_tuple(f.ib, f.fb, f.is_signed);
    PyObject* r = check(PyObject_Call(FixedFormatCls, args.ptr(), kw.ptr()));
    g_formats.emplace(key, r);
    return r;
}

// Exact value of a Python operand as a dyadic or a ratio (fixed.py _to_fraction).
struct Rat {
    BigInt n;
    BigInt d = 1;   // > 0
};

static Rat rat_from_fraction(PyObject* frac) {
    static PyObject* sn = PyUnicode_InternFromString("_numerator");
    static PyObject* sd = PyUnicode_InternFromString("_denominator");
    nb::object n = steal_checked(PyObject_GetAttr(frac, sn)), d = steal_checked(PyObject_GetAttr(frac, sd));
    return {big_from_long(n.ptr()), big_from_long(d.ptr())};
}

static Rat rat_from_dy(const Dy<BigInt>& v) {
    Rat r;
    BigInt s = v.neg ? BigInt(-v.sig) : v.sig;
    if (v.exp >= 0) r.n = s << (unsigned)v.exp;
    else { r.n = s; r.d = BigInt(1) << (unsigned)-v.exp; }
    return r;
}

static Rat to_rat(PyObject* v) {
    if (is_fixed(v)) {
        PyFixed* x = (PyFixed*)v;
        return rat_from_dy({x->val < 0, -x->f->fb, abs(x->val), false});
    }
    if (is_fp(v)) {
        PyFP* x = (PyFP*)v;
        if (!is_finite(x->c, *x->f)) {   // FP.exact raises
            nb::object e = steal_checked(PyObject_GetAttrString(v, "exact"));
        }
        return rat_from_dy(decode<BigInt>(x->c, *x->f));
    }
    if (is_uint(v)) {
        nb::object i = uint_val(v);
        return {big_from_long(i.ptr()), 1};
    }
    if (PyLong_Check(v)) return {big_from_long(v), 1};
    if (PyFloat_Check(v)) {
        double d = PyFloat_AS_DOUBLE(v);
        if (!double_finite(d)) raise(PyExc_ValueError, "fixed point has no inf/NaN");
        uint64_t sig;
        int64_t exp;
        double_parts(d, sig, exp);
        Dy<BigInt> dy{sig != 0 && double_neg(d), exp, BigInt(sig), false};
        return rat_from_dy(dy);
    }
    nb::object frac = is_fraction(v) ? nb::borrow(v) : steal_checked(PyObject_CallOneArg(S.Fraction, v));
    return rat_from_fraction(frac.ptr());
}

// _round_to_int(n / d, rounding) for d > 0: (q, inexact)
static std::pair<BigInt, bool> round_ratio(const BigInt& n, const BigInt& d, uint8_t rounding) {
    bool neg = n < 0;
    BigInt a = abs(n), q, r;
    boost::multiprecision::divide_qr(a, d, q, r);
    if (r != 0) {
        bool up;
        BigInt r2 = r * 2;
        switch (rounding) {
            case RNE: up = r2 > d || (r2 == d && bit(q, 0)); break;
            case RNA: up = r2 >= d; break;
            case RTZ: up = false; break;
            case RUP: up = !neg; break;
            default: up = neg; break;   // RDN (SR is rejected by FixedFormat)
        }
        if (up) q += 1;
    }
    return {neg ? BigInt(-q) : q, r != 0};
}

// FIXED.cast_value(value, fmt)
static PyObject* cast_value(PyObject* value, PyObject* fmt) {
    const FixFmt& f = ffmt(fmt);
    Rat x = to_rat(value);
    // x / ulp = x * 2**fb
    BigInt n = x.n, d = x.d;
    if (f.fb >= 0) n <<= (unsigned)f.fb;
    else d <<= (unsigned)-f.fb;
    auto [q, inexact] = round_ratio(n, d, f.rounding);
    uint8_t flags = inexact ? INEXACT : 0;
    BigInt lo = min_raw(f), hi = max_raw(f);
    if (q < lo || q > hi) {
        flags |= OVERFLOW;
        q = f.sat ? (q < lo ? lo : hi) : wrap(q, f);
    }
    return fixed_new(fmt, q, flags);
}

static nb::object exact_of(PyFixed* x) {
    return fraction_dy<BigInt>(x->val < 0, abs(x->val), -x->f->fb);
}

// _other: a FIXED, or a literal cast into self's format (warning if lossy)
static PyObject* other_of(PyFixed* self, PyObject* o, nb::object& hold) {
    if (is_fixed(o)) return o;
    if ((PyLong_Check(o) && !PyBool_Check(o)) || is_fraction(o) || PyFloat_Check(o)) {
        hold = nb::steal(cast_value(o, self->fmt));
        PyFixed* r = (PyFixed*)hold.ptr();
        if (r->flags) {
            nb::object ex = exact_of(r);
            nb::object fl = steal_checked(PyNumber_Float(ex.ptr()));
            nb::object name = steal_checked(PyObject_GetAttrString(S.flags[r->flags], "name"));
            warn(CastWarningCls, "implicit cast of " + py_repr(o) + " to " + py_str(self->fmt) + " is lossy: " +
                                     py_repr(fl.ptr()) + " (" + py_str(name.ptr()) + ")");
        }
        return hold.ptr();
    }
    return nullptr;
}

// FIXED._grow
static FixFmt grow(PyFixed* a, PyFixed* b, char op) {
    const FixFmt &fa = *a->f, &fb = *b->f;
    if (fa.is_signed != fb.is_signed)
        warn(S.IntCastWarning, "mixing signedness: " + py_str(a->fmt) + " and " + py_str(b->fmt) +
                                   ", the unsigned operand gains a sign bit");
    bool sgn = fa.is_signed || fb.is_signed || op == '-';
    int64_t ia = fa.ib + (sgn && !fa.is_signed), ib = fb.ib + (sgn && !fb.is_signed);
    FixFmt r;
    r.is_signed = sgn;
    r.rounding = fa.rounding;
    r.sat = fa.sat;
    if (op == '*') { r.ib = ia + ib; r.fb = fa.fb + fb.fb; }
    else { r.ib = std::max(ia, ib) + 1; r.fb = std::max(fa.fb, fb.fb); }
    return r;
}

static PyObject* binop(PyObject* x, PyObject* y, char op) {
    VF_TRY
    bool reverse = !is_fixed(x);
    PyFixed* self = (PyFixed*)(reverse ? y : x);
    nb::object hold;
    PyObject* o = other_of(self, reverse ? x : y, hold);
    if (!o) Py_RETURN_NOTIMPLEMENTED;
    PyFixed *a = self, *b = (PyFixed*)o;
    if (reverse) std::swap(a, b);
    FixFmt g = grow(a, b, op);
    BigInt v;
    if (op == '*') v = a->val * b->val;
    else {
        BigInt va = a->val << (unsigned)(g.fb - a->f->fb), vb = b->val << (unsigned)(g.fb - b->f->fb);
        v = op == '+' ? BigInt(va + vb) : BigInt(va - vb);
    }
    return fixed_new(format_for(g), v, 0);
    VF_CATCH(nullptr)
}
static PyObject* x_add(PyObject* a, PyObject* b) { return binop(a, b, '+'); }
static PyObject* x_sub(PyObject* a, PyObject* b) { return binop(a, b, '-'); }
static PyObject* x_mul(PyObject* a, PyObject* b) { return binop(a, b, '*'); }

static PyObject* x_neg(PyObject* s) {
    VF_TRY
    PyFixed* x = (PyFixed*)s;
    FixFmt g = *x->f;
    g.ib += 1;
    g.is_signed = true;
    return fixed_new(format_for(g), -x->val, 0);
    VF_CATCH(nullptr)
}

static PyObject* shift(PyObject* s, PyObject* n, bool right) {
    VF_TRY
    if (!is_fixed(s)) Py_RETURN_NOTIMPLEMENTED;
    PyFixed* x = (PyFixed*)s;
    nb::object k = steal_checked(PyNumber_Index(n));
    long long kk = PyLong_AsLongLong(k.ptr());
    if (kk == -1 && PyErr_Occurred()) raise_current();
    if (right) kk = -kk;
    // f.replace(int_bits=ib + n, frac_bits=fb - n)
    nb::object fmt = nb::handle(x->fmt).attr("replace")(nb::arg("int_bits") = x->f->ib + kk,
                                                        nb::arg("frac_bits") = x->f->fb - kk);
    return fixed_new(fmt.ptr(), x->val, 0);
    VF_CATCH(nullptr)
}
static PyObject* x_lshift(PyObject* s, PyObject* n) { return shift(s, n, false); }
static PyObject* x_rshift(PyObject* s, PyObject* n) { return shift(s, n, true); }

static int x_bool(PyObject* s) { return ((PyFixed*)s)->val != 0; }

static PyObject* x_float(PyObject* s) {
    VF_TRY
    nb::object e = exact_of((PyFixed*)s);
    return PyNumber_Float(e.ptr());
    VF_CATCH(nullptr)
}

static PyObject* x_int(PyObject* s) {
    VF_TRY
    PyFixed* x = (PyFixed*)s;
    int64_t fb = x->f->fb;
    BigInt v = fb <= 0 ? BigInt(x->val << (unsigned)-fb) : BigInt(abs(x->val) >> (unsigned)fb);
    if (fb > 0 && x->val < 0) v = -v;
    return long_from_big(v).release().ptr();
    VF_CATCH(nullptr)
}

static int cmp_rat(const Rat& a, const Rat& b) {
    BigInt l = a.n * b.d, r = b.n * a.d;
    return l < r ? -1 : l > r ? 1 : 0;
}

static PyObject* x_richcompare(PyObject* s, PyObject* o, int op) {
    VF_TRY
    PyFixed* x = (PyFixed*)s;
    Rat a = rat_from_dy({x->val < 0, -x->f->fb, abs(x->val), false});
    Rat b;
    if (op == Py_EQ || op == Py_NE) {
        try {
            b = to_rat(o);
        } catch (nb::python_error& e) {
            if (e.matches(PyExc_TypeError) || e.matches(PyExc_ValueError)) Py_RETURN_NOTIMPLEMENTED;
            throw;
        }
        bool eq = cmp_rat(a, b) == 0;
        return PyBool_FromLong(op == Py_EQ ? eq : !eq);
    }
    b = to_rat(o);
    int k = cmp_rat(a, b);
    bool r = op == Py_LT ? k < 0 : op == Py_LE ? k <= 0 : op == Py_GT ? k > 0 : k >= 0;
    return PyBool_FromLong(r);
    VF_CATCH(nullptr)
}

static Py_hash_t x_hash(PyObject* s) {
    VF_TRY
    nb::object e = exact_of((PyFixed*)s);
    return PyObject_Hash(e.ptr());
    VF_CATCH(-1)
}

static PyObject* x_repr(PyObject* s) {
    VF_TRY
    PyFixed* x = (PyFixed*)s;
    nb::object e = exact_of(x);
    nb::object fl = steal_checked(PyNumber_Float(e.ptr()));
    std::string r = "FIXED(" + py_repr(fl.ptr()) + ", " + py_str(x->fmt);
    if (x->flags) {
        nb::object name = steal_checked(PyObject_GetAttrString(S.flags[x->flags], "name"));
        r += ", flags=" + py_str(name.ptr());
    }
    r += ")";
    return PyUnicode_FromStringAndSize(r.data(), (Py_ssize_t)r.size());
    VF_CATCH(nullptr)
}

static void x_dealloc(PyObject* s) {
    PyFixed* x = (PyFixed*)s;
    x->val.~BigInt();
    Py_XDECREF(x->fmt);
    PyTypeObject* tp = Py_TYPE(s);
    tp->tp_free(s);
    Py_DECREF(tp);
}

static nb::object raw_of(PyFixed* x) {
    BigInt r = x->val;
    if (r < 0) r += BigInt(1) << (unsigned)x->f->bits();
    return long_from_big(r);
}

static PyObject* g_format(PyObject* s, void*) { return Py_NewRef(((PyFixed*)s)->fmt); }
static PyObject* g_val(PyObject* s, void*) {
    VF_TRY return long_from_big(((PyFixed*)s)->val).release().ptr(); VF_CATCH(nullptr)
}
static PyObject* g_raw(PyObject* s, void*) {
    VF_TRY return raw_of((PyFixed*)s).release().ptr(); VF_CATCH(nullptr)
}
static PyObject* g_exact(PyObject* s, void*) {
    VF_TRY return exact_of((PyFixed*)s).release().ptr(); VF_CATCH(nullptr)
}
static PyObject* g_flags(PyObject* s, void*) { return Py_NewRef(S.flags[((PyFixed*)s)->flags & 31]); }
static PyObject* g_signed(PyObject* s, void*) { return PyBool_FromLong(((PyFixed*)s)->f->is_signed); }
static PyObject* g_int_bits(PyObject* s, void*) { return PyLong_FromLongLong(((PyFixed*)s)->f->ib); }
static PyObject* g_frac_bits(PyObject* s, void*) { return PyLong_FromLongLong(((PyFixed*)s)->f->fb); }
static PyObject* g_bits(PyObject* s, void*) { return PyLong_FromLongLong(((PyFixed*)s)->f->bits()); }

static PyObject* format_raw(PyObject* s, const std::string& spec) {
    nb::object raw = raw_of((PyFixed*)s);
    nb::object sp = steal_checked(PyUnicode_FromString(spec.c_str()));
    return PyObject_Format(raw.ptr(), sp.ptr());
}
static PyObject* m_to_bin(PyObject* s, PyObject*) {
    VF_TRY return format_raw(s, "0" + std::to_string(((PyFixed*)s)->f->bits()) + "b"); VF_CATCH(nullptr)
}
static PyObject* m_to_hex(PyObject* s, PyObject*) {
    VF_TRY return format_raw(s, "0" + std::to_string((((PyFixed*)s)->f->bits() + 3) / 4) + "x"); VF_CATCH(nullptr)
}

// div(other, fmt): quotient rounded into an explicit format
static PyObject* m_div(PyObject* s, PyObject* const* args0, Py_ssize_t nargs, PyObject* kw) {
    VF_TRY
    PyObject* args[2];
    parse_args("div", args0, nargs, kw, {"other", "fmt"}, args, 2);
    PyFixed* self = (PyFixed*)s;
    nb::object hold;
    PyObject* o = other_of(self, args[0], hold);
    if (!o) raise(PyExc_AttributeError, "'NoneType' object has no attribute 'exact'");
    PyFixed* b = (PyFixed*)o;
    if (b->val == 0) raise(PyExc_ZeroDivisionError, "fixed-point division by zero");
    PyObject* fmt = args[1];
    if (!is_fixfmt(fmt)) raise(PyExc_TypeError, "fmt must be a FixedFormat");
    const FixFmt& f = ffmt(fmt);
    // (va / 2**fa) / (vb / 2**fb) * 2**F
    BigInt n = self->val, d = b->val;
    int64_t sh = b->f->fb + f.fb - self->f->fb;
    if (d < 0) { n = -n; d = -d; }
    if (sh >= 0) n <<= (unsigned)sh;
    else d <<= (unsigned)-sh;
    auto [q, inexact] = round_ratio(n, d, f.rounding);
    uint8_t flags = inexact ? INEXACT : 0;
    BigInt lo = min_raw(f), hi = max_raw(f);
    if (q < lo || q > hi) {
        flags |= OVERFLOW;
        q = f.sat ? (q < lo ? lo : hi) : wrap(q, f);
    }
    return fixed_new(fmt, q, flags);
    VF_CATCH(nullptr)
}

static PyMethodDef x_methods[] = {
    {"to_bin", m_to_bin, METH_NOARGS, "to_bin($self, /)\n--\n\n"},
    {"to_hex", m_to_hex, METH_NOARGS, "to_hex($self, /)\n--\n\n"},
    {"div", (PyCFunction)(void (*)(void))m_div, METH_FASTCALL | METH_KEYWORDS,
     "div($self, /, other, fmt)\n--\n\nQuotient rounded into an explicit format (division is not exact)."},
    {nullptr, nullptr, 0, nullptr}};

static PyGetSetDef x_getset[] = {
    {"format", g_format, nullptr, nullptr, nullptr},
    {"val", g_val, nullptr, "The scaled integer (signed if the format is).", nullptr},
    {"raw", g_raw, nullptr, "Bit pattern (two's complement).", nullptr},
    {"exact", g_exact, nullptr, nullptr, nullptr},
    {"flags", g_flags, nullptr, "INEXACT and/or OVERFLOW from the cast that produced this value.", nullptr},
    {"signed", g_signed, nullptr, nullptr, nullptr},
    {"int_bits", g_int_bits, nullptr, nullptr, nullptr},
    {"frac_bits", g_frac_bits, nullptr, nullptr, nullptr},
    {"bits", g_bits, nullptr, nullptr, nullptr},
    {"size", g_bits, nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

static PyObject* x_tp_new(PyTypeObject*, PyObject*, PyObject*) {
    PyErr_SetString(PyExc_TypeError, "build FIXED values with a FixedFormat: FixedFormat(4, 4)(1.5)");
    return nullptr;
}

// ---------------------------------------------------------------- FixedBase

static PyObject* fb_new(PyTypeObject* type, PyObject*, PyObject*) {
    PyFixFmt* o = (PyFixFmt*)type->tp_alloc(type, 0);
    if (!o) return nullptr;
    new (&o->f) FixFmt();
    return (PyObject*)o;
}
static void fb_dealloc(PyObject* s) {
    PyTypeObject* tp = Py_TYPE(s);
    if (PyType_IS_GC(tp)) PyObject_GC_UnTrack(s);
    tp->tp_free(s);
    Py_DECREF(tp);
}
static PyObject* fb_setup(PyObject* s, PyObject* args) {
    long long ib, fbits;
    int sg, rnd, sat;
    if (!PyArg_ParseTuple(args, "LLpip", &ib, &fbits, &sg, &rnd, &sat)) return nullptr;
    FixFmt& f = ((PyFixFmt*)s)->f;
    f.ib = ib; f.fb = fbits; f.is_signed = sg; f.rounding = (uint8_t)rnd; f.sat = sat;
    Py_RETURN_NONE;
}
static PyObject* fb_call(PyObject* s, PyObject* args, PyObject* kw) {
    VF_TRY
    // fmt(value): positional or by name
    Py_ssize_t n = PyTuple_GET_SIZE(args), nk = kw ? PyDict_GET_SIZE(kw) : 0;
    PyObject* value = n == 1 && nk == 0 ? PyTuple_GET_ITEM(args, 0) : nullptr;
    if (n == 0 && nk == 1) value = PyDict_GetItemString(kw, "value");
    if (!value) raise(PyExc_TypeError, "FixedFormat() takes exactly one value");
    return cast_value(value, s);
    VF_CATCH(nullptr)
}
static PyObject* fb_from_raw(PyObject* s, PyObject* const* args, Py_ssize_t nargs, PyObject* kw) {
    VF_TRY
    PyObject* a[1];
    parse_args("from_raw", args, nargs, kw, {"raw"}, a, 1);
    nb::object i = steal_checked(PyNumber_Index(a[0]));
    BigInt r = big_from_long(i.ptr());
    return fixed_new(s, wrap(r, ffmt(s)), 0);
    VF_CATCH(nullptr)
}
static PyMethodDef fb_methods[] = {
    {"_setup", fb_setup, METH_VARARGS, nullptr},
    {"from_raw", (PyCFunction)(void (*)(void))fb_from_raw, METH_FASTCALL | METH_KEYWORDS,
     "from_raw($self, /, raw)\n--\n\nBuild from a bit pattern (two's complement if signed)."},
    {nullptr, nullptr, 0, nullptr}};

static PyObject* py_cast_value(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    VF_TRY
    if (nargs != 2 || !is_fixfmt(args[1])) raise(PyExc_TypeError, "cast_value(value, fmt)");
    return cast_value(args[0], args[1]);
    VF_CATCH(nullptr)
}

void register_fixed(nb::module_& m) {
    static PyType_Slot fb_slots[] = {
        {Py_tp_new, (void*)fb_new}, {Py_tp_dealloc, (void*)fb_dealloc}, {Py_tp_call, (void*)fb_call},
        {Py_tp_methods, fb_methods}, {0, nullptr}};
    static PyType_Spec fb_spec = {"verifloat._core.FixedBase", sizeof(PyFixFmt), 0,
                                  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, fb_slots};
    FixBase = (PyTypeObject*)check(PyType_FromSpec(&fb_spec));
    FixBase->tp_name = "FixedBase";
    m.attr("FixedBase") = nb::handle((PyObject*)FixBase);

    static PyType_Slot slots[] = {
        {Py_tp_new, (void*)x_tp_new}, {Py_tp_dealloc, (void*)x_dealloc}, {Py_tp_repr, (void*)x_repr},
        {Py_tp_hash, (void*)x_hash}, {Py_tp_richcompare, (void*)x_richcompare},
        {Py_tp_methods, x_methods}, {Py_tp_getset, x_getset},
        {Py_nb_add, (void*)x_add}, {Py_nb_subtract, (void*)x_sub}, {Py_nb_multiply, (void*)x_mul},
        {Py_nb_negative, (void*)x_neg}, {Py_nb_lshift, (void*)x_lshift}, {Py_nb_rshift, (void*)x_rshift},
        {Py_nb_bool, (void*)x_bool}, {Py_nb_float, (void*)x_float}, {Py_nb_int, (void*)x_int},
        {Py_tp_doc, (void*)"A fixed-point value: an integer ``val`` scaled by 2**-frac_bits."},
        {0, nullptr}};
    static PyType_Spec spec = {"verifloat.fixed.FIXED", sizeof(PyFixed), 0,
                               Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, slots};
    FixedT = (PyTypeObject*)check(PyType_FromSpec(&spec));
    FixedT->tp_name = "FIXED";
    m.attr("FIXED") = nb::handle((PyObject*)FixedT);

    static PyMethodDef cv = {"fixed_cast_value", (PyCFunction)(void (*)(void))py_cast_value, METH_FASTCALL, nullptr};
    m.attr("fixed_cast_value") = steal_checked(PyCFunction_New(&cv, nullptr));
    m.def("fixed_with_flags", [](nb::handle x, int flags) {
        if (!is_fixed(x.ptr())) raise(PyExc_TypeError, "expected a FIXED");
        PyFixed* f = (PyFixed*)x.ptr();
        return nb::steal(fixed_new(f->fmt, f->val, (uint8_t)flags));
    });
    m.def("_init_fixed", [](nb::handle fixed_format, nb::handle cast_warning) {
        FixedFormatCls = Py_NewRef(fixed_format.ptr());
        CastWarningCls = Py_NewRef(cast_warning.ptr());
    });
}

}  // namespace vf
