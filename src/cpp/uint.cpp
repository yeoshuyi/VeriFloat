// UINT / INT: a port of verifloat_py/uint.py and sint.py. Values up to 64
// bits use native arithmetic; wider ones use Python ints.
#include "uint.hpp"
#include "args.hpp"

#include <string>

namespace vf {

// ---------------------------------------------------------------- helpers

static nb::object pyint(i128 v) { return long_from_i128(v); }
static nb::object pyint64(int64_t v) { return steal_checked(PyLong_FromLongLong(v)); }

static nb::object py_shl1(int64_t k) {   // 1 << k
    nb::object one = pyint64(1), kk = pyint64(k);
    return steal_checked(PyNumber_Lshift(one.ptr(), kk.ptr()));
}
static nb::object py_op(int op, PyObject* a, PyObject* b) {
    PyObject* r;
    switch (op) {
        case '+': r = PyNumber_Add(a, b); break;
        case '-': r = PyNumber_Subtract(a, b); break;
        case '*': r = PyNumber_Multiply(a, b); break;
        case '&': r = PyNumber_And(a, b); break;
        case '|': r = PyNumber_Or(a, b); break;
        case '^': r = PyNumber_Xor(a, b); break;
        case '<': r = PyNumber_Lshift(a, b); break;
        case '>': r = PyNumber_Rshift(a, b); break;
        default: r = nullptr;
    }
    return steal_checked(r);
}
static bool py_lt(PyObject* a, PyObject* b) {
    int r = PyObject_RichCompareBool(a, b, Py_LT);
    if (r < 0) raise_current();
    return r;
}

static i128 min_i(bool sgn, int64_t bits) { return sgn ? -((i128)1 << (bits - 1)) : 0; }
static i128 max_i(bool sgn, int64_t bits) { return ((i128)1 << (bits - (sgn ? 1 : 0))) - 1; }
static nb::object min_py(bool sgn, int64_t bits) {
    if (!sgn) return pyint64(0);
    return steal_checked(PyNumber_Negative(py_shl1(bits - 1).ptr()));
}
static nb::object max_py(bool sgn, int64_t bits) {
    nb::object p = py_shl1(bits - (sgn ? 1 : 0)), one = pyint64(1);
    return steal_checked(PyNumber_Subtract(p.ptr(), one.ptr()));
}
static nb::object mask_py(int64_t bits) {
    nb::object p = py_shl1(bits), one = pyint64(1);
    return steal_checked(PyNumber_Subtract(p.ptr(), one.ptr()));
}

static PyUInt* alloc(PyTypeObject* t) {
    PyUInt* o;
    if (t == S.UINT || t == S.INT) {
        o = (PyUInt*)PyObject_Malloc(t->tp_basicsize);
        if (!o) { PyErr_NoMemory(); raise_current(); }
        PyObject_Init((PyObject*)o, t);
    } else {
        o = (PyUInt*)t->tp_alloc(t, 0);
        if (!o) raise_current();
    }
    o->big = nullptr;
    return o;
}

PyObject* uint_from_raw64(PyTypeObject* t, uint64_t raw, int64_t bits, bool sat) {
    PyUInt* o = alloc(t);
    o->bits = bits;
    o->sat = sat;
    o->sgn = type_signed(t);
    o->v = raw & mask64(bits);
    if (bits > 64) {
        o->v = 0;
        o->big = steal_checked(PyLong_FromUnsignedLongLong(raw)).release().ptr();
    }
    return (PyObject*)o;
}

// type(val, bits, clamp) with an i128 value (bits <= 64): clamp, then wrap.
static PyObject* make_small(PyTypeObject* t, i128 v, int64_t bits, bool clamp, bool sat) {
    bool sgn = type_signed(t);
    if (clamp) v = std::min(std::max(v, min_i(sgn, bits)), max_i(sgn, bits));
    PyUInt* o = alloc(t);
    o->bits = bits;
    o->sat = sat;
    o->sgn = sgn;
    o->v = (uint64_t)(u128)v & mask64(bits);
    return (PyObject*)o;
}

// type(val, bits, clamp) for any Python int value.
static nb::object make_any(PyTypeObject* t, PyObject* val, int64_t bits, bool clamp, bool sat) {
    bool sgn = type_signed(t);
    if (bits <= 64) {
        int overflow = 0;
        long long x = PyLong_AsLongLongAndOverflow(val, &overflow);
        if (x == -1 && PyErr_Occurred()) raise_current();
        if (!overflow) return nb::steal(make_small(t, x, bits, clamp, sat));
    }
    nb::object v = nb::borrow(val);
    if (clamp) {
        nb::object lo = min_py(sgn, bits), hi = max_py(sgn, bits);
        if (py_lt(v.ptr(), lo.ptr())) v = lo;
        else if (py_lt(hi.ptr(), v.ptr())) v = hi;
    }
    nb::object m = mask_py(bits);
    nb::object raw = py_op('&', v.ptr(), m.ptr());
    PyUInt* o = alloc(t);
    o->bits = bits;
    o->sat = sat;
    o->sgn = sgn;
    if (bits <= 64) {
        o->v = PyLong_AsUnsignedLongLong(raw.ptr());
        if (PyErr_Occurred()) { Py_DECREF(o); raise_current(); }
    } else {
        o->v = 0;
        o->big = raw.release().ptr();
    }
    return nb::steal((PyObject*)o);
}

nb::object uint_make(PyTypeObject* t, PyObject* val, int64_t bits, bool sat) {
    return make_any(t, val, bits, sat, sat);
}

static inline PyUInt* U(PyObject* o) { return (PyUInt*)o; }
static inline bool small(PyUInt* x) { return x->bits <= 64; }

static i128 val_i(PyUInt* x) {   // small only
    if (x->sgn && x->bits && (x->v >> (x->bits - 1)) & 1) return (i128)x->v - ((i128)1 << x->bits);
    return (i128)x->v;
}
static nb::object raw_py(PyUInt* x) {
    if (!small(x)) return nb::borrow(x->big);
    return steal_checked(PyLong_FromUnsignedLongLong(x->v));
}
static nb::object val_py(PyUInt* x) {
    if (small(x)) return pyint(val_i(x));
    if (x->sgn) {
        nb::object top = py_shl1(x->bits - 1);
        if (!py_lt(x->big, top.ptr())) {
            nb::object m = py_shl1(x->bits);
            return steal_checked(PyNumber_Subtract(x->big, m.ptr()));
        }
    }
    return nb::borrow(x->big);
}
nb::object uint_val(PyObject* o) { return val_py(U(o)); }

static std::string kind(bool sgn, int64_t bits) { return (sgn ? "i" : "u") + std::to_string(bits); }

// ---------------------------------------------------------------- coercion

struct ICo {
    bool fast = false;
    i128 a = 0, b = 0;
    nb::object pa, pb;
    int64_t bits = 0;
    PyTypeObject* cls = nullptr;
};

// UINT._coerce: operands as values (both signed) or raw bits (otherwise).
static bool coerce(PyUInt* self, PyObject* other, ICo& c) {
    if (is_uint(other)) {
        PyUInt* o = U(other);
        if (self->sat != o->sat)
            warn(S.IntCastWarning, std::string("mismatched saturate: ") + (self->sat ? "True" : "False") +
                                       " (left) vs " + (o->sat ? "True" : "False") + " (right), using " +
                                       (self->sat ? "True" : "False"));
        c.bits = std::max(self->bits, o->bits);
        bool both = self->sgn && o->sgn;
        c.cls = both ? Py_TYPE(self) : (self->sgn ? Py_TYPE(o) : Py_TYPE(self));
        if (self->sgn != o->sgn)
            warn(S.IntCastWarning, "mixing signedness: " + kind(self->sgn, self->bits) + " and " +
                                       kind(o->sgn, o->bits) + ", signed operand reinterpreted as u" +
                                       std::to_string(c.bits) + " raw bits");
        else if (self->bits != o->bits)
            warn(S.IntCastWarning, "mixing widths: " + kind(self->sgn, self->bits) + " and " +
                                       kind(o->sgn, o->bits) + ", " +
                                       (self->sgn ? "sign-extended" : "zero-extended") + " to " +
                                       kind(type_signed(c.cls), c.bits));
        if (small(self) && small(o)) {
            c.fast = true;
            c.a = both ? val_i(self) : (i128)self->v;
            c.b = both ? val_i(o) : (i128)o->v;
        } else {
            c.pa = both ? val_py(self) : raw_py(self);
            c.pb = both ? val_py(o) : raw_py(o);
        }
        return true;
    }
    if (PyLong_Check(other)) {
        // A literal lives in a register of the operand's type: cast it.
        c.bits = self->bits;
        c.cls = Py_TYPE(self);
        int overflow = 0;
        long long x = PyLong_AsLongLongAndOverflow(other, &overflow);
        if (x == -1 && PyErr_Occurred()) raise_current();
        if (small(self) && !overflow) {
            PyUInt tmp;
            tmp.bits = self->bits;
            tmp.sgn = self->sgn;
            i128 v = x;
            if (self->sat) v = std::min(std::max(v, min_i(self->sgn, self->bits)), max_i(self->sgn, self->bits));
            tmp.v = (uint64_t)(u128)v & mask64(self->bits);
            i128 cast = val_i(&tmp);
            if (x < min_i(self->sgn, self->bits) || x > max_i(self->sgn, self->bits)) {
                nb::object s = steal_checked(PyObject_Str(other));
                warn(S.IntCastWarning, "implicit cast of " + py_str(s.ptr()) + " to " + kind(self->sgn, self->bits) +
                                           " is lossy: " + py_str(pyint(cast).ptr()));
            }
            c.fast = true;
            c.a = val_i(self);
            c.b = cast;
            return true;
        }
        nb::object cast = make_any(Py_TYPE(self), other, self->bits, self->sat, self->sat);
        nb::object cv = val_py(U(cast.ptr()));
        nb::object lo = min_py(self->sgn, self->bits), hi = max_py(self->sgn, self->bits);
        if (py_lt(other, lo.ptr()) || py_lt(hi.ptr(), other))
            warn(S.IntCastWarning, "implicit cast of " + py_str(other) + " to " + kind(self->sgn, self->bits) +
                                       " is lossy: " + py_str(cv.ptr()));
        c.pa = val_py(self);
        c.pb = cv;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------- arithmetic

static i128 trunc_div(i128 a, i128 b) { return a / b; }   // C++ truncates toward zero
static i128 trunc_mod(i128 a, i128 b) { return a % b; }

[[noreturn]] static void zero_div() { raise(PyExc_ZeroDivisionError, "division by zero"); }

static nb::object py_trunc_div(PyObject* a, PyObject* b) {
    int z = PyObject_Not(b);
    if (z) zero_div();
    nb::object aa = steal_checked(PyNumber_Absolute(a)), ab = steal_checked(PyNumber_Absolute(b));
    nb::object q = steal_checked(PyNumber_FloorDivide(aa.ptr(), ab.ptr()));
    nb::object zero = pyint64(0);
    bool na = py_lt(a, zero.ptr()), nb_ = py_lt(b, zero.ptr());
    return na == nb_ ? q : steal_checked(PyNumber_Negative(q.ptr()));
}

static PyObject* binop(PyObject* x, PyObject* y, int op, bool arith) {
    VF_TRY
    bool reverse = !is_uint(x);
    PyUInt* self = U(reverse ? y : x);
    PyObject* other = reverse ? x : y;
    ICo c;
    if (!coerce(self, other, c)) Py_RETURN_NOTIMPLEMENTED;
    bool clamp = self->sat && arith;
    if (c.fast && c.bits <= 64) {
        i128 a = reverse ? c.b : c.a, b = reverse ? c.a : c.b, r;
        bool ok = true;
        switch (op) {
            case '+': r = a + b; break;
            case '-': r = a - b; break;
            case '*': {
                int64_t la = a < 0 ? bitlen((u128)-a) : bitlen((u128)a);
                int64_t lb = b < 0 ? bitlen((u128)-b) : bitlen((u128)b);
                ok = la + lb <= 126;
                r = ok ? a * b : 0;
                break;
            }
            case '/': if (b == 0) zero_div(); r = trunc_div(a, b); break;
            case '%': if (b == 0) zero_div(); r = trunc_mod(a, b); break;
            case '&': r = a & b; break;
            case '|': r = a | b; break;
            default: r = a ^ b; break;
        }
        if (ok) return make_small(c.cls, r, c.bits, clamp, self->sat);
        c.pa = pyint(reverse ? c.b : c.a);
        c.pb = pyint(reverse ? c.a : c.b);
        reverse = false;
    } else if (c.fast) {
        c.pa = pyint(c.a);
        c.pb = pyint(c.b);
    }
    PyObject* a = reverse ? c.pb.ptr() : c.pa.ptr();
    PyObject* b = reverse ? c.pa.ptr() : c.pb.ptr();
    nb::object r;
    if (op == '/') r = py_trunc_div(a, b);
    else if (op == '%') {
        nb::object q = py_trunc_div(a, b);
        nb::object qb = py_op('*', q.ptr(), b);
        r = py_op('-', a, qb.ptr());
    } else r = py_op(op, a, b);
    return make_any(c.cls, r.ptr(), c.bits, clamp, self->sat).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* u_add(PyObject* a, PyObject* b) { return binop(a, b, '+', true); }
static PyObject* u_sub(PyObject* a, PyObject* b) { return binop(a, b, '-', true); }
static PyObject* u_mul(PyObject* a, PyObject* b) { return binop(a, b, '*', true); }
static PyObject* u_floordiv(PyObject* a, PyObject* b) { return binop(a, b, '/', true); }
static PyObject* u_mod(PyObject* a, PyObject* b) { return binop(a, b, '%', true); }
static PyObject* u_and(PyObject* a, PyObject* b) { return binop(a, b, '&', false); }
static PyObject* u_or(PyObject* a, PyObject* b) { return binop(a, b, '|', false); }
static PyObject* u_xor(PyObject* a, PyObject* b) { return binop(a, b, '^', false); }

static PyObject* u_neg(PyObject* s) {
    VF_TRY
    PyUInt* x = U(s);
    if (small(x)) return make_small(Py_TYPE(s), -val_i(x), x->bits, x->sat, x->sat);
    nb::object v = val_py(x);
    nb::object n = steal_checked(PyNumber_Negative(v.ptr()));
    return make_any(Py_TYPE(s), n.ptr(), x->bits, x->sat, x->sat).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* u_invert(PyObject* s) {
    VF_TRY
    PyUInt* x = U(s);
    if (small(x)) return make_small(Py_TYPE(s), (i128)(~x->v & mask64(x->bits)), x->bits, false, x->sat);
    nb::object n = steal_checked(PyNumber_Invert(x->big));
    return make_any(Py_TYPE(s), n.ptr(), x->bits, false, x->sat).release().ptr();
    VF_CATCH(nullptr)
}

static int64_t shift_count(PyObject* n) {
    nb::object i = steal_checked(PyNumber_Long(n));
    int overflow = 0;
    long long k = PyLong_AsLongLongAndOverflow(i.ptr(), &overflow);
    if (k == -1 && PyErr_Occurred()) raise_current();
    if (overflow < 0 || (!overflow && k < 0)) raise(PyExc_ValueError, "negative shift count");
    return overflow ? INT64_MAX : k;
}

static PyObject* u_lshift(PyObject* s, PyObject* n) {
    VF_TRY
    if (!is_uint(s)) Py_RETURN_NOTIMPLEMENTED;
    PyUInt* x = U(s);
    int64_t k = shift_count(n);
    if (small(x)) {
        uint64_t r = k >= 64 ? 0 : x->v << k;
        return make_small(Py_TYPE(s), (i128)(r & mask64(x->bits)), x->bits, false, x->sat);
    }
    nb::object kk = pyint64(k);
    nb::object r = py_op('<', x->big, kk.ptr());
    return make_any(Py_TYPE(s), r.ptr(), x->bits, false, x->sat).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* u_rshift(PyObject* s, PyObject* n) {
    VF_TRY
    if (!is_uint(s)) Py_RETURN_NOTIMPLEMENTED;
    PyUInt* x = U(s);
    int64_t k = shift_count(n);
    if (small(x)) {
        i128 v = val_i(x);
        i128 r = k >= 127 ? (v < 0 ? -1 : 0) : v >> k;
        return make_small(Py_TYPE(s), r, x->bits, false, x->sat);
    }
    nb::object v = val_py(x), kk = pyint64(k);
    nb::object r = py_op('>', v.ptr(), kk.ptr());
    return make_any(Py_TYPE(s), r.ptr(), x->bits, false, x->sat).release().ptr();
    VF_CATCH(nullptr)
}

static int u_bool(PyObject* s) {
    PyUInt* x = U(s);
    if (small(x)) return x->v != 0;
    return PyObject_IsTrue(x->big);
}

static PyObject* u_index(PyObject* s) {
    VF_TRY
    return val_py(U(s)).release().ptr();
    VF_CATCH(nullptr)
}

// ---------------------------------------------------------------- comparison

// fn(a, b) on coerced operands: -1/0/1 for <, ==; -2 NotImplemented.
static int cmp3(PyObject* s, PyObject* other) {
    PyUInt* self = U(s);
    ICo c;
    if (!coerce(self, other, c)) return -2;
    if (c.fast) return c.a < c.b ? -1 : c.a > c.b ? 1 : 0;
    if (py_lt(c.pa.ptr(), c.pb.ptr())) return -1;
    if (py_lt(c.pb.ptr(), c.pa.ptr())) return 1;
    return 0;
}

static PyObject* u_richcompare(PyObject* s, PyObject* other, int op) {
    VF_TRY
    // __eq__/__lt__ are defined; functools.total_ordering derives the rest.
    if (op == Py_EQ || op == Py_NE) {
        int k = cmp3(s, other);
        if (k == -2) Py_RETURN_NOTIMPLEMENTED;
        return PyBool_FromLong(op == Py_EQ ? k == 0 : k != 0);
    }
    int k = cmp3(s, other);
    if (k == -2) Py_RETURN_NOTIMPLEMENTED;
    bool lt = k == -1;
    switch (op) {
        case Py_LT: return PyBool_FromLong(lt);
        case Py_GE: return PyBool_FromLong(!lt);
        case Py_LE:   // op_result or self == other
            if (lt) Py_RETURN_TRUE;
            return PyObject_RichCompare(s, other, Py_EQ);
        default:      // GT: not op_result and self != other
            if (lt) Py_RETURN_FALSE;
            return PyObject_RichCompare(s, other, Py_NE);
    }
    VF_CATCH(nullptr)
}

static Py_hash_t u_hash(PyObject* s) {
    VF_TRY
    nb::object v = val_py(U(s));
    return PyObject_Hash(v.ptr());
    VF_CATCH(-1)
}

// ---------------------------------------------------------------- construction

static PyObject* u_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    VF_TRY
    static const char* kwlist[] = {"val", "bits", "saturate", nullptr};
    PyObject *val = nullptr, *bits = nullptr, *saturate = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOO", (char**)kwlist, &val, &bits, &saturate)) return nullptr;
    bool sgn = type_signed(type);
    int64_t nbits = 32;
    if (bits) {
        if (!PyLong_Check(bits) || PyBool_Check(bits)) raise(PyExc_ValueError, "bits must be a positive integer");
        int overflow = 0;
        long long b = PyLong_AsLongLongAndOverflow(bits, &overflow);
        if (overflow > 0) raise(PyExc_OverflowError, "bits is too large");
        nbits = b;
    }
    if (nbits < 0 || (nbits == 0 && sgn)) raise(PyExc_ValueError, "bits must be a positive integer");
    if (nbits > kMaxIntBits) raise(PyExc_OverflowError, "bits above 2**24 is not supported");
    bool sat = false;
    if (saturate) {
        int t = PyObject_IsTrue(saturate);
        if (t < 0) raise_current();
        sat = t;
    }
    nb::object v = val ? steal_checked(PyNumber_Long(val)) : pyint64(0);
    return make_any(type, v.ptr(), nbits, sat, sat).release().ptr();
    VF_CATCH(nullptr)
}

static void u_dealloc(PyObject* s) {
    Py_XDECREF(U(s)->big);
    PyTypeObject* tp = Py_TYPE(s);
    tp->tp_free(s);
    Py_DECREF(tp);
}

// ---------------------------------------------------------------- methods

static PyObject* g_bits(PyObject* s, void*) { return PyLong_FromLongLong(U(s)->bits); }
static PyObject* g_val(PyObject* s, void*) { return u_index(s); }
static PyObject* g_raw(PyObject* s, void*) {
    VF_TRY
    return raw_py(U(s)).release().ptr();
    VF_CATCH(nullptr)
}
static PyObject* g_sat(PyObject* s, void*) { return PyBool_FromLong(U(s)->sat); }
static PyObject* g_mask(PyObject* s, void*) {
    VF_TRY
    return mask_py(U(s)->bits).release().ptr();
    VF_CATCH(nullptr)
}
static PyObject* g_min(PyObject* s, void*) {
    VF_TRY
    return min_py(U(s)->sgn, U(s)->bits).release().ptr();
    VF_CATCH(nullptr)
}
static PyObject* g_max(PyObject* s, void*) {
    VF_TRY
    return max_py(U(s)->sgn, U(s)->bits).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* m_resize(PyObject* s, PyObject* const* args, Py_ssize_t nargs, PyObject* kw) {
    VF_TRY
    PyObject* a[1];
    parse_args("resize", args, nargs, kw, {"bits"}, a, 1);
    PyUInt* x = U(s);
    nb::object v = val_py(x);
    return PyObject_CallFunction((PyObject*)Py_TYPE(s), "OOO", v.ptr(), a[0], x->sat ? Py_True : Py_False);
    VF_CATCH(nullptr)
}

static PyObject* u_repr(PyObject* s) {
    VF_TRY
    PyUInt* x = U(s);
    nb::object v = val_py(x);
    nb::object name = steal_checked(PyObject_GetAttrString((PyObject*)Py_TYPE(s), "__name__"));
    std::string r = py_str(name.ptr()) + "(" + py_str(v.ptr()) + ", " + kind(x->sgn, x->bits) +
                    (x->sat ? ", sat" : "") + ")";
    return PyUnicode_FromStringAndSize(r.data(), (Py_ssize_t)r.size());
    VF_CATCH(nullptr)
}

static PyObject* fmt_raw(PyUInt* x, const std::string& spec) {
    nb::object raw = raw_py(x);
    nb::object sp = steal_checked(PyUnicode_FromString(spec.c_str()));
    return PyObject_Format(raw.ptr(), sp.ptr());
}
static PyObject* m_to_bin(PyObject* s, PyObject*) {
    VF_TRY
    PyUInt* x = U(s);
    if (!x->bits) return PyUnicode_FromString("");
    return fmt_raw(x, "0" + std::to_string(x->bits) + "b");
    VF_CATCH(nullptr)
}
static PyObject* m_to_hex(PyObject* s, PyObject*) {
    VF_TRY
    PyUInt* x = U(s);
    return fmt_raw(x, "0" + std::to_string((x->bits + 3) / 4) + "x");
    VF_CATCH(nullptr)
}

static PyObject* m_reduce(PyObject* s, PyObject*) {
    VF_TRY
    PyUInt* x = U(s);
    nb::object v = val_py(x);
    return Py_BuildValue("(O(OLO))", (PyObject*)Py_TYPE(s), v.ptr(), (long long)x->bits, x->sat ? Py_True : Py_False);
    VF_CATCH(nullptr)
}

static PyMethodDef u_methods[] = {
    {"resize", (PyCFunction)(void (*)(void))m_resize, METH_FASTCALL | METH_KEYWORDS,
     "resize($self, /, bits)\n--\n\nChange width: truncate (or clamp, in saturate mode) or extend."},
    {"to_bin", m_to_bin, METH_NOARGS, "to_bin($self, /)\n--\n\n"},
    {"to_hex", m_to_hex, METH_NOARGS, "to_hex($self, /)\n--\n\n"},
    {"__reduce__", m_reduce, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static PyGetSetDef u_getset[] = {
    {"bits", g_bits, nullptr, nullptr, nullptr},
    {"size", g_bits, nullptr, "Storage width in bits.", nullptr},
    {"val", g_val, nullptr, nullptr, nullptr},
    {"raw", g_raw, nullptr, nullptr, nullptr},
    {"saturate", g_sat, nullptr, nullptr, nullptr},
    {"mask", g_mask, nullptr, nullptr, nullptr},
    {"min", g_min, nullptr, nullptr, nullptr},
    {"max", g_max, nullptr, nullptr, nullptr},
    {"_bits", g_bits, nullptr, nullptr, nullptr},
    {"_val", g_raw, nullptr, nullptr, nullptr},
    {"_sat", g_sat, nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

void register_uint(PyObject* module) {
    static PyType_Slot slots[] = {
        {Py_tp_new, (void*)u_new},
        {Py_tp_dealloc, (void*)u_dealloc},
        {Py_tp_repr, (void*)u_repr},
        {Py_tp_hash, (void*)u_hash},
        {Py_tp_richcompare, (void*)u_richcompare},
        {Py_tp_methods, u_methods},
        {Py_tp_getset, u_getset},
        {Py_nb_add, (void*)u_add},
        {Py_nb_subtract, (void*)u_sub},
        {Py_nb_multiply, (void*)u_mul},
        {Py_nb_floor_divide, (void*)u_floordiv},
        {Py_nb_remainder, (void*)u_mod},
        {Py_nb_negative, (void*)u_neg},
        {Py_nb_and, (void*)u_and},
        {Py_nb_or, (void*)u_or},
        {Py_nb_xor, (void*)u_xor},
        {Py_nb_invert, (void*)u_invert},
        {Py_nb_lshift, (void*)u_lshift},
        {Py_nb_rshift, (void*)u_rshift},
        {Py_nb_bool, (void*)u_bool},
        {Py_nb_index, (void*)u_index},
        {Py_nb_int, (void*)u_index},
        {Py_tp_doc, (void*)"UINT(val=0, bits=32, saturate=False)\n--\n\n"
                           "Fixed-width unsigned integer that wraps (or saturates)."},
        {0, nullptr}};
    static PyType_Spec spec = {"verifloat.uint.UINT", sizeof(PyUInt), 0, Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, slots};
    S.UINT = (PyTypeObject*)check(PyType_FromSpec(&spec));
    S.UINT->tp_name = "UINT";   // bare, as for a Python class

    static PyType_Slot islots[] = {{Py_tp_doc, (void*)"INT(val=0, bits=32, saturate=False)\n--\n\n"
                                                      "Fixed-width two's-complement signed integer."},
                                   {0, nullptr}};
    static PyType_Spec ispec = {"verifloat.sint.INT", sizeof(PyUInt), 0, Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, islots};
    S.INT = (PyTypeObject*)check(PyType_FromSpecWithBases(&ispec, (PyObject*)S.UINT));
    S.INT->tp_name = "INT";

    if (PyObject_SetAttrString((PyObject*)S.UINT, "signed", Py_False) < 0 ||
        PyObject_SetAttrString((PyObject*)S.INT, "signed", Py_True) < 0 ||
        PyModule_AddObjectRef(module, "UINT", (PyObject*)S.UINT) < 0 ||
        PyModule_AddObjectRef(module, "INT", (PyObject*)S.INT) < 0)
        raise_current();
}

}  // namespace vf
