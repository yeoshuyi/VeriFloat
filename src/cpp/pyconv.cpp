// Conversions between Python objects and native integers/dyadics.
#include "py.hpp"

#include "kernel.hpp"

#include <string>

namespace vf {

nb::object long_from_u128(u128 v) {
    uint64_t hi = (uint64_t)(v >> 64), lo = (uint64_t)v;
    if (!hi) return steal_checked(PyLong_FromUnsignedLongLong(lo));
    nb::object h = steal_checked(PyLong_FromUnsignedLongLong(hi));
    nb::object s = steal_checked(PyLong_FromLong(64));
    nb::object r = steal_checked(PyNumber_Lshift(h.ptr(), s.ptr()));
    nb::object l = steal_checked(PyLong_FromUnsignedLongLong(lo));
    return steal_checked(PyNumber_Or(r.ptr(), l.ptr()));
}

nb::object long_from_i128(i128 v) {
    if (v >= 0) return long_from_u128((u128)v);
    nb::object m = long_from_u128((u128)(-(v + 1)) + 1);
    return steal_checked(PyNumber_Negative(m.ptr()));
}

nb::object long_from_big(const BigInt& v) {
    bool neg = v < 0;
    BigInt a = neg ? BigInt(-v) : v;
    nb::object r;
    if (bitlen(a) <= 128) r = long_from_u128(to_u128(a));
    else {
        std::string hex = a.str(0, std::ios_base::hex);
        r = steal_checked(PyLong_FromString(hex.c_str(), nullptr, 16));
    }
    return neg ? steal_checked(PyNumber_Negative(r.ptr())) : r;
}

BigInt big_from_long(PyObject* o) {
    int overflow = 0;
    long long x = PyLong_AsLongLongAndOverflow(o, &overflow);
    if (x == -1 && PyErr_Occurred()) raise_current();
    if (!overflow) return BigInt(x);
    nb::object a = steal_checked(PyNumber_Absolute(o));
    nb::object hex = steal_checked(PyNumber_ToBase(a.ptr(), 16));   // "0x..."
    BigInt r(PyUnicode_AsUTF8(hex.ptr()));
    return overflow < 0 ? BigInt(-r) : r;
}

static nb::object make_fraction(nb::object n, nb::object d) {
    if (S.from_coprime) return steal_checked(PyObject_CallFunctionObjArgs(S.from_coprime, n.ptr(), d.ptr(), nullptr));
    return steal_checked(PyObject_CallFunctionObjArgs(S.Fraction, n.ptr(), d.ptr(), nullptr));
}

nb::object fraction_int(nb::object n) { return make_fraction(n, nb::int_(1)); }

nb::object fraction_nd(nb::object n, nb::object d) {
    return steal_checked(PyObject_CallFunctionObjArgs(S.Fraction, n.ptr(), d.ptr(), nullptr));
}

static int64_t ctz(u128 x) {
    uint64_t lo = (uint64_t)x;
    return lo ? (int)std::countr_zero((uint64_t)lo) : 64 + (int)std::countr_zero((uint64_t)(x >> 64));
}
static int64_t ctz(const BigInt& x) { return (int64_t)boost::multiprecision::lsb(x); }

static nb::object signed_long(bool neg, const u128& v) {
    nb::object r = long_from_u128(v);
    return neg ? steal_checked(PyNumber_Negative(r.ptr())) : r;
}
static nb::object signed_long(bool neg, const BigInt& v) { return long_from_big(neg ? BigInt(-v) : v); }

template <class U> nb::object fraction_dy(bool neg, const U& sig, int64_t exp) {
    if (sig == 0) return make_fraction(nb::int_(0), nb::int_(1));
    if (exp >= 0) {
        BigInt n = to_big(sig) << (unsigned)exp;
        return make_fraction(long_from_big(neg ? BigInt(-n) : n), nb::int_(1));
    }
    int64_t tz = std::min<int64_t>(ctz(sig), -exp);
    U s = sig >> (unsigned)tz;
    exp += tz;
    nb::object n = signed_long(neg, s);
    if (exp == 0) return make_fraction(n, nb::int_(1));
    nb::object one = nb::int_(1), k = steal_checked(PyLong_FromLongLong(-exp));
    nb::object d = steal_checked(PyNumber_Lshift(one.ptr(), k.ptr()));
    return make_fraction(n, d);
}
template nb::object fraction_dy<u128>(bool, const u128&, int64_t);
template nb::object fraction_dy<BigInt>(bool, const BigInt&, int64_t);

std::string py_str(PyObject* o) {
    nb::object s = steal_checked(PyObject_Str(o));
    const char* c = PyUnicode_AsUTF8(s.ptr());
    if (!c) raise_current();
    return c;
}

std::string py_repr(PyObject* o) {
    nb::object s = steal_checked(PyObject_Repr(o));
    const char* c = PyUnicode_AsUTF8(s.ptr());
    if (!c) raise_current();
    return c;
}

void warn(PyObject* category, const std::string& msg) {
    nb::object m = steal_checked(PyUnicode_FromStringAndSize(msg.data(), (Py_ssize_t)msg.size()));
    steal_checked(PyObject_CallFunctionObjArgs(S.warn, category, m.ptr(), nullptr));
}

}  // namespace vf

// ---------------------------------------------------------------- the core's host

namespace vf {

// An error from the core, as the Python exception the reference raises.
void fail(Err kind, const std::string& msg) {
    PyObject* type = PyExc_RuntimeError;
    switch (kind) {
        case Err::VALUE: type = PyExc_ValueError; break;
        case Err::ZERO_DIVISION: type = PyExc_ZeroDivisionError; break;
        case Err::OVERFLOW: type = PyExc_OverflowError; break;
        case Err::TYPE: type = PyExc_TypeError; break;
        case Err::RUNTIME: break;
    }
    raise(type, msg);
}

// `sr = sr or 0`
static void sr_set(SRArg& sr, PyObject* v) {
    int truth = PyObject_IsTrue(v);
    if (truth < 0) raise_current();
    if (!truth) { sr.set((int64_t)0); return; }
    int overflow = 0;
    long long x = PyLong_AsLongLongAndOverflow(v, &overflow);
    if (x == -1 && PyErr_Occurred()) raise_current();
    if (!overflow) sr.set((int64_t)x);
    else sr.set(big_from_long(v));
}

// Random bits for stochastic rounding: the explicit sr_rand argument, or
// drawn from the Python source (set_sr_source).
void sr_draw(SRArg& sr, const Fmt& f) {
    PyObject* given = (PyObject*)sr.given;
    if (given && given != Py_None) { sr_set(sr, given); return; }
    PyObject* src = PyDict_GetItemWithError(S.sr_state, S.str_source);
    if (!src) { if (PyErr_Occurred()) raise_current(); raise(PyExc_RuntimeError, "no SR source"); }
    nb::object hold = steal_checked(PyObject_CallFunction(src, "L", (long long)f.sr_bits));
    sr_set(sr, hold.ptr());
}

}  // namespace vf
