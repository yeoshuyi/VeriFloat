// Python glue: interpreter state the core needs, error helpers, and
// conversions between Python ints/Fractions and native integers.
#pragma once

#include <nanobind/nanobind.h>   // first: Python.h sets feature macros

#include <Python.h>

#include "common.hpp"

#include <string>
#include <string_view>

namespace nb = nanobind;

namespace vf {

// Objects handed over by the Python package at import time (see _init).
struct PyState {
    PyObject* Fraction = nullptr;          // fractions.Fraction
    PyObject* from_coprime = nullptr;      // Fraction._from_coprime_ints (may be null)
    PyObject* flags[32] = {};              // FPFlags(i)
    PyObject* rounding_enum[6] = {};       // Rounding members by index
    PyObject* nanmode_enum[4] = {};
    PyObject* warn = nullptr;              // _warnings.warn(category, msg)
    PyObject* IntCastWarning = nullptr;
    PyObject* report = nullptr;            // fp._report(event, unrounded, fmt, result)
    PyObject* warn_mismatch = nullptr;     // fp._warn_mismatch(fa, fb, fmt)
    PyObject* common_of = nullptr;         // fp._common_of_py(fa, fb) -> FPFormat
    PyObject* cast_warn = nullptr;         // fp._cast_warn(other, fmt, b, flags)
    PyObject* sr_state = nullptr;          // dict {"source": callable}
    PyObject* str_source = nullptr;        // "source"
    PyObject* inf = nullptr;               // float('inf')
    PyObject* ninf = nullptr;
    PyObject* FPFormat = nullptr;          // Python FPFormat class
    PyTypeObject* FP = nullptr;
    PyTypeObject* UINT = nullptr;
    PyTypeObject* INT = nullptr;
    PyTypeObject* FormatBase = nullptr;
    PyObject* event_names[8] = {};
};
extern PyState S;

// Raise a Python exception from C++ (propagates as nb::python_error).
[[noreturn]] inline void raise(PyObject* type, const std::string& msg) {
    PyErr_SetString(type, msg.c_str());
    throw nb::python_error();
}
[[noreturn]] inline void raise_current() { throw nb::python_error(); }
inline PyObject* check(PyObject* o) {
    if (!o) raise_current();
    return o;
}
inline nb::object steal_checked(PyObject* o) { return nb::steal(check(o)); }

// f(arg), as PyObject_CallOneArg does it. That function is not in nanobind's
// list of the CPython symbols a macOS module may leave for the interpreter to
// supply (darwin-ld-cpython.sym), so the module would not link there.
inline PyObject* call_one(PyObject* f, PyObject* arg) {
    PyObject* args[2] = {nullptr, arg};
    return PyObject_Vectorcall(f, args + 1, 1 | PY_VECTORCALL_ARGUMENTS_OFFSET, nullptr);
}

// Bodies of CPython slot functions: translate C++ exceptions to Python.
#define VF_TRY try {
#define VF_CATCH(ret)                                                        \
    }                                                                        \
    catch (nb::python_error & e) { e.restore(); return ret; }                \
    catch (std::bad_alloc &) { PyErr_NoMemory(); return ret; }               \
    catch (std::exception & e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return ret; }

// ---- input limits
// Arrays and tensors have at most kMaxDims axes, like NumPy's 64, and at most
// kMaxElems elements, which keeps every element count, stride and byte count
// far from overflowing whatever axes they are made of.
constexpr size_t kMaxDims = 64;
constexpr size_t kMaxElems = size_t(1) << 48;

// a * b, or an error if that exceeds kMaxElems.
inline size_t elems_mul(size_t a, size_t b) {
    if (b != 0 && a > kMaxElems / b) raise(PyExc_ValueError, "array too large (more than 2**48 elements)");
    return a * b;
}
// The element count of a shape: every axis at least 1, at most kMaxDims of
// them, at most kMaxElems in all.
inline size_t checked_count(const std::vector<int64_t>& shape) {
    if (shape.size() > kMaxDims) raise(PyExc_ValueError, "too many dimensions (at most 64)");
    size_t n = 1;
    for (int64_t d : shape) {
        if (d < 1) raise(PyExc_ValueError, "every dimension must be at least 1");
        n = elems_mul(n, (size_t)d);
    }
    return n;
}

// The items of a sequence (or any iterable) as a tuple of strong references.
// Code that walks the items while calling back into Python uses this rather
// than the list's own item array, which that Python code could change or free.
inline nb::object snapshot(PyObject* o, const char* msg) {
    PyObject* t = PySequence_Tuple(o);
    if (!t) {
        if (PyErr_ExceptionMatches(PyExc_TypeError)) {
            PyErr_Clear();
            raise(PyExc_TypeError, msg);
        }
        raise_current();
    }
    return nb::steal(t);
}
inline Py_ssize_t tuple_size(const nb::object& t) { return PyTuple_GET_SIZE(t.ptr()); }
inline PyObject* tuple_item(const nb::object& t, Py_ssize_t i) { return PyTuple_GET_ITEM(t.ptr(), i); }

// A Python int as an int64 in [lo, hi], else `what` in a ValueError.
inline int64_t int_in(PyObject* o, int64_t lo, int64_t hi, const char* what) {
    nb::object i = steal_checked(PyNumber_Index(o));
    int overflow = 0;
    const long long v = PyLong_AsLongLongAndOverflow(i.ptr(), &overflow);
    if (v == -1 && PyErr_Occurred()) raise_current();
    if (overflow || v < lo || v > hi) raise(PyExc_ValueError, what);
    return v;
}

inline bool is_fraction(PyObject* o) {
    return Py_TYPE(o) == (PyTypeObject*)S.Fraction || PyObject_TypeCheck(o, (PyTypeObject*)S.Fraction);
}

// Python int <-> native
nb::object long_from_u128(u128 v);
nb::object long_from_big(const BigInt& v);
nb::object long_from_i128(i128 v);
BigInt big_from_long(PyObject* o);         // any int (sign kept)
inline int64_t long_bitlen(PyObject* o) {   // |o|.bit_length()
    nb::object r = steal_checked(PyObject_CallMethod(o, "bit_length", nullptr));
    return PyLong_AsLongLong(r.ptr());
}

// Exact Fraction from a dyadic (-1)**neg * sig * 2**exp.
template <class U> nb::object fraction_dy(bool neg, const U& sig, int64_t exp);
nb::object fraction_int(nb::object n);
nb::object fraction_nd(nb::object n, nb::object d);   // normalizing constructor

std::string py_str(PyObject* o);
std::string py_repr(PyObject* o);

// Warnings (attributed to the first caller outside the package by Python).
void warn(PyObject* category, const std::string& msg);

}  // namespace vf
