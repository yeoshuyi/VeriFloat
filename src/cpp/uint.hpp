// UINT / INT: fixed-width two's-complement integers.
#pragma once

#include "py.hpp"

namespace vf {

// Values up to 64 bits live in `v` (raw bits, masked); wider ones in `big`
// (a non-negative Python int holding the raw bits).
struct PyUInt {
    PyObject_HEAD
    int64_t bits;
    bool sat;
    bool sgn;        // INT (or a subclass)
    uint64_t v;
    PyObject* big;   // owned, or nullptr when bits <= 64
};

inline bool is_uint(PyObject* o) { return PyObject_TypeCheck(o, S.UINT); }
inline bool type_signed(PyTypeObject* t) { return PyType_IsSubtype(t, S.INT); }

// New UINT/INT of type `t` from raw bits (already masked to `bits`).
PyObject* uint_from_raw64(PyTypeObject* t, uint64_t raw, int64_t bits, bool sat);
// type(cls)(val, bits) for an arbitrary Python int value (wraps).
nb::object uint_make(PyTypeObject* t, PyObject* val, int64_t bits, bool sat);
// The value of a UINT/INT as a Python int (signed for INT).
nb::object uint_val(PyObject* o);

void register_uint(PyObject* module);

}  // namespace vf
