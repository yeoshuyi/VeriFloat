// The FP Python type and the operations other translation units reuse.
#pragma once

#include "py.hpp"

#include "ops.hpp"

#include <vector>

namespace vf {

// Python object behind FPFormat (the dataclass subclasses it).
struct PyFormat {
    PyObject_HEAD
    Fmt f;
};
inline bool is_format(PyObject* o) { return PyObject_TypeCheck(o, S.FormatBase); }
inline const Fmt& fmt_of(PyObject* o) { return ((PyFormat*)o)->f; }

// The infinitely precise result before rounding, built lazily: the exact
// dyadic terms of the operation (UnrD), turned into a Fraction on access.
struct Unr : UnrD {
    nb::object obj;   // OBJ: a number Fraction() accepts; also the cache
    Unr() = default;
    Unr(UnrD&& d) : UnrD(std::move(d)) {}
};

struct PyFP {
    PyObject_HEAD
    PyObject* fmt;    // strong ref to the Python FPFormat
    const Fmt* f;
    Code c;
    uint8_t flags;
    Unr u;
};
inline bool is_fp(PyObject* o) { return PyObject_TypeCheck(o, S.FP); }

inline Opnd opnd_of(PyFP* x) { return {x->c, x->f, x->fmt}; }

// Allocate an FP in Python format `fmt` (no warnings, no unrounded).
PyFP* fp_new(PyObject* fmt, const Code& c, uint8_t flags = 0);
PyFP* fp_new(PyObject* fmt, const Code& c, uint8_t flags, Unr&& u);
// FP._finish: set flags and unrounded; warn on the event if the format asks.
nb::object fp_finish(PyObject* fmt, const RoundOut& r, Unr&& u);
inline nb::object fp_finish(PyObject* fmt, OpOut&& o) { return fp_finish(fmt, o.r, Unr(std::move(o.u))); }

// Materialize the unrounded value (Fraction, +-inf float, or None).
nb::object unrounded_of(PyFP* x);
nb::object fraction_of_term(const Term& t);

// Encode a Python number (int, float, Fraction, ...) into `fmt`; this is
// FP.from_value for non-FP values.
nb::object fp_from_number(PyObject* value, PyObject* fmt, PyObject* sr_rand);
// Round an exact Python number into a format without finishing (FP._round).
RoundOut round_number(PyObject* value, const Fmt& f, bool zero_sign, SRArg& sr, nb::object* unr_obj);
nb::object fp_plain_obj(PyObject* fmt, const Code& c);

// The exact value of a finite encoding as a Fraction.
nb::object fp_exact(const Code& c, const Fmt& f);

}  // namespace vf
