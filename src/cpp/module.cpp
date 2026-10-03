// Module definition: types, and the Python objects the core calls back into.
#include "fp.hpp"
#include "uint.hpp"

namespace vf {
void register_fp(nb::module_& m);
void register_accum(nb::module_& m);
void register_fixed(nb::module_& m);
void register_block(nb::module_& m);
void register_array(nb::module_& m);
void register_selfcheck(nb::module_& m);
}  // namespace vf

using namespace vf;

static PyObject* keep(nb::handle h) { return Py_NewRef(h.ptr()); }

NB_MODULE(_core, m) {
    register_uint(m.ptr());
    register_fp(m);
    register_accum(m);
    register_fixed(m);
    register_block(m);
    register_array(m);
    register_selfcheck(m);

    // Objects from the Python package, handed over once at import time.
    m.def("_init", [](nb::handle fraction, nb::list flags, nb::list roundings, nb::list nan_modes,
                      nb::handle warn_fn, nb::handle int_cast_warning, nb::handle report,
                      nb::handle warn_mismatch, nb::handle common_of, nb::handle cast_warn,
                      nb::handle sr_state, nb::handle fpformat) {
        S.Fraction = keep(fraction);
        S.from_coprime = PyObject_GetAttrString(fraction.ptr(), "_from_coprime_ints");
        if (!S.from_coprime) PyErr_Clear();
        for (size_t i = 0; i < 32; ++i) S.flags[i] = keep(flags[i]);
        for (size_t i = 0; i < 6; ++i) S.rounding_enum[i] = keep(roundings[i]);
        for (size_t i = 0; i < 4; ++i) S.nanmode_enum[i] = keep(nan_modes[i]);
        S.warn = keep(warn_fn);
        S.IntCastWarning = keep(int_cast_warning);
        S.report = keep(report);
        S.warn_mismatch = keep(warn_mismatch);
        S.common_of = keep(common_of);
        S.cast_warn = keep(cast_warn);
        S.sr_state = keep(sr_state);
        S.str_source = PyUnicode_InternFromString("source");
        S.inf = PyFloat_FromDouble(INFINITY);
        S.ninf = PyFloat_FromDouble(-INFINITY);
        S.FPFormat = keep(fpformat);
        const char* events[8] = {nullptr, "saturated", "overflowed", "wrapped-up", "wrapped-down",
                                 "clamped", "underflowed", "flushed"};
        S.event_names[0] = Py_NewRef(Py_None);
        for (int i = 1; i < 8; ++i) S.event_names[i] = PyUnicode_InternFromString(events[i]);
    });
}
