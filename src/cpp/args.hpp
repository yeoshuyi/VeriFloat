// Minimal METH_FASTCALL | METH_KEYWORDS argument parsing.
#pragma once

#include "py.hpp"

#include <initializer_list>

namespace vf {

// Fill out[i] (borrowed) from positional args, then keywords by name.
// Entries not given keep their preset default; the first `nreq` must be given.
inline void parse_args(const char* fname, PyObject* const* args, Py_ssize_t nargs, PyObject* kwnames,
                       std::initializer_list<const char*> names, PyObject** out, size_t nreq) {
    size_t n = names.size();
    if ((size_t)nargs > n)
        raise(PyExc_TypeError, std::string(fname) + "() takes at most " + std::to_string(n) + " arguments");
    bool given[8] = {};
    for (Py_ssize_t i = 0; i < nargs; ++i) { out[i] = args[i]; given[i] = true; }
    if (kwnames) {
        Py_ssize_t nk = PyTuple_GET_SIZE(kwnames);
        for (Py_ssize_t j = 0; j < nk; ++j) {
            PyObject* key = PyTuple_GET_ITEM(kwnames, j);
            size_t i = 0;
            for (const char* nm : names) {
                if (PyUnicode_CompareWithASCIIString(key, nm) == 0) break;
                ++i;
            }
            if (i == n)
                raise(PyExc_TypeError, std::string(fname) + "() got an unexpected keyword argument '" +
                                           PyUnicode_AsUTF8(key) + "'");
            if (given[i])
                raise(PyExc_TypeError, std::string(fname) + "() got multiple values for argument '" +
                                           PyUnicode_AsUTF8(key) + "'");
            out[i] = args[nargs + j];
            given[i] = true;
        }
    }
    for (size_t i = 0; i < nreq; ++i)
        if (!given[i])
            raise(PyExc_TypeError, std::string(fname) + "() missing required argument '" + names.begin()[i] + "'");
}

}  // namespace vf
