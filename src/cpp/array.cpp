// FPArray: an N-dimensional array of FP values in one format, stored as raw
// codes. Every operation gives, element by element, exactly what the scalar
// FP operation gives (code, flags and warnings); the common cases run on the
// fast kernels (fast.hpp) and everything else goes through the scalar code.
#include "accum.hpp"
#include "args.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace vf {

nb::object arith(char op, const Opnd& a, const Opnd& b, PyObject* fmt);
nb::object convert_to(PyFP* self, PyObject* fmt, PyObject* sr_rand);

struct PyFPArray {
    PyObject_HEAD
    PyObject* fmt;   // strong ref to the Python FPFormat
    const Fmt* f;
    std::vector<uint64_t> raw;
    std::vector<uint8_t> flags;
    std::vector<int64_t> shape;
};

static PyTypeObject* g_array_type = nullptr;
static inline bool is_array(PyObject* o) { return PyObject_TypeCheck(o, g_array_type); }

static void check_array_format(PyObject* fmt) {
    if (!is_format(fmt)) raise(PyExc_TypeError, "fmt must be an FPFormat");
    if (fmt_of(fmt).size > 64)
        raise(PyExc_ValueError, "FPArray holds formats of at most 64 bits, not " + fmt_str(fmt_of(fmt)));
}

static PyFPArray* array_new(PyObject* fmt, std::vector<int64_t> shape) {
    check_array_format(fmt);
    const size_t n = checked_count(shape);   // before anything is allocated
    // The storage first: if it cannot be had, no half-built object exists.
    std::vector<uint64_t> raw(n);
    std::vector<uint8_t> flags(n);
    PyTypeObject* t = g_array_type;
    PyFPArray* o = (PyFPArray*)t->tp_alloc(t, 0);
    if (!o) raise_current();
    Py_INCREF(fmt);
    o->fmt = fmt;
    o->f = &fmt_of(fmt);
    new (&o->raw) std::vector<uint64_t>(std::move(raw));      // moves do not throw
    new (&o->flags) std::vector<uint8_t>(std::move(flags));
    new (&o->shape) std::vector<int64_t>(std::move(shape));
    return o;
}

static void array_dealloc(PyObject* s) {
    PyFPArray* o = (PyFPArray*)s;
    o->raw.~vector();
    o->flags.~vector();
    o->shape.~vector();
    Py_XDECREF(o->fmt);
    PyTypeObject* tp = Py_TYPE(s);
    tp->tp_free(s);
    Py_DECREF(tp);
}

static inline void set_elem(PyFPArray* a, size_t i, PyObject* fp) {
    PyFP* p = (PyFP*)fp;
    a->raw[i] = raw64_of(p->c, *p->f);
    a->flags[i] = p->flags;
}

static inline nb::object elem_fp(const PyFPArray* a, size_t i) {
    return nb::steal((PyObject*)fp_new(a->fmt, code_of_raw64(a->raw[i], *a->f), a->flags[i]));
}

// ---------------------------------------------------------------- nested lists

// Shape and leaves of nested lists/tuples. The leaves are held as strong
// references: converting one may run Python code that changes the lists. No
// Python code runs during the walk itself. The depth is limited, which also
// stops a list that contains itself.
static void walk(PyObject* o, size_t depth, std::vector<int64_t>& shape, std::vector<nb::object>& leaves,
                 bool& shape_known) {
    if (PyList_Check(o) || PyTuple_Check(o)) {
        if (depth >= kMaxDims) raise(PyExc_ValueError, "too many dimensions (at most 64), or a list that contains itself");
        Py_ssize_t n = PySequence_Fast_GET_SIZE(o);
        if (n == 0) raise(PyExc_ValueError, "empty tensor");
        if (depth == shape.size()) {
            if (shape_known) raise(PyExc_ValueError, "ragged nested lists: all rows must have the same shape");
            shape.push_back(n);
        } else if (shape[depth] != n)
            raise(PyExc_ValueError, "ragged nested lists: all rows must have the same shape");
        PyObject** items = PySequence_Fast_ITEMS(o);
        for (Py_ssize_t i = 0; i < n; ++i) walk(items[i], depth + 1, shape, leaves, shape_known);
        return;
    }
    if (depth != shape.size()) raise(PyExc_ValueError, "ragged nested lists: all rows must have the same shape");
    shape_known = true;
    leaves.push_back(nb::borrow(o));
}

static void flatten(PyObject* values, std::vector<int64_t>& shape, std::vector<nb::object>& leaves) {
    bool known = false;
    walk(values, 0, shape, leaves, known);
    if (shape.empty()) raise(PyExc_TypeError, "expected nested lists of values, not a scalar");
}

template <class Make> static nb::object nest(const std::vector<int64_t>& shape, size_t dim, size_t& pos, Make&& make) {
    nb::list out;
    for (int64_t i = 0; i < shape[dim]; ++i) {
        if (dim + 1 == shape.size()) out.append(make(pos++));
        else out.append(nest(shape, dim + 1, pos, make));
    }
    return out;
}

static nb::object shape_tuple(const std::vector<int64_t>& shape) {
    nb::object t = steal_checked(PyTuple_New((Py_ssize_t)shape.size()));
    for (size_t i = 0; i < shape.size(); ++i) PyTuple_SET_ITEM(t.ptr(), (Py_ssize_t)i, check(PyLong_FromLongLong(shape[i])));
    return t;
}

static std::string shape_str(const std::vector<int64_t>& shape) { return py_repr(shape_tuple(shape).ptr()); }

static size_t count_of(const std::vector<int64_t>& shape) { return checked_count(shape); }

// ---------------------------------------------------------------- broadcasting

// NumPy's rule: shapes are aligned at the last axis, and an axis of length 1
// repeats along the other operand's.
static std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    const size_t n = std::max(a.size(), b.size());
    std::vector<int64_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        const int64_t da = i < n - a.size() ? 1 : a[i - (n - a.size())];
        const int64_t db = i < n - b.size() ? 1 : b[i - (n - b.size())];
        if (da != db && da != 1 && db != 1)
            raise(PyExc_ValueError, "shape mismatch: " + shape_str(a) + " vs " + shape_str(b));
        out[i] = da == 1 ? db : da;
    }
    return out;
}

// For each element of an array of shape `out` (C order), the flat index of
// the element of shape `src` that broadcasts to it. `src` must broadcast to
// `out`.
static std::vector<size_t> broadcast_index(const std::vector<int64_t>& src, const std::vector<int64_t>& out) {
    const size_t nd = out.size(), lead = nd - src.size();
    std::vector<size_t> stride(nd, 0);
    size_t acc = 1;
    for (size_t i = src.size(); i-- > 0;) {
        stride[lead + i] = src[i] == out[lead + i] ? acc : 0;   // a repeated axis does not advance
        acc *= (size_t)src[i];
    }
    std::vector<size_t> map(count_of(out));
    std::vector<int64_t> idx(nd, 0);
    size_t cur = 0;
    for (size_t o = 0; o < map.size(); ++o) {
        map[o] = cur;
        for (size_t i = nd; i-- > 0;) {
            if (++idx[i] < out[i]) {
                cur += stride[i];
                break;
            }
            cur -= stride[i] * (size_t)(out[i] - 1);
            idx[i] = 0;
        }
    }
    return map;
}

// ---------------------------------------------------------------- numeric buffers

// A numeric buffer (a NumPy array, array.array, memoryview, ...) read element
// by element in C order, whatever its strides. Only native-endian numbers
// are read here; anything else is left to the caller's list path.
struct NumBuffer {
    Py_buffer view{};
    bool held = false;
    char kind = 0;   // 'f' float, 'i' signed int, 'u' unsigned int (bool included)
    std::vector<int64_t> shape;
    size_t n = 0;

    NumBuffer() = default;
    NumBuffer(const NumBuffer&) = delete;
    NumBuffer& operator=(const NumBuffer&) = delete;
    ~NumBuffer() {
        if (held) PyBuffer_Release(&view);
    }

    bool open(PyObject* o) {
        if (PyList_Check(o) || PyTuple_Check(o) || PyBytes_Check(o) || PyByteArray_Check(o) || PyUnicode_Check(o))
            return false;
        if (!PyObject_CheckBuffer(o)) return false;
        if (PyObject_GetBuffer(o, &view, PyBUF_RECORDS_RO) != 0) {
            PyErr_Clear();
            return false;
        }
        held = true;
        const char* f = view.format ? view.format : "B";
        // The elements are read in the host's byte order: a format that names
        // that order, or none, is taken here; the other order goes to the
        // list path.
        const bool little = std::endian::native == std::endian::little;
        if (*f == '@' || *f == '=' || *f == (little ? '<' : '>') || (!little && *f == '!')) ++f;
        const Py_ssize_t size = view.itemsize;
        bool ok = f[0] != 0 && f[1] == 0 && view.suboffsets == nullptr;
        switch (f[0]) {
            case 'd': ok = ok && size == 8; kind = 'f'; break;
            case 'f': ok = ok && size == 4; kind = 'f'; break;
            case 'e': ok = ok && size == 2; kind = 'f'; break;
            case 'b': case 'h': case 'i': case 'l': case 'q': case 'n':
                ok = ok && (size == 1 || size == 2 || size == 4 || size == 8); kind = 'i'; break;
            case '?': case 'B': case 'H': case 'I': case 'L': case 'Q': case 'N':
                ok = ok && (size == 1 || size == 2 || size == 4 || size == 8); kind = 'u'; break;
            default: ok = false;
        }
        if (!ok) {
            PyBuffer_Release(&view);
            held = false;
            return false;
        }
        if (view.ndim == 0) raise(PyExc_TypeError, "expected nested lists of values, not a scalar");
        if ((size_t)view.ndim > kMaxDims) raise(PyExc_ValueError, "too many dimensions (at most 64)");
        n = 1;
        for (int d = 0; d < view.ndim; ++d) {
            if (view.shape[d] == 0) raise(PyExc_ValueError, "empty tensor");
            shape.push_back(view.shape[d]);
            n = elems_mul(n, (size_t)view.shape[d]);   // a broadcast view can claim any size
        }
        return true;
    }

    // fn(pointer to element) for every element in C order
    template <class Fn> void for_each(Fn&& fn) const { walk_dim(0, (const char*)view.buf, fn); }
    template <class Fn> void walk_dim(int d, const char* p, Fn& fn) const {
        const Py_ssize_t len = view.shape[d], step = view.strides[d];
        if (d + 1 == view.ndim) {
            for (Py_ssize_t i = 0; i < len; ++i, p += step) fn(p);
        } else {
            for (Py_ssize_t i = 0; i < len; ++i, p += step) walk_dim(d + 1, p, fn);
        }
    }

    // The element as a double (floats: exact widening of binary16/32, done
    // on the bits: a conversion instruction would read a subnormal as zero
    // when the processor is set to denormals-are-zero).
    double as_double(const char* p) const {
        if (view.itemsize == 8) {
            double d;
            std::memcpy(&d, p, 8);
            return d;
        }
        const uint32_t b = (uint32_t)load(p);
        const int M = view.itemsize == 4 ? 23 : 10, E = view.itemsize == 4 ? 8 : 5;
        const uint64_t sign = (uint64_t)(b >> (E + M)) << 63, top = (uint64_t(1) << E) - 1;
        const uint64_t field = (b >> M) & top, mant = b & ((uint32_t(1) << M) - 1);
        const int64_t bias = (int64_t)(top >> 1);
        if (field == top) return double_of_bits(sign | 0x7ff0000000000000u | (mant << (52 - M)));     // inf, NaN
        if (field) return double_of_bits(sign | ((field - (uint64_t)bias + 1023) << 52) | (mant << (52 - M)));
        if (!mant) return double_of_bits(sign);
        const int64_t n = bitlen(mant);                     // subnormal: mant * 2**(1 - bias - M)
        const int64_t e = n - 1 + 1 - bias - M;
        return double_of_bits(sign | ((uint64_t)(e + 1023) << 52) | ((mant << (53 - n)) & ((uint64_t(1) << 52) - 1)));
    }
    // The element's bytes as an unsigned integer of its size, in the host's byte order.
    uint64_t load(const char* p) const {
        switch (view.itemsize) {
            case 1: return (uint8_t)*p;
            case 2: { uint16_t v; std::memcpy(&v, p, 2); return v; }
            case 4: { uint32_t v; std::memcpy(&v, p, 4); return v; }
            default: { uint64_t v; std::memcpy(&v, p, 8); return v; }
        }
    }
    // The element's integer value as its low 64 bits (two's complement) and a Python int.
    uint64_t as_bits(const char* p) const {
        uint64_t v = load(p);
        if (kind == 'i' && view.itemsize < 8 && ((v >> (view.itemsize * 8 - 1)) & 1))
            v |= ~uint64_t(0) << (view.itemsize * 8);   // sign-extend
        return v;
    }
    nb::object as_int(const char* p) const {
        const uint64_t v = as_bits(p);
        return kind == 'i' ? steal_checked(PyLong_FromLongLong((long long)v))
                           : steal_checked(PyLong_FromUnsignedLongLong(v));
    }
};

// Objects that are neither nested lists nor a numeric buffer but know how to
// become lists (object arrays, big-endian arrays, array-likes): their
// tolist(), else the object itself.
static nb::object as_lists(PyObject* values) {
    if (!PyList_Check(values) && !PyTuple_Check(values) && PyObject_HasAttrString(values, "tolist") &&
        PyObject_HasAttrString(values, "shape"))
        return steal_checked(PyObject_CallMethod(values, "tolist", nullptr));
    return nb::borrow(values);
}

// ---------------------------------------------------------------- construction

// Round one Python value into the array's format (FPFormat.__call__).
static inline void quantize_elem(PyFPArray* a, size_t i, PyObject* v, bool fast) {
    const Fmt& F = *a->f;
    if (fast && PyFloat_CheckExact(v)) {
        double d = PyFloat_AS_DOUBLE(v);
        if (double_finite(d)) {
            uint64_t bits;
            std::memcpy(&bits, &d, 8);
            FV<uint64_t> x;
            x.neg = bits >> 63;
            uint64_t field = (bits >> 52) & 0x7ff, mant = bits & ((uint64_t(1) << 52) - 1);
            x.sig = field ? mant | (uint64_t(1) << 52) : mant;
            x.exp = (int64_t)(field ? field : 1) - 1075;
            unsigned fl = 0;
            FCode c;
            if (fast_round_code(x, false, F, c, fl)) {
                a->raw[i] = c.raw(F);
                a->flags[i] = (uint8_t)fl;
                ++g_fast_hits;
                return;
            }
        }
    }
    if (is_fp(v)) {
        // A normal value or a zero of the array's own format converts to
        // itself (flags cleared). Not NaN/inf (NaN conventions), and not a
        // subnormal: formats that flush or wrap do not keep it.
        const PyFP* p = (const PyFP*)v;
        if (p->f->id == F.id && is_finite(p->c, F) && (!F.has_zero || p->c.field != 0 || mant_zero(p->c, F))) {
            a->raw[i] = raw64_of(p->c, F);
            a->flags[i] = 0;
            return;
        }
    }
    if (fast) ++g_fast_misses;
    nb::object r = fp_from_number(v, a->fmt, Py_None);
    set_elem(a, i, r.ptr());
}

static PyFPArray* array_convert(PyFPArray* src, PyObject* fmt);

static PyObject* array_tp_new(PyTypeObject*, PyObject* args, PyObject* kwargs) {
    VF_TRY
    static const char* kw[] = {"values", "fmt", nullptr};
    PyObject *values, *fmt;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:FPArray", (char**)kw, &values, &fmt)) return nullptr;
    check_array_format(fmt);
    if (is_array(values)) return (PyObject*)array_convert((PyFPArray*)values, fmt);
    static const EwFmt f64{52, 1023, 63, 2047, (uint64_t(1) << 52) - 1, 2046, 0, true, false, 0};
    NumBuffer buf;
    if (buf.open(values)) {
        // A numeric buffer: its numbers rounded into the format, exactly as
        // the same Python floats or ints would be.
        PyFPArray* a = array_new(fmt, buf.shape);
        nb::object hold = nb::steal((PyObject*)a);
        const bool fast = g_fast && fast_target(*a->f);
        const SimdKernels* simd = fast && buf.kind == 'f' ? simd_active() : nullptr;
        const size_t n = buf.n;
        if (simd) {
            // binary64 codes of the elements: the buffer itself when it is a
            // C-contiguous float64 array, else a widened copy
            std::vector<uint64_t> copy;
            const uint64_t* bits;
            if (buf.view.itemsize == 8 && PyBuffer_IsContiguous(&buf.view, 'C')) bits = (const uint64_t*)buf.view.buf;
            else {
                copy.resize(n);
                size_t i = 0;
                buf.for_each([&](const char* p) {
                    const double d = buf.as_double(p);
                    std::memcpy(&copy[i++], &d, 8);
                });
                bits = copy.data();
            }
            const size_t W = simd->width;
            std::vector<uint8_t> redo(n / W + 1);
            simd->ew_conv(bits, n, f64, ew_fmt(*a->f), a->raw.data(), a->flags.data(), redo.data());
            size_t again = 0;
            for (size_t v = 0; v * W < n; ++v) {
                if (!redo[v]) continue;
                for (size_t l = 0; l < W; ++l)
                    if ((redo[v] >> l) & 1) {
                        double d;
                        std::memcpy(&d, &bits[v * W + l], 8);
                        nb::object x = steal_checked(PyFloat_FromDouble(d));
                        quantize_elem(a, v * W + l, x.ptr(), fast);
                        ++again;
                    }
            }
            g_fast_hits += n - again;
        } else {
            size_t i = 0;
            buf.for_each([&](const char* p) {
                nb::object x = buf.kind == 'f' ? steal_checked(PyFloat_FromDouble(buf.as_double(p))) : buf.as_int(p);
                quantize_elem(a, i++, x.ptr(), fast);
            });
        }
        return hold.release().ptr();
    }
    nb::object lists = as_lists(values);
    std::vector<int64_t> shape;
    std::vector<nb::object> leaves;
    flatten(lists.ptr(), shape, leaves);
    PyFPArray* a = array_new(fmt, shape);
    nb::object hold = nb::steal((PyObject*)a);
    const bool fast = g_fast && fast_target(*a->f);
    const SimdKernels* simd = fast ? simd_active() : nullptr;
    if (simd) {
        // Python floats as binary64 codes through the conversion kernel;
        // anything else is given a NaN code, which the kernel hands back.
        const size_t n = leaves.size();
        std::vector<uint64_t> bits(n);
        const size_t W = simd->width;
        std::vector<uint8_t> redo(n / W + 1);
        for (size_t i = 0; i < n; ++i) {
            if (PyFloat_CheckExact(leaves[i].ptr())) {
                const double d = PyFloat_AS_DOUBLE(leaves[i].ptr());
                std::memcpy(&bits[i], &d, 8);
            } else bits[i] = ~uint64_t(0);
        }
        simd->ew_conv(bits.data(), n, f64, ew_fmt(*a->f), a->raw.data(), a->flags.data(), redo.data());
        size_t again = 0;
        for (size_t v = 0; v * W < n; ++v) {
            if (!redo[v]) continue;
            for (size_t l = 0; l < W; ++l)
                if ((redo[v] >> l) & 1) {
                    quantize_elem(a, v * W + l, leaves[v * W + l].ptr(), fast);
                    ++again;
                }
        }
        g_fast_hits += n - again;
    } else
        for (size_t i = 0; i < leaves.size(); ++i) quantize_elem(a, i, leaves[i].ptr(), fast);
    return hold.release().ptr();
    VF_CATCH(nullptr)
}

[[noreturn]] static void raw_out_of_range(const std::string& v, const Fmt& f) {
    raise(PyExc_ValueError, "raw code " + v + " does not fit the " + std::to_string(f.size) + " bits of " + fmt_str(f));
}

// A raw code: an int in [0, 2**size), as FPFormat.from_raw takes it.
static uint64_t raw_from_long(PyObject* o, const Fmt& f) {
    if (!PyLong_Check(o)) raise(PyExc_TypeError, "raw codes must be ints");
    const uint64_t x = PyLong_AsUnsignedLongLong(o);   // OverflowError if negative or above 64 bits
    if (x == (uint64_t)-1 && PyErr_Occurred()) {
        if (!PyErr_ExceptionMatches(PyExc_OverflowError)) raise_current();
        PyErr_Clear();
        raw_out_of_range(py_repr(o), f);
    }
    if (f.size < 64 && (x >> f.size) != 0) raw_out_of_range(std::to_string(x), f);
    return x;
}

// FPArray.from_raw(codes, fmt)
static PyObject* m_from_raw(PyObject*, PyObject* const* args, Py_ssize_t nargs, PyObject* kw) {
    VF_TRY
    PyObject* a[2];
    parse_args("from_raw", args, nargs, kw, {"codes", "fmt"}, a, 2);
    check_array_format(a[1]);
    NumBuffer buf;
    if (buf.open(a[0])) {
        if (buf.kind == 'f') raise(PyExc_TypeError, "raw codes must be ints");
        PyFPArray* r = array_new(a[1], buf.shape);
        nb::object hold = nb::steal((PyObject*)r);
        const Fmt& f = *r->f;
        // Codes in [0, 2**size). A signed array as wide as the format holds
        // the codes as its bit patterns (FP16 codes in int16), and is read so.
        const bool bit_patterns = buf.kind == 'i' && buf.view.itemsize * 8 == f.size;
        size_t i = 0;
        buf.for_each([&](const char* p) {
            const uint64_t v = bit_patterns ? buf.load(p) : buf.as_bits(p);
            const bool neg = !bit_patterns && buf.kind == 'i' && (int64_t)v < 0;
            if (neg || (f.size < 64 && (v >> f.size) != 0))
                raw_out_of_range(neg ? std::to_string((int64_t)v) : std::to_string(v), f);
            r->raw[i++] = v;
        });
        return hold.release().ptr();
    }
    nb::object lists = as_lists(a[0]);
    std::vector<int64_t> shape;
    std::vector<nb::object> leaves;
    flatten(lists.ptr(), shape, leaves);
    PyFPArray* r = array_new(a[1], shape);
    nb::object hold = nb::steal((PyObject*)r);
    for (size_t i = 0; i < leaves.size(); ++i) r->raw[i] = raw_from_long(leaves[i].ptr(), *r->f);
    return hold.release().ptr();
    VF_CATCH(nullptr)
}

// ---------------------------------------------------------------- access

static PyObject* g_format(PyObject* s, void*) { return Py_NewRef(((PyFPArray*)s)->fmt); }
static PyObject* g_shape(PyObject* s, void*) {
    VF_TRY
    return shape_tuple(((PyFPArray*)s)->shape).release().ptr();
    VF_CATCH(nullptr)
}
static PyObject* g_ndim(PyObject* s, void*) { return PyLong_FromSize_t(((PyFPArray*)s)->shape.size()); }
static PyObject* g_flags(PyObject* s, void*) {
    PyFPArray* a = (PyFPArray*)s;
    uint8_t f = 0;
    for (uint8_t x : a->flags) f |= x;
    return Py_NewRef(S.flags[f & 31]);
}
static PyObject* g_raw(PyObject* s, void*) {
    VF_TRY
    PyFPArray* a = (PyFPArray*)s;
    size_t pos = 0;
    return nest(a->shape, 0, pos, [&](size_t i) { return steal_checked(PyLong_FromUnsignedLongLong(a->raw[i])); })
        .release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* m_tolist(PyObject* s, PyObject*) {
    VF_TRY
    PyFPArray* a = (PyFPArray*)s;
    size_t pos = 0;
    return nest(a->shape, 0, pos, [&](size_t i) { return elem_fp(a, i); }).release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* m_to_float(PyObject* s, PyObject*) {
    VF_TRY
    PyFPArray* a = (PyFPArray*)s;
    size_t pos = 0;
    return nest(a->shape, 0, pos, [&](size_t i) {
        return steal_checked(PyFloat_FromDouble(fp_to_double(code_of_raw64(a->raw[i], *a->f), *a->f)));
    }).release().ptr();
    VF_CATCH(nullptr)
}

// Flat list of the elements as FP values (C order).
static nb::object array_flat(PyFPArray* a) {
    nb::object out = steal_checked(PyList_New((Py_ssize_t)a->raw.size()));
    for (size_t i = 0; i < a->raw.size(); ++i) PyList_SET_ITEM(out.ptr(), (Py_ssize_t)i, elem_fp(a, i).release().ptr());
    return out;
}
static PyObject* m_flat(PyObject* s, PyObject*) {
    VF_TRY
    return array_flat((PyFPArray*)s).release().ptr();
    VF_CATCH(nullptr)
}

// The elements as packed native words, for NumPy: 0 raw codes (uint64),
// 1 flags (uint8), 2 values (float64, rounded as float(x) rounds).
static PyObject* m_bytes(PyObject* s, PyObject* what) {
    VF_TRY
    PyFPArray* a = (PyFPArray*)s;
    const long w = PyLong_AsLong(what);
    if (w == -1 && PyErr_Occurred()) raise_current();
    if (w == 0) return PyBytes_FromStringAndSize((const char*)a->raw.data(), (Py_ssize_t)(a->raw.size() * 8));
    if (w == 1) return PyBytes_FromStringAndSize((const char*)a->flags.data(), (Py_ssize_t)a->flags.size());
    std::vector<double> v(a->raw.size());
    for (size_t i = 0; i < v.size(); ++i) v[i] = fp_to_double(code_of_raw64(a->raw[i], *a->f), *a->f);
    return PyBytes_FromStringAndSize((const char*)v.data(), (Py_ssize_t)(v.size() * 8));
    VF_CATCH(nullptr)
}

static Py_ssize_t array_len(PyObject* s) { return (Py_ssize_t)((PyFPArray*)s)->shape[0]; }

// a[key]: integers, slices, `...` and None (a new axis of length 1), as in
// NumPy's basic indexing. An FP when every axis is indexed by an integer,
// else a copy of the selected elements (each with its flags).
static PyObject* array_index(PyFPArray* a, PyObject* const* idx, size_t n) {
    const size_t nd = a->shape.size();
    size_t used = 0, dots = 0;
    for (size_t k = 0; k < n; ++k) {
        if (idx[k] == Py_Ellipsis) ++dots;
        else if (idx[k] != Py_None) ++used;
    }
    if (dots > 1) raise(PyExc_IndexError, "an index can only have a single ellipsis ('...')");
    if (used > nd) raise(PyExc_IndexError, "too many indices for FPArray");
    std::vector<int64_t> stride(nd);
    int64_t acc = 1;
    for (size_t d = nd; d-- > 0;) {
        stride[d] = acc;
        acc *= a->shape[d];
    }
    std::vector<int64_t> oshape, ostep;   // per result axis: length, and step in flat source elements
    int64_t base = 0;
    size_t d = 0;
    auto whole = [&](size_t upto) {
        for (; d < upto; ++d) {
            oshape.push_back(a->shape[d]);
            ostep.push_back(stride[d]);
        }
    };
    for (size_t k = 0; k < n; ++k) {
        PyObject* key = idx[k];
        if (key == Py_Ellipsis) {
            whole(d + (nd - used));
        } else if (key == Py_None) {
            oshape.push_back(1);
            ostep.push_back(0);
        } else if (PySlice_Check(key)) {
            Py_ssize_t start, stop, step;
            if (PySlice_Unpack(key, &start, &stop, &step) < 0) raise_current();
            const Py_ssize_t count = PySlice_AdjustIndices((Py_ssize_t)a->shape[d], &start, &stop, step);
            if (count == 0) raise(PyExc_IndexError, "slice selects no elements (an FPArray cannot be empty)");
            base += (int64_t)start * stride[d];
            oshape.push_back(count);
            // With two or more elements |step| < the axis length, so this
            // stays within the array; a single element never steps.
            ostep.push_back(count > 1 ? (int64_t)step * stride[d] : 0);
            ++d;
        } else {
            if (!PyIndex_Check(key) || PyBool_Check(key))
                raise(PyExc_TypeError, "FPArray indices must be integers or slices");
            Py_ssize_t i = PyNumber_AsSsize_t(key, PyExc_IndexError);
            if (i == -1 && PyErr_Occurred()) raise_current();
            const int64_t len = a->shape[d];
            if (i < 0) i += len;
            if (i < 0 || i >= len) raise(PyExc_IndexError, "FPArray index out of range");
            base += (int64_t)i * stride[d];
            ++d;
        }
    }
    whole(nd);
    if (oshape.empty()) return elem_fp(a, (size_t)base).release().ptr();
    PyFPArray* r = array_new(a->fmt, oshape);
    const size_t on = oshape.size();
    std::vector<int64_t> pos(on, 0);
    int64_t cur = base;
    for (size_t o = 0; o < r->raw.size(); ++o) {
        r->raw[o] = a->raw[(size_t)cur];
        r->flags[o] = a->flags[(size_t)cur];
        for (size_t i = on; i-- > 0;) {
            if (++pos[i] < oshape[i]) {
                cur += ostep[i];
                break;
            }
            cur -= ostep[i] * (oshape[i] - 1);
            pos[i] = 0;
        }
    }
    return (PyObject*)r;
}

static PyObject* array_subscript(PyObject* s, PyObject* key) {
    VF_TRY
    if (PyTuple_Check(key))
        return array_index((PyFPArray*)s, &PyTuple_GET_ITEM(key, 0), (size_t)PyTuple_GET_SIZE(key));
    return array_index((PyFPArray*)s, &key, 1);
    VF_CATCH(nullptr)
}

static PyObject* array_item(PyObject* s, Py_ssize_t i) {
    VF_TRY
    // No negative-index fix-up here: CPython adds the length before calling.
    if (i < 0 || i >= (Py_ssize_t)((PyFPArray*)s)->shape[0]) raise(PyExc_IndexError, "FPArray index out of range");
    nb::object k = steal_checked(PyLong_FromSsize_t(i));
    PyObject* kp = k.ptr();
    return array_index((PyFPArray*)s, &kp, 1);
    VF_CATCH(nullptr)
}

static PyObject* array_repr(PyObject* s) {
    VF_TRY
    PyFPArray* a = (PyFPArray*)s;
    std::string r = "FPArray(shape=" + py_repr(shape_tuple(a->shape).ptr()) + ", [" + py_str(a->fmt) + "])";
    return PyUnicode_FromStringAndSize(r.data(), (Py_ssize_t)r.size());
    VF_CATCH(nullptr)
}

// Bit-exact equality: same format, shape and codes.
static PyObject* array_richcompare(PyObject* x, PyObject* y, int op) {
    if ((op != Py_EQ && op != Py_NE) || !is_array(y)) Py_RETURN_NOTIMPLEMENTED;
    PyFPArray *a = (PyFPArray*)x, *b = (PyFPArray*)y;
    bool eq = a->f->id == b->f->id && a->shape == b->shape && a->raw == b->raw;
    return PyBool_FromLong(eq == (op == Py_EQ));
}

static PyObject* m_transpose(PyObject* s, PyObject* args) {
    VF_TRY
    PyFPArray* a = (PyFPArray*)s;
    size_t nd = a->shape.size();
    std::vector<size_t> perm(nd);
    Py_ssize_t given = PyTuple_GET_SIZE(args);
    if (given == 1 && PyTuple_Check(PyTuple_GET_ITEM(args, 0))) {
        args = PyTuple_GET_ITEM(args, 0);
        given = PyTuple_GET_SIZE(args);
    }
    if (given == 0) {
        for (size_t i = 0; i < nd; ++i) perm[i] = nd - 1 - i;
    } else {
        if ((size_t)given != nd) raise(PyExc_ValueError, "axes must be a permutation of the array's axes");
        std::vector<bool> seen(nd);
        for (size_t i = 0; i < nd; ++i) {
            Py_ssize_t ax = PyNumber_AsSsize_t(PyTuple_GET_ITEM(args, (Py_ssize_t)i), PyExc_IndexError);
            if (ax == -1 && PyErr_Occurred()) raise_current();
            if (ax < 0) ax += (Py_ssize_t)nd;
            if (ax < 0 || (size_t)ax >= nd || seen[(size_t)ax])
                raise(PyExc_ValueError, "axes must be a permutation of the array's axes");
            seen[(size_t)ax] = true;
            perm[i] = (size_t)ax;
        }
    }
    std::vector<int64_t> shape(nd);
    for (size_t i = 0; i < nd; ++i) shape[i] = a->shape[perm[i]];
    std::vector<size_t> src_stride(nd);
    size_t acc = 1;
    for (size_t i = nd; i-- > 0;) { src_stride[i] = acc; acc *= (size_t)a->shape[i]; }
    PyFPArray* r = array_new(a->fmt, shape);
    std::vector<int64_t> idx(nd, 0);
    for (size_t o = 0; o < r->raw.size(); ++o) {
        size_t src = 0;
        for (size_t i = 0; i < nd; ++i) src += (size_t)idx[i] * src_stride[perm[i]];
        r->raw[o] = a->raw[src];
        r->flags[o] = a->flags[src];
        for (size_t i = nd; i-- > 0;) {
            if (++idx[i] < shape[i]) break;
            idx[i] = 0;
        }
    }
    return (PyObject*)r;
    VF_CATCH(nullptr)
}
static PyObject* g_T(PyObject* s, void*) {
    nb::object none = nb::steal(PyTuple_New(0));
    return m_transpose(s, none.ptr());
}

static PyObject* m_reshape(PyObject* s, PyObject* args) {
    VF_TRY
    PyFPArray* a = (PyFPArray*)s;
    if (PyTuple_GET_SIZE(args) == 1 && (PyTuple_Check(PyTuple_GET_ITEM(args, 0)) || PyList_Check(PyTuple_GET_ITEM(args, 0))))
        args = PyTuple_GET_ITEM(args, 0);
    nb::object seq = snapshot(args, "shape must be a sequence of ints");
    if ((size_t)tuple_size(seq) > kMaxDims) raise(PyExc_ValueError, "too many dimensions (at most 64)");
    std::vector<int64_t> shape;
    size_t total = 1;
    Py_ssize_t infer = -1;      // the axis given as -1: its length follows from the others
    for (Py_ssize_t i = 0; i < tuple_size(seq); ++i) {
        Py_ssize_t d = PyNumber_AsSsize_t(tuple_item(seq, i), PyExc_OverflowError);
        if (d == -1 && PyErr_Occurred()) raise_current();
        if (d == -1 && infer < 0) {
            infer = i;
            d = 1;
        } else if (d < 1)
            raise(PyExc_ValueError, "shape dimensions must be positive (or one -1)");
        shape.push_back(d);
        total = elems_mul(total, (size_t)d);
    }
    if (infer >= 0 && a->raw.size() % total == 0) {
        shape[(size_t)infer] = (int64_t)(a->raw.size() / total);
        total = a->raw.size();
    }
    if (shape.empty() || total != a->raw.size())
        raise(PyExc_ValueError, "cannot reshape " + std::to_string(a->raw.size()) + " elements into that shape");
    PyFPArray* r = array_new(a->fmt, shape);
    r->raw = a->raw;
    r->flags = a->flags;
    return (PyObject*)r;
    VF_CATCH(nullptr)
}

// The array repeated along new leading axes and along its axes of length 1.
static PyFPArray* array_broadcast(PyFPArray* a, const std::vector<int64_t>& shape) {
    const size_t lead = shape.size() - std::min(shape.size(), a->shape.size());
    bool ok = a->shape.size() <= shape.size();
    for (size_t i = 0; ok && i < a->shape.size(); ++i) ok = a->shape[i] == 1 || a->shape[i] == shape[lead + i];
    if (!ok) raise(PyExc_ValueError, "cannot broadcast shape " + shape_str(a->shape) + " to " + shape_str(shape));
    const std::vector<size_t> map = broadcast_index(a->shape, shape);
    PyFPArray* r = array_new(a->fmt, shape);
    for (size_t o = 0; o < map.size(); ++o) {
        r->raw[o] = a->raw[map[o]];
        r->flags[o] = a->flags[map[o]];
    }
    return r;
}

static std::vector<int64_t> shape_arg(PyObject* arg) {
    std::vector<int64_t> shape;
    if (PyIndex_Check(arg) && !PyBool_Check(arg)) {
        shape.push_back(PyNumber_AsSsize_t(arg, PyExc_OverflowError));
        if (shape[0] == -1 && PyErr_Occurred()) raise_current();
    } else {
        nb::object seq = snapshot(arg, "shape must be an int or a sequence of ints");
        if ((size_t)tuple_size(seq) > kMaxDims) raise(PyExc_ValueError, "too many dimensions (at most 64)");
        for (Py_ssize_t i = 0; i < tuple_size(seq); ++i) {
            const Py_ssize_t d = PyNumber_AsSsize_t(tuple_item(seq, i), PyExc_OverflowError);
            if (d == -1 && PyErr_Occurred()) raise_current();
            shape.push_back(d);
        }
    }
    if (shape.empty()) raise(PyExc_ValueError, "an FPArray has at least one axis");
    for (int64_t d : shape)
        if (d < 1) raise(PyExc_ValueError, "shape dimensions must be positive");
    checked_count(shape);
    return shape;
}

static PyObject* m_broadcast_to(PyObject* s, PyObject* shape) {
    VF_TRY
    return (PyObject*)array_broadcast((PyFPArray*)s, shape_arg(shape));
    VF_CATCH(nullptr)
}

// ---------------------------------------------------------------- arithmetic

// a op b for operands of one fast format; strides of 0 repeat a scalar. OP
// is fast_add_code, fast_sub_code, ... : one loop per operation.
template <class OP>
VF_NOINLINE static void fast_binop_loop(OP&& fast_one, char op, PyObject* fmt, const uint64_t* a, size_t sa,
                                        const uint64_t* b, size_t sb, size_t n, uint64_t* out, uint8_t* flags) {
    const Fmt& F = fmt_of(fmt);
    const RawFmt rf(F);
    const bool has_special = F.has_nan();
    const uint64_t top = F.top;
    const int64_t M = F.M;
    uint64_t misses = 0;
    // SIMD first, where there is a kernel for the operation; it marks the
    // elements it leaves to the scalar code below.
    const SimdKernels* simd = simd_active();
    if (!(simd && (op == '*' ? M <= kEwMaxMulM : op != '/' && M <= kEwMaxAddM))) simd = nullptr;
    const size_t W = simd ? simd->width : 1;
    std::vector<uint8_t> redo;
    if (simd) {
        redo.resize(n / W + 1);
        simd->ew_bin(op, a, sa, b, sb, n, ew_fmt(F), out, flags, redo.data());
    }
    for (size_t i = 0; i < n; ++i) {
        if (simd) {
            if (!redo[i / W]) { i |= W - 1; continue; }        // the whole vector is done
            if (!((redo[i / W] >> (i % W)) & 1)) continue;
        }
        const uint64_t ra = a[i * sa], rb = b[i * sb];
        // The all-ones exponent holds inf/NaN (and, for 'fn', ordinary values:
        // those take the general path too).
        const bool special = has_special & ((((ra >> M) & top) == top) | (((rb >> M) & top) == top));
        unsigned fl = 0;
        FCode c;
        if (!special && fast_one(fv_of_raw<uint64_t>(ra, rf), fv_of_raw<uint64_t>(rb, rf), F, c, fl)) [[likely]] {
            out[i] = c.raw(F);
            flags[i] = (uint8_t)fl;
            continue;
        }
        ++misses;
        const Code ca = code_of_raw64(ra, F), cb = code_of_raw64(rb, F);
        nb::object r = arith(op, {ca, &F, fmt}, {cb, &F, fmt}, fmt);
        PyFP* p = (PyFP*)r.ptr();
        out[i] = raw64_of(p->c, F);
        flags[i] = p->flags;
    }
    g_fast_hits += n - misses;
    g_fast_misses += misses;
}

static void fast_binop(char op, PyObject* fmt, const uint64_t* a, size_t sa, const uint64_t* b, size_t sb, size_t n,
                       uint64_t* out, uint8_t* flags) {
    using X = const FV<uint64_t>&;
    switch (op) {
        case '+':
            return fast_binop_loop([](X x, X y, const Fmt& f, FCode& c, unsigned& fl) { return fast_add_code(x, y, f, c, fl); },
                                   op, fmt, a, sa, b, sb, n, out, flags);
        case '-':
            return fast_binop_loop([](X x, X y, const Fmt& f, FCode& c, unsigned& fl) { return fast_sub_code(x, y, f, c, fl); },
                                   op, fmt, a, sa, b, sb, n, out, flags);
        case '*':
            return fast_binop_loop([](X x, X y, const Fmt& f, FCode& c, unsigned& fl) { return fast_mul_code(x, y, f, c, fl); },
                                   op, fmt, a, sa, b, sb, n, out, flags);
        default:
            return fast_binop_loop([](X x, X y, const Fmt& f, FCode& c, unsigned& fl) { return fast_div_code(x, y, f, c, fl); },
                                   op, fmt, a, sa, b, sb, n, out, flags);
    }
}

static PyObject* num_op(char op, PyObject* x, PyObject* y) {
    switch (op) {
        case '+': return PyNumber_Add(x, y);
        case '-': return PyNumber_Subtract(x, y);
        case '*': return PyNumber_Multiply(x, y);
        default: return PyNumber_TrueDivide(x, y);
    }
}

static PyObject* array_binop(PyObject* x, PyObject* y, char op) {
    VF_TRY
    const bool xa = is_array(x), ya = is_array(y);
    PyFPArray* A = xa ? (PyFPArray*)x : nullptr;
    PyFPArray* B = ya ? (PyFPArray*)y : nullptr;
    nb::object hold_a, hold_b;   // operands repeated to the common shape
    if (xa && ya && A->shape != B->shape) {
        const std::vector<int64_t> shape = broadcast_shape(A->shape, B->shape);
        if (A->shape != shape) hold_a = nb::steal((PyObject*)(A = array_broadcast(A, shape)));
        if (B->shape != shape) hold_b = nb::steal((PyObject*)(B = array_broadcast(B, shape)));
    }
    PyFPArray* any = xa ? A : B;
    const size_t n = any->raw.size();
    // Scalars: an FP, or a Python number (handled by the scalar operator).
    PyObject* scalar = xa && ya ? nullptr : (xa ? y : x);
    if (scalar && !is_fp(scalar) && !(PyLong_Check(scalar) && !PyBool_Check(scalar)) && !PyFloat_Check(scalar) &&
        !is_fraction(scalar))
        Py_RETURN_NOTIMPLEMENTED;
    const Fmt* fx = xa ? A->f : (is_fp(x) ? ((PyFP*)x)->f : nullptr);
    const Fmt* fy = ya ? B->f : (is_fp(y) ? ((PyFP*)y)->f : nullptr);
    if (g_fast && fx && fy && fx->id == fy->id && fast_target(*fx)) {
        const Fmt& F = *fx;
        PyFPArray* r = array_new(any->fmt, any->shape);
        nb::object hold = nb::steal((PyObject*)r);
        uint64_t sx = 0, sy = 0;
        if (!xa) sx = raw64_of(((PyFP*)x)->c, F);
        if (!ya) sy = raw64_of(((PyFP*)y)->c, F);
        const uint64_t* pa = xa ? A->raw.data() : &sx;
        const uint64_t* pb = ya ? B->raw.data() : &sy;
        fast_binop(op, any->fmt, pa, xa, pb, ya, n, r->raw.data(), r->flags.data());
        return hold.release().ptr();
    }
    // General: the scalar operator on each pair of elements.
    PyFPArray* r = nullptr;
    nb::object hold;
    for (size_t i = 0; i < n; ++i) {
        nb::object ex = xa ? elem_fp(A, i) : nb::borrow(x);
        nb::object ey = ya ? elem_fp(B, i) : nb::borrow(y);
        nb::object v = steal_checked(num_op(op, ex.ptr(), ey.ptr()));
        if (!is_fp(v.ptr())) raise(PyExc_TypeError, "unsupported operand for an FPArray operation");
        if (!r) {
            r = array_new(((PyFP*)v.ptr())->fmt, any->shape);
            hold = nb::steal((PyObject*)r);
        }
        set_elem(r, i, v.ptr());
    }
    return hold.release().ptr();
    VF_CATCH(nullptr)
}

static PyObject* array_add(PyObject* a, PyObject* b) { return array_binop(a, b, '+'); }
static PyObject* array_sub(PyObject* a, PyObject* b) { return array_binop(a, b, '-'); }
static PyObject* array_mul(PyObject* a, PyObject* b) { return array_binop(a, b, '*'); }
static PyObject* array_div(PyObject* a, PyObject* b) { return array_binop(a, b, '/'); }

static PyObject* array_unary(PyObject* s, PyObject* (*fn)(PyObject*)) {
    VF_TRY
    PyFPArray* a = (PyFPArray*)s;
    PyFPArray* r = nullptr;
    nb::object hold;
    for (size_t i = 0; i < a->raw.size(); ++i) {
        nb::object v = steal_checked(fn(elem_fp(a, i).ptr()));
        if (!r) {
            r = array_new(((PyFP*)v.ptr())->fmt, a->shape);
            hold = nb::steal((PyObject*)r);
        }
        set_elem(r, i, v.ptr());
    }
    return hold.release().ptr();
    VF_CATCH(nullptr)
}
static PyObject* array_neg(PyObject* s) { return array_unary(s, PyNumber_Negative); }
static PyObject* array_pos(PyObject* s) { return array_unary(s, PyNumber_Positive); }
static PyObject* array_abs(PyObject* s) { return array_unary(s, PyNumber_Absolute); }

// ---------------------------------------------------------------- conversion

static PyFPArray* array_convert(PyFPArray* src, PyObject* fmt) {
    check_array_format(fmt);
    const Fmt &G = *src->f, &F = fmt_of(fmt);
    PyFPArray* r = array_new(fmt, src->shape);
    nb::object hold = nb::steal((PyObject*)r);
    const bool fast = g_fast && fast_target(F) && G.M <= 61;
    const SimdKernels* simd = fast && ew_source(G) ? simd_active() : nullptr;
    const size_t W = simd ? simd->width : 1;
    std::vector<uint8_t> redo;
    if (simd) {
        redo.resize(src->raw.size() / W + 1);
        simd->ew_conv(src->raw.data(), src->raw.size(), ew_fmt(G), ew_fmt(F), r->raw.data(), r->flags.data(),
                      redo.data());
    }
    for (size_t i = 0; i < src->raw.size(); ++i) {
        if (simd && !((redo[i / W] >> (i % W)) & 1)) continue;
        const Code c = code_of_raw64(src->raw[i], G);
        if (fast && is_finite(c, G)) {
            FCode out;
            unsigned fl = 0;
            if (fast_round_code(fv_of<uint64_t>(c, G), false, F, out, fl)) {
                r->raw[i] = out.raw(F);
                r->flags[i] = (uint8_t)fl;
                continue;
            }
        }
        nb::object v = convert_to((PyFP*)elem_fp(src, i).ptr(), fmt, Py_None);
        set_elem(r, i, v.ptr());
    }
    return (PyFPArray*)hold.release().ptr();
}

static PyObject* m_convert(PyObject* s, PyObject* fmt) {
    VF_TRY
    return (PyObject*)array_convert((PyFPArray*)s, fmt);
    VF_CATCH(nullptr)
}

// ---------------------------------------------------------------- matmul

// _core.array_matmul(a, b, nbatch, m, k, n, batch_a, batch_b, acc, shape):
// both operands flat in C order with b already K x N; one FP per output.
static nb::object py_array_matmul(nb::handle ha, nb::handle hb, int64_t nbatch, int64_t m, int64_t k, int64_t n,
                                  bool batch_a, bool batch_b, nb::handle acc, nb::handle shape) {
    if (!is_array(ha.ptr()) || !is_array(hb.ptr())) raise(PyExc_TypeError, "expected FPArray operands");
    PyFPArray *A = (PyFPArray*)ha.ptr(), *B = (PyFPArray*)hb.ptr();
    AccSpec spec;
    if (!spec_from(acc, spec)) raise(PyExc_TypeError, "acc must be an FPFormat or an Accumulator");
    MatDims d{nbatch, m, k, n, batch_a, batch_b};
    if (nbatch < 1 || m < 1 || k < 1 || n < 1) raise(PyExc_ValueError, "matmul dimensions must be positive");
    const size_t ua = (size_t)(batch_a ? nbatch : 1), ub = (size_t)(batch_b ? nbatch : 1);
    if (A->raw.size() != elems_mul(elems_mul(ua, (size_t)m), (size_t)k) ||
        B->raw.size() != elems_mul(elems_mul(ub, (size_t)k), (size_t)n))
        raise(PyExc_ValueError, "operand sizes do not match the matmul dimensions");
    nb::object sh = snapshot(shape.ptr(), "shape must be a sequence of ints");
    std::vector<int64_t> oshape;
    for (Py_ssize_t i = 0; i < tuple_size(sh); ++i)
        oshape.push_back(int_in(tuple_item(sh, i), 1, INT64_MAX, "bad matmul result shape"));
    PyFPArray* R = array_new(spec.fmt, oshape);
    nb::object hold = nb::steal((PyObject*)R);
    if (R->raw.size() != elems_mul(elems_mul((size_t)nbatch, (size_t)m), (size_t)n))
        raise(PyExc_ValueError, "result shape does not match the matmul dimensions");

    std::vector<uint8_t> ok;
    bool fast = false;
    auto decode_all = [](PyFPArray* X, std::vector<FV<u128>>& out) {
        const Fmt& f = *X->f;
        if (f.M > 61) return false;
        out.resize(X->raw.size());
        for (size_t i = 0; i < out.size(); ++i) {
            const Code c = code_of_raw64(X->raw[i], f);
            if (!is_finite(c, f)) return false;
            out[i] = fv_of<u128>(c, f);
        }
        return true;
    };
    {
        std::vector<FV<u128>> qa, qb;
        fast = decode_all(A, qa) && decode_all(B, qb) &&
               fast_matmul(qa, A->f->M + 1, qb, B->f->M + 1, d, spec, R->raw, R->flags, ok, nullptr);
    }
    bool all_ok = fast;
    if (fast)
        for (uint8_t o : ok) all_ok = all_ok && o;
    if (all_ok) return hold;
    // General path for the remaining outputs, on FP objects.
    nb::object la = array_flat(A), lb = array_flat(B);
    std::vector<In> va, vb;
    bool nf = false, nums = false;
    parse_all(la, va, nf, nums);
    parse_all(lb, vb, nf, nums);
    size_t idx = 0;
    for (int64_t bi = 0; bi < nbatch; ++bi) {
        int64_t oa = batch_a ? bi * m * k : 0, ob = batch_b ? bi * k * n : 0;
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < n; ++j, ++idx) {
                if (fast && ok[idx]) continue;
                nb::object r = sum_products(va.data() + oa + i * k, vb.data() + ob + j, (size_t)k, 1, (size_t)n,
                                            nullptr, spec);
                set_elem(R, idx, r.ptr());
            }
    }
    return hold;
}

// ---------------------------------------------------------------- element-wise methods

// _core.array_map(name, operands): the FP method `name` called on every
// element of operands[0], with the other operands as its arguments. FPArray
// operands broadcast and give their elements; anything else is passed as it
// is. FP results make an FPArray; results that are (value, flags) pairs
// (the comparisons) give (nested lists of the values, the OR of the flags);
// other results give nested lists.
static nb::object py_array_map(nb::handle name, nb::handle operands) {
    // A tuple: the methods called below run Python code, which must not be
    // able to change the operands under this loop.
    nb::object seq = snapshot(operands.ptr(), "operands must be a sequence");
    const size_t n = (size_t)tuple_size(seq);
    PyObject** items = &PyTuple_GET_ITEM(seq.ptr(), 0);
    if (n < 1 || !is_array(items[0])) raise(PyExc_TypeError, "expected an FPArray");
    std::vector<int64_t> shape = ((PyFPArray*)items[0])->shape;
    for (size_t i = 1; i < n; ++i)
        if (is_array(items[i])) shape = broadcast_shape(shape, ((PyFPArray*)items[i])->shape);
    std::vector<std::vector<size_t>> maps(n);
    for (size_t i = 0; i < n; ++i)
        if (is_array(items[i])) maps[i] = broadcast_index(((PyFPArray*)items[i])->shape, shape);
    const size_t total = count_of(shape);
    std::vector<PyObject*> args(n + 1);     // args[0] is scratch space for the call
    std::vector<nb::object> elems(n);
    PyFPArray* r = nullptr;
    nb::object hold;
    std::vector<nb::object> others;
    bool pairs = true;
    unsigned flags = 0;
    for (size_t o = 0; o < total; ++o) {
        for (size_t i = 0; i < n; ++i) {
            if (is_array(items[i])) {
                elems[i] = elem_fp((PyFPArray*)items[i], maps[i][o]);
                args[i + 1] = elems[i].ptr();
            } else
                args[i + 1] = items[i];
        }
        nb::object v = steal_checked(
            PyObject_VectorcallMethod(name.ptr(), args.data() + 1, n | PY_VECTORCALL_ARGUMENTS_OFFSET, nullptr));
        if (is_fp(v.ptr())) {
            PyFP* p = (PyFP*)v.ptr();
            if (!r) {
                if (o != 0) raise(PyExc_TypeError, "results of mixed kinds");
                r = array_new(p->fmt, shape);
                hold = nb::steal((PyObject*)r);
            } else if (p->f->id != r->f->id)
                raise(PyExc_TypeError, "results of mixed formats");
            set_elem(r, o, v.ptr());
            continue;
        }
        if (r) raise(PyExc_TypeError, "results of mixed kinds");
        if (pairs && PyTuple_Check(v.ptr()) && PyTuple_GET_SIZE(v.ptr()) == 2 &&
            PyLong_Check(PyTuple_GET_ITEM(v.ptr(), 1))) {
            const long fl = PyLong_AsLong(PyTuple_GET_ITEM(v.ptr(), 1));
            if (fl == -1 && PyErr_Occurred()) raise_current();
            flags |= (unsigned)fl;
            others.push_back(nb::borrow(PyTuple_GET_ITEM(v.ptr(), 0)));
        } else {
            if (o != 0 && pairs) raise(PyExc_TypeError, "results of mixed kinds");
            pairs = false;
            others.push_back(std::move(v));
        }
    }
    if (r) return hold;
    size_t pos = 0;
    nb::object lists = nest(shape, 0, pos, [&](size_t i) { return others[i]; });
    if (!pairs) return lists;
    return nb::make_tuple(lists, nb::borrow(S.flags[flags & 31]));
}

// _core.array_from_fps(values, shape): an array of these FP values (one
// format), each keeping its code and flags.
static nb::object py_array_from_fps(nb::handle values, nb::handle shape) {
    nb::object seq = snapshot(values.ptr(), "values must be a sequence of FP");
    const size_t n = (size_t)tuple_size(seq);
    PyObject** items = n ? &PyTuple_GET_ITEM(seq.ptr(), 0) : nullptr;
    const std::vector<int64_t> sh = shape_arg(shape.ptr());
    if (n == 0 || count_of(sh) != n) raise(PyExc_ValueError, "the shape does not match the number of values");
    for (size_t i = 0; i < n; ++i)
        if (!is_fp(items[i]) || ((PyFP*)items[i])->f->id != ((PyFP*)items[0])->f->id)
            raise(PyExc_TypeError, "values must be FP values of one format");
    PyFPArray* r = array_new(((PyFP*)items[0])->fmt, sh);
    for (size_t i = 0; i < n; ++i) set_elem(r, i, items[i]);
    return nb::steal((PyObject*)r);
}

// ---------------------------------------------------------------- pickling

static nb::object py_array_restore(nb::handle fmt, nb::handle shape, nb::handle raw, nb::bytes flags) {
    nb::object st = snapshot(shape.ptr(), "bad FPArray shape");
    std::vector<int64_t> sh;
    for (Py_ssize_t i = 0; i < tuple_size(st); ++i)
        sh.push_back(int_in(tuple_item(st, i), 1, INT64_MAX, "bad FPArray shape"));
    if (sh.empty()) raise(PyExc_ValueError, "bad FPArray shape");
    nb::object seq = snapshot(raw.ptr(), "raw codes must be a sequence");
    PyFPArray* r = array_new(fmt.ptr(), sh);
    nb::object hold = nb::steal((PyObject*)r);
    if ((size_t)tuple_size(seq) != r->raw.size() || flags.size() != r->raw.size())
        raise(PyExc_ValueError, "bad FPArray state");
    for (size_t i = 0; i < r->raw.size(); ++i) {
        r->raw[i] = raw_from_long(tuple_item(seq, (Py_ssize_t)i), *r->f);
        r->flags[i] = (uint8_t)flags.c_str()[i] & 31;
    }
    return hold;
}

static PyObject* m_reduce(PyObject* s, PyObject*) {
    VF_TRY
    PyFPArray* a = (PyFPArray*)s;
    nb::object restore = steal_checked(PyObject_GetAttrString((PyObject*)g_array_type, "_restore"));
    nb::object raw = steal_checked(PyList_New((Py_ssize_t)a->raw.size()));
    for (size_t i = 0; i < a->raw.size(); ++i)
        PyList_SET_ITEM(raw.ptr(), (Py_ssize_t)i, check(PyLong_FromUnsignedLongLong(a->raw[i])));
    nb::object fl = steal_checked(PyBytes_FromStringAndSize((const char*)a->flags.data(), (Py_ssize_t)a->flags.size()));
    nb::object sh = shape_tuple(a->shape);
    return Py_BuildValue("(O(OOOO))", restore.ptr(), a->fmt, sh.ptr(), raw.ptr(), fl.ptr());
    VF_CATCH(nullptr)
}

// ---------------------------------------------------------------- type

#define FASTKW (METH_FASTCALL | METH_KEYWORDS)
static PyMethodDef array_methods[] = {
    {"from_raw", (PyCFunction)(void (*)(void))m_from_raw, FASTKW | METH_STATIC,
     "from_raw(codes, fmt)\n--\n\nAn array from nested lists of raw codes."},
    {"tolist", m_tolist, METH_NOARGS, "tolist($self, /)\n--\n\nNested lists of FP values (each with its flags)."},
    {"to_float", m_to_float, METH_NOARGS, "to_float($self, /)\n--\n\nNested lists of Python floats."},
    {"convert", m_convert, METH_O, "convert($self, fmt, /)\n--\n\nEvery element rounded into another format."},
    {"transpose", m_transpose, METH_VARARGS,
     "transpose($self, /, *axes)\n--\n\nPermute the axes (default: reverse them)."},
    {"reshape", m_reshape, METH_VARARGS,
     "reshape($self, /, *shape)\n--\n\nThe same elements in C order with a new shape (one axis may be -1)."},
    {"broadcast_to", m_broadcast_to, METH_O,
     "broadcast_to($self, shape, /)\n--\n\nThe array repeated to a larger shape (NumPy's broadcasting rule)."},
    {"_flat", m_flat, METH_NOARGS, nullptr},
    {"_bytes", m_bytes, METH_O, nullptr},
    {"__reduce__", m_reduce, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static PyGetSetDef array_getset[] = {
    {"format", g_format, nullptr, nullptr, nullptr},
    {"shape", g_shape, nullptr, nullptr, nullptr},
    {"ndim", g_ndim, nullptr, nullptr, nullptr},
    {"raw", g_raw, nullptr, "Nested lists of raw codes.", nullptr},
    {"flags", g_flags, nullptr, "OR of the status flags of every element.", nullptr},
    {"T", g_T, nullptr, "The array with its axes reversed.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

void register_array(nb::module_& m) {
    static PyType_Slot slots[] = {
        {Py_tp_new, (void*)array_tp_new},
        {Py_tp_dealloc, (void*)array_dealloc},
        {Py_tp_repr, (void*)array_repr},
        {Py_tp_richcompare, (void*)array_richcompare},
        {Py_tp_hash, (void*)PyObject_HashNotImplemented},
        {Py_tp_methods, array_methods},
        {Py_tp_getset, array_getset},
        {Py_sq_length, (void*)array_len},
        {Py_sq_item, (void*)array_item},
        {Py_mp_length, (void*)array_len},
        {Py_mp_subscript, (void*)array_subscript},
        {Py_nb_add, (void*)array_add},
        {Py_nb_subtract, (void*)array_sub},
        {Py_nb_multiply, (void*)array_mul},
        {Py_nb_true_divide, (void*)array_div},
        {Py_nb_negative, (void*)array_neg},
        {Py_nb_positive, (void*)array_pos},
        {Py_nb_absolute, (void*)array_abs},
        {Py_tp_doc,
         (void*)"FPArray(values, fmt)\n--\n\nAn N-dimensional array of FP values in one format.\n\n"
                "Each operation gives, element by element, exactly the result of the scalar FP\n"
                "operation (code and flags)."},
        {0, nullptr}};
    static PyType_Spec spec = {"verifloat.array.FPArray", sizeof(PyFPArray), 0, Py_TPFLAGS_DEFAULT, slots};
    g_array_type = (PyTypeObject*)check(PyType_FromSpec(&spec));
    g_array_type->tp_name = "FPArray";
    m.attr("FPArray") = nb::handle((PyObject*)g_array_type);
    m.def("array_matmul", &py_array_matmul);
    m.def("array_restore", &py_array_restore);
    m.def("array_map", &py_array_map);
    m.def("array_from_fps", &py_array_from_fps);
}

}  // namespace vf
