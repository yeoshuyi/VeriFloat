// Dot-product accumulation shared by dot/matmul on lists (accum.cpp) and on
// FPArray (array.cpp).
#pragma once

#include "fp.hpp"

#include "fast.hpp"

#include <vector>

namespace vf {

using DB = Dy<BigInt>;

// A value with the sign used for zero results (accum.py `_signed`).
struct SV {
    DB v;
    bool sign = false;
};

// A dot-product operand on the general path.
struct In {
    PyFP* fp = nullptr;   // borrowed FP, or nullptr for a number
    SV n;                 // the number's exact value (v < 0 as sign)
};

struct AccSpec {
    PyObject* fmt;
    int order;            // 0 exact, 1 sequential, 2 pairwise
    PyObject* product;    // FPFormat or nullptr
    int64_t group;
    bool aligned;
    int64_t align_bits;
};

// acc = (fmt, order, product, group, align_bits); false if not native.
bool spec_from(nb::handle acc, AccSpec& s);

// Parse a flat sequence of operands; false if the Python path must handle it.
bool parse_all(nb::handle seq, std::vector<In>& out, bool& nonfinite_fp, bool& numbers);

// Accumulator.sum_products on the general path.
nb::object sum_products(const In* a, const In* b, size_t n, size_t sa, size_t sb, const In* init,
                        const AccSpec& spec);

struct MatDims {
    int64_t nbatch, m, k, n;
    bool batch_a, batch_b;
    size_t outputs() const { return (size_t)(nbatch * m * n); }
};

// Matmul over finite operands (flat, C order) on the fast kernel. `bits_a`
// and `bits_b` bound the operands' significand widths. Returns false when
// the accumulator or the widths rule the fast kernel out. Otherwise each
// output either has ok = 1 with its code and flags, or ok = 0 and must be
// computed on the general path. `unr` (optional) receives the exact sums of
// an order="exact" accumulator.
bool fast_matmul(const std::vector<FV<u128>>& a, int64_t bits_a, const std::vector<FV<u128>>& b, int64_t bits_b,
                 const MatDims& d, const AccSpec& spec, std::vector<uint64_t>& raw, std::vector<uint8_t>& flags,
                 std::vector<uint8_t>& ok, std::vector<DB>* unr);

// An exact Python number (int, finite float, Fraction with a power-of-two
// denominator) as sig * 2**exp with sig odd and below 2**62; false if it is
// something else or too wide.
bool number_fv(PyObject* o, FV<u128>& v);

// The tight-kernel driver alone (no fallback to the other fast kernels), for
// the self-check: false if it does not apply, else ok marks what it computed.
bool lean_matmul_check(const std::vector<FV<u128>>& a, const std::vector<FV<u128>>& b, const MatDims& d, const Fmt& F,
                       const Fmt* P, int order, std::vector<uint64_t>& raw, std::vector<uint8_t>& flags,
                       std::vector<uint8_t>& ok, std::vector<DB>* unr);

}  // namespace vf
