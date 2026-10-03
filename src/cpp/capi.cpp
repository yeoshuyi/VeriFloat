// The C library (include/verifloat.h): the core's operations on raw codes,
// for SystemVerilog DPI-C and C callers. No Python here; this file is the
// core's host in the library, as pyconv.cpp is in the extension.
#include "verifloat.h"

#include "ops.hpp"

#include <initializer_list>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#ifndef VF_VERSION
#define VF_VERSION "unknown"
#endif

struct vf_fmt {
    vf::Fmt f;
    std::string name;
    size_t words = 0;   // 32-bit words of a code
};

namespace vf {

// ---------------------------------------------------------------- the core's host

namespace {

struct CoreError : std::exception {
    std::string msg;
    explicit CoreError(std::string m) : msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

thread_local std::string t_error;

uint64_t g_sr_value = 0;
uint64_t (*g_sr_source)(int, void*) = nullptr;
void* g_sr_ctx = nullptr;

}  // namespace

void fail(Err, const std::string& msg) { throw CoreError(msg); }

void sr_draw(SRArg& sr, const Fmt& f) {
    const uint64_t v = g_sr_source ? g_sr_source((int)f.sr_bits, g_sr_ctx) : g_sr_value;
    if (v <= (uint64_t)INT64_MAX) sr.set((int64_t)v);
    else sr.set(BigInt(v));
}

namespace {

// Run `fn`; an error becomes result 0, flags VF_ERROR and a message.
template <class R, class Fn> R guarded(int* flags, Fn&& fn) {
    try {
        return fn();
    } catch (const std::exception& e) {
        t_error = e.what();
    } catch (...) {
        t_error = "unknown error";
    }
    if (flags) *flags = VF_ERROR;
    return R();
}

const vf_fmt& checked(const vf_fmt* f) {
    if (!f) fail(Err::VALUE, "null format (vf_format failed?)");
    return *f;
}

// ---------------------------------------------------------------- codes

Code code64(uint64_t raw, const vf_fmt& h) {
    const Fmt& f = h.f;
    if (f.size > 64) fail(Err::VALUE, h.name + " is wider than 64 bits: use the _w functions");
    Code c;   // at most 64 bits with E >= 1: the mantissa has under 64 bits
    c.sign = f.is_signed && ((raw >> (f.E + f.M)) & 1);
    c.field = (raw >> f.M) & f.top;
    c.mant = raw & f.mask();
    return c;
}

uint64_t raw64(const Code& c, const Fmt& f) {
    uint64_t r = (c.field << f.M) | c.mant;
    if (c.sign && f.is_signed) r |= uint64_t(1) << (f.E + f.M);
    return r;
}

Code code_w(const uint32_t* w, const vf_fmt& h) {
    if (!w) fail(Err::VALUE, "null operand");
    BigInt raw = 0;
    for (size_t i = h.words; i-- > 0;) {
        raw <<= 32;
        raw |= w[i];
    }
    return code_from_raw(raw, h.f);
}

void put_w(const Code& c, const vf_fmt& h, uint32_t* out) {
    if (!out) fail(Err::VALUE, "null result");
    BigInt raw = raw_of_code(c, h.f);
    for (size_t i = 0; i < h.words; ++i) {
        out[i] = static_cast<uint32_t>(raw & BigInt(0xffffffffu));
        raw >>= 32;
    }
}

// ---------------------------------------------------------------- operations

uint8_t rounding_of(int rounding, const Fmt& f) {
    if (rounding == VF_ROUND_FORMAT) return f.rounding;
    if (rounding < VF_RNE || rounding > VF_SR) fail(Err::VALUE, "unknown rounding mode " + std::to_string(rounding));
    return (uint8_t)rounding;
}

OpOut plain_out(const Code& c) { return {{c, 0, EV_NONE}, UnrD()}; }

// One operation of the VF_OP_* list on operands of format F.
OpOut run(int op, const Fmt& F, const Code& ca, const Code& cb, const Code& cc) {
    const Opnd a{ca, &F}, b{cb, &F}, c{cc, &F};
    switch (op) {
        case VF_OP_ADD: return op_arith('+', a, b, F);
        case VF_OP_SUB: return op_arith('-', a, b, F);
        case VF_OP_MUL: return op_arith('*', a, b, F);
        case VF_OP_DIV: return op_arith('/', a, b, F);
        case VF_OP_REM: return op_rem(a, b, F, false);
        case VF_OP_FMOD: return op_rem(a, b, F, true);
        case VF_OP_MIN: return op_minmax(a, b, F, false, false);
        case VF_OP_MAX: return op_minmax(a, b, F, true, false);
        case VF_OP_MINNUM: return op_minmax(a, b, F, false, true);
        case VF_OP_MAXNUM: return op_minmax(a, b, F, true, true);
        case VF_OP_SGNJ: return plain_out(op_sgnj(a, cb.sign, 0));
        case VF_OP_SGNJN: return plain_out(op_sgnj(a, cb.sign, 1));
        case VF_OP_SGNJX: return plain_out(op_sgnj(a, cb.sign, 2));
        case VF_OP_FMA: return op_fma(a, b, c, F);
        case VF_OP_SQRT: return op_sqrt(a);
        case VF_OP_NEG: return op_neg(a);
        case VF_OP_ABS: {
            Code r = ca;
            r.sign = false;
            return plain_out(r);
        }
        case VF_OP_NEXT_UP: return op_next(a, true);
        case VF_OP_NEXT_DOWN: return op_next(a, false);
        case VF_OP_LOGB: return op_logb(a);
    }
    fail(Err::VALUE, "unknown operation " + std::to_string(op));
}

int arity(int op) { return op == VF_OP_FMA ? 3 : op >= VF_OP_SQRT ? 1 : 2; }

uint64_t done(const OpOut& o, const Fmt& f, int* flags) {
    if (flags) *flags = o.r.flags;
    return raw64(o.r.c, f);
}

uint64_t op64(const vf_fmt* f, int op, uint64_t a, uint64_t b, uint64_t c, int* flags) {
    return guarded<uint64_t>(flags, [&] {
        const vf_fmt& h = checked(f);
        const int n = arity(op);
        const Code ca = code64(a, h), cb = n > 1 ? code64(b, h) : Code(), cc = n > 2 ? code64(c, h) : Code();
        return done(run(op, h.f, ca, cb, cc), h.f, flags);
    });
}

void scale_check(int64_t n) {
    if (n > (int64_t(1) << 62) || n < -(int64_t(1) << 62)) fail(Err::OVERFLOW, "scaleb exponent beyond 2**62");
}

int compare(const Fmt& F, const Code& ca, const Code& cb, int signaling, int* flags) {
    const Opnd a{ca, &F}, b{cb, &F};
    auto [rel, fl] = op_compare(a, b, signaling != 0);
    if (flags) *flags = fl;
    return rel;
}

uint64_t from_big(const vf_fmt* f, const BigInt& v, int* flags) {
    return guarded<uint64_t>(flags, [&] {
        const vf_fmt& h = checked(f);
        code64(0, h);   // the width check
        SRArg sr;
        const Dy<BigInt> d{v < 0, 0, abs(v), false};
        const RoundOut r = round_big(d, h.f, false, sr);
        if (flags) *flags = r.flags;
        return raw64(r.c, h.f);
    });
}

// ---------------------------------------------------------------- formats

std::mutex g_lock;
std::map<std::string, const vf_fmt*>& registry() {
    static std::map<std::string, const vf_fmt*> r;
    return r;
}

}  // namespace
}  // namespace vf

using namespace vf;

extern "C" {

const vf_fmt* vf_format(const char* name) {
    return guarded<const vf_fmt*>(nullptr, [&]() -> const vf_fmt* {
        if (!name) fail(Err::VALUE, "null format name");
        std::lock_guard<std::mutex> hold(g_lock);
        auto& reg = registry();
        const std::string key(name);
        if (auto it = reg.find(key); it != reg.end()) return it->second;
        Fmt f = fmt_parse(key);
        const std::string canonical = fmt_str(f);
        const vf_fmt* h;
        if (auto it = reg.find(canonical); it != reg.end()) h = it->second;
        else {
            f.id = (int64_t)reg.size();
            auto* made = new vf_fmt{f, canonical, (size_t)((f.size + 31) / 32)};   // lives until exit
            reg.emplace(canonical, made);
            h = made;
        }
        reg.emplace(key, h);
        return h;
    });
}

int vf_format_size(const vf_fmt* f) { return f ? (int)f->f.size : 0; }
const char* vf_format_name(const vf_fmt* f) { return f ? f->name.c_str() : ""; }
const char* vf_last_error(void) { return t_error.c_str(); }
const char* vf_version(void) { return VF_VERSION; }

uint64_t vf_op(const vf_fmt* f, int op, uint64_t a, uint64_t b, uint64_t c, int* flags) {
    return op64(f, op, a, b, c, flags);
}

#define VF_BIN(name, op) \
    uint64_t name(const vf_fmt* f, uint64_t a, uint64_t b, int* flags) { return op64(f, op, a, b, 0, flags); }
#define VF_UN(name, op) \
    uint64_t name(const vf_fmt* f, uint64_t a, int* flags) { return op64(f, op, a, 0, 0, flags); }
VF_BIN(vf_add, VF_OP_ADD)
VF_BIN(vf_sub, VF_OP_SUB)
VF_BIN(vf_mul, VF_OP_MUL)
VF_BIN(vf_div, VF_OP_DIV)
VF_BIN(vf_rem, VF_OP_REM)
VF_BIN(vf_fmod, VF_OP_FMOD)
VF_BIN(vf_min, VF_OP_MIN)
VF_BIN(vf_max, VF_OP_MAX)
VF_BIN(vf_minnum, VF_OP_MINNUM)
VF_BIN(vf_maxnum, VF_OP_MAXNUM)
VF_BIN(vf_sgnj, VF_OP_SGNJ)
VF_BIN(vf_sgnjn, VF_OP_SGNJN)
VF_BIN(vf_sgnjx, VF_OP_SGNJX)
VF_UN(vf_sqrt, VF_OP_SQRT)
VF_UN(vf_neg, VF_OP_NEG)
VF_UN(vf_abs, VF_OP_ABS)
VF_UN(vf_next_up, VF_OP_NEXT_UP)
VF_UN(vf_next_down, VF_OP_NEXT_DOWN)
VF_UN(vf_logb, VF_OP_LOGB)
#undef VF_BIN
#undef VF_UN

uint64_t vf_fma(const vf_fmt* f, uint64_t a, uint64_t b, uint64_t c, int* flags) {
    return op64(f, VF_OP_FMA, a, b, c, flags);
}

uint64_t vf_scaleb(const vf_fmt* f, uint64_t a, int64_t n, int* flags) {
    return guarded<uint64_t>(flags, [&] {
        const vf_fmt& h = checked(f);
        scale_check(n);
        return done(op_scaleb({code64(a, h), &h.f}, n), h.f, flags);
    });
}

int vf_fclass(const vf_fmt* f, uint64_t a) {
    return guarded<int>(nullptr, [&] {
        const vf_fmt& h = checked(f);
        return (int)op_fclass(code64(a, h), h.f);
    });
}

int vf_compare(const vf_fmt* f, uint64_t a, uint64_t b, int signaling, int* flags) {
    return guarded<int>(flags, [&] {
        const vf_fmt& h = checked(f);
        return compare(h.f, code64(a, h), code64(b, h), signaling, flags);
    });
}
int vf_eq(const vf_fmt* f, uint64_t a, uint64_t b, int signaling, int* flags) {
    return vf_compare(f, a, b, signaling, flags) == VF_EQ;
}
int vf_lt(const vf_fmt* f, uint64_t a, uint64_t b, int signaling, int* flags) {
    return vf_compare(f, a, b, signaling, flags) == VF_LT;
}
int vf_le(const vf_fmt* f, uint64_t a, uint64_t b, int signaling, int* flags) {
    const int r = vf_compare(f, a, b, signaling, flags);
    return r == VF_LT || r == VF_EQ;
}

uint64_t vf_convert(const vf_fmt* to, const vf_fmt* from, uint64_t a, int* flags) {
    return guarded<uint64_t>(flags, [&] {
        const vf_fmt &t = checked(to), &s = checked(from);
        code64(0, t);
        SRArg sr;
        return done(op_convert({code64(a, s), &s.f}, t.f, sr), t.f, flags);
    });
}

uint64_t vf_round_to_integral(const vf_fmt* f, uint64_t a, int rounding, int exact, int* flags) {
    return guarded<uint64_t>(flags, [&] {
        const vf_fmt& h = checked(f);
        SRArg sr;
        return done(op_round_to_integral({code64(a, h), &h.f}, rounding_of(rounding, h.f), exact != 0, sr), h.f,
                    flags);
    });
}

uint64_t vf_to_int(const vf_fmt* f, uint64_t a, int bits, int is_signed, int rounding, int exact, int* flags) {
    return guarded<uint64_t>(flags, [&] {
        const vf_fmt& h = checked(f);
        if (bits < 1 || bits > 64) fail(Err::VALUE, "vf_to_int: bits must be 1 to 64");
        SRArg sr;
        IntOut o = op_to_int({code64(a, h), &h.f}, bits, is_signed != 0, rounding_of(rounding, h.f), exact != 0, sr);
        if (flags) *flags = o.flags;
        if (o.v < 0) o.v += BigInt(1) << (unsigned)bits;   // two's complement
        return static_cast<uint64_t>(o.v);
    });
}

uint64_t vf_from_int(const vf_fmt* f, int64_t value, int* flags) { return from_big(f, BigInt(value), flags); }
uint64_t vf_from_uint(const vf_fmt* f, uint64_t value, int* flags) { return from_big(f, BigInt(value), flags); }

uint64_t vf_from_double(const vf_fmt* f, double value, int* flags) {
    return guarded<uint64_t>(flags, [&] {
        const vf_fmt& h = checked(f);
        code64(0, h);
        SRArg sr;
        return done(op_from_double(value, h.f, sr), h.f, flags);
    });
}

double vf_to_double(const vf_fmt* f, uint64_t a) {
    return guarded<double>(nullptr, [&] {
        const vf_fmt& h = checked(f);
        return fp_to_double(code64(a, h), h.f);
    });
}

void vf_set_sr(uint64_t bits) {
    g_sr_source = nullptr;
    g_sr_value = bits;
}
void vf_set_sr_source(uint64_t (*source)(int bits, void* ctx), void* ctx) {
    g_sr_source = source;
    g_sr_ctx = ctx;
}

// ---- any width

void vf_op_w(const vf_fmt* f, int op, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* result,
             int* flags) {
    guarded<int>(flags, [&] {
        const vf_fmt& h = checked(f);
        const int n = arity(op);
        const Code ca = code_w(a, h), cb = n > 1 ? code_w(b, h) : Code(), cc = n > 2 ? code_w(c, h) : Code();
        const OpOut o = run(op, h.f, ca, cb, cc);
        put_w(o.r.c, h, result);
        if (flags) *flags = o.r.flags;
        return 0;
    });
}

void vf_scaleb_w(const vf_fmt* f, const uint32_t* a, int64_t n, uint32_t* result, int* flags) {
    guarded<int>(flags, [&] {
        const vf_fmt& h = checked(f);
        scale_check(n);
        const OpOut o = op_scaleb({code_w(a, h), &h.f}, n);
        put_w(o.r.c, h, result);
        if (flags) *flags = o.r.flags;
        return 0;
    });
}

int vf_fclass_w(const vf_fmt* f, const uint32_t* a) {
    return guarded<int>(nullptr, [&] {
        const vf_fmt& h = checked(f);
        return (int)op_fclass(code_w(a, h), h.f);
    });
}

int vf_compare_w(const vf_fmt* f, const uint32_t* a, const uint32_t* b, int signaling, int* flags) {
    return guarded<int>(flags, [&] {
        const vf_fmt& h = checked(f);
        return compare(h.f, code_w(a, h), code_w(b, h), signaling, flags);
    });
}

void vf_convert_w(const vf_fmt* to, const vf_fmt* from, const uint32_t* a, uint32_t* result, int* flags) {
    guarded<int>(flags, [&] {
        const vf_fmt &t = checked(to), &s = checked(from);
        SRArg sr;
        const OpOut o = op_convert({code_w(a, s), &s.f}, t.f, sr);
        put_w(o.r.c, t, result);
        if (flags) *flags = o.r.flags;
        return 0;
    });
}

void vf_round_to_integral_w(const vf_fmt* f, const uint32_t* a, int rounding, int exact, uint32_t* result,
                            int* flags) {
    guarded<int>(flags, [&] {
        const vf_fmt& h = checked(f);
        SRArg sr;
        const OpOut o = op_round_to_integral({code_w(a, h), &h.f}, rounding_of(rounding, h.f), exact != 0, sr);
        put_w(o.r.c, h, result);
        if (flags) *flags = o.r.flags;
        return 0;
    });
}

// ---- 128-bit vectors: the whole result is written, zero above the format

static bool fits128(std::initializer_list<const vf_fmt*> fmts, uint32_t* result, int* flags) {
    if (result) result[0] = result[1] = result[2] = result[3] = 0;
    return guarded<int>(flags, [&] {
        for (const vf_fmt* f : fmts) {
            const vf_fmt& h = checked(f);
            if (h.f.size > 128) fail(Err::VALUE, h.name + " is wider than 128 bits: use the _w functions");
        }
        return 1;
    }) != 0;
}

void vf_op128(const vf_fmt* f, int op, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* result,
              int* flags) {
    if (fits128({f}, result, flags)) vf_op_w(f, op, a, b, c, result, flags);
}

void vf_scaleb128(const vf_fmt* f, const uint32_t* a, int64_t n, uint32_t* result, int* flags) {
    if (fits128({f}, result, flags)) vf_scaleb_w(f, a, n, result, flags);
}

int vf_fclass128(const vf_fmt* f, const uint32_t* a) {
    return fits128({f}, nullptr, nullptr) ? vf_fclass_w(f, a) : 0;
}

int vf_compare128(const vf_fmt* f, const uint32_t* a, const uint32_t* b, int signaling, int* flags) {
    return fits128({f}, nullptr, flags) ? vf_compare_w(f, a, b, signaling, flags) : 0;
}

void vf_convert128(const vf_fmt* to, const vf_fmt* from, const uint32_t* a, uint32_t* result, int* flags) {
    if (fits128({to, from}, result, flags)) vf_convert_w(to, from, a, result, flags);
}

void vf_round_to_integral128(const vf_fmt* f, const uint32_t* a, int rounding, int exact, uint32_t* result,
                             int* flags) {
    if (fits128({f}, result, flags)) vf_round_to_integral_w(f, a, rounding, exact, result, flags);
}

}  // extern "C"
