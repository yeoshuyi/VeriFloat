// SIMD kernels, written once against a small vector interface and included by
// simd_avx2.cpp and simd_avx512.cpp, each of which defines that interface:
//
//   V  a vector of W unsigned 64-bit lanes        M  a per-lane mask
//
//   V::set1, V::load, store, + - & | ^
//   shlv/shrv(v, counts)   shift each lane by its own count (>= 64 gives 0)
//   mul32(a, b)            low 32 bits x low 32 bits -> 64
//   abs64(v)               |v| as a signed value
//   inc_if(v, m)           v + 1 in the lanes of m
//   addsub(v, x, m)        v - x in the lanes of m, v + x elsewhere
//   eq, gt, ge (signed), gt_half (unsigned > 2**63), eq0, nz -> M
//   kand, kor, kxor, kandn(a, b) = ~a & b, knot, ksel(c, a, b) = c ? a : b
//   flip_if(m, bit)        m, complemented when bit is 1
//   select(m, a, b)        m ? a : b per lane
//   one_if(m)              1 in the lanes of m, else 0
//   bits(m), M::from_bits  the mask as an integer, lane 0 in bit 0
//   lznorm(s, lz)          s shifted left until its leading bit is at 63
//
// Everything is integer arithmetic. Each kernel mirrors a scalar kernel step
// for step (lean.hpp for dot products, fast.hpp for single operations) and
// the tests require identical results from both. Constants are broadcast once
// per call (the structs below), not once per step.

struct Cst {
    V zero, one, half;
};
static inline Cst cst() { return {V::set1(0), V::set1(1), V::set1(uint64_t(1) << 63)}; }

struct RoundK {   // constants for rounding into one format at a fixed position
    V emin, emax, fn, k, k64, m1, up;
    bool has_fn;
    uint8_t rounding;
};
// Rounding of a value with its leading bit at `lead` to M + 1 bits, to be
// stored with the leading bit at 62.
static inline RoundK round_k(const RT& f, int64_t lead) {
    RoundK c;
    c.emin = V::set1((uint64_t)f.emin);
    c.emax = V::set1((uint64_t)f.emax);
    c.fn = V::set1(f.fn_mask);
    c.k = V::set1((uint64_t)(lead - f.M));          // discarded bits
    c.k64 = V::set1((uint64_t)(64 - (lead - f.M)));
    c.m1 = V::set1((uint64_t)(f.M + 1));
    c.up = V::set1((uint64_t)(62 - f.M));
    c.has_fn = f.fn_mask != 0;
    c.rounding = f.rounding;
    return c;
}

// Round up? `low` is the discarded fraction, left-justified (bit 63 = half).
static inline M round_up(V low, V s, M neg, uint8_t rounding, const Cst& c) {
    switch (rounding) {
        case kRNE: return gt_half(low | (s & c.one), c.half);
        case kRNA: return gt(c.zero, low);   // the half bit is set
        case kRTZ: return M::none();
        case kRUP: return kandn(neg, nz(low));
        default: return kand(neg, nz(low));   // RDN
    }
}

// lean_out_of_range: e0 is the leading-bit exponent before rounding and e
// after it, s the rounded significand.
static inline M out_of_range(V e0, V e, V s, const RoundK& f) {
    M r = kor(gt(f.emin, e0), gt(e, f.emax));
    if (f.has_fn) r = kor(r, kand(eq(e, f.emax), eq(s & f.fn, f.fn)));
    return r;
}

// a + b for two values with their leading bits at 62: A * 2**ae and T * 2**te
// (lean_add); a zero A is allowed. Returns the sum normalized to bit 63, with
// `lost` set where bits were shifted out of the smaller operand and `sz`
// where the sum is exactly zero.
struct Sum {
    V N, lz, be;
    M bn, sz, lost;
};
static inline Sum add_aligned(V A, V ae, M an, V T, V te, M tn, const Cst& c) {
    const V d = ae - te;
    // |acc| >= |term|? (never for a zero accumulator)
    const M ge_ = kandn(eq0(A), kor(gt(d, c.zero), kand(eq0(d), ge(A, T))));
    const V big = select(ge_, A, T), small = select(ge_, T, A);
    Sum r;
    r.be = select(ge_, ae, te);
    r.bn = ksel(ge_, an, tn);
    const V dd = abs64(d);
    const V sm = shrv(small, dd);
    r.lost = knot(eq(shlv(sm, dd), small));
    const V S = select(kxor(an, tn), big - sm - one_if(r.lost), big + sm);
    r.sz = eq0(S);
    r.N = lznorm(S, r.lz);
    return r;
}

// ---------------------------------------------------------------- dot products

struct SeqK {
    Cst c;
    RoundK P, F;
    V c2, c60, c61, c62, c63;
    M rdn;        // all lanes set when F rounds toward negative
    bool has_p;
};
static inline SeqK seq_k(const LeanSpec& sp) {
    SeqK K;
    K.c = cst();
    K.P = round_k(sp.P, 60);   // a product has its leading bit at 60 or 61
    K.F = round_k(sp.F, 63);   // a normalized sum has it at 63
    K.c2 = V::set1(2);
    K.c60 = V::set1(60);
    K.c61 = V::set1(61);
    K.c62 = V::set1(62);
    K.c63 = V::set1(63);
    K.rdn = sp.F.rounding == kRDN ? M::all() : M::none();
    K.has_p = sp.has_p;
    return K;
}

// One accumulator per lane. inexv and badv are nonzero in the set lanes.
struct Acc {
    V A, ae, inexv, badv;
    M an;
};

// A product x * y, rounded by P when there is one (lean_term): T with its
// leading bit at 62 (0 for a zero product), exponent te, sign tn. `lowv` is
// nonzero where the rounding was inexact and `bad` set where it left the
// kernel's range; both are meaningless in the lanes of a zero product.
struct Term {
    V T, te, lowv;
    M tn, tz, bad;
};
VF_SIMD_INLINE Term vterm(const SeqK& K, V xs, V xe, uint64_t xneg, V ys, V ye, M yn) {
    Term t;
    const V p = mul32(xs, ys);   // 0, or leading bit at 60 or 61
    const V pe = xe + ye;
    t.tn = flip_if(yn, xneg);
    t.tz = eq0(p);
    const V top = shrv(p, K.c61);
    if (K.has_p) {
        const RoundK& P = K.P;
        V s = shrv(p, P.k + top);
        t.lowv = shlv(p, P.k64 - top);
        s = inc_if(s, round_up(t.lowv, s, t.tn, P.rounding, K.c));
        const V cy = shrv(s, P.m1);
        s = shrv(s, cy);
        const V e0 = pe + K.c60 + top, e = e0 + cy;
        t.bad = out_of_range(e0, e, s, P);
        t.T = shlv(s, P.up);
        t.te = e - K.c62;
    } else {
        const V sh = K.c2 - top;
        t.lowv = K.c.zero;
        t.bad = M::none();
        t.T = shlv(p, sh);   // stays 0 for a zero product
        t.te = pe - sh;
    }
    return t;
}

// acc = round_F(acc + T * 2**te), as lean_add / lean_add_zero: tz marks zero
// terms, extra_low/extra_bad are the term's own rounding results.
VF_SIMD_INLINE void vadd(Acc& acc, const SeqK& K, V T, V te, M tn, M tz, V extra_low, M extra_bad) {
    const Cst& c = K.c;
    const RoundK& F = K.F;
    const M az = eq0(acc.A);
    const Sum u = add_aligned(acc.A, acc.ae, acc.an, T, te, tn, c);
    V s = shrv(u.N, F.k);
    const V low = shlv(u.N, F.k64) | one_if(u.lost);
    s = inc_if(s, round_up(low, s, u.bn, F.rounding, c));
    const V cy = shrv(s, F.m1);
    s = shrv(s, cy);
    const V e0 = u.be + K.c63 - u.lz, e = e0 + cy;
    const M abad = out_of_range(e0, e, s, F);
    // A zero term keeps the accumulator (two zeros of opposite sign give the
    // rounding mode's zero); an exactly zero sum gives that zero; otherwise
    // the rounded sum.
    const M zero_sum = kandn(tz, u.sz);          // nonzero term, sum exactly zero
    const M keep = tz;
    const M to_zero_sign = kor(zero_sum, kand(kand(tz, az), kxor(acc.an, tn)));
    const V rounded = shlv(s, F.up);
    Acc nx;
    nx.A = select(keep, acc.A, select(zero_sum, c.zero, rounded));
    nx.ae = select(keep, acc.ae, select(zero_sum, c.zero, e - K.c62));
    nx.an = ksel(to_zero_sign, K.rdn, ksel(keep, acc.an, u.bn));
    nx.inexv = acc.inexv | select(keep, c.zero, extra_low | select(zero_sum, c.zero, low));
    nx.badv = acc.badv | one_if(kandn(keep, kor(extra_bad, kandn(zero_sum, abad))));
    acc = nx;
}

static inline void acc_out(const Acc& acc, size_t lanes, LeanState* st) {
    alignas(64) uint64_t fA[W], fe[W];
    acc.A.store(fA);
    acc.ae.store(fe);
    const unsigned fan = bits(acc.an), fin = bits(nz(acc.inexv)), fbad = bits(nz(acc.badv));
    for (unsigned l = 0; l < lanes; ++l) {
        st[l].A = fA[l];
        st[l].ae = (int64_t)fe[l];
        st[l].an = (fan >> l) & 1;
        st[l].inexact = (fin >> l) & 1;
        st[l].bad = (fbad >> l) & 1;
    }
}

// Sequential: two rows at a time (each sum is one dependency chain, and two
// independent chains overlap in the CPU; they also share the loads of b).
static void seq_lean_simd(const LeanVec* a, size_t rows, const LeanVec& b, size_t stride, size_t k, size_t lanes,
                          const LeanSpec& sp, LeanState* st) {
    const SeqK K = seq_k(sp);
    const Acc init{K.c.zero, K.c.zero, K.c.zero, K.c.zero, M::none()};
    Acc s0 = init, s1 = init;
    const LeanVec a0 = a[0], a1 = a[rows == 2 ? 1 : 0];
    if (rows == 2) {
        for (size_t i = 0; i < k; ++i) {
            const size_t o = i * stride;
            const V ys = V::load(b.sig + o), ye = V::load((const uint64_t*)(b.exp + o));
            const M yn = nz(V::load(b.neg + o));
            const Term t0 = vterm(K, V::set1(a0.sig[i]), V::set1((uint64_t)a0.exp[i]), a0.neg[i], ys, ye, yn);
            const Term t1 = vterm(K, V::set1(a1.sig[i]), V::set1((uint64_t)a1.exp[i]), a1.neg[i], ys, ye, yn);
            vadd(s0, K, t0.T, t0.te, t0.tn, t0.tz, t0.lowv, t0.bad);
            vadd(s1, K, t1.T, t1.te, t1.tn, t1.tz, t1.lowv, t1.bad);
        }
        acc_out(s1, lanes, st + W);
    } else {
        for (size_t i = 0; i < k; ++i) {
            const size_t o = i * stride;
            const V ys = V::load(b.sig + o), ye = V::load((const uint64_t*)(b.exp + o));
            const M yn = nz(V::load(b.neg + o));
            const Term t0 = vterm(K, V::set1(a0.sig[i]), V::set1((uint64_t)a0.exp[i]), a0.neg[i], ys, ye, yn);
            vadd(s0, K, t0.T, t0.te, t0.tn, t0.tz, t0.lowv, t0.bad);
        }
    }
    acc_out(s0, lanes, st);
}

// Pairwise: every term rounded into F, then a balanced tree of rounded
// additions, level by level. scratch holds A, ae and sign (0/1) per term.
static void pair_lean_simd(const LeanVec& a, const LeanVec& b, size_t stride, size_t k, size_t lanes,
                           const LeanSpec& sp, uint64_t* scratch, LeanState* st) {
    const SeqK K = seq_k(sp);
    const Acc zero{K.c.zero, K.c.zero, K.c.zero, K.c.zero, M::none()};
    uint64_t* const sA = scratch;
    uint64_t* const sE = scratch + (k ? k : 1) * W;
    uint64_t* const sN = scratch + 2 * (k ? k : 1) * W;
    V inexv = K.c.zero, badv = K.c.zero;
    K.c.zero.store(sA);
    K.c.zero.store(sE);
    K.c.zero.store(sN);
    for (size_t i = 0; i < k; ++i) {
        const size_t o = i * stride;
        const V ys = V::load(b.sig + o), ye = V::load((const uint64_t*)(b.exp + o));
        const M yn = nz(V::load(b.neg + o));
        const Term t = vterm(K, V::set1(a.sig[i]), V::set1((uint64_t)a.exp[i]), a.neg[i], ys, ye, yn);
        Acc x = zero;
        x.an = t.tn;   // 0 + (a zero term) keeps the term's sign
        vadd(x, K, t.T, t.te, t.tn, t.tz, t.lowv, t.bad);
        inexv = inexv | x.inexv;
        badv = badv | x.badv;
        x.A.store(sA + i * W);
        x.ae.store(sE + i * W);
        one_if(x.an).store(sN + i * W);
    }
    size_t len = k ? k : 1;
    while (len > 1) {
        size_t o = 0;
        for (size_t i = 0; i + 1 < len; i += 2, ++o) {
            Acc x{V::load(sA + i * W), V::load(sE + i * W), K.c.zero, K.c.zero, nz(V::load(sN + i * W))};
            const V T = V::load(sA + (i + 1) * W);
            vadd(x, K, T, V::load(sE + (i + 1) * W), nz(V::load(sN + (i + 1) * W)), eq0(T), K.c.zero, M::none());
            inexv = inexv | x.inexv;
            badv = badv | x.badv;
            x.A.store(sA + o * W);
            x.ae.store(sE + o * W);
            one_if(x.an).store(sN + o * W);
        }
        if (len % 2) {
            V::load(sA + (len - 1) * W).store(sA + o * W);
            V::load(sE + (len - 1) * W).store(sE + o * W);
            V::load(sN + (len - 1) * W).store(sN + o * W);
            ++o;
        }
        len = o;
    }
    const Acc r{V::load(sA), V::load(sE), inexv, badv, nz(V::load(sN))};
    acc_out(r, lanes, st);
}

// Exact: each term shifted onto its lane's grid and added, with its sign,
// into four 32-bit limbs held in 64-bit lanes: no carries inside the loop.
static void exact_lean_simd(const LeanVec& a, const LeanVec& b, size_t stride, size_t k, size_t lanes,
                            const LeanSpec& sp, const int64_t* base, ExactSum* out) {
    const SeqK K = seq_k(sp);
    const RoundK& P = K.P;
    alignas(64) uint64_t tb[W] = {};
    for (size_t l = 0; l < lanes; ++l) tb[l] = (uint64_t)base[l];
    const V basev = V::load(tb), c64 = V::set1(64), c32 = V::set1(32), m32 = V::set1(0xffffffff);
    V l0 = K.c.zero, l1 = K.c.zero, l2 = K.c.zero, l3 = K.c.zero, inexv = K.c.zero, badv = K.c.zero;
    for (size_t i = 0; i < k; ++i) {
        const size_t o = i * stride;
        const V ys = V::load(b.sig + o), ye = V::load((const uint64_t*)(b.exp + o));
        const M neg = flip_if(nz(V::load(b.neg + o)), a.neg[i]);
        const V p = mul32(V::set1(a.sig[i]), ys);   // 0, or leading bit at 60 or 61
        const V pe = V::set1((uint64_t)a.exp[i]) + ye;
        V v = p, sh = pe - basev;
        if (K.has_p) {
            const M pz = eq0(p);
            const V top = shrv(p, K.c61);
            V s = shrv(p, P.k + top);
            const V low = shlv(p, P.k64 - top);
            s = inc_if(s, round_up(low, s, neg, P.rounding, K.c));
            const V cy = shrv(s, P.m1);
            const V e0 = pe + K.c60 + top;
            inexv = inexv | low;   // a zero product has low == 0
            badv = badv | one_if(kandn(pz, out_of_range(e0, e0 + cy, shrv(s, cy), P)));
            v = s;                 // s * 2**(pe + 60 - P.M + top), carry included
            sh = sh + top;
        }
        // v * 2**sh as 128 bits (a zero v gives zero whatever its shift is)
        const V lo = shlv(v, sh), hi = shrv(v, c64 - sh);
        l0 = addsub(l0, lo & m32, neg);
        l1 = addsub(l1, shrv(lo, c32), neg);
        l2 = addsub(l2, hi & m32, neg);
        l3 = addsub(l3, shrv(hi, c32), neg);
    }
    alignas(64) uint64_t r0[W], r1[W], r2[W], r3[W];
    l0.store(r0);
    l1.store(r1);
    l2.store(r2);
    l3.store(r3);
    const unsigned fin = bits(nz(inexv)), fbad = bits(nz(badv));
    for (unsigned l = 0; l < lanes; ++l) {
        out[l].limb[0] = (int64_t)r0[l];
        out[l].limb[1] = (int64_t)r1[l];
        out[l].limb[2] = (int64_t)r2[l];
        out[l].limb[3] = (int64_t)r3[l];
        out[l].inexact = (fin >> l) & 1;
        out[l].bad = (fbad >> l) & 1;
    }
}

// ---------------------------------------------------------------- element-wise

struct EwK {   // one format's geometry, broadcast
    V M, top, mmask, signbit, magmask, lead, bias, maxf, fn, m1, k, k64, up, c64;
    bool has_special, ftz, has_fn;
    uint8_t rounding;
};
static inline EwK ew_k(const EwFmt& f) {
    EwK c;
    const uint64_t signbit = f.sign_shift < 64 ? uint64_t(1) << f.sign_shift : 0;
    c.M = V::set1((uint64_t)f.M);
    c.top = V::set1(f.top);
    c.mmask = V::set1(f.mmask);
    c.signbit = V::set1(signbit);
    c.magmask = V::set1(signbit - 1);
    c.lead = V::set1(uint64_t(1) << f.M);
    c.bias = V::set1((uint64_t)f.bias);
    c.maxf = V::set1(f.max_field);
    c.fn = V::set1(f.fn_mask);
    c.m1 = V::set1((uint64_t)(f.M + 1));
    c.k = V::set1((uint64_t)(63 - f.M));     // rounding a value with its leading bit at 63
    c.k64 = V::set1((uint64_t)(f.M + 1));
    c.up = V::set1((uint64_t)(62 - f.M));
    c.c64 = V::set1(64);
    c.has_special = f.has_special;
    c.ftz = f.ftz;
    c.has_fn = f.fn_mask != 0;
    c.rounding = f.rounding;
    return c;
}

struct Dec {
    V field, mant;
    M neg, normal, zero, finite;
};
static inline Dec decode(V raw, const EwK& f) {
    Dec d;
    d.field = shrv(raw, f.M) & f.top;
    d.mant = raw & f.mmask;
    d.neg = nz(raw & f.signbit);
    d.finite = f.has_special ? knot(eq(d.field, f.top)) : M::all();
    d.normal = kandn(eq0(d.field), d.finite);
    d.zero = f.ftz ? eq0(d.field) : eq0(d.field | d.mant);
    return d;
}

// Is a result with exponent field e0 before rounding and e after it, and
// rounded significand s, outside what the kernel may return?
static inline M field_out_of_range(V e0, V e, V s, const EwK& f, const Cst& c) {
    M r = kor(gt(c.one, e0), gt(e, f.maxf));
    if (f.has_fn) r = kor(r, kand(eq(e, f.maxf), eq(s & f.fn, f.fn)));
    return r;
}

static inline V encode(M neg, V field, V s, const EwK& f, const Cst& c) {
    return select(neg, f.signbit, c.zero) | shlv(field, f.M) | (s & f.mmask);
}

// Bit l of bits_ as byte l (0 or 1), W bytes.
static inline void put_bytes(uint8_t* dst, unsigned bits_) {
    // spread the bits to one per byte: multiply places a copy of the bits at
    // each byte, the mask keeps bit l of copy l
    const uint64_t spread = ((uint64_t)(bits_ & 0xff) * 0x0101010101010101ULL) & 0x8040201008040201ULL;
    const uint64_t bytes = ((spread + 0x7f7f7f7f7f7f7f7fULL) >> 7) & 0x0101010101010101ULL;
    __builtin_memcpy(dst, &bytes, W);
}

// One vector of a op b; returns the lanes to redo.
static inline unsigned ew_bin_step(char op, V ra, V rb, const EwK& f, const Cst& c, uint64_t* out, uint8_t* flags) {
    const Dec a = decode(ra, f);
    Dec b = decode(rb, f);
    V res;
    M done, inexact;
    if (op == '*') {
        const V p = mul32(a.mant | f.lead, b.mant | f.lead);   // leading bit at 2M or 2M + 1
        const M neg = kxor(a.neg, b.neg);
        const V top = shrv(p, f.M + f.m1);
        const V kv = f.M + top;
        V s = shrv(p, kv);
        const V low = shlv(p, f.c64 - kv);
        s = inc_if(s, round_up(low, s, neg, f.rounding, c));
        const V cy = shrv(s, f.m1);
        s = shrv(s, cy);
        const V e0 = a.field + b.field - f.bias + top, e = e0 + cy;
        const M ok = kandn(field_out_of_range(e0, e, s, f, c), kand(a.normal, b.normal));
        const M zr = kor(kand(a.zero, b.finite), kand(a.finite, b.zero));   // a zero times a finite value
        res = select(zr, select(neg, f.signbit, c.zero), encode(neg, e, s, f, c));
        inexact = kandn(zr, nz(low));
        done = kor(ok, zr);
    } else {
        if (op == '-') b.neg = knot(b.neg);
        const Sum u = add_aligned(shlv(a.mant | f.lead, f.up), a.field, a.neg, shlv(b.mant | f.lead, f.up), b.field,
                                  b.neg, c);
        V s = shrv(u.N, f.k);
        const V low = shlv(u.N, f.k64) | one_if(u.lost);
        s = inc_if(s, round_up(low, s, u.bn, f.rounding, c));
        const V cy = shrv(s, f.m1);
        s = shrv(s, cy);
        // the larger operand's leading bit is at 62, the sum's at 63 - lz
        const V e0 = u.be + c.one - u.lz, e = e0 + cy;
        const M ok = kandn(kor(u.sz, field_out_of_range(e0, e, s, f, c)), kand(a.normal, b.normal));
        // x + 0 is x and 0 + y is y (0 + 0 depends on the rounding mode: redone)
        const M bz = kand(a.normal, b.zero), az = kand(a.zero, b.normal);
        const V rb_signed = (rb & f.magmask) | select(b.neg, f.signbit, c.zero);
        res = select(bz, ra, select(az, rb_signed, encode(u.bn, e, s, f, c)));
        inexact = kand(ok, nz(low));
        done = kor(ok, kor(bz, az));
    }
    res.store(out);
    put_bytes(flags, bits(inexact));
    return ~bits(done) & ((1u << W) - 1);
}

static void ew_bin_simd(char op, const uint64_t* a, size_t sa, const uint64_t* b, size_t sb, size_t n, const EwFmt& fmt,
                        uint64_t* out, uint8_t* flags, uint8_t* redo) {
    const EwK f = ew_k(fmt);
    const Cst c = cst();
    size_t i = 0;
    for (; i + W <= n; i += W) {
        const V ra = sa ? V::load(a + i) : V::set1(a[0]);
        const V rb = sb ? V::load(b + i) : V::set1(b[0]);
        redo[i / W] = (uint8_t)ew_bin_step(op, ra, rb, f, c, out + i, flags + i);
    }
    if (i < n) {   // the last, partial vector through padded copies
        alignas(64) uint64_t ta[W] = {}, tb[W] = {}, to[W];
        uint8_t tf[W];
        for (size_t l = 0; i + l < n; ++l) {
            ta[l] = a[(i + l) * sa];
            tb[l] = b[(i + l) * sb];
        }
        redo[i / W] = (uint8_t)(ew_bin_step(op, V::load(ta), V::load(tb), f, c, to, tf) & ((1u << (n - i)) - 1));
        for (size_t l = 0; i + l < n; ++l) {
            out[i + l] = to[l];
            flags[i + l] = tf[l];
        }
    }
}

// One vector of codes of format s rounded into format f.
static inline unsigned ew_conv_step(V r, const EwK& s, const EwK& f, const Cst& c, V norm, V rebias, uint64_t* out,
                                    uint8_t* flags) {
    const Dec x = decode(r, s);
    const V N = shlv(x.mant | s.lead, norm);   // leading bit at 63
    V q = shrv(N, f.k);
    const V low = shlv(N, f.k64);
    q = inc_if(q, round_up(low, q, x.neg, f.rounding, c));
    const V cy = shrv(q, f.m1);
    q = shrv(q, cy);
    const V e0 = x.field + rebias, e = e0 + cy;
    const M ok = kandn(field_out_of_range(e0, e, q, f, c), x.normal);
    const V res = select(x.zero, select(x.neg, f.signbit, c.zero), encode(x.neg, e, q, f, c));
    res.store(out);
    put_bytes(flags, bits(kand(ok, nz(low))));
    return ~bits(kor(ok, x.zero)) & ((1u << W) - 1);
}

static void ew_conv_simd(const uint64_t* src, size_t n, const EwFmt& sfmt, const EwFmt& ffmt, uint64_t* out,
                         uint8_t* flags, uint8_t* redo) {
    const EwK s = ew_k(sfmt), f = ew_k(ffmt);
    const Cst c = cst();
    const V norm = V::set1((uint64_t)(63 - sfmt.M));
    const V rebias = V::set1((uint64_t)(ffmt.bias - sfmt.bias));
    size_t i = 0;
    for (; i + W <= n; i += W)
        redo[i / W] = (uint8_t)ew_conv_step(V::load(src + i), s, f, c, norm, rebias, out + i, flags + i);
    if (i < n) {
        alignas(64) uint64_t ts[W] = {}, to[W];
        uint8_t tf[W];
        for (size_t l = 0; i + l < n; ++l) ts[l] = src[i + l];
        redo[i / W] = (uint8_t)(ew_conv_step(V::load(ts), s, f, c, norm, rebias, to, tf) & ((1u << (n - i)) - 1));
        for (size_t l = 0; i + l < n; ++l) {
            out[i + l] = to[l];
            flags[i + l] = tf[l];
        }
    }
}
