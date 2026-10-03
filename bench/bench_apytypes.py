"""Throughput of verifloat against APyTypes (C++ core, NumPy-like arrays).

The scalar rows time one operation on one value. The array rows use FPArray
against APyFloatArray; the last row runs matmul on plain lists of FP values,
the form a cocotb scoreboard usually has. Every case first checks that both
libraries produce the same bits. Run: python bench/bench_apytypes.py
"""

from __future__ import annotations

import argparse
import random
import time
import warnings

import apytypes as apy

import verifloat as vf


def rate(fn, budget: float) -> float:
    """Calls per second of fn() (best of 3 runs of about `budget` seconds)."""
    t = time.perf_counter(); fn(); one = max(time.perf_counter() - t, 1e-7)
    n = max(1, int(budget / one))
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, time.perf_counter() - t)
    return n / best


def cases(rng):
    N = 4096
    vals = [rng.uniform(-100, 100) for _ in range(N)]
    h = [vf.FP16(v) for v in vals]
    g = [vf.FP16(v) for v in reversed(vals)]
    ah = [apy.APyFloat.from_float(v, 5, 10) for v in vals]
    ag = [apy.APyFloat.from_float(v, 5, 10) for v in reversed(vals)]
    s32 = [vf.FP32(v) for v in vals]
    a32 = [apy.APyFloat.from_float(v, 8, 23) for v in vals]
    assert [x.raw for x in h] == [a.to_bits() for a in ah]

    # Scalars: (name, verifloat fn, apytypes fn, elements per call)
    i = [0]

    def nxt():
        i[0] = (i[0] + 1) & (N - 1)
        return i[0]
    yield "FP16 a + b", lambda: h[nxt()] + g[i[0]], lambda: ah[nxt()] + ag[i[0]], 1
    yield "FP16 a * b", lambda: h[nxt()] * g[i[0]], lambda: ah[nxt()] * ag[i[0]], 1
    yield "FP16 a / b", lambda: h[nxt()] / g[i[0]], lambda: ah[nxt()] / ag[i[0]], 1
    yield "FP16 from float", (lambda: vf.FP16(vals[nxt()])), (lambda: apy.APyFloat.from_float(vals[nxt()], 5, 10)), 1
    yield "FP32 -> FP16", (lambda: s32[nxt()].convert(vf.FP16)), (lambda: a32[nxt()].cast(5, 10)), 1
    fx = vf.FixedFormat(8, 8)
    xf = [fx(v / 2) for v in vals]
    xa = [apy.APyFixed(x.raw, int_bits=8, frac_bits=8) for x in xf]
    yield "FIXED s8.8 a * b", lambda: xf[nxt()] * xf[(i[0] + 1) & (N - 1)], \
        lambda: xa[nxt()] * xa[(i[0] + 1) & (N - 1)], 1

    # Arrays: FPArray vs APyFloatArray
    A, B = apy.APyFloatArray.from_float(vals, 5, 10), apy.APyFloatArray.from_float(vals[::-1], 5, 10)
    VA, VB = vf.FP16.array(vals), vf.FP16.array(vals[::-1])
    assert VA.raw == A.to_bits()
    for sym, op in (("+", lambda x, y: x + y), ("*", lambda x, y: x * y), ("/", lambda x, y: x / y)):
        assert op(VA, VB).raw == op(A, B).to_bits()
        yield f"FP16 array a {sym} b, 4096 elements", (lambda op=op: op(VA, VB)), (lambda op=op: op(A, B)), N
    yield "FP16 array from 4096 floats", lambda: vf.FP16.array(vals), \
        lambda: apy.APyFloatArray.from_float(vals, 5, 10), N

    m = [[rng.uniform(-2, 2) for _ in range(64)] for _ in range(64)]
    k = [[rng.uniform(-2, 2) for _ in range(16)] for _ in range(64)]
    vm, vk = vf.FP32.array(m), vf.FP32.array(k)
    lm, lk = vm.tolist(), vk.tolist()
    am, ak = apy.APyFloatArray.from_float(m, 8, 23), apy.APyFloatArray.from_float(k, 8, 23)
    acc = vf.Accumulator(vf.FP32, "sequential", product=vf.FP32)   # APyTypes' matmul
    assert vf.matmul(vm, vk, acc=acc).raw == (am @ ak).to_bits()
    assert [[x.raw for x in r] for r in vf.matmul(lm, lk, acc=acc)] == (am @ ak).to_bits()
    yield "FP32 array matmul 64x64 @ 64x16", lambda: vf.matmul(vm, vk, acc=acc), lambda: am @ ak, 64 * 64 * 16
    yield "FP32 matmul on lists of FP", lambda: vf.matmul(lm, lk, acc=acc), lambda: am @ ak, 64 * 64 * 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    budget = 0.05 if args.quick else 0.3
    if not args.markdown:
        print(f"{'case':36} {'verifloat':>16} {'apytypes':>16} {'vf/apy':>8}")
    for name, fv, fa, unit in cases(random.Random(1)):
        rv, ra = rate(fv, budget) * unit, rate(fa, budget) * unit
        if args.markdown:
            print(f"| {name} | {rv:,.0f}/s | {ra:,.0f}/s | {rv / ra:.2f}× |")
        else:
            print(f"{name:36} {rv:14,.0f}/s {ra:14,.0f}/s {rv / ra:7.2f}x")


if __name__ == "__main__":
    main()
