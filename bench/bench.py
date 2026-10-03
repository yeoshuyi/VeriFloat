"""Throughput of the C++-backed verifloat against the frozen pure-Python
version (verifloat_py). Run: python bench/bench.py [--quick]"""

from __future__ import annotations

import argparse
import random
import time
import warnings

import verifloat as cpp
import verifloat_py as py


def rate(fn, n: int) -> float:
    """Calls per second of fn(i) over n iterations (best of 3)."""
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        for i in range(n):
            fn(i)
        best = min(best, time.perf_counter() - t)
    return n / best


def cases(m, rng):
    vals = [rng.uniform(-100, 100) for _ in range(256)]
    f16 = [m.FP16(v) for v in vals]
    f32 = [m.FP32(v) for v in vals]
    e4 = [m.E4M3(v / 4) for v in vals]
    u8 = [m.UINT(rng.randrange(256), 8) for _ in range(256)]
    i16 = [m.INT(rng.randrange(-30000, 30000), 16) for _ in range(256)]
    fx = [m.FixedFormat(8, 8)(v / 2) for v in vals]
    sr = m.FPFormat(5, 10, rounding=m.Rounding.SR)
    srv = [sr(v) for v in vals]
    k = 255
    yield "FP16 from float", lambda i: m.FP16(vals[i & k]), 1
    yield "FP16 a + b", lambda i: f16[i & k] + f16[(i + 1) & k], 1
    yield "FP16 a * b", lambda i: f16[i & k] * f16[(i + 7) & k], 1
    yield "FP16 a / b", lambda i: f16[i & k] / f16[(i + 3) & k], 1
    yield "FP32 fma", lambda i: f32[i & k].fma(f32[(i + 1) & k], f32[(i + 2) & k]), 1
    yield "FP32 sqrt", lambda i: abs(f32[i & k]).sqrt(), 1
    yield "E4M3 a * b", lambda i: e4[i & k] * e4[(i + 5) & k], 1
    yield "FP16 SR a + b", lambda i: srv[i & k] + srv[(i + 1) & k], 1
    yield "FP16 raw/flags", lambda i: (f16[i & k].raw, f16[i & k].flags), 1
    yield "FP16 a < b", lambda i: f16[i & k] < f16[(i + 1) & k], 1
    yield "FP32 -> FP16", lambda i: f32[i & k].convert(m.FP16), 1
    yield "UINT8 a + b", lambda i: u8[i & k] + u8[(i + 1) & k], 1
    yield "INT16 a * b", lambda i: i16[i & k] * i16[(i + 1) & k], 1
    yield "FIXED a * b", lambda i: fx[i & k] * fx[(i + 1) & k], 1
    a = [[rng.uniform(-3, 3) for _ in range(64)] for _ in range(64)]
    b = [[rng.uniform(-3, 3) for _ in range(64)] for _ in range(64)]
    ta = m.BlockTensor.quantize(a, m.NVFP4)
    tb = m.BlockTensor.quantize(b, m.NVFP4)
    acc = m.Accumulator(m.FP32, order="sequential", group=16, align_bits=25)
    yield "NVFP4 quantize 64x64", lambda i: m.BlockTensor.quantize(a, m.NVFP4), 64 * 64
    yield "NVFP4 64x64x16 matmul, FP32 acc", (
        lambda i: m.matmul(ta, [r[:16] for r in tb.dequantize()], acc=acc)), 64 * 64 * 16
    vec_a, vec_b = ta.dequantize()[0], tb.dequantize()[0]
    yield "dot 64 (exact Fraction)", lambda i: m.dot(vec_a, vec_b), 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--markdown", action="store_true", help="print README table rows")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    budget = 0.05 if args.quick else 0.3
    if not args.markdown:
        print(f"{'case':36} {'python':>14} {'c++':>14} {'speedup':>9}")
    rows = zip(cases(py, random.Random(1)), cases(cpp, random.Random(1)))
    for (name, fpy, unit), (_, fcpp, _) in rows:
        # size the loop so the Python run takes about `budget` seconds
        t = time.perf_counter(); fpy(0); one = max(time.perf_counter() - t, 1e-7)
        n = max(3, int(budget / one))
        r_py, r_cpp = rate(fpy, n), rate(fcpp, max(n, int(n * 20)))
        u = "op/s" if unit == 1 else "el/s"
        if args.markdown:
            print(f"| {name} | {r_py * unit:,.0f} {u} | {r_cpp * unit:,.0f} {u} | {r_cpp / r_py:.0f}× |")
        else:
            print(f"{name:36} {r_py * unit:12,.0f}{u[-2:]} {r_cpp * unit:12,.0f}{u[-2:]} {r_cpp / r_py:8.1f}x")


if __name__ == "__main__":
    main()
