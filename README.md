<div align="center">

# VeriFloat

**Bit-accurate golden models for number-format hardware.**

Floating point of any width, fixed point, integers and block-scaled tensors, with IEEE status flags on every result,<br>
for verifying RTL from [cocotb](https://www.cocotb.org/) testbenches in Python, or from SystemVerilog through DPI-C.

[![tests](https://github.com/yeoshuyi/VeriFloat/actions/workflows/tests.yml/badge.svg)](https://github.com/yeoshuyi/VeriFloat/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)](https://github.com/yeoshuyi/VeriFloat/tree/main/src/cpp)
[![Platforms](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md#platforms)
[![cocotb](https://img.shields.io/badge/cocotb-2.x-6D2077)](https://www.cocotb.org/)
[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md#limitations)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/yeoshuyi/VeriFloat/blob/main/LICENSE)

[Documentation](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md) ·
[Validation](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md#validation) ·
[Examples](https://github.com/yeoshuyi/VeriFloat/tree/main/examples) ·
[Contributing](https://github.com/yeoshuyi/VeriFloat/blob/main/CONTRIBUTING.md)

> This project is written with AI assistance, through Claude Code (Opus 5.5)
</div>

---

## Highlights

- **Any FP format.** Any exponent and mantissa width, any bias, signed or unsigned, with IEEE, OCP "FN" or no inf/NaN. Presets for FP16, BF16, FP32, FP64, OCP FP8/FP6/FP4 and E8M0.
- **Hardware options.** Five IEEE rounding modes plus stochastic rounding, saturation, exponent wrap, flush-to-zero, tininess before or after rounding, and x86, ARM or RISC-V NaN conventions.
- **Everything a scoreboard needs.** Each result carries its raw bits, its IEEE status flags and its exact unrounded value. Every operation is exact and rounds once.
- **Full IEEE 754 operation set.** `+ − × ÷`, fma, sqrt, remainder, min/max, nextUp/Down, scaleB/logB, classification, integer conversion, round-to-integral and comparisons.
- **Arrays and tensors.** `FPArray` for N-D arrays with NumPy interop and SIMD kernels. `BlockTensor` for NVFP4, OCP MX and custom block-scaled formats, with models of how hardware accumulates dot products.
- **Fixed point and integers.** `FIXED` with exact bit growth, plus wrapping or saturating `UINT`/`INT`.
- **Testbench tools.** A scoreboard that explains each mismatch, and test-vector files (TestFloat, CSV, JSON lines, `$readmemh`).
- **No Python needed.** `libverifloat` is the same core as a C library with a SystemVerilog DPI-C package.
- **Fast and checked.** A C++ core runs millions of scalar operations and hundreds of millions of array elements per second. Results are checked against Berkeley SoftFloat/TestFloat, MPFR, the x86 FPU, ml_dtypes, gfloat and APyTypes.

## Installation

```bash
pip install verifloat
```

Requires Python 3.12 or later, with no runtime dependencies. Wheels are published for Linux (x86-64 and aarch64, glibc and musl), macOS (Apple silicon and Intel) and Windows (x86-64). Elsewhere pip builds from source, which needs a C++20 compiler (GCC, Clang or MSVC) and git.

## Quick start

```python
from verifloat import FP16, E4M3, FPFormat, Rounding

FP16(1) / FP16(0)                    # FP(inf, e5m10, flags=DIVZERO)
E4M3(470)                            # FP(nan, e4m3, fn, flags=INEXACT|OVERFLOW)   OCP FP8: overflow gives NaN
FP16(1.5).fma(FP16(2), FP16(0.25))   # FP(3.25, e5m10)

# Describe the format your RTL implements, then use it like a constructor
FP8 = FPFormat(4, 3, bias=10, ftz=True, rounding=Rounding.RTZ)
y = FP8(3) * FP8(0.6875)             # FP(2.0, e4m3, bias=10, ftz, rtz, flags=INEXACT)
y.raw, y.flags, y.unrounded          # (88, <FPFlags.INEXACT: 1>, Fraction(33, 16))
```

### In a cocotb testbench

```python
from verifloat import FP32
from verifloat.scoreboard import Scoreboard

sb = Scoreboard("adder", fail_fast=False, logger=dut._log)
a, b = FP32.from_raw(a_bits), FP32.from_raw(b_bits)
sb.check(a + b, dut.z, flags=dut.fflags, op="add", a=a, b=b)   # compares bits and flags
sb.assert_clean()
```

A mismatch report shows both values, their encodings, the distance in ulps and the exact result:

```
value mismatch (e5m10)
  op       add
  expected 0.2998046875     0x34cc  0|01101|0011001100
  actual   0.300048828125   0x34cd  0|01101|0011001101
  diff     +1 ulp (actual - expected)
  exact    0.2999267578125  (off by: expected -0.5 ulp, actual +0.5 ulp)
```

### From SystemVerilog

```systemverilog
import verifloat_pkg::*;

chandle fp32 = vf_format("e8m23");
expected = vf_add(fp32, a, b, flags);   // raw code and IEEE flags, no Python in the simulation
```

## Documentation

| Topic | |
|---|---|
| [Floating point](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md#floating-point) | formats, operations, IEEE semantics, stochastic rounding, flags, arrays |
| [Integers and fixed point](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md#integers-and-fixed-point) | `UINT`, `INT`, `FIXED` |
| [Block-scaled tensors](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md#block-scaled-tensors) | NVFP4, MXFP4, MXINT8, custom formats, accumulator models |
| [cocotb and the scoreboard](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md#cocotb) | bus packing, mismatch reports, three complete testbenches |
| [Test vectors](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md#test-vectors) | generating and checking vector files |
| [SystemVerilog and C](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md#systemverilog-and-c) | the DPI-C package and the C API |
| [Validation](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md#validation) and [platforms](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md#platforms) | what each outside reference checks; bit-identical results across machines |
| [Performance](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md#performance) and [limitations](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md#limitations) | throughput against APyTypes; what is not supported |
| [When to use something else](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md#when-to-use-something-else) | APyTypes, ml_dtypes, gfloat, pychop, SoftFloat |

## Status

VeriFloat is **alpha** (0.2), so the API may still change. It has been run bit for bit across Linux distributions, compilers, C libraries and CPU architectures. See [Platforms](https://github.com/yeoshuyi/VeriFloat/blob/main/docs/README.md#platforms) for what has and hasn't been run. The original pure-Python implementation (0.1) is frozen in [`archive/python`](https://github.com/yeoshuyi/VeriFloat/tree/main/archive/python) and is the reference the C++ core is tested against.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](https://github.com/yeoshuyi/VeriFloat/blob/main/CONTRIBUTING.md) for how to build, test and validate a change. Please report security problems privately, as described in [SECURITY.md](https://github.com/yeoshuyi/VeriFloat/blob/main/SECURITY.md).

## License

[MIT](https://github.com/yeoshuyi/VeriFloat/blob/main/LICENSE)
