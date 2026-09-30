# VeriFloat

VeriFloat is a small, pure-Python library of bit-accurate number types, written as golden reference models for verifying number-format hardware (RTL) in [cocotb](https://www.cocotb.org/) testbenches. It covers:

- **`FP`**: floating point with any exponent/mantissa width (including zero-width mantissas), any bias, signed or unsigned, with or without zero.
  - The inf/NaN encoding can be IEEE 754, OCP "FN" (NaN only), or none.
  - Five IEEE rounding modes plus stochastic rounding, tininess detected before or after rounding, and x86/ARM/RISC-V NaN conventions.
  - Options for saturation, exponent wrap and flush-to-zero.
  - Operations: `+ − × ÷`, fused multiply-add, square root, IEEE min/max, integer conversion, round-to-integral and comparisons, each with IEEE status flags.
- **`UINT` / `INT`**: fixed-width two's-complement integers that wrap or saturate.
- **`FIXED`**: fixed-point numbers with exact bit-growing arithmetic and rounding/overflow on cast.
- **`BlockTensor`**: block-scaled N-D tensors (NVFP4, OCP MX, or custom), plus models of how hardware accumulates dot products.

Values are exact rationals internally, so every operation rounds once into its target format. Each result carries its raw bits, its status flags and its unrounded value, which is what a scoreboard compares against a DUT.

**Status:** alpha (0.1). The API may still change. See [Validation](#validation) for what has and hasn't been checked, and [Limitations](#limitations).

```python
from verifloat import FP16, E4M3

FP16(1) / FP16(0)            # FP(inf, e5m10, flags=DIVZERO)
E4M3(470)                    # FP(nan, e4m3, fn, flags=INEXACT|OVERFLOW)   OCP FP8: overflow gives NaN
FP16(1.5).fma(FP16(2), FP16(0.25))   # FP(3.25, e5m10)
```

## When to use something else

Other projects cover parts of this space, and several are more mature:

| Project | Consider it when you need |
|---|---|
| [APyTypes](https://apytypes.org/) | Fast (C++ backend) bit-accurate fixed- and floating-point arrays for hardware design. VeriFloat is much slower |
| [ml_dtypes](https://github.com/jax-ml/ml_dtypes) | NumPy dtypes for the standard ML formats at NumPy speed |
| [gfloat](https://github.com/graphcore-research/gfloat) | Encoding/decoding of IEEE P3109, OCP FP8 and MX formats with many rounding modes |
| [pychop](https://pypi.org/project/pychop/) | Simulating low precision in numerical algorithms, with NumPy/PyTorch/JAX backends |
| [Berkeley SoftFloat / TestFloat](http://www.jhauser.us/arithmetic/) | The standard C reference and test-vector generator for IEEE binary16/32/64/128 |

VeriFloat's focus is narrower: a per-operation golden model for cocotb testbenches of FP4/FP8/custom-format datapaths. It gives raw codes and status flags per result, covers hardware options (saturate, wrap, FTZ/DAZ, NaN conventions, tininess), and models block-scaled tensors and accumulators. It trades speed for exactness and simplicity: roughly 100k scalar operations per second.

## Installation

```bash
pip install verifloat
```

Requires Python 3.12+. There are no runtime dependencies.

## Quick start

```python
from verifloat import FPFormat, FP16, FP32, E2M1, Rounding, UINT, INT

FP16(1 / 3)                 # FP(0.333251953125, e5m10, flags=INEXACT)
FP16(1 / 3).raw             # 13653 (0x3555)
E2M1(2.5)                   # FP(2.0, e2m1, finite, flags=INEXACT)   NVFP4 element, ties to even

# Describe the format your RTL implements, then use it like a constructor
FP8 = FPFormat(4, 3, bias=10, ftz=True, rounding=Rounding.RTZ)
y = FP8(3) * FP8(0.6875)    # FP(2.0, e4m3, bias=10, ftz, rtz, flags=INEXACT)
y.raw, y.flags, y.unrounded # (88, <FPFlags.INEXACT: 1>, Fraction(33, 16))

UINT(250, 8) + 10                    # UINT(4, u8)            wraps
INT(100, 8, saturate=True) + 100     # INT(127, i8, sat)      saturates
```

## Floating point

### Formats

An `FPFormat` describes a format completely. The fields after `signed` are keyword-only:

| Field | Default | Meaning |
|---|---|---|
| `exp_bits`, `mantissa_bits` | 2, 1 | field widths (`mantissa_bits` may be 0) |
| `bias` | `2**(exp_bits-1) - 1` | any integer |
| `signed` | `True` | `False` removes the sign bit |
| `inf_nan` | `True` | `True`: IEEE 754 (all-ones exponent holds inf/NaN). `"fn"`: OCP style, only the all-ones code is NaN. `False`: all codes finite |
| `has_zero` | `True` | `False`: exponent field 0 is an ordinary binade, so there is no zero and no subnormal (as in OCP E8M0) |
| `saturate` | `False` | overflow gives ±max instead of inf/NaN |
| `wrap` | `False` | overflow/underflow wraps the exponent field, as a datapath that drops the carry would |
| `ftz` | `False` | flush-to-zero outputs and denormals-are-zero inputs |
| `rounding` | `Rounding.RNE` | `RNE`, `RNA`, `RTZ`, `RUP`, `RDN`, or `SR` (stochastic) |
| `sr_bits` | 8 | random bits per stochastic rounding |
| `nan_mode` | `NaNMode.CANONICAL` | which NaN results are returned: `CANONICAL` (RISC-V), `PROPAGATE`, `X86`, `ARM` |
| `tininess` | `"after"` | underflow tininess detected `"after"` rounding (x86, RISC-V) or `"before"` (ARM) |

A format is frozen and hashable. `str()` and `FPFormat.parse()` round-trip its short name (e.g. `'ue4m3, bias=10, fn, rtz'`).

```python
fmt = FPFormat(5, 10)
fmt(1.5), fmt.from_raw(0x3C00), fmt.max, fmt.min_normal, fmt.min_subnormal
fmt.all_values()                        # every encoding, for exhaustive sweeps
fmt.replace(rounding=Rounding.RTZ)      # a variant
x.format, x.convert(E4M3)               # a value's format; convert it
```

Presets:

| Preset | Format | Notes |
|---|---|---|
| `FP16`, `BF16`, `FP32`, `FP64` | IEEE binary16, bfloat16, binary32, binary64 | `FP32` is also exported as `E8M23` |
| `E5M2`, `E4M3` | OCP FP8 | E4M3 is `inf_nan="fn"`, max 448, overflow gives NaN |
| `E3M2`, `E2M3`, `E2M1` | OCP FP6 / FP4 | no inf/NaN, saturate. `E2M1` is the NVFP4/MXFP4 element |
| `UE4M3`, `UE8M0` | unsigned E4M3; OCP E8M0 (`2**(code-127)`, no zero, 0xFF is NaN) | |

The default `FPFormat()` is a 4-bit IEEE-style E2M1 with inf/NaN, **not** the NVFP4 element. Use `E2M1` for NVFP4.

### Operations

Every result is computed exactly and rounded once:

| Operation | Notes |
|---|---|
| `a + b`, `a - b`, `a * b`, `a / b` | with Python numbers too (cast into the FP's format first) |
| `a.fma(b, c)` | fused multiply-add |
| `a.sqrt()` | |
| `a.minimum(b)`, `a.maximum(b)` | IEEE 754-2019: NaN propagates, −0 < +0 |
| `a.minimum_number(b)`, `a.maximum_number(b)` | IEEE 754-2019 / RISC-V `fmin`/`fmax`: a NaN operand is ignored |
| `a.to_int(bits, signed, rounding, exact)` | returns `(INT or UINT, flags)`. An invalid conversion returns what the `nan_mode` platform returns: RISC-V saturates, x86 gives the "integer indefinite", ARM saturates and maps NaN to 0 |
| `a.round_to_integral(rounding, exact)` | IEEE `roundToIntegral` (`exact=True` is `roundToIntegralExact`) |
| `a.compare(b, signaling)`, `a.eq/lt/le(b, signaling)` | return `(result, flags)`. Quiet comparisons flag only signaling NaNs; signaling ones flag any NaN |
| `==`, `<`, … | IEEE ordering without flags (NaN is unordered) |
| `& \| ^ ~` | on the raw bit patterns |

```python
FP32(3.7).to_int(8)                                 # (INT(4, i8), <FPFlags.INEXACT: 1>)
FP32(1e10).to_int(8)                                # (INT(127, i8), <FPFlags.INVALID: 16>)   RISC-V
FP32.replace(nan_mode=NaNMode.X86)(1e10).to_int(8)  # (INT(-128, i8), <FPFlags.INVALID: 16>)  x86
FP16(1).lt(FP16(float("nan")))                      # (False, <FPFlags.INVALID: 16>)
FP16(1).minimum_number(FP16(float("nan")))          # FP(1.0, e5m10)
```

### IEEE semantics

- **Overflow** follows IEEE 754-2019 §7.4: RNE, RNA and SR give ±inf, RTZ gives ±max, and RUP/RDN give inf only in their own direction. `"fn"` formats give NaN where IEEE gives inf. `saturate=True` or `inf_nan=False` always gives ±max.
- **Underflow:** gradual (subnormals). The UNDERFLOW flag is raised for tiny, inexact results, with tininess judged after rounding by default (`tininess="before"` for ARM-style).
- **Special operands:** `inf − inf`, `0 × inf`, `0/0`, `inf/inf` and `sqrt(−x)` give NaN and raise INVALID. `x/0` gives ±inf and raises DIVZERO. Signaling-NaN operands raise INVALID.
- **fma with a quiet-NaN addend:** for `0 × inf + qNaN`, IEEE leaves signaling to the implementation. RISC-V and ARM signal INVALID, and x86 hardware does not; `nan_mode` selects which.
- **Signed zero:** an exact zero sum of opposite signs is +0, except −0 under RDN.
- **Formats without NaN** (`inf_nan=False`): division by zero raises `ZeroDivisionError`, and invalid operations raise `ValueError`, since there is nothing IEEE-correct to return.
- **Unsigned formats** clamp negative results to 0.
- **Formats without zero** return NaN (INVALID) for a zero or negative result, and round tiny values up to the smallest value.

### Stochastic rounding

`Rounding.SR` follows the hardware definition used by gfloat. The discarded fraction is rounded to `sr_bits` bits, added to `sr_bits` random bits, and the result rounds away from zero if that sum carries. Pass the random bits explicitly to reproduce a DUT, or set a source for operators:

```python
f = E4M3.replace(rounding=Rounding.SR, sr_bits=4)
f(1.3, sr_rand=5), f(1.3, sr_rand=12)     # (FP(1.25, ...), FP(1.375, ...))
verifloat.set_sr_source(lfsr_model)      # callable(nbits) -> int, or an int seed
```

### Status flags and diagnostics

| Attribute | Meaning |
|---|---|
| `flags` | `FPFlags`: `INEXACT` (bit 0), `UNDERFLOW` (1), `OVERFLOW` (2), `DIVZERO` (3), `INVALID` (4). This is the RISC-V `fflags` bit order |
| `unrounded` | the exact result before rounding (`Fraction`, ±inf, or `None`) |
| `exact`, `ulp`, `error_ulps(ref=None)` | exact value, spacing at this value, error in ulps |
| `raw`, `to_hex()`, `to_bin(sep=" ")` | encoding |
| `is_nan`, `is_snan`, `is_inf`, `is_finite`, `is_zero`, `is_subnormal` | classification |

## Integers and fixed point

`UINT(val, bits, saturate=False)` and `INT(val, bits, saturate=False)` wrap modulo 2^bits, or clamp to [min, max] with `saturate=True`. Bit operations always wrap. `INT` division truncates toward zero. A Python `int` operand is cast into the operand's type first (with a warning if it doesn't fit). Mixing widths gives the wider width, and mixing `UINT` with `INT` gives `UINT` over raw bits (C rules).

`FixedFormat(int_bits, frac_bits, signed=True, *, rounding, overflow)` describes fixed point. `int_bits` includes the sign bit, as in APyTypes and VHDL `fixed_pkg`. Arithmetic grows the result exactly:
- `+`/`−`: one more integer bit.
- `×`: the widths add.

Rounding and overflow (`"wrap"` or `"saturate"`) happen only on `cast`:

```python
q = FixedFormat(4, 4)                  # signed, 4 integer bits (incl. sign), 4 fraction bits
x = q(1.3)                             # FIXED(1.3125, sfix4.4, flags=INEXACT)
y = x * q(-2)                          # FIXED(-2.625, sfix8.8)        exact
y.cast(4, 2, rounding=Rounding.RDN, overflow="saturate")   # FIXED(-2.75, sfix4.2, rdn, saturate, flags=INEXACT)
```

## Warnings

Warnings flag behaviour that is easy to miss in a testbench. All derive from `VeriFloatWarning` (a `RuntimeWarning`) and point at the calling line:

| Warning | Raised when |
|---|---|
| `CastWarning` | a Python number is cast lossily into an operand's format; `reblock` changes values |
| `FPFormatWarning` | FP operands of different width/bias are promoted |
| `FPModeWarning` | FP operands differ in any other format field (one warning per field) |
| `IntCastWarning` | integer width/signedness/saturate mismatch, or an int literal that doesn't fit |
| `BlockFormatWarning` | block tensors of different formats are combined |
| `FPOverflowWarning`, `FPUnderflowWarning` | an unsigned format overflows, underflows or clamps, or any format's exponent wraps |

When FP formats are mixed, the result has the smallest format holding both operands exactly. It is signed if either is, and has inf/NaN if either does. The other fields come from the left operand.

## Block-scaled tensors

`BlockTensor.quantize(values, fmt, axis=-1)` quantizes an N-D tensor (nested lists) in blocks along `axis`:

```
value = tensor_scale * block_scale * (element - zero_point)
```

A `BlockFormat` takes:
- `elem`: an `FPFormat`, or an `IntFormat(bits, signed, rounding, frac_bits)`.
- `block_size`.
- `scale`: an `FPFormat`, a `Pow2Format` (E8M0-style), or `None`.
- Optionally `scale_max`, a per-block `zero_point`, and a `tensor_scale` format.

Three **recipe** options select how scales are computed:
- `compute`: the intermediate arithmetic format (`None` = exact).
- `scale_min`: a lower clamp on block scales.
- `zero_scale`: the scale stored for an all-zero block.

| Preset | Elements | Block | Scale | Tensor scale | Recipe |
|---|---|---|---|---|---|
| `NVFP4` | E2M1 | 16 | E4M3 | FP32 | exact |
| `NVFP4_MODELOPT` | E2M1 | 16 | E4M3 | FP32 | NVIDIA ModelOpt's: FP32 arithmetic, scales clamped to [2⁻⁹, 448], all-zero blocks scale 1 |
| `MXFP4` | E2M1 | 32 | E8M0 | none | OCP MX |
| `MXINT8` | INT8 as 1.6 fixed point | 32 | E8M0 | none | OCP MX |

```python
t = BlockTensor.quantize([6, 3, 1.5, 0, -0.5, -6], NVFP4)
t.elem_raw, t.scale_raw, t.tensor_scale.to_hex()   # ([7, 5, 3, 0, 9, 15], [126], '3b124925')
BlockTensor.quantize([[0.0]*16, [1.0]*16], NVFP4).scale_raw            # [[0], [126]]
BlockTensor.quantize([[0.0]*16, [1.0]*16], NVFP4_MODELOPT).scale_raw   # [[56], [126]]
```

**Which recipe?** `NVFP4_MODELOPT` matched NVIDIA's quantizer on every code tested (see Validation). The exact `NVFP4` recipe differs from it in two ways:
- **Scale clamps**, the main difference. On test matrices with a wide dynamic range, 57% of block scales differed.
- **FP32 rounding of intermediates**, which caused only rare tie differences (6 of 9,034 scales).

Use the recipe that matches the quantizer your RTL is verified against.

**Operations:**
- `+ − *` (elementwise or by a scalar), unary `−` and `@` compute exact values, then requantize into the left operand's format.
- `t[i, j, …]` indexes the exact values, which are computed once and cached.
- `t.transpose(*axes)` / `t.T` moves the blocks with their axis, so it is exact and no requantization happens.
- `t.reblock(axis)` requantizes along another axis, with a `CastWarning` if that changes values.
- `dot(a, b, acc)` and `matmul(a, b, acc, out, transpose_b=False)` take 1-D, 2-D or batched N-D operands. Their outputs are exact `Fraction`s, or rounded per `acc`.

### Accumulation

`acc` can be an `FPFormat` (exact sum, rounded once) or an `Accumulator` that models hardware:

```python
Accumulator(FP32, "sequential")               # s = round(s + a[i]*b[i]): an FMA chain
Accumulator(FP32, "pairwise", product=FP32)   # round products, then a rounded adder tree
Accumulator(FP32, "sequential", group=16, align_bits=24)
    # sum 16 products at a time in an adder that aligns to the largest exponent
    # and truncates below 24 bits, then add to the running sum with one rounding

Accumulator(FP32, "sequential").sum_products([FP32(1e8), FP32(1), FP32(-1e8)], [FP32(1)] * 3)
# FP(0.0, e8m23, flags=INEXACT)      (the exact answer is 1)
```

## cocotb

`verifloat.bus` packs codes onto flat buses (lane 0 in the low bits) and back. It doesn't import cocotb:

```python
from verifloat.bus import pack, unpack, pack_fp, unpack_fp, pack_block_row, unpack_block_row

dut.a.value = pack_fp([FP16(1), FP16(-2)])            # 0xC0003C00
b = pack_block_row(t)                                 # elems / scales / zeros / tensor_scale buses
```

Two complete testbenches run open-source RTL on Verilator. Each downloads its RTL at a pinned commit:

- [`examples/fp32_adder`](examples/fp32_adder) checks the [dawsonjon/fpu](https://github.com/dawsonjon/fpu) IEEE single-precision adder against `FP32`.
- [`examples/nvfp4_dotprod`](examples/nvfp4_dotprod) checks the NVFP4 mode of [sashakatne/nvfp4-dotprod-formal-dv](https://github.com/sashakatne/nvfp4-dotprod-formal-dv), a 16-element block dot product, against `BlockTensor` and `dot`.

Both passed tens of thousands of constrained random vectors with no mismatches, and both fail immediately when the golden model is deliberately changed.

## Validation

The tests (`tests/`) use seeded, constrained random stimulus. Every operation is compared with a separately written reference model (`tests/reference.py`) that predicts result bits, flags and warnings across random formats, including unsigned, no-zero, zero-mantissa, wrap, FTZ, every rounding mode and NaN convention. Both models were written for this project, so the tests also compare against references VeriFloat didn't write:

| Reference | What is compared (bits and flags unless noted) |
|---|---|
| [Berkeley TestFloat 3e](http://www.jhauser.us/arithmetic/TestFloat.html) with SoftFloat 3e, specialized for RISC-V, x86 SSE and ARM VFPv2 | binary16/32/64 `+ − × ÷`, fma, sqrt, round-to-integral, float↔int32/64 conversions, float↔float conversions, quiet and signaling comparisons. All five rounding modes and both tininess conventions (RISC-V); round-to-nearest-even for x86/ARM NaN conventions |
| Host x86 SSE/FMA unit, driven through MXCSR | binary32/64 `+ − × ÷`, sqrt, fma in four rounding modes, with and without FTZ+DAZ; FMA chains with signed zeros |
| C library `fma`/`fmaf`, numpy float32 | accumulator models (FMA chain, rounded products, pairwise trees) |
| numpy / host FPU | binary64/32 arithmetic and flags, conversions to binary32/16, IEEE min/max (except ±0 pairs, which C leaves open) |
| [ml_dtypes](https://github.com/jax-ml/ml_dtypes) | every encoding, conversion and arithmetic for bfloat16, E5M2, E4M3 (FN and IEEE-style), E3M4, E3M2, E2M3, E2M1, E8M0 (values only) |
| [gfloat](https://github.com/graphcore-research/gfloat) | all five rounding modes, saturation and stochastic rounding for the OCP formats, binary16 and bfloat16; E8M0 ties; MXFP4/MXINT8 block encoding |
| [APyTypes](https://apytypes.org/) | fixed-point casts (every rounding mode, wrap/saturate), bit growth on `+ − ×`, saturating integer arithmetic |
| [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (optional, see below) | NVFP4 quantization: `NVFP4_MODELOPT` matched every element, block-scale and tensor-scale code on the matrices tested |
| IEEE 754-2019, OCP FP8 and MX v1.0 | known encodings: max, min normal/subnormal, NaN/inf codes, 1/3 |

These checks found real bugs, all fixed:
- MXINT8 scale codes were off by 6 (its elements are 1.6 fixed point).
- Exact-zero sums under round-toward-negative were +0 instead of −0.
- With DAZ, `sqrt`, min/max and round-to-integral passed subnormal inputs through instead of returning zero.
- ARM's NaN priority in fma was wrong.
- An accumulator crashed when its running sum overflowed.

**Documented divergences between references** (VeriFloat follows the first one named in each case):
- **fma(0, ∞, qNaN):** x86 hardware doesn't signal INVALID, but SoftFloat's x86 model does.
- **Signaling-NaN operand to `minimum_number`:** IEEE 754-2019 and RISC-V return the number, while C `fmin` returns NaN.
- **E8M0 ties:** gfloat and IEEE P3109 break ties by code parity, while ml_dtypes always rounds up.
- **E8M0 inputs below 2⁻¹²⁶:** ml_dtypes rounds these to code 1.

**Not validated externally:**
- the `wrap` option
- `align_bits` accumulation
- `PROPAGATE`/`ARM`/`X86` NaN conventions under non-RNE rounding
- formats other than the IEEE/OCP ones, beyond VeriFloat's own reference model

## Limitations

- **Slow:** pure Python with `Fraction` arithmetic. Fine for unit-level cocotb tests, not for large regressions.
- **Missing operations:** no IEEE `remainder`, `nextUp`/`nextDown`, `scaleB`/`logB`, or decimal formats.
- **`BlockTensor` is list-based** and suited to the tensor sizes of unit tests (hundreds to thousands of elements).
- **Accumulator models** cover common structures (sequential, pairwise, aligned groups), but real accumulators vary. Check your RTL's documentation.
- **The default `FPFormat()`** is an IEEE-style E2M1, rarely what you want. Pass a format.

## Development

```bash
uv sync                                  # dev tools: pytest, numpy, ml_dtypes, gfloat, apytypes
uv run pytest
VERIFLOAT_SEED=1234 uv run pytest        # reproduce a run (the seed is printed in the header)
VERIFLOAT_ITERS=20000 uv run pytest      # more random cases per test
```

- **TestFloat** (`tests/test_testfloat.py`) downloads and builds Berkeley SoftFloat/TestFloat 3e on first run, into `~/.cache/verifloat` (set `VERIFLOAT_CACHE` to change this). It needs network access, gcc and make, and is skipped without them. `VERIFLOAT_TESTFLOAT_N` sets the number of vectors per case.
- **The x86 tests** (`tests/test_hwfpu.py`) compile a small C helper with gcc and run only on x86-64.
- **The NVIDIA comparison** (`tests/test_nvidia.py`) needs a separate Python with torch and nvidia-modelopt, set in `VERIFLOAT_MODELOPT_PYTHON`. See that file for setup.
- **The cocotb examples** need Verilator and `uv sync --group examples`.

## License

MIT. See [LICENSE](LICENSE).
