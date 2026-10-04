# VeriFloat documentation

The user guide and reference for VeriFloat. For an overview, see the [project README](../README.md); for building and testing, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Floating point](#floating-point)
  - [Formats](#formats)
  - [Operations](#operations)
  - [IEEE semantics](#ieee-semantics)
  - [Stochastic rounding](#stochastic-rounding)
  - [Status flags and diagnostics](#status-flags-and-diagnostics)
  - [Arrays](#arrays)
- [Integers and fixed point](#integers-and-fixed-point)
- [Warnings](#warnings)
- [Block-scaled tensors](#block-scaled-tensors)
  - [Accumulation](#accumulation)
- [cocotb](#cocotb)
  - [Scoreboard](#scoreboard)
- [Test vectors](#test-vectors)
- [SystemVerilog and C](#systemverilog-and-c)
- [Validation](#validation)
- [Platforms](#platforms)
- [Performance](#performance)
- [Limitations](#limitations)
- [When to use something else](#when-to-use-something-else)

## Installation

```bash
pip install verifloat
```

Requires Python 3.12+. There are no runtime dependencies. Installing from source builds the C++ extension, which needs a C++20 compiler; pip fetches CMake, scikit-build-core and nanobind, and Boost.Multiprecision (header-only) is taken from the system or downloaded.

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
| `a.remainder(b)`, `a.fmod(b)` | IEEE 754 `remainder` (`a − n·b` with `n = a/b` rounded to nearest, ties to even) and C `fmod` (`n` truncated) |
| `a.next_up()`, `a.next_down()` | IEEE 754 `nextUp` / `nextDown`: the neighbouring value of the format |
| `a.scaleb(n)`, `a.logb()` | IEEE 754 `scaleB` (`a × 2**n`, rounded into the format) and `logB` (`floor(log2(\|a\|))` as a value of the format) |
| `a.copysign(b)`, `a.fsgnj(b)`, `a.fsgnjn(b)`, `a.fsgnjx(b)` | sign injection (IEEE `copySign`, RISC-V `fsgnj*`): bits only, no flags |
| `a.fclass()` | RISC-V `fclass`: a 10-bit mask with one bit set (bit 0 −inf … bit 9 quiet NaN) |
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
FP32(5.0).remainder(FP32(3.0)), FP32(5.0).fmod(FP32(3.0))   # (FP(-1.0, e8m23), FP(2.0, e8m23))
FP16(1.0).next_up().to_hex(), FP16(1.0).scaleb(-3), FP16(8.5).logb()   # ('3c01', FP(0.125, e5m10), FP(3.0, e5m10))
FP16(-1.5).fclass(), FP16(1.5).copysign(FP16(-2.0))  # (2, FP(-1.5, e5m10))
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

### Arrays

`FPArray` holds an N-D array of values in one format (of at most 64 bits) as raw codes. Each element of a result is exactly what the scalar operation gives: the same code, the same flags, and the same warnings and errors.

```python
a = FP16.array([[1.5, 2.25], [0.1, -3.0]])        # round nested lists (or a NumPy array) into the format
b = FPArray.from_raw([0x3c00, 0x4000], FP16)      # from encodings: [1.0, 2.0]
c = a * b                                         # also + - /; with an array (shapes broadcast), an FP or a number
c.raw                                             # [[15872, 17536], [11878, 50688]]
c.flags                                           # <FPFlags: 0>: the OR over all elements
c[1, 0]                                           # FP(0.0999755859375, e5m10)
a[:, 1], a[::-1, None, 0], a[..., 0]              # integers, slices, ... and None, as NumPy's basic indexing
a.sqrt(), a.fma(b, 1), a.maximum(2), a.lt(b)      # the FP methods, element by element
a.sum()                                           # Fraction(6963, 8192): the exact sum
a.sum(1, Accumulator(FP16, "sequential"))         # each row added in order, rounding after every addition
c.tolist(), c.to_float(), c.shape, c.T, c.reshape(-1), c.convert(E4M3), b.broadcast_to((2, 2))
FPArray.zeros((2, 3), FP16), FPArray.full(3, 0.1, E4M3)
matmul(a, a, acc=Accumulator(FP32, "sequential"))  # an FPArray in FP32
a.to_numpy(), a.to_numpy("raw"), a.to_numpy("flags")   # float64 values, codes (uint16 here), flags per element
```

- **Construction.** `FPArray(values, fmt)` and `fmt.array(values)` round numbers or FP values, taken from nested lists or from a numeric buffer (a NumPy array of floats or integers of any layout, `array.array`, a `memoryview`). `FPArray.from_raw(codes, fmt)` takes encodings, as lists or an integer array. A code must fit the format (0 ≤ code < 2**size; anything else raises `ValueError` rather than being cut to its low bits), except that a signed integer array exactly as wide as the format is read as bit patterns (FP16 codes in `int16`). `FPArray.full(shape, value, fmt)`, `zeros` and `ones` repeat one rounded value.
- **NumPy is optional.** `to_numpy("value" | "raw" | "flags")` returns new arrays; nothing converts implicitly, because float64 cannot hold every format's values.
- **Indexing** with integers gives an `FP` (with its own flags); slices, `...` and `None` select sub-arrays as in NumPy. The result is always a copy, arrays cannot be modified, and an array cannot be empty (a slice that selects nothing raises `IndexError`). `==` is bit-exact equality of whole arrays.
- **Broadcasting.** The operators and the element-wise methods take operands of different shapes by NumPy's rule (shapes aligned at the last axis, axes of length 1 repeated).
- **Element-wise methods** have the names, arguments and defaults of the `FP` methods: `sqrt`, `fma`, `minimum`, `maximum`, `minimum_number`, `maximum_number`, `remainder`, `fmod`, `copysign`, `fsgnj`, `fsgnjn`, `fsgnjx`, `next_up`, `next_down`, `scaleb`, `logb`, `round_to_integral` return an array. `eq`, `lt`, `le`, `compare` and `to_int` return `(nested lists of results, OR of the flags)`, and `fclass` nested lists.
- **`sum(axis=None, acc=None)`** adds elements in index order: exactly (a `Fraction`), rounded once into an `FPFormat`, or as an `Accumulator` models it. It is `dot(elements, [1, 1, ...], acc)` for each sum. With an `axis` the result is an array (nested lists of Fractions when exact).
- `dot` and `matmul` accept arrays wherever they accept nested lists. With an `acc` and two arrays, `matmul` returns an array.
- Arrays do not keep `unrounded` values.

`+ − × ÷`, conversion, construction from floats and `matmul` run on fast kernels (SIMD where the CPU has AVX2 or AVX-512) for the common cases: finite operands and a result in range, in a signed format with a zero and a non-stochastic rounding mode. Anything else (NaN, inf, overflow, unsigned or no-zero formats, stochastic rounding) is computed by the general code, element by element. The results do not depend on which path ran; see [Performance](#performance) for how that is ensured. The element-wise methods and `sum` have no kernels yet: they call the scalar operation for each element (4 to 6 million elements per second for `sqrt`, `fma` and `maximum` on FP32, where `+` reaches hundreds of millions).

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
- `dot(a, b, acc)` and `matmul(a, b, acc, out, transpose_b=False)` take 1-D, 2-D or batched N-D operands: nested lists, `BlockTensor`s or `FPArray`s. Their outputs are exact `Fraction`s, or rounded per `acc`.

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

Three complete testbenches run open-source RTL on Verilator. Each downloads its RTL at a pinned commit:

- [`examples/fp32_adder`](../examples/fp32_adder) checks the [dawsonjon/fpu](https://github.com/dawsonjon/fpu) IEEE single-precision adder against `FP32`.
- [`examples/nvfp4_dotprod`](../examples/nvfp4_dotprod) checks the NVFP4 mode of [sashakatne/nvfp4-dotprod-formal-dv](https://github.com/sashakatne/nvfp4-dotprod-formal-dv), a 16-element block dot product, against `BlockTensor` and `dot`.

- [`examples/dpi_adder`](../examples/dpi_adder) checks the same adder from a SystemVerilog testbench, with the model called through DPI-C (see [SystemVerilog and C](#systemverilog-and-c)).

All three passed tens of thousands of constrained random vectors with no mismatches, and each fails immediately when the golden model is deliberately changed. The two cocotb testbenches report through `verifloat.scoreboard`.

### Scoreboard

`verifloat.scoreboard` compares DUT outputs with the model and explains every difference. It doesn't import cocotb. A DUT value can be an `int`, a bit string, a cocotb `LogicArray`, or a signal handle; X/Z bits are reported as their own kind of mismatch.

```python
from verifloat.scoreboard import Scoreboard, compare

a, b = FP16(0.1), FP16(0.2)
want = a + b
print(compare(want, want.raw + 1, flags=0b00000, op="add", a=a, b=b))
```
```
value mismatch (e5m10)
  op       add
  a        0.0999755859375  0x2e66  0|01011|1001100110
  b        0.199951171875   0x3266  0|01100|1001100110
  expected 0.2998046875     0x34cc  0|01101|0011001100
  actual   0.300048828125   0x34cd  0|01101|0011001101
  diff     +1 ulp (actual - expected)
  exact    0.2999267578125  (off by: expected -0.5 ulp, actual +0.5 ulp)
  flags    expected INEXACT, actual none; differ: INEXACT
```

`compare(expected, actual, *, flags=None, nan="exact", zero_sign=True, tol_ulps=0, flag_map=None, **context)` returns a `Mismatch` or `None`.

| Option | Meaning |
|---|---|
| `nan` | `"exact"`: NaN codes must match; `"quiet"`: any quiet NaN matches a NaN; `"any"`: any NaN matches a NaN |
| `zero_sign` | `False` accepts +0 for -0 |
| `tol_ulps` | accept finite results within this many ulps (default 0: bit-exact) |
| `flags` | the DUT's flag value, in RISC-V `fflags` order unless `flag_map={FPFlags.INEXACT: 0, FPFlags.INVALID: 1}` gives the bit positions. Flags not in the map are not compared |

`Mismatch.kind` is `value`, `nan`, `zero_sign`, `flags` or `unresolved`.

```python
print(compare(FP16(1.0), "0011110000x00000"))          # also LogicArray, handles, ints
```
```
unresolved bits (X/Z) (e5m10)
  expected 1.0  0x3c00  0|01111|0000000000
  actual   unresolved 0b0011110000x00000
```

A `Scoreboard` counts a stream of checks. By default a mismatch raises `AssertionError`, which fails a cocotb test; `fail_fast=False` logs it (cocotb's logger in a simulation) and goes on.

```python
sb = Scoreboard("sum", fail_fast=False, max_report=2)
sb.check(want, dut.z, flags=dut.fflags, op="add", a=a, b=b)
sb.check_array(fp16_array, dut_codes, op="load")        # FPArray or nested FP lists
sb.check_block(block_tensor, elems=..., scales=..., tensor_scale=...)
print(sb.summary())
sb.assert_clean()                                      # also fails if nothing was checked
```
```
sum: 4 checked, 0 passed, 4 failed (value 4)
first 2 mismatches:
value mismatch at [0] (e5m10)
  op       load
  expected 1.0           0x3c00  0|01111|0000000000
  actual   1.0009765625  0x3c01  0|01111|0000000001
  diff     +1 ulp (actual - expected)
value mismatch at [1] (e5m10)
  op       load
  expected 2.0          0x4000  0|10000|0000000000
  actual   2.001953125  0x4001  0|10000|0000000001
  diff     +1 ulp (actual - expected)
```

## Test vectors

`verifloat.vectors` writes files of (operands, expected result, expected flags) for any format and operation, for testbenches that read vectors from files, or for FPGA/silicon bring-up. It is not imported by `import verifloat`:

```bash
python -m verifloat.vectors generate --op add --fmt "e4m3, fn" --count 5 --seed 1 --style mixed --format testfloat -o add.txt
python -m verifloat.vectors verify add.txt --op add --fmt "e4m3, fn"
```

```
$ cat add.txt
01 75 75 01
30 B1 98 00
81 07 06 00
01 00 01 00
FF 7F 7F 00
$ python -m verifloat.vectors verify add.txt --op add --fmt "e4m3, fn"
add(e4m3, fn): checked 5 vectors, 0 mismatches
```

Each line is `a b result flags`: hex, zero-padded to the format's width, flags as in `fflags` (`01` = INEXACT). This is what `testfloat_gen` prints, so a TestFloat-driven testbench reads it unchanged. `verify` exits with status 1 on any difference (2 for a bad file or arguments), and names the line:

```
$ python -m verifloat.vectors verify bad.txt --op add --fmt "e4m3, fn"
line 42: add(0x7e, 0x09): file 0x7f [INEXACT]; model 0x7e [INEXACT]
line 701: add(0xff, 0xf4): file 0x7f [OVERFLOW]; model 0x7f [none]
add(e4m3, fn): checked 1000 vectors, 2 mismatches
```

`verify` also checks vectors made elsewhere: real `testfloat_gen` output for binary16/32/64 passes (`tests/test_vectors.py`).

| Option | Meaning |
|---|---|
| `--op` | `add sub mul div fma sqrt remainder fmod minimum maximum minimum_number maximum_number copysign next_up next_down logb eq lt le` (and `eq_signaling`, `lt_quiet`, `le_quiet`) `compare convert round_to_integral to_int from_int`; TestFloat's `mulAdd` and `roundToInt` work too |
| `--fmt`, `--to` | a format name as `str(fmt)` prints it (`"e5m10"`, `"e4m3, fn, rtz"`), or an uppercase preset with tags (`FP16`, `"BF16, rtz"`). Rounding mode, NaN convention and tininess are tags of the format. `--to` is the target of `convert` |
| `--bits`, `--unsigned`, `--rounding`, `--exact` / `--no-exact` | `to_int` / `from_int` width and signedness; rounding of `to_int` / `round_to_integral`, and whether they raise INEXACT (default: `to_int` yes, `round_to_integral` no, as in the model) |
| `--style` | `mixed` (default): random codes biased toward zeros, subnormals, extremes, inf/NaN (quiet and signaling), neighbouring codes, exponents at the format's limits, and cancelling / equal / reciprocal operands. `random`: uniform. `edges`: every combination of the edge encodings. `exhaustive`: every operand combination (refused above 2**22 unless `--count` samples it) |
| `--format` | `testfloat` (default), `csv`, `jsonl`, `readmemh`; guessed from the `-o` extension |

The other formats:

```
$ python -m verifloat.vectors generate --op fma --fmt FP16 --count 3 --format csv
a,b,c,result,flags
FFFF,4248,BE03,7E00,00
8401,B800,C17C,C17C,01
3FEE,4F65,E61A,E5DF,01

$ python -m verifloat.vectors generate --op div --fmt FP16 --count 2 --format jsonl
{"verifloat_vectors": 1, "version": "0.2.0", "op": "div", "fmt": "e5m10", "seed": 0, "style": "mixed", "count": 2}
{"a": "FFFF", "b": "4248", "result": "7E00", "flags": "00"}
{"a": "C001", "b": "BC02", "result": "3FFE", "flags": "01"}

$ python -m verifloat.vectors generate --op sqrt --fmt "e5m2, rtz" --count 3 --format readmemh --seed 3
// version: 0.2.0
// op: sqrt
...
// localparam int VEC_A_LSB = 13;  // [20:13] a, 8 bits
// localparam int VEC_RESULT_LSB = 5;  // [12:5] result, 8 bits
// localparam int VEC_FLAGS_LSB = 0;  // [4:0] flags, 5 bits
...
008400
0FCFC0
1D0FD0
```

- A `jsonl` file describes itself (`python -m verifloat.vectors verify cvt.jsonl` needs no `--op`/`--fmt`).
- `readmemh` words concatenate the fields, first field in the top bits, for `$readmemh`; `--sv FILE` writes the matching packed struct. `testfloat` and `csv` carry no comment header unless you pass `--header` (`--meta FILE` writes the metadata as a JSON sidecar instead).
- The model cannot answer some stimuli (`x/0` in a format without NaN raises); these are skipped, counted and reported on stderr: `div(e3m2, finite): wrote 3968 vectors, skipped 128 (ZeroDivisionError)`.
- Stochastic rounding: the random bits are exported in an `sr` field after the operands, so a file replays without a shared random source. `to_int` and `round_to_integral` refuse SR formats: give `--rounding`.

From Python:

```python
from verifloat import FP16, vectors

stream = vectors.generate("fma", FP16, 1000, seed=1)       # iterator of Vector(operands, result, flags, sr)
vectors.write("fma.jsonl", stream)                         # format from the extension
vectors.verify("fma.jsonl", out=sys.stdout).ok             # jsonl carries op and format
vectors.verify(open("f16_add.txt"), "add", FP16)           # testfloat_gen output
```

## SystemVerilog and C

The C++ core is also built as a C library, `libverifloat`, with a plain C API (`verifloat.h`) and a SystemVerilog package of DPI-C imports (`verifloat_pkg.sv`). A SystemVerilog or UVM testbench calls the model directly, with no Python in the simulation:

```systemverilog
import verifloat_pkg::*;

chandle fp32 = vf_format("e8m23");               // once; any name the Python package prints
int flags;
expected = vf_add(fp32, a, b, flags);            // raw code and IEEE flags
if (z !== expected[31:0]) $error("got %h, want %h [%s]", z, expected[31:0], vf_flags_str(flags));
```

All three files are installed with the Python package, and `python -m verifloat.dpi` prints where (`--sv`, `--cflags`, `--ldflags`, `--include`, `--lib` print one value):

```bash
verilator --binary --timing tb.sv rtl.v $(python -m verifloat.dpi --sv) \
    -CFLAGS "$(python -m verifloat.dpi --cflags)" -LDFLAGS "$(python -m verifloat.dpi --ldflags)"
```

- **Formats** are named as `str(fmt)` prints them in Python: `"e8m23"`, `"e5m10, rtz"`, `"e4m3, fn"`, `"ue4m3, bias=10, saturate, tininess=before"`. Every format option is available (rounding mode, NaN convention, saturate, wrap, FTZ, no-zero, stochastic rounding).
- **Codes** are raw encodings in the low bits of a 64-bit value, and `flags` uses the RISC-V `fflags` bit order.
- **Functions:** `vf_add`, `vf_sub`, `vf_mul`, `vf_div`, `vf_fma`, `vf_sqrt`, `vf_rem`, `vf_fmod`, `vf_min`, `vf_max`, `vf_minnum`, `vf_maxnum`, `vf_neg`, `vf_abs`, `vf_sgnj`, `vf_sgnjn`, `vf_sgnjx`, `vf_next_up`, `vf_next_down`, `vf_scaleb`, `vf_logb`, `vf_fclass`, `vf_compare`, `vf_eq`, `vf_lt`, `vf_le`, `vf_convert`, `vf_round_to_integral`, `vf_to_int`, `vf_from_int`, `vf_from_uint`, `vf_from_double`, `vf_to_double`, and `vf_op` (any operation by number).
- **Formats wider than 64 bits** (binary128 is `"e15m112"`) use `vf_op128`, `vf_scaleb128`, `vf_fclass128`, `vf_compare128`, `vf_convert128` and `vf_round_to_integral128`, which take `bit [127:0]` values. In C, the `_w` functions take codes of any width as 32-bit words.
- **Errors:** where the model has no answer (an invalid operation in a format without NaN, an unknown format name), the result is 0, `flags` is `VF_ERROR` and `vf_last_error()` gives the reason.
- **Stochastic rounding:** `vf_set_sr(bits)` sets the random bits the next roundings use (in C, `vf_set_sr_source` installs a callback).
- The library always runs the general code (no fast kernels) and keeps no Python state. `vf_last_error` and the stochastic-rounding source are per thread and per process respectively.

From C or C++:

```c
#include <stdio.h>
#include <verifloat.h>

int main(void) {
    const vf_fmt* fp16 = vf_format("e5m10");                 /* binary16 */
    int flags;
    uint64_t r = vf_add(fp16, 0x3c00, 0x0001, &flags);       /* 1.0 + the smallest subnormal */
    printf("%04llx %d\n", (unsigned long long)r, flags);     /* 3c00 1: 1.0, INEXACT */
    return 0;
}
```

```bash
gcc ex.c $(python -m verifloat.dpi --cflags) $(python -m verifloat.dpi --ldflags) -o ex
```

[`examples/dpi_adder`](../examples/dpi_adder) is the `fp32_adder` check written as a SystemVerilog testbench.

## Validation

| Reference | What is compared (bits and flags unless noted) |
|---|---|
| [Berkeley TestFloat 3e](http://www.jhauser.us/arithmetic/TestFloat.html) with SoftFloat 3e, specialized for RISC-V, x86 SSE and ARM VFPv2 | binary16/32/64/128 `+ − × ÷`, fma, sqrt, round-to-integral, float↔int32/64 conversions, float↔float conversions, quiet and signaling comparisons. All five rounding modes under each of the three NaN conventions, and both tininess conventions for `+ − × ÷`. A default run samples TestFloat's level-1 sequences; `VERIFLOAT_TESTFLOAT_N=0` takes all of them (395 million results, 368 million of them fused multiply-adds), which pass on the fast paths and on the general code |
| Berkeley SoftFloat 3e on the designed cases (`tests/test_softfloat.py`), same three specializations | binary16/32/64/128: `+ − × ÷`, fma, sqrt, remainder, the six comparisons, float↔float in all twelve directions, float→int32/64 signed and unsigned (exact and not), int→float, round-to-integral. Five rounding modes, both tininess conventions, each NaN convention. Scalar operations on the general code and on the fast paths, the array kernels at every SIMD level, and the C library |
| [GNU MPFR](https://www.mpfr.org/) (through gmpy2), with an exponent range and subnormal emulation set per format | IEEE-style formats of any width, mantissas from 1 to 500 bits: `+ − × ÷`, fma, sqrt, rounding of exact rationals and doubles, conversions. Four rounding modes (MPFR has no ties-away). The designed cases in eleven formats from e3m2 to e12m150 (`+ − × ÷`, fma, sqrt, conversions) |
| Host x86 SSE/FMA unit, driven through MXCSR | binary32/64 `+ − × ÷`, sqrt, fma in four rounding modes, with and without FTZ+DAZ; FMA chains with signed zeros. Random operands and the designed cases (3.8 million results in a default run) |
| C library `fma`/`fmaf`, numpy float32 | accumulator models (FMA chain, rounded products, pairwise trees) |
| numpy / host FPU | binary64/32 arithmetic and flags, conversions to binary32/16, IEEE min/max (except ±0 pairs, which C leaves open) |
| [ml_dtypes](https://github.com/jax-ml/ml_dtypes) | every encoding, conversion and arithmetic for bfloat16, E5M2, E4M3 (FN and IEEE-style), E3M4, E3M2, E2M3, E2M1, E8M0 (values only) |
| [gfloat](https://github.com/graphcore-research/gfloat) | all five rounding modes, saturation and stochastic rounding for the OCP formats, binary16 and bfloat16; E8M0 ties; MXFP4/MXINT8 block encoding |
| [APyTypes](https://apytypes.org/) | `APyFloat`: `+ − × ÷`, casts between formats, conversion from floats, classification and comparisons, over random exponent/mantissa widths and biases in the five IEEE rounding modes (`tests/test_apytypes_fp.py`). `APyFloatArray` against `FPArray`: the same element-wise, and matmul. `APyFixed`: fixed-point casts (every rounding mode, wrap/saturate), bit growth on `+ − ×`, saturating integer arithmetic |
| [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (optional, see below) | NVFP4 quantization: `NVFP4_MODELOPT` matched every element, block-scale and tensor-scale code on the matrices tested |
| IEEE 754-2019, OCP FP8 and MX v1.0 | known encodings: max, min normal/subnormal, NaN/inf codes, 1/3 |

**Documented divergences between references** (VeriFloat follows the first one named in each case):
- **fma(0, ∞, qNaN):** x86 hardware doesn't signal INVALID, but SoftFloat's x86 model does.
- **Signaling-NaN operand to `minimum_number`:** IEEE 754-2019 and RISC-V return the number, while C `fmin` returns NaN.
- **E8M0 ties:** gfloat and IEEE P3109 break ties by code parity, while ml_dtypes always rounds up.
- **E8M0 inputs below 2⁻¹²⁶:** ml_dtypes rounds these to code 1.
- **Underflow on exact subnormal results:** IEEE 754's default raises no flag, while MPFR's subnormal emulation raises underflow.
- **A quiet NaN passing through an operation:** IEEE 754 raises nothing, while MPFR raises its NaN flag for any NaN result.
- **A running sum that overflows:** `Accumulator` stops at the infinity. APyTypes' matmul keeps adding, so a later product of the opposite sign that also overflows gives NaN.
- **`!=` with a NaN operand:** True under IEEE 754, False in APyFloat.
- **The double just below half the smallest value, rounded to nearest with ties away:** it is nearer to zero, and SoftFloat gives zero; gfloat 0.5 gives the smallest value.

**Not validated externally:**
- the `wrap` option
- `align_bits` accumulation
- the `PROPAGATE` NaN convention
- unsigned formats other than E8M0, `fn` and finite encodings other than the OCP formats, and `FTZ` other than on binary32/64, beyond VeriFloat's own reference model
- status flags outside IEEE-style formats: ml_dtypes, gfloat and APyTypes give values only, so the flags of `fn`, finite and saturating formats rest on VeriFloat's reference model (SoftFloat, the x86 processor and MPFR cover the flags of IEEE-style formats, MPFR at any width)
- stochastic rounding beyond the formats gfloat has (the OCP formats, binary16, bfloat16)
- `scaleb`, `logb`, `next_up`/`next_down`, `fclass` and sign injection outside binary16/32/64, where the C library and NumPy are the references: in other formats they are checked against the model written from their definitions

## Platforms

A result depends neither on the machine that computes it nor on the state the processor is in. All arithmetic is done on integers (64-bit, 128-bit, or of any size), square roots included, and a Python float or a NumPy float buffer is taken apart and put together through its bits. The core executes no floating-point instruction to produce a result, with one exception that cannot change one: the AVX2 kernels count leading zeros with an integer-to-float conversion whose operand is masked so that no rounding mode can alter the count. The SIMD kernels are chosen at run time and are required to give what the general code gives.

That is the design. What has been run:

| Platform | Compiler, C library, Python | Golden corpus | Test suite |
|---|---|---|---|
| x86-64, Ubuntu 26.04 (where it is developed) | GCC 15.2, glibc 2.43, Python 3.12 | written here | 6702 passed on each of AVX-512, AVX2, the scalar kernels and the general code; 3400 on the pure-Python 0.1 |
| Debian 13 | GCC 14.2, glibc 2.41, Python 3.13 | identical | 6696 passed |
| AlmaLinux 9 | GCC 11.5, glibc 2.34, Python 3.12 | identical | 6696 passed |
| Alpine | GCC 15.2, musl, Python 3.14 | identical | 6238 passed (ml_dtypes has no musl wheel) |
| Ubuntu 24.04 | GCC 13.3, glibc 2.39, Python 3.12 | identical | 6634 passed ¹ |
| Fedora | GCC 16.2, glibc 2.43, Python 3.14 | identical | 6634 passed ¹ |
| Arch Linux | GCC 16.2, glibc 2.44, Python 3.14 | identical | 6634 passed ¹ |
| openSUSE Tumbleweed | GCC 16.2, glibc 2.44, Python 3.13 | identical | 6634 passed ¹ |
| Debian 13 with Clang | Clang 19.1, libstdc++, Python 3.13 | identical | 6634 passed ¹ |
| Debian 13 with Clang and libc++ | Clang 19.1, libc++, Python 3.13 | identical | 6634 passed ¹ |
| aarch64 (emulated), Debian 13 | GCC 14.2, glibc 2.41, Python 3.13 | identical | 3575 passed ² |
| riscv64 (emulated), Debian 13 | GCC 14.2, glibc 2.41, Python 3.13 | identical | 3561 passed ² |
| ppc64le (emulated), Debian 13 | GCC 14.2, glibc 2.41, Python 3.13 | identical | 3561 passed ² |
| s390x, big-endian (emulated), Debian 13 | GCC 14.2, glibc 2.41, Python 3.13 | identical | 3561 passed ² |
| 64-bit Windows ABI: MinGW-w64 under Wine 10 | GCC 14, C library only | identical | (C library only) |

¹ Without the edge grids (62 tests), to save time. Containers have no Verilator, cocotb or NVIDIA Model Optimizer, so the six tests that need them are skipped everywhere but on the first line.
² A bounded part of the suite at a twentieth of the usual random cases, without the exhaustive sweeps and with SoftFloat's RISC-V specialization only: the reference-model, C library, array, NumPy, IEEE-operation, fixed-point, accumulator and block-tensor tests, then TestFloat, the designed SoftFloat cases, parity with 0.1, the array kernels and MPFR. ml_dtypes and APyTypes have no wheels for riscv64, ppc64le and s390x. On s390x the binary128 cases of the designed SoftFloat test first failed because the test's own driver handed SoftFloat 128-bit values in little-endian word order; with the driver corrected they pass (TestFloat's binary128 vectors passed from the start).

- **Golden corpus.** `tools/golden_corpus generate` writes 1.4 million vectors (every operation, 23 formats covering every format option, random, biased and edge-case operands) with the build at hand; `verify` recomputes them with another build, through the scalar operations and through the array kernels at each SIMD level the CPU offers, and requires every result bit and flag to be the same. The corpus was written on x86-64 with AVX-512 and reproduced everywhere above.

```bash
tools/in_container debian:13                         # build and run the suite in a container (podman)
tools/in_container --arch arm64 debian:13            # another architecture (QEMU user mode registered with binfmt_misc)
CXX=clang++ CXXFLAGS=-stdlib=libc++ tools/in_container debian:13
python tools/golden_corpus generate corpus && CORPUS=$PWD/corpus tools/in_container alpine:latest
```
## Performance

Against [APyTypes](https://apytypes.org/) 0.5, which also has a C++ core (`bench/bench_apytypes.py`; both libraries are first checked to give the same bits):

| Operation | VeriFloat | APyTypes | Ratio |
|---|---|---|---|
| FP16 a + b | 6,269,372 op/s | 4,760,266 op/s | 1.3× |
| FP16 a * b | 6,600,551 op/s | 4,722,914 op/s | 1.4× |
| FP16 a / b | 6,322,948 op/s | 3,034,874 op/s | 2.1× |
| FP16 from float | 6,584,645 op/s | 5,037,951 op/s | 1.3× |
| FP32 → FP16 convert | 5,482,113 op/s | 5,300,719 op/s | 1.0× |
| FIXED s8.8 a * b | 5,440,237 op/s | 4,680,921 op/s | 1.2× |
| FP16 array a + b, 4096 elements | 480,935,842 el/s | 175,723,210 el/s | 2.7× |
| FP16 array a * b, 4096 elements | 721,297,211 el/s | 216,381,513 el/s | 3.3× |
| FP16 array a / b, 4096 elements | 112,384,109 el/s | 9,774,989 el/s | 11× |
| FP16 array from 4096 floats | 355,774,026 el/s | 43,197,736 el/s | 8.2× |
| FP32 array matmul 64×64 @ 64×16 | 524,984,774 el/s | 313,780,715 el/s | 1.7× |
| FP32 matmul 64×64 @ 64×16 on lists of FP | 295,891,136 el/s | 318,469,418 el/s | 0.9× |

Matrix products by accumulator model and kernel level (FP32 arrays, 64×64 @ 64×16, million multiply-adds per second):

| Accumulator | scalar kernels | AVX2 | AVX-512 |
|---|---|---|---|
| `Accumulator(FP32, product=FP32)` (round products, add in order) | 94 | 226 | 283 |
| `Accumulator(FP32)` (FMA chain) | 109 | 244 | 315 |
| `Accumulator(FP32, "pairwise", product=FP32)` | 63 | 166 | 215 |
| `FP32` (exact sum, one rounding) | 164 | 441 | 523 |
| `Accumulator(FP32, group=16, align_bits=24)` | about 100 | (scalar) | (scalar) |

See [Validation](#validation) for how each path is checked against the others and against exact arithmetic.

## Limitations

- **Format limits:** `exp_bits` ≤ 60, \|`bias`\| ≤ 2⁶⁰, `mantissa_bits` ≤ 2²⁴, `sr_bits` ≤ 2²⁰. Integer and fixed-point widths (`UINT`/`INT` bits, `to_int` bits, `FixedFormat` and `IntFormat` widths) are at most 2²⁴ bits. Version 0.1 had no limits, but formats near them were unusably slow there anyway.
- **Array limits:** an `FPArray` has at most 64 axes (NumPy's limit) and 2⁴⁸ elements; nested lists deeper than 64 levels, or that contain themselves, are refused. Larger requests raise `ValueError` before any memory is allocated.
- **Raw codes** passed to `from_raw` must fit the format: a code with bits above the format's width raises `ValueError`. (Version 0.1 and earlier 0.2 builds kept the low bits silently.)
- **Pickling** `FP`, `FPArray` and the other types works, but as with any pickle, load only data you trust: unpickling can run arbitrary code.
- **Missing:** decimal formats, and the IEEE transcendental functions (`exp`, `log`, `pow` ...).
- **`BlockTensor` is list-based** and suited to the tensor sizes of unit tests (hundreds to thousands of elements).
- **`FPArray`** holds formats of at most 64 bits, is immutable (indexing copies, there is no item assignment) and cannot be empty. Fancy indexing (index arrays, boolean masks) is not supported. Only `+ − × ÷`, conversion, construction and `matmul`/`dot` have fast kernels; the other element-wise methods and `sum` run the scalar operation per element. Mixing array formats works when the promoted format still fits 64 bits.
- **The C library** runs the general code only (no fast kernels), has no arrays, accumulators or block tensors, and its `uint64_t` entry points take formats of at most 64 bits (wider ones use the `128` and `_w` functions). The SystemVerilog package was tried with Verilator only.
- **SIMD kernels are x86-64 only** (AVX2 or AVX-512, GCC or Clang). Other platforms use the scalar kernels, with the same results (see [Platforms](#platforms)).
- **Compilers and targets:** GCC 11 or later, Clang, or MSVC (Visual Studio 2022), on a 64-bit target. Where the compiler has `__int128` the core uses it; elsewhere it uses `src/cpp/int128.hpp` (`-DVF_PORTABLE_INT128=ON` forces that, to test it). See [Platforms](#platforms) for what has been run.
- **Accumulator models** cover common structures (sequential, pairwise, aligned groups), but real accumulators vary. Check your RTL's documentation.
- **The default `FPFormat()`** is an IEEE-style E2M1, rarely what you want. Pass a format.

## When to use something else

Other projects cover parts of this space, and several are more mature:

| Project | Consider it when you need |
|---|---|
| [APyTypes](https://apytypes.org/) | Bit-accurate fixed- and floating-point arrays for hardware design, with broadcasting, slicing, reductions and convolution, and fixed-point arrays. The designed cases in six formats, five rounding modes |
| [ml_dtypes](https://github.com/jax-ml/ml_dtypes) | NumPy dtypes for the standard ML formats at NumPy speed. For the formats of 8 bits or fewer, every pair of encodings through `+ − × ÷`; for all of them, conversion on every rounding boundary |
| [gfloat](https://github.com/graphcore-research/gfloat) | Encoding/decoding of IEEE P3109, OCP FP8 and MX formats with many rounding modes. Every rounding boundary of each format (each midpoint and the doubles on either side of it) in the five modes |
| [pychop](https://pypi.org/project/pychop/) | Simulating low precision in numerical algorithms, with NumPy/PyTorch/JAX backends |
| [Berkeley SoftFloat / TestFloat](http://www.jhauser.us/arithmetic/) | The standard C reference and test-vector generator for IEEE binary16/32/64/128 |

VeriFloat's focus is narrower: a per-operation golden model for cocotb testbenches of FP4/FP8/custom-format datapaths. It gives raw codes and status flags per result, covers hardware options (saturate, wrap, FTZ/DAZ, NaN conventions, tininess), and models block-scaled tensors and accumulators. Every result is exact before its single rounding, and the C++ core still runs several million scalar operations per second (see [Performance](#performance)).
