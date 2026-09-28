# customtypes

Bit-accurate fixed-width integers and custom minifloats for Python.

- **`UINT`**: unsigned integer of any width, wraps modulo 2ⁿ
- **`INT`**: two's-complement signed integer of any width
- **`FP`**: IEEE-style minifloat with a configurable exponent and mantissa width. The default is **NVFP4 / E2M1**.

It is pure Python with no dependencies. Arithmetic is exact (FP rounding goes through `fractions.Fraction`), so results match the hardware bit for bit. That makes it useful as a golden reference model for RTL, quantization experiments and number-format exploration.

## Installation

```bash
pip install customtypes
# or
uv add customtypes
```

Requires Python 3.13+.

## Quick start

```python
from customtypes import UINT, INT, FP

UINT(250, 8) + 10          # UINT(4, u8)       wraps modulo 2**8
INT(127, 8) + 1            # INT(-128, i8)     two's-complement overflow
FP.from_value(2.5)         # FP(2.0, e2m1)     round-to-nearest-even in NVFP4

UINT(0, 8).size            # 8
INT(0, 5).size             # 5
FP.from_value(1).size      # 4  (1 sign + 2 exponent + 1 mantissa)
```

Every type has a `size` property that gives the number of bits the value occupies in storage.

## `UINT`: unsigned integers

```python
UINT(val=0, bits=32)
```

The value is masked to `bits` on construction, and every operation wraps modulo `2**bits`.

```python
a = UINT(250, 8)
a + 10                 # UINT(4, u8)
UINT(3, 8) - 5         # UINT(254, u8)
~UINT(0, 4)            # UINT(15, u4)
UINT(1, 8) << 9        # UINT(0, u8)
UINT(7, 8) // 2        # UINT(3, u8)
UINT(0x1FF, 12).resize(8)   # UINT(255, u8)  truncates
```

| Attribute | Meaning |
|---|---|
| `bits` / `size` | width in bits |
| `val` | numeric value (`int`) |
| `raw` | underlying bit pattern (`int`) |
| `mask` | `2**bits - 1` |
| `min`, `max` | representable range |
| `to_bin()` | zero-padded binary string |
| `resize(bits)` | new value of a different width (truncates, or zero-extends) |

Supported operators: `+ - * // %`, `& | ^ ~ << >>`, and comparisons. The other operand can be a plain `int`. `int()`, `bool()` and `operator.index()` work, so a `UINT` can be used as a list index.

## `INT`: signed integers

`INT` is a subclass of `UINT` with two's-complement semantics.

```python
INT(200, 8)            # INT(-56, i8)
INT(-1, 8).raw         # 255
INT(-2, 4).to_bin()    # '1110'
INT(-8, 8) >> 1        # INT(-4, i8)   arithmetic shift
INT(-7, 8) // 2        # INT(-3, i8)   division truncates toward zero (C-style)
INT(-7, 8) % 2         # INT(-1, i8)   remainder takes the sign of the dividend
INT(-1, 4).resize(8)   # INT(-1, i8)   sign-extends
```

### Mixing widths and signedness

When two integers of different widths meet, the result takes the **wider** width. When they differ in signedness, the operation follows C rules: if either operand is `UINT`, both are treated as their raw bit patterns and the result is `UINT`.

```python
INT(-1, 8) + INT(1, 4)     # INT(0, i8)
UINT(200, 8) + INT(-1, 8)  # UINT(199, u8)   (-1 is read as 0xFF)
```

## `FP`: custom minifloats

```python
FP(sign: bool, exp: UINT, mantissa: UINT)
```

The layout is IEEE-754: sign, then a biased exponent (bias = `2**(E-1) - 1`), then a mantissa with an implicit leading 1. Subnormals are used when the exponent field is 0. In practice you usually build values with the constructors:

```python
FP.from_value(value, exp_bits=2, mantissa_bits=1)   # round an int/float/Fraction/FP
FP.from_raw(raw, exp_bits=2, mantissa_bits=1)       # decode a bit pattern
FP.zero(exp_bits, mantissa_bits, sign=False)
FP.max_value(exp_bits, mantissa_bits, sign=False)
x.convert(exp_bits, mantissa_bits)                  # re-round into another format
```

```python
y = FP.from_value(0.1, 4, 3)   # FP(0.1015625, e4m3)
y.size, y.bias                 # (8, 7)
y.convert(2, 1)                # FP(0.0, e2m1)

x = FP.from_value(2.5)         # FP(2.0, e2m1)
x.raw, x.to_bin()              # (4, '0 10 0')
float(x), int(x)               # (2.0, 2)
```

### Semantics

- **Rounding:** round-to-nearest, ties-to-even, computed exactly. Each operation is rounded once.
- **No infinities or NaN:** every bit pattern is a finite number. Out-of-range results **saturate** to ±max. Passing `nan` or `inf` to `from_value` raises `ValueError`.
- **Signed zero:** `-0.0` is preserved (`FP.from_value(-0.0).raw == 0b1000`). It compares equal to `+0.0`, and IEEE sign rules apply to zero results (`1 + -1` is `+0`, `-0 + -0` is `-0`).
- **Division by zero** raises `ZeroDivisionError`.
- **Mixed formats:** both operands are widened to `max(E)`, `max(M)` (an exact conversion) before the operation, and the result is in that format. Plain numbers are first rounded into the FP operand's format.
- **Bitwise ops** (`& | ^ ~`) act on the raw bit patterns.

```python
FP.from_value(1) + FP.from_value(0.125, 4, 3)   # FP(1.125, e4m3)
FP.from_value(6) + 6                            # FP(6.0, e2m1)   saturates
FP.from_value(3) * 1.5                          # FP(4.0, e2m1)   4.5 rounds to 4
~FP.from_raw(0b0111)                            # FP(-0.0, e2m1)
```

| Attribute | Meaning |
|---|---|
| `size` | total width: `1 + exp_bits + mantissa_bits` |
| `sign`, `exp`, `mantissa` | fields (`bool`, `UINT`, `UINT`) |
| `exp_bits`, `mantissa_bits`, `bias` | format parameters |
| `raw` | packed bit pattern (`int`) |
| `is_zero`, `is_subnormal` | classification |
| `to_bin()` | `'s eee mmm'` string |

### NVFP4 (E2M1)

This is the default format. Its 16 codes are:

| code (magnitude bits) | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|---|---|---|---|---|---|---|---|---|
| value | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |

Bit 3 is the sign. Other common formats: `FP.from_value(x, 4, 3)` for FP8 E4M3 (without the OCP NaN encoding) and `FP.from_value(x, 5, 2)` for E5M2 (without inf/NaN).

## Development

```bash
uv sync
uv run pytest
```

The tests use **constrained random** stimulus: the widths, values and FP formats are drawn at random but biased toward edge cases (0, ±max, min, subnormals, rounding midpoints). The results are checked against an independent reference model (`tests/reference.py`).

- `tests/test_generic.py`: `UINT`, `INT`, mixed signedness and FP across random formats
- `tests/test_nvfp4.py`: dedicated tests for NVFP4 / E2M1

Each run prints its seed in the pytest header, and any failure prints a reproduce command. Control the tests with environment variables:

```bash
CUSTOMTYPES_SEED=1234 uv run pytest     # reproduce a run
CUSTOMTYPES_ITERS=50000 uv run pytest   # more random cases per test (default 2000)
```

## License

MIT. See [LICENSE](LICENSE).
