# Changelog

All notable changes to VeriFloat. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, a minor version may
change the API).

## [0.2.0] - unreleased

The first release on PyPI. The library is rewritten on a C++ core; the API of
0.1 is kept, and 0.1 itself is kept in `archive/python` as the reference the
core is tested against.

### Added

- A C++ core (nanobind) for every type: the same results as 0.1, bit for
  bit, several million scalar operations per second.
- `FPArray`: N-D arrays of FP values with NumPy input and output, slicing,
  broadcasting, element-wise methods, `sum` and `matmul`, on SIMD kernels
  (AVX2, AVX-512) where the CPU has them.
- IEEE 754 operations: `remainder`, `fmod`, `next_up`, `next_down`, `scaleb`,
  `logb`, `fclass`, `copysign` and the RISC-V sign injections, `is_normal`.
- `verifloat.scoreboard`: compares DUT outputs with the model and explains
  each mismatch.
- `verifloat.vectors`: test-vector files (TestFloat, CSV, JSON lines,
  `$readmemh`) for any format and operation, written and checked.
- `libverifloat`, the same core as a C library, and a SystemVerilog DPI-C
  package (`verifloat_pkg.sv`); `python -m verifloat.dpi` prints the paths
  and flags.
- Wheels for Linux (x86-64 and aarch64, glibc and musl), macOS (Apple silicon
  and Intel) and Windows (x86-64), Python 3.12 to 3.14.

### Changed

- `from_raw` (on `FPFormat`, `FP` and `FPArray`) refuses a code that does not
  fit the format (negative, or with bits above its width) with `ValueError`;
  0.1 kept the low bits. A signed NumPy array exactly as wide as the format is
  read as bit patterns (FP16 codes in `int16`).
- Format limits: `exp_bits` ≤ 60, |`bias`| ≤ 2⁶⁰, `mantissa_bits` ≤ 2²⁴,
  `sr_bits` ≤ 2²⁰; integer and fixed-point widths at most 2²⁴ bits.
- Arrays have at most 64 axes and 2⁴⁸ elements; nested lists deeper than 64
  levels, or that contain themselves, are refused.

### Fixed

- Results no longer depend on the processor's floating-point state:
  flush-to-zero / denormals-are-zero (set process-wide by any library built
  with `-ffast-math`), the rounding mode, or x87 registers left in use. The
  core reads and writes floats through their bits and takes square roots
  with integers only.
- Big-endian hosts accept `array.array` and `memoryview` buffers.
- Builds with MSVC, MinGW-w64 and on macOS.
- Shapes whose element count overflowed, and lists that contain themselves,
  could crash the interpreter or read outside an array; they now raise.

### Security

- CI actions are pinned to commits, the Boost dependency to a commit, and
  every file the tests and examples download to a SHA-256. See
  [SECURITY.md](SECURITY.md) for how to report a problem.

## [0.1.0]

The pure-Python implementation (git tag `python-v0.1.0`; not published on
PyPI), kept in `archive/python` as `verifloat_py`.

[0.2.0]: https://github.com/yeoshuyi/VeriFloat/compare/python-v0.1.0...v0.2.0
[0.1.0]: https://github.com/yeoshuyi/VeriFloat/releases/tag/python-v0.1.0
