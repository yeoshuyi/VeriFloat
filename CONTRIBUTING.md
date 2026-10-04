# Contributing to VeriFloat

Bug reports, format requests and pull requests are welcome at <https://github.com/yeoshuyi/VeriFloat/issues>.

VeriFloat is a golden model, so a change to a result must be backed by a reference that VeriFloat did not write: every new or changed operation is checked against an outside oracle (Berkeley SoftFloat/TestFloat, MPFR, the host FPU, ml_dtypes, gfloat or APyTypes), and fast paths must give what the general code gives, bit for bit. [docs/README.md](docs/README.md#validation) lists what each reference covers.

## Development

```bash
uv sync                                  # dev tools: pytest, numpy, ml_dtypes, gfloat, apytypes, gmpy2
uv run pytest
VERIFLOAT_SEED=1234 uv run pytest        # reproduce a run (the seed is printed in the header)
VERIFLOAT_ITERS=20000 uv run pytest      # more random cases per test
VERIFLOAT_IMPL=py uv run pytest          # the same suite against the frozen pure-Python 0.1
uv run pytest -k "not test_edge"        # skip the edge grids (the full suite takes 15 to 25 minutes)
VERIFLOAT_FAST=0 uv run pytest           # the C++ core on its general path only (no fast kernels)
VERIFLOAT_SIMD=avx2 uv run pytest        # a SIMD level: none, avx2 or avx512
VERIFLOAT_MEM_GB=4 uv run pytest         # memory cap of the test process (default 8 GiB, 0 for none)
MEM=10G tools/capped uv run pytest       # the same under a hard limit: a Slurm job, or a systemd scope
uv run python bench/bench.py             # throughput, C++ core vs pure-Python 0.1
uv run python bench/bench_apytypes.py    # throughput against APyTypes
```

The C++ core is in `src/cpp`, the Python package in `src/verifloat`, and the tests in `tests` (C and C++ helpers the tests compile are in `tests/native`). The frozen pure-Python 0.1 in `archive/python` is the oracle for `tests/test_parity.py`.

## Checking a change across platforms

```bash
python tools/golden_corpus generate corpus           # 1.4 million result vectors from the build at hand
python tools/golden_corpus verify corpus             # recompute them with another build: every bit and flag must match
tools/in_container debian:13                         # build and run the suite in a container (podman)
tools/in_container --arch arm64 debian:13            # another architecture (QEMU user mode)
pip install -C cmake.define.VF_PORTABLE_INT128=ON .  # the portable 128-bit integers that MSVC builds use
```

CI (`.github/workflows/tests.yml`) runs the suite and the corpus on Linux x86-64 and ARM, macOS (Apple silicon and Intel), Windows with MSVC and with MinGW-w64, and Linux with the portable 128-bit integers.
