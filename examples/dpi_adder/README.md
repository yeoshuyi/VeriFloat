# Example: a SystemVerilog testbench with VeriFloat through DPI-C

The same check as [`fp32_adder`](../fp32_adder), without Python in the loop:
`tb.sv` drives the single-precision adder from
[dawsonjon/fpu](https://github.com/dawsonjon/fpu) (MIT licence, Verilog) and
compares every sum with `vf_add`, the VeriFloat model called through DPI-C.

```systemverilog
import verifloat_pkg::*;

chandle fp32 = vf_format("e8m23");            // once; any format the Python package can name
want = vf_add(fp32, 64'(a), 64'(b), flags);   // raw code and IEEE flags
```

```bash
uv sync
uv run python examples/dpi_adder/run.py
uv run python examples/dpi_adder/run.py +n=20000 +seed=7
uv run python examples/dpi_adder/run.py +rounding=rtz     # a wrong model: mismatches are reported
```

Requirements: Verilator 5.x on `PATH`. `run.py` downloads `adder.v` at a pinned
commit into `sim_build/`, then builds with the paths `python -m verifloat.dpi`
prints:

```bash
verilator --binary --timing tb.sv adder.v $(python -m verifloat.dpi --sv) \
    -CFLAGS "$(python -m verifloat.dpi --cflags)" -LDFLAGS "$(python -m verifloat.dpi --ldflags)"
```

With the vectors we ran (2,000 with seed 1 and 20,000 with seed 7) there were
no mismatches; with `+rounding=rtz` the model rounds toward zero and about
45% of the sums differ. As in the cocotb example, any NaN is accepted where
the model's result is NaN (the adder returns `0xFFC00000`, the model's
canonical NaN is `0x7FC00000`), and flags aren't compared because the adder
has no flag outputs. Passing random vectors isn't a proof of correctness.

The testbench changes the adder's inputs on the falling clock edge and uses
blocking assignments: Verilator runs non-blocking assignments in an `initial`
block as blocking ones, which would race with the adder's rising-edge logic.
