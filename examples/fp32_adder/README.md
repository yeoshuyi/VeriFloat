# Example: verifying an open-source FP32 adder with cocotb

This testbench checks the single-precision adder from
[dawsonjon/fpu](https://github.com/dawsonjon/fpu) (MIT licence, Verilog)
against VeriFloat's `FP32` format as the golden model.

```bash
uv sync --group examples                       # installs cocotb
uv run --group examples python examples/fp32_adder/run.py
N_VECTORS=20000 SEED=7 uv run --group examples python examples/fp32_adder/run.py
```

Requirements: Verilator 5.x on `PATH`. `run.py` downloads `adder.v` at a pinned
commit into `sim_build/`, so the RTL isn't copied into this repository.

The stimulus is constrained random: specials (±0, ±inf, NaN), subnormals,
operands of similar magnitude (which exercise cancellation and rounding), and
uniformly random bit patterns. Results must match bit for bit. The one
exception is NaN: the adder returns `0xFFC00000`, while VeriFloat's canonical
NaN is `0x7FC00000`, so any NaN is accepted when the model's result is NaN.

With the vectors we ran (2,000 with seed 1 and 20,000 with seed 7) there were
no mismatches. As a sanity check, switching the model to round-toward-zero
makes the testbench report mismatches immediately. Passing random vectors
isn't a proof of correctness. The adder has no status-flag outputs, so flags
aren't checked here.
