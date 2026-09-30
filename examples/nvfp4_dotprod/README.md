# Example: an open-source NVFP4 dot-product core

This testbench checks the NVFP4 mode of the dot-product core from
[sashakatne/nvfp4-dotprod-formal-dv](https://github.com/sashakatne/nvfp4-dotprod-formal-dv)
(Apache-2.0, SystemVerilog) against VeriFloat.

Each operand is one NVFP4 block: 16 E2M1 elements plus a UE4M3 scale. The
core returns the FP32 dot product. Its documented contract is that the result
is exact, and the golden model checks this too: `dot(a, b, acc=FP32)` must
raise no flags. A NaN scale must give the canonical FP32 quiet NaN and set
`invalid` and `is_nan`.

```bash
uv sync --group examples
uv run --group examples python examples/nvfp4_dotprod/run.py
N_VECTORS=20000 SEED=7 uv run --group examples python examples/nvfp4_dotprod/run.py
```

This needs Verilator 5.x. `run.py` downloads the RTL at a pinned commit into
`sim_build/`. `dotprod_wrap.sv` (part of this example) flattens the core's
array ports into two 128-bit buses, and `verifloat.bus.pack_block_row` fills them.

The stimulus mixes raw codes (every element pattern, occasional NaN scales)
with blocks quantized from random reals. In the runs we did (3,000 pairs with
seed 1 and 20,000 with seed 7) there were no mismatches in result bits or
status. Accumulating in BF16 instead of FP32 makes the testbench report
mismatches immediately. Only the NVFP4 mode is exercised here, not the core's
INT8 or BF16 modes.
