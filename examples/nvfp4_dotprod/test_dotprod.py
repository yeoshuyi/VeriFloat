"""cocotb testbench: an open-source NVFP4 dot-product core vs VeriFloat.

The core (github.com/sashakatne/nvfp4-dotprod-formal-dv, Apache-2.0) takes one
NVFP4 block per operand: 16 E2M1 elements and a UE4M3 scale, and returns the
FP32 dot product (exact: the design's contract is that no bit is dropped).
The golden model is a VeriFloat BlockTensor per operand and dot(..., acc=FP32).
"""

import os
import random

import cocotb
from cocotb.triggers import Timer

from verifloat import E2M1, FP32, UE4M3, BlockFormat, BlockTensor, FPFlags, dot
from verifloat.bus import pack_block_row
from verifloat.scoreboard import Scoreboard

N = int(os.environ.get("N_VECTORS", 3000))
SEED = int(os.environ.get("SEED", 1))
NVFP4 = BlockFormat(E2M1, 16, UE4M3)          # the core's operand format
FMT_NVFP4 = 2


def operand_bus(t: BlockTensor) -> int:
    """16 x 4-bit elements in lanes 0-3, the 8-bit scale in lane 4."""
    b = pack_block_row(t)
    return b.elems | (b.scales << 64)


def random_block(rng):
    if rng.random() < 0.5:               # every code pattern, via raw codes
        scale = rng.choice([0x7F] if rng.random() < 0.05 else list(range(0x7F)))
        return BlockTensor.from_raw([rng.getrandbits(4) for _ in range(16)], [scale], NVFP4)
    e = rng.randint(-12, 8)              # or quantized from random reals
    return BlockTensor.quantize([rng.uniform(-1, 1) * 2.0 ** e for _ in range(16)], NVFP4)


@cocotb.test()
async def nvfp4_dot_matches_verifloat(dut):
    rng = random.Random(SEED)
    dut.mode.value = FMT_NVFP4
    # The result is compared bit for bit. status is {sat, invalid, is_nan,
    # is_inf}: its invalid bit (bit 2) is compared as a VeriFloat flag, the
    # other three are derived from the model's result below.
    sb = Scoreboard("nvfp4_dotprod", fail_fast=False, logger=dut._log,
                    flag_map={FPFlags.INVALID: 2})
    status_errors = 0
    for i in range(N):
        a, b = random_block(rng), random_block(rng)
        dut.a_flat.value = operand_bus(a)
        dut.b_flat.value = operand_bus(b)
        await Timer(1, unit="ns")

        if UE4M3.from_raw(a.scale_raw[0]).is_nan or UE4M3.from_raw(b.scale_raw[0]).is_nan:
            # A NaN scale makes the block NaN: the core returns FP32's
            # canonical quiet NaN and flags invalid. inf - inf is that result.
            want = FP32.inf() - FP32.inf()
        else:
            want = dot(a, b, acc=FP32)                   # exact sum, one FP32 rounding
            assert not want.flags, "the core's contract: the result is exact"
        sb.check(want, dut.result, flags=dut.status, op="dot", a_elems=a.elem_raw,
                 a_scale=a.scale_raw, b_elems=b.elem_raw, b_scale=b.scale_raw)

        status = dut.status.value.to_unsigned()
        if status & 0b1011 != (int(want.is_nan) << 1 | int(want.is_inf)):
            status_errors += 1
            dut._log.error(f"#{i}: status {status:04b}: is_nan/is_inf/sat disagree "
                           f"with the model result {want.to_hex()}")
    dut._log.info(f"{sb.summary().splitlines()[0]} (seed {SEED}), "
                  f"{status_errors} status errors")
    sb.assert_clean()
    assert status_errors == 0
