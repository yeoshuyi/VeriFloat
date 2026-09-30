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

from verifloat import E2M1, FP32, UE4M3, BlockFormat, BlockTensor, dot
from verifloat.bus import pack_block_row

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
    mismatches = 0
    for i in range(N):
        a, b = random_block(rng), random_block(rng)
        dut.a_flat.value = operand_bus(a)
        dut.b_flat.value = operand_bus(b)
        await Timer(1, unit="ns")
        got = FP32.from_raw(dut.result.value.to_unsigned())
        status = dut.status.value.to_unsigned()          # {sat, invalid, is_nan, is_inf}

        if UE4M3.from_raw(a.scale_raw[0]).is_nan or UE4M3.from_raw(b.scale_raw[0]).is_nan:
            # A NaN scale makes the block NaN: the core returns FP32's
            # canonical quiet NaN and flags invalid.
            want_raw, want_status = 0x7FC00000, 0b0110
        else:
            want = dot(a, b, acc=FP32)                   # exact sum, one FP32 rounding
            assert not want.flags, "the core's contract: the result is exact"
            want_raw, want_status = want.raw, 0
        if (got.raw, status) != (want_raw, want_status):
            mismatches += 1
            dut._log.error(f"#{i}: a={a.elem_raw} s={a.scale_raw} b={b.elem_raw} "
                           f"s={b.scale_raw}: RTL {got.to_hex()} status {status:04b}, "
                           f"model {want_raw:08x} status {want_status:04b}")
    dut._log.info(f"{N} NVFP4 block pairs, {mismatches} mismatches (seed {SEED})")
    assert mismatches == 0
