"""cocotb testbench: dawsonjon/fpu single-precision adder vs. VeriFloat FP32.

The adder (https://github.com/dawsonjon/fpu, MIT) takes two IEEE 754
binary32 operands over valid/ack handshakes and returns their sum rounded to
nearest-even. VeriFloat's FP32 format is the golden model.
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge

from verifloat import FP32

N = int(os.environ.get("N_VECTORS", 2000))
SEED = int(os.environ.get("SEED", 1))


def stimulus(rng: random.Random) -> int:
    """Constrained random binary32 encodings: specials, subnormals,
    cancellation-prone values and uniformly random bits."""
    r = rng.random()
    sign = rng.getrandbits(1) << 31
    if r < 0.1:
        return sign | rng.choice([0, 1, 0x007FFFFF, 0x00800000, 0x7F7FFFFF,
                                  0x7F800000, 0x7FC00000, 0x3F800000])
    if r < 0.5:
        return sign | (rng.randint(120, 134) << 23) | rng.getrandbits(23)
    if r < 0.6:
        return sign | (rng.randint(0, 2) << 23) | rng.getrandbits(23)
    return rng.getrandbits(32)


async def send(dut, data, stb, ack, value):
    """Drive a valid/ack handshake until the DUT accepts the word."""
    data.value = value
    stb.value = 1
    while True:
        await ReadOnly()
        accepted = ack.value == 1
        await RisingEdge(dut.clk)
        if accepted:
            break
    stb.value = 0


async def receive(dut) -> int:
    while True:
        await ReadOnly()
        if dut.output_z_stb.value == 1:
            z = dut.output_z.value.to_unsigned()
            await RisingEdge(dut.clk)       # ack is held high: handshake done
            return z
        await RisingEdge(dut.clk)


@cocotb.test()
async def adder_matches_verifloat(dut):
    Clock(dut.clk, 10, unit="ns").start()
    dut.rst.value = 1
    dut.input_a_stb.value = 0
    dut.input_b_stb.value = 0
    dut.output_z_ack.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0

    rng = random.Random(SEED)
    mismatches = []
    for i in range(N):
        a_bits, b_bits = stimulus(rng), stimulus(rng)
        await send(dut, dut.input_a, dut.input_a_stb, dut.input_a_ack, a_bits)
        await send(dut, dut.input_b, dut.input_b_stb, dut.input_b_ack, b_bits)
        got_bits = await receive(dut)

        want = FP32.from_raw(a_bits) + FP32.from_raw(b_bits)   # golden model
        got = FP32.from_raw(got_bits)
        # The adder returns a negative quiet NaN (0xFFC00000); any NaN is
        # accepted for NaN results, everything else must match bit for bit.
        ok = got.is_nan if want.is_nan else got.raw == want.raw
        if not ok:
            mismatches.append((a_bits, b_bits, got_bits, want))
            dut._log.error(
                f"#{i}: {a_bits:#010x} + {b_bits:#010x}: RTL {got.to_hex()} "
                f"({float(got)!r}) != model {want.to_hex()} ({float(want)!r})"
                + (f", model error {float(want.error_ulps()):.3g} ulp,"
                   f" RTL error {float(got.error_ulps(want.unrounded)):.3g} ulp"
                   if got.is_finite and want.is_finite else ""))
    dut._log.info(f"{N} vectors, {len(mismatches)} mismatches (seed {SEED})")
    assert not mismatches, f"{len(mismatches)} mismatches, first: {mismatches[0][:3]}"
