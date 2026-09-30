"""Bus packing helpers: constrained random round trips."""

from __future__ import annotations

import pytest

from verifloat import E4M3, FP16, NVFP4, BlockTensor
from verifloat.bus import (pack, pack_block_row, pack_fp, unpack, unpack_block_row,
                           unpack_fp)
from reference import rand_tensor
from test_blockscale import rand_block_format


def test_pack_unpack_random(rng, iters):
    for _ in range(iters):
        width, n = rng.randint(1, 16), rng.randint(1, 20)
        codes = [rng.getrandbits(width) for _ in range(n)]
        msb = rng.random() < 0.5
        bus = pack(codes, width, msb_first=msb)
        assert bus < 1 << (width * n)
        assert unpack(bus, width, n, msb_first=msb) == codes
        # Lane 0 is in the low bits unless msb_first.
        lane0 = (bus >> ((n - 1) * width if msb else 0)) & ((1 << width) - 1)
        assert lane0 == codes[0]


def test_pack_anchor():
    assert pack([0x1, 0x2, 0x3], 4) == 0x321
    assert pack([0x1, 0x2, 0x3], 4, msb_first=True) == 0x123
    assert unpack(0x321, 4, 3) == [1, 2, 3]
    assert pack_fp([FP16(1), FP16(-2)]) == 0xC000_3C00
    assert [float(x) for x in unpack_fp(0xC000_3C00, FP16, 2)] == [1.0, -2.0]
    with pytest.raises(ValueError):
        pack([16], 4)
    with pytest.raises(ValueError):
        unpack(1 << 12, 4, 3)
    with pytest.raises(ValueError):
        pack_fp([FP16(1), E4M3(1)])


def test_block_row_roundtrip(rng, iters):
    for _ in range(max(iters // 20, 20)):
        bf = rand_block_format(rng)
        n = rng.randint(1, 40)
        t = BlockTensor.quantize(rand_tensor(rng, (n,))[0], bf)
        msb = rng.random() < 0.5
        buses = pack_block_row(t, msb_first=msb)
        assert buses.elem_width == bf.elem.size
        assert unpack_block_row(buses, bf, n, msb_first=msb) == t


def test_nvfp4_row_layout():
    t = BlockTensor.quantize([[6, 3, 1.5, 0, -0.5, -6] * 3, [1] * 18], NVFP4)
    b = pack_block_row(t, row=0)
    assert (b.elem_width, b.scale_width) == (4, 8)
    assert b.elems & 0xFFFFFF == 0xF90357            # codes 7,5,3,0,9,15, lane 0 low
    assert unpack(b.scales, 8, 2) == t.scale_raw[0]
    assert b.tensor_scale == t.tensor_scale.raw
