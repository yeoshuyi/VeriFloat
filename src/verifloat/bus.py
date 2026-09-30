"""Pack VeriFloat codes onto flat buses and back, for cocotb (or any simulator
that exposes a bus as one integer).

Lane 0 sits in the least significant bits by default, which matches a
Verilog port declared as ``input [N*W-1:0] bus`` indexed as ``bus[i*W +: W]``.
None of this imports cocotb: ``dut.sig.value = pack(...)`` and
``unpack(int(dut.sig.value), ...)`` work with any version.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .blockscale import BlockFormat, BlockTensor
from .fp import FP, FPFormat
from .uint import UINT


def pack(codes: Iterable[int], width: int, *, msb_first: bool = False) -> int:
    """Concatenate lane codes of ``width`` bits into one integer."""
    codes = list(codes)
    if msb_first:
        codes.reverse()
    bus, mask = 0, (1 << width) - 1
    for i, c in enumerate(codes):
        c = int(c.raw) if isinstance(c, (FP, UINT)) else int(c)
        if c & ~mask:
            raise ValueError(f"lane {i} code {c:#x} does not fit in {width} bits")
        bus |= c << (i * width)
    return bus


def unpack(bus: int, width: int, count: int, *, msb_first: bool = False) -> list[int]:
    """Split an integer bus into ``count`` lane codes of ``width`` bits."""
    bus = int(bus)
    if bus >> (width * count):
        raise ValueError(f"bus value has bits above {width * count}")
    mask = (1 << width) - 1
    codes = [(bus >> (i * width)) & mask for i in range(count)]
    return codes[::-1] if msb_first else codes


def pack_fp(values: Iterable[FP], fmt: FPFormat | None = None, **kw) -> int:
    """Pack FP values (all in one format) onto a bus."""
    values = list(values)
    fmt = fmt or values[0].format
    if any(v.format != fmt for v in values):
        raise ValueError(f"all values must be in {fmt}")
    return pack((v.raw for v in values), fmt.size, **kw)


def unpack_fp(bus: int, fmt: FPFormat, count: int, **kw) -> list[FP]:
    """Decode a bus of ``count`` FP lanes."""
    return [fmt.from_raw(c) for c in unpack(bus, fmt.size, count, **kw)]


@dataclass(frozen=True)
class BlockBuses:
    """One row of a block tensor as bus integers, with their widths."""
    elems: int
    scales: int | None
    zeros: int | None
    tensor_scale: int | None
    elem_width: int
    scale_width: int | None
    zero_width: int | None


def _width(f) -> int | None:
    return None if f is None else f.size


def pack_block_row(t: BlockTensor, row: int = 0, **kw) -> BlockBuses:
    """Pack one row (or a 1-D tensor) of a BlockTensor: all elements on one
    bus, all block scales on another, plus zero-points and the tensor scale."""
    fmt = t.fmt
    rows = lambda x: [x] if t.ndim == 1 else x
    elems = rows(t.elem_raw)[row]
    scales = rows(t.scale_raw)[row] if fmt.scale is not None else None
    zeros = rows(t.zero_raw)[row] if fmt.zero_point is not None else None
    return BlockBuses(
        elems=pack(elems, _width(fmt.elem), **kw),
        scales=None if scales is None else pack(scales, _width(fmt.scale), **kw),
        zeros=None if zeros is None else pack(zeros, _width(fmt.zero_point), **kw),
        tensor_scale=None if t.tensor_scale is None else t.tensor_scale.raw,
        elem_width=_width(fmt.elem), scale_width=_width(fmt.scale),
        zero_width=_width(fmt.zero_point))


def unpack_block_row(buses: BlockBuses | dict, fmt: BlockFormat, length: int,
                     **kw) -> BlockTensor:
    """Rebuild a 1-D BlockTensor of ``length`` elements from bus integers
    (e.g. read back from the DUT)."""
    b = buses if isinstance(buses, dict) else buses.__dict__
    nblocks = -(-length // fmt.block_size)
    elems = unpack(b["elems"], _width(fmt.elem), length, **kw)
    scales = (None if fmt.scale is None
              else unpack(b["scales"], _width(fmt.scale), nblocks, **kw))
    zeros = (None if fmt.zero_point is None
             else unpack(b["zeros"], _width(fmt.zero_point), nblocks, **kw))
    return BlockTensor.from_raw(elems, scales, fmt, zeros, b.get("tensor_scale"))
