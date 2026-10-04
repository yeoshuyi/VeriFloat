"""Hostile and malformed inputs: each is refused with an exception.

Every case here once crashed the interpreter, read memory outside an array,
or would have taken memory without bound: shapes whose element count
overflows, lists nested without end or containing themselves, widths that
allocate gigabytes, raw codes that do not fit their format, and private
helpers called with inconsistent arguments. A regression shows up as a
crash of the test process, or as a test that fails to see an error.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

if os.environ.get("VERIFLOAT_IMPL", "cpp") != "cpp":
    pytest.skip("these limits are part of the C++ package", allow_module_level=True)

from verifloat import (FP16, NVFP4, BlockTensor, FixedFormat, FPArray, INT, UINT,  # noqa: E402
                       _core)
from verifloat.blockscale import IntFormat  # noqa: E402

TOO_LARGE = "too large|too many dimensions"


# ---------------------------------------------------------------- shapes

@pytest.mark.parametrize("shape", [
    (3, 0x5555555555555556),      # 3 * that = 2 (mod 2**64): the old check passed
    (2**62, 4),                   # 2**64 elements: wrapped to 0
    (2**25, 2**24),               # 2**49: above the limit without overflowing
])
def test_shapes_whose_size_overflows_are_refused(shape):
    a = FP16.array([1.0, 2.0])
    with pytest.raises(ValueError, match=TOO_LARGE + "|cannot reshape"):
        a.reshape(*shape)
    with pytest.raises(ValueError, match=TOO_LARGE + "|cannot broadcast"):
        FP16.array([1.0]).broadcast_to(shape)
    with pytest.raises(ValueError, match=TOO_LARGE):
        _core.array_restore(FP16, shape, [], b"")


def test_at_most_64_axes():
    a = FP16.array([1.0, 2.0])
    assert a.reshape(*([1] * 63 + [2])).tolist()      # NumPy's limit is allowed
    with pytest.raises(ValueError, match="too many dimensions"):
        a.reshape(*([1] * 64 + [2]))
    with pytest.raises(ValueError, match="too many dimensions"):
        a[(None,) * 64]


def test_numpy_view_claiming_a_huge_size():
    np = pytest.importorskip("numpy")
    def view(dtype):   # 2**49 elements over one: a broadcast view, no memory
        return np.lib.stride_tricks.as_strided(np.zeros(1, dtype), shape=(2**25, 2**24), strides=(0, 0))

    with pytest.raises(ValueError, match=TOO_LARGE):
        FPArray(view(np.float32), FP16)
    with pytest.raises(ValueError, match=TOO_LARGE):
        FPArray.from_raw(view(np.uint16), FP16)


def test_slices_with_huge_steps():
    a = FP16.array([[1.0, 2.0, 3.0]])
    assert a[:, ::2**62].raw == [[FP16(1.0).raw]]
    assert a[:, ::-(2**62)].raw == [[FP16(3.0).raw]]


# ---------------------------------------------------------------- nested lists

def _self_containing():
    x = [1.0]
    x[0] = x
    return x


def _deep(n):
    x = [1.0]
    for _ in range(n):
        x = [x]
    return x


@pytest.mark.parametrize("make", [_self_containing, lambda: _deep(10**5)], ids=["self", "deep"])
def test_lists_nested_without_end_are_refused(make):
    values = make()
    with pytest.raises(ValueError, match="too many dimensions"):
        FPArray(values, FP16)
    with pytest.raises(ValueError, match="too many dimensions"):
        FPArray.from_raw(values, FP16)
    with pytest.raises(ValueError, match="too many dimensions"):
        BlockTensor.quantize(values, NVFP4)


def test_conversion_that_empties_its_own_list():
    """A value whose conversion clears the list being converted: the other
    values are held, so they are read intact."""
    values = []

    class Clearing(Fraction):
        @property
        def numerator(self):
            values.clear()
            return super().numerator

    values.extend([Clearing(1, 2)] + [float(i) + 0.5 for i in range(200)])
    a = FPArray(values, FP16)
    assert a.shape == (201,) and a[200] == FP16(199.5)


def test_private_helpers_check_their_arguments():
    with pytest.raises(ValueError):
        _core.permute([1, 2, 3], (2, 2), (1, 0))           # 3 values for 4 elements
    with pytest.raises(ValueError):
        _core.permute([1, 2, 3, 4], (2, 2), (1, 1))        # not a permutation
    with pytest.raises(ValueError):
        _core.permute([1, 2, 3, 4], (2, 2), (0, 5))        # no such axis
    assert _core.permute([1, 2, 3, 4], (2, 2), (1, 0)) == [1, 3, 2, 4]
    with pytest.raises(ValueError):
        _core.array_from_fps([FP16(1)] * 2, (3, 0x5555555555555556))
    with pytest.raises(ValueError):
        _core.array_restore(FP16, (2,), [1], b"\0\0")     # 1 code for 2 elements
    with pytest.raises(ValueError):
        FP16._setup(5, 10, 15, True, 1, True, False, False, False, 99, 8, 0, False, 0)
    spec = NVFP4._native()
    values = [1.0] * 16
    with pytest.raises(ValueError):
        _core.block_quantize(values, (32,), 0, spec)            # shape claims 32 values
    with pytest.raises(ValueError):
        _core.block_quantize(values, (16,), 5, spec)            # no axis 5
    assert len(_core.block_quantize(values, (16,), 0, spec)) == 5     # consistent arguments still work
    ones = [FP16(1)] * 4
    with pytest.raises(ValueError):
        _core.matmul_reduce(ones, ones, 1, 2, 3, 2, False, False, None)   # 2x3 @ 3x2 needs 6 each
    with pytest.raises(ValueError):
        _core.matmul_reduce(ones, ones, 1, 2**40, 2**40, 1, False, False, None)
    assert _core.matmul_reduce(ones, ones, 1, 2, 2, 2, False, False, None) == [2, 2, 2, 2]


# ---------------------------------------------------------------- widths

def test_widths_beyond_the_limit_are_refused():
    with pytest.raises(OverflowError):
        UINT(1, 10**10)
    with pytest.raises(OverflowError):
        INT(1, 2**24 + 1)
    with pytest.raises(OverflowError):
        FP16(1.0).to_int(10**10)
    with pytest.raises(OverflowError):
        FixedFormat(10**9, 0)
    with pytest.raises(OverflowError):
        FixedFormat(10**9 + 4, -(10**9))          # 4 bits wide, from huge parts
    with pytest.raises(OverflowError):
        IntFormat(10**9)
    assert UINT(5, 2**16).bits == 2**16            # wide but within the limit


# ---------------------------------------------------------------- raw codes

@pytest.mark.parametrize("raw", [-1, 2**16, 2**100])
def test_raw_codes_must_fit_the_format(raw):
    with pytest.raises(ValueError, match="does not fit"):
        FP16.from_raw(raw)
    with pytest.raises(ValueError, match="does not fit"):
        FPArray.from_raw([raw], FP16)


def test_raw_codes_from_numpy():
    np = pytest.importorskip("numpy")
    # A signed array as wide as the format holds codes as bit patterns.
    assert FPArray.from_raw(np.array([-1, 0x3C00], np.int16), FP16).raw == [0xFFFF, 0x3C00]
    for bad in (np.array([-1], np.int32), np.array([70000], np.uint32), np.array([-1], np.int8)):
        with pytest.raises(ValueError, match="does not fit"):
            FPArray.from_raw(bad, FP16)
    assert FPArray.from_raw(np.array([0x3C00], np.uint32), FP16).raw == [0x3C00]


def test_bit_operators_still_wrap():
    assert (~FP16(1.0)).raw == 0xFFFF ^ FP16(1.0).raw


# ---------------------------------------------------------------- tools

@pytest.mark.skipif(not shutil.which("bash") or os.name != "posix", reason="needs bash")
@pytest.mark.parametrize("var, value", [("MEM", "x[$(touch {marker})]G"), ("CPUS", "1;touch {marker}"),
                                        ("TIME", "$(touch {marker})")])
def test_capped_takes_its_settings_as_text(tmp_path, var, value):
    marker = tmp_path / "ran"
    env = dict(os.environ, CAPPED="ulimit", **{var: value.format(marker=marker)})
    tool = Path(__file__).parent.parent / "tools" / "capped"
    r = subprocess.run([str(tool), "true"], env=env, capture_output=True, text=True)
    assert r.returncode == 2 and not marker.exists(), r.stderr
