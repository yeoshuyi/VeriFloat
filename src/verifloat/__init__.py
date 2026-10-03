"""VeriFloat: bit-exact custom integer and floating-point types for RTL verification."""

from ._warnings import (BlockFormatWarning, CastWarning, FPFormatWarning,
                        FPModeWarning, FPOverflowWarning, FPUnderflowWarning,
                        IntCastWarning, VeriFloatWarning)
from .accum import Accumulator
from .array import FPArray
from .blockscale import (E8M0, MXFP4, MXINT8, NVFP4, NVFP4_MODELOPT, BlockFormat, BlockTensor,
                         IntFormat, Pow2Format, dot, matmul)
from .fixed import FIXED, FixedFormat
from .fp import (BF16, E2M1, E2M3, E3M2, E4M3, E5M2, E8M23, FP, FP16, FP32, FP64,
                 UE4M3, UE8M0, FPFlags, FPFormat, NaNMode, Rounding, set_sr_source)
from .sint import INT
from .uint import UINT

__all__ = [
    "UINT", "INT", "FIXED", "FixedFormat", "FP", "FPArray", "FPFormat", "FPFlags", "Rounding", "NaNMode",
    "E2M1", "E2M3", "E3M2", "E4M3", "E5M2", "UE4M3", "UE8M0", "set_sr_source",
    "FP16", "BF16", "FP32", "E8M23", "FP64",
    "BlockTensor", "BlockFormat", "IntFormat", "Pow2Format",
    "E8M0", "NVFP4", "NVFP4_MODELOPT", "MXFP4", "MXINT8", "dot", "matmul", "Accumulator",
    "VeriFloatWarning", "CastWarning", "FPFormatWarning", "FPModeWarning",
    "IntCastWarning", "BlockFormatWarning", "FPOverflowWarning",
    "FPUnderflowWarning",
]
__version__ = "0.1.0"
