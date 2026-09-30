"""Run NVIDIA ModelOpt's NVFP4 quantizer on matrices given as float32 bit
patterns (JSON on stdin) and print its codes as JSON. This script runs in a
separate environment that has torch and nvidia-modelopt installed (see
test_nvidia.py); it does not import verifloat."""

import json
import sys

import torch
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

out = []
for bits in json.load(sys.stdin):
    x = torch.tensor(bits, dtype=torch.int64).to(torch.int32).view(torch.float32)
    q, scale, scale2 = NVFP4QTensor.quantize(x, 16)
    packed = q._quantized_data.to(torch.int64)
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(x.shape[0], -1)
    out.append({
        "elem": codes[:, :x.shape[1]].tolist(),
        "scale": scale.view(torch.uint8).to(torch.int64).tolist(),
        "tensor_scale": int(scale2.view(torch.int32)) & 0xFFFFFFFF,
    })
json.dump(out, sys.stdout)
