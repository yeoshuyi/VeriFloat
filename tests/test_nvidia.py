"""NVFP4 against NVIDIA's own quantizer (Model Optimizer, NVFP4QTensor).

ModelOpt needs torch and many other packages, so it is not a dev dependency.
Point VERIFLOAT_MODELOPT_PYTHON at a Python that has them, e.g.:

    uv venv /tmp/mo && VIRTUAL_ENV=/tmp/mo uv pip install torch nvidia-modelopt \\
        requests huggingface_hub
    VERIFLOAT_MODELOPT_PYTHON=/tmp/mo/bin/python uv run pytest tests/test_nvidia.py

ModelOpt computes its scales in float32 and clamps block scales to
[2**-9, 448], so it is compared with the NVFP4_MODELOPT recipe; NVFP4 (the
exact recipe) is compared where the two recipes agree by construction.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path

import pytest

from verifloat import NVFP4_MODELOPT, BlockTensor

PY = os.environ.get("VERIFLOAT_MODELOPT_PYTHON")
pytestmark = pytest.mark.skipif(not PY, reason="set VERIFLOAT_MODELOPT_PYTHON to run")


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x: float) -> int:
    return struct.unpack("<i", struct.pack("<f", x))[0]


def rand_matrix(rng):
    rows, blocks = rng.randint(1, 4), rng.randint(1, 4)
    m = []
    for _ in range(rows):
        row = []
        for _ in range(blocks):
            e = rng.randint(-30, 30)
            kind = rng.random()
            for _ in range(16):
                if kind < 0.1:
                    row.append(0.0)                         # all-zero block
                elif kind < 0.25:                          # exact E2M1 ties
                    row.append(f32(rng.choice([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
                                   * rng.choice([-1, 1]) * 2.0 ** e))
                else:
                    row.append(f32(rng.uniform(-1, 1) * 2.0 ** e))
        m.append(row)
    if all(v == 0 for r in m for v in r):
        m[0][0] = 1.0                                     # ModelOpt divides by amax
    return m


def test_nvfp4_vs_modelopt(rng, iters):
    mats = [rand_matrix(rng) for _ in range(max(iters // 20, 50))]
    script = Path(__file__).parent / "nvidia_modelopt_ref.py"
    res = subprocess.run([PY, str(script)], input=json.dumps(
        [[[bits(v) for v in r] for r in m] for m in mats]), capture_output=True,
        text=True, check=True)
    for m, ref in zip(mats, json.loads(res.stdout)):
        t = BlockTensor.quantize(m, NVFP4_MODELOPT)
        assert t.tensor_scale.raw == ref["tensor_scale"], m
        assert t.scale_raw == ref["scale"], m
        assert t.elem_raw == ref["elem"], m
