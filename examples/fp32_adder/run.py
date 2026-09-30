"""Fetch the dawsonjon/fpu adder at a pinned commit and run the cocotb
testbench on Verilator:  uv run --group examples python examples/fp32_adder/run.py
"""

import urllib.request
from pathlib import Path

from cocotb_tools.runner import get_runner

COMMIT = "c45d5e0a9f1b7e945e2be6770dbc1203825a05c4"
URL = f"https://raw.githubusercontent.com/dawsonjon/fpu/{COMMIT}/adder/adder.v"

here = Path(__file__).resolve().parent
build = here / "sim_build"
build.mkdir(exist_ok=True)
src = build / "adder.v"
if not src.exists():
    urllib.request.urlretrieve(URL, src)

runner = get_runner("verilator")
runner.build(sources=[src], hdl_toplevel="adder", build_dir=build,
             build_args=["-Wno-fatal", "-Wno-WIDTH"], always=True)
runner.test(hdl_toplevel="adder", test_module="test_adder", test_dir=here,
            build_dir=build)
