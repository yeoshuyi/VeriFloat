"""Fetch sashakatne/nvfp4-dotprod-formal-dv's RTL at a pinned commit and run
the cocotb testbench on Verilator:
    uv run --group examples python examples/nvfp4_dotprod/run.py
"""

import urllib.request
from pathlib import Path

from cocotb_tools.runner import get_runner

REPO = "sashakatne/nvfp4-dotprod-formal-dv"
COMMIT = "125db5918dee0c4b69a031d4cd52a6bd96bc1725"
FILES = ["dotprod_pkg.sv", "nvfp4_unpack.svh", "front_end_int8.sv", "mul_lane.sv",
         "align_to_fixed.sv", "exact_acc_tree.sv", "final_round.sv",
         "front_end_bf16.sv", "mul_lane_bf16.sv", "align_bf16.sv",
         "special_case_bf16.sv", "final_round_bf16.sv", "front_end_nvfp4.sv",
         "mul_lane_nvfp4.sv", "align_nvfp4.sv", "scale_mul_nvfp4.sv",
         "final_round_nvfp4.sv", "dotprod_top.sv"]

here = Path(__file__).resolve().parent
build = here / "sim_build"
rtl = build / "rtl"
rtl.mkdir(parents=True, exist_ok=True)
# The package also includes the project's SystemVerilog reference headers.
HEADERS = ["ref/dotprod_ref.svh", "ref/dotprod_ref_bf16.svh", "ref/dotprod_ref_nvfp4.svh"]
for path in [f"rtl/{f}" for f in FILES] + HEADERS:
    dest = rtl / Path(path).name
    if not dest.exists():
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}", dest)

sources = [rtl / f for f in FILES if f.endswith(".sv")] + [here / "dotprod_wrap.sv"]
runner = get_runner("verilator")
runner.build(sources=sources, includes=[rtl], hdl_toplevel="dotprod_wrap",
             build_dir=build, build_args=["-Wno-fatal", "-Wno-WIDTH"], always=True)
runner.test(hdl_toplevel="dotprod_wrap", test_module="test_dotprod", test_dir=here,
            build_dir=build)
