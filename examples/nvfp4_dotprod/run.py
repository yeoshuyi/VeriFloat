"""Fetch sashakatne/nvfp4-dotprod-formal-dv's RTL at a pinned commit and run
the cocotb testbench on Verilator:
    uv run --group examples python examples/nvfp4_dotprod/run.py
"""

import hashlib
import urllib.request
from pathlib import Path

from cocotb_tools.runner import get_runner

REPO = "sashakatne/nvfp4-dotprod-formal-dv"
COMMIT = "125db5918dee0c4b69a031d4cd52a6bd96bc1725"
# The pinned content of every file fetched below.
SHA256 = {
    "rtl/dotprod_pkg.sv": "b9f185659b1876720ae939bc4c59eefd8e885bd9887ea6f9e3eb0b88262aad95",
    "rtl/nvfp4_unpack.svh": "904a752c73f7260e41aeb528d0b967728c4813c52145cfbe751988a6201c402c",
    "rtl/front_end_int8.sv": "5e48f98110d53193b4135a5e5b3ad6b537ca53d2d3fa4db1fb445a9a835f6f26",
    "rtl/mul_lane.sv": "276a6d416fb0dd3c82c9130c006766117f66ca329d0e1fddcbd779c36255d786",
    "rtl/align_to_fixed.sv": "11dc8382296cb5ab0fe525baff16569814ad79464540f87382d07448707e80d8",
    "rtl/exact_acc_tree.sv": "f0415d52fd036e8b6ecf0b29f93cefa1fdead52ba7afab29d602a6a946f26a3e",
    "rtl/final_round.sv": "c6b9dfed8ae06491daf977c536256a2950a71d4dfe29b8e927d5a7a26b8c21f5",
    "rtl/front_end_bf16.sv": "182f345e41b7ab58cf2378ebff8ecac47a5af6c79a12113b2b299998f6d99324",
    "rtl/mul_lane_bf16.sv": "d65177000361a7e962ec06eb05f0cecfe95cc4b13a5e3c1c24722df53b71e336",
    "rtl/align_bf16.sv": "4f8724547841b1559f1f9e0705486975533701d3976f4ddd67adca5279e0064d",
    "rtl/special_case_bf16.sv": "cc3f820ded0faa29859b8b7269151c84a82f289f64c327c36aa1b9cf2fd3860c",
    "rtl/final_round_bf16.sv": "637b879d440464ccbb1c827e997c8ef48ec34e9ae7b649daf6b00b68e3fa1ef2",
    "rtl/front_end_nvfp4.sv": "21f0258047defdeb130bc79bd04115b4b0bb68940172285157e436ef65258ad6",
    "rtl/mul_lane_nvfp4.sv": "fe0201b912cd86ad747336b2e4d49b5f651fbcb827844e04d1d9bd3c5d6f674d",
    "rtl/align_nvfp4.sv": "cd1c779316fa078e1634a512bd1dee07cb4869094771a6e66696d27b3b3e6aae",
    "rtl/scale_mul_nvfp4.sv": "76e8ef05cbce042a5a6e3d080e902fe1a2042387c5d003addb3f07f85f3bda5c",
    "rtl/final_round_nvfp4.sv": "ce8a4dc45d82194fb6a307bbf364c5985cbe3e4d096068740e94ca2a74c83716",
    "rtl/dotprod_top.sv": "c6760810198539bdc6b047c79258f0a19b951322bef0e5a035a59144d9c833f6",
    "ref/dotprod_ref.svh": "301eeda5cb73704e74289a40bb05059588c4634b8fe94f6d62101d88a65e506d",
    "ref/dotprod_ref_bf16.svh": "8dfccd37d5956998a95809058f52da3721569687e49d87d2fd6379ef7ff56971",
    "ref/dotprod_ref_nvfp4.svh": "dc832cc4f4e78faf645045de3f98910802bd3cfbeb0ebec212551801e9887788",
}


def fetch(url: str, dest: Path, sha256: str) -> None:
    """Download url to dest unless dest already holds exactly that content;
    refuse anything whose SHA-256 is not the pinned one."""
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha256:
        return
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError(f"{url}: content does not match the pinned SHA-256")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)


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
    fetch(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}", dest, SHA256[path])

sources = [rtl / f for f in FILES if f.endswith(".sv")] + [here / "dotprod_wrap.sv"]
runner = get_runner("verilator")
runner.build(sources=sources, includes=[rtl], hdl_toplevel="dotprod_wrap",
             build_dir=build, build_args=["-Wno-fatal", "-Wno-WIDTH"], always=True)
runner.test(hdl_toplevel="dotprod_wrap", test_module="test_dotprod", test_dir=here,
            build_dir=build)
