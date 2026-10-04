"""Build and run the SystemVerilog testbench on Verilator:

    uv run python examples/dpi_adder/run.py [+n=20000] [+seed=7] [+rounding=rtz]

The adder is fetched from dawsonjon/fpu at a pinned commit; the golden model
is libverifloat, linked with the flags `python -m verifloat.dpi` prints.
"""

import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

from verifloat import dpi

COMMIT = "c45d5e0a9f1b7e945e2be6770dbc1203825a05c4"
URL = f"https://raw.githubusercontent.com/dawsonjon/fpu/{COMMIT}/adder/adder.v"
SHA256 = "0cd444cd873fda6068e2b91a92574679d61303aa75c025971a6d2ecb72c7980f"


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


here = Path(__file__).resolve().parent
build = here / "sim_build"
build.mkdir(exist_ok=True)
src = build / "adder.v"
fetch(URL, src, SHA256)

subprocess.run(["verilator", "--binary", "--timing", "-Wno-fatal", "-Wno-WIDTH", "--top-module", "tb",
                "--Mdir", str(build / "obj_dir"), str(dpi.sv_package()), str(here / "tb.sv"), str(src),
                "-CFLAGS", dpi.cflags(), "-LDFLAGS", dpi.ldflags()], check=True, stdout=subprocess.DEVNULL)
sys.exit(subprocess.run([str(build / "obj_dir" / "Vtb"), *sys.argv[1:]]).returncode)
