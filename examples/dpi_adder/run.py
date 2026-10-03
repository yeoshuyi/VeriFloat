"""Build and run the SystemVerilog testbench on Verilator:

    uv run python examples/dpi_adder/run.py [+n=20000] [+seed=7] [+rounding=rtz]

The adder is fetched from dawsonjon/fpu at a pinned commit; the golden model
is libverifloat, linked with the flags `python -m verifloat.dpi` prints.
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

from verifloat import dpi

COMMIT = "c45d5e0a9f1b7e945e2be6770dbc1203825a05c4"
URL = f"https://raw.githubusercontent.com/dawsonjon/fpu/{COMMIT}/adder/adder.v"

here = Path(__file__).resolve().parent
build = here / "sim_build"
build.mkdir(exist_ok=True)
src = build / "adder.v"
if not src.exists():
    urllib.request.urlretrieve(URL, src)

subprocess.run(["verilator", "--binary", "--timing", "-Wno-fatal", "-Wno-WIDTH", "--top-module", "tb",
                "--Mdir", str(build / "obj_dir"), str(dpi.sv_package()), str(here / "tb.sv"), str(src),
                "-CFLAGS", dpi.cflags(), "-LDFLAGS", dpi.ldflags()], check=True, stdout=subprocess.DEVNULL)
sys.exit(subprocess.run([str(build / "obj_dir" / "Vtb"), *sys.argv[1:]]).returncode)
