"""The C library and its SystemVerilog package: where they are installed.

The same core as the Python package, built as ``libverifloat`` with a plain
C API (``verifloat.h``) and DPI-C imports (``verifloat_pkg.sv``), so that a
SystemVerilog or C++ testbench can call the model directly::

    verilator --binary tb.sv $(python -m verifloat.dpi --sv) \\
        -CFLAGS "$(python -m verifloat.dpi --cflags)" -LDFLAGS "$(python -m verifloat.dpi --ldflags)"

``python -m verifloat.dpi`` prints all the paths; one option prints one value.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import _core

__all__ = ["cflags", "include_dir", "ldflags", "library", "main", "sv_package"]

_ROOT = Path(_core.__file__).resolve().parent
_LIB_NAMES = ("libverifloat.so", "libverifloat.dylib", "verifloat.dll", "libverifloat.dll")


def library() -> Path:
    """The shared library (libverifloat.so, .dylib or .dll)."""
    for name in _LIB_NAMES:
        if (_ROOT / name).exists():
            return _ROOT / name
    raise FileNotFoundError(f"no libverifloat in {_ROOT}: this build of verifloat has no C library")


def include_dir() -> Path:
    """The directory of verifloat.h."""
    return _ROOT / "include"


def sv_package() -> Path:
    """verifloat_pkg.sv: the DPI-C imports, to compile with the testbench."""
    return _ROOT / "sv" / "verifloat_pkg.sv"


def cflags() -> str:
    return f"-I{include_dir()}"


def ldflags() -> str:
    """Link flags; the library's directory is also set as a run-time search path."""
    lib = library()
    return f"-L{lib.parent} -lverifloat -Wl,-rpath,{lib.parent}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m verifloat.dpi", description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    for flag, text in (("--lib", "path of the shared library"), ("--include", "directory of verifloat.h"),
                       ("--sv", "path of verifloat_pkg.sv"), ("--cflags", "compiler flags"),
                       ("--ldflags", "linker flags")):
        g.add_argument(flag, action="store_true", help=text)
    a = p.parse_args(argv)
    try:
        values = {"lib": library(), "include": include_dir(), "sv": sv_package(), "cflags": cflags(),
                  "ldflags": ldflags()}
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    chosen = [k for k in values if getattr(a, k)]
    if chosen:
        print(values[chosen[0]])
    else:
        for k, v in values.items():
            print(f"{k:8} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
