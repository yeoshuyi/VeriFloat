"""The portable 128-bit integers that MSVC builds use, against the compiler's own.

src/cpp/int128.hpp stands in for unsigned __int128 and __int128 where the
compiler has neither. tests/native/int128_check.cpp compares every operator
with the built-in types, on all pairs of boundary values and on random
operands; it needs a C++20 compiler that has __int128 (GCC or Clang). The
rest of the suite runs against the portable integers when the package is
built with -DVF_PORTABLE_INT128=ON.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
CXX = next((c for c in (os.environ.get("CXX"), "c++", "g++", "clang++") if c and shutil.which(c)), None)


@pytest.mark.skipif(CXX is None, reason="no C++ compiler")
def test_portable_int128_matches_the_builtin_one(tmp_path, iters):
    exe = tmp_path / ("int128_check.exe" if sys.platform == "win32" else "int128_check")
    build = subprocess.run([CXX, "-std=c++20", "-O2", f"-I{ROOT / 'src' / 'cpp'}",
                            str(ROOT / "tests" / "native" / "int128_check.cpp"), "-o", str(exe)],
                           capture_output=True, text=True)
    if build.returncode != 0:
        pytest.skip(f"{CXX} could not build the check (it needs C++20 and __int128): {build.stderr[-300:]}")
    run = subprocess.run([str(exe), str(iters * 500)], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "no difference" in run.stdout
