"""Download and build Berkeley SoftFloat 3 + TestFloat 3 (testfloat_gen) for
several SoftFloat "specializations" (the NaN / invalid-integer conventions of
RISC-V, x86 SSE and ARM VFP). Builds are cached per commit under
$VERIFLOAT_CACHE (default ~/.cache/verifloat) and never deleted by this code.
Needs network access (first run only), a C compiler and make.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

# Release 3e (2018) of both, plus a later SoftFloat fix for RISC-V f128 NaNs.
SOFTFLOAT = ("ucb-bar/berkeley-softfloat-3", "b64af41c3276f97f0e181920400ee056b9c88037")
TESTFLOAT = ("ucb-bar/berkeley-testfloat-3", "06b20075dd3c1a5d0dd007a93643282832221612")
SPECIALIZATIONS = ("RISCV", "8086-SSE", "ARM-VFPv2")


def _cache() -> Path:
    return Path(os.environ.get("VERIFLOAT_CACHE", Path.home() / ".cache" / "verifloat"))


def _fetch(repo: str, sha: str, dest: Path) -> None:
    if dest.exists():
        return
    url = f"https://github.com/{repo}/archive/{sha}.tar.gz"
    data = urllib.request.urlopen(url, timeout=60).read()
    tmp = dest.with_name(dest.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(tmp, filter="data")
    (inner,) = tmp.iterdir()
    inner.rename(dest)
    tmp.rmdir()


def testfloat_gen(spec: str) -> Path:
    """Path to a testfloat_gen built against SoftFloat with ``spec``."""
    if not (shutil.which("make") and (shutil.which("cc") or shutil.which("gcc"))):
        raise RuntimeError("a C compiler and make are needed to build TestFloat")
    root = _cache() / "testfloat" / f"{SOFTFLOAT[1][:12]}-{TESTFLOAT[1][:12]}"
    src = root / "src"
    _fetch(*SOFTFLOAT, src / "berkeley-softfloat-3")
    _fetch(*TESTFLOAT, src / "berkeley-testfloat-3")
    build = root / spec
    gen = build / "berkeley-testfloat-3" / "build" / "Linux-x86_64-GCC" / "testfloat_gen"
    if gen.exists():
        return gen
    for name in ("berkeley-softfloat-3", "berkeley-testfloat-3"):
        if not (build / name).exists():
            shutil.copytree(src / name, build / name)
    run = lambda d, *args: subprocess.run(
        ["make", "-s", "-j8", f"SPECIALIZE_TYPE={spec}", *args],
        cwd=build / d / "build" / "Linux-x86_64-GCC", check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    run("berkeley-softfloat-3")
    run("berkeley-testfloat-3", "testfloat_gen")
    return gen
