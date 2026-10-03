"""Download and build Berkeley SoftFloat 3 + TestFloat 3 (testfloat_gen) for
several SoftFloat "specializations" (the NaN / invalid-integer conventions of
RISC-V, x86 SSE and ARM VFP). Builds are cached per commit under
$VERIFLOAT_CACHE (default ~/.cache/verifloat) and never deleted by this code.
Needs network access (first run only), a C compiler and make.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import sys
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


# The build directory of the generic 64-bit GCC target. Its settings fit any
# 64-bit little-endian GCC or Clang host; on a big-endian host the one line
# that says otherwise is removed (see _build).
PLATFORM = "Linux-x86_64-GCC"


def _build(spec: str) -> Path:
    """Build SoftFloat and testfloat_gen for ``spec``; the build's root directory."""
    if not (shutil.which("make") and shutil.which("gcc")):
        raise RuntimeError("gcc and make are needed to build TestFloat")
    root = _cache() / "testfloat" / f"{SOFTFLOAT[1][:12]}-{TESTFLOAT[1][:12]}"
    src = root / "src"
    _fetch(*SOFTFLOAT, src / "berkeley-softfloat-3")
    _fetch(*TESTFLOAT, src / "berkeley-testfloat-3")
    build = root / spec
    gen = build / "berkeley-testfloat-3" / "build" / PLATFORM / "testfloat_gen"
    if gen.exists():
        return build
    for name in ("berkeley-softfloat-3", "berkeley-testfloat-3"):
        if not (build / name).exists():
            shutil.copytree(src / name, build / name)
        if sys.byteorder == "big":
            header = build / name / "build" / PLATFORM / "platform.h"
            header.write_text(header.read_text().replace("#define LITTLEENDIAN 1", ""))
    run = lambda d, *args: subprocess.run(
        ["make", "-s", "-j8", f"SPECIALIZE_TYPE={spec}", *args],
        cwd=build / d / "build" / PLATFORM, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    run("berkeley-softfloat-3")
    run("berkeley-testfloat-3", "testfloat_gen")
    return build


def testfloat_gen(spec: str) -> Path:
    """Path to a testfloat_gen built against SoftFloat with ``spec``."""
    return _build(spec) / "berkeley-testfloat-3" / "build" / PLATFORM / "testfloat_gen"


def softfloat_ref(spec: str) -> Path:
    """Path to tests/native/softfloat_ref.c built against SoftFloat with
    ``spec``: SoftFloat's answer for operands of the caller's choosing."""
    build = _build(spec)
    source = Path(__file__).parent / "native" / "softfloat_ref.c"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    exe = build / f"softfloat_ref-{digest}"
    if not exe.exists():
        soft = build / "berkeley-softfloat-3"
        subprocess.run(["gcc", "-O2", "-DSOFTFLOAT_FAST_INT64", "-o", str(exe), str(source),
                        f"-I{soft / 'source' / 'include'}",
                        str(soft / "build" / PLATFORM / "softfloat.a")], check=True, stderr=subprocess.PIPE)
    return exe
