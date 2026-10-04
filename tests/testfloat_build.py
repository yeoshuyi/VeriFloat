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

# Release 3e (2018) of both, plus a later SoftFloat fix for RISC-V f128 NaNs:
# (repository, commit, SHA-256 of the extracted file tree; see tree_sha256).
SOFTFLOAT = ("ucb-bar/berkeley-softfloat-3", "b64af41c3276f97f0e181920400ee056b9c88037",
             "565192c517a79f7a749fbaa8de752554125bdf8da4ee0cea108e4b7f786cf7d4")
TESTFLOAT = ("ucb-bar/berkeley-testfloat-3", "06b20075dd3c1a5d0dd007a93643282832221612",
             "73f07482a3da4fe6a7c6a259947cc509d3b5057f9d6cd786b04cafae5560dcd7")
SPECIALIZATIONS = ("RISCV", "8086-SSE", "ARM-VFPv2")


def cache_dir() -> Path:
    """$VERIFLOAT_CACHE (default ~/.cache/verifloat): programs and libraries
    the tests build are kept here and later run or loaded, so on POSIX it
    must belong to this user and be writable by no one else. Write access for
    others is removed from a directory of this user's; one that belongs to
    someone else is refused."""
    path = Path(os.environ.get("VERIFLOAT_CACHE", Path.home() / ".cache" / "verifloat"))
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        st = path.stat()
        if st.st_uid != os.getuid():
            raise RuntimeError(f"{path} belongs to another user; it holds programs the tests run")
        if st.st_mode & 0o022:
            path.chmod(st.st_mode & ~0o022 & 0o7777)
    return path



def shared_library(source: Path, name: str) -> Path:
    """tests/native/<source> built with gcc as a shared library in the
    cache, named by its source's hash. Built under a temporary name and
    renamed when complete, so an interrupted build is never loaded."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    lib = cache_dir() / f"{name}-{digest}.so"
    if not lib.exists():
        tmp = lib.with_name(f"{lib.name}.{os.getpid()}.part")
        try:
            subprocess.run(["gcc", "-O1", "-shared", "-fPIC", "-o", str(tmp), str(source)], check=True)
            os.replace(tmp, lib)
        finally:
            tmp.unlink(missing_ok=True)
    return lib


def tree_sha256(root: Path) -> str:
    """SHA-256 over a directory tree: each path with its file's SHA-256 (or
    its link target), in sorted order. Unlike a hash of the downloaded
    archive, it does not change if the server compresses differently."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*"), key=lambda q: q.relative_to(root).as_posix()):
        rel = p.relative_to(root).as_posix().encode()
        if p.is_symlink():
            h.update(b"L" + rel + b"\0" + os.readlink(p).encode() + b"\0")
        elif p.is_file():
            h.update(b"F" + rel + b"\0" + hashlib.sha256(p.read_bytes()).digest())
        elif p.is_dir():
            h.update(b"D" + rel + b"\0")
    return h.hexdigest()


def _fetch(repo: str, sha: str, tree: str, dest: Path) -> None:
    if dest.exists():
        return
    url = f"https://github.com/{repo}/archive/{sha}.tar.gz"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    tmp = dest.with_name(dest.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(tmp, filter="data")
    (inner,) = tmp.iterdir()
    if tree_sha256(inner) != tree:
        shutil.rmtree(tmp)
        raise RuntimeError(f"{url}: content does not match the pinned SHA-256")
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
    root = cache_dir() / "testfloat" / f"{SOFTFLOAT[1][:12]}-{TESTFLOAT[1][:12]}"
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
