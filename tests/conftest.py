"""Seeded RNG for constrained random tests.

Set VERIFLOAT_SEED to reproduce a run and VERIFLOAT_ITERS to change
how many random cases each test draws. VERIFLOAT_IMPL=py runs the suite
against the frozen pure-Python version (archive/python) instead of the
C++-backed package. VERIFLOAT_MEM_GB caps the memory of a test run (default
8 GiB, 0 for none).
"""

from __future__ import annotations

import collections
import importlib
import os
import random
import sys

import pytest

# A test that goes wrong on a wide format can ask for an integer of 2**(2**40)
# bits. Cap the address space of the test process (and of what it spawns), so
# such a test fails with MemoryError instead of taking the machine down.
# VERIFLOAT_MEM_GB sets the cap in GiB; 0 removes it.
MEM_GB = float(os.environ.get("VERIFLOAT_MEM_GB", 8))
try:
    import resource
except ImportError:                     # not a POSIX system: no cap
    MEM_GB = 0.0
if MEM_GB > 0:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    _cap = int(MEM_GB * (1 << 30))
    if _hard != resource.RLIM_INFINITY:
        _cap = min(_cap, _hard)
    try:
        if _soft == resource.RLIM_INFINITY or _soft > _cap:
            resource.setrlimit(resource.RLIMIT_AS, (_cap, _hard))
        MEM_GB = resource.getrlimit(resource.RLIMIT_AS)[0] / (1 << 30)
    except (ValueError, OSError):       # a system that does not take this limit (macOS)
        MEM_GB = 0.0

IMPL = os.environ.get("VERIFLOAT_IMPL", "cpp")
if IMPL == "py":
    # Alias the package and its submodules before any test imports them.
    import verifloat_py
    sys.modules["verifloat"] = verifloat_py
    for _sub in ("accum", "blockscale", "bus", "fixed", "fp", "sint", "uint", "_warnings"):
        sys.modules[f"verifloat.{_sub}"] = importlib.import_module(f"verifloat_py.{_sub}")
elif IMPL != "cpp":
    raise RuntimeError(f"VERIFLOAT_IMPL must be 'cpp' or 'py', not {IMPL!r}")

# VERIFLOAT_FAST=0 runs the C++ core on its general path only (no fast
# kernels), VERIFLOAT_SIMD=none|avx2|avx512 picks the SIMD level: every
# setting must pass the whole suite with identical results.
FAST = os.environ.get("VERIFLOAT_FAST", "1") != "0"
SIMD = os.environ.get("VERIFLOAT_SIMD")
if IMPL == "cpp":
    from verifloat import _core
    _core.set_fast(FAST)
    if SIMD is not None:
        _core.set_simd(SIMD)
    SIMD = _core.simd()

# How many results a run compared with each outside reference: tests add to
# it, the summary at the end of the run reports it.
COMPARED: collections.Counter = collections.Counter()

SEED = int(os.environ.get("VERIFLOAT_SEED", random.randrange(1 << 32)))
ITERS = int(os.environ.get("VERIFLOAT_ITERS", 2000))


def pytest_report_header(config):
    paths = f" VERIFLOAT_FAST={int(FAST)} VERIFLOAT_SIMD={SIMD}" if IMPL == "cpp" else ""
    mem = f" memory cap {MEM_GB:g} GiB" if MEM_GB > 0 else " no memory cap"
    return f"verifloat: VERIFLOAT_IMPL={IMPL}{paths} VERIFLOAT_SEED={SEED} VERIFLOAT_ITERS={ITERS}{mem}"


def pytest_terminal_summary(terminalreporter):
    if COMPARED:
        terminalreporter.section("results compared with outside references")
        for name, n in sorted(COMPARED.items()):
            terminalreporter.write_line(f"{n:>13,}  {name}")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.failed:
        report.sections.append(
            ("reproduce", f"VERIFLOAT_SEED={SEED} pytest '{item.nodeid}'"))
    return report


@pytest.fixture
def rng(request) -> random.Random:
    # Per-test stream so a test reproduces regardless of which others run.
    return random.Random(f"{SEED}:{request.node.nodeid}")


@pytest.fixture
def iters() -> int:
    return ITERS
