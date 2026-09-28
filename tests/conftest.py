"""Seeded RNG for constrained random tests.

Set CUSTOMTYPES_SEED to reproduce a run and CUSTOMTYPES_ITERS to change
how many random cases each test draws.
"""

from __future__ import annotations

import os
import random

import pytest

SEED = int(os.environ.get("CUSTOMTYPES_SEED", random.randrange(1 << 32)))
ITERS = int(os.environ.get("CUSTOMTYPES_ITERS", 2000))


def pytest_report_header(config):
    return f"customtypes: CUSTOMTYPES_SEED={SEED} CUSTOMTYPES_ITERS={ITERS}"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.failed:
        report.sections.append(
            ("reproduce", f"CUSTOMTYPES_SEED={SEED} pytest '{item.nodeid}'"))
    return report


@pytest.fixture
def rng(request) -> random.Random:
    # Per-test stream so a test reproduces regardless of which others run.
    return random.Random(f"{SEED}:{request.node.nodeid}")


@pytest.fixture
def iters() -> int:
    return ITERS
