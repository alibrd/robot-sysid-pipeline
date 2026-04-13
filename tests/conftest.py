"""Pytest helpers for the standalone sysid_pipeline test suite."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run the slower literature-verification tests.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: slower literature-verification tests; enable with --run-slow",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(
        reason="need --run-slow to run slow literature-verification tests"
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
