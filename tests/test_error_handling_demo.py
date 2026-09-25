"""Smoke test for examples/error_handling_demo.py."""

import asyncio
import importlib.util
import logging
from pathlib import Path
import sys

import pytest

_DEMO_PATH = Path(__file__).resolve().parents[1] / "examples" / "error_handling_demo.py"


def _load_error_handling_demo():
    """Load the demo module from examples/ without requiring a package."""
    spec = importlib.util.spec_from_file_location("error_handling_demo", _DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_error_handling_demo_runs_without_api_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run the scripted error-handling demo end to end; it must not crash."""
    logger = logging.getLogger("dobby")
    handlers = list(logger.handlers)
    level = logger.level
    propagate = logger.propagate
    try:
        demo = _load_error_handling_demo()
        asyncio.run(demo.main())
    finally:
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate

    output = capsys.readouterr().out
    assert "Dobby error-handling demonstration" in output
    assert "No API keys required." in output
    assert "SCENARIO 1" in output
    assert "SCENARIO 6" in output
