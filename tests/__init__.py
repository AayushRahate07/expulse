"""
Expulse Test Suite Package.
Provides unittest compatibility for function-based tests.
"""

import inspect
import unittest
from pathlib import Path
import importlib


def load_tests(loader, standard_tests, pattern):
    """Dynamically loads function-based test_* tests into unittest TestSuite."""
    suite = unittest.TestSuite()
    tests_dir = Path(__file__).parent
    for test_file in sorted(tests_dir.glob("test_*.py")):
        mod = importlib.import_module(f"tests.{test_file.stem}")
        for name, func in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith("test_"):
                suite.addTest(unittest.FunctionTestCase(func))
    return suite
