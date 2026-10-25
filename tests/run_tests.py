"""
Standalone Test Runner for Expulse Test Suite.
Discovers and executes all test functions in tests/test_*.py.
Can be executed with: `python tests/run_tests.py` or via standard `pytest tests/`.
"""

import importlib
import inspect
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_all_tests():
    tests_dir = PROJECT_ROOT / "tests"
    test_files = sorted(tests_dir.glob("test_*.py"))

    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    failures = []

    print("=" * 70)
    print(f"EXPULSE TEST RUNNER - Discovering tests in {tests_dir}...")
    print("=" * 70)

    start_time = time.perf_counter()

    for test_file in test_files:
        module_name = f"tests.{test_file.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            print(f"[FAIL] Could not import module {module_name}: {e}")
            failed_tests += 1
            failures.append((module_name, "Module Import", str(e)))
            continue

        test_funcs = [
            (name, func)
            for name, func in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_")
        ]

        for name, func in test_funcs:
            total_tests += 1
            try:
                func()
                print(f"  [PASS] {module_name}::{name}")
                passed_tests += 1
            except AssertionError as ae:
                print(f"  [FAIL] {module_name}::{name} -> Assertion failed: {ae}")
                failed_tests += 1
                failures.append((module_name, name, str(ae)))
            except Exception as ex:
                print(f"  [ERROR] {module_name}::{name} -> Exception: {ex}")
                failed_tests += 1
                failures.append((module_name, name, str(ex)))

    duration = time.perf_counter() - start_time
    print("=" * 70)
    print(f"RESULTS: {passed_tests} passed, {failed_tests} failed out of {total_tests} tests ({duration:.2f}s)")
    print("=" * 70)

    if failures:
        print("\nFailures:")
        for mod, test, err in failures:
            print(f" - {mod}::{test}: {err}")
        sys.exit(1)
    else:
        print("\nAll tests PASSED successfully!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
