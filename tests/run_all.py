"""Minimal runner so tests work without pytest: python -m tests.run_all (pytest also works)."""
import importlib
import sys
import traceback
from pathlib import Path

# every tests/test_*.py file, so new test files are picked up without editing this list
MODULES = sorted(f"tests.{p.stem}" for p in Path(__file__).parent.glob("test_*.py"))


def main() -> int:
    failed = passed = 0
    for name in MODULES:
        mod = importlib.import_module(name)
        for attr in sorted(dir(mod)):
            if attr.startswith("test_"):
                try:
                    getattr(mod, attr)()
                    passed += 1
                except Exception:
                    failed += 1
                    print(f"FAIL {name}.{attr}")
                    traceback.print_exc()
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
