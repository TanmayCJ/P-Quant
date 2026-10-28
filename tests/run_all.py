"""Minimal runner so tests work without pytest: python -m tests.run_all (pytest also works)."""
import importlib
import sys
import traceback

MODULES = ["tests.test_metrics", "tests.test_pipeline", "tests.test_phase2", "tests.test_phase3", "tests.test_confirm", "tests.test_artifact", "tests.test_phaseb", "tests.test_export"]


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
