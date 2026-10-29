"""Forwarding stub for backward compatibility with existing README instructions."""

from pathlib import Path
import sys
import importlib.util

ROOT = Path(__file__).resolve().parent.parent.parent
target_script = ROOT / "src" / "06_compare_models.py"

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("compare_models", target_script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_models"] = module
    spec.loader.exec_module(module)
    module.run_comparison()
