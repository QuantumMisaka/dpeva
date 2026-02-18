import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
src_str = str(SRC_DIR)
sys.path = [src_str] + [p for p in sys.path if p != src_str]

for name in list(sys.modules.keys()):
    if name == "dpeva" or name.startswith("dpeva."):
        del sys.modules[name]
