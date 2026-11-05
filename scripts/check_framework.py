"""Run meaningful local tests without requiring pytest or downloading models."""
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
suite = unittest.defaultTestLoader.discover(str(root / "tests"))
raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
