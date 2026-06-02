"""Unit tests for the background CPU load monitor."""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.serving.load_monitor import LoadMonitor


class TestLoadMonitor(unittest.TestCase):
    def test_starts_and_stops(self):
        monitor = LoadMonitor(interval_s=0.05)
        monitor.start()
        try:
            time.sleep(0.2)
            sample = monitor.sample()
            self.assertIsNotNone(sample)
            self.assertGreaterEqual(sample.cpu_percent, 0.0)
            self.assertLessEqual(sample.cpu_percent, 100.0)
        finally:
            monitor.stop()

    def test_context_manager(self):
        with LoadMonitor(interval_s=0.05) as monitor:
            time.sleep(0.15)
            self.assertGreaterEqual(monitor.smoothed_percent(), 0.0)

    def test_rejects_bad_alpha(self):
        with self.assertRaises(ValueError):
            LoadMonitor(alpha=0.0)
        with self.assertRaises(ValueError):
            LoadMonitor(alpha=1.5)


if __name__ == "__main__":
    unittest.main()
