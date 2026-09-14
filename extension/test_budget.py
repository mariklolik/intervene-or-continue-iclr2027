import importlib.util
import tempfile
import subprocess
import sys
import time
import unittest
from pathlib import Path


class BudgetTest(unittest.TestCase):
    def test_child_deadline_stops_work_without_host_guard(self):
        self.assertTrue(Path(__file__).with_name("container_deadline.py").exists())
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "over_budget"
            started = time.time()
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("container_deadline.py")), str(started + 0.3), sys.executable, "-c", "import time,pathlib; time.sleep(10); pathlib.Path(" + repr(str(marker)) + ").touch()"], timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertLess(time.time() - started, 3)
            self.assertFalse(marker.exists())

    def test_overlapping_reservations_cannot_exceed_cap(self):
        self.assertIsNotNone(importlib.util.find_spec("server_guard"))
        from server_guard import reserve

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.jsonl"
            reserve(path, "first", 60, 2, 180)
            with self.assertRaises(ValueError):
                reserve(path, "second", 31, 2, 180)
            reserve(path, "third", 30, 2, 180)
            with self.assertRaises(ValueError):
                reserve(path, "fourth", 1, 1, 180)


if __name__ == "__main__":
    unittest.main()
