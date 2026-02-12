import os
import tempfile
import unittest
from unittest import mock

from trinity.manager.log_manager import LogManager


class TestLogManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self.temp_dir.name, "test.log")
        with open(self.log_path, "w") as f:
            f.write("INFO Start\nWARN Something\nERROR FindMe\nDEBUG End\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    @mock.patch("builtins.print")
    def test_init_and_tracking(self, mock_print):
        manager = LogManager(
            log_dir=self.temp_dir.name,
            min_level="INFO",
            last_n_lines=2,
            color_output=False,
        )
        manager.scan_new_files()
        self.assertEqual(mock_print.call_count, 1 + 2)  # 1 for tracking, 2 for last_n_lines
        self.assertIn(self.log_path, manager.trackers)
        tracker = manager.trackers[self.log_path]
        with open(self.log_path, "a") as f:
            f.write("INFO line 4\nERROR line 5\nDEBUG line 6\n")
        lines = tracker.read_new_lines()
        filtered = [line for line in lines if tracker.should_display(line)]
        self.assertEqual(filtered, ["INFO line 4", "ERROR line 5"])

    @mock.patch("builtins.print")
    def test_file_rotation(self, mock_print):
        manager = LogManager(
            log_dir=self.temp_dir.name,
            min_level="DEBUG",
            color_output=False,
            last_n_lines=1,
        )
        manager.scan_new_files()
        tracker = manager.trackers[self.log_path]
        with open(self.log_path, "w") as f:
            f.write("INFO AfterRotation\n")
        tracker.read_new_lines()
        self.assertEqual(
            mock_print.call_count, 4
        )  # 1 for last_n_lines, 1 for tracking, 1 for file rotation, 1 for last_n_lines
        import re

        def strip_ansi(s):
            return re.sub(r"\x1b\[[0-9;]*m", "", s)

        self.assertIn("Detected file rotation", strip_ansi(mock_print.call_args_list[2][0][0]))
        self.assertIn("INFO AfterRotation", strip_ansi(mock_print.call_args_list[3][0][0]))

    @mock.patch("builtins.print")
    def test_keyword_filter_and_search_pattern(self, mock_print):
        log2 = os.path.join(self.temp_dir.name, "other.log")
        with open(log2, "w") as f:
            f.write("INFO Other\n")
        manager = LogManager(
            log_dir=self.temp_dir.name,
            keyword="test",
            min_level="INFO",
            color_output=False,
            search_pattern="FindMe",
        )
        manager.scan_new_files()
        self.assertIn(self.log_path, manager.trackers)
        self.assertNotIn(log2, manager.trackers)
        self.assertTrue(any("FindMe" in call[0][0] for call in mock_print.call_args_list))
