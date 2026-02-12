"""logger manager"""

import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Set


class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


LOG_LEVELS = {
    "DEBUG": 0,
    "INFO": 1,
    "WARNING": 2,
    "WARN": 2,
    "ERROR": 3,
    "CRITICAL": 4,
    "FATAL": 4,
}


class LogFileTracker:
    """Single log file tracker"""

    def __init__(
        self,
        filepath: str,
        min_level: int = 0,
        color_output: bool = True,
        last_n_lines: int = 0,
        search_pattern: str | None = None,
    ):
        self.filepath = filepath
        self.min_level = min_level
        self.color_output = color_output
        self.last_n_lines = last_n_lines
        self.search_pattern = search_pattern
        self.file = None
        self.file_size = 0
        self.inode = None

    def open_file(self):
        """Open file and optionally read last N lines"""
        try:
            self.file = open(self.filepath, "r", encoding="utf-8", errors="ignore")
            if self.search_pattern:
                print(
                    f"{Colors.CYAN}[INFO] Searching for pattern '{self.search_pattern}' in {self.filepath}{Colors.RESET}"
                )
                self.file.seek(0)
                lines = self.file.readlines()
                match_indices = [i for i, line in enumerate(lines) if self.search_pattern in line]
                for idx in match_indices:
                    start = max(0, idx - 5)
                    end = min(len(lines), idx + 6)
                    print(f"{Colors.MAGENTA}[{self.filepath}:{idx + 1}]{Colors.RESET}")
                    for i in range(start, end):
                        prefix = f"{Colors.MAGENTA}>> {Colors.RESET}" if i == idx else "   "
                        print(prefix + self.format_output(lines[i].rstrip("\n")))
                print(
                    f"{Colors.CYAN}[INFO] Finished searching in {self.filepath}, now monitoring for new lines...{Colors.RESET}"
                )
            stat = os.stat(self.filepath)
            self.inode = stat.st_ino
            if self.last_n_lines > 0:
                # Read last N lines
                self.file.seek(0, 2)
                file_size = self.file.tell()
                block_size = 4096
                blocks = []
                lines_found = 0
                pos = file_size
                while pos > 0 and lines_found < self.last_n_lines:
                    read_size = min(block_size, pos)
                    pos -= read_size
                    self.file.seek(pos)
                    block = self.file.read(read_size)
                    blocks.insert(0, block)
                    lines_found = sum(b.count("\n") for b in blocks)
                all_data = "".join(blocks)
                last_lines = all_data.splitlines()[-self.last_n_lines :]
                for line in last_lines:
                    print(self.format_output(line))
                self.file.seek(0, 2)
            else:
                self.file.seek(0, 2)
            self.file_size = self.file.tell()
            return True
        except Exception as e:
            print(f"{Colors.RED}[ERROR] Failed to open {self.filepath}: {e}{Colors.RESET}")
            return False

    def check_rotation(self):
        """Check if file has been rotated"""
        try:
            stat = os.stat(self.filepath)
            # detect rotation: inode changed or file size decreased
            if stat.st_ino != self.inode or stat.st_size < self.file_size:
                print(f"{Colors.CYAN}[INFO] Detected file rotation: {self.filepath}{Colors.RESET}")
                if self.file:
                    self.file.close()
                return True
            return False
        except FileNotFoundError:
            return True
        except Exception as e:
            print(
                f"{Colors.RED}[ERROR] Error checking file rotation for {self.filepath}: {e}{Colors.RESET}"
            )
            return False

    def read_new_lines(self) -> List[str]:
        """Read newly added lines"""
        lines = []
        try:
            if self.check_rotation():
                if self.open_file():
                    return []

            while True:
                line = self.file.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))

            self.file_size = self.file.tell()
        except Exception as e:
            print(f"{Colors.RED}[ERROR] Error reading file {self.filepath}: {e}{Colors.RESET}")

        return lines

    def parse_log_level(self, line: str) -> int:
        """Parse log level"""
        match = re.match(r"^(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\b", line)
        if match:
            level = match.group(1).upper()
            return LOG_LEVELS.get(level, -1)
        return -1

    def should_display(self, line: str) -> bool:
        """Determine if this log line should be displayed"""
        priority = self.parse_log_level(line)

        if priority >= self.min_level:
            return True
        return False

    def format_output(self, line: str) -> str:
        """Format output"""
        filename = Path(self.filepath).name

        if self.color_output:
            return f"{Colors.BLUE}[{filename}]{Colors.RESET} {line}"
        else:
            return f"[{filename}] {line}"

    def close(self):
        """Close file"""
        if self.file:
            self.file.close()


class LogManager:
    """A manager to track multiple log files in real-time."""

    def __init__(
        self,
        log_dir: str,
        keyword: str | None = None,
        min_level: str = "DEBUG",
        scan_interval: float = 0.5,
        last_n_lines: int = 0,
        search_pattern: str | None = None,
        color_output: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.keyword = keyword
        self.min_level_name = min_level.upper()
        self.min_level_priority = LOG_LEVELS.get(self.min_level_name, 0)
        self.scan_interval = scan_interval
        self.color_output = color_output
        self.search_pattern = search_pattern
        self.trackers: Dict[str, LogFileTracker] = {}
        self.running = False
        self.last_n_lines = last_n_lines
        self.lock = threading.Lock()

    def find_log_files(self) -> Set[str]:
        """Find matching log files"""
        log_files = set()
        try:
            for file in self.log_dir.iterdir():
                if file.is_file() and (self.keyword is None or self.keyword in file.name):
                    log_files.add(str(file.resolve()))
        except Exception as e:
            print(f"{Colors.RED}[ERROR] Failed to scan directory {self.log_dir}: {e}{Colors.RESET}")

        return log_files

    def scan_new_files(self):
        """Scan for newly added log files"""
        current_files = self.find_log_files()

        with self.lock:
            for filepath in current_files:
                if filepath not in self.trackers:
                    tracker = LogFileTracker(
                        filepath,
                        self.min_level_priority,
                        self.color_output,
                        self.last_n_lines,
                        self.search_pattern,
                    )
                    if tracker.open_file():
                        self.trackers[filepath] = tracker
                        print(f"{Colors.GREEN}[INFO] Started tracking: {filepath}{Colors.RESET}")

            removed_files = set(self.trackers.keys()) - current_files
            for filepath in removed_files:
                self.trackers[filepath].close()
                del self.trackers[filepath]
                print(f"{Colors.YELLOW}[INFO] Stopped tracking: {filepath}{Colors.RESET}")

    def monitor(self):
        """Main monitoring loop"""
        self.running = True
        last_scan = 0

        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.CYAN}Log monitoring started{Colors.RESET}")
        print(f"{Colors.CYAN}Monitoring directory: {self.log_dir}{Colors.RESET}")
        print(f"{Colors.CYAN}Keyword: {self.keyword}{Colors.RESET}")
        print(f"{Colors.CYAN}Minimum level: {self.min_level_name}{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")

        # Initial scan
        self.scan_new_files()

        try:
            while self.running:
                current_time = time.time()

                # Periodically scan for new files (every 5 seconds)
                if current_time - last_scan > 5:
                    self.scan_new_files()
                    last_scan = current_time

                # Read new content from all files
                with self.lock:
                    for tracker in list(self.trackers.values()):
                        lines = tracker.read_new_lines()
                        for line in lines:
                            if tracker.should_display(line):
                                output = tracker.format_output(line)
                                print(output)

                time.sleep(self.scan_interval)

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}[INFO] Received stop signal, exiting...{Colors.RESET}")
        finally:
            self.stop()

    def stop(self):
        """Stop monitoring"""
        self.running = False
        with self.lock:
            for tracker in self.trackers.values():
                tracker.close()
        print(f"{Colors.GREEN}[INFO] Monitoring stopped{Colors.RESET}")
