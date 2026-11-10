"""The process limit must reject allocations, not merely report memory use."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_with_memory_limit.py"
LIMIT = 192 * 1024 * 1024
spec = importlib.util.spec_from_file_location("memory_limit", SCRIPT)
limiter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(limiter)


def has_user_manager():
    if sys.platform != "linux":
        return False
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show", "--property=Version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@unittest.skipUnless(
    has_user_manager(), "requires Linux cgroup v2 and a systemd user manager"
)
class MemoryLimitTests(unittest.TestCase):
    def run_query(self, code, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--bytes",
                str(LIMIT),
                "--",
                sys.executable,
                "-c",
                code,
                *extra,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )

    def test_limit_is_installed_before_command_and_arguments_are_preserved(self):
        result = self.run_query(
            "from pathlib import Path; import sys; "
            "p=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip()[4:]; "
            "print((p/'memory.max').read_text().strip()); print(sys.argv[1])",
            "a literal argument; $(no shell) $HOME ${USER}",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                str(LIMIT),
                "a literal argument; $(no shell) $HOME ${USER}",
            ],
        )

    def test_small_allocation_succeeds_and_large_heap_allocation_fails(self):
        small = self.run_query("print(len(bytearray(1024 * 1024)))")
        self.assertEqual(small.returncode, 0, small.stderr)
        self.assertEqual(small.stdout.strip(), "1048576")
        large = self.run_query(
            "print('started',flush=True)\na=bytearray(512 * 1024 * 1024)\n"
            "for i in range(0,len(a),4096): a[i]=1\nprint('finished')"
        )
        self.assertNotEqual(large.returncode, 0)
        self.assertIn("started", large.stdout, large.stderr)
        self.assertNotIn("finished", large.stdout)

    def test_native_mapping_is_limited_when_pages_are_committed(self):
        reserve = self.run_query("import mmap; m=mmap.mmap(-1, 512 * 1024 * 1024)")
        self.assertEqual(reserve.returncode, 0, reserve.stderr)
        touch = self.run_query(
            "import mmap\nprint('started',flush=True)\nm=mmap.mmap(-1,512*1024*1024)\n"
            "for i in range(0,len(m),4096): m[i]=1\nprint('finished')"
        )
        self.assertNotEqual(touch.returncode, 0)
        self.assertIn("started", touch.stdout, touch.stderr)
        self.assertNotIn("finished", touch.stdout)

    def test_child_process_allocations_share_the_limit(self):
        result = self.run_query(
            "import subprocess, sys\nprint('started',flush=True)\n"
            "a=bytearray(120*1024*1024)\n"
            "for i in range(0,len(a),4096): a[i]=1\n"
            "subprocess.run([sys.executable,'-c','a=bytearray(120*1024*1024); print(len(a))'],check=True)\n"
            "print('finished')"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("started", result.stdout, result.stderr)
        self.assertNotIn("finished", result.stdout)

    def test_command_exit_status_is_preserved(self):
        self.assertEqual(self.run_query("raise SystemExit(37)").returncode, 37)


class MemoryLimitArguments(unittest.TestCase):
    def test_invalid_limit_never_runs_command(self):
        for limit in ("0", "-1", "bad", str(2**63)):
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--bytes",
                    limit,
                    "--",
                    sys.executable,
                    "-c",
                    "print('ran')",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")

    def test_unsupported_platform_fails_before_running(self):
        with patch.object(
            limiter.sys, "platform", "darwin"
        ), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                limiter.main(["--bytes", str(LIMIT), "--", "never-run"])
        self.assertEqual(error.exception.code, 2)

    def test_uninstalled_limits_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name, value in [
                ("memory.max", str(LIMIT)),
                ("memory.swap.max", "0"),
                ("memory.oom.group", "1"),
            ]:
                (path / name).write_text(value)
            limiter.verify_limit(path, LIMIT)
            for name, value in [
                ("memory.max", "max"),
                ("memory.max", str(LIMIT + 1)),
                ("memory.swap.max", "1"),
                ("memory.oom.group", "0"),
            ]:
                old = (path / name).read_text()
                (path / name).write_text(value)
                with self.assertRaises(OSError):
                    limiter.verify_limit(path, LIMIT)
                (path / name).write_text(old)


if __name__ == "__main__":
    unittest.main()
