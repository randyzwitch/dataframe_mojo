"""CI orchestration must preserve failures and avoid implicit stale builds."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TestRunner(unittest.TestCase):
    def run_suite(self, *, compile_exit=0, run_exit=0, reuse=False, missing=False):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            calls = directory / "calls"
            mojo = directory / "mojo"
            mojo.write_text(
                f"#!{sys.executable}\n"
                "import os, pathlib, sys\n"
                "with open(os.environ['MOJO_CALLS'], 'a') as f: f.write('build\\n')\n"
                "if int(os.environ['COMPILE_EXIT']): sys.exit(int(os.environ['COMPILE_EXIT']))\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
                "out.write_text('#!/bin/sh\\nexit ' + os.environ['RUN_EXIT'] + '\\n')\n"
                "out.chmod(0o755)\n"
            )
            mojo.chmod(0o755)
            binary = directory / "parquet"
            if not missing:
                binary.write_text(f"#!/bin/sh\nexit {run_exit}\n")
                binary.chmod(0o755)
            env = dict(
                os.environ, PATH=f"{tmp}:{os.environ['PATH']}",
                MOJO_CALLS=str(calls), COMPILE_EXIT=str(compile_exit),
                RUN_EXIT=str(run_exit), TEST_JOBS="1",
                CI_TIMINGS_DIR=str(directory / "timings"),
                TEST_PARQUET_BINARY=str(binary) if reuse else "",
            )
            result = subprocess.run(
                ["bash", "scripts/run_tests.sh", "tests/test_parquet.mojo"],
                cwd=ROOT, env=env, capture_output=True, text=True,
            )
            records = [json.loads(p.read_text()) for p in (directory / "timings").glob("*.json")]
            return result, calls.read_text() if calls.exists() else "", records

    def test_build_and_execute(self):
        result, calls, records = self.run_suite()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, "build\n")
        self.assertEqual(len(records), 2)

    def test_compile_and_runtime_failures_propagate(self):
        for options in (dict(compile_exit=3), dict(run_exit=4)):
            result, _, _ = self.run_suite(**options)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FAILED: tests/test_parquet.mojo", result.stdout)

    def test_reuse_is_explicit_and_never_rebuilds(self):
        for options in ({}, dict(run_exit=4), dict(missing=True)):
            result, calls, _ = self.run_suite(reuse=True, **options)
            self.assertEqual(calls, "")
            self.assertEqual(result.returncode == 0, not options)


if __name__ == "__main__":
    unittest.main()
