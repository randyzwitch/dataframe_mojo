"""Check local Python imports without hiding unused test helpers."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TestReachability(unittest.TestCase):
    def check_tree(self, files):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scripts").mkdir()
            shutil.copyfile(
                ROOT / "scripts/check_test_reachability.py",
                root / "scripts/check_test_reachability.py",
            )
            # The checker also validates its configured runner entry points.
            for name in (
                "tests/run_tests.sh",
                "tests/runner.mojo",
                "tests/tooling/contract_runner.mojo",
                "tests/tooling/package_smoke.sh",
                "tests/tooling/suite_policy.mojo",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            for name, source in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source)
            for args in (("init", "-q"), ("add", ".")):
                subprocess.run(
                    ["git", *args], cwd=root, check=True, capture_output=True
                )
            result = subprocess.run(
                [sys.executable, "scripts/check_test_reachability.py"],
                cwd=root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stderr, "")
            self.assertNotIn("that do not exist", result.stdout)
            return result

    def test_sibling_imports_are_reachable(self):
        for statement in (
            "from helper import value",
            "from helper import (\n    value as fixture,\n)",
            "import helper as fixture",
            "import json, helper",
        ):
            with self.subTest(statement=statement):
                result = self.check_tree(
                    {
                        "tests/tooling/helper.py": "value = 1\n",
                        "tests/tooling/test_consumer.py": statement + "\n",
                    }
                )
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )

    def test_unused_helper_is_rejected(self):
        result = self.check_tree({"tests/tooling/helper.py": "value = 1\n"})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("tests/tooling/helper.py", result.stdout)

    def test_import_text_is_not_an_import(self):
        result = self.check_tree(
            {
                "tests/tooling/helper.py": "value = 1\n",
                "tests/tooling/test_consumer.py": (
                    '# from helper import value\ntext = "import helper"\n'
                ),
            }
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("tests/tooling/helper.py", result.stdout)

    def test_self_import_does_not_make_helper_reachable(self):
        result = self.check_tree(
            {"tests/tooling/helper.py": "import helper\n"}
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("tests/tooling/helper.py", result.stdout)

    def test_import_does_not_reach_other_directory(self):
        result = self.check_tree(
            {
                "tests/tooling/helper.py": "value = 1\n",
                "tests/tooling/test_consumer.py": "from helper import value\n",
                "tests/support/helper.py": "value = 2\n",
            }
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("tests/support/helper.py", result.stdout)
        self.assertNotIn("  tests/tooling/helper.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
