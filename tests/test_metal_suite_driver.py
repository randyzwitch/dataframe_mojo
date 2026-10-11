"""Forced-Metal suite derivation and complete failure accounting; no compiler."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "metal_suite_driver", ROOT / "scripts/bench_metal_suites.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class MetalSuiteDriver(unittest.TestCase):
    def test_derivation_preserves_original_queries_and_forces_every_collect(self):
        original = {
            name: (ROOT / "benchmarks/suites" / (name + ".mojo")).read_text()
            for name in ("h2o", "pdsh", "tpcds", "clickbench")
        }
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            manifest = driver.stage_sources(destination)
            for name, text in original.items():
                cpu = (destination / f"cpu_{name}.mojo").read_text()
                self.assertEqual(cpu, text.replace(".collect()", '.collect(engine="cpu")'))
                metal = (destination / f"metal_{name}.mojo").read_text()
                self.assertNotIn(".collect()", metal)
                self.assertNotIn('engine="auto"', metal)
                expected = 15 if name == "h2o" else text.count(".collect()")
                self.assertEqual(metal.count(".collect(accelerator=MetalRuntime())"), expected)
                self.assertEqual((ROOT / "benchmarks/suites" / (name + ".mojo")).read_text(), text)
            self.assertEqual(len(manifest["derived_source_sha256"]), 14)

    def test_unreviewed_materialization_changes_fail_closed(self):
        with self.assertRaises(ValueError):
            driver.derive('x.collect(streaming=False)', "pdsh", "metal")
        text = (ROOT / "benchmarks/suites/h2o.mojo").read_text()
        with self.assertRaises(ValueError):
            driver.derive(text.replace('ref x = t["x"]', 'ref x = missing'), "h2o", "metal")
        with self.assertRaises(ValueError):
            driver.derive(text.replace('if query == "q5":', 'if query == "q5":\n        return x'), "h2o", "metal")

    def test_no_partial_protocol_can_claim_success(self):
        for output in ("", "coverage\tq1\tok\t\n", "coverage\tq2\tok\t\n", "coverage\tq1\tother\t\n"):
            with self.assertRaises(ValueError):
                driver.parse(output, "q1")
        result = driver.parse("coverage\tq1\tok\t\nordered_equal\tFalse\ncollections\t2\noutput_rows\t3\ncollection_ns\t17\n", "q1")
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["ordered_equal"])
        self.assertEqual(result["collections"], 2)

    def test_every_failure_category_is_retained(self):
        for status in ("unsupported", "execution_failed", "wrong_answer", "reference_unsupported", "reference_failed"):
            result = driver.parse(f"coverage\tq6\t{status}\tprecise reason\n", "q6")
            self.assertEqual(result, dict(status=status, reason="precise reason"))


if __name__ == "__main__":
    unittest.main()
