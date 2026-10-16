"""Forced-Metal suite derivation and complete failure accounting; no compiler."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "metal_suite_driver", ROOT / "benchmarks/tools/bench_metal_suites.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class MetalSuiteDriver(unittest.TestCase):
    def test_derivation_preserves_original_queries_and_forces_every_collect(
        self,
    ):
        original = {
            name: (ROOT / "benchmarks/suites" / (name + ".mojo")).read_text()
            for name in ("h2o", "pdsh", "tpcds", "clickbench")
        }
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            manifest = driver.stage_sources(destination)
            for name, text in original.items():
                cpu = (destination / f"cpu_{name}.mojo").read_text()
                self.assertEqual(
                    cpu, text.replace(".collect()", '.collect(engine="cpu")')
                )
                metal = (destination / f"metal_{name}.mojo").read_text()
                self.assertNotIn(".collect()", metal)
                self.assertNotIn('engine="auto"', metal)
                expected = 15 if name == "h2o" else text.count(".collect()")
                self.assertEqual(
                    metal.count(".collect(accelerator=MetalRuntime())"),
                    expected,
                )
                self.assertEqual(
                    (ROOT / "benchmarks/suites" / (name + ".mojo")).read_text(),
                    text,
                )
            self.assertEqual(len(manifest["derived_source_sha256"]), 16)

    def test_unreviewed_materialization_changes_fail_closed(self):
        with self.assertRaises(ValueError):
            driver.derive("x.collect(streaming=False)", "pdsh", "metal")
        text = (ROOT / "benchmarks/suites/h2o.mojo").read_text()
        with self.assertRaises(ValueError):
            driver.derive(
                text.replace('ref x = t["x"]', "ref x = missing"),
                "h2o",
                "metal",
            )
        with self.assertRaises(ValueError):
            driver.derive(
                text.replace(
                    'if query == "q5":', 'if query == "q5":\n        return x'
                ),
                "h2o",
                "metal",
            )

    def test_no_partial_protocol_can_claim_success(self):
        for output in (
            "",
            "coverage\tq1\tok\t\n",
            "coverage\tq2\tok\t\n",
            "coverage\tq1\tother\t\n",
        ):
            with self.assertRaises(ValueError):
                driver.parse(output, "q1")
        result = driver.parse(
            "coverage\tq1\tok\t\nordered_equal\tFalse\ncollections\t2\noutput_rows\t3\ncollection_ns\t17\n",
            "q1",
        )
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["ordered_equal"])
        self.assertEqual(result["collections"], 2)

    def test_every_failure_category_is_retained(self):
        for status in (
            "unsupported",
            "execution_failed",
            "wrong_answer",
            "reference_unsupported",
            "reference_failed",
        ):
            result = driver.parse(
                f"coverage\tq6\t{status}\tprecise reason\n", "q6"
            )
            self.assertEqual(
                result, dict(status=status, reason="precise reason")
            )

    def test_full_answer_diagnostics_preserve_primary_failure(self):
        output = (
            "coverage\tq1\twrong_answer\texact multiset differs\n"
            "expected_rows\t2\nactual_rows\t2\n"
            "cpu_repeat_equal\tFalse\n"
            "diagnostic\tcpu_metal.schema_equal\tTrue\n"
            "diagnostic\tcpu_metal.alignment\tverified_unique_keys\n"
            "column_bag\tcpu_metal\tkey\tTrue\n"
            "column_diagnostic\tcpu_metal\tmean\tFloat64\t1\t0\t1\t0\t18446744073709551615\t1e-15\t2e-15\n"
            "column_diagnostic\tcpu_repeat\tcount\tInt64\t1\t0\t0\t0\tunavailable\tunavailable\tunavailable\n"
            "diagnostic_error\tcpu_repeat\tsecondary comparison failed\n"
        )
        result = driver.parse(output, "q1")
        self.assertEqual(result["status"], "wrong_answer")
        facts = result["diagnostics"]
        self.assertFalse(facts["cpu_repeat_ordered_equal"])
        self.assertTrue(facts["cpu_metal.schema_equal"])
        self.assertTrue(facts["cpu_metal.column_bags"]["key"])
        self.assertEqual(
            facts["cpu_metal.columns"]["mean"]["max_ulp"], (1 << 64) - 1
        )
        self.assertEqual(
            facts["cpu_metal.columns"]["mean"]["max_absolute"], "1e-15"
        )
        self.assertIsNone(facts["cpu_repeat.columns"]["count"]["max_ulp"])
        self.assertEqual(len(facts["errors"]), 1)

    def test_malformed_diagnostics_fail_closed(self):
        prefix = "coverage\tq1\twrong_answer\tdiff\n"
        numeric = "column_diagnostic\tcpu_metal\tv\tFloat64\t1\t0\t1\t0\t1\t1e-15\t1e-15\n"
        invalid = [
            "diagnostic\tcpu_metal.schema_equal\tmaybe\n",
            "diagnostic\tunknown.schema_equal\tTrue\n",
            "column_bag\tcpu_metal\tv\tTrue\n" * 2,
            numeric * 2,
            numeric.replace("\t1\t0\t1\t0\t", "\t-1\t0\t1\t0\t"),
            numeric.replace("\t1e-15\t1e-15", "\tnan\t1e-15"),
            numeric.replace("\t1e-15\t1e-15", "\t-1\t1e-15"),
            numeric.replace("\t1\t1e-15", "\t18446744073709551616\t1e-15"),
            numeric.replace("\t1\t0\t1\t0\t", "\t1\t0\t0\t0\t"),
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                driver.parse(prefix + payload, "q1")

    def test_source_pinned_variant_audits_and_invalid_keys(self):
        path = "benchmarks/suites/h2o.mojo"
        entry = dict(
            workload_kind="canonical",
            execution_boundary="gpu_collects",
            source_path=path,
            source_sha256="hash",
            audit_reason="group keys",
            alignment_keys=["id1"],
        )
        metadata = {
            "cases": {
                "h2o_groupby/q1": entry,
                "h2o_groupby/base/q1": dict(entry, alignment_keys=["id2"]),
            }
        }
        selected = driver.case_audit(
            metadata, "h2o_groupby", "base", "q1", {path: "hash"}
        )
        self.assertEqual(selected["alignment_keys"], ["id2"])
        stale = driver.case_audit(
            metadata, "h2o_groupby", "base", "q1", {path: "changed"}
        )
        self.assertEqual(stale["execution_boundary"], "unknown")
        self.assertEqual(stale["alignment_keys"], [])
        for keys in (["id1", "id1"], ["bad\tkey"], "id1"):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                driver.case_audit(
                    {
                        "cases": {
                            "h2o_groupby/q1": dict(entry, alignment_keys=keys)
                        }
                    },
                    "h2o_groupby",
                    "base",
                    "q1",
                    {path: "hash"},
                )

    def test_diagnostics_do_not_relax_the_exact_comparator(self):
        source = (ROOT / "benchmarks/metal_suite_common.mojo").read_text()
        self.assertIn(
            "or not actual.sort(actual.columns()).equals(canonical)", source
        )
        self.assertNotIn("tolerance", source)


if __name__ == "__main__":
    unittest.main()
