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
                adapted = manifest["scalar_materialization_proofs"][
                    "benchmarks/suites/" + name + ".mojo"
                ]["adapted"]
                expected = 15 if name == "h2o" else text.count(
                    ".collect()"
                ) + len(adapted)
                self.assertEqual(
                    metal.count(".collect(accelerator=MetalRuntime())"),
                    expected,
                )
                self.assertEqual(
                    (ROOT / "benchmarks/suites" / (name + ".mojo")).read_text(),
                    text,
                )
            self.assertEqual(len(manifest["derived_source_sha256"]), 17)
            self.assertIn("metal_coverage_audit.mojo", manifest["derived_source_sha256"])
            self.assertIn(
                "benchmarks/metal_coverage_audit.mojo",
                manifest["original_source_sha256"],
            )
            self.assertIn(
                "from metal_coverage_audit import CoverageAuditRuntime as MetalRuntime",
                (destination / "metal_h2o.mojo").read_text(),
            )
            self.assertEqual(
                sum(
                    len(p["adapted"])
                    for p in manifest["scalar_materialization_proofs"].values()
                ),
                1,
            )
            for path, proof in manifest[
                "scalar_materialization_proofs"
            ].items():
                self.assertEqual(
                    proof["canonical_source_sha256"],
                    manifest["original_source_sha256"][path],
                )
                name = Path(path).stem
                self.assertEqual(
                    proof["metal_derived_source_sha256"],
                    manifest["derived_source_sha256"][
                        "metal_" + name + ".mojo"
                    ],
                )

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
            "coverage\tq1\tok\t\nordered_equal\tFalse\ncollections\t2\n"
            "match\twithin_float_tolerance\n"
            "runtime_audit\tno_marker\nruntime_audit\tcpu\n"
            "output_rows\t3\ncollection_ns\t17\n",
            "q1",
        )
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["ordered_equal"])
        self.assertEqual(result["match"], "within_float_tolerance")
        self.assertEqual(result["collections"], 2)
        self.assertEqual(result["runtime_adjudication"]["observed_calls"], 2)
        self.assertEqual(result["runtime_adjudication"]["events"], ["no_marker", "cpu"])
        self.assertEqual(result["runtime_adjudication"]["cpu_adjudicated_calls"], 1)
        self.assertEqual(
            result["runtime_adjudication"]["no_cpu_adjudication_marker_calls"], 1
        )
        self.assertIn(
            "not a GPU-only certificate",
            result["runtime_adjudication"]["interpretation"],
        )

    def test_runtime_audit_is_required_and_unknown_fails_closed(self):
        success = (
            "coverage\tq1\tok\t\nordered_equal\tTrue\ncollections\t2\n"
            "match\texact\n"
            "runtime_audit\tno_marker\nruntime_audit\tcpu\n"
        )
        for output in (
            success.replace("runtime_audit\tcpu\n", ""),
            success.replace("runtime_audit\tno_marker\n", ""),
            success.replace("runtime_audit\tcpu\n", "runtime_audit\tunknown\n"),
            success.replace("runtime_audit\tcpu\n", "runtime_audit\tbogus\n"),
            success.replace("runtime_audit\tcpu\n", "runtime_audit\tcpu\textra\n"),
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                driver.parse(output, "q1")
        failed = driver.parse(
            "coverage\tq1\tunsupported\tunsupported operator\n"
            "runtime_audit\tfailed\n",
            "q1",
        )
        self.assertEqual(failed["status"], "unsupported")
        self.assertEqual(failed["runtime_adjudication"]["failed_calls"], 1)

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

    def test_derived_worker_pin_cannot_certify_another_adapter(self):
        path = "benchmarks/suites/tpcds.mojo"
        entry = dict(
            workload_kind="canonical",
            execution_boundary="gpu_collects",
            source_path=path,
            source_sha256="canonical",
            metal_derived_source_sha256="adapted",
            whole_query_gpu_certificate=False,
            alignment_keys=["key"],
            order_keys=["key"],
            audit_reason="Reviewed canonical and derived materialization boundaries",
        )
        metadata = {"cases": {"tpcds/q24": entry}}
        valid = driver.case_audit(
            metadata,
            "tpcds",
            "base",
            "q24",
            {path: "canonical"},
            {"metal_tpcds.mojo": "adapted"},
        )
        self.assertEqual(valid["execution_boundary"], "gpu_collects")
        for derived in (None, {}, {"metal_tpcds.mojo": "old"}):
            stale = driver.case_audit(
                metadata, "tpcds", "base", "q24", {path: "canonical"}, derived
            )
            self.assertEqual(stale["execution_boundary"], "unknown")
            self.assertEqual(stale["alignment_keys"], [])
            self.assertEqual(stale["order_keys"], [])
            self.assertFalse(stale["whole_query_gpu_certificate"])
        legacy = dict(entry, execution_boundary="hybrid")
        legacy.pop("metal_derived_source_sha256")
        unchanged = driver.case_audit(
            {"cases": {"tpcds/q24": legacy}},
            "tpcds",
            "base",
            "q24",
            {path: "canonical"},
            {"metal_tpcds.mojo": "adapted"},
        )
        self.assertEqual(unchanged["execution_boundary"], "hybrid")

    def test_only_floating_cells_are_compared_with_tolerance(self):
        source = (ROOT / "benchmarks/metal_suite_common.mojo").read_text()
        self.assertIn("float_tolerance_match", source)
        self.assertIn("actual.sort(actual.columns()).equals(canonical)", source)
        self.assertIn('print("match\\t"', source)
        diagnostics = (
            ROOT / "benchmarks/metal_suite_diagnostics.mojo"
        ).read_text()
        self.assertIn("FLOAT_TOLERANCE = 1e-6", diagnostics)
        # The rule is bench_suites._close; keep the two in step.
        self.assertIn(
            "abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))",
            (ROOT / "benchmarks/tools/bench_suites.py").read_text(),
        )

    def test_a_match_kind_is_required_and_validated(self):
        base = (
            "coverage\tq1\tok\t\nordered_equal\tTrue\ncollections\t1\n"
            "runtime_audit\tno_marker\n"
        )
        with self.assertRaises(ValueError):
            driver.parse(base, "q1")
        with self.assertRaises(ValueError):
            driver.parse(base + "match\tclose\n", "q1")
        self.assertEqual(
            driver.parse(base + "match\texact\n", "q1")["match"], "exact"
        )


if __name__ == "__main__":
    unittest.main()
