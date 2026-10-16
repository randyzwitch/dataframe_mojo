"""Coverage receipts must not silently inflate parity or mix measurements."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "metal_query_report", ROOT / "benchmarks/tools/metal_query_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def case(query, variant="base", status="ok", rows=1):
    return dict(
        suite="example",
        query=query,
        variant=variant,
        status=status,
        reason="",
        output_rows=rows,
        ordered_equal=True,
        collections=2,
    )


def receipt(cases):
    return {
        "format": "dataframe-metal-suite-coverage-v1",
        "provenance": {
            "engine_revision": "a" * 40,
            "implementation": "mojo",
            "engine_status": "",
            "mojo": "fixture",
            "threads": 8,
            "scale": "smoke",
            "repetitions": 2,
            "driver_sha256": "driver",
            "comparison": "exact",
            "original_source_sha256": {"query.mojo": "source"},
        },
        "cases": cases,
    }


def definitions():
    return {
        "example": {
            "queries": ["q1", "q2", "q3"],
            "variants": ["base", "decimal"],
            "translated_queries": ["q1", "q2"],
        }
    }


class MetalQueryReport(unittest.TestCase):
    def test_catalog_is_source_derived_and_untranslated_cells_remain_visible(self):
        catalog = report.catalog()
        self.assertEqual(sum(len(d["queries"]) for d in catalog.values()), 179)
        self.assertEqual(
            sum(len(d["translated_queries"]) for d in catalog.values()), 168
        )
        self.assertEqual(
            sum(len(d["queries"]) * len(d["variants"]) for d in catalog.values()), 345
        )
        self.assertNotIn("q28", catalog["clickbench"]["translated_queries"])
        self.assertEqual(
            set(catalog["tpcds"]["queries"])
            - set(catalog["tpcds"]["translated_queries"]),
            {"q5", "q14", "q18", "q22", "q36", "q67", "q70", "q77", "q80", "q86"},
        )

    def test_partial_grid_cannot_claim_all_variants_or_complete_percent(self):
        c = case("q1")
        summary = report.summarize({report.key(c): c}, definitions(), {}, {})
        self.assertFalse(summary["complete"])
        self.assertEqual(len(summary["unmeasured_cases"]), 5)
        self.assertIsNone(
            summary["raw_adapter"]["matching_percent_of_translated_cases"]
        )
        self.assertEqual(summary["raw_adapter"]["all_variants_matching_queries"], [])

    def test_complete_counts_keep_wrong_unsupported_empty_and_hybrid_separate(self):
        rows = [
            case("q1", rows=0),
            case("q1", "decimal"),
            case("q2", status="wrong_answer"),
            case("q2", "decimal", status="unsupported"),
            case("q3", status="reference_unsupported"),
            case("q3", "decimal", status="reference_unsupported"),
        ]
        rows[2]["diagnostics"] = {"cpu_repeat_ordered_equal": False}
        metadata = {
            "cases": {
                "example/q1": {
                    "execution_boundary": "hybrid",
                    "source_path": "query.mojo",
                    "source_sha256": "source",
                    "audit_reason": "eager mean",
                }
            }
        }
        summary = report.summarize(
            {report.key(c): c for c in rows},
            definitions(),
            {"query.mojo": "source"},
            metadata,
        )
        self.assertTrue(summary["complete"])
        raw = summary["raw_adapter"]
        self.assertEqual((raw["matching_cases"], raw["runnable_cases"]), (2, 3))
        self.assertEqual(
            (raw["matching_empty_cases"], raw["matching_populated_cases"]), (1, 1)
        )
        self.assertEqual(raw["matching_percent_of_translated_cases"], 50)
        self.assertEqual(raw["all_variants_matching_queries"], ["example/q1"])
        conservative = summary["known_hybrid_and_unknown_audits_excluded"]
        self.assertEqual(
            (conservative["matching_cases"], conservative["excluded_cases"]), (0, 2)
        )
        self.assertEqual(summary["cpu_repeat_ordered_equality"], {"False": 1})
        self.assertEqual(
            raw["statuses"]["wrong_answer"], 1
        )  # instability never excuses a difference

    def test_stale_execution_audit_fails_closed(self):
        metadata = {
            "cases": {
                "example/q1": {
                    "execution_boundary": "gpu_collects",
                    "source_path": "query.mojo",
                    "source_sha256": "old",
                    "audit_reason": "reviewed",
                }
            }
        }
        self.assertEqual(
            report.execution_audit(case("q1"), {"query.mojo": "new"}, metadata)[
                "execution_boundary"
            ],
            "unknown",
        )

    def test_untranslated_success_cannot_inflate_support_counts(self):
        c = case("q3")
        with self.assertRaisesRegex(ValueError, "absent from the translated"):
            report.summarize({report.key(c): c}, definitions(), {}, {})

    def test_cli_cannot_overwrite_immutable_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            original = json.dumps(receipt([case("q1")]))
            path.write_text(original)
            process = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks/tools/metal_query_report.py"),
                    str(path),
                    "--output",
                    str(path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("must not overwrite", process.stderr)
            self.assertEqual(path.read_text(), original)

    def test_duplicate_dirty_mixed_revision_and_partial_success_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "one.json", Path(tmp) / "two.json"
            original = receipt([case("q1")])
            first.write_text(json.dumps(original))
            second.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                report.load_receipts([first, second])
            altered = copy.deepcopy(original)
            altered["provenance"]["engine_revision"] = "b" * 40
            second.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "Mixed measurement"):
                report.load_receipts([first, second])
            with self.assertRaisesRegex(ValueError, "Unexpected engine"):
                report.load_receipts([first], revision="b" * 40)
            altered = copy.deepcopy(original)
            altered["provenance"]["engine_status"] = " M dataframe/frame.mojo"
            second.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "Dirty"):
                report.load_receipts([second])
            altered = copy.deepcopy(original)
            del altered["cases"][0]["ordered_equal"]
            second.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "lacks comparison"):
                report.load_receipts([second])

    def test_cpu_repeat_log_is_diagnostic_only_and_missing_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            c = case("q1", status="wrong_answer")
            c["log"] = "case.txt"
            self.assertEqual(report.cpu_repeat(c, path), {})
            (path.parent / "case.txt").write_text("cpu_repeat_equal\tFalse\n")
            self.assertEqual(
                report.cpu_repeat(c, path), {"cpu_repeat_ordered_equal": False}
            )
            self.assertEqual(c["status"], "wrong_answer")


if __name__ == "__main__":
    unittest.main()
