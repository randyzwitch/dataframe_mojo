"""Synthetic fixtures exercise rendering/admission; never performance evidence."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "apple_html", ROOT / "benchmarks/tools/apple_performance_html.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
from suite_answer_fixture import export_fixture


def fixture():
    engine = dict(
        revision="a" * 40,
        status="",
        package_sha256={"dataframe/__init__.mojo": "SYNTHETIC"},
    )
    sources = {"benchmarks/suites/h2o.mojo": "SYNTHETIC"}
    cases = [
        dict(
            suite="h2o_groupby",
            variant=variant,
            query="q1",
            inputs={"x": dict(path="synthetic.parquet", sha256=variant)},
            strict_basis=dict(
                receipt_sha256="strict",
                verified=True,
                status="ok",
                execution_boundary="gpu_collects",
            ),
        )
        for variant in ("k100", "k10")
    ]
    wrapper_path = "benchmarks/metal_coverage_audit.mojo"
    strict = dict(
        receipt_sha256="strict",
        format="dataframe-metal-suite-coverage-v1",
        provenance=dict(
            engine_revision=engine["revision"],
            engine_status="",
            implementation="mojo",
            mojo="SYNTHETIC SDK",
            threads=8,
            scale="default",
            repetitions=3,
            driver_sha256=report.collector.sha(report.collector.metal.__file__),
            comparison="exact",
            original_source_sha256={
                **sources,
                wrapper_path: report.collector.sha(ROOT / wrapper_path),
            },
            derived_source_sha256={"worker.mojo": "SYNTHETIC"},
        ),
        cases=[
            dict(
                c,
                status="ok",
                ordered_equal=True,
                collections=3,
                output_rows=2,
                runtime_adjudication=dict(
                    status="complete",
                    observed_calls=3,
                    cpu_adjudicated_calls=0,
                    no_cpu_adjudication_marker_calls=3,
                    unknown_calls=0,
                    failed_calls=0,
                    events=["no_marker", "no_marker", "no_marker"],
                ),
            )
            for c in cases
        ],
    )
    data = dict(
        format=report.collector.FORMAT,
        provenance=dict(
            engine=engine,
            mojo="SYNTHETIC SDK",
            rounds=3,
            repetitions=3,
            warmups=1,
            threads=8,
            scale="default",
            harness_revision="b" * 40,
            harness_sha256={"collector": "SYNTHETIC"},
            staged=dict(original_source_sha256=sources),
            recorded_at_utc="SYNTHETIC TEST FIXTURE",
            physical_memory_bytes=16 * 1024**3,
            headline_full_tier_attempt=False,
        ),
        cases=cases,
        runs=[],
        oracle={},
        engine_unchanged_after=True,
        inputs_unchanged_after=True,
        all_scheduled_workers_attempted=True,
    )
    for case in cases:
        data["oracle"]["/".join(report.coverage.key(case))] = dict(
            status="completed",
            answer=dict(
                status="ok",
                summary=dict(height=2, values=["3.0"], names=["value"]),
                oracles=[export_fixture(oracle=True)],
            ),
        )
        for round_index in range(3):
            for engine_name in ("cpu", "metal"):
                base = 100 + 10 * round_index
                if engine_name == "metal" and case["variant"] == "k100":
                    base //= 2
                samples = [
                    dict(phase="cold", index=0, ns=base * 3),
                    dict(phase="warmup", index=0, ns=base * 2),
                ]
                samples += [
                    dict(phase="warm", index=i, ns=base + i * 10)
                    for i in range(3)
                ]
                for ordinal, sample in enumerate(samples):
                    sample["summary"] = dict(
                        height=2, values=["3.0"], names=["value"]
                    )
                    sample["complete_answer"] = export_fixture(
                        repetition=ordinal
                    )
                data["runs"].append(
                    dict(
                        case,
                        engine=engine_name,
                        round=round_index,
                        status="ok",
                        samples=samples,
                        oracle=dict(status="matches"),
                        compiler_overlap=False,
                        busy_host=False,
                        host_before=dict(
                            compilers=[], load_average=[0.1, 0.1, 0.1]
                        ),
                        host_after=dict(
                            compilers=[], load_average=[0.2, 0.2, 0.2]
                        ),
                    )
                )
    return data, strict


class ApplePerformanceHTML(unittest.TestCase):
    def test_metrics_use_only_admitted_paired_cases_and_distinguish_queries(self):
        data, strict = fixture()
        result = report.analyze(data, [strict], {})
        self.assertEqual(
            (result["admitted_cases"], result["admitted_query_ids"]), (2, 1)
        )
        proof = result["rows"][0]["confirmed_strict_basis"][
            "runtime_adjudication"
        ]
        self.assertEqual(proof["status"], "complete")
        self.assertEqual(proof["cpu_adjudicated_calls"], 0)
        self.assertEqual(proof["no_cpu_adjudication_marker_calls"], 3)
        self.assertAlmostEqual(result["geometric_mean"], math.sqrt(65 / 120))
        self.assertAlmostEqual(result["policy_geometric_mean"], 1 / math.sqrt(2))
        row = result["rows"][0]
        self.assertEqual(row["times"]["cpu"]["warm"]["median_round_minimum_ns"], 110)
        self.assertEqual(row["times"]["cpu"]["warm"]["median_ns"], 120)
        self.assertEqual(row["times"]["cpu"]["warm"]["minimum_ns"], 100)
        self.assertEqual(row["times"]["cpu"]["warm"]["maximum_ns"], 140)
        self.assertEqual(row["times"]["cpu"]["cold"]["median_ns"], 330)
        self.assertEqual(result["worst_variants"][0]["variant"], "k10")
        self.assertAlmostEqual(
            result["suite_geometric_means"]["h2o_groupby"], math.sqrt(65 / 120)
        )
        # Below 1 means Metal was faster; the worst variant is the highest.
        self.assertAlmostEqual(result["worst_variants"][0]["ratio"], 1)
        self.assertLess(result["rows"][0]["ratio"], 1)
        page = report.render(data, result, receipt_path="x", receipt_sha="y")
        self.assertIn("Metal / CPU geometric mean", page)
        # Round-minimum geometric mean in the table, raw median in the text.
        self.assertIn("0.71", page)
        self.assertIn("0.736", page)
        for header in ("cpu ms", "metal ms", "vs cpu", "worst vs cpu (variant)", "Status"):
            self.assertIn(header, page)
        # Tables come before the provenance detail, and the page stays small.
        self.assertLess(page.index('id="h2o_groupby"'), page.index('id="excluded"'))
        self.assertLess(page.index('id="excluded"'), page.index('id="provenance"'))
        self.assertLess(len(page), 200_000)
        self.assertNotIn("CPU / Metal", page)
        self.assertEqual(result["catalog_cases"], 345)
        self.assertEqual(len(result["unplanned_cases"]), 343)

    def test_wrong_answer_never_promoted_by_cached_eligible_or_oracle(self):
        data, strict = fixture()
        strict["cases"][0]["status"] = "wrong_answer"
        data["cases"][0]["ratio_basis"] = "eligible"
        result = report.analyze(data, [strict], {})
        self.assertIsNone(result["rows"][0]["ratio"])
        self.assertEqual(result["admitted_cases"], 1)
        self.assertEqual(result["geometric_mean"], 1)
        self.assertEqual(
            result["rows"][0]["confirmed_strict_basis"]["status"], "wrong_answer"
        )

    def test_rechecks_sdk_input_source_and_supplied_exact_receipt(self):
        for mutate in (
            lambda d, s: s["provenance"].update(mojo="different"),
            lambda d, s: s["cases"][0]["inputs"]["x"].update(sha256="different"),
            lambda d, s: s["provenance"].update(
                original_source_sha256={"benchmarks/suites/h2o.mojo": "different"}
            ),
            lambda d, s: d["cases"][0]["strict_basis"].update(receipt_sha256="other"),
            lambda d, s: s["cases"][0].pop("runtime_adjudication"),
            lambda d, s: s["cases"][0]["runtime_adjudication"].update(
                status="incomplete"
            ),
            lambda d, s: s["cases"][0]["runtime_adjudication"].update(
                observed_calls=2
            ),
            lambda d, s: s["cases"][0]["runtime_adjudication"].update(
                cpu_adjudicated_calls=1
            ),
            lambda d, s: s["cases"][0]["runtime_adjudication"].update(
                unknown_calls=1
            ),
            lambda d, s: s["provenance"]["original_source_sha256"].update(
                {"benchmarks/metal_coverage_audit.mojo": "different"}
            ),
            lambda d, s: s["provenance"].update(driver_sha256="different"),
        ):
            data, strict = fixture()
            strict = copy.deepcopy(strict)
            mutate(data, strict)
            self.assertIsNone(report.analyze(data, [strict], {})["rows"][0]["ratio"])
        data, strict = fixture()
        self.assertEqual(report.analyze(data, [], {})["admitted_cases"], 0)

    def test_rechecks_individual_complete_answers_not_cached_oracle_status(
        self,
    ):
        data, strict = fixture()
        data["runs"][0]["samples"][0]["complete_answer"] = export_fixture(
            {"id1": ["a", "b"], "v1": [9999, 2]}
        )
        self.assertIsNone(
            report.analyze(data, [strict], {})["rows"][0]["ratio"]
        )
        data, strict = fixture()
        data["oracle"].clear()
        self.assertEqual(
            report.analyze(data, [strict], {})["admitted_cases"], 0
        )

    def test_missing_failure_busy_or_changed_identity_stays_visible_without_ratio(self):
        for mode in (
            "missing",
            "unsupported",
            "process_failed",
            "timeout",
            "answer_failed",
            "busy",
            "compiler",
            "missing_host",
            "identity",
        ):
            data, strict = fixture()
            if mode == "missing":
                data["runs"].pop(0)
            elif mode in {"unsupported", "process_failed", "timeout", "answer_failed"}:
                data["runs"][0].update(
                    status=mode, reason="resource / operation evidence"
                )
            elif mode == "busy":
                data["runs"][0]["host_before"]["load_average"][0] = 9
            elif mode == "compiler":
                data["runs"][0]["host_after"]["compilers"] = [["123", "mojo"]]
            elif mode == "missing_host":
                data["runs"][0].pop("host_before")
            else:
                data["inputs_unchanged_after"] = False
            result = report.analyze(data, [strict], {})
            self.assertIsNone(result["rows"][0]["ratio"], mode)
            self.assertIsNotNone(result["rows"][0]["times"]["cpu"]["warm"], mode)

    def test_hybrid_excluded_and_full_boundary_record_survives(self):
        data, strict = fixture()
        audit = dict(
            workload_kind="canonical",
            execution_boundary="hybrid",
            source_path="benchmarks/suites/h2o.mojo",
            source_sha256="SYNTHETIC",
            audit_reason="SYNTHETIC post-collect operator",
            whole_query_gpu_certificate=False,
            execution_boundaries=["collect", "cpu aggregate"],
        )
        strict["cases"][0]["execution_audit"] = audit
        result = report.analyze(data, [strict], {})
        self.assertIsNone(result["rows"][0]["ratio"])
        self.assertEqual(
            result["rows"][0]["confirmed_strict_basis"]["execution_audit"], audit
        )

    def test_malformed_or_duplicate_measurements_fail_closed(self):
        for mutate in (
            lambda d: d["runs"].append(d["runs"][0]),
            lambda d: d["cases"].append(d["cases"][0]),
            lambda d: d["runs"][0]["samples"].append(d["runs"][0]["samples"][0]),
            lambda d: d["runs"][0]["samples"][0].update(ns=-1),
            lambda d: d["runs"][0]["samples"][0].update(ns=True),
            lambda d: d["runs"][0]["samples"].pop(),
            lambda d: d["runs"][0].update(round=99),
            lambda d: d["provenance"].update(staged={}),
        ):
            data, strict = fixture()
            mutate(data)
            with self.assertRaises(ValueError):
                report.analyze(data, [strict], {})

    def test_standalone_html_escapes_evidence_and_exposes_exclusions(self):
        data, strict = fixture()
        data["runs"][0].update(
            status="process_failed", reason="<script>alert('x')</script>"
        )
        rendered = report.render(
            data,
            report.analyze(data, [strict], {}),
            receipt_path="raw.json",
            receipt_sha="hash",
        )
        self.assertNotIn("<script>alert", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("SYNTHETIC TEST FIXTURE", rendered)
        self.assertIn("cpu ms", rendered)
        self.assertIn("Metal / CPU geometric mean", rendered)
        self.assertIn("unverified", rendered)
        self.assertIn("Excluded cases", rendered)
        self.assertIn("Unmeasured / receipt not supplied", rendered)
        self.assertNotIn("https://", rendered)

    def test_coverage_counts_keep_variants_and_empty_separate(self):
        data, strict = fixture()
        definitions = report.coverage.catalog()
        strict["provenance"]["original_source_sha256"] = {
            v["source_path"]: v["source_sha256"] for v in definitions.values()
        }
        strict["cases"][0]["output_rows"] = 0
        strict["cases"][1]["status"] = "wrong_answer"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coverage.json"
            path.write_text(json.dumps(strict))
            evidence = report.coverage_evidence([path], {})
        summary = evidence["summary"]
        self.assertEqual(summary["raw_adapter"]["matching_empty_cases"], 1)
        self.assertEqual(summary["raw_adapter"]["matching_cases"], 1)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["raw_adapter"]["all_variants_matching_queries"], [])
        self.assertIn("partial", report.coverage_comparison(evidence, evidence))
        changed = copy.deepcopy(evidence)
        changed["derived_workers"] = [{"worker.mojo": "changed adapter"}]
        self.assertIn(
            "not attributed to kernel gains",
            report.coverage_comparison(evidence, changed),
        )

    def test_pinned_historical_catalog_ignores_cached_summary_counts(self):
        data, strict = fixture()
        definitions = report.coverage.catalog()
        for item in definitions.values():
            item["source_sha256"] = "SYNTHETIC HISTORICAL " + item["source_path"]
        strict["provenance"]["original_source_sha256"] = {
            v["source_path"]: v["source_sha256"] for v in definitions.values()
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coverage.json"
            catalog = Path(tmp) / "catalog.json"
            path.write_text(json.dumps(strict))
            catalog.write_text(
                json.dumps(
                    dict(
                        format="dataframe-metal-suite-summary-v1",
                        catalog=definitions,
                        summary=dict(matching_cases=999999),
                    )
                )
            )
            with self.assertRaisesRegex(ValueError, "source-pinned summary"):
                report.coverage_evidence([path], {})
            evidence = report.coverage_evidence([path], {}, catalog)
            self.assertEqual(evidence["summary"]["raw_adapter"]["matching_cases"], 2)
            self.assertEqual(evidence["catalog_receipt"]["sha256"], report.sha(catalog))

    def test_complete_like_for_like_coverage_delta_requires_same_audits(self):
        definitions = report.coverage.catalog()
        data, strict = fixture()
        strict["provenance"]["original_source_sha256"] = {
            v["source_path"]: v["source_sha256"] for v in definitions.values()
        }
        strict["cases"] = []
        for suite, spec in definitions.items():
            for variant in spec["variants"]:
                for query in spec["queries"]:
                    strict["cases"].append(
                        dict(
                            suite=suite,
                            variant=variant,
                            query=query,
                            inputs={"x": dict(sha256="same inputs")},
                            status="ok" if query
                            in spec["translated_queries"] else "reference_unsupported",
                            ordered_equal=True,
                            collections=3,
                            output_rows=1,
                        )
                    )
        initial = copy.deepcopy(strict)
        initial["cases"][0]["status"] = "wrong_answer"
        with tempfile.TemporaryDirectory() as tmp:
            before_path, after_path = (
                Path(tmp) / "before.json",
                Path(tmp) / "after.json",
            )
            before_path.write_text(json.dumps(initial))
            after_path.write_text(json.dumps(strict))
            before = report.coverage_evidence([before_path], {})
            after = report.coverage_evidence([after_path], {})
        self.assertIn("changed by +1", report.coverage_comparison(before, after))
        changed = copy.deepcopy(after)
        first = next(iter(changed["execution_audits"]))
        changed["execution_audits"][first]["execution_boundary"] = "hybrid"
        self.assertIn(
            "not attributed to kernel gains",
            report.coverage_comparison(before, changed),
        )

    def test_cli_admits_separate_timing_receipts_without_changing_support_counts(self):
        data, strict = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strict_path = root / "default-strict.json"
            strict_path.write_text(json.dumps(strict))
            digest = report.sha(strict_path)
            for case in data["cases"]:
                case["strict_basis"]["receipt_sha256"] = digest
            source = root / "timings.json"
            source.write_text(json.dumps(data))
            output = root / "report.html"
            script = ROOT / "benchmarks/tools/apple_performance_html.py"
            metadata = root / "synthetic-metadata.json"
            metadata.write_text(json.dumps(dict(format="dataframe-metal-case-metadata-v1", cases={})))
            argv = [sys.executable, str(script), str(source), "--strict-coverage", str(strict_path),
                    "--case-metadata", str(metadata)]
            result = subprocess.run(argv + ["--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("2 admitted pairs", result.stdout)
            self.assertIn(digest, output.read_text())
            self.assertIn("Unmeasured / receipt not supplied", output.read_text())
            unchanged = strict_path.read_bytes()
            refused = subprocess.run(argv + ["--output", str(strict_path)], capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertEqual(strict_path.read_bytes(), unchanged)

    def test_cli_never_overwrites_input_and_missing_coverage_yields_no_ratios(self):
        data, strict = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "timings.json"
            output = Path(tmp) / "report.html"
            original = json.dumps(data)
            source.write_text(original)
            script = ROOT / "benchmarks/tools/apple_performance_html.py"
            bad = subprocess.run(
                [sys.executable, str(script), str(source), "--output", str(source)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(bad.returncode, 0)
            self.assertEqual(source.read_text(), original)
            good = subprocess.run(
                [sys.executable, str(script), str(source), "--output", str(output)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertIn("0 admitted pairs", good.stdout)
            self.assertIn("Metal / CPU geometric mean", output.read_text())


if __name__ == "__main__":
    unittest.main()
