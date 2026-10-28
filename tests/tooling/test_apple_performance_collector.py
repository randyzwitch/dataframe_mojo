"""Performance protocol and evidence admission; no compiler or timed queries."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "apple_performance", ROOT / "benchmarks/tools/bench_apple_performance.py"
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
from suite_answer_fixture import export_fixture


def protocol():
    result = ""
    for phase, count in (("cold", 1), ("warmup", 1), ("warm", 3)):
        for index in range(count):
            result += f"sample\t{phase}\t{index}\t{index + 1}\nanswer\t{phase}\t{index}\t1\t3.0\tv\n"
    return result + "outcome\tok\t\n"


class ApplePerformanceCollector(unittest.TestCase):
    def test_complete_answer_protocol_binds_every_phase(self):
        with tempfile.TemporaryDirectory() as folder:
            records = [
                export_fixture(repetition=i, directory=folder) for i in range(5)
            ]
            output = (
                "".join(
                    "complete_answer\tq1\t" + json.dumps(record) + "\n"
                    for record in records
                )
                + protocol()
            )
            run = dict(
                collector.parse(
                    output, 3, 1, answer_directory=folder, query="q1"
                ),
                suite="h2o_groupby",
                query="q1",
            )
            self.assertEqual(
                [
                    (s["phase"], s["index"], s["complete_answer"]["repetition"])
                    for s in run["samples"]
                ],
                [
                    ("cold", 0, 0),
                    ("warmup", 0, 1),
                    ("warm", 0, 2),
                    ("warm", 1, 3),
                    ("warm", 2, 4),
                ],
            )
            reference = dict(status="ok", oracles=[export_fixture(oracle=True)])
            self.assertEqual(
                collector.oracle_checks(run, reference)["status"], "matches"
            )
            run["samples"][-1].pop("complete_answer")
            self.assertEqual(
                collector.oracle_checks(run, reference)["status"], "unverified"
            )

    def test_rotated_order_retains_every_case_engine_round(self):
        cases = [dict(query="q1"), dict(query="q2")]
        plan = collector.schedule(cases, 3)
        self.assertEqual(len(plan), 12)
        self.assertEqual(
            [(r["query"], r["engine"]) for r in plan[:4]],
            [("q1", "cpu"), ("q1", "metal"), ("q2", "metal"), ("q2", "cpu")],
        )
        self.assertEqual([r["engine"] for r in plan[4:6]], ["metal", "cpu"])
        self.assertEqual(cases, [dict(query="q1"), dict(query="q2")])

    def test_complete_samples_keep_cold_warmup_warm_separate(self):
        result = collector.parse(protocol(), 3, 1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            [s["phase"] for s in result["samples"]],
            ["cold", "warmup", "warm", "warm", "warm"],
        )
        self.assertEqual(result["samples"][0]["summary"]["values"], ["3.0"])
        json.dumps(result, allow_nan=False)
        nonfinite = collector.parse(protocol().replace("3.0", "nan"), 3, 1)
        json.dumps(nonfinite, allow_nan=False)

    def test_protocol_never_invents_missing_or_duplicate_success(self):
        for text in (
            "",
            "outcome\tok\t\n",
            protocol() + "outcome\tok\t\n",
            protocol().replace("sample\tcold\t0\t1\n", ""),
            protocol().replace("sample\tcold\t0\t1\n", "sample\tcold\t0\t-1\n"),
            protocol().replace("sample\tcold\t0\t1\n", "sample\tcold\t0\t1\n" * 2),
            protocol().replace(
                "answer\tcold\t0\t1\t3.0\tv", "answer\tcold\t0\t1\t3.0,4.0\tv"
            ),
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                collector.parse(text, 3, 1)
        failed = collector.parse(
            "sample\tcold\t0\t5\noutcome\texecution_failed\tresource budget exceeded\n",
            3,
            1,
        )
        self.assertEqual(failed["samples"][0]["ns"], 5)
        self.assertEqual(failed["reason"], "resource budget exceeded")

    def test_process_failure_keeps_completed_raw_collections_without_success(self):
        result = collector.parse("sample\tcold\t0\t123\n", 3, 1, allow_incomplete=True)
        self.assertEqual(result["status"], "uncompleted_protocol")
        self.assertEqual(result["samples"], [dict(phase="cold", index=0, ns=123)])
        failed = collector.parse(
            "sample\tcold\t0\t123\noutcome\tanswer_failed\tuntimed answer summary: memory\n",
            3,
            1,
        )
        self.assertEqual(failed["status"], "answer_failed")

    def test_staged_queries_use_existing_forced_derivation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = collector.stage(root)
            expected = {
                f"{prefix}_{name}.mojo"
                for prefix in ("cpu", "metal", "suite", "performance")
                for name in ("h2o", "pdsh", "tpcds", "clickbench")
            }
            expected |= {
                "suite_common.mojo",
                "metal_suite_common.mojo",
                "metal_suite_timing.mojo",
            }
            self.assertTrue(expected <= set(receipt["derived_source_sha256"]))
            self.assertEqual(
                set(receipt["derived_source_sha256"]),
                {p.name for p in root.glob("*.mojo")},
            )
            for name in ("h2o", "pdsh", "tpcds", "clickbench"):
                original = (ROOT / "benchmarks/suites" / (name + ".mojo")).read_text()
                self.assertEqual(
                    (root / ("cpu_" + name + ".mojo")).read_text(),
                    collector.metal.derive(original, name, "cpu"),
                )
                self.assertEqual(
                    (root / ("metal_" + name + ".mojo")).read_text(),
                    collector.metal.derive(original, name, "metal"),
                )
            # These callbacks must return completed DataFrames. The plan()
            # exports of PDS-H/TPC-DS return LazyFrames and cannot be timed here.
            for name in ("pdsh", "tpcds", "clickbench"):
                wrapper = (root / f"performance_{name}.mojo").read_text()
                for engine in ("cpu", "metal"):
                    self.assertIn(
                        f"from {engine}_{name} import query as {engine}_query",
                        wrapper,
                    )
            self.assertNotIn("def main", (root / "metal_suite_timing.mojo").read_text())
            self.assertNotIn(
                "reference(", (root / "metal_suite_timing.mojo").read_text()
            )

    def test_input_preparation_covers_variants_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.parquet"
            path.write_bytes(b"literal fixture")
            with patch.object(collector.suites, "tables", return_value={"x": path}):
                plan = collector.input_plan(["h2o_groupby"], {"q1"}, "smoke")
            self.assertEqual(len(plan["cases"]), 5)
            collector.validate_input_plan(plan, ["h2o_groupby"], {"q1"}, "smoke")
            self.assertTrue(collector.inputs_unchanged(plan["cases"]))
            altered = copy.deepcopy(plan)
            altered["cases"].pop()
            with self.assertRaises(ValueError):
                collector.validate_input_plan(altered, ["h2o_groupby"], {"q1"}, "smoke")
            altered = copy.deepcopy(plan)
            altered["cases"].append(altered["cases"][0])
            with self.assertRaises(ValueError):
                collector.validate_input_plan(altered, ["h2o_groupby"], {"q1"}, "smoke")
            path.write_bytes(b"changed")
            self.assertFalse(collector.inputs_unchanged(plan["cases"]))

    def test_strict_basis_requires_matching_engine_sdk_threads_source_inputs(self):
        case = dict(
            suite="example",
            variant="base",
            query="q1",
            inputs={"x": {"sha256": "input"}},
        )
        engine = dict(revision="engine", status="")
        sources = {"benchmarks/suites/example.mojo": "source"}
        row = dict(
            case,
            status="ok",
            ordered_equal=False,
            collections=2,
            runtime_adjudication=dict(
                status="complete",
                observed_calls=2,
                cpu_adjudicated_calls=0,
                no_cpu_adjudication_marker_calls=2,
                unknown_calls=0,
                failed_calls=0,
                events=["no_marker", "no_marker"],
            ),
        )
        wrapper_path = "benchmarks/metal_coverage_audit.mojo"
        wrapper_sha = collector.sha(ROOT / wrapper_path)
        built_driver_provenance = collector.metal.driver_provenance()
        receipt = dict(
            provenance=dict(
                engine_revision="engine",
                engine_status="",
                mojo="sdk",
                threads=8,
                **built_driver_provenance,
                original_source_sha256={**sources, wrapper_path: wrapper_sha},
                derived_source_sha256={"metal_example.mojo": "derived"},
            ),
            cases=[row],
        )
        metadata = {
            "format": "dataframe-metal-case-metadata-v1",
            "cases": {
                "example/base/q1": {
                    "execution_boundary": "gpu_collects",
                    "workload_kind": "canonical",
                    "audit_reason": "Fixture with pinned derived worker.",
                    "source_path": "benchmarks/suites/example.mojo",
                    "source_sha256": "source",
                    "metal_derived_source_sha256": "derived",
                }
            },
        }
        self.assertEqual(
            built_driver_provenance["driver_sha256"],
            collector.sha(collector.metal.__file__),
        )
        result = collector.strict_basis(
            case, [receipt], engine, sources, "sdk", 8, metadata
        )
        self.assertTrue(result["verified"])
        self.assertTrue(result["runtime_audit_verified"])
        for field, value in (
            ("engine_revision", "old"),
            ("mojo", "old"),
            ("threads", 4),
            ("engine_status", "dirty"),
        ):
            changed = copy.deepcopy(receipt)
            changed["provenance"][field] = value
            self.assertFalse(
                collector.strict_basis(
                    case, [changed], engine, sources, "sdk", 8, metadata
                )["verified"]
            )
        for field, value in (
            ("status", "wrong_answer"),
            ("collections", 0),
            ("ordered_equal", None),
        ):
            changed = copy.deepcopy(receipt)
            changed["cases"][0][field] = value
            self.assertFalse(
                collector.strict_basis(
                    case, [changed], engine, sources, "sdk", 8, metadata
                )["verified"]
            )
        self.assertFalse(
            collector.strict_basis(
                dict(case, inputs={"x": {"sha256": "different"}}),
                [receipt],
                engine,
                sources,
                "sdk",
                8,
                metadata,
            )["verified"]
        )
        self.assertFalse(
            collector.strict_basis(
                case,
                [receipt],
                engine,
                {"benchmarks/suites/example.mojo": "new"},
                "sdk",
                8,
                metadata,
            )["verified"]
        )
        stale_wrapper = copy.deepcopy(receipt)
        stale_wrapper["provenance"]["original_source_sha256"][wrapper_path] = "old"
        self.assertFalse(
            collector.strict_basis(
                case, [stale_wrapper], engine, sources, "sdk", 8, metadata
            )["runtime_audit_verified"]
        )
        stale_driver = copy.deepcopy(receipt)
        stale_driver["provenance"]["driver_sha256"] = collector.sha(
            collector.coverage.__file__
        )
        self.assertFalse(
            collector.strict_basis(
                case, [stale_driver], engine, sources, "sdk", 8, metadata
            )["runtime_audit_verified"]
        )
        stale_derived = copy.deepcopy(receipt)
        stale_derived["provenance"]["derived_source_sha256"][
            "metal_example.mojo"
        ] = "old"
        stale_result = collector.strict_basis(
            case, [stale_derived], engine, sources, "sdk", 8, metadata
        )
        self.assertEqual(stale_result["execution_boundary"], "unknown")
        self.assertFalse(stale_result["runtime_audit_verified"])
        self.assertFalse(stale_result["verified"])
        missing_derived = copy.deepcopy(receipt)
        missing_derived["provenance"]["derived_source_sha256"].pop(
            "metal_example.mojo"
        )
        missing_result = collector.strict_basis(
            case, [missing_derived], engine, sources, "sdk", 8, metadata
        )
        self.assertEqual(missing_result["execution_boundary"], "unknown")
        self.assertFalse(missing_result["runtime_audit_verified"])
        old_receipt = copy.deepcopy(receipt)
        old_receipt["cases"][0].pop("runtime_adjudication")
        self.assertFalse(
            collector.strict_basis(
                case, [old_receipt], engine, sources, "sdk", 8, metadata
            )["verified"]
        )

    def test_oracle_policy_remains_separate_from_exact_outcome(self):
        run = dict(collector.parse(protocol(), 3, 1), suite="h2o_groupby", query="q1")
        self.assertEqual(
            collector.oracle_checks(
                run,
                dict(
                    status="ok",
                    summary=dict(height=1, values=[3.0], names=["v"]),
                ),
            )["status"],
            "unverified",
        )
        self.assertEqual(
            collector.oracle_checks(
                run,
                dict(
                    status="ok",
                    summary=dict(height=1, values=[4.0], names=["v"]),
                ),
            )["status"],
            "unverified",
        )
        self.assertEqual(run["status"], "ok")

    def test_ratio_admission_rejects_failed_hybrid_or_incomplete_evidence(self):
        strict = dict(
            verified=True,
            execution_boundary="gpu_collects",
            runtime_audit_verified=True,
            runtime_adjudication=dict(cpu_adjudicated_calls=0),
        )
        runs = [
            dict(
                collector.parse(protocol(), 3, 1),
                engine=engine,
                round=index,
                oracle=dict(status="matches", format=collector.ANSWER_POLICY),
            )
            for index in range(3)
            for engine in ("cpu", "metal")
        ]
        self.assertTrue(
            collector.ratio_basis(runs, strict, 3, 3).startswith("eligible")
        )
        for basis in (
            dict(strict, verified=False),
            dict(strict, execution_boundary="hybrid"),
            dict(strict, runtime_audit_verified=False),
        ):
            self.assertFalse(
                collector.ratio_basis(runs, basis, 3, 3).startswith("eligible")
            )
        self.assertEqual(
            collector.ratio_basis(runs[:-1], strict, 3, 3), "incomplete_rounds"
        )
        self.assertEqual(
            collector.ratio_basis(
                runs,
                dict(
                    strict,
                    runtime_adjudication=dict(cpu_adjudicated_calls=1),
                ),
                3,
                3,
            ),
            "cpu_data_fault_adjudication_observed",
        )
        for field, value in (
            ("status", "process_failed"),
            ("compiler_overlap", True),
            ("busy_host", True),
            ("oracle", dict(status="differs")),
        ):
            changed = copy.deepcopy(runs)
            changed[0][field] = value
            self.assertFalse(
                collector.ratio_basis(changed, strict, 3, 3).startswith(
                    "eligible"
                )
            )

    def test_inherited_exports_cannot_contaminate_actual_worker_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            exported = Path(tmp) / "unexpected-answer.txt"
            inherited = dict(
                DATAFRAME_BENCH_ANSWER_EXPORT_PREFIX=str(exported),
                DATAFRAME_TRACE_PATHS="1",
                DATAFRAME_EXECUTION_REPORT="1",
                DATAFRAME_THREADS="99",
                BENCH_KEEP_SENTINEL="preserved",
            )
            env = collector.worker_environment(8, inherited)
            code = "import os,json; from pathlib import Path; prefix=os.getenv('DATAFRAME_BENCH_ANSWER_EXPORT_PREFIX'); Path(prefix).write_text('contaminated') if prefix else None; print(json.dumps(dict(os.environ)))"
            worker = collector.execute([sys.executable, "-c", code], env, 5)
            self.assertEqual(worker["status"], "completed")
            observed = json.loads(worker["stdout"])
            for name in (
                "DATAFRAME_BENCH_ANSWER_EXPORT_PREFIX",
                "DATAFRAME_TRACE_PATHS",
                "DATAFRAME_EXECUTION_REPORT",
            ):
                self.assertNotIn(name, observed)
            self.assertFalse(exported.exists())
            self.assertEqual(observed["DATAFRAME_THREADS"], "8")
            self.assertEqual(observed["BENCH_THREADS"], "8")
            self.assertEqual(observed["BENCH_KEEP_SENTINEL"], "preserved")
            self.assertEqual(
                inherited["DATAFRAME_BENCH_ANSWER_EXPORT_PREFIX"], str(exported)
            )

    def test_timeout_terminates_only_the_owned_worker_group(self):
        process = Mock(pid=12345, returncode=-9)
        process.communicate.side_effect = [
            subprocess.TimeoutExpired("worker", 1),
            ("partial", "error"),
        ]
        with patch.object(
            collector.subprocess, "Popen", return_value=process
        ) as start, patch.object(collector.os, "killpg") as kill:
            result = collector.execute(["worker"], {}, 1)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["stdout"], "partial")
        self.assertTrue(start.call_args.kwargs["start_new_session"])
        kill.assert_called_once_with(12345, collector.signal.SIGKILL)

    def test_oracle_runtime_version_is_observed_or_explicitly_unavailable(self):
        with patch.object(
            collector,
            "command",
            return_value='{"duckdb_version":"fixture-version","python_version":"fixture-python"}',
        ):
            self.assertEqual(
                collector.oracle_runtime_identity(),
                dict(
                    available=True,
                    duckdb_version="fixture-version",
                    python_version="fixture-python",
                ),
            )
        with patch.object(
            collector, "command", side_effect=subprocess.CalledProcessError(1, "python")
        ):
            self.assertFalse(collector.oracle_runtime_identity()["available"])

    def test_darwin_peak_memory_is_explicit_and_not_inferred_from_signal(self):
        self.assertEqual(
            collector.peak_rss("  123456 maximum resident set size\n"), 123456
        )
        self.assertIsNone(collector.peak_rss("Command terminated by signal 9\n"))
        self.assertIsNone(
            collector.peak_rss(
                "1 maximum resident set size\n2 maximum resident set size\n"
            )
        )


if __name__ == "__main__":
    unittest.main()
