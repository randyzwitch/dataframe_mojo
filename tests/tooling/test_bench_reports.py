"""Benchmark exposure, immutable rerendering, and complete query coverage."""

import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "tools"))
import bench_html
import bench_policy
import bench_suites as bench


def fixture():
    result = {
        "provenance": {
            "utc": "2026-10-04T00:00:00Z",
            "revision": "a" * 40,
            "dirty": False,
            "mojo": "fixture",
            "polars": "fixture",
            "duckdb": "fixture",
            "cpu": "fixture",
            "threads": 8,
            "scale": "smoke",
            "rounds": 1,
            "reps": 1,
        },
        "engines": ["mojo", "polars", "duckdb"],
        "runs": [],
        "trace": {},
    }
    for suite, spec in bench.SUITES.items():
        for variant in bench.VARIANTS[suite]:
            for query in spec["queries"]:
                case = f"{suite}/{variant}/{query}"
                result["trace"][case] = [
                    "group_by.partitioned.indexed"
                ] if suite == "h2o_groupby" and query == "q1" else []
                for engine in result["engines"]:
                    result["runs"].append(
                        {
                            "suite": suite,
                            "variant": variant,
                            "query": query,
                            "engine": engine,
                            "round": 0,
                            "status": "unsupported",
                            "reason": "synthetic report fixture",
                        }
                    )
    return result


class BenchmarkReports(unittest.TestCase):
    def test_catalog_preserves_holdouts_and_records_prior_exposure(self):
        self.assertEqual(
            set(bench.TIERS["full"]["suites"].split(",")), set(bench.SUITES)
        )
        self.assertEqual(
            sum(len(s["queries"]) for s in bench.SUITES.values()), 179
        )
        self.assertEqual(
            {
                name
                for name, spec in bench.SUITES.items()
                if spec["role"] == "heldout"
            },
            {"clickbench"},
        )
        # PDS-H is development since 2026-10-05 and TPC-DS since 2026-10-15;
        # the quick tier runs every development suite and no held-out one.
        self.assertEqual(bench.SUITES["pdsh"]["role"], "dev")
        self.assertEqual(bench.SUITES["tpcds"]["role"], "dev")
        self.assertEqual(
            bench.TIERS["quick"]["suites"].split(","),
            ["h2o_groupby", "h2o_join", "pdsh", "tpcds"],
        )
        info = bench_policy.record(ROOT, bench.SUITES, bench.SUITES)
        self.assertEqual(
            info["policy"], "development_tuning_holdout_validation"
        )
        self.assertEqual(info["policy_version"], 4)
        self.assertEqual(info["suites"]["pdsh"]["intended_use"], "tuning")
        self.assertEqual(info["suites"]["tpcds"]["intended_use"], "tuning")
        self.assertEqual(
            info["suites"]["tpcds"]["prior_exposure"], "used_for_development"
        )
        self.assertEqual(
            info["suites"]["clickbench"]["intended_use"], "validation_only"
        )
        self.assertEqual(
            info["suites"]["clickbench"]["prior_exposure"],
            "some_prior_optimization_use",
        )
        for suite in info["suites"].values():
            self.assertIsNone(suite["upstream_revision"])
        self.assertTrue(
            all(len(v) == 64 for v in info["source_sha256"].values())
        )

    def test_html_retains_every_query_and_counts_variants_separately(self):
        result = fixture()
        result["provenance"]["evaluation"] = bench_policy.record(
            ROOT, bench.SUITES, bench.SUITES
        )
        original = copy.deepcopy(result)
        cells = bench.evaluate(result)
        page = bench_html.render(
            result,
            cells,
            bench.SUITES,
            bench.VARIANTS,
            bench.instrumented_paths(),
        )
        for suite, spec in bench.SUITES.items():
            section = page.split(f'<section id="{suite}">', 1)[1].split(
                "</section>", 1
            )[0]
            self.assertEqual(
                re.findall(r'<td class="q">(q\d+)</td>', section),
                spec["queries"],
            )
        self.assertEqual(
            len(re.findall(r'<td class="q">q\d+</td>', page)), 179
        )
        self.assertIn("held-out validation", page)
        self.assertIn("TPC-DS until 2026-10-15", page)
        self.assertIn('<th class="num">Distinct queries</th>', page)
        self.assertIn('<th class="num">Query/variant cases</th>', page)
        self.assertRegex(
            page, r"group_by\.partitioned\.indexed.*one query.*\n?"
        )
        markdown = bench.report(result)
        self.assertIn(
            "`group_by.partitioned.indexed` **(one query)** | 1 | 5 |", markdown
        )
        self.assertEqual(result, original)

    def test_operator_reports_are_parsed_and_rendered(self):
        """`dataframe-operator:` lines map to one report per query, the last
        run's rows win, and both report formats list the joins that indexed
        more rows than probed them."""
        columns = "\t".join
        stderr = "\n".join(
            [
                "dataframe-query: q3",
                "dataframe-operator:\t" + columns(["0", "SCAN frame", "streaming", "", "", "0", "0", "10", "0", "1"]),
                "dataframe-operator:\t" + columns(["2", "JOIN inner on k = k", "streaming", "hash_index", "right", "10", "500", "7", "1", "1"]),
                "dataframe-operator:\t" + columns(["0", "SCAN frame", "streaming", "", "", "0", "0", "10", "0", "1"]),
                "dataframe-operator:\t" + columns(["2", "JOIN inner on k = k", "streaming", "hash_index", "right", "10", "600", "7", "1", "1"]),
                "dataframe-operator:\t" + columns(["4", "JOIN inner on j = j", "eager", "eager_hash", "left", "900", "20", "9", "1", "1"]),
                "dataframe-query: q5",
                "dataframe-path: join.hash_index",
            ]
        )
        reports = bench.operator_reports(stderr)
        self.assertEqual(sorted(reports), ["q3", "q5"])
        self.assertEqual([r["node"] for r in reports["q3"]], [0, 2, 4])
        self.assertEqual(reports["q3"][1]["build_rows"], 600)
        self.assertEqual(reports["q3"][1]["executor"], "streaming")
        self.assertEqual(reports["q5"], [])
        result = fixture()
        result["operators"] = {"pdsh/base/q3": reports["q3"]}
        markdown = bench.report(result)
        self.assertIn("## Join builds larger than their probe", markdown)
        self.assertIn("Of 2 joins that built an index", markdown)
        self.assertIn("| pdsh/base/q3 | JOIN inner on k = k | streaming | hash_index | 10 | 600 | 7 |", markdown)
        self.assertNotIn("JOIN inner on j = j", markdown)
        page = bench_html.render(
            result, bench.evaluate(result), bench.SUITES, bench.VARIANTS, bench.instrumented_paths()
        )
        self.assertIn('<section id="joins">', page)
        self.assertIn('<a href="#joins">joins</a>', page)
        self.assertIn("<td>JOIN inner on k = k</td>", page)

    def test_legacy_rerender_preserves_raw_samples_and_warns(self):
        result = fixture()
        # Include an actual timing cell so immutability covers timing samples.
        result["runs"][0].update(
            status="ok",
            times=[101, 102],
            summary={"height": 1, "values": [7], "names": ["v"]},
        )
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / "legacy.json"
            raw.write_text(json.dumps(result))
            before = raw.read_bytes()
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks/tools/bench_suites.py"),
                    "--report-from",
                    str(raw),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(raw.read_bytes(), before)
            for suffix in (".md", ".html"):
                report = raw.with_suffix(suffix).read_text()
                self.assertIn("Legacy result", report)
                self.assertIn("Raw measurements are unchanged", report)
                self.assertIn("held", report.lower())

    def test_instrumentation_includes_nested_and_other_operator_names(self):
        paths = bench.instrumented_paths()
        self.assertIn("group_by.partitioned.indexed", paths)
        self.assertIn("lazy.join_order", paths)
        self.assertIn("rank.one_sort", paths)

    def test_nested_library_changes_invalidate_runner_binaries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            nested = root / "dataframe/planning/model.mojo"
            nested.parent.mkdir(parents=True)
            nested.write_text("comptime SCAN_FRAME = 0\n")
            suites = root / "benchmarks/suites"
            suites.mkdir(parents=True)
            suite = suites / "example.mojo"
            suite.write_text("def main(): pass\n")
            binary = root / "build/suites/suite_example"
            binary.parent.mkdir(parents=True)
            binary.touch()
            os.utime(nested, (10, 10))
            os.utime(suite, (10, 10))
            os.utime(binary, (20, 20))
            with patch.object(bench, "ROOT", root), patch.object(
                bench, "SUITES_DIR", suites
            ), patch.object(bench, "_build") as build:
                self.assertEqual(
                    bench.build_runners(["example"]), {"example": binary}
                )
                build.assert_not_called()
                os.utime(nested, (30, 30))
                bench.build_runners(["example"])
                build.assert_called_once_with(root, "example", binary)

    def test_reference_cache_is_invalidated_by_query_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "table.parquet"
            data.write_bytes(b"fixture")
            source = root / "engines.py"
            source.write_text("query version 1")
            (root / "oracle_sql.py").write_text("oracle version 1")
            from types import SimpleNamespace

            args = SimpleNamespace(threads=8, reps=3)
            with patch.object(bench, "SUITES_DIR", root):
                old = bench._cache_key("duckdb", {"x": data}, args)
                source.write_text("query version 2")
                new = bench._cache_key("duckdb", {"x": data}, args)
            self.assertNotEqual(old, new)

    def test_reports_made_under_the_earlier_policy_keep_its_note(self):
        result = fixture()
        result["provenance"]["evaluation"] = bench_policy.record(
            ROOT, bench.SUITES, bench.SUITES
        )
        self.assertIn(
            "use ClickBench as held-out", bench_policy.report_note(result)
        )
        result["provenance"]["evaluation"]["policy_version"] = 3
        earlier = bench_policy.report_note(result)
        self.assertIn("use TPC-DS and ClickBench as held-out", earlier)
        result["provenance"]["evaluation"]["policy_version"] = 2
        earlier = bench_policy.report_note(result)
        self.assertIn("PDS-H and ClickBench as held-out", earlier)
        self.assertNotIn("TPC-DS", earlier)

    def test_a_crashed_worker_fails_one_query_and_the_rest_still_run(self):
        worker = (
            "import os, sys\n"
            "for q in sys.argv[1].split(','):\n"
            "    if q == 'q2':\n"
            "        sys.stderr.write('boom\\n')\n"
            "        os._exit(3)\n"
            "    print(f'time\\t{q}\\t1000')\n"
            "    print(f'summary\\t{q}\\t1\\t1.0\\tx')\n"
            "    sys.stdout.flush()\n"
        )
        queries = ["q1", "q2", "q3", "q4"]
        found, _ = bench.run_worker(
            [sys.executable, "-c", worker, ",".join(queries)],
            dict(os.environ),
            60,
            True,
            queries,
        )
        self.assertEqual(
            [found[q]["status"] for q in queries],
            ["ok", "failed", "ok", "ok"],
        )
        self.assertIn("worker exited 3", found["q2"]["reason"])
        self.assertIn("boom", found["q2"]["reason"])

    def test_heldout_flag_remains_supported_without_duplicate_suites(self):
        captured = {}

        def measure(args):
            captured["suites"] = args.suites.split(",")
            return fixture()

        with tempfile.TemporaryDirectory() as folder:
            output = str(Path(folder) / "result.json")
            with patch.object(bench, "measure", measure), patch.object(
                sys,
                "argv",
                ["bench", "--tier", "full", "--heldout", "--output", output],
            ), patch("sys.stdout"), patch("sys.stderr") as stderr:
                bench.main()
                self.assertFalse(
                    any(
                        "deprecated" in str(call)
                        for call in stderr.write.call_args_list
                    )
                )
        self.assertEqual(captured["suites"], list(bench.SUITES))


class CorrectnessOnly(unittest.TestCase):
    def setUp(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("complete answer fixtures require PyArrow")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.receipts = []
        for label, values in (("good", [2]), ("wrong", [2, 3])):
            folder = root / label
            folder.mkdir()
            path, schema = folder / "q1-0.parquet", folder / "q1-0.schema.json"
            pq.write_table(
                pa.table({"id1": ["a"] * len(values), "v1": values}), path
            )
            schema.write_text(
                json.dumps(
                    {
                        "format": "suite-answer-v1",
                        "query": "q1",
                        "repetition": 0,
                        "rows": len(values),
                        "columns": [
                            {"name": "id1", "arrow": "string"},
                            {"name": "v1", "arrow": "int64"},
                        ],
                        "oracle": {
                            "width": 2,
                            "order": [],
                            "limit": None,
                            "offset": 0,
                        },
                    }
                )
            )
            self.receipts.append(
                bench.suite_answers.pin_export(
                    {
                        "path": str(path),
                        "schema_path": str(schema),
                        "repetition": 0,
                    },
                    folder,
                    "q1",
                )
            )

    def result(self, status="ok", reference=True):
        result = fixture()
        result["provenance"]["reps"] = 0
        result["query_plan"] = [dict(suite="h2o_groupby", variant="base", queries=["q1"])]
        result["runs"] = [
            dict(
                suite="h2o_groupby",
                variant="base",
                query="q1",
                engine=engine,
                status=status if engine == "mojo" else "ok",
                summary=dict(height=1, values=[2.0], names=["value"]),
                answers=[self.receipts[0]],
                oracles=[self.receipts[0]] if engine == "duckdb" else [],
            )
            for engine in (["mojo", "duckdb"] if reference else ["mojo"])
        ]
        return result

    def test_untimed_answers_still_detect_wrong_missing_and_failed_results(
        self,
    ):
        with patch("sys.stdout"):
            self.assertFalse(bench.check_correctness(self.result()))
            for status in ("failed", "timeout"):
                self.assertTrue(bench.check_correctness(self.result(status)))
            self.assertFalse(
                bench.check_correctness(self.result("unsupported"))
            )
            self.assertTrue(
                bench.check_correctness(self.result(reference=False))
            )
            result = self.result()
            result["runs"][0]["answers"] = [self.receipts[1]]
            self.assertTrue(bench.check_correctness(result))
            result["runs"] = []
            self.assertTrue(bench.check_correctness(result))

    def test_cli_runs_once_without_reports_or_reference_cache(self):
        result = self.result()
        with patch.object(bench, "measure", return_value=result) as measure, patch.object(
            bench, "report"
        ) as report, patch.object(bench.bench_html, "render") as render, patch.object(
            sys, "argv", ["bench", "--correctness-only", "--engines", "mojo,duckdb"]
        ), patch("sys.stdout"), patch("sys.stderr"):
            bench.main()
        args = measure.call_args.args[0]
        self.assertEqual((args.rounds, args.reps), (1, 0))
        self.assertFalse(args.reuse_references)
        self.assertFalse(args.trace)
        self.assertEqual(args.output.name, "answers.json")
        report.assert_not_called()
        render.assert_not_called()

    def test_cli_wrong_answer_exits_nonzero(self):
        with patch.object(bench, "measure", return_value=self.result("failed")), patch.object(
            sys, "argv", ["bench", "--correctness-only"]
        ), patch("sys.stdout"), patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            bench.main()
        self.assertEqual(error.exception.code, 1)

    def test_timed_and_saved_report_commands_share_one_complete_validation(self):
        result = self.result()
        path = Path(self.directory.name) / "report.json"
        path.write_text(json.dumps(result))
        for saved in (False, True):
            argv = ["bench", "--check"] + (
                ["--report-from", str(path)] if saved
                else ["--output", str(path)]
            )
            with self.subTest(saved=saved), patch.object(
                bench, "measure", return_value=result
            ), patch.object(bench, "evaluate", wraps=bench.evaluate) as evaluate, patch.object(
                bench, "report", return_value="markdown"
            ) as report, patch.object(
                bench.bench_html, "render", return_value="<html>report</html>"
            ) as render, patch.object(sys, "argv", argv), patch("sys.stdout"):
                bench.main()
            self.assertEqual(evaluate.call_count, 1)
            cells = render.call_args.args[1]
            self.assertIs(report.call_args.args[1], cells)
            self.assertEqual(cells[("h2o_groupby", "base", "q1", "mojo")]["status"], "ok")

    def test_each_report_command_revalidates_tampered_answer_evidence(self):
        path = Path(self.directory.name) / "report.json"
        path.write_text(json.dumps(self.result()))
        with patch.object(
            bench, "evaluate", wraps=bench.evaluate
        ) as evaluate, patch.object(bench, "report", return_value="markdown"), patch.object(
            bench.bench_html, "render", return_value="<html>report</html>"
        ) as render, patch.object(
            sys, "argv", ["bench", "--report-from", str(path)]
        ), patch("sys.stdout"):
            bench.main()
            self.assertEqual(render.call_args.args[1][("h2o_groupby", "base", "q1", "mojo")]["status"], "ok")
            Path(self.receipts[0]["path"]).write_bytes(b"corrupt answer")
            bench.main()
        self.assertEqual(evaluate.call_count, 2)
        cell = render.call_args.args[1][("h2o_groupby", "base", "q1", "mojo")]
        self.assertEqual(cell["status"], "unverified")
        self.assertNotIn("ms", cell)

    def test_one_validation_keeps_failed_timed_workers_failing_the_command(self):
        path = Path(self.directory.name) / "failed.json"
        with patch.object(
            bench, "measure", return_value=self.result("failed")
        ), patch.object(bench, "evaluate", wraps=bench.evaluate) as evaluate, patch.object(
            bench, "report", return_value="markdown"
        ), patch.object(
            bench.bench_html, "render", return_value="<html>report</html>"
        ), patch.object(
            sys, "argv", ["bench", "--check", "--output", str(path)]
        ), patch("sys.stdout"), patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            bench.main()
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(evaluate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
