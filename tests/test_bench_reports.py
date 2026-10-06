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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
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
            {"tpcds", "clickbench"},
        )
        # PDS-H is development since 2026-10-05, and the quick tier runs it.
        self.assertEqual(bench.SUITES["pdsh"]["role"], "dev")
        self.assertIn("pdsh", bench.TIERS["quick"]["suites"].split(","))
        self.assertNotIn("tpcds", bench.TIERS["quick"]["suites"].split(","))
        info = bench_policy.record(ROOT, bench.SUITES, bench.SUITES)
        self.assertEqual(
            info["policy"], "development_tuning_holdout_validation"
        )
        self.assertEqual(info["policy_version"], 3)
        self.assertEqual(info["suites"]["pdsh"]["intended_use"], "tuning")
        self.assertEqual(
            info["suites"]["tpcds"]["intended_use"], "validation_only"
        )
        self.assertEqual(
            info["suites"]["tpcds"]["prior_exposure"], "none_recorded"
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
        self.assertIn("TPC-DS replaced it", page)
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
                    str(ROOT / "scripts/bench_suites.py"),
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

    def test_reference_cache_is_invalidated_by_query_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "table.parquet"
            data.write_bytes(b"fixture")
            source = root / "engines.py"
            source.write_text("query version 1")
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
        self.assertIn("TPC-DS and ClickBench", bench_policy.report_note(result))
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


if __name__ == "__main__":
    unittest.main()
