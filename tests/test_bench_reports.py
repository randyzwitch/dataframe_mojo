"""Benchmark exposure, immutable rerendering, and complete query coverage."""

import copy
import json
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
    def test_catalog_and_full_tier_do_not_claim_independence(self):
        self.assertEqual(
            set(bench.TIERS["full"]["suites"].split(",")), set(bench.SUITES)
        )
        self.assertEqual(
            sum(len(s["queries"]) for s in bench.SUITES.values()), 80
        )
        self.assertTrue(all(s["role"] == "dev" for s in bench.SUITES.values()))
        info = bench_policy.record(ROOT, bench.SUITES, bench.SUITES)
        self.assertFalse(info["independent_validation"])
        for suite in info["suites"].values():
            self.assertEqual(suite["exposure"], "used_for_development")
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
        self.assertEqual(len(re.findall(r'<td class="q">q\d+</td>', page)), 80)
        self.assertIn("No independent validation is claimed", page)
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
                self.assertNotIn("Report-only", report)

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

    def test_old_cli_flag_warns_without_duplicating_suites(self):
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
                self.assertTrue(
                    any(
                        "deprecated" in str(call)
                        for call in stderr.write.call_args_list
                    )
                )
        self.assertEqual(captured["suites"], list(bench.SUITES))


if __name__ == "__main__":
    unittest.main()
