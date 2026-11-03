"""Operator parity harness: cell matrices, data generators, the DuckDB and
Polars query builders, output parsers and the report, at tiny sizes."""

import collections
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "tools"))
import bench_operators as bench  # noqa: E402

import generate  # noqa: E402
import matrices  # noqa: E402
import operator_engines  # noqa: E402


def tiny(cell, rows=3_000):
    return bench.smoke(cell, rows)


class Matrices(unittest.TestCase):
    def test_sizes_and_identity(self):
        every = matrices.matrices()
        sizes = {name: len(cells) for name, cells in every.items()}
        self.assertEqual(
            sizes,
            {
                "join": 200,
                "agg": 150,
                "filter": 24,
                "sort": 16,
                "window": 104,
                "sched": 4,
            },
        )
        ids = [c["id"] for cells in every.values() for c in cells]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(json.dumps(every), json.dumps(matrices.matrices()))

    def test_latin_hypercube_balances_every_factor(self):
        cells = matrices.latin_hypercube(matrices.JOIN_FACTORS, 200, 1)
        for name, levels in matrices.JOIN_FACTORS.items():
            counts = collections.Counter(c[name] for c in cells)
            self.assertEqual(set(counts), set(levels), name)
            low = 200 // len(levels)
            self.assertTrue(
                all(low <= n <= low + 1 for n in counts.values()), name
            )

    def test_join_caps_probe_and_output(self):
        for cell in matrices.matrices()["join"]:
            spec = cell["dataset"]
            self.assertLessEqual(spec["probe_rows"], matrices.PROBE_CAP)
            dups = matrices.mean_dups(spec["dups"])
            self.assertLessEqual(
                spec["probe_rows"] * dups, matrices.OUTPUT_CAP + dups
            )


class Generators(unittest.TestCase):
    def test_bijections(self):
        import numpy as np

        ids = np.arange(100_000, dtype=np.uint64)
        for kind in ("int32", "int64", "date", "decimal", "str_short",
                     "str_long", "int_int", "int_str"):
            columns = generate.key_values(kind, ids, 99)
            rows = list(zip(*(c.to_pylist() for c in columns)))
            self.assertEqual(len(set(rows)), len(ids), kind)
        # Not a progression: neighbours in id are far apart in value.
        values = generate.key_values("int64", ids, 99)[0].to_numpy()
        self.assertGreater(np.abs(np.diff(values)).min(), 0)
        self.assertLess((np.diff(values) > 0).mean(), 0.6)

    def test_string_lengths(self):
        import numpy as np

        ids = np.arange(5_000, dtype=np.uint64)
        for fn, low, high in (
            (generate.str_short, 7, 12),
            (generate.str_long, 20, 40),
            (generate.str25, 25, 25),
        ):
            lengths = [len(s) for s in fn(ids, 3).to_pylist()]
            self.assertGreaterEqual(min(lengths), low)
            self.assertLessEqual(max(lengths), high)

    def test_datasets_are_deterministic_and_shaped(self):
        every = matrices.matrices()
        for name, cells in every.items():
            for cell in cells[:3]:
                spec = tiny(cell)["dataset"]
                first = generate.build_tables(spec)
                again = generate.build_tables(spec)
                for table in first:
                    self.assertTrue(first[table].equals(again[table]), name)
                if name == "join":
                    self.assertEqual(
                        first["build"].num_rows, spec["build_rows"]
                    )
                    self.assertEqual(
                        first["probe"].num_rows, spec["probe_rows"]
                    )

    def test_join_match_rate(self):
        import pyarrow.compute as pc

        cell = next(
            c
            for c in matrices.matrices()["join"]
            if c["params"]["match"] == 0.5
            and c["params"]["nulls"] == 0.0
            and c["params"]["key"] == "int64"
        )
        spec = dict(tiny(cell, 20_000)["dataset"])
        tables = generate.build_tables(spec)
        hit = pc.is_in(tables["probe"]["k0"], tables["build"]["bk0"])
        share = pc.mean(hit.cast("float64")).as_py()
        self.assertAlmostEqual(share, 0.5, delta=0.05)


class EnginesAgree(unittest.TestCase):
    """DuckDB SQL and Polars plans give the same answers on tiny inputs,
    through the harness's own comparison."""

    def test_one_cell_per_kind(self):
        import duckdb
        import polars as pl

        every = matrices.matrices()
        picks = [
            every["join"][i] for i in range(0, 200, 23)
        ] + [every["agg"][i] for i in range(0, 150, 19)] + [
            every["filter"][5],
            every["sort"][1],
            every["sort"][6],
        ] + [every["window"][i] for i in range(0, 52, 5)] + [
            every["sched"][0]
        ]
        for cell in picks:
            cell = tiny(cell, 2_000)
            tables = generate.build_tables(cell["dataset"])
            con = duckdb.connect()
            for name, table in tables.items():
                con.register(name, table)
            expected = con.sql(
                operator_engines.duckdb_sql(cell["query"])
            ).to_arrow_table()
            frames = {
                name: pl.from_arrow(table).lazy()
                for name, table in tables.items()
            }
            actual = (
                operator_engines.polars_plan(cell["query"], frames)
                .collect()
                .to_arrow()
            )
            with self.subTest(cell=cell["id"]):
                self.assertEqual(actual.num_columns, expected.num_columns)
                if cell["query"]["op"] == "sort":
                    with tempfile.TemporaryDirectory() as directory:
                        import pyarrow.parquet as pq

                        path = Path(directory) / "t.parquet"
                        pq.write_table(tables["t"], path)
                        relation, spec = bench.sort_oracle(
                            cell, {"t": str(path)}
                        )
                    self.assertEqual(relation.num_rows, 2_000)
                    self.assertEqual(spec["width"], actual.num_columns)
                    continue
                a, e, _ = bench._canonical_pair(actual, expected)
                import suite_answers

                result = suite_answers.compare_answers(
                    bench._answer(a),
                    bench._answer(e),
                    tolerant_columns=[
                        n
                        for n in bench.computed_floats(cell)
                        if str(e.schema.field(n).type) == "double"
                    ],
                )
                self.assertEqual(result["status"], "equal", result)


RUNNER_OUTPUT = """wall\t0\t2000000
op\t0\t0\tSCAN frame\tpipeline\t0\t0\t10\t0.0\t0.0
op\t0\t1\tJOIN inner on k0 = bk0\tpipeline\t10\t10\t7\t4.5\t0.0
op\t0\t2\tSELECT k0\tpipeline\t7\t0\t7\t0.1\t1.9
wall\t1\t1000000
op\t1\t0\tSCAN frame\tpipeline\t0\t0\t10\t0.0\t0.0
op\t1\t1\tJOIN inner on k0 = bk0\tpipeline\t10\t10\t7\t2.5\t0.0
op\t1\t2\tSELECT k0\tpipeline\t7\t0\t7\t0.1\t0.9
rows\t7
"""

POLARS_STDERR = """polars-rep 0
Streaming query took 35.66ms (247.22ms CPU), detailed breakdown:
equi-join: tot(246.70ms), poll(231.32ms, n=321, max=11.67ms, stolen=22.4%), update(15.38ms, n=8, max=15.28ms), recv(row=3000000, morsel=40, max=83334), sent(row=2000000, morsel=24, max=83334), io(total_active_time=0.00ns)
in-memory-source: tot(402.74µs), poll(401.82µs, n=24, max=105.70µs, stolen=45.8%), update(920.00ns, n=4, max=520.00ns), recv(row=0, morsel=0, max=0), sent(row=1000000, morsel=16, max=62500), io(total_active_time=0.00ns)
polars-rep end
"""


class Parsers(unittest.TestCase):
    def test_runner_lines(self):
        out = bench.parse_runner(RUNNER_OUTPUT)
        self.assertEqual(out["rows"], 7)
        self.assertEqual([r["wall_ms"] for r in out["runs"]], [2.0, 1.0])
        summary = bench.summarize("join", "mojo", dict(out, status="ok"))
        self.assertEqual(summary["op_busy_ms"], 3.5)
        self.assertEqual(summary["wall_ms"], 1.5)
        self.assertEqual(summary["operators"], ["JOIN inner on k0 = bk0"])

    def test_polars_metrics(self):
        runs = operator_engines.polars_nodes(POLARS_STDERR)
        self.assertEqual(list(runs), [0])
        self.assertAlmostEqual(runs[0][0]["busy_ms"], 246.70)
        self.assertAlmostEqual(runs[0][1]["busy_ms"], 0.40274)
        self.assertEqual(runs[0][0]["rows"], 2_000_000)

    def test_runner_arguments(self):
        cell = matrices.matrices()["agg"][0]
        args = bench.runner_args(cell["query"])
        self.assertIn("op=agg", args)
        aggs = next(a for a in args if a.startswith("aggs="))
        self.assertEqual(aggs.count(":"), 2 * len(cell["query"]["aggs"]))


class Report(unittest.TestCase):
    def test_flags_slow_and_failed_cells(self):
        def engine(wall, busy, status="ok"):
            return {"status": status, "wall_ms": wall, "op_busy_ms": busy}

        result = {
            "provenance": {"revision": "a" * 40, "engines": list(bench.ENGINES)},
            "cells": {
                "join-000": {
                    "matrix": "join",
                    "params": {"key": "int64"},
                    "engines": {
                        "mojo": engine(30.0, 100.0),
                        "duckdb": engine(10.0, 50.0),
                        "polars": engine(20.0, 60.0),
                    },
                    "answers": {
                        "mojo": {"status": "equal"},
                        "polars": {"status": "equal"},
                    },
                },
                "agg-000": {
                    "matrix": "agg",
                    "params": {"key": "int32_dense"},
                    "engines": {
                        "mojo": {"status": "failed", "reason": "boom"},
                        "duckdb": engine(10.0, 50.0),
                        "polars": engine(20.0, 60.0),
                    },
                    "answers": {},
                },
            },
        }
        self.assertEqual(bench.ratios(result["cells"]["join-000"]), (3.0, 2.0))
        text = bench.report(result)
        self.assertIn("## join", text)
        self.assertIn("## Worst cells", text)
        self.assertIn("**slow**", text)
        self.assertIn("**failed**", text)
        self.assertIn("1 of 2 cells", text)


if __name__ == "__main__":
    unittest.main()
