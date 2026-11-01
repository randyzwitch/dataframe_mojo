"""Complete-answer contracts independent of either dataframe engine."""
import copy
from decimal import Decimal
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/tools"))
sys.path.insert(0, str(ROOT / "tests/tooling"))
import suite_answers as answers
import bench_suites as bench
import suite_schema
from suite_answer_fixture import rows_answer as answer


class CompleteComparisonTests(unittest.TestCase):
    def test_rejects_summary_collisions(self):
        cases = [
            (
                answer([("aa",), ("bb",)], types=("string",)),
                answer([("cc",), ("dd",)], types=("string",)),
            ),
            (answer([(2**53,)]), answer([(2**53 + 1,)])),
            (answer([(None,), (1,)]), answer([(0,), (1,)])),
            (
                answer(
                    [(0, 0), (1, 1), (2, 2)], ("a", "b"), ("int64", "int64")
                ),
                answer(
                    [(0, 1), (1, -1), (2, 3)], ("a", "b"), ("int64", "int64")
                ),
            ),
        ]
        for actual, expected in cases:
            with self.subTest(actual=actual.table.to_pylist()):
                self.assertEqual(
                    answers.compare_answers(actual, expected)["status"],
                    "different",
                )

    def test_preserves_multiplicity(self):
        actual, expected = answer([(1,), (1,), (2,)]), answer(
            [(2,), (1,), (1,)]
        )
        self.assertEqual(
            answers.compare_answers(actual, expected)["status"], "equal"
        )
        self.assertEqual(
            answers.compare_answers(actual, answer([(1,), (2,), (2,)]))[
                "status"
            ],
            "different",
        )

    def test_schema_checked_even_for_empty_results(self):
        for other in (answer([], ("b",)), answer([], types=("uint64",))):
            self.assertEqual(
                answers.compare_answers(answer([]), other)["reason"],
                "schema_mismatch",
            )

    def test_decimals_and_integer_fields_stay_exact(self):
        a = answer(
            [(Decimal("9007199254740992.01"),)], types=("decimal128(20, 2)",)
        )
        b = answer([(Decimal("9007199254740992.02"),)], types=a.types)
        self.assertEqual(answers.compare_answers(a, b)["status"], "different")
        with self.assertRaises(ValueError):
            answers.compare_answers(
                answer([(1,)]), answer([(1,)]), tolerant_columns=["a"]
            )

    def test_only_declared_float_fields_have_tolerance(self):
        a, b = answer([(1.0,)], types=("double",)), answer(
            [(1.0 + 1e-10,)], types=("double",)
        )
        self.assertEqual(answers.compare_answers(a, b)["status"], "different")
        self.assertEqual(
            answers.compare_answers(a, b, tolerant_columns=["a"])["status"],
            "equal",
        )
        for left, right, status in [
            (float("nan"), float("nan"), "equal"),
            (float("inf"), float("-inf"), "different"),
            (None, 0.0, "different"),
        ]:
            self.assertEqual(
                answers.compare_answers(
                    answer([(left,)], types=("double",)),
                    answer([(right,)], types=("double",)),
                    tolerant_columns=["a"],
                )["status"],
                status,
            )

    def test_integer_sum_transport_equivalence_remains_exact(self):
        native = answer(
            [("a", 2**53 + 1)], ("id1", "v1"), ("string", "int64")
        )
        reference = answer(
            [("a", Decimal(2**53 + 1))],
            ("id1", "v1"),
            ("string", "decimal128(38, 0)"),
        )
        native = suite_schema.normalize(native, "h2o_groupby", "q1")
        reference = suite_schema.normalize(reference, "h2o_groupby", "q1")
        self.assertEqual(
            answers.compare_answers(native, reference)["status"], "equal"
        )
        reference = suite_schema.normalize(
            answer(
                [("a", Decimal(2**53))],
                ("id1", "v1"),
                ("string", "decimal128(38, 0)"),
            ),
            "h2o_groupby",
            "q1",
        )
        self.assertEqual(
            answers.compare_answers(native, reference)["status"], "different"
        )
        # A decimal source field elsewhere cannot acquire integer equivalence.
        untouched = suite_schema.normalize(
            answer([(Decimal(1),)], types=("decimal128(38, 0)",)), "pdsh", "q2"
        )
        self.assertEqual(untouched.types, ["decimal128(38, 0)"])

    def test_only_reviewed_schema_aliases_are_accepted(self):
        original = answer([(1,)], ("count_star()",))
        self.assertEqual(
            suite_schema.normalize(original, "clickbench", "q0").names,
            ["count"],
        )
        self.assertEqual(
            suite_schema.normalize(original, "clickbench", "q19").names,
            ["count_star()"],
        )


try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None


@unittest.skipIf(pa is None, "PyArrow is required for answer export tests")
class ExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def export(self, values, dtype=None, name="a", repetition=0):
        table = pa.table({name: pa.array(values, type=dtype)})
        path = self.root / f"q1-{repetition}.parquet"
        schema_path = self.root / f"q1-{repetition}.schema.json"
        pq.write_table(table, path)
        schema_path.write_text(
            json.dumps(
                {
                    "format": answers.FORMAT,
                    "query": "q1",
                    "repetition": repetition,
                    "rows": len(values),
                    "columns": [
                        {"name": name, "arrow": str(table.schema[0].type)}
                    ],
                    "oracle": {
                        "width": 1,
                        "order": [],
                        "limit": None,
                        "offset": 0,
                    },
                }
            )
        )
        return answers.pin_export(
            {
                "path": str(path),
                "schema_path": str(schema_path),
                "repetition": repetition,
            },
            self.root,
            "q1",
        )

    def test_round_trip_keeps_integer_and_nanosecond_precision(self):
        receipt = self.export([2**53 + 1], pa.int64())
        self.assertEqual(
            answers.load_answer(receipt).table.column(0).to_pylist(),
            [2**53 + 1],
        )
        receipt = self.export([1234567890123456789], pa.timestamp("ns"))
        self.assertEqual(
            answers.load_answer(receipt).table.column(0).to_pylist(),
            [1234567890123456789],
        )

    def test_modification_or_foreign_paths_fail_closed(self):
        receipt = self.export([1])
        Path(receipt["schema_path"]).write_text("{}")
        with self.assertRaises(ValueError):
            answers.load_answer(receipt)
        with self.assertRaises(ValueError):
            answers.pin_export(receipt, self.root / "other", "q1")

    def test_summary_only_runs_are_unverified(self):
        runs = [
            {
                "suite": "h2o_groupby",
                "variant": "k100",
                "query": "q1",
                "engine": e,
                "status": "ok",
                "times": [10],
                "summary": {"height": 1, "values": [1.0]},
            }
            for e in ("duckdb", "mojo")
        ]
        cells = bench.evaluate({"runs": runs})
        self.assertTrue(
            all(c["status"] == "unverified" for c in cells.values())
        )

    def test_all_rounds_and_repetitions_are_checked(self):
        receipt = self.export([1])
        base = {
            "suite": "fixture",
            "variant": "k100",
            "query": "q1",
            "status": "ok",
            "times": [10],
            "round": 0,
            "answers": [receipt],
        }
        runs = [
            dict(base, engine="duckdb", oracles=[receipt]),
            dict(base, engine="mojo"),
        ]
        self.assertEqual(
            bench.evaluate({"runs": runs})[("fixture", "k100", "q1", "mojo")][
                "status"
            ],
            "ok",
        )
        bad = dict(
            base,
            engine="mojo",
            round=1,
            status="failed",
            reason="later round failed",
        )
        self.assertEqual(
            bench.evaluate({"runs": runs + [bad]})[
                ("fixture", "k100", "q1", "mojo")
            ]["status"],
            "failed",
        )
        # A second timed repetition without its answer cannot qualify timing.
        runs[1] = dict(runs[1], times=[10, 20])
        self.assertEqual(
            bench.evaluate({"runs": runs})[("fixture", "k100", "q1", "mojo")][
                "status"
            ],
            "unverified",
        )

    def test_wrong_later_answer_cannot_hide_behind_first_success(self):
        first = self.export([1])
        second = self.export([999], repetition=1)
        base = {
            "suite": "fixture",
            "variant": "base",
            "query": "q1",
            "status": "ok",
            "round": 0,
        }
        result = {
            "runs": [
                dict(base, engine="duckdb", answers=[first], oracles=[first]),
                dict(
                    base, engine="mojo", times=[10, 20], answers=[first, second]
                ),
            ]
        }
        cell = bench.evaluate(result)[("fixture", "base", "q1", "mojo")]
        self.assertEqual(cell["status"], "wrong answer")
        self.assertNotIn("ms", cell)

    def test_reused_verdict_still_checks_each_receipt(self):
        # Repetition 1 has the same Parquet bytes as repetition 0, so its
        # comparison is reused, but its own files are still verified.
        oracle = self.export([1, 2])
        first = self.export([1, 2])
        second = self.export([1, 2], repetition=1)
        self.assertEqual(first["path_sha256"], second["path_sha256"])
        base = {
            "suite": "fixture",
            "variant": "base",
            "query": "q1",
            "status": "ok",
            "round": 0,
            "times": [10, 20],
        }
        result = {
            "runs": [
                dict(
                    base,
                    engine="duckdb",
                    answers=[first, second],
                    oracles=[oracle],
                ),
                dict(base, engine="mojo", answers=[first, second]),
            ]
        }
        key = ("fixture", "base", "q1", "mojo")
        self.assertEqual(bench.evaluate(result)[key]["status"], "ok")
        Path(second["schema_path"]).write_text("{}")
        self.assertEqual(bench.evaluate(result)[key]["status"], "unverified")

    def test_missing_round_is_not_a_successful_measurement(self):
        first = self.export([1])
        run = {
            "suite": "fixture",
            "variant": "base",
            "query": "q1",
            "status": "ok",
            "round": 0,
            "engine": "duckdb",
            "answers": [first],
            "oracles": [first],
        }
        cell = bench.evaluate({"provenance": {"rounds": 2}, "runs": [run]})[
            ("fixture", "base", "q1", "duckdb")
        ]
        self.assertEqual(cell["status"], "unverified")


if __name__ == "__main__":
    unittest.main()
