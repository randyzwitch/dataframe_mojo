"""The suite answer check (#606): infinities, and results that swap values
between rows."""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "tools"))
sys.path.insert(0, str(ROOT / "benchmarks" / "suites"))

import bench_suites  # noqa: E402


class CloseTest(unittest.TestCase):
    def test_infinities(self):
        inf = float("inf")
        self.assertTrue(bench_suites._close(inf, inf))
        self.assertTrue(bench_suites._close(-inf, -inf))
        self.assertFalse(bench_suites._close(1.0, inf))
        self.assertFalse(bench_suites._close(inf, 1.0))
        self.assertFalse(bench_suites._close(inf, -inf))
        self.assertFalse(bench_suites._close(1e308, inf))

    def test_nan_and_finite(self):
        nan = float("nan")
        self.assertTrue(bench_suites._close(nan, nan))
        self.assertFalse(bench_suites._close(nan, 1.0))
        self.assertFalse(bench_suites._close(1.0, nan))
        self.assertTrue(bench_suites._close(1.0, 1.0 + 1e-9))
        self.assertFalse(bench_suites._close(1.0, 1.001))
        self.assertTrue(bench_suites._close(0.0, 1e-9))


def _summary(height, values, pairs=None):
    out = {"height": height, "values": values, "names": []}
    if pairs is not None:
        out["pairs"] = pairs
    return out


class SameAnswerTest(unittest.TestCase):
    def test_equal_length_strings_collide(self):
        # The worker exports byte-length sums, not string contents. These
        # distinct results therefore have identical protocol records.
        def strings(values):
            return _summary(
                len(values), [float(sum(len(v.encode()) for v in values))]
            )

        self.assertTrue(
            bench_suites.same_answer(
                "x", "q", strings(["aa", "bb"]), strings(["cc", "dd"])
            )
        )

    def test_q17_accepts_arbitrary_same_height_values(self):
        self.assertTrue(
            bench_suites.same_answer(
                "clickbench",
                "q17",
                _summary(10, [123.0, 456.0, 789.0]),
                _summary(10, [-999.0, 0.0, 1e30]),
            )
        )

    def test_large_integer_precision_is_lost(self):
        self.assertNotEqual(2**53, 2**53 + 1)
        self.assertTrue(
            bench_suites.same_answer(
                "x",
                "q",
                _summary(1, [float(2**53)]),
                _summary(1, [float(2**53 + 1)]),
            )
        )

    def test_null_zero_and_column_name_changes_are_invisible(self):
        # [null, 1] and [0, 1] have the same height and sum.
        got = _summary(2, [1.0])
        want = dict(got, names=["different_name"])
        self.assertTrue(bench_suites.same_answer("x", "q", got, want))

    def test_adjacent_products_can_also_collide(self):
        # [(0, 0), (1, 1), (2, 2)] versus [(0, 1), (1, -1), (2, 3)]:
        # both column sums are 3 and both product sums are 5.
        self.assertTrue(
            bench_suites.same_answer(
                "x",
                "q",
                _summary(3, [3.0, 3.0], [5.0]),
                _summary(3, [3.0, 3.0], [5.0]),
            )
        )

    def test_swapped_rows_are_rejected(self):
        # [(1, 0), (2, 2)] against [(1, 2), (2, 0)]: the same column sums,
        # different products of neighbouring columns (1*0 + 2*2 = 4 against
        # 1*2 + 2*0 = 2).
        got = _summary(2, [3.0, 2.0], [4.0])
        want = _summary(2, [3.0, 2.0], [2.0])
        self.assertFalse(
            bench_suites.same_answer("x", "q", got, want)
        )
        self.assertTrue(bench_suites.same_answer("x", "q", got, got))

    def test_reference_without_pairs_still_compares_columns(self):
        got = _summary(2, [3.0, 2.0], [4.0])
        old = _summary(2, [3.0, 2.0])
        self.assertTrue(bench_suites.same_answer("x", "q", got, old))
        self.assertFalse(
            bench_suites.same_answer(
                "x", "q", got, _summary(2, [3.0, 3.0])
            )
        )

    def test_parse_reads_pairs(self):
        line = "summary\tq1\t2\t3.0,2.0\ta,b\t4.0"
        parsed = bench_suites.parse(line, ["q1"])["q1"]
        self.assertEqual(parsed["summary"]["pairs"], [4.0])
        legacy = "summary\tq1\t2\t3.0,2.0\ta,b"
        parsed = bench_suites.parse(legacy, ["q1"])["q1"]
        self.assertNotIn("pairs", parsed["summary"])


class EngineSummaryTest(unittest.TestCase):
    def test_pairs_keep_rows_together(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars is in the oracle environment")
        import engines

        first = pl.DataFrame({"g": [1, 2], "v": [0, 2]})
        second = pl.DataFrame({"g": [1, 2], "v": [2, 0]})
        height_a, values_a, pairs_a = engines.summary(first)
        height_b, values_b, pairs_b = engines.summary(second)
        self.assertEqual((height_a, values_a), (height_b, values_b))
        self.assertEqual(pairs_a, [4.0])
        self.assertEqual(pairs_b, [2.0])

    def test_nulls_count_as_zero_in_pairs(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars is in the oracle environment")
        import engines

        frame = pl.DataFrame({"a": [1, None, 3], "s": ["xy", "z", None]})
        _, values, pairs = engines.summary(frame)
        self.assertEqual(values, [4.0, 3.0])
        self.assertEqual(pairs, [2.0])  # 1*2 + 0*1 + 3*0


if __name__ == "__main__":
    unittest.main()
