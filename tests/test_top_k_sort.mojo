"""Sorts cut short by a slice (#332): `sort(...).head(k)` and
`sort(...).slice(offset, k)` select their rows instead of sorting every
row, lazily (a TOP_K plan node, streamed batch by batch) and eagerly
(`top_k`, `bottom_k`). Each result equals the full sort's rows, including
stable ties, both null placements, NaN, and long strings whose ties are
settled by comparing the strings.
"""
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from dataframe import Column, DataFrame, Series, col, lit
from dataframe.packed_sort import packed_top_rows


struct Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self, bound: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(bound))


def frame(rows: Int, seed: UInt64) raises -> DataFrame:
    var rng = Lcg(seed)
    var nan = Float64(0) / Float64(0)
    var ints = List[Int64]()
    var int_valid = List[Bool]()
    var floats = List[Float64]()
    var float_valid = List[Bool]()
    var words = List[String]()
    var word_valid = List[Bool]()
    var urls = List[String]()
    for i in range(rows):
        # Few distinct values, so heads cut through runs of ties.
        ints.append(Int64(rng.next(20)))
        int_valid.append(rng.next(10) != 0)
        var pick = rng.next(12)
        floats.append(
            nan if pick
            == 0 else (-0.0 if pick == 1 else Float64(rng.next(500)) / 3 - 80)
        )
        float_valid.append(rng.next(9) != 0)
        words.append("w" + String(rng.next(40)))
        word_valid.append(rng.next(8) != 0)
        # Strings longer than a key holds and sharing most bytes, so the
        # boundary of a head falls inside runs that need settling.
        urls.append(
            "https://example.org/a/b/c/"
            + String(rng.next(4))
            + "/"
            + "x" * rng.next(12)
            + String(rng.next(30))
        )
        _ = i
    return DataFrame(
        [
            Series("i", Column[Int64](ints^, int_valid^)),
            Series("f", Column[Float64](floats^, float_valid^)),
            Series("w", Column[String](words^, word_valid^)),
            Series("u", Column[String](urls^)),
        ]
    ).with_row_index()


def check_heads(
    data: DataFrame,
    by: List[String],
    descending: List[Bool],
    nulls_last: List[Bool],
) raises:
    var full = data.sort(by, descending=descending, nulls_last=nulls_last)
    for k in [0, 1, 7, 50, 333, data.height() + 5]:
        var expected = full.slice(0, k)
        # Eager selection of the head of the order.
        var order = data._arg_sort_head(by, descending, nulls_last, k)
        assert_true(data.take(order).equals(expected), String(k))
        # Lazy, streamed in small batches and materialized.
        for streaming in [True, False]:
            var lazy = (
                data.lazy()
                .sort(by, descending=descending, nulls_last=nulls_last)
                .head(k)
                .collect(streaming=streaming, batch_size=997)
            )
            assert_true(lazy.equals(expected), String(k))
        # A filter below the sort keeps the plan streaming, so each batch
        # selects its own first rows on a worker.
        var kept = data.filter(col("index") % lit(Int64(3)) != lit(Int64(0)))
        var filtered = kept.sort(
            by, descending=descending, nulls_last=nulls_last
        ).slice(0, k)
        var streamed = (
            data.lazy()
            .filter(col("index") % lit(Int64(3)) != lit(Int64(0)))
            .sort(by, descending=descending, nulls_last=nulls_last)
            .head(k)
            .collect(batch_size=997)
        )
        assert_true(streamed.equals(filtered), String(k))
        # An offset: the sort keeps offset + k rows, the slice drops some.
        var sliced = (
            data.lazy()
            .sort(by, descending=descending, nulls_last=nulls_last)
            .slice(3, k)
            .collect(batch_size=997)
        )
        assert_true(sliced.equals(full.slice(3, k)), String(k))


def test_heads_match_the_full_sort() raises:
    var data = frame(20_000, 3)
    var keys: List[List[String]] = [
        ["i"],
        ["f"],
        ["w"],
        ["u"],
        ["i", "f"],
        ["w", "i", "u"],
    ]
    for by in keys:
        var n = len(by)
        for pattern in range(4):
            var descending = List[Bool]()
            var nulls_last = List[Bool]()
            for k in range(n):
                descending.append(((pattern + k) & 1) == 1)
                nulls_last.append(((pattern >> 1) & 1) == 1)
            check_heads(data, by, descending, nulls_last)


def test_selection_is_used_and_settles_long_string_ties() raises:
    var data = frame(50_000, 9)
    var packed = packed_top_rows([data.column("u")], [False], [True], 25)
    assert_true(Bool(packed))
    var expected = data.arg_sort(["u"])
    expected.shrink(25)
    assert_equal(packed.value(), expected)
    # Keys that do not pack still select through dense ranks.
    var wide = DataFrame(
        [
            Series("a", Column[Int64]([Int64.MIN, Int64.MAX, 0, 5, -5])),
            Series("b", Column[UInt64]([UInt64.MAX, 0, UInt64(1) << 63, 3, 3])),
        ]
    )
    assert_false(
        Bool(
            packed_top_rows(
                [wide.column("a"), wide.column("b")],
                [False, False],
                [True, True],
                2,
            )
        )
    )
    assert_equal(
        wide._arg_sort_head(["a", "b"], [False, False], [True, True], 2),
        [0, 4],
    )


def test_explain_shows_top_k() raises:
    var data = frame(1_000, 5)
    var plan = data.lazy().sort(["f"]).head(10).explain()
    assert_true("TOP_K 10 by f" in plan, plan)
    # A filter between them still moves below, and the head still fuses.
    plan = (
        data.lazy()
        .sort(["f"], descending=True)
        .filter(col("i") > lit(Int64(3)))
        .slice(5, 10)
        .explain()
    )
    assert_true("TOP_K 15 by f" in plan, plan)
    # A sort with no slice above stays a sort.
    plan = data.lazy().sort(["f"]).explain()
    assert_true("SORT f" in plan and "TOP_K" not in plan, plan)


def test_top_and_bottom_k() raises:
    var data = frame(30_000, 13)
    for k in [0, 3, 100, 40_000]:
        var top = data.top_k(k, ["f", "i"])
        var expected = data.sort(
            ["f", "i"], descending=[True, True], nulls_last=[True, True]
        ).slice(0, k)
        assert_true(top.equals(expected), String(k))
        var bottom = data.bottom_k(k, "w")
        assert_true(bottom.equals(data.sort(["w"]).slice(0, k)), String(k))


def test_large_inputs_select_per_chunk() raises:
    # Enough rows for several 65,536-row chunks, each selecting on its own
    # worker before the candidates are selected again.
    var data = frame(300_000, 21)
    var keys: List[List[String]] = [["f"], ["u"], ["w", "i"]]
    for by in keys:
        for d in [False, True]:
            var n = len(by)
            var descending = List[Bool](length=n, fill=d)
            var nulls_last = List[Bool](length=n, fill=not d)
            var full = data.arg_sort(
                by, descending=descending, nulls_last=nulls_last
            )
            for k in [1, 40, 5000]:
                var expected = full.copy()
                expected.shrink(k)
                assert_equal(
                    data._arg_sort_head(by, descending, nulls_last, k),
                    expected,
                    String(k),
                )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
