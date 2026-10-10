"""Thread-owned pipelines give the eager result for row-local plans."""
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import Column, DataFrame, Expr, Series, StringColumn, col, lit
from dataframe.pipeline import _morsel_ranges


def table(n: Int, chunks: Int) raises -> DataFrame:
    var a = List[Int64](capacity=n)
    var a_valid = List[Bool](capacity=n)
    var b = List[Float64](capacity=n)
    var s = List[String](capacity=n)
    for i in range(n):
        a.append(Int64(i % 97))
        a_valid.append(i % 11 != 3)
        b.append(Float64(i % 13) * 0.5)
        s.append("row" + String(i % 7))
    var frame = DataFrame(
        [
            Series("a", Column[Int64](a^, a_valid^)),
            Series("b", Column[Float64](b^)),
            Series("s", StringColumn(s^)),
        ]
    )
    if chunks <= 1:
        return frame^
    # The same rows as `chunks` physical chunks per column, aligned.
    var columns = List[Series]()
    for c in range(frame.width()):
        var parts = List[Series]()
        for k in range(chunks):
            var start = n * k // chunks
            var stop = n * (k + 1) // chunks
            parts.append(frame._columns[c].slice(start, stop - start))
        columns.append(Series._from_chunks(parts^))
    return DataFrame(columns^)


def test_row_local_plans_match_eager_on_chunked_frames() raises:
    for chunks in [1, 3]:
        var t = table(20_011, chunks)
        var eager = (
            t.filter((col("a") > 40) & (col("b") < 5.0))
            .with_columns([(col("a") * 2).alias("a2")])
            .filter(col("a2") != 100)
            .select_exprs([col("s"), col("a2"), (col("b") + 1.0).alias("b1")])
        )
        var lazy = (
            t.lazy()
            .filter((col("a") > 40) & (col("b") < 5.0))
            .with_columns([(col("a") * 2).alias("a2")])
            .filter(col("a2") != 100)
            .select_exprs([col("s"), col("a2"), (col("b") + 1.0).alias("b1")])
            .collect(batch_size=1024)
        )
        assert_true(eager.equals(lazy))
        assert_equal(lazy.height(), eager.height())
        var dropped = (
            t.lazy().drop(["b"]).filter(col("a") == 5).collect(batch_size=1024)
        )
        assert_true(t.drop(["b"]).filter(col("a") == 5).equals(dropped))


def test_pipeline_reductions_and_empty_results() raises:
    var t = table(20_011, 4)
    var counted = (
        t.lazy()
        .filter(col("a") > 40)
        .select_exprs(
            [col("a").len().alias("n"), col("b").sum().alias("total")]
        )
        .collect(batch_size=1024)
    )
    var eager = t.filter(col("a") > 40)
    assert_equal(counted.height(), 1)
    assert_equal(counted.column("n").get(0).int64(), Int64(eager.height()))
    assert_equal(
        counted.column("total").get(0).float64(),
        eager.select_exprs([col("b").sum().alias("t")])
        .column("t")
        .get(0)
        .float64(),
    )
    var none = (
        t.lazy().filter(col("a") > 1000).select(["s"]).collect(batch_size=1024)
    )
    assert_equal(none.height(), 0)
    assert_equal(none.columns(), ["s"])
    var none_counted = (
        t.lazy()
        .filter(col("a") > 1000)
        .select_exprs([col("a").len().alias("n")])
        .collect(batch_size=1024)
    )
    assert_equal(none_counted.column("n").get(0).int64(), Int64(0))


def test_grouped_reductions_match_eager_in_both_orders() raises:
    # Few groups and many groups, plain and chunked frames, after a filter
    # and a with_columns: the thread-owned grouped sink gives the eager
    # group-by's result, in first-occurrence order when asked and as a
    # set otherwise. Integer sums, so association order cannot matter.
    for chunks in [1, 5]:
        var t = table(60_011, chunks)
        var few_keys: List[String] = ["s"]
        var many_keys: List[String] = ["a", "s"]
        for which in range(2):
            var keys = few_keys.copy() if which == 0 else many_keys.copy()
            var aggregates: List[Expr] = [
                col("a").sum().alias("total"),
                col("b").min().alias("low"),
                col("a").len().alias("n"),
            ]
            var eager = (
                t.filter(col("b") < 5.5)
                .with_columns([(col("a") * 3).alias("a")])
                .group_by(keys, maintain_order=True)
                .agg(aggregates)
            )
            var ordered = (
                t.lazy()
                .filter(col("b") < 5.5)
                .with_columns([(col("a") * 3).alias("a")])
                .group_by(keys, maintain_order=True)
                .agg(aggregates)
                .collect(batch_size=1024)
            )
            assert_true(eager.equals(ordered))
            var unordered = (
                t.lazy()
                .filter(col("b") < 5.5)
                .with_columns([(col("a") * 3).alias("a")])
                .group_by(keys)
                .agg(aggregates)
                .collect(batch_size=1024)
            )
            assert_equal(unordered.height(), eager.height())
            var sort_keys = keys.copy()
            assert_true(eager.sort(sort_keys).equals(unordered.sort(sort_keys)))
    # Nothing survives the filter: an empty result with the schema.
    var t = table(5_000, 2)
    var none = (
        t.lazy()
        .filter(col("a") > 1000)
        .group_by(["s"])
        .agg([col("a").sum().alias("total")])
        .collect(batch_size=1024)
    )
    assert_equal(none.height(), 0)
    assert_equal(none.columns(), ["s", "total"])


def test_morsel_ranges_never_cross_a_chunk() raises:
    var t = table(10_000, 3)
    var ranges = _morsel_ranges(t, 1024, 4)
    var ends = t._columns[0]._chunked.value()[].ends.copy()
    var covered = 0
    for k in range(len(ranges) // 2):
        var low = ranges[2 * k]
        var high = low + ranges[2 * k + 1]
        assert_equal(low, covered)
        covered = high
        assert_true(ranges[2 * k + 1] <= 1024 + 512 + 64)
        var inside = False
        var start = 0
        for end in ends:
            if low >= start and high <= end:
                inside = True
            start = end
        assert_true(inside)
    assert_equal(covered, 10_000)
    # Boundaries inside a chunk are 64-row aligned to the chunk's start.
    var chunk_starts = List[Int]()
    var at = 0
    for end in ends:
        chunk_starts.append(at)
        at = end
    for k in range(len(ranges) // 2):
        var low = ranges[2 * k]
        if low not in chunk_starts:
            var base = 0
            for c in chunk_starts:
                if c <= low:
                    base = c
            assert_equal((low - base) % 64, 0)
    # About two pieces of 2,048 (fewer than four a worker) are cut to four.
    var plain = _morsel_ranges(table(5_000, 1), 2048, 4)
    assert_equal(len(plain) // 2, 4)
    # A chunk a little longer than the target stays one morsel.
    var one = _morsel_ranges(table(2_900, 1), 2048, 4)
    assert_equal(len(one) // 2, 1)
    # Many pieces stay as they are.
    var many = _morsel_ranges(table(50_000, 1), 1000, 4)
    assert_equal(len(many) // 2, 50)
    var single = _morsel_ranges(table(100, 1), 2048, 4)
    assert_equal(len(single) // 2, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
