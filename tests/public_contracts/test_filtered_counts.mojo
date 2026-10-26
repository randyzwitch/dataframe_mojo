"""Counts after a filter equal the rows the filter keeps: a column without
nulls counts every kept row, one with nulls only its kept valid rows, and a
column also summed still gives its kept values."""
# contract-group: table.filter
from std.testing import assert_equal
from dataframe import Column, DataFrame, Expr, LazyFrame, Series, col
from tests.support.coverage_generators import chunk_layout


def _frame(n: Int) raises -> DataFrame:
    var keys = List[Int32]()
    var ints = List[Int64]()
    var floats = List[Float64]()
    var floats_valid = List[Bool]()
    var texts = List[String]()
    for row in range(n):
        keys.append(Int32((row * 37) % 100))
        ints.append(Int64(row % 1000))
        floats.append(Float64(row) / 4)
        floats_valid.append(row % 3 != 1)
        texts.append(String(row % 7))
    return DataFrame(
        [
            Series("k", Column[Int32](keys^)),
            Series("x", Column[Int64](ints^)),
            Series("y", Column[Float64](floats^, floats_valid)),
            Series("s", Column[String](texts^)),
        ]
    )


def _expect(plan: LazyFrame, want: List[Int64]) raises:
    for optimize in [True, False]:
        for streaming in [True, False]:
            for batch_size in [65536, 1000]:
                var out = plan.collect(
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=batch_size,
                )
                assert_equal(out.height(), 1)
                var names = out.columns()
                for c in range(len(want)):
                    assert_equal(
                        out.item(0, names[c]).int64(), want[c], names[c]
                    )


def test_filtered_counts_match_the_kept_rows() raises:
    for n in [0, 1, 999, 70_001, 200_000]:
        var frame = _frame(n)
        for chunked in [False, True]:
            var source = chunk_layout(frame) if chunked else frame.copy()
            for limit in [0, 30, 100]:
                var kept = Int64(0)
                var kept_valid = Int64(0)
                var total = Int64(0)
                for row in range(n):
                    if Int((row * 37) % 100) < limit:
                        kept += 1
                        total += Int64(row % 1000)
                        if row % 3 != 1:
                            kept_valid += 1
                var exprs: List[Expr] = [
                    col("x").count().alias("x"),
                    col("y").count().alias("y"),
                    col("s").count().alias("s"),
                    col("x").len().alias("rows"),
                ]
                var filtered = source.lazy().filter(col("k") < limit)
                _expect(
                    filtered.select_exprs(exprs),
                    [kept, kept_valid, kept, kept],
                )
                # A counted column the same reduction also sums.
                var both: List[Expr] = [
                    col("x").sum().alias("sum"),
                    col("x").count().alias("count"),
                ]
                _expect(filtered.select_exprs(both), [total, kept])
            # The predicate reads the counted column itself.
            var below = Int64(0)
            for row in range(n):
                if row % 1000 < 500:
                    below += 1
            var own = source.lazy().filter(col("x") < 500)
            _expect(own.select_exprs([col("x").count()]), [below])
