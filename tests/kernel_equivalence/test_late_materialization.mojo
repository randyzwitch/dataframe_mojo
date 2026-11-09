"""A small top-N over filters and projections on a table runs on its key
and filter columns with row ids, then fetches the rest of its rows (late
materialization); the rows match the plain eager evaluation."""
# contract-group: kernel.order_keys
from std.testing import assert_true
from dataframe import Column, DataFrame, Series, col, lit
from dataframe.storage.string import StringColumn

comptime ROWS = 300_000


def _lcg(mut state: UInt64) -> Int:
    state = state * 6364136223846793005 + 1442695040888963407
    return Int(state >> 33)


def _table() raises -> DataFrame:
    var state = UInt64(5)
    var k = List[Int64](capacity=ROWS)
    var valid = List[Bool](capacity=ROWS)
    var c = List[Int64](capacity=ROWS)
    var b = List[Float64](capacity=ROWS)
    var s = List[String](capacity=ROWS)
    var ids = List[Int64](capacity=ROWS)
    for i in range(ROWS):
        k.append(Int64(_lcg(state) % 5000))
        valid.append(i % 13 != 0)
        c.append(Int64(_lcg(state) % 100))
        b.append(Float64(_lcg(state) % 1000) / 8)
        s.append("row-" + String(i % 977))
        ids.append(Int64(i))
    return DataFrame(
        [
            Series("k", Column[Int64](k^, valid)),
            Series("c", Column[Int64](c^)),
            Series("b", Column[Float64](b^)),
            Series("s", StringColumn(s)),
            Series("id", Column[Int64](ids^)),
        ]
    )._with_storage_statistics()


def test_top_n_fetches_unread_columns_late() raises:
    var frame = _table()
    var limits = [1, 10, 50]
    var offsets = [0, 7, 1000]
    for descending in [False, True]:
        for nulls_last in [True, False]:
            for c in range(len(limits)):
                var limit = limits[c]
                var offset = offsets[c]
                # A filter, then a projection that computes a column the
                # order never reads.
                var got = (
                    frame.lazy()
                    .filter(col("c") < lit(Int64(60)))
                    .select_exprs(
                        [
                            col("k"),
                            col("s"),
                            (col("b") * 2).alias("b2"),
                            col("id"),
                        ]
                    )
                    .sort(
                        ["k"],
                        descending=[descending],
                        nulls_last=[nulls_last],
                        maintain_order=True,
                    )
                    .slice(offset, limit)
                    .collect()
                )
                var expected = (
                    frame.filter(col("c") < lit(Int64(60)))
                    .select_exprs(
                        [
                            col("k"),
                            col("s"),
                            (col("b") * 2).alias("b2"),
                            col("id"),
                        ]
                    )
                    .sort(
                        ["k"],
                        descending=[descending],
                        nulls_last=[nulls_last],
                        maintain_order=True,
                    )
                    .slice(offset, limit)
                )
                assert_true(
                    got.equals(expected),
                    "select: desc="
                    + String(descending)
                    + " nulls_last="
                    + String(nulls_last)
                    + " limit="
                    + String(limit),
                )


def test_computed_key_and_passthrough_columns() raises:
    var frame = _table()
    var got = (
        frame.lazy()
        .with_columns([(col("c") * 3 + col("id") % 11).alias("m")])
        .sort(["m", "id"], descending=[True, False], nulls_last=[True, True])
        .head(20)
        .collect()
    )
    var expected = (
        frame.with_columns([(col("c") * 3 + col("id") % 11).alias("m")])
        .sort(["m", "id"], descending=[True, False], nulls_last=[True, True])
        .head(20)
    )
    assert_true(got.equals(expected), "with_columns key")
    # Every column read by the order: nothing is deferred.
    var whole = (
        frame.lazy()
        .select(["k", "id"])
        .sort(["k", "id"], descending=[False, False], nulls_last=[True, True])
        .head(10)
        .collect()
    )
    var whole_expected = (
        frame.select(["k", "id"])
        .sort(["k", "id"], descending=[False, False], nulls_last=[True, True])
        .head(10)
    )
    assert_true(whole.equals(whole_expected), "all columns required")


def test_top_n_keeps_the_limit_when_ties_cross_workers() raises:
    # Every row ties on the best key across workers: the shared boundary
    # is that key, and the combine must still keep `limit` rows.
    var frame = _table().with_columns([(col("id") % 3).alias("few")])
    for limit in [100, 5000]:
        var got = (
            frame.lazy()
            .sort(["few"], descending=[False], nulls_last=[True])
            .head(limit)
            .collect()
        )
        var expected = frame.sort(
            ["few"], descending=[False], nulls_last=[True]
        ).head(limit)
        assert_true(got.height() == limit, "row count")
        assert_true(
            got.select(["few"]).equals(expected.select(["few"])), "key column"
        )
