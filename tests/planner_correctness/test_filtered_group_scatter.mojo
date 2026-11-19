"""Filtered aggregates must not suppress ordinary states on scatter chunks."""
# contract-group: reduce.grouped
# contract-group: planner.routing
from std.testing import assert_true
from dataframe import Column, DataFrame, Expr, Series, col
from tests.support.group_strategy_reference import ScopedEnvironment


def _check(frame: DataFrame, outputs: List[Expr], expected: DataFrame) raises:
    assert_true(
        frame.group_by(["k", "j"], maintain_order=True)
        .agg(outputs)
        .equals(expected)
    )
    var query = (
        frame.lazy().group_by(["k", "j"], maintain_order=True).agg(outputs)
    )
    for streaming in [False, True]:
        for optimize in [False, True]:
            var actual = query.collect(
                engine="cpu",
                streaming=streaming,
                optimize=optimize,
                batch_size=127,
            )
            assert_true(
                actual.equals(expected),
                String(actual) + "\n!=\n" + String(expected),
            )


def test_grouped_first_last_mixed_with_null_skipping_reductions() raises:
    var valid: List[Bool] = [False, True, True, True, True, True, False, True]
    var frame = DataFrame(
        [
            Series(
                "k",
                Column[Int32](
                    [2, 1, 2, 1, 2, 1, 3, 3],
                    [True, True, True, True, True, True, False, False],
                ),
            ),
            Series(
                "j",
                Column[Int64](
                    [
                        1 << 60,
                        (1 << 60) + 1,
                        1 << 60,
                        (1 << 60) + 1,
                        1 << 60,
                        (1 << 60) + 1,
                        (1 << 60) + 2,
                        (1 << 60) + 2,
                    ]
                ),
            ),
            Series("n", Column[Int32]([9, 4, 4, -2, -2, 1, 8, 3], valid)),
            Series(
                "w",
                Column[Int64](
                    [
                        (1 << 60) + 9,
                        (1 << 60) + 4,
                        (1 << 60) + 4,
                        (1 << 60) - 2,
                        (1 << 60) - 2,
                        (1 << 60) + 1,
                        (1 << 60) + 8,
                        (1 << 60) + 3,
                    ],
                    valid,
                ),
            ),
            Series("f", Column[Float32]([9, 4, 4, -2, -2, 1, 8, 3], valid)),
            Series(
                "b",
                Column[Bool](
                    [True, False, True, False, True, False, True, True], valid
                ),
            ),
        ]
    )
    for chunked in [False, True]:
        if chunked:
            var pieces = List[Series]()
            for name in frame.columns():
                var value = frame.column(name)
                pieces.append(
                    Series._from_chunks(
                        [
                            value.slice(0, 1),
                            value.slice(1, 4),
                            value.slice(5, 3),
                        ]
                    )
                )
            frame = DataFrame(pieces^)
        for name in ["n", "w", "f", "b"]:
            var value = col(name)
            var source = frame.column(name)
            # Explicit positions in each group's original row order, including
            # null first rows. These expectations do not call an aggregate.
            var expected = DataFrame(
                [
                    frame.column("k").take([0, 1, 6]),
                    frame.column("j").take([0, 1, 6]),
                    source.take([0, 1, 6]).renamed("first"),
                    source.take([4, 5, 7]).renamed("last"),
                    source.take([2, 1, 7]).renamed("nonnull_first"),
                    source.take([4, 5, 7]).renamed("nonnull_last"),
                ]
            )
            _check(
                frame,
                [
                    value.first().alias("first"),
                    value.last().alias("last"),
                    value.first(ignore_nulls=True).alias("nonnull_first"),
                    value.last(ignore_nulls=True).alias("nonnull_last"),
                ],
                expected,
            )


def test_grouped_scatter_updates_counts_sums_and_extremes_beside_filters() raises:
    for rows in [8, 4097]:
        var keys = List[Int32]()
        var wide_keys = List[Int64]()
        var values = List[Int64]()
        var valid = List[Bool]()
        var counts = List[Int64](length=4, fill=0)
        var lengths = List[Int64](length=4, fill=0)
        var sums = List[Int64](length=4, fill=0)
        var filtered = List[Int64](length=4, fill=0)
        var minimum = List[Int64](length=4, fill=Int64.MAX)
        var maximum = List[Int64](length=4, fill=Int64.MIN)
        for row in range(rows):
            var group = row % 4
            var value = Int64(row % 19 - 9)
            var present = row % 5 != 0
            keys.append(Int32(group))
            wide_keys.append(Int64(1 << 60) + Int64(group))
            values.append(value)
            valid.append(present)
            lengths[group] += 1
            if present:
                counts[group] += 1
                sums[group] += value
                minimum[group] = min(minimum[group], value)
                maximum[group] = max(maximum[group], value)
                if value > 0:
                    filtered[group] += value
        var frame = DataFrame(
            [
                Series("k", Column[Int32](keys^)),
                Series("j", Column[Int64](wide_keys^)),
                Series("v", Column[Int64](values^, valid)),
            ]
        )
        var expected = DataFrame(
            [
                frame.column("k").take([0, 1, 2, 3]),
                frame.column("j").take([0, 1, 2, 3]),
                Series("count", Column[Int64](counts^)),
                Series("len", Column[Int64](lengths^)),
                Series("sum", Column[Int64](sums^)),
                Series("min", Column[Int64](minimum^)),
                Series("max", Column[Int64](maximum^)),
                Series("positive", Column[Int64](filtered^)),
            ]
        )
        var expressions: List[Expr] = [
            col("v").count().alias("count"),
            col("v").len().alias("len"),
            col("v").sum().alias("sum"),
            col("v").min().alias("min"),
            col("v").max().alias("max"),
            col("v").filter(col("v") > 0).sum().alias("positive"),
        ]
        for workers in [1, 3]:
            var threads = ScopedEnvironment(
                "DATAFRAME_THREADS", String(workers)
            )
            _check(frame, expressions, expected)
            _ = threads^
