"""Ordered windows preserve typed cells and restore every original row."""
# contract-group: expr.ordered_over
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, Expr, Series, col
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import chunk_layout, offset_layout


def test_ordered_over_scatter_ties_partitions_and_lazy_key_dependencies() raises:
    var frame = DataFrame(
        [
            Series(
                "g",
                Column[Int64](
                    [0, 1, 0, 1, 0, 1], [False, True, False, True, False, True]
                ),
            ),
            Series("t", Column[Int64]([2, 2, 1, 1, 2, 0])),
            Series("z", Column[Int64]([2, 0, 0, 0, 1, 0])),
            Series("x", Column[Int64]([10, 20, 30, 40, 50, 60])),
        ]
    )
    var expressions: List[Expr] = [
        col("x").cum_sum().over("g", order_by="t").alias("sum"),
        col("x").shift().over([String("g")], order_by="t").alias("previous"),
        col("x")
        .cum_sum()
        .over("g", order_by="t", descending=True)
        .alias("descending"),
        col("x").cum_sum().over(order_by="t").alias("global"),
        col("x").first().over("g", order_by="t").alias("first"),
        col("x")
        .cum_sum()
        .over("g", order_by=[String("t"), "z"])
        .alias("multi"),
    ]
    var expected = DataFrame(
        [
            Series("sum", Column[Int64]([40, 120, 30, 100, 90, 60])),
            Series(
                "previous",
                Column[Int64](
                    [30, 40, 0, 60, 10, 0],
                    [True, True, False, True, True, False],
                ),
            ),
            Series("descending", Column[Int64]([10, 20, 90, 60, 60, 120])),
            Series("global", Column[Int64]([140, 160, 90, 130, 210, 60])),
            Series("first", Column[Int64]([30, 60, 30, 60, 30, 60])),
            Series("multi", Column[Int64]([90, 120, 30, 100, 80, 60])),
        ]
    )
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        assert_frame(source.select_exprs(expressions, batch_size=2), expected)
        for optimize in [False, True]:
            for streaming in [False, True]:
                assert_frame(
                    source.lazy()
                    .select_exprs(expressions)
                    .collect(
                        engine="cpu",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=2,
                    ),
                    expected,
                )
        var empty = source.head(0)
        assert_frame(empty.select_exprs(expressions), expected.head(0))


def test_ordered_over_nan_null_placement_and_escaped_order_names() raises:
    var nan = Float64(0) / Float64(0)
    var order_name = String("t\x1f\x1e")
    var frame = DataFrame(
        [
            Series(
                order_name,
                Column[Float64](
                    [nan, 0, 2, 0, 2, 0], [True, True, True, False, True, True]
                ),
            ),
            Series("x", Column[Int64]([1, 2, 4, 8, 16, 32])),
        ]
    )
    var expressions: List[Expr] = [
        col("x").cum_sum().over(order_by=order_name).alias("first"),
        col("x")
        .cum_sum()
        .over(order_by=order_name, nulls_last=True)
        .alias("last"),
        col("x")
        .cum_sum()
        .over(order_by=order_name, descending=True, nulls_last=True)
        .alias("descending"),
    ]
    var expected = DataFrame(
        [
            Series("first", Column[Int64]([63, 10, 46, 8, 62, 42])),
            Series("last", Column[Int64]([55, 2, 38, 63, 54, 34])),
            Series("descending", Column[Int64]([1, 23, 5, 63, 21, 55])),
        ]
    )
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        assert_frame(source.select_exprs(expressions), expected)
        assert_frame(
            source.lazy()
            .select_exprs(expressions)
            .collect(engine="cpu", batch_size=1),
            expected,
        )
    with assert_raises(contains="Unknown window column"):
        _ = (
            frame.lazy()
            .select(col("x").sum().over(order_by="missing"))
            .collect(engine="cpu")
        )
    with assert_raises(contains="over requires"):
        _ = frame.select(col("x").sum().over())


def test_ordered_over_sorts_categorical_labels_instead_of_dictionary_codes() raises:
    var frame = DataFrame(
        [
            Series("t", Column[String](["z", "a", "z", "b", "a", "c"])).cast(
                "categorical"
            ),
            Series("x", Column[Int64]([1, 2, 4, 8, 16, 32])),
        ]
    )
    var expected = DataFrame(
        [Series("x", Column[Int64]([59, 2, 63, 26, 18, 58]))]
    )
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        var expr = col("x").cum_sum().over(order_by="t")
        assert_frame(source.select(expr), expected)
        assert_frame(
            source.lazy().select(expr).collect(engine="cpu", batch_size=2),
            expected,
        )


def test_shift_fill_only_replaces_vacated_partition_positions() raises:
    var frame = DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 0, 1, 0, 1])),
            Series("t", Column[Int64]([2, 2, 1, 1, 2, 0])),
            Series(
                "x",
                Column[Int64](
                    [10, 20, 30, 40, 50, 60],
                    [False, True, True, False, True, True],
                ),
            ),
        ]
    )
    var expressions: List[Expr] = [
        col("x").shift(1, fill_value=-1).over("g", order_by="t").alias("lag"),
        col("x").shift(-1, fill_value=-1).over("g", order_by="t").alias("lead"),
        col("x").shift(0, fill_value=-1).over("g", order_by="t").alias("same"),
        col("x")
        .shift(100, fill_value=-1)
        .over("g", order_by="t")
        .alias("outside"),
    ]
    var expected = DataFrame(
        [
            Series(
                "lag",
                Column[Int64](
                    [30, 0, -1, 60, 0, -1],
                    [True, False, True, True, False, True],
                ),
            ),
            Series(
                "lead",
                Column[Int64](
                    [50, -1, 0, 20, -1, 0],
                    [True, True, False, True, True, False],
                ),
            ),
            frame.column("x").renamed("same"),
            Series("outside", Column[Int64]([-1, -1, -1, -1, -1, -1])),
        ]
    )
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        assert_frame(source.select_exprs(expressions), expected)
        assert_frame(
            source.lazy()
            .select_exprs(expressions)
            .collect(engine="cpu", batch_size=1),
            expected,
        )
    var text = DataFrame(
        [Series("x", Column[String](["a", "", "c"], [True, False, True]))]
    )
    assert_frame(
        text.select(col("x").shift(fill_value="empty")),
        DataFrame(
            [
                Series(
                    "x", Column[String](["empty", "a", ""], [True, True, False])
                ),
            ]
        ),
    )


def test_global_multi_key_windows_and_empty_partition_names_keep_distinct_dependencies() raises:
    var frame = DataFrame(
        [
            Series("t", Column[Int64]([2, 1, 1, 2])),
            Series("r", Column[Int64]([0, 1, 2, 3])),
            Series("x", Column[Int64]([10, 20, 30, 40])),
            Series("", Column[Int64]([0, 1, 0, 1])),
        ]
    )
    var expressions: List[Expr] = [
        col("x").cum_sum().over(order_by=[String("t"), "r"]).alias("global"),
        col("x")
        .cum_sum()
        .over([String("")], order_by=[String("t")])
        .alias("partition"),
        col("x").sum().over("").alias("sum"),
        col("x").cum_sum().over(order_by=[String("")]).alias("empty_order"),
        col("x")
        .cum_sum()
        .over(order_by=[String("t"), "r"], descending=True)
        .alias("descending"),
    ]
    var expected = DataFrame(
        [
            Series("global", Column[Int64]([60, 20, 50, 100])),
            Series("partition", Column[Int64]([40, 20, 30, 60])),
            Series("sum", Column[Int64]([40, 60, 40, 60])),
            Series("empty_order", Column[Int64]([10, 60, 40, 100])),
            Series("descending", Column[Int64]([50, 100, 80, 40])),
        ]
    )
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        assert_frame(source.select_exprs(expressions), expected)
        for optimize in [False, True]:
            for streaming in [False, True]:
                assert_frame(
                    source.lazy()
                    .select_exprs(expressions)
                    .collect(
                        engine="cpu",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=1,
                    ),
                    expected,
                )


def test_ordered_over_composes_with_group_sequences_and_shift_defaults() raises:
    var frame = DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 0, 1, 0, 1])),
            Series("t", Column[Int64]([2, 2, 1, 1, 2, 0])),
            Series("x", Column[Int64]([10, 20, 30, 40, 50, 60])),
        ]
    )
    var expressions: List[Expr] = [
        col("x").head(2).sum().over("g", order_by="t").alias("head"),
        col("x").gather([0, -1]).sum().over("g", order_by="t").alias("gather"),
        col("x")
        .sort(descending=True)
        .first()
        .over("g", order_by="t")
        .alias("sort"),
        col("x")
        .filter(col("x") >= 20)
        .first()
        .over("g", order_by="t")
        .alias("filter"),
        col("x")
        .shift(fill_value=-1)
        .head(2)
        .sum()
        .over("g", order_by="t")
        .alias("shift"),
    ]
    var expected = DataFrame(
        [
            Series("head", Column[Int64]([40, 100, 40, 100, 40, 100])),
            Series("gather", Column[Int64]([80, 80, 80, 80, 80, 80])),
            Series("sort", Column[Int64]([50, 60, 50, 60, 50, 60])),
            Series("filter", Column[Int64]([30, 60, 30, 60, 30, 60])),
            Series("shift", Column[Int64]([29, 59, 29, 59, 29, 59])),
        ]
    )
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        assert_frame(source.select_exprs(expressions), expected)
        for optimize in [False, True]:
            for streaming in [False, True]:
                assert_frame(
                    source.lazy()
                    .select_exprs(expressions)
                    .collect(
                        engine="cpu",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=1,
                    ),
                    expected,
                )


def _rolling_reference(
    keys: List[Int64],
    key_valid: List[Bool],
    order: List[Int64],
    values: List[Int64],
    valid: List[Bool],
    width: Int,
    needed: Int,
    mean: Bool,
) raises -> Series:
    """rolling_sum / rolling_mean over (partition, order) by brute force:
    each partition's rows in order, every frame summed from scratch."""
    var n = len(keys)
    var by_order = List[Int](length=n, fill=0)
    for row in range(n):
        by_order[Int(order[row])] = row
    var parts = Dict[Int64, List[Int]]()
    for k in range(n):
        var row = by_order[k]
        # A null key is a partition of its own.
        var key = keys[row] if key_valid[row] else Int64.MIN
        if key not in parts:
            parts[key] = List[Int]()
        parts[key].append(row)
    var ints = List[Int64](length=n, fill=0)
    var floats = List[Float64](length=n, fill=0)
    var ok = List[Bool](length=n, fill=False)
    var at_least = width if needed < 0 else needed
    for entry in parts.items():
        ref rows = entry.value
        for k in range(len(rows)):
            var total = Int64(0)
            var count = 0
            for f in range(max(0, k - width + 1), k + 1):
                if valid[rows[f]]:
                    total += values[rows[f]]
                    count += 1
            var row = rows[k]
            ok[row] = count >= at_least and count > 0
            ints[row] = total if ok[row] else 0
            floats[row] = Float64(total) / Float64(count) if ok[row] else 0
    if mean:
        return Series("w", Column[Float64](floats^, ok))
    return Series("w", Column[Int64](ints^, ok))


def test_ordered_over_rolling_sums_partitions_of_varied_sizes_and_nulls() raises:
    # Rows in more than one 2048-row block, so partitions are evaluated by
    # several tasks. Partitions of varied sizes: one half of the rows, a
    # seventh, a key per thirteenth and a null key.
    var n = 5003
    var keys = List[Int64]()
    var key_valid = List[Bool]()
    var order = List[Int64]()
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(n):
        keys.append(
            0 if row % 2 == 0 else (1 if row % 7 == 1 else Int64(row % 13 + 2))
        )
        key_valid.append(row % 97 != 5)
        order.append(Int64((row * 7919) % n))
        values.append(Int64((row * 31) % 201 - 100))
    for null_values in [False, True]:
        valid.clear()
        for row in range(n):
            valid.append(not null_values or row % 11 != 3)
        var frame = DataFrame(
            [
                Series("g", Column[Int64](keys.copy(), key_valid)),
                Series("t", Column[Int64](order.copy())),
                Series("x", Column[Int64](values.copy(), valid)),
            ]
        )
        for width in [1, 11]:
            for needed in [-1, 2]:
                for mean in [False, True]:
                    var e = (
                        col("x")
                        .rolling_mean(width, needed) if mean else col("x")
                        .rolling_sum(width, needed)
                    )
                    var expressions: List[Expr] = [
                        e.over("g", order_by="t").alias("w")
                    ]
                    var expected = _rolling_reference(
                        keys,
                        key_valid,
                        order,
                        values,
                        valid,
                        width,
                        needed,
                        mean,
                    )
                    # Whole columns compared at once: a cell-by-cell
                    # comparison of every row dominates the test's time.
                    assert_true(
                        frame.select_exprs(expressions)
                        .column("w")
                        .equals(expected)
                    )
                    assert_true(
                        chunk_layout(frame)
                        .lazy()
                        .select_exprs(expressions)
                        .collect(engine="cpu")
                        .column("w")
                        .equals(expected)
                    )


def test_ordered_over_rolling_sum_overflow_raises_only_for_counted_frames() raises:
    var frame = DataFrame(
        [
            Series("g", Column[Int64]([0, 0, 1, 1])),
            Series("t", Column[Int64]([0, 1, 2, 3])),
            Series("x", Column[Int64]([Int64.MAX, 1, Int64.MIN, -1])),
        ]
    )
    with assert_raises():
        _ = frame.select_exprs(
            [col("x").rolling_sum(2).over("g", order_by="t").alias("w")]
        )
    with assert_raises():
        _ = (
            frame.lazy()
            .select_exprs(
                [col("x").rolling_sum(2).over("g", order_by="t").alias("w")]
            )
            .collect(engine="cpu")
        )
    # Frames under min_samples are null and never summed into a result.
    assert_frame(
        frame.select_exprs(
            [col("x").rolling_sum(2, 3).over("g", order_by="t").alias("w")]
        ),
        DataFrame(
            [
                Series(
                    "w",
                    Column[Int64]([0, 0, 0, 0], [False, False, False, False]),
                )
            ]
        ),
    )
    # A mean divides the exact total, so it does not overflow.
    var means = frame.select_exprs(
        [col("x").rolling_mean(2, 1).over("g", order_by="t").alias("w")]
    )
    assert_equal(means.item(1, "w").float64(), 4611686018427387904.0)
    assert_equal(means.item(3, "w").float64(), -4611686018427387904.0)
