"""Binding precedes evaluation; nested branches evaluate only selected rows."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.execution import _total_cast
from tests.support.coverage_generators import offset_layout, chunk_layout


def test_nested_and_chained_branch_masks_suppress_only_unselected_overflow() raises:
    var raw = DataFrame(
        [Series("x", Column[Int64]([Int64.MIN, Int64.MAX, -2, 2]))]
    )
    var expression = (
        when(col("x") == lit(Int64.MIN))
        .then(0)
        .when(col("x") > 0)
        .then(
            when(col("x") == lit(Int64.MAX))
            .then(col("x"))
            .otherwise(lit(Int64.MAX - 2) + col("x"))
        )
        .otherwise(-col("x"))
    )
    var scalar = (
        when(col("x") < lit(Int64.MIN))
        .then(lit(Int64.MAX) + 1)
        .otherwise(col("x"))
    )
    var quotient = (
        when(col("x") == lit(Int64.MIN)).then(0).otherwise(col("x") // -1)
    )
    var expected: List[List[Int64]] = [
        [0, Int64.MAX, 2, Int64.MAX],
        [Int64.MIN, Int64.MAX, -2, 2],
        [0, -Int64.MAX, 2, -2],
    ]
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        for batch in [1, 3, 64]:
            var actual = frame.select_exprs(
                [
                    expression.copy().alias("branch"),
                    scalar.copy().alias("scalar"),
                    quotient.copy().alias("quotient"),
                ],
                batch_size=batch,
            )
            assert_equal(actual.columns(), ["branch", "scalar", "quotient"])
            assert_equal(actual.height(), 4)
            for operation in range(3):
                var name = ["branch", "scalar", "quotient"][operation]
                assert_true(actual.column(name).dtype() == DataType.INT64)
                for row in range(4):
                    assert_equal(
                        actual.item(row, name).int64(), expected[operation][row]
                    )
        with assert_raises(contains="overflow"):
            _ = frame.select(
                when(col("x") > 0).then(lit(Int64.MAX) + col("x")).otherwise(0)
            )
        assert_equal(frame.clear().select(expression.copy()).height(), 0)


def test_binding_error_precedes_overflow_and_aggregate_conditions_keep_group_shapes() raises:
    var dangerous = DataFrame([Series("x", Column[Int64]([Int64.MAX, 1]))])
    for frame in [dangerous.copy(), dangerous.clear()]:
        with assert_raises(contains="Unknown expression column"):
            _ = frame.select_exprs(
                [col("x").sum().alias("overflow"), col("missing")]
            )
        with assert_raises(contains="Unknown expression column"):
            _ = frame.select(
                when(lit(False)).then(col("missing")).otherwise(col("x"))
            )
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([2, -1, -3, 1])),
            Series("g", Column[Int64]([1, 1, 0, 0])),
        ]
    )
    var selection = (
        when(col("x").sum() > 0)
        .then(col("x").max())
        .otherwise(col("x").min())
        .alias("selected")
    )
    var grouped = frame.group_by("g", maintain_order=True).agg(
        [selection.copy()]
    )
    assert_equal(grouped.columns(), ["g", "selected"])
    assert_equal(grouped.height(), 2)
    assert_true(grouped.column("selected").dtype() == DataType.INT64)
    for row in range(2):
        assert_equal(grouped.item(row, "g").int64(), Int64(1 - row))
        assert_equal(
            grouped.item(row, "selected").int64(), Int64(2 if row == 0 else -3)
        )
    var global_result = frame.select(selection.copy())
    assert_equal(global_result.shape(), (1, 1))
    assert_equal(global_result.item().int64(), Int64(-3))
    var broadcast = frame.with_columns(selection.copy())
    assert_equal(broadcast.height(), 4)
    for row in range(4):
        assert_equal(broadcast.item(row, "selected").int64(), Int64(-3))


def test_masked_means_under_a_scalar_condition_and_total_strict_casts() raises:
    # A scalar `when` choosing between reductions of null-otherwise
    # conditionals whose branches cast strictly: total casts take the
    # validity route, and plain, offset and chunked layouts agree. A strict
    # narrowing cast still never raises on rows its predicate drops, under
    # the taken scalar branch and the untaken one.
    assert_true(_total_cast(DataType.FLOAT64, DataType.FLOAT64))
    assert_true(_total_cast(DataType.INT64, DataType.FLOAT32))
    assert_true(_total_cast(DataType.INT8, DataType.INT64))
    assert_true(_total_cast(DataType.UINT32, DataType.INT64))
    assert_true(_total_cast(DataType.UINT16, DataType.UINT16))
    assert_true(not _total_cast(DataType.INT64, DataType.INT8))
    assert_true(not _total_cast(DataType.UINT64, DataType.INT64))
    assert_true(not _total_cast(DataType.INT16, DataType.UINT32))
    assert_true(not _total_cast(DataType.FLOAT64, DataType.INT64))
    var n = 300
    var quantity = List[Int64]()
    var quantity_valid = List[Bool]()
    var paid = List[Float64]()
    var paid_valid = List[Bool]()
    var wide = List[Int64]()
    for row in range(n):
        quantity.append(Int64(row % 100))
        quantity_valid.append(row % 17 != 4)
        paid.append(Float64(row) * 0.5)
        paid_valid.append(row % 11 != 6)
        wide.append(Int64(row % 100) * 1000)
    var raw = DataFrame(
        [
            Series("q", Column[Int64](quantity.copy(), quantity_valid.copy())),
            Series("p", Column[Float64](paid.copy(), paid_valid.copy())),
            Series("w", Column[Int64](wide^)),
        ]
    )
    # The mean of p over valid rows with 20 <= q <= 39, and of q itself.
    var p_total = Float64(0)
    var p_count = 0
    var q_total = Float64(0)
    var q_count = 0
    var banded = 0
    for row in range(n):
        if not quantity_valid[row] or quantity[row] < 20 or quantity[row] > 39:
            continue
        banded += 1
        q_total += Float64(quantity[row])
        q_count += 1
        if paid_valid[row]:
            p_total += paid[row]
            p_count += 1
    var band = col("q").is_between(lit(Int64(20)), lit(Int64(39)))
    var flag = when(band.copy()).then(lit(Int64(1))).otherwise(lit(Int64(0)))
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        for threshold in [Int64(0), Int64(1000)]:
            var taken = Int64(banded) > threshold
            var bucket = (
                when(flag.copy().sum() > lit(threshold))
                .then(
                    when(band.copy())
                    .then(col("p").cast(DataType.FLOAT64))
                    .end()
                    .mean()
                )
                .otherwise(
                    when(band.copy())
                    .then(col("q").cast(DataType.FLOAT64))
                    .end()
                    .mean()
                )
            )
            # Only rows with q < 30 hold w below 30,000: the strict Int16
            # cast overflows on every other row, which the predicate drops.
            var narrow = (
                when(flag.copy().sum() > lit(threshold))
                .then(
                    when(col("q") < lit(Int64(30)))
                    .then(col("w").cast(DataType.INT16))
                    .end()
                    .max()
                )
                .otherwise(
                    when(col("q") < lit(Int64(30)))
                    .then(col("w").cast(DataType.INT16))
                    .end()
                    .min()
                )
            )
            var actual = (
                frame.lazy()
                .select_exprs([bucket.alias("bucket"), narrow.alias("narrow")])
                .collect()
            )
            var expected = (p_total / Float64(p_count)) if taken else (
                q_total / Float64(q_count)
            )
            assert_true(
                abs(actual.item(0, "bucket").float64() - expected) < 1e-9
            )
            assert_true(actual.column("narrow").dtype() == DataType.INT16)
            assert_equal(
                Int(actual.column("narrow")._data[Column[Int16]]._get(0)),
                29000 if taken else 0,
            )
