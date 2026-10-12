"""Reduction accumulators and equality use the full logical decimal domain."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from tests.support.coverage_generators import offset_layout, chunk_layout


def _layout(frame: DataFrame, layout: Int) raises -> DataFrame:
    return (
        frame.copy() if layout
        == 0 else offset_layout(frame) if layout
        == 1 else chunk_layout(frame)
    )


def test_decimal_means_allow_exact_totals_beyond_input_precision() raises:
    # Precision three admits each +/-9.00 but not a total of 18.00. A mean
    # must divide a wide total, never constrain that total to input precision.
    var values: List[Int128] = [900, 900, -900, 900, 900, 900, 999]
    var groups: List[Int64] = [1, 0, 1, 0, 1, 0, 2]
    # Decimal128's precision-limited total is isolated in the six focused
    # executor-family cases below; do not hide passing narrow-width coverage.
    for width in [32, 64]:
        var dtype = DataType.decimal(3, 2, width)
        for mode in range(3):
            var valid = List[Bool](length=7, fill=mode != 2)
            valid[6] = False
            if mode == 1:
                valid[2] = False
            var source = (
                Series("x", Column[Int128](values.copy(), valid.copy()))
                .with_dtype(DataType.decimal(3, 2))
                .cast(dtype)
            )
            var raw = DataFrame(
                [source^, Series("g", Column[Int64](groups.copy()))]
            )
            var total = Int128(0)
            var count = 0
            for row in range(len(values)):
                if valid[row]:
                    total += values[row]
                    count += 1
            if count > 0:
                assert_true(total > 999)
            for layout in range(3):
                var frame = _layout(raw, layout)
                for execution in range(5):
                    var actual = frame.select(
                        col("x").mean().alias("m")
                    ) if execution == 0 else frame.lazy().select(
                        col("x").mean().alias("m")
                    ).collect(
                        optimize=execution > 2,
                        streaming=execution % 2 == 0,
                        batch_size=2,
                        engine="cpu",
                    )
                    assert_equal(actual.shape(), (1, 1))
                    assert_equal(actual.columns(), ["m"])
                    assert_true(actual.column("m").dtype() == DataType.FLOAT64)
                    assert_equal(actual.item().is_null(), count == 0)
                    if count:
                        assert_equal(
                            actual.item().float64(),
                            Float64(total) / Float64(count) / 100,
                        )
                    var grouped = frame.group_by("g", maintain_order=True).agg(
                        [col("x").mean().alias("m")]
                    ) if execution == 0 else frame.lazy().group_by(
                        ["g"], maintain_order=True
                    ).agg(
                        [col("x").mean().alias("m")]
                    ).collect(
                        optimize=execution > 2,
                        streaming=execution % 2 == 0,
                        batch_size=2,
                        engine="cpu",
                    )
                    assert_equal(grouped.columns(), ["g", "m"])
                    assert_equal(grouped.height(), 3)
                    assert_true(grouped.column("g").dtype() == DataType.INT64)
                    assert_true(grouped.column("m").dtype() == DataType.FLOAT64)
                    for output in range(3):
                        var group = [1, 0, 2][output]
                        total = 0
                        count = 0
                        for row in range(len(values)):
                            if groups[row] == Int64(group) and valid[row]:
                                total += values[row]
                                count += 1
                        assert_equal(
                            grouped.item(output, "g").int64(), Int64(group)
                        )
                        assert_equal(
                            grouped.item(output, "m").is_null(), count == 0
                        )
                        if count:
                            assert_equal(
                                grouped.item(output, "m").float64(),
                                Float64(total) / Float64(count) / 100,
                            )
                    # Recompute global state for the next execution variant.
                    total = 0
                    count = 0
                    for row in range(len(values)):
                        if valid[row]:
                            total += values[row]
                            count += 1
                var empty = frame.clear().select(col("x").mean())
                assert_equal(empty.shape(), (1, 1))
                assert_true(empty.column("x").dtype() == DataType.FLOAT64)
                assert_true(empty.item().is_null())


def test_decimal_distinct_counts_compare_full_int128_values() raises:
    # n_unique accepts any logical type; a Decimal128 value is not limited
    # to Int64. Independent equality over scaled integers counts null once.
    var huge = Int128(Int64.MAX) * 1000
    var values: List[Int128] = [huge, huge, huge + 1, -huge, 5, 0]
    var valid: List[Bool] = [True, True, True, True, True, False]
    var expected = 1
    for row in range(len(values)):
        if not valid[row]:
            continue
        var seen = False
        for previous in range(row):
            seen = seen or (valid[previous] and values[previous] == values[row])
        if not seen:
            expected += 1
    assert_equal(expected, 5)
    var frame = DataFrame(
        [
            Series("x", Column[Int128](values^, valid^)).with_dtype(
                DataType.decimal(38, 2)
            )
        ]
    )
    var actual = frame.select(col("x").n_unique().alias("u"))
    assert_equal(actual.shape(), (1, 1))
    assert_true(actual.column("u").dtype() == DataType.INT64)
    assert_equal(actual.item().int64(), Int64(expected))


def _wide_mean(grouped: Bool, lazy: Bool, streaming: Bool) raises:
    var values: List[Int128] = [900, 900, 900, 900, 999]
    var valid: List[Bool] = [True, True, True, True, False]
    var raw = DataFrame(
        [
            Series("x", Column[Int128](values.copy(), valid.copy())).with_dtype(
                DataType.decimal(3, 2)
            ),
            Series("g", Column[Int64]([1, 0, 1, 0, 2])),
        ]
    )
    var total = Int128(0)
    for row in range(5):
        if valid[row]:
            total += values[row]
    assert_equal(total, Int128(3600))
    assert_true(total > 999)
    var expected = Float64(total) / 4 / 100
    for layout in range(3):
        var frame = _layout(raw, layout)
        for optimize in [False, True]:
            if not grouped:
                var actual = (
                    frame.lazy()
                    .select(col("x").mean().alias("m"))
                    .collect(
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=2,
                        engine="cpu",
                    ) if lazy else frame.select(col("x").mean().alias("m"))
                )
                assert_equal(actual.shape(), (1, 1))
                assert_true(actual.column("m").dtype() == DataType.FLOAT64)
                assert_equal(actual.item().float64(), expected)
            else:
                var actual = (
                    frame.lazy()
                    .group_by(["g"], maintain_order=True)
                    .agg([col("x").mean().alias("m")])
                    .collect(
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=2,
                        engine="cpu",
                    ) if lazy else frame.group_by("g", maintain_order=True)
                    .agg([col("x").mean().alias("m")])
                )
                assert_equal(actual.columns(), ["g", "m"])
                assert_equal(actual.height(), 3)
                assert_true(actual.column("m").dtype() == DataType.FLOAT64)
                for row in range(3):
                    assert_equal(
                        actual.item(row, "g").int64(), Int64([1, 0, 2][row])
                    )
                    assert_equal(actual.item(row, "m").is_null(), row == 2)
                    if row != 2:
                        assert_equal(actual.item(row, "m").float64(), expected)


def test_decimal128_eager_global_mean_uses_wide_total() raises:
    _wide_mean(False, False, False)


def test_decimal128_eager_grouped_mean_uses_wide_total() raises:
    _wide_mean(True, False, False)


def test_decimal128_lazy_materialized_global_mean_uses_wide_total() raises:
    _wide_mean(False, True, False)


def test_decimal128_lazy_materialized_grouped_mean_uses_wide_total() raises:
    _wide_mean(True, True, False)


def test_decimal128_lazy_streaming_global_mean_uses_wide_total() raises:
    _wide_mean(False, True, True)


def test_decimal128_lazy_streaming_grouped_mean_uses_wide_total() raises:
    _wide_mean(True, True, True)


def test_decimal128_sum_checks_final_precision_and_ignores_null_payloads() raises:
    var limit = Int128(1)
    for _ in range(38):
        limit *= 10
    var big = limit // 10 * 6
    var examples: List[List[Int128]] = [[big, big], [-big, -big], [big, big]]
    for example in range(len(examples)):
        var values = examples[example].copy()
        var valid = List[Bool](length=len(values), fill=True)
        if example == 2:
            valid[1] = False
        var total = Int128(0)
        for row in range(len(values)):
            if valid[row]:
                total += values[row]
        var overflow = total <= -limit or total >= limit
        assert_equal(overflow, example == 0 or example == 1)
        var raw = DataFrame(
            [
                Series("x", Column[Int128](values^, valid^)).with_dtype(
                    DataType.decimal(38, 0)
                ),
                Series(
                    "g",
                    Column[Int64](
                        List[Int64](length=len(examples[example]), fill=0)
                    ),
                ),
            ]
        )
        for layout in range(3):
            var frame = _layout(raw, layout)
            for grouped in [False, True]:
                for execution in range(5):
                    var plan = (
                        frame.lazy()
                        .group_by(["g"])
                        .agg(
                            [col("x").sum().alias("s")]
                        ) if grouped else frame.lazy()
                        .select(col("x").sum().alias("s"))
                    )
                    if overflow:
                        with assert_raises(
                            contains="exceeds decimal precision"
                        ):
                            if execution == 0:
                                _ = frame.group_by("g").agg(
                                    [col("x").sum().alias("s")]
                                ) if grouped else frame.select(
                                    col("x").sum().alias("s")
                                )
                            else:
                                _ = plan.collect(
                                    optimize=execution > 2,
                                    streaming=execution % 2 == 0,
                                    batch_size=2,
                                    engine="cpu",
                                )
                    else:
                        var actual: DataFrame
                        if execution == 0:
                            actual = frame.group_by("g").agg(
                                [col("x").sum().alias("s")]
                            ) if grouped else frame.select(
                                col("x").sum().alias("s")
                            )
                        else:
                            actual = plan.collect(
                                optimize=execution > 2,
                                streaming=execution % 2 == 0,
                                batch_size=2,
                                engine="cpu",
                            )
                        assert_equal(actual.height(), 1)
                        assert_true(
                            actual.column("s").dtype()
                            == DataType.decimal(38, 0)
                        )
                        assert_equal(actual.column("s")._decimal_raw(0), total)
