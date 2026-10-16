"""Scalar GPU mode has independently anchored typed, ordered winners."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_raises
from std.memory import bitcast
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    StructColumn,
    col,
    lit,
    when,
)
from dataframe.metal import MetalRuntime
from dataframe.decimal import pow10
from dataframe.accelerator.distributions import lower_scalar_modes
from tests.public_contracts.test_scalar_mode import (
    mode_sources,
    mode_expected,
    float_mode_source,
    assert_float_mode_bits,
)
from tests.support.coverage_compare import assert_frame, assert_value
from tests.support.coverage_generators import offset_layout, chunk_layout


# requires: metal
def test_mojo_gpu_scalar_mode_typed_winners_nulls_empty_and_scalar_siblings() raises:
    for source in mode_sources():
        var raw = DataFrame([source.copy()])
        var expected = mode_expected(source)
        for layout in range(3):
            var frame = (
                raw.copy() if layout
                == 0 else offset_layout(raw) if layout
                == 1 else chunk_layout(raw)
            )
            var query = frame.lazy().select(col("x").mode())
            var actual = query.collect(
                accelerator=MetalRuntime(), engine="accel"
            )
            assert_frame(actual, expected)
            var empty = (
                frame.clear()
                .lazy()
                .select(col("x").mode())
                .collect(accelerator=MetalRuntime(), engine="accel")
            )
            assert_equal(empty.height(), 0)
            assert_equal(empty.dtypes(), expected.dtypes())
            var nulls = (
                frame.tail(2)
                .lazy()
                .select(col("x").mode())
                .collect(accelerator=MetalRuntime(), engine="accel")
            )
            assert_equal(nulls.height(), 1)
            assert_true(nulls.item().is_null())
            var siblings = (
                frame.lazy()
                .select_exprs(
                    [
                        col("x").mode(),
                        lit(Int64(7)).alias("scalar"),
                        col("x").len().alias("rows"),
                    ]
                )
                .collect(accelerator=MetalRuntime(), engine="accel")
            )
            assert_equal(siblings.height(), 3)
            for row in range(3):
                assert_value(siblings.item(row, "x"), expected.item(row, "x"))
                assert_equal(siblings.item(row, "scalar").int64(), Int64(7))
                assert_equal(siblings.item(row, "rows").int64(), Int64(6))
            var empty_siblings = (
                frame.clear()
                .lazy()
                .select_exprs([col("x").mode(), lit(Int64(7)).alias("scalar")])
                .collect(accelerator=MetalRuntime(), engine="accel")
            )
            assert_equal(empty_siblings.height(), 0)


# requires: metal
def test_mojo_gpu_scalar_mode_float_equality_first_bits_and_budget() raises:
    for single in [False, True]:
        var source = float_mode_source(single)
        var expected = DataFrame([source.take([4, 0, 6, 2, 8])])
        for layout in range(3):
            var raw = DataFrame([source.copy()])
            var frame = (
                raw.copy() if layout
                == 0 else offset_layout(raw) if layout
                == 1 else chunk_layout(raw)
            )
            var query = frame.lazy().select(col("x").mode())
            var actual = query.collect(
                accelerator=MetalRuntime(), engine="accel"
            )
            assert_frame(actual, expected)
            assert_float_mode_bits(actual, single)
            if layout == 0:
                var measured = query.profile(accelerator=MetalRuntime())
                var peak = Int(
                    measured[1].item(0, "query_peak_requested_bytes").int64()
                )
                assert_true(peak > 0)
                assert_frame(
                    query.collect(
                        accelerator=MetalRuntime(memory_limit_bytes=peak),
                        engine="accel",
                    ),
                    expected,
                )
                with assert_raises(contains="budget"):
                    _ = query.collect(
                        accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
                        engine="accel",
                    )


# requires: metal
def test_mojo_gpu_scalar_mode_reports_remaining_alignment_and_literal_shape() raises:
    var frame = DataFrame([Series("x", Column[Int64]([3, 1, 3, 1]))])
    with assert_raises(contains="intermediate value_counts struct consumers"):
        _ = lower_scalar_modes(
            frame.lazy()
            .select(col("x").value_counts().alias("counts"))
            .select(col("counts"))
        )
    for empty in [False, True]:
        var input = frame.clear() if empty else frame.copy()
        var query = input.lazy().select(lit(Int64(7)).mode())
        with assert_raises(
            contains="Aggregate input must be row-valued without nested aggregates"
        ):
            _ = query.collect(engine="cpu", streaming=False)
        with assert_raises(
            contains="Aggregate input must be row-valued without nested aggregates"
        ):
            _ = query.collect(accelerator=MetalRuntime(), engine="accel")


# requires: metal
def test_mojo_gpu_multiple_modes_zip_exact_ordinals_and_cardinality() raises:
    var magnitude = Int128(1) << 90
    var valid: List[Bool] = [True, True, True, True, False, False]
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([3, 1, 3, 1, 9, 9], valid.copy())),
            Series(
                "s",
                StringColumn(
                    ["z", "a", "z", "a", "poison", "poison"], valid.copy()
                ),
            ),
            Series(
                "d",
                Column[Int128](
                    [
                        magnitude + 1,
                        -magnitude,
                        magnitude + 1,
                        -magnitude,
                        Int128.MIN,
                        0,
                    ],
                    valid.copy(),
                ),
            ).with_dtype(DataType.decimal(38, 2)),
        ]
    )
    var expected = DataFrame(
        [
            Series("a", Column[Int64]([1, 3, 0], [True, True, False])),
            Series("scalar", Column[Int64]([7, 7, 7])),
            Series("b", StringColumn(["a", "z", ""], [True, True, False])),
            Series(
                "c",
                Column[Int128](
                    [-magnitude, magnitude + 1, 0], [True, True, False]
                ),
            ).with_dtype(DataType.decimal(38, 2)),
            Series("rows", Column[Int64]([6, 6, 6])),
        ]
    )
    for layout in range(3):
        var input = (
            frame.copy() if layout
            == 0 else offset_layout(frame) if layout
            == 1 else chunk_layout(frame)
        )
        var expressions: List[Expr] = [
            col("x").mode().alias("a"),
            lit(Int64(7)).alias("scalar"),
            col("s").mode().alias("b"),
            col("d").mode().alias("c"),
            col("x").len().alias("rows"),
        ]
        var query = input.lazy().select_exprs(expressions)
        assert_frame(query.collect(engine="cpu", streaming=False), expected)
        assert_frame(
            query.collect(accelerator=MetalRuntime(), engine="accel"), expected
        )
        var empty = (
            input.clear()
            .lazy()
            .select_exprs(expressions)
            .collect(accelerator=MetalRuntime(), engine="accel")
        )
        assert_equal(empty.height(), 0)
        assert_equal(empty.dtypes(), expected.dtypes())
        var nulls = (
            input.tail(2)
            .lazy()
            .select_exprs(expressions)
            .collect(accelerator=MetalRuntime(), engine="accel")
        )
        assert_equal(nulls.height(), 1)
        for name in ["a", "b", "c"]:
            assert_true(nulls.item(0, name).is_null())
        assert_equal(nulls.item(0, "scalar").int64(), Int64(7))
        assert_equal(nulls.item(0, "rows").int64(), Int64(2))
        if layout == 0:
            var measured = query.profile(accelerator=MetalRuntime())
            var peak = Int(
                measured[1].item(0, "query_peak_requested_bytes").int64()
            )
            assert_frame(
                query.collect(
                    accelerator=MetalRuntime(memory_limit_bytes=peak),
                    engine="accel",
                ),
                expected,
            )
            with assert_raises(contains="budget"):
                _ = query.collect(
                    accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
                    engine="accel",
                )

    var left_cases: List[List[Int64]] = [[1, 1, 1, 1], [1, 2, 3, 4]]
    for left in left_cases:
        var mismatch = DataFrame(
            [
                Series("left", Column[Int64](left.copy())),
                Series("right", Column[Int64]([5, 5, 6, 6])),
            ]
        )
        var query = mismatch.lazy().select_exprs(
            [col("left").mode(), col("right").mode()]
        )
        with assert_raises(
            contains="explode: columns have different element counts at row 0"
        ):
            _ = query.collect(engine="cpu", streaming=False)
        with assert_raises(
            contains="explode: columns have different element counts at row 0"
        ):
            _ = query.collect(accelerator=MetalRuntime(), engine="accel")


def _counts_expected(
    source: Series, sorted: Bool, normalize: Bool
) raises -> DataFrame:
    var values = source.cast(
        DataType.STRING
    ) if source.dtype().is_categorical() else source.copy()
    var indices: List[Int] = [0, 1, 4] if sorted else [1, 0, 4]
    values = values.take(indices^).renamed("x")
    var counts: List[Int64] = [3, 2, 1] if sorted else [2, 3, 1]
    var tally: Series
    if normalize:
        var shares = List[Float64]()
        for count in counts:
            shares.append(Float64(count) / Float64(6))
        tally = Series("frequency", Column[Float64](shares^))
    else:
        var integers = List[UInt32]()
        for count in counts:
            integers.append(UInt32(count))
        tally = Series("frequency", Column[UInt32](integers^))
    return DataFrame([Series("counts", StructColumn([values^, tally^]))])


# requires: metal
def test_mojo_gpu_value_counts_typed_counts_sort_proportions_empty_and_null_categories() raises:
    var serial = 0
    for source in mode_sources():
        var raw = DataFrame([source.take([0, 1, 0, 1, 0, 4])])
        var frame = (
            raw.copy() if serial % 3
            == 0 else offset_layout(raw) if serial % 3
            == 1 else chunk_layout(raw)
        )
        serial += 1
        for sorted in [False, True]:
            for normalize in [False, True]:
                var expected = _counts_expected(source, sorted, normalize)
                var expression = (
                    col("x")
                    .value_counts(sorted, name="frequency", normalize=normalize)
                    .alias("counts")
                )
                var query = frame.lazy().select(expression)
                assert_frame(
                    query.collect(engine="cpu", streaming=False), expected
                )
                var actual = query.collect(
                    accelerator=MetalRuntime(), engine="accel"
                )
                assert_frame(actual, expected)
                for row in range(3):
                    assert_true(not actual.item(row, "counts").is_null())
                var empty = (
                    frame.clear()
                    .lazy()
                    .select(expression)
                    .collect(accelerator=MetalRuntime(), engine="accel")
                )
                assert_equal(empty.height(), 0)
                assert_equal(empty.dtypes(), expected.dtypes())
                var allnull = (
                    DataFrame([source.tail(2)])
                    .lazy()
                    .select(expression)
                    .collect(accelerator=MetalRuntime(), engine="accel")
                )
                assert_equal(allnull.height(), 1)
                var cell = allnull.item()
                assert_true(not cell.is_null())
                assert_true(cell.struct_field("x").is_null())
                if normalize:
                    assert_equal(
                        cell.struct_field("frequency").float64(), Float64(1)
                    )
                else:
                    assert_equal(
                        cell.struct_field("frequency").numeric[DType.uint32](),
                        UInt32(2),
                    )


# requires: metal
def test_mojo_gpu_mixed_mode_value_counts_zip_scalar_broadcast_and_mismatches() raises:
    var source = Series(
        "x",
        Column[Int64](
            [3, 1, 3, 1, 9, 9], [True, True, True, True, False, False]
        ),
    )
    var query = (
        DataFrame([source.copy()])
        .lazy()
        .select_exprs(
            [
                col("x").mode().alias("m"),
                col("x").value_counts(name="n").alias("counts"),
                lit(Int64(7)).alias("scalar"),
                col("x").value_counts(True, normalize=True).alias("shares"),
            ]
        )
    )
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    assert_equal(actual.height(), 3)
    assert_equal(actual.columns(), ["m", "counts", "scalar", "shares"])
    for row in range(3):
        var expected = source.get(1 if row == 0 else 0 if row == 1 else 4)
        assert_value(actual.item(row, "m"), expected)
        var counts = actual.item(row, "counts")
        assert_true(not counts.is_null())
        assert_value(counts.struct_field("x"), expected)
        assert_equal(
            counts.struct_field("n").numeric[DType.uint32](), UInt32(2)
        )
        assert_equal(actual.item(row, "scalar").int64(), Int64(7))
        var shares = actual.item(row, "shares")
        assert_true(not shares.is_null())
        assert_value(shares.struct_field("x"), expected)
        assert_equal(
            shares.struct_field("proportion").float64(), Float64(1) / Float64(3)
        )
    assert_frame(actual, query.collect(engine="cpu", streaming=False))
    var measured = query.profile(accelerator=MetalRuntime())
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    assert_frame(
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
        ),
        actual,
    )
    with assert_raises(contains="budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
        )
    var unequal = (
        DataFrame([source.take([0, 1, 0, 1, 0, 4])])
        .lazy()
        .select_exprs(
            [
                col("x").mode().alias("m"),
                col("x").value_counts().alias("counts"),
            ]
        )
    )
    with assert_raises(
        contains="explode: columns have different element counts at row 0"
    ):
        _ = unequal.collect(engine="cpu", streaming=False)
    with assert_raises(
        contains="explode: columns have different element counts at row 0"
    ):
        _ = unequal.collect(accelerator=MetalRuntime(), engine="accel")


# requires: metal
def test_mojo_gpu_value_counts_float_nan_zero_first_representations() raises:
    for single in [False, True]:
        var source = float_mode_source(single)
        var values = source.take([4, 0, 6, 2, 8])
        for sorted in [False, True]:
            for normalize in [False, True]:
                var tally = Series(
                    "count", Column[Float64]([0.2, 0.2, 0.2, 0.2, 0.2])
                ) if normalize else Series(
                    "count", Column[UInt32]([2, 2, 2, 2, 2])
                )
                var expected = DataFrame(
                    [Series("counts", StructColumn([values.copy(), tally^]))]
                )
                var query = (
                    DataFrame([source.copy()])
                    .lazy()
                    .select(
                        col("x")
                        .value_counts(sorted, name="count", normalize=normalize)
                        .alias("counts")
                    )
                )
                var actual = query.collect(
                    accelerator=MetalRuntime(), engine="accel"
                )
                assert_frame(actual, expected)
                var field = actual.column("counts").struct_column().field("x")
                assert_float_mode_bits(DataFrame([field^]), single)


# requires: metal
def test_mojo_gpu_float32_distributions_quiet_valid_signaling_nan_first_occurrence() raises:
    var signaling: List[UInt32] = [0x7F800042, 0xFF800073]
    var quiet: List[UInt32] = [0x7FC00042, 0xFFC00073]
    for sign in range(2):
        var values = List[Float32]()
        for word in List[UInt32](
            [signaling[sign], 0x7FC00099, 0x80000000, 0, 1, 0x80000001]
        ):
            values.append(bitcast[DType.float32](word))
        var frame = DataFrame([Series("x", Column[Float32](values^))])
        for accelerator in [False, True]:
            var mode_query = frame.lazy().select(col("x").mode())
            var mode = mode_query.collect(
                accelerator=MetalRuntime(), engine="accel"
            ) if accelerator else mode_query.collect(
                engine="cpu", streaming=False
            )
            assert_equal(mode.height(), 2)
            var mode_values = mode.column("x")
            assert_equal(
                bitcast[DType.uint32](
                    mode_values._data[Column[Float32]]._get(0)
                ),
                UInt32(0x80000000),
            )
            assert_equal(
                bitcast[DType.uint32](
                    mode_values._data[Column[Float32]]._get(1)
                ),
                quiet[sign],
            )
            var counts_query = frame.lazy().select(col("x").value_counts())
            var counts = counts_query.collect(
                accelerator=MetalRuntime(), engine="accel"
            ) if accelerator else counts_query.collect(
                engine="cpu", streaming=False
            )
            assert_equal(counts.height(), 4)
            var fields = counts.column("x").struct_column()
            var ordered = fields.field("x")
            var expected: List[UInt32] = [
                0x80000001,
                0x80000000,
                1,
                quiet[sign],
            ]
            for row in range(4):
                assert_equal(
                    bitcast[DType.uint32](
                        ordered._data[Column[Float32]]._get(row)
                    ),
                    expected[row],
                )
                assert_equal(
                    fields.field("count").get(row).numeric[DType.uint32](),
                    UInt32(2 if row == 1 or row == 3 else 1),
                )


# requires: metal
def test_mojo_gpu_distributions_observe_all_original_producers_before_cardinality() raises:
    var amount = Int128(6) * pow10(37)
    var frame = DataFrame(
        [
            Series("a", Column[Int64]([1, 1])),
            Series("b", Column[Int64]([1, 2])),
            Series("raw", Column[Int64]([128, 129])),
            Series("v", Column[Int128]([amount, amount])).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("z", Column[Int64]([Int64.MAX, 1])),
        ]
    )
    var cases: List[List[Expr]] = [
        [
            col("a").mode().alias("a"),
            col("b").mode().alias("b"),
            col("v").sum().alias("total"),
        ],
        [
            col("a").mode().alias("a"),
            col("v").sum().alias("total"),
            col("b").mode().alias("b"),
        ],
        [
            col("v").sum().alias("total"),
            col("raw").cast(DataType.INT8).mode().alias("bad"),
        ],
        [
            col("raw").cast(DataType.INT8).mode().alias("bad"),
            col("v").sum().alias("total"),
        ],
        [
            col("a").mode().alias("a"),
            col("b").mode().alias("b"),
            col("z").sum().alias("total"),
        ],
    ]
    var cpu_errors: List[String] = [
        "decimal precision",
        "decimal precision",
        "decimal precision",
        "range",
        "sum overflow",
    ]
    var gpu_errors: List[String] = [
        "decimal precision",
        "decimal precision",
        "decimal precision",
        "strict cast failed",
        "overflow",
    ]
    for case_index in range(len(cases)):
        var query = frame.lazy().select_exprs(cases[case_index])
        with assert_raises(contains=cpu_errors[case_index]):
            _ = query.collect(engine="cpu", streaming=False)
        with assert_raises(contains=gpu_errors[case_index]):
            _ = query.collect(accelerator=MetalRuntime(), engine="accel")

    # Reduction states are computed before scalar WHEN selects its branch.
    # A false condition cannot hide an overflowing Decimal SUM, regardless
    # of the scalar's original position relative to MODE.
    var runtime = MetalRuntime()
    var scalar = (
        when(lit(False))
        .then(col("v").sum())
        .otherwise(lit(Int64(0)).cast(DataType.decimal(38, 0)))
        .alias("scalar")
    )
    for scalar_first in [False, True]:
        var expressions = List[Expr]()
        if scalar_first:
            expressions.append(scalar.copy())
        expressions.append(col("a").mode())
        if not scalar_first:
            expressions.append(scalar.copy())
        var query = frame.lazy().select_exprs(expressions)
        with assert_raises(contains="decimal precision"):
            _ = query.collect(engine="cpu", streaming=False)
        with assert_raises(contains="decimal precision"):
            _ = query.collect(accelerator=runtime, engine="accel")

    # MODE input is a row expression: the invalid narrowing cast on row0
    # is unselected and must remain masked at the producer boundary.
    var mixed = DataFrame(
        [
            Series("raw", Column[UInt64]([UInt64.MAX, 3])),
            Series("use", Column[Bool]([False, True])),
        ]
    )
    var masked_mode = (
        when(col("use"))
        .then(col("raw").cast(DataType.INT8))
        .otherwise(lit(Int8(0)))
        .mode()
        .alias("mode")
    )
    var expected_modes = DataFrame([Series("mode", Column[Int8]([0, 3]))])
    for layout in range(3):
        var source = (
            mixed.copy() if layout
            == 0 else offset_layout(mixed) if layout
            == 1 else chunk_layout(mixed)
        )
        var query = source.lazy().select(masked_mode)
        assert_frame(
            query.collect(engine="cpu", streaming=False), expected_modes
        )
        assert_frame(
            query.collect(accelerator=runtime, engine="accel"), expected_modes
        )

    # The SUM state128 itself is valid. The unselected Int8 cast of that
    # state belongs to WHEN, and must not become an eagerly observed leaf.
    var safe = DataFrame(
        [
            Series("a", Column[Int64]([1, 1])),
            Series("z", Column[Int64]([64, 64])),
        ]
    )
    var projected = safe.lazy().select_exprs(
        [
            col("a").mode(),
            when(lit(False))
            .then(col("z").sum().cast(DataType.INT8))
            .otherwise(lit(Int8(0)))
            .alias("scalar"),
        ]
    )
    var expected = DataFrame(
        [
            Series("a", Column[Int64]([1])),
            Series("scalar", Column[Int8]([0])),
        ]
    )
    assert_frame(projected.collect(engine="cpu", streaming=False), expected)
    assert_frame(
        projected.collect(accelerator=runtime, engine="accel"), expected
    )
