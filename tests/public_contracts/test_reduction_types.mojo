"""Public reduction semantics across numeric and extended logical types."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from tests.support.reduction_types_reference import (
    typed_reduction_columns,
    reduction_validity,
    typed_scalar_reductions,
    assert_typed_reductions,
    near,
)
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.coverage_compare import assert_float
from tests.support.numeric_model import FLOAT_DTYPES


def test_global_reductions_match_scalar_models_across_logical_types() raises:
    for mode in range(3):
        var valid = reduction_validity(mode)
        for original in typed_reduction_columns(valid):
            # Categorical expressions decode to String; focused cases below
            # check that policy instead of asserting physical storage identity.
            if original.dtype().is_categorical():
                continue
            var raw = DataFrame([original.copy()])
            for layout in range(3):
                var frame = (
                    raw.copy() if layout
                    == 0 else offset_layout(raw) if layout
                    == 1 else chunk_layout(raw)
                )
                var actual = frame.select_exprs(
                    typed_scalar_reductions(original.dtype())
                )
                assert_equal(actual.height(), 1)
                var names = List[String]()
                for expression in typed_scalar_reductions(original.dtype()):
                    names.append(expression._name)
                assert_equal(actual.columns(), names)
                assert_typed_reductions(
                    actual, 0, original, valid, [0, 1, 2, 3, 4, 5]
                )
                var empty = frame.clear().select_exprs(
                    typed_scalar_reductions(original.dtype())
                )
                assert_equal(empty.height(), 1)
                assert_typed_reductions(empty, 0, original, valid, [])


def test_global_float_extrema_and_statistics_follow_nonfinite_policy() raises:
    comptime for i in range(len(FLOAT_DTYPES)):
        comptime D = FLOAT_DTYPES[i]
        var nan = Float64(0) / Float64(0)
        var inf = Float64(1) / Float64(0)
        var examples: List[List[Float64]] = [
            [-0.0, 0.0, nan, 3],
            [0.0, -0.0, nan, 3],
            [nan, nan, nan, nan],
            [-inf, inf, 0.0, nan],
            [inf, 1, inf, 2],
            [-inf, -1, -inf, -2],
        ]
        var minima: List[Float64] = [-0.0, 0.0, nan, -inf, 1, -inf]
        var maxima: List[Float64] = [nan, nan, nan, nan, inf, -1]
        var argmin: List[UInt32] = [0, 0, 0, 0, 1, 0]
        var argmax: List[UInt32] = [3, 3, 0, 1, 0, 1]
        var unique: List[Int64] = [3, 3, 1, 4, 3, 3]
        for example in range(len(examples)):
            var values = List[Scalar[D]]()
            for value in examples[example]:
                values.append(value.cast[D]())
            var raw = DataFrame([Series("x", Column[Scalar[D]](values^))])
            for layout in range(3):
                var frame = (
                    raw.copy() if layout
                    == 0 else offset_layout(raw) if layout
                    == 1 else chunk_layout(raw)
                )
                var actual = frame.select_exprs(
                    [
                        col("x").min().alias("min"),
                        col("x").max().alias("max"),
                        col("x").arg_min().alias("arg_min"),
                        col("x").arg_max().alias("arg_max"),
                        col("x").n_unique().alias("unique"),
                        col("x").var().alias("var"),
                        col("x").std().alias("std"),
                        col("x").sum().alias("sum"),
                        col("x").sum(min_count=4).alias("sum4"),
                        col("x").sum(min_count=5).alias("sum5"),
                        col("x").mean().alias("mean"),
                    ]
                )
                assert_float(
                    Float64(actual.item(0, "min").numeric[D]()), minima[example]
                )
                assert_float(
                    Float64(actual.item(0, "max").numeric[D]()), maxima[example]
                )
                assert_equal(
                    actual.item(0, "arg_min").numeric[DType.uint32](),
                    argmin[example],
                )
                assert_equal(
                    actual.item(0, "arg_max").numeric[DType.uint32](),
                    argmax[example],
                )
                assert_equal(actual.item(0, "unique").int64(), unique[example])
                assert_float(actual.item(0, "var").float64(), nan)
                assert_float(actual.item(0, "std").float64(), nan)
                var total = (
                    nan if example < 4 else inf if example == 4 else -inf
                )
                for name in ["sum", "sum4"]:
                    assert_true(actual.column(name).dtype() == DataType.of(D))
                    assert_float(
                        Float64(actual.item(0, name).numeric[D]()), total
                    )
                assert_true(actual.item(0, "sum5").is_null())
                assert_true(actual.column("mean").dtype() == DataType.FLOAT64)
                assert_float(actual.item(0, "mean").float64(), total)


def test_unsupported_reduction_type_combinations_reject_even_empty_inputs() raises:
    for source in typed_reduction_columns(reduction_validity(0)):
        var dtype = source.dtype()
        var invalid = List[Expr]()
        if not dtype.is_numeric() and not dtype.is_duration():
            invalid.append(col("x").sum())
        if not dtype.is_numeric():
            invalid.extend(
                [
                    col("x").mean(),
                    col("x").var(),
                    col("x").std(),
                    col("x").median(),
                    col("x").quantile(0.5),
                    col("x").skew(),
                    col("x").kurtosis(),
                ]
            )
        if dtype != DataType.BOOL:
            invalid.extend([col("x").any(), col("x").all()])
        for frame in [
            DataFrame([source.copy()]),
            DataFrame([source.slice(0, 0)]),
        ]:
            for expression in invalid:
                with assert_raises():
                    _ = frame.select(expression)
                with assert_raises():
                    _ = frame.group_by("x").agg([expression.alias("bad")])


def test_global_modes_and_value_counts_keep_typed_values_nulls_and_frequency_order() raises:
    from tests.support.reduction_types_reference import assert_distribution

    for mode in range(3):
        var valid = reduction_validity(mode)
        for source in typed_reduction_columns(valid):
            if source.dtype().is_categorical() or source.dtype().is_decimal():
                continue
            var raw = DataFrame([source.copy()])
            for layout in range(3):
                var frame = (
                    raw.copy() if layout
                    == 0 else offset_layout(raw) if layout
                    == 1 else chunk_layout(raw)
                )
                for empty in [False, True]:
                    var input = frame.clear() if empty else frame.copy()
                    var rows = List[Int]()
                    if not empty:
                        rows.extend([0, 1, 2, 3, 4, 5])
                    var modes = input.select(col("x").mode()).column("x")
                    for sort in [False, True]:
                        for normalize in [False, True]:
                            var counts = input.select(
                                col("x").value_counts(sort, normalize=normalize)
                            ).column("x")
                            assert_distribution(
                                modes,
                                counts,
                                source,
                                valid,
                                rows,
                                sort,
                                normalize,
                            )


def test_reduction_rank_histogram_has_literal_null_and_tie_anchors() raises:
    from tests.support.reduction_types_reference import histogram

    var source = Series("x", Column[Int64]([3, 1, 3, 2, 1, 9]))
    var model = histogram(
        source, [True, True, True, True, True, True], [0, 1, 2, 3, 4, 5]
    )
    assert_equal(model[0], [1, 3, 0, 5])
    assert_equal(model[1], [2, 1, 2, 1])
    model = histogram(
        source, [True, False, True, True, True, False], [0, 1, 2, 3, 4, 5]
    )
    assert_equal(model[0], [4, 3, 0, 1])
    assert_equal(model[1], [1, 1, 2, 2])
    model = histogram(
        source, [False, False, False, False, False, False], [0, 2, 5]
    )
    assert_equal(model[0], [0])
    assert_equal(model[1], [3])
    assert_equal(len(histogram(source, reduction_validity(0), [])[0]), 0)


def test_uint64_quantiles_do_not_expose_internal_ordering_bias() raises:
    var source = Series("x", Column[UInt64]([1, 3]))
    var raw = DataFrame([source.copy(), Series("g", Column[Int64]([0, 0]))])
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        var expressions: List[Expr] = [
            col("x").median().alias("median"),
            col("x").quantile(0.25, "linear").alias("quarter"),
        ]
        for grouped in [False, True]:
            var actual = frame.group_by("g").agg(
                expressions
            ) if grouped else frame.select_exprs(expressions)
            assert_equal(actual.height(), 1)
            for name in ["median", "quarter"]:
                assert_true(actual.column(name).dtype() == DataType.FLOAT64)
                assert_true(not actual.item(0, name).is_null())
                assert_true(
                    actual.item(0, name).float64() >= 0,
                    "UInt64 percentile must not expose signed ordering bias",
                )
                assert_equal(
                    actual.item(0, name).float64(),
                    2.0 if name == "median" else 1.5,
                )


def _advertised_decimal_scalar(operation: String) raises:
    from std.math import sqrt

    for width in [128, 64, 32]:
        var source = (
            Series("x", Column[Int128]([100, 300]))
            .with_dtype(DataType.decimal(8, 2))
            .cast(DataType.decimal(8, 2, width))
        )
        var raw = DataFrame([source.copy(), Series("g", Column[Int64]([0, 0]))])
        var expression = col("x").var()
        var expected = 2.0
        if operation == "std":
            expression = col("x").std()
            expected = sqrt(2.0)
        elif operation == "median":
            expression = col("x").median()
        elif operation == "quantile":
            expression = col("x").quantile(0.25)
            expected = 1.5
        elif operation == "skew":
            expression = col("x").skew()
            expected = 0.0
        elif operation == "kurtosis":
            expression = col("x").kurtosis()
            expected = -2.0
        elif operation == "arg_min":
            expression = col("x").arg_min()
            expected = 0.0
        elif operation == "arg_max":
            expression = col("x").arg_max()
            expected = 1.0
        for grouped in [False, True]:
            var actual = raw.group_by("g").agg(
                [expression.copy()]
            ) if grouped else raw.select(expression)
            assert_equal(actual.height(), 1)
            var value = actual.item(0, "x")
            assert_true(not value.is_null())
            if operation == "arg_min" or operation == "arg_max":
                assert_true(actual.column("x").dtype() == DataType.UINT32)
                assert_equal(value.numeric[DType.uint32](), UInt32(expected))
            else:
                assert_true(actual.column("x").dtype() == DataType.FLOAT64)
                near(value.float64(), expected)


def test_decimal_variance_retains_numeric_scale() raises:
    _advertised_decimal_scalar("var")


def test_decimal_standard_deviation_retains_numeric_scale() raises:
    _advertised_decimal_scalar("std")


def test_decimal_median_retains_numeric_scale() raises:
    _advertised_decimal_scalar("median")


def test_decimal_quantile_retains_numeric_scale() raises:
    _advertised_decimal_scalar("quantile")


def test_decimal_skew_accepts_advertised_numeric_input() raises:
    _advertised_decimal_scalar("skew")


def test_decimal_kurtosis_accepts_advertised_numeric_input() raises:
    _advertised_decimal_scalar("kurtosis")


def test_decimal_arg_min_accepts_advertised_ordered_input() raises:
    _advertised_decimal_scalar("arg_min")


def test_decimal_arg_max_accepts_advertised_ordered_input() raises:
    _advertised_decimal_scalar("arg_max")


def _advertised_decimal_distribution(mode: Bool) raises:
    from tests.support.coverage_compare import assert_value

    for width in [128, 64, 32]:
        var source = (
            Series("x", Column[Int128]([100, 300]))
            .with_dtype(DataType.decimal(8, 2))
            .cast(DataType.decimal(8, 2, width))
        )
        var raw = DataFrame([source.copy(), Series("g", Column[Int64]([0, 0]))])
        var expression = col("x").mode() if mode else col("x").value_counts()
        for grouped in [False, True]:
            var actual = raw.group_by("g").agg(
                [expression.copy()]
            ) if grouped else raw.select(expression)
            var values = actual.item(
                0, "x"
            ).list() if grouped else actual.column("x")
            assert_equal(len(values), 2)
            if mode:
                assert_true(values.dtype() == source.dtype())
                for row in range(2):
                    assert_value(values.get(row), source.get(row))
            else:
                assert_true(
                    values.dtype()
                    == DataType.struct(
                        ["x", "count"], [source.dtype(), DataType.UINT32]
                    )
                )
                for row in range(2):
                    assert_value(
                        values.get(row).struct_field("x"), source.get(row)
                    )
                    assert_equal(
                        values.get(row)
                        .struct_field("count")
                        .numeric[DType.uint32](),
                        UInt32(1),
                    )


def test_decimal_mode_retains_typed_values() raises:
    _advertised_decimal_distribution(True)


def test_decimal_value_counts_retains_typed_values() raises:
    _advertised_decimal_distribution(False)


def _categorical_source() raises -> Series:
    return Series("x", Column[String](["z", "a", "z"])).cast(
        DataType.CATEGORICAL
    )


def test_categorical_extrema_preserve_documented_result_dtype() raises:
    var source = _categorical_source()
    var actual = DataFrame([source.copy()]).select_exprs(
        [col("x").min().alias("min"), col("x").max().alias("max")]
    )
    for name in ["min", "max"]:
        assert_true(
            actual.column(name).dtype() == DataType.STRING,
            "categorical extrema expressions must return decoded String",
        )
        assert_equal(
            actual.item(0, name).string(), "a" if name == "min" else "z"
        )


def test_categorical_modes_preserve_documented_value_dtype() raises:
    var source = _categorical_source()
    var actual = DataFrame([source.copy()]).select(col("x").mode())
    assert_equal(actual.height(), 1)
    assert_true(
        actual.column("x").dtype() == DataType.STRING,
        "categorical mode expressions must return decoded String",
    )
    assert_equal(actual.item().string(), "z")


def test_categorical_value_counts_preserve_documented_value_dtype() raises:
    var source = _categorical_source()
    var actual = DataFrame([source.copy()]).select(col("x").value_counts())
    assert_equal(actual.height(), 2)
    assert_true(
        actual.column("x").dtype()
        == DataType.struct(["x", "count"], [DataType.STRING, DataType.UINT32]),
        "categorical value_counts expressions must return decoded String",
    )
    for row in range(2):
        assert_equal(
            actual.item(row, "x").struct_field("x").string(),
            "a" if row == 0 else "z",
        )
        assert_equal(
            actual.item(row, "x").struct_field("count").numeric[DType.uint32](),
            UInt32(1 if row == 0 else 2),
        )


def test_uint64_quantiles_preserve_full_domain_values_and_interpolation() raises:
    var raw = DataFrame(
        [
            Series(
                "x",
                Column[UInt64](
                    [0, UInt64.MAX, UInt64(1) << 63, 4],
                    [True, True, True, False],
                ),
            ),
            Series("g", Column[Int64]([0, 0, 0, 0])),
        ]
    )
    var a = Float64("4611686018427387904")
    var b = Float64("9223372036854775808")
    var c = Float64("13835058055282163712")
    var d = Float64("18446744073709551616")
    var probabilities: List[Float64] = [0, 0.25, 0.5, 0.75, 1]
    var linear: List[Float64] = [0, a, b, c, d]
    var lower: List[Float64] = [0, 0, b, b, d]
    var higher: List[Float64] = [0, b, b, d, d]
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        for q in range(len(probabilities)):
            var expressions: List[Expr] = [
                col("x").median().alias("median"),
                col("x").quantile(probabilities[q], "linear").alias("linear"),
                col("x").quantile(probabilities[q], "lower").alias("lower"),
                col("x").quantile(probabilities[q], "higher").alias("higher"),
                col("x").quantile(probabilities[q], "nearest").alias("nearest"),
                col("x")
                .quantile(probabilities[q], "midpoint")
                .alias("midpoint"),
            ]
            for grouped in [False, True]:
                var query = (
                    frame.lazy()
                    .group_by("g")
                    .agg(expressions) if grouped else frame.lazy()
                    .select_exprs(expressions)
                )
                for streaming in [False, True]:
                    var actual = query.collect(
                        engine="cpu", streaming=streaming
                    )
                    assert_equal(actual.item(0, "median").float64(), b)
                    assert_equal(actual.item(0, "linear").float64(), linear[q])
                    assert_equal(actual.item(0, "lower").float64(), lower[q])
                    assert_equal(actual.item(0, "higher").float64(), higher[q])
                    assert_equal(actual.item(0, "nearest").float64(), higher[q])
                    assert_equal(
                        actual.item(0, "midpoint").float64(), linear[q]
                    )
