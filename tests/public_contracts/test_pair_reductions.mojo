"""Public two-input reductions use complete pairs and independently ranked ties."""
from std.math import isnan, abs
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, corr, cov
from tests.support.numeric_model import INTEGER_DTYPES, FLOAT_DTYPES
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.pair_reduction_reference import PairReference, average_ranks


def _numeric_columns(
    name: String, values: List[Float64], valid: List[Bool]
) raises -> List[Series]:
    var columns = List[Series]()
    comptime types = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.float32,
        DType.float64,
    )
    comptime for i in range(len(types)):
        comptime D = types[i]
        var payload = List[Scalar[D]]()
        for value in values:
            payload.append(value.cast[D]())
        columns.append(Series(name, Column[Scalar[D]](payload^, valid.copy())))
    return columns^


def _expressions() -> List[Expr]:
    return [
        corr(col("x"), col("y")).alias("pearson"),
        corr(col("x"), col("y"), method="spearman").alias("spearman"),
        cov(col("x"), col("y"), ddof=0).alias("cov0"),
        cov(col("x"), col("y"), ddof=1).alias("cov1"),
        cov(col("x"), col("y"), ddof=6).alias("cov6"),
    ]


def _near(actual: Float64, expected: Float64) raises:
    if isnan(expected):
        assert_true(isnan(actual))
    else:
        assert_true(not isnan(actual))
        # Values and ranks <=9, at most6 rows. Two-pass independent sums
        # vs online Chan/Welford reductions receive this explicit budget.
        assert_true(abs(actual - expected) <= 2e-12 * (1 + abs(expected)))


def _assert_pairs(
    actual: DataFrame, output: Int, reference: PairReference
) raises:
    for name in ["pearson", "spearman", "cov0", "cov1", "cov6"]:
        assert_true(actual.column(name).dtype() == DataType.FLOAT64)
    for name in ["pearson", "spearman"]:
        assert_true(not actual.item(output, name).is_null())
        _near(
            actual.item(output, name).float64(),
            reference.pearson if name == "pearson" else reference.spearman,
        )
    for ddof in [0, 1, 6]:
        var name = "cov" + String(ddof)
        assert_equal(actual.item(output, name).is_null(), reference.count == 0)
        if reference.count:
            _near(
                actual.item(output, name).float64(), reference.covariance(ddof)
            )


def test_pairwise_reference_has_literal_centered_sum_and_average_rank_anchors() raises:
    assert_equal(average_ranks([4.0, 1.0, 4.0, 2.0]), [3.5, 1.0, 3.5, 2.0])
    var model = PairReference(
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [True, True, True],
        [True, True, True],
        [0, 1, 2],
    )
    assert_equal(model.comoment, 4.0)
    assert_equal(model.pearson, 1.0)
    assert_equal(model.spearman, 1.0)
    assert_equal(model.covariance(0), 4.0 / 3.0)
    assert_equal(model.covariance(1), 2.0)
    model = PairReference(
        [1.0, 99.0, 3.0],
        [2.0, 4.0, 99.0],
        [True, False, True],
        [True, True, False],
        [0, 1, 2],
    )
    assert_equal(model.count, 1)
    assert_true(isnan(model.pearson))
    assert_equal(model.covariance(1), 0.0)


def test_public_pairwise_statistics_cover_every_numeric_width_and_complete_group_rows() raises:
    var xs: List[Float64] = [3, 1, 3, 2, 1, 9]
    var ys: List[Float64] = [4, 1, 4, 3, 2, 8]
    var members: List[List[Int]] = [[0, 2, 4], [1, 3, 5]]
    for mode in range(4):
        var xv = List[Bool](length=6, fill=mode < 2)
        var yv = List[Bool](length=6, fill=mode < 3)
        if mode == 1:
            xv = [True, False, True, True, False, True]
            yv = [False, True, True, True, True, False]
        if mode == 3:
            xv[0] = True
            yv[0] = True
        for x in _numeric_columns("x", xs, xv):
            for y in _numeric_columns("y", ys, yv):
                var raw = DataFrame(
                    [
                        x.copy(),
                        y.copy(),
                        Series("g", Column[Int64]([0, 1, 0, 1, 0, 1])),
                    ]
                )
                for layout in range(3):
                    var frame = (
                        raw.copy() if layout
                        == 0 else offset_layout(raw) if layout
                        == 1 else chunk_layout(raw)
                    )
                    var actual = frame.select_exprs(_expressions())
                    assert_equal(
                        actual.columns(),
                        ["pearson", "spearman", "cov0", "cov1", "cov6"],
                    )
                    assert_equal(actual.height(), 1)
                    _assert_pairs(
                        actual,
                        0,
                        PairReference(xs, ys, xv, yv, [0, 1, 2, 3, 4, 5]),
                    )
                    var grouped = frame.group_by("g").agg(_expressions())
                    assert_equal(
                        grouped.columns(),
                        ["g", "pearson", "spearman", "cov0", "cov1", "cov6"],
                    )
                    assert_equal(grouped.height(), 2)
                    assert_true(grouped.column("g").dtype() == DataType.INT64)
                    var seen = List[Bool](length=2, fill=False)
                    for output in range(2):
                        var group = Int(grouped.item(output, "g").int64())
                        assert_true(group >= 0 and group < 2)
                        assert_true(not seen[group])
                        seen[group] = True
                        _assert_pairs(
                            grouped,
                            output,
                            PairReference(xs, ys, xv, yv, members[group]),
                        )
                    var empty = frame.clear().select_exprs(_expressions())
                    assert_equal(empty.height(), 1)
                    assert_equal(empty.columns(), actual.columns())
                    _assert_pairs(empty, 0, PairReference(xs, ys, xv, yv, []))
                    var empty_grouped = (
                        frame.clear().group_by("g").agg(_expressions())
                    )
                    assert_equal(empty_grouped.height(), 0)
                    assert_equal(empty_grouped.columns(), grouped.columns())
                    for name in grouped.columns():
                        assert_true(
                            empty_grouped.column(name).dtype()
                            == grouped.column(name).dtype()
                        )


def test_pairwise_float_constants_nonfinite_values_and_hidden_null_payloads() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var x_examples: List[List[Float64]] = [
        [-0.0, 0.0, -0.0, 0.0, -0.0, 0.0],
        [1, 2, nan, 3, 4, 5],
        [1, 2, inf, 3, 4, 5],
        [1, 2, -inf, 3, 4, 5],
        [3, 1, 3, 2, 1, 9],
    ]
    var y_examples: List[List[Float64]] = [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [7, 7, 7, 7, 7, 7],
    ]
    comptime for width in range(len(FLOAT_DTYPES)):
        comptime D = FLOAT_DTYPES[width]
        for example in range(len(x_examples)):
            for poison_null in [False, True]:
                var valid = List[Bool](length=6, fill=True)
                valid[2] = not poison_null
                var x = List[Scalar[D]]()
                var y = List[Scalar[D]]()
                for row in range(6):
                    x.append(x_examples[example][row].cast[D]())
                    y.append(y_examples[example][row].cast[D]())
                var raw = DataFrame(
                    [
                        Series("x", Column[Scalar[D]](x^, valid.copy())),
                        Series("y", Column[Scalar[D]](y^)),
                        Series("g", Column[Int64]([0, 1, 0, 1, 0, 1])),
                    ]
                )
                for layout in range(3):
                    var frame = (
                        raw.copy() if layout
                        == 0 else offset_layout(raw) if layout
                        == 1 else chunk_layout(raw)
                    )
                    var expressions: List[Expr] = [
                        corr(col("x"), col("y")).alias("pearson"),
                        corr(col("x"), col("y"), method="spearman").alias(
                            "spearman"
                        ),
                        cov(col("x"), col("y"), ddof=0).alias("cov0"),
                        cov(col("x"), col("y"), ddof=1).alias("cov1"),
                    ]
                    for grouped in [False, True]:
                        var actual = frame.group_by(
                            "g", maintain_order=True
                        ).agg(expressions) if grouped else frame.select_exprs(
                            expressions
                        )
                        assert_equal(actual.height(), 2 if grouped else 1)
                        for output in range(actual.height()):
                            var rows: List[Int] = [0, 1, 2, 3, 4, 5]
                            if grouped:
                                rows = [0, 2, 4] if output == 0 else [1, 3, 5]
                                assert_equal(
                                    actual.item(output, "g").int64(),
                                    Int64(output),
                                )
                            var model = PairReference(
                                x_examples[example],
                                y_examples[example],
                                valid,
                                List[Bool](length=6, fill=True),
                                rows,
                            )
                            for name in ["pearson", "spearman", "cov0", "cov1"]:
                                assert_true(
                                    actual.column(name).dtype()
                                    == DataType.FLOAT64
                                )
                                assert_true(
                                    not actual.item(output, name).is_null()
                                )
                                var expected = (
                                    model.pearson if name
                                    == "pearson" else model.spearman if name
                                    == "spearman" else model.covariance(
                                        0 if name == "cov0" else 1
                                    )
                                )
                                _near(
                                    actual.item(output, name).float64(),
                                    expected,
                                )


def test_pairwise_binding_rejects_invalid_types_methods_and_degrees_of_freedom() raises:
    var raw = DataFrame(
        [
            Series("x", Column[Int64]([1, 2])),
            Series("s", Column[String](["1", "2"])),
            Series("b", Column[Bool]([False, True])),
        ]
    )
    var expressions: List[Expr] = [
        corr(col("x"), col("s")),
        corr(col("b"), col("x")),
        cov(col("s"), col("x")),
        cov(col("x"), col("b")),
        corr(col("x"), col("x"), method="kendall"),
        cov(col("x"), col("x"), ddof=-1),
    ]
    for frame in [raw.copy(), raw.clear()]:
        for expression in expressions:
            with assert_raises():
                _ = frame.select(expression)
            with assert_raises():
                _ = frame.group_by("b").agg([expression.alias("bad")])


def _decimal_pair_contract(method: String) raises:
    for width in [128, 64, 32]:
        var decimal = (
            Series("x", Column[Int128]([100, 300]))
            .with_dtype(DataType.decimal(8, 2))
            .cast(DataType.decimal(8, 2, width))
        )
        var floating = Series("y", Column[Float64]([2, 4]))
        for reversed in [False, True]:
            var frame = DataFrame(
                [
                    decimal.copy(),
                    floating.copy(),
                    Series("g", Column[Int64]([0, 0])),
                ]
            )
            var left = col("y") if reversed else col("x")
            var right = col("x") if reversed else col("y")
            var expression = cov(left, right) if method == "cov" else corr(
                left, right, method=method
            )
            for grouped in [False, True]:
                var actual = frame.group_by("g").agg(
                    [expression.alias("value")]
                ) if grouped else frame.select(expression.alias("value"))
                assert_true(actual.column("value").dtype() == DataType.FLOAT64)
                assert_equal(actual.height(), 1)
                assert_true(not actual.item(0, "value").is_null())
                _near(
                    actual.item(0, "value").float64(),
                    2.0 if method == "cov" else 1.0,
                )


def test_decimal_pearson_accepts_advertised_numeric_pairs() raises:
    _decimal_pair_contract("pearson")


def test_decimal_spearman_accepts_advertised_numeric_pairs() raises:
    _decimal_pair_contract("spearman")


def test_decimal_covariance_accepts_advertised_numeric_pairs() raises:
    _decimal_pair_contract("cov")
