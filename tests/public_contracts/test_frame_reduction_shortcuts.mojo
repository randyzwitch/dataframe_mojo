"""Frame reductions retain columns and select results by input dtype."""
# contract-group: reduce.global
from std.math import sqrt
from std.testing import assert_equal, assert_raises
from dataframe import Column, DataFrame, DataType, Series, ListColumn
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout


def _source() raises -> DataFrame:
    var valid: List[Bool] = [True, False, True]
    return DataFrame(
        [
            Series("i", Column[Int32]([1, 99, 4], valid)),
            Series("f", Column[Float32]([1, 99, 4], valid)),
            Series("b", Column[Bool]([True, False, False], valid)),
            Series("s", Column[String](["b", "ignored", "a"], valid)),
            Series("d", Column[Int64]([0, 99, 2], valid)).with_dtype(
                DataType.DATE
            ),
            Series(
                "l",
                ListColumn(
                    [Int64(0), 1, 1, 2],
                    Series("item", Column[Int64]([1, 2])),
                    [UInt8(0b101)],
                ),
            ),
        ]
    )


def _expected(operation: String, empty: Bool = False) raises -> DataFrame:
    var columns = List[Series]()
    var statistical = (
        operation == "mean"
        or operation == "std"
        or operation == "var"
        or operation == "median"
        or operation == "quantile"
    )
    var value = Float64(0)
    if operation == "sum":
        value = 5
    elif operation == "min":
        value = 1
    elif operation == "max" or operation == "product":
        value = 4
    elif (
        operation == "mean" or operation == "median" or operation == "quantile"
    ):
        value = 2.5
    elif operation == "var":
        value = 4.5
    elif operation == "std":
        value = sqrt(Float64(4.5))
    if empty:
        value = 1 if operation == "product" else 0
    var valid: List[Bool] = [
        not empty
        or operation == "sum"
        or operation == "count"
        or operation == "product"
    ]
    if operation == "count":
        for name in [String("i"), "f", "b", "s", "d", "l"]:
            columns.append(
                Series(name, Column[UInt32]([UInt32(0 if empty else 2)]))
            )
        return DataFrame(columns^)
    if statistical:
        columns.append(Series("i", Column[Float64]([value], valid)))
    elif operation == "product":
        columns.append(Series("i", Column[Int64]([Int64(value)], valid)))
    else:
        columns.append(Series("i", Column[Int32]([Int32(value)], valid)))
    columns.append(Series("f", Column[Float32]([Float32(value)], valid)))
    if operation == "sum":
        columns.append(Series("b", Column[UInt32]([UInt32(0 if empty else 1)])))
    elif operation == "product":
        columns.append(Series("b", Column[Int64]([Int64(1 if empty else 0)])))
    elif operation == "min" or operation == "max":
        columns.append(Series("b", Column[Bool]([operation == "max"], valid)))
    else:
        var boolean = Float64(0.5) if operation != "std" else sqrt(Float64(0.5))
        columns.append(Series("b", Column[Float64]([boolean], valid)))
    if operation == "min" or operation == "max":
        columns.append(
            Series(
                "s",
                Column[String](
                    [String("a") if operation == "min" else "b"], valid
                ),
            )
        )
        columns.append(
            Series(
                "d",
                Column[Int64]([Int64(0 if operation == "min" else 2)], valid),
            ).with_dtype(DataType.DATE)
        )
    else:
        columns.append(
            Series.full_null(
                "s",
                DataType.NULL if operation == "product" else DataType.STRING,
                1,
            )
        )
        columns.append(
            Series.full_null(
                "d",
                DataType.NULL if operation == "product" else DataType.DATE,
                1,
            )
        )
    columns.append(
        Series.full_null(
            "l",
            DataType.NULL if operation
            == "product" else DataType.list(DataType.INT64),
            1,
        )
    )
    return DataFrame(columns^)


def _reduce(
    frame: DataFrame, operation: String, lazy: Bool
) raises -> DataFrame:
    if lazy:
        var plan = frame.lazy()
        if operation == "sum":
            return plan.sum().collect(engine="cpu")
        if operation == "mean":
            return plan.mean().collect(engine="cpu")
        if operation == "min":
            return plan.min().collect(engine="cpu")
        if operation == "max":
            return plan.max().collect(engine="cpu")
        if operation == "count":
            return plan.count().collect(engine="cpu")
        if operation == "product":
            return plan.product().collect(engine="cpu")
        if operation == "median":
            return plan.median().collect(engine="cpu")
        if operation == "std":
            return plan.std().collect(engine="cpu")
        if operation == "var":
            return plan.var().collect(engine="cpu")
        return plan.quantile(0.5, "linear").collect(engine="cpu")
    if operation == "sum":
        return frame.sum()
    if operation == "mean":
        return frame.mean()
    if operation == "min":
        return frame.min()
    if operation == "max":
        return frame.max()
    if operation == "count":
        return frame.count()
    if operation == "product":
        return frame.product()
    if operation == "median":
        return frame.median()
    if operation == "std":
        return frame.std()
    if operation == "var":
        return frame.var()
    return frame.quantile(0.5, "linear")


def test_frame_reductions_cover_mixed_types_eager_lazy_offsets_and_chunks() raises:
    var raw = _source()
    for frame in [raw.copy(), offset_layout(raw), chunk_layout(raw)]:
        for operation in [
            String("sum"),
            "mean",
            "min",
            "max",
            "std",
            "var",
            "median",
            "quantile",
            "count",
            "product",
        ]:
            for lazy in [False, True]:
                assert_frame(
                    _reduce(frame, operation, lazy), _expected(operation)
                )


def test_frame_reductions_empty_all_null_and_zero_column_frames_still_return_one_row() raises:
    var source = _source()
    var columns = List[Series]()
    for name in source.columns():
        columns.append(Series.full_null(name, source.column(name).dtype(), 3))
    var all_null = DataFrame(columns^)
    var no_columns = DataFrame(List[Series](), height=5)
    for operation in [
        String("sum"),
        "mean",
        "min",
        "max",
        "std",
        "var",
        "median",
        "quantile",
        "count",
        "product",
    ]:
        for lazy in [False, True]:
            assert_frame(
                _reduce(source.clear(), operation, lazy),
                _expected(operation, empty=True),
            )
            assert_frame(
                _reduce(all_null, operation, lazy),
                _expected(operation, empty=True),
            )
            assert_frame(
                _reduce(no_columns, operation, lazy),
                DataFrame(List[Series](), height=1),
            )


def test_frame_reduction_quantile_defaults_ddof_and_invalid_options_are_explicit() raises:
    var frame = DataFrame([Series("x", Column[Int64]([1, 4]))])
    assert_equal(frame.quantile(0.25).item().float64(), 1.0)
    assert_equal(
        frame.lazy().quantile(0.25).collect(engine="cpu").item().float64(), 1.0
    )
    assert_equal(frame.quantile(0.25, "linear").item().float64(), 1.75)
    assert_equal(frame.var(0).item().float64(), 2.25)
    assert_equal(
        frame.lazy().var(0).collect(engine="cpu").item().float64(), 2.25
    )
    assert_equal(frame.std(0).item().float64(), 1.5)
    var strings = DataFrame([Series("x", Column[String](["a"]))])
    with assert_raises(contains="ddof"):
        _ = strings.std(-1)
    with assert_raises(contains="ddof"):
        _ = strings.lazy().var(-1)
    for q in [Float64(-0.1), 1.1, Float64(0) / Float64(0)]:
        with assert_raises(contains="quantile"):
            _ = strings.quantile(q)
        with assert_raises(contains="quantile"):
            _ = strings.lazy().quantile(q)
    with assert_raises(contains="interpolation"):
        _ = strings.quantile(0.5, "bad")
    var overflow = DataFrame([Series("x", Column[Int64]([Int64.MAX, 1]))])
    with assert_raises(contains="overflow"):
        _ = overflow.sum()


def test_frame_reductions_handle_decimals_unsigned_duration_binary_and_categorical() raises:
    var unsigned = DataFrame([Series("x", Column[UInt64]([1, 4]))])
    assert_frame(unsigned.sum(), DataFrame([Series("x", Column[UInt64]([5]))]))
    assert_frame(
        unsigned.product(), DataFrame([Series("x", Column[UInt64]([4]))])
    )
    assert_equal(unsigned.median().item().float64(), 2.5)
    for width in [32, 64, 128]:
        var dtype = DataType.decimal(8, 2, width)
        var decimal = DataFrame(
            [
                Series("x", Column[Int128]([100, 400]))
                .with_dtype(DataType.decimal(8, 2))
                .cast(dtype)
            ]
        )
        var total = decimal.lazy().sum().collect(engine="cpu")
        assert_equal(total.column("x").dtype(), DataType.decimal(38, 2))
        assert_equal(total.column("x")._data[Column[Int128]]._get(0), 500)
        assert_equal(decimal.mean().item().float64(), 2.5)
        assert_equal(decimal.var(0).item().float64(), 2.25)
        assert_equal(decimal.product().item().float64(), 4.0)
    var duration = DataFrame(
        [Series("x", Column[Int64]([1, 4])).with_dtype(DataType.duration("us"))]
    )
    assert_frame(
        duration.sum(),
        DataFrame(
            [
                Series("x", Column[Int64]([5])).with_dtype(
                    DataType.duration("us")
                )
            ]
        ),
    )
    assert_equal(duration.mean().item().is_null(), True)
    var binary = DataFrame([Series.binary("x", [[255], [0]])])
    assert_frame(binary.min(), DataFrame([Series.binary("x", [[0]])]))
    var categorical = DataFrame(
        [Series("x", Column[String](["b", "a"])).cast(DataType.CATEGORICAL)]
    )
    var minimum = categorical.lazy().min().collect(engine="cpu")
    assert_equal(minimum.column("x").dtype().is_categorical(), True)
    assert_frame(
        DataFrame([minimum.column("x").cast(DataType.STRING)]),
        DataFrame([Series("x", Column[String](["a"]))]),
    )
    var category_nulls = categorical.sum()
    assert_equal(category_nulls.item().is_null(), True)
    assert_equal(category_nulls.column("x").dtype().is_categorical(), True)
    assert_equal(
        categorical.lazy().sum().collect(engine="cpu").item().is_null(), True
    )
