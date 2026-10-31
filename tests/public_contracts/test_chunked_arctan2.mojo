"""atan2 keeps scalar libm semantics while reading physical numeric spans."""
# contract-group: expr.chunk_windows
# contract-group: expr.arithmetic
from std.math import isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    BoolColumn,
    StringColumn,
    DataFrame,
    DataType,
    Series,
    arctan2,
    col,
    lit,
    when,
)
from dataframe.kernels.binary import arctan2_series
from tests.support.coverage_compare import assert_frame


def _parts(source: Series, right: Bool) raises -> Series:
    if right:
        return Series._from_chunks(
            [
                source.slice(0, 3),
                source.slice(3, 0),
                source.slice(3, 1),
                source.slice(4, 9),
                source.slice(13, 3),
            ]
        )
    return Series._from_chunks(
        [
            source.slice(0, 1),
            source.slice(1, 7),
            source.slice(8, 0),
            source.slice(8, 3),
            source.slice(11, 5),
        ]
    )


def _fixture[D: DType](right: Bool) raises -> Series:
    var inf = Float64(1) / Float64(0)
    var nan = Float64(0) / Float64(0)
    var seed = List[Float64]()
    if right:
        seed = [
            0.0,
            0.0,
            -0.0,
            -0.0,
            0,
            0,
            inf,
            inf,
            -inf,
            -inf,
            1,
            nan,
            3,
            4,
            1,
            -1,
        ]
    else:
        seed = [
            0.0,
            -0.0,
            0.0,
            -0.0,
            1,
            -1,
            inf,
            -inf,
            inf,
            -inf,
            nan,
            1,
            99,
            88,
            1,
            1,
        ]
    var values = List[Scalar[D]]()
    var valid = List[Bool]()
    for row in range(3):
        values.append(0)
        valid.append(True)
    for row in range(16):
        values.append(seed[row].cast[D]())
        valid.append(row != (13 if right else 12))
    return Series(
        "x" if right else "y", Column[Scalar[D]](values^, valid)
    ).slice(3, 16)


def _cold(source: Series) raises:
    if source.is_chunked():
        assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))


def _check(
    actual: Series,
    dtype: DataType,
    name: String,
    values: List[Float64],
    valid: List[Bool],
) raises:
    assert_equal(actual.name(), name)
    assert_true(actual.dtype() == dtype)
    assert_equal(len(actual), len(values))
    assert_equal(len(valid), len(values))
    for row in range(len(values)):
        assert_equal(actual.get(row).is_null(), not valid[row])
        if not valid[row]:
            if dtype == DataType.FLOAT32:
                assert_equal(
                    bitcast[DType.uint32](
                        actual.numeric[DType.float32]()._get(row)
                    ),
                    UInt32(0),
                )
            else:
                assert_equal(
                    bitcast[DType.uint64](
                        actual.numeric[DType.float64]()._get(row)
                    ),
                    UInt64(0),
                )
            continue
        var value = (
            Float64(actual.get(row).numeric[DType.float32]()) if dtype
            == DataType.FLOAT32 else actual.get(row).float64()
        )
        if isnan(values[row]):
            assert_true(isnan(value))
        elif values[row] == 0:
            assert_equal(
                bitcast[DType.uint64](value), bitcast[DType.uint64](values[row])
            )
        else:
            var eps = Float64(5e-7) if dtype == DataType.FLOAT32 else Float64(
                1e-14
            )
            assert_true(
                abs(value - values[row])
                <= eps * max(Float64(1), abs(values[row]))
            )


def _angles() -> List[Float64]:
    var nan = Float64(0) / Float64(0)
    return [
        0.0,
        -0.0,
        3.141592653589793,
        -3.141592653589793,
        1.5707963267948966,
        -1.5707963267948966,
        0.7853981633974483,
        -0.7853981633974483,
        2.356194490192345,
        -2.356194490192345,
        nan,
        nan,
        0,
        0,
        0.7853981633974483,
        2.356194490192345,
    ]


def _angle_valid() -> List[Bool]:
    return [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
    ]


def _owned[D: DType]() raises -> Series:
    var y = _parts(_fixture[D](False), False)
    var x = _parts(_fixture[D](True), True)
    var result = arctan2_series(y, x, DataType.of(D))
    _cold(y)
    _cold(x)
    return result^


def _physical_answers[D: DType]() raises:
    _check(_owned[D](), DataType.of(D), "", _angles(), _angle_valid())
    var y = _parts(_fixture[D](False), False)
    var x = _parts(_fixture[D](True), True)
    var direct = arctan2_series(y, x, DataType.of(D))
    var flat = arctan2_series(
        _fixture[D](False), _fixture[D](True), DataType.of(D)
    )
    assert_frame(
        DataFrame([direct.renamed("a")]), DataFrame([flat.renamed("a")])
    )
    var public_result = (
        DataFrame([y.copy(), x.copy()])
        .lazy()
        .select_exprs([arctan2(col("y"), col("x")).alias("angle")])
        .collect(engine="cpu", batch_size=3)
    )
    var names: List[String] = ["angle"]
    assert_equal(public_result.columns(), names)
    _check(
        public_result.column("angle"),
        DataType.of(D),
        "angle",
        _angles(),
        _angle_valid(),
    )
    _cold(y)
    _cold(x)


def test_arctan2_sliced_unequal_windows_preserve_complete_ieee_answers_and_owners() raises:
    _physical_answers[DType.float32]()
    _physical_answers[DType.float64]()


def test_arctan2_whole_scalars_and_ready_cache_keep_physical_row_order() raises:
    var flat = Series(
        "y", Column[Float64]([0, 1, -1, 99, 0], [True, True, True, False, True])
    )
    var rows = Series._from_chunks(
        [
            flat.slice(0, 1),
            flat.slice(1, 1),
            flat.slice(2, 1),
            flat.slice(3, 1),
            flat.slice(4, 1),
        ]
    )
    var one = Series("x", Column[Float64]([1]))
    _check(
        arctan2_series(rows, one, DataType.FLOAT64),
        DataType.FLOAT64,
        "",
        [0, 0.7853981633974483, -0.7853981633974483, 0, 0],
        [True, True, True, False, True],
    )
    var xflat = Series(
        "x", Column[Float64]([1, -1, 0, 88, 1], [True, True, True, False, True])
    )
    var xs = Series._from_chunks([xflat.slice(0, 2), xflat.slice(2, 3)])
    _check(
        arctan2_series(
            Series("y", Column[Float64]([-0.0])), xs, DataType.FLOAT64
        ),
        DataType.FLOAT64,
        "",
        [-0.0, -3.141592653589793, -0.0, 0, -0.0],
        [True, True, True, False, True],
    )
    var missing = Series("x", Column[Float64]([99], [False]))
    _check(
        arctan2_series(rows, missing, DataType.FLOAT64),
        DataType.FLOAT64,
        "",
        [0, 0, 0, 0, 0],
        [False, False, False, False, False],
    )
    _cold(rows)
    _cold(xs)
    var merged_y = rows.rechunk()
    var merged_x = xs.rechunk()
    _ = merged_y^
    _ = merged_x^
    _check(
        arctan2_series(rows, one, DataType.FLOAT64),
        DataType.FLOAT64,
        "",
        [0, 0.7853981633974483, -0.7853981633974483, 0, 0],
        [True, True, True, False, True],
    )
    _check(
        arctan2_series(rows, xs, DataType.FLOAT64),
        DataType.FLOAT64,
        "",
        [0, 2.356194490192345, -1.5707963267948966, 0, 0],
        [True, True, True, False, True],
    )
    assert_equal(rows._chunked.value()[].cache[].state.load(), Int64(2))
    assert_equal(xs._chunked.value()[].cache[].state.load(), Int64(2))


def _decimal_case(value: Series) raises:
    var frame = DataFrame(
        [value.renamed("d"), Series("f", Column[Float32]([1, 1, 1]))]
    )
    var result = frame.select_exprs(
        [
            arctan2(col("d"), lit(Float64(1))).alias("decimal_y"),
            arctan2(col("f"), col("d")).alias("decimal_x"),
            arctan2(col("d") + col("d"), lit(Float64(2))).alias("computed"),
        ],
        batch_size=2,
    )
    var valid: List[Bool] = [True, True, False]
    _check(
        result.column("decimal_y"),
        DataType.FLOAT64,
        "decimal_y",
        [-0.7853981633974483, 0.7853981633974483, 0],
        valid,
    )
    _check(
        result.column("decimal_x"),
        DataType.FLOAT32,
        "decimal_x",
        [2.356194490192345, 0.7853981633974483, 0],
        valid,
    )
    _check(
        result.column("computed"),
        DataType.FLOAT64,
        "computed",
        [-0.7853981633974483, 0.7853981633974483, 0],
        valid,
    )


def test_arctan2_target_dtype_casts_x_before_libm_and_preserves_decimal_scale() raises:
    var y = Series._from_chunks(
        [Series("y", Column[Float32]([1])), Series("y", Column[Float32]([-1]))]
    )
    var x = Series._from_chunks(
        [
            Series("x", Column[Float64]([1e39])),
            Series("x", Column[Float64]([1e39])),
        ]
    )
    _check(
        arctan2_series(y, x, DataType.FLOAT32),
        DataType.FLOAT32,
        "",
        [0.0, -0.0],
        [True, True],
    )
    var result = DataFrame([y.copy(), x.copy()]).select_exprs(
        [arctan2(col("y"), col("x")).alias("a")]
    )
    _check(result.column("a"), DataType.FLOAT32, "a", [0.0, -0.0], [True, True])
    var other = DataFrame(
        [Series("i", Column[Int64]([1])), Series("f", Column[Float32]([1]))]
    ).select_exprs([arctan2(col("i"), col("f")).alias("a")])
    _check(
        other.column("a"), DataType.FLOAT64, "a", [0.7853981633974483], [True]
    )
    _decimal_case(
        Series(
            "d", Column[Int32]([-100, 100, 999], [True, True, False])
        ).with_dtype(DataType.decimal(9, 2, 32))
    )
    _decimal_case(
        Series(
            "d", Column[Int64]([-100, 100, 999], [True, True, False])
        ).with_dtype(DataType.decimal(18, 2, 64))
    )
    _decimal_case(
        Series(
            "d", Column[Int128]([-100, 100, 999], [True, True, False])
        ).with_dtype(DataType.decimal(38, 2, 128))
    )


def test_arctan2_public_logical_type_validation_precedes_empty_or_inactive_rows() raises:
    var base = DataFrame(
        [
            Series("n", Column[Int64]([1, 2])),
            Series("b", BoolColumn([True, False])),
            Series("date", Column[Int64]([0, 1])).with_dtype(DataType.DATE),
            Series("text", StringColumn(["1", "2"])),
        ]
    )
    for frame in [base.copy(), base.clear()]:
        for name in ["b", "date", "text"]:
            for reverse in [False, True]:
                var bad = arctan2(col("n"), col(name)) if reverse else arctan2(
                    col(name), col("n")
                )
                with assert_raises(
                    contains="arctan2 requires numeric operands"
                ):
                    _ = frame.select_exprs([bad.copy()])
                with assert_raises(
                    contains="arctan2 requires numeric operands"
                ):
                    _ = frame.select_exprs(
                        [when(lit(False)).then(bad).otherwise(lit(Float64(0)))]
                    )


def test_arctan2_private_cast_errors_precede_length_and_empty_rules() raises:
    var ybad = Series._from_chunks(
        [
            Series("y", StringColumn(["1"])),
            Series("y", StringColumn(["bad", "2"])),
        ]
    )
    var xbad = Series("x", StringColumn(["oops", "1"]))
    with assert_raises(contains="failed at row 1"):
        _ = arctan2_series(ybad, xbad, DataType.FLOAT64)
    with assert_raises(contains="failed at row 0"):
        _ = arctan2_series(
            Series("y", StringColumn(["1", "2", "3"])), xbad, DataType.FLOAT64
        )
    with assert_raises(contains="failed at row 1"):
        _ = arctan2_series(
            ybad, Series("x", Column[Float64]([])), DataType.FLOAT64
        )
    var empty = Series("y", Column[Float64]([]))
    var one = Series("x", Column[Float64]([1]))
    var two = Series("x", Column[Float64]([1, 2]))
    with assert_raises(contains="Incompatible expression lengths"):
        _ = arctan2_series(two, empty, DataType.FLOAT64)
    with assert_raises(contains="Incompatible expression lengths"):
        _ = arctan2_series(empty, two, DataType.FLOAT64)
    _check(
        arctan2_series(one, empty, DataType.FLOAT64),
        DataType.FLOAT64,
        "",
        [],
        [],
    )
    _check(
        arctan2_series(empty, one, DataType.FLOAT64),
        DataType.FLOAT64,
        "",
        [],
        [],
    )
    _check(
        arctan2_series(empty, empty, DataType.FLOAT64),
        DataType.FLOAT64,
        "",
        [],
        [],
    )
