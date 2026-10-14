"""Exact primitive identity CAST chains retain source Arrow field metadata."""
from std.memory import ArcPointer, bitcast
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    BoolColumn,
    Column,
    DataFrame,
    DataType,
    Expr,
    ListColumn,
    Series,
    StructColumn,
    col,
    lit,
    when,
)
from dataframe.binding import bind
from dataframe.identity_metadata import identity_column_source
from dataframe.execution import _RowsJob, _full
from tests.support.coverage_generators import offset_layout, chunk_layout


def _source() raises -> DataFrame:
    var magnitude = Int128(1) << 80
    var fields: List[Series] = [
        Series("i", Column[Int64]([-7, Int64.MAX, 11], [True, False, True])),
        Series(
            "u",
            Column[UInt64](
                [UInt64.MAX, 0, UInt64(1) << 63], [True, False, True]
            ),
        ),
        Series(
            "f32",
            Column[Float32](
                [Float32(-0.0), Float32(0), Float32(1.5)], [True, False, True]
            ),
        ),
        Series(
            "f64",
            Column[Float64](
                [-0.0, 0.0, bitcast[DType.float64](UInt64(1))],
                [True, False, True],
            ),
        ),
        Series("b", BoolColumn([True, False, False], [True, False, True])),
        Series(
            "d32", Column[Int32]([-123, 0, 456], [True, False, True])
        ).with_dtype(DataType.decimal(9, 2, 32)),
        Series(
            "d64", Column[Int64]([-123, 0, 456], [True, False, True])
        ).with_dtype(DataType.decimal(18, 3, 64)),
        Series(
            "d128",
            Column[Int128](
                [-magnitude, 0, magnitude + Int128(7)], [True, False, True]
            ),
        ).with_dtype(DataType.decimal(38, 2, 128)),
        Series(
            "date", Column[Int64]([0, Int64.MIN, 1], [True, False, True])
        ).with_dtype(DataType.DATE),
        Series(
            "datetime", Column[Int64]([-1, Int64.MIN, 1], [True, False, True])
        ).with_dtype(DataType.datetime("ns", "UTC")),
        Series(
            "duration", Column[Int64]([-1, Int64.MIN, 1], [True, False, True])
        ).with_dtype(DataType.duration("us")),
        Series(
            "time", Column[Int64]([0, Int64.MIN, 17], [True, False, True])
        ).with_dtype(DataType.TIME),
    ]
    for at in range(len(fields)):
        var metadata: List[UInt8] = [
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            107,
            1,
            0,
            0,
            0,
            UInt8(65 + at),
        ]
        fields[at]._field_metadata = ArcPointer(metadata^)
    return DataFrame(fields^)


def _owner(actual: Series, source: Series) raises:
    assert_true(Bool(actual._field_metadata))
    assert_equal(
        actual._field_metadata.value()[], source._field_metadata.value()[]
    )
    assert_true(
        actual._field_metadata.value().ptr()
        == source._field_metadata.value().ptr()
    )


def _expressions(frame: DataFrame, strict: Bool) raises -> List[Expr]:
    var result = List[Expr]()
    for field in frame._columns:
        result.append(
            col(field.name())
            .cast(field.dtype(), strict=strict)
            .cast(field.dtype(), strict=strict)
            .alias(field.name())
        )
    return result^


def _check(actual: DataFrame, source: DataFrame) raises:
    assert_true(actual.equals(source))
    for at in range(source.width()):
        _owner(actual._columns[at], source._columns[at])
    if actual.height():
        assert_equal(
            bitcast[DType.uint32](
                actual.column("f32").rechunk().float32()._get(0)
            ),
            UInt32(0x80000000),
        )
        assert_equal(
            bitcast[DType.uint64](
                actual.column("f64").rechunk().float64()._get(0)
            ),
            UInt64(0x8000000000000000),
        )
        assert_equal(
            actual.column("d128").rechunk()._decimal_raw(2),
            (Int128(1) << 80) + Int128(7),
        )


def test_identity_column_metadata_primitive_logical_empty_and_batched() raises:
    var raw = _source()
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        for empty in [False, True]:
            var source = frame.clear() if empty else frame.copy()
            for strict in [False, True]:
                for batch in [1, 64]:
                    _check(
                        source.select_exprs(
                            _expressions(source, strict), batch_size=batch
                        ),
                        source,
                    )
    var typed = raw.select_exprs(
        [
            col("i")
            .cast(DataType.FLOAT64)
            .cast(DataType.INT64)
            .alias("roundtrip"),
            (col("i") + lit(Int64(0))).cast(DataType.INT64).alias("computed"),
            col("date").cast(DataType.datetime("ms")).alias("changed"),
        ],
        batch_size=1,
    )
    for column in typed._columns:
        assert_false(Bool(column._field_metadata))
    assert_equal(typed.item(0, "roundtrip").int64(), Int64(-7))
    assert_equal(typed.item(2, "roundtrip").int64(), Int64(11))
    assert_equal(typed.item(2, "changed").to_physical(), Int64(86400000))
    var dangerous = raw.with_column(
        Series("bad", Column[Int64]([128, 128, 128]))
    )
    var hidden = dangerous.select(
        when(lit(False))
        .then(col("bad").cast(DataType.INT8))
        .otherwise(lit(Int8(0)))
        .alias("masked"),
        batch_size=1,
    )
    assert_false(Bool(hidden.column("masked")._field_metadata))
    for row in range(hidden.height()):
        assert_equal(hidden.item(row, "masked").int8(), Int8(0))


def test_identity_column_metadata_full_worker_and_nested_boundaries() raises:
    var frame = _source()
    var expression = col("i").cast(DataType.INT64).cast(DataType.INT64)
    var bound = bind(expression, frame._columns)
    assert_equal(identity_column_source(bound, len(bound.expr._nodes) - 1), 0)
    var states = List[Series]()
    for dtype in bound.dtypes:
        states.append(Series.full_null("", dtype, 0))
    var full = _full[8](
        bound, frame._columns, states, len(bound.expr._nodes) - 1, 3, 1, False
    )
    _owner(full, frame.column("i"))
    var job = _RowsJob[8](
        bound,
        frame._columns,
        states,
        len(bound.expr._nodes) - 1,
        1,
        3,
        1,
        False,
    )
    job.run()
    var part = job^.into_result()
    _owner(part, frame.column("i"))
    assert_equal(len(part), 2)
    assert_false(part.int64()._valid(0))
    assert_equal(part.int64()._get(1), Int64(11))
    var running = frame.select(
        col("i").cast(DataType.INT64).cum_sum().alias("sum"), batch_size=1
    )
    assert_false(Bool(running.column("sum")._field_metadata))
    assert_equal(running.item(0, "sum").int64(), Int64(-7))
    assert_equal(running.item(2, "sum").int64(), Int64(4))
    var child = frame.column("i").renamed("value")
    var nested = DataFrame(
        [
            Series("record", StructColumn([child.copy()])),
            Series("list", ListColumn([0, 1, 2, 3], child.copy())),
        ]
    )
    var field = nested.select(
        col("record").field("value").cast(DataType.INT64).alias("field")
    )
    _owner(field.column("field"), child)
    for name in ["record", "list"]:
        var dtype = nested.column(name).dtype()
        with assert_raises(contains="list and struct casts are not supported"):
            _ = nested.select(col(name).cast(dtype))
        var binding = bind(col(name).cast(dtype), nested._columns)
        assert_equal(
            identity_column_source(binding, len(binding.expr._nodes) - 1), -1
        )
