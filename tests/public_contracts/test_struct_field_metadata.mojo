"""Primitive source Struct fields retain child owners across row accumulators."""
# contract-group: public.accelerator_staging
# contract-group: public.apple_parity
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StructColumn,
    as_struct,
    col,
    lit,
    when,
)
from dataframe.binding import bind
from dataframe.execution import _RowsJob


def _source() raises -> DataFrame:
    # One packed Arrow application metadata entry: tag -> raw.
    var metadata: List[UInt8] = [
        1,
        0,
        0,
        0,
        3,
        0,
        0,
        0,
        116,
        97,
        103,
        3,
        0,
        0,
        0,
        114,
        97,
        119,
    ]
    var value = Series(
        "value",
        Column[Int64]([7, Int64.MAX, 11, 13], [True, True, False, True]),
    )
    value._field_metadata = ArcPointer(metadata^)
    var fields: List[Series] = [value^]
    var bits: List[UInt8] = [13]
    return DataFrame([Series("record", StructColumn(fields^, bits^))])


def _owner(actual: Series, original: Series) raises:
    assert_true(Bool(actual._field_metadata))
    assert_equal(
        actual._field_metadata.value()[], original._field_metadata.value()[]
    )
    assert_true(
        actual._field_metadata.value().ptr()
        == original._field_metadata.value().ptr()
    )


def test_struct_field_metadata_empty_batched_nullable_and_computed_roots() raises:
    var raw = _source()
    var child = raw.column("record").struct_column().field("value")
    for offset in [0, 1]:
        var frame = raw.slice(offset, 4 - offset)
        for empty in [False, True]:
            var source = frame.clear() if empty else frame.copy()
            for batch in [1, 64]:
                var actual = source.select_exprs(
                    [
                        col("record").field("value").alias("out"),
                        (col("record").field("value") + 1).alias("computed"),
                    ],
                    batch_size=batch,
                )
                assert_equal(actual.columns(), ["out", "computed"])
                assert_equal(actual.height(), 0 if empty else 4 - offset)
                assert_equal(actual.column("out").dtype(), DataType.INT64)
                _owner(actual.column("out"), child)
                assert_false(Bool(actual.column("computed")._field_metadata))
                for row in range(actual.height()):
                    var source_row = offset + row
                    assert_equal(
                        actual.item(row, "out").is_null(),
                        source_row == 1 or source_row == 2,
                    )
                    assert_equal(
                        actual.item(row, "computed").is_null(),
                        source_row == 1 or source_row == 2,
                    )
                    if source_row == 0 or source_row == 3:
                        assert_equal(
                            actual.item(row, "out").int64(),
                            Int64(7 if source_row == 0 else 13),
                        )
                        assert_equal(
                            actual.item(row, "computed").int64(),
                            Int64(8 if source_row == 0 else 14),
                        )
    # The _full accumulator is used for the field feeding a window, whereas
    # the computed cumulative result itself starts with fresh metadata.
    var running = raw.select_exprs(
        [col("record").field("value").cum_sum().alias("running")], batch_size=1
    )
    assert_false(Bool(running.column("running")._field_metadata))
    assert_equal(running.item(0, "running").int64(), Int64(7))
    assert_true(running.item(1, "running").is_null())
    assert_true(running.item(2, "running").is_null())
    assert_equal(running.item(3, "running").int64(), Int64(20))
    # Reserving/appending the metadata seed must not mutate shared inputs.
    assert_equal(child.int64()._get(0), Int64(7))
    assert_equal(child.int64()._get(1), Int64.MAX)
    assert_false(child.int64()._valid(2))
    assert_equal(child.int64()._get(3), Int64(13))


def test_struct_field_metadata_worker_accumulator_interval() raises:
    var frame = _source()
    var child = frame.column("record").struct_column().field("value")
    var bound = bind(col("record").field("value"), frame._columns)
    var states = List[Series]()
    for dtype in bound.dtypes:
        states.append(Series.full_null("", dtype, 0))
    # Exercise the worker-owned accumulator independent of host core count
    # or DATAFRAME_THREADS, including a nonzero input row interval.
    var job = _RowsJob[8](
        bound,
        frame._columns,
        states,
        len(bound.expr._nodes) - 1,
        1,
        4,
        1,
        False,
    )
    job.run()
    var actual = job^.into_result()
    _owner(actual, child)
    assert_equal(len(actual), 3)
    assert_false(actual.int64()._valid(0))
    assert_false(actual.int64()._valid(1))
    assert_equal(actual.int64()._get(2), Int64(13))


def test_struct_field_metadata_zero_rows_do_not_evaluate_packed_siblings() raises:
    var child = _source().column("record").struct_column().field("value")
    var source = DataFrame([child.renamed("seed")])
    var expression = (
        as_struct(
            [
                col("seed").alias("value"),
                lit(Int64(128)).cast(DataType.INT8).alias("checked"),
            ],
            "record",
        )
        .field("value")
        .alias("out")
    )
    var actual = source.clear().select_exprs([expression.copy()], batch_size=1)
    assert_equal(actual.height(), 0)
    assert_equal(actual.column("out").dtype(), DataType.INT64)
    _owner(actual.column("out"), child)
    with assert_raises(contains="cast from int64 to int8 failed"):
        _ = source.select_exprs([expression.copy()], batch_size=1)


def _identity_field() -> Expr:
    return (
        col("record").field("value").cast(DataType.INT64).cast(DataType.INT64)
    )


def test_struct_field_identity_cast_metadata_empty_batched_masks_and_computed() raises:
    var raw = _source()
    var child = raw.column("record").struct_column().field("value")
    for offset in [0, 1]:
        var sliced = raw.slice(offset, 4 - offset)
        for empty in [False, True]:
            var source = sliced.clear() if empty else sliced.copy()
            for batch in [1, 64]:
                var actual = source.select_exprs(
                    [
                        col("record")
                        .field("value")
                        .cast(DataType.INT64)
                        .alias("once"),
                        _identity_field().alias("chain"),
                        (_identity_field() + 1).alias("computed"),
                        col("record")
                        .field("value")
                        .cast(DataType.INT32)
                        .cast(DataType.INT32)
                        .alias("changed"),
                    ],
                    batch_size=batch,
                )
                assert_equal(actual.height(), 0 if empty else 4 - offset)
                assert_equal(
                    actual.columns(), ["once", "chain", "computed", "changed"]
                )
                _owner(actual.column("once"), child)
                _owner(actual.column("chain"), child)
                assert_false(Bool(actual.column("computed")._field_metadata))
                assert_false(Bool(actual.column("changed")._field_metadata))
                assert_equal(actual.column("changed").dtype(), DataType.INT32)
                for row in range(actual.height()):
                    var source_row = offset + row
                    var missing = source_row == 1 or source_row == 2
                    for name in ["once", "chain", "computed", "changed"]:
                        assert_equal(actual.item(row, name).is_null(), missing)
                    if not missing:
                        var value = Int64(7 if source_row == 0 else 13)
                        assert_equal(actual.item(row, "once").int64(), value)
                        assert_equal(actual.item(row, "chain").int64(), value)
                        assert_equal(
                            actual.item(row, "computed").int64(), value + 1
                        )
                        assert_equal(
                            actual.item(row, "changed").int32(), Int32(value)
                        )
    var source = DataFrame(
        [
            raw.column("record"),
            Series("keep", Column[Bool]([True, True, False, False])),
        ]
    )
    var chosen = source.select_exprs(
        [
            when(col("keep"))
            .then(_identity_field() + 1)
            .otherwise(lit(Int64(0)))
            .alias("chosen")
        ],
        batch_size=1,
    )
    assert_false(Bool(chosen.column("chosen")._field_metadata))
    assert_equal(chosen.item(0, "chosen").int64(), Int64(8))
    assert_true(chosen.item(1, "chosen").is_null())
    assert_equal(chosen.item(2, "chosen").int64(), Int64(0))
    assert_equal(chosen.item(3, "chosen").int64(), Int64(0))
    # The _full window-input accumulator also follows identity field chains.
    var running = raw.select_exprs(
        [_identity_field().cum_sum().alias("running")], batch_size=1
    )
    assert_false(Bool(running.column("running")._field_metadata))
    assert_equal(running.item(0, "running").int64(), Int64(7))
    assert_true(running.item(1, "running").is_null())
    assert_true(running.item(2, "running").is_null())
    assert_equal(running.item(3, "running").int64(), Int64(20))
    assert_equal(child.int64()._get(1), Int64.MAX)
    assert_false(child.int64()._valid(2))


def test_struct_field_identity_cast_worker_and_zero_row_packed_siblings() raises:
    var frame = _source()
    var child = frame.column("record").struct_column().field("value")
    var bound = bind(_identity_field(), frame._columns)
    var states = List[Series]()
    for dtype in bound.dtypes:
        states.append(Series.full_null("", dtype, 0))
    for end in [1, 4]:
        var job = _RowsJob[8](
            bound,
            frame._columns,
            states,
            len(bound.expr._nodes) - 1,
            1,
            end,
            1,
            False,
        )
        job.run()
        var actual = job^.into_result()
        _owner(actual, child)
        assert_equal(len(actual), end - 1)
        if end == 4:
            assert_false(actual.int64()._valid(0))
            assert_false(actual.int64()._valid(1))
            assert_equal(actual.int64()._get(2), Int64(13))
    var source = DataFrame([child.renamed("seed")])
    var expression = (
        as_struct(
            [
                col("seed").alias("value"),
                lit(Int64(128)).cast(DataType.INT8).alias("checked"),
            ],
            "record",
        )
        .field("value")
        .cast(DataType.INT64)
        .cast(DataType.INT64)
        .alias("out")
    )
    var empty = source.clear().select_exprs([expression.copy()], batch_size=1)
    assert_equal(empty.height(), 0)
    _owner(empty.column("out"), child)
    with assert_raises(contains="cast from int64 to int8 failed"):
        _ = source.select_exprs([expression.copy()], batch_size=1)
