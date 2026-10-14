"""Narrow primitive source List GET metadata accumulator contracts."""
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, DataType, ListColumn, Series, col
from dataframe.binding import bind
from dataframe.execution import _RowsJob
from tests.support.coverage_generators import offset_layout, chunk_layout


def _source() raises -> DataFrame:
    var child = Series("item", Column[Int64]([7, 8, Int64.MAX, 13, 17]))
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    child._field_metadata = ArcPointer(metadata^)
    var offsets: List[Int64] = [0, 2, 3, 3, 5]
    var parent: List[UInt8] = [13]
    return DataFrame([Series("xs", ListColumn(offsets^, child^, parent^))])


def test_source_list_get_primitive_metadata_seed_empty_batched_identity_and_workers() raises:
    var original = _source()
    var owner = (
        original.column("xs").list_column().child()._field_metadata.value()
    )
    var direct = col("xs").list().get(0)
    var identity = direct.cast(DataType.INT64).cast(DataType.INT64)
    for offset in [0, 1]:
        var frame = original.slice(offset, 4 - offset)
        for empty in [False, True]:
            var source = frame.clear() if empty else frame.copy()
            for batch in [1, 64]:
                var result = source.select_exprs(
                    [
                        direct.alias("direct"),
                        identity.alias("identity"),
                        (direct + 1).alias("fresh"),
                        direct.cast(DataType.INT32).alias("changed"),
                    ],
                    batch_size=batch,
                )
                assert_equal(result.height(), 0 if empty else 4 - offset)
                for name in ["direct", "identity"]:
                    assert_true(
                        result.column(name)._field_metadata.value().ptr()
                        == owner.ptr()
                    )
                for name in ["fresh", "changed"]:
                    assert_false(Bool(result.column(name)._field_metadata))
                for row in range(result.height()):
                    var logical = row + offset
                    var missing = logical == 1 or logical == 2
                    assert_equal(result.item(row, "direct").is_null(), missing)
                    assert_equal(
                        result.item(row, "identity").is_null(), missing
                    )
                    if not missing:
                        assert_equal(
                            result.item(row, "direct").int64(),
                            Int64(7 if logical == 0 else 13),
                        )
    var bound = bind(identity, original._columns)
    var states = List[Series]()
    for dtype in bound.dtypes:
        states.append(Series.full_null("", dtype, 0))
    for end in [1, 4]:
        var job = _RowsJob[8](
            bound,
            original._columns,
            states,
            len(bound.expr._nodes) - 1,
            1,
            end,
            1,
            False,
        )
        job.run()
        var result = job^.into_result()
        assert_equal(len(result), end - 1)
        assert_true(result._field_metadata.value().ptr() == owner.ptr())
    var running = original.select_exprs(
        [identity.cum_sum().alias("sum")], batch_size=1
    )
    assert_false(Bool(running.column("sum")._field_metadata))
    assert_equal(running.item(0, "sum").int64(), Int64(7))
    assert_true(running.item(1, "sum").is_null())
    assert_true(running.item(2, "sum").is_null())
    assert_equal(running.item(3, "sum").int64(), Int64(20))


def test_source_list_null_parent_nonempty_span_cpu_explode_current_contract() raises:
    # Literal CPU witness only: documents actual current storage/operator
    # behavior. It does not enable or repair any GPU EXPLODE implementation.
    var offsets: List[Int64] = [0, 2, 2, 3]
    var bits: List[UInt8] = [6]
    var source = DataFrame(
        [
            Series(
                "xs",
                ListColumn(
                    offsets^, Series("item", Column[Int64]([7, 8, 9])), bits^
                ),
            ),
            Series("id", Column[Int64]([0, 1, 2])),
        ]
    )
    var fields = source.select_exprs(
        [
            col("xs").list().len().alias("len"),
            col("xs").list().get(0).alias("get"),
        ]
    )
    assert_true(fields.item(0, "len").is_null())
    assert_true(fields.item(0, "get").is_null())
    assert_equal(fields.item(1, "len").int64(), Int64(0))
    assert_true(fields.item(1, "get").is_null())
    assert_equal(fields.item(2, "get").int64(), Int64(9))
    for variant in range(3):
        var frame = (
            source.copy() if variant
            == 0 else offset_layout(source) if variant
            == 1 else chunk_layout(source)
        )
        var exploded = frame.explode("xs")
        assert_equal(exploded.height(), 4)
        assert_equal(
            exploded.column("id").int64().to_list(),
            [Int64(0), Int64(0), Int64(1), Int64(2)],
        )
        assert_equal(exploded.item(0, "xs").int64(), Int64(7))
        assert_equal(exploded.item(1, "xs").int64(), Int64(8))
        assert_true(exploded.item(2, "xs").is_null())
        assert_equal(exploded.item(3, "xs").int64(), Int64(9))
        for optimize in [False, True]:
            for streaming in [False, True]:
                for batch in [1, 64]:
                    var lazy = (
                        frame.lazy()
                        .explode("xs")
                        .collect(
                            engine="cpu",
                            optimize=optimize,
                            streaming=streaming,
                            batch_size=batch,
                        )
                    )
                    assert_true(lazy.equals(exploded))
