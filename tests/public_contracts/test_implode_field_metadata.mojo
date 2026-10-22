"""IMPLODE accumulators retain prepared child metadata and fresh computations."""
# contract-group: public.apple_parity
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.binding import bind
from dataframe.execution import evaluate, _RowsJob, _full
from dataframe.expr import OVER
from tests.support.coverage_generators import offset_layout, chunk_layout


def _source() raises -> DataFrame:
    var value = Series(
        "value",
        Column[Int64]([7, Int64.MAX, -3, 11], [True, False, True, True]),
    )
    # Packed Arrow application metadata: tag -> raw.
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
    value._field_metadata = ArcPointer(metadata^)
    return DataFrame([value^, Series("key", Column[Int32]([0, 1, 0, 1]))])


def _owner(actual: Series, source: Series) raises:
    assert_true(Bool(actual._field_metadata))
    assert_equal(
        actual._field_metadata.value()[], source._field_metadata.value()[]
    )
    assert_true(
        actual._field_metadata.value().ptr()
        == source._field_metadata.value().ptr()
    )


def _values(actual: Series, values: List[Int64], valid: List[Bool]) raises:
    assert_equal(actual.dtype(), DataType.INT64)
    assert_equal(len(actual), len(values))
    for row in range(len(values)):
        assert_equal(actual.int64()._valid(row), valid[row])
        if valid[row]:
            assert_equal(actual.int64()._get(row), values[row])


def test_implode_field_metadata_scalar_empty_batches_identity_and_computed() raises:
    var original = _source()
    var owner = original.column("value")
    for layout in range(3):
        var frame = (
            original.copy() if layout
            == 0 else offset_layout(original) if layout
            == 1 else chunk_layout(original)
        )
        for empty in [False, True]:
            var input = frame.clear() if empty else frame.copy()
            for batch in [1, 64]:
                for identity in [False, True]:
                    var operand = (
                        col("value")
                        .cast(DataType.INT64)
                        .cast(DataType.INT64) if identity else col("value")
                    )
                    var expression = operand.implode().alias("list")
                    var result = input.select(expression, batch_size=batch)
                    assert_equal(result.height(), 1)
                    assert_equal(
                        result.column("list").dtype(),
                        DataType.list(DataType.INT64),
                    )
                    assert_false(result.column("list").list_column().is_null(0))
                    assert_false(Bool(result.column("list")._field_metadata))
                    var child = result.item().list()
                    _owner(child, owner)
                    if empty:
                        assert_equal(len(child), 0)
                    else:
                        _values(
                            child, [7, 0, -3, 11], [True, False, True, True]
                        )
                var computed = input.select(
                    (col("value") + lit(Int64(1))).implode().alias("computed"),
                    batch_size=batch,
                )
                var computed_child = computed.item().list()
                assert_false(Bool(computed_child._field_metadata))
                if empty:
                    assert_equal(len(computed_child), 0)
                else:
                    _values(
                        computed_child,
                        [8, 0, -2, 12],
                        [True, False, True, True],
                    )
    # Metadata and raw null payload owners remain unchanged by seed append.
    assert_equal(owner.int64()._get(1), Int64.MAX)
    _owner(original.column("value"), owner)


def test_implode_field_metadata_grouped_over_full_and_worker_accumulators() raises:
    var frame = _source()
    var owner = frame.column("value")
    var expression = col("value").cast(DataType.INT64).implode().alias("list")
    var grouped = frame.group_by("key", maintain_order=True).agg(
        [expression.copy()]
    )
    assert_equal(grouped.height(), 2)
    assert_equal(grouped.item(0, "key").int32(), Int32(0))
    assert_equal(grouped.item(1, "key").int32(), Int32(1))
    _values(grouped.column("list").list_column().row(0), [7, -3], [True, True])
    _values(grouped.column("list").list_column().row(1), [0, 11], [False, True])
    _owner(grouped.column("list").list_column().child(), owner)
    var empty_grouped = (
        frame.clear()
        .group_by("key", maintain_order=True)
        .agg([expression.copy()])
    )
    assert_equal(empty_grouped.height(), 0)
    _owner(empty_grouped.column("list").list_column().child(), owner)
    var bound = bind(expression, frame._columns)
    var direct = evaluate(
        bound,
        frame._columns,
        4,
        batch_size=1,
        grouped=True,
        groups=[0, 1, 0, 1],
        group_count=2,
    )
    _owner(direct.list_column().child(), owner)
    _values(direct.list_column().row(0), [7, -3], [True, True])
    var over = col("value").implode().over("key").alias("lists")
    var aligned = frame.select(over, batch_size=1).column("lists")
    _owner(aligned.list_column().child(), owner)
    for row in range(4):
        if row % 2 == 0:
            _values(aligned.list_column().row(row), [7, -3], [True, True])
        else:
            _values(aligned.list_column().row(row), [0, 11], [False, True])
    # Exercise _full and worker-owned accumulators with an actual prepared
    # aligned state, independent of core count and thread environment.
    bound = bind(over, frame._columns)
    var states = List[Series]()
    for dtype in bound.dtypes:
        states.append(Series.full_null("", dtype, 0))
    for at in range(len(bound.expr._nodes)):
        if bound.expr._nodes[at].op == OVER:
            states[at] = aligned.renamed("")
    var full = _full[8](
        bound, frame._columns, states, len(bound.expr._nodes) - 1, 4, 1, False
    )
    _owner(full.list_column().child(), owner)
    assert_true(full.equals(aligned.renamed("")))
    var pieces = List[Series]()
    for start in [0, 2]:
        var job = _RowsJob[8](
            bound,
            frame._columns,
            states,
            len(bound.expr._nodes) - 1,
            start,
            start + 2,
            1,
            False,
        )
        job.run()
        var part = job^.into_result()
        _owner(part.list_column().child(), owner)
        pieces.append(part^)
    var combined = Series._from_chunks(pieces)
    _owner(combined.rechunk().list_column().child(), owner)
    assert_true(combined.equals(aligned.renamed("")))


def test_implode_field_metadata_seed_does_not_evaluate_inactive_payloads() raises:
    var frame = _source()
    for empty in [False, True]:
        var input = frame.clear() if empty else frame.copy()
        for selected in [False, True]:
            var expression = (
                when(lit(selected))
                .then(lit(Int64(128)).cast(DataType.INT8).cast(DataType.INT64))
                .otherwise(col("value"))
                .implode()
                .alias("list")
            )
            if selected and not empty:
                with assert_raises(contains="cast from int64 to int8 failed"):
                    _ = input.select(expression, batch_size=1)
            else:
                var result = input.select(expression, batch_size=1)
                var child = result.item().list()
                assert_false(Bool(child._field_metadata))
                if empty:
                    assert_equal(len(child), 0)
                else:
                    _values(child, [7, 0, -3, 11], [True, False, True, True])
