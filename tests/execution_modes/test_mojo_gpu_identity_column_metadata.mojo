"""CPU and GPU share exact primitive identity-cast metadata provenance."""
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_false
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StructColumn,
    col,
    lit,
)
from dataframe.metal import MetalRuntime
from dataframe.accelerator.gpu.runtime import _preflight
from tests.public_contracts.test_identity_column_metadata import (
    _source,
    _owner,
    _expressions,
    _check,
)
from tests.support.coverage_generators import offset_layout, chunk_layout


def test_mojo_gpu_identity_column_metadata_logical_views_empty_and_budget() raises:
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
                var query = source.lazy().select_exprs(
                    _expressions(source, strict)
                )
                var plan = _preflight(query)
                for at in range(len(plan.outputs)):
                    assert_true(Bool(plan.outputs[at].field_metadata))
                    assert_equal(
                        plan.outputs[at].field_metadata.value()[],
                        source._columns[at]._field_metadata.value()[],
                    )
                var actual = query.collect(
                    accelerator=MetalRuntime(device_id=999999 if empty else 0),
                    optimize=False,
                )
                _check(actual, source)
                _check(
                    query.collect(
                        engine="cpu", streaming=False, optimize=False
                    ),
                    source,
                )
    var safe = raw.lazy().select_exprs(_expressions(raw, True))
    var measured = safe.profile(accelerator=MetalRuntime())
    var peak = Int(measured[1].item(0, "peak_requested_bytes").int64())
    _check(safe.collect(accelerator=MetalRuntime(memory_limit_bytes=peak)), raw)


def test_mojo_gpu_identity_column_metadata_struct_transport_and_fresh_results() raises:
    var raw = _source()
    var child = Series("value", Column[Int64]([-7, Int64.MAX, 11]))
    child._field_metadata = raw.column("i")._field_metadata
    var parent_bits: List[UInt8] = [5]
    var fields: List[Series] = [child.copy()]
    var source = raw.with_column(
        Series("record", StructColumn(fields^, parent_bits^))
    )
    # Source-Struct normalization must not overwrite ordinary numeric identity
    # casts with a computed-fresh prototype merely because a Struct is present.
    var expressions = _expressions(raw, True)
    expressions.append(
        col("record").field("value").cast(DataType.INT64).alias("field")
    )
    expressions.append(
        (col("record").field("value") + lit(Int64(1)))
        .cast(DataType.INT64)
        .alias("computed")
    )
    var query = source.lazy().select_exprs(expressions^)
    var actual = query.collect(accelerator=MetalRuntime(), optimize=False)
    for at in range(raw.width()):
        _owner(actual._columns[at], raw._columns[at])
        assert_true(actual._columns[at].equals(raw._columns[at]))
    _owner(actual.column("field"), child)
    assert_equal(actual.item(0, "field").int64(), Int64(-7))
    assert_true(actual.item(1, "field").is_null())
    assert_equal(actual.item(2, "field").int64(), Int64(11))
    assert_false(Bool(actual.column("computed")._field_metadata))
    assert_equal(actual.item(0, "computed").int64(), Int64(-6))
    assert_true(actual.item(1, "computed").is_null())
    assert_equal(actual.item(2, "computed").int64(), Int64(12))
    assert_true(
        actual.equals(
            query.collect(engine="cpu", streaming=False, optimize=False)
        )
    )
    # Actual dtype-changing casts and computations still create fresh fields.
    var changed = raw.lazy().select_exprs(
        [
            col("i")
            .cast(DataType.FLOAT64)
            .cast(DataType.INT64)
            .alias("roundtrip"),
            (col("i") + lit(Int64(0))).cast(DataType.INT64).alias("computed"),
            col("date").cast(DataType.datetime("ms")).alias("changed"),
        ]
    )
    var fresh = changed.collect(accelerator=MetalRuntime(), optimize=False)
    for column in fresh._columns:
        assert_false(Bool(column._field_metadata))
    assert_equal(fresh.item(0, "roundtrip").int64(), Int64(-7))
    assert_equal(fresh.item(2, "changed").to_physical(), Int64(86400000))
