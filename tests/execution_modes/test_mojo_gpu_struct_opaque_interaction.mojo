"""Source Struct identity fields with ordinary opaque byte columns."""
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    BoolColumn,
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    StructColumn,
    col,
    from_wkb,
    lit,
    when,
)
from dataframe.metal import MetalRuntime
from dataframe.accelerator.source_structs import lower_source_structs
from tests.support.coverage_generators import offset_layout, chunk_layout


def _owner(tag: UInt8) -> ArcPointer[List[UInt8]]:
    var bytes: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, tag]
    return ArcPointer(bytes^)


def _point(kind: Int) -> List[UInt8]:
    var result: List[UInt8] = [1, 1, 0, 0, 0]
    if kind == 1:
        result = [0, 0, 0, 0, 1]
    result.extend(List[UInt8](length=16, fill=0))
    if kind == 2:
        result[12] = 128  # Little-endian negative zero X, distinct raw WKB.
    return result^


def _source() raises -> DataFrame:
    var child = Series(
        "n",
        Column[Int64](
            [5, Int64.MAX, Int64.MAX, 9, Int64.MAX, Int64.MAX, 11],
            [True, True, False, True, True, True, True],
        ),
    )
    child._field_metadata = _owner(110)
    # Parent-null rows1/5 hide valid MAX child poison; row2 is child-null.
    var parent: List[UInt8] = [93]
    var record = Series("record", StructColumn([child^], parent^))
    record._field_metadata = _owner(114)
    var raw: List[List[UInt8]] = [
        [255],
        [195, 40],
        [0],
        [],
        [255, 0],
        [237, 160, 128],
        [97, 0, 98],
    ]
    var binary = Series.binary(
        "b", raw^, [True, True, True, False, True, True, True]
    )
    binary._field_metadata = _owner(98)
    var wkb: List[List[UInt8]] = [
        _point(0),
        _point(1),
        _point(2),
        [],
        _point(1),
        _point(2),
        _point(0),
    ]
    var geometry = from_wkb(
        Series.binary("g", wkb^, [True, True, True, False, True, True, True]),
        '{"crs":"EPSG:4326"}',
    )
    geometry._field_metadata = _owner(103)
    return DataFrame(
        [
            record^,
            binary^,
            geometry^,
            Series(
                "use", BoolColumn([True, True, True, False, False, True, True])
            ),
        ]
    )


def _expressions() raises -> List[Expr]:
    var field = (
        col("record").field("n").cast(DataType.INT64).cast(DataType.INT64)
    )
    return [
        field.copy().alias("identity"),
        when(col("use"))
        .then(field + 1)
        .otherwise(lit(Int64(0)))
        .alias("masked"),
        col("b").alias("bytes"),
        col("g").alias("geometry"),
        (col("b") == lit("\x00")).alias("zero"),
    ]


def _raw_equal(actual: Series, source: Series) raises:
    assert_equal(actual.dtype(), source.dtype())
    assert_equal(len(actual), len(source))
    var a = actual.rechunk()
    var b = source.rechunk()
    ref left = a._data[StringColumn]
    ref right = b._data[StringColumn]
    for row in range(len(a)):
        assert_equal(left._valid(row), right._valid(row))
        if left._valid(row):
            assert_equal(len(left._row_bytes(row)), len(right._row_bytes(row)))
            for at in range(len(left._row_bytes(row))):
                assert_equal(
                    left._row_bytes(row)[at], right._row_bytes(row)[at]
                )


def _metadata_equal(actual: Series, expected: Series) raises:
    assert_true(Bool(actual._field_metadata))
    assert_true(
        actual._field_metadata.value().ptr()
        == expected._field_metadata.value().ptr()
    )
    assert_equal(
        actual._field_metadata.value()[], expected._field_metadata.value()[]
    )


def _literal_result(
    actual: DataFrame, original: DataFrame, start: Int, rows: Int
) raises:
    assert_equal(actual.height(), rows)
    assert_equal(
        actual.columns(), ["identity", "masked", "bytes", "geometry", "zero"]
    )
    var child = original.column("record").struct_column().field("n")
    _metadata_equal(actual.column("identity"), child)
    _metadata_equal(actual.column("bytes"), original.column("b"))
    _metadata_equal(actual.column("geometry"), original.column("g"))
    assert_false(Bool(actual.column("masked")._field_metadata))
    assert_false(Bool(actual.column("zero")._field_metadata))
    _raw_equal(actual.column("bytes"), original.column("b").slice(start, rows))
    _raw_equal(
        actual.column("geometry"), original.column("g").slice(start, rows)
    )
    var identity: List[Int64] = [5, 0, 0, 9, Int64.MAX, 0, 11]
    var masked: List[Int64] = [6, 0, 0, 0, 0, 0, 12]
    for row in range(rows):
        var source_row = start + row
        var missing = source_row == 1 or source_row == 2 or source_row == 5
        assert_equal(actual.item(row, "identity").is_null(), missing)
        assert_equal(actual.item(row, "masked").is_null(), missing)
        if not missing:
            assert_equal(
                actual.item(row, "identity").int64(), identity[source_row]
            )
            assert_equal(actual.item(row, "masked").int64(), masked[source_row])
        assert_equal(actual.item(row, "zero").is_null(), source_row == 3)
        if source_row != 3:
            assert_equal(actual.item(row, "zero").bool(), source_row == 2)


def test_mojo_gpu_struct_identity_and_opaque_transport_share_owners_masks_and_budget() raises:
    var original = _source()
    var expressions = _expressions()
    for sliced in [False, True]:
        var source = original.slice(1, 5) if sliced else original.copy()
        for variant in range(3):
            var frame = (
                source.copy() if variant
                == 0 else offset_layout(source) if variant
                == 1 else chunk_layout(source)
            )
            var query = frame.lazy().select_exprs(expressions)
            var measured = query.profile(
                accelerator=MetalRuntime(), engine="accel"
            )
            _literal_result(
                measured[0], original, 1 if sliced else 0, 5 if sliced else 7
            )
            for batch in [1, 64]:
                var cpu = frame.select_exprs(expressions, batch_size=batch)
                _literal_result(
                    cpu, original, 1 if sliced else 0, 5 if sliced else 7
                )
                assert_true(measured[0].equals(cpu))
            assert_true(measured[1].item(0, "kernel_launches").int64() > 0)
            assert_true(measured[1].item(0, "opaque_staging_bytes").int64() > 0)
            assert_true(
                measured[1].item(0, "opaque_restoration_bytes").int64() > 0
            )
            var flattened = lower_source_structs(query)
            assert_equal(
                measured[1].item(0, "source_struct_staging_bytes").int64(),
                Int64(flattened.staging_bytes),
            )
            if variant == 2 and not sliced:
                # Independent composition check: the outer Struct reservation
                # surrounds the complete opaque inner peak, not its first region.
                assert_true(flattened.staging_bytes > 0)
                var inner = flattened.query.profile(
                    accelerator=MetalRuntime(), engine="accel"
                )
                var peak = Int(
                    measured[1].item(0, "query_peak_requested_bytes").int64()
                )
                assert_equal(
                    peak,
                    flattened.staging_bytes
                    + Int(
                        inner[1].item(0, "query_peak_requested_bytes").int64()
                    ),
                )
                var exact = query.collect(
                    accelerator=MetalRuntime(memory_limit_bytes=peak),
                    engine="accel",
                )
                _literal_result(exact, original, 0, 7)
                var limited = MetalRuntime(memory_limit_bytes=peak - 1)
                with assert_raises(contains="budget"):
                    _ = query.collect(accelerator=limited, engine="accel")
                var safe = (
                    DataFrame([Series("x", Column[Int64]([7]))])
                    .lazy()
                    .select(col("x"))
                )
                assert_equal(
                    safe.collect(accelerator=limited, engine="accel")
                    .item()
                    .int64(),
                    Int64(7),
                )
    # Faulting the same visible MAX confirms the safe result above relies on
    # WHEN/parent masking, rather than silently dropping checked row errors.
    var fault = original.lazy().select_exprs(
        [
            (
                col("record")
                .field("n")
                .cast(DataType.INT64)
                .cast(DataType.INT64)
                + 1
            ).alias("bad"),
            col("b"),
            col("g"),
        ]
    )
    with assert_raises(contains="overflow"):
        _ = fault.collect(engine="cpu", streaming=False)
    with assert_raises(contains="overflow"):
        _ = fault.collect(accelerator=MetalRuntime(), engine="accel")
