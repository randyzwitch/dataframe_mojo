"""Opaque schema agreement and parent-offset import across worker dispatch."""
from std.testing import assert_equal, assert_true, assert_raises
from std.sys import CompilationTarget
from dataframe.io.arrow import _ARROW_IMPORT_MIN_CELLS, _arrow_import_workers
from dataframe import (
    ArrowArray,
    ArrowSchema,
    Column,
    DataFrame,
    DataType,
    Series,
    col,
    concat,
    export_arrow_series,
    import_arrow,
    import_arrow_series,
)
from tests.support.arrow_foreign import (
    ForeignArrow,
    integers,
    text_bytes,
    packed_metadata,
    read_signed,
)
from tests.support.lifecycle_stream import at, array_release, schema_release
from tests.support.lifecycle_jobs import WorkerSetting


def edge_metadata(
    reverse: Bool = False, different: Bool = False
) -> List[UInt8]:
    var keys: List[List[UInt8]] = [
        text_bytes("ARROW:extension:name"),
        text_bytes("ARROW:extension:metadata"),
        [255, 0, 107],
        text_bytes("empty"),
    ]
    var values: List[List[UInt8]] = [
        text_bytes("external.opaque"),
        [0, 254, 128, 255],
        [128, 255, 0],
        [],
    ]
    if different:
        values[1][1] = 253
    if reverse:
        var reordered_keys = List[List[UInt8]]()
        var reordered_values = List[List[UInt8]]()
        for row in range(len(keys) - 1, -1, -1):
            reordered_keys.append(keys[row].copy())
            reordered_values.append(values[row].copy())
        return packed_metadata(reordered_keys^, reordered_values^)
    return packed_metadata(keys^, values^)


def opaque_series(
    reverse: Bool = False, different: Bool = False
) raises -> Series:
    var producer = ForeignArrow(1)
    producer.nodes[0].describe("l", "x", 4, 0, 1)
    producer.nodes[0].buffer([13])
    producer.nodes[0].buffer(integers([3, 999, 1, 2], 8))
    producer.nodes[0].metadata = edge_metadata(reverse, different)
    producer.seal()
    var result = import_arrow_series(
        producer.nodes[0].array, producer.nodes[0].schema
    )
    producer.assert_released()
    producer.poison()
    return result^


def assert_opaque(
    series: Series, rows: List[Int], check_cells: Bool = True
) raises:
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(series, array, schema)
    var expected = edge_metadata()
    assert_true(schema.metadata != 0)
    # Metadata maps may reorder pairs; inspect the foreign length-prefixed map
    # rather than depending on dataframe's metadata comparison implementation.
    var seen: List[Bool] = [False, False, False, False]
    var position = 4
    assert_equal(read_signed(schema.metadata, 0, 4), Int128(4))
    for pair in range(4):
        var key_size = Int(read_signed(schema.metadata + position, 0, 4))
        position += 4
        var key = List[UInt8]()
        for byte in range(key_size):
            key.append(
                at[UInt8](schema.metadata + position)[unsafe_offset=byte]
            )
        position += key_size
        var value_size = Int(read_signed(schema.metadata + position, 0, 4))
        position += 4
        var value = List[UInt8]()
        for byte in range(value_size):
            value.append(
                at[UInt8](schema.metadata + position)[unsafe_offset=byte]
            )
        position += value_size
        var keys: List[List[UInt8]] = [
            text_bytes("ARROW:extension:name"),
            text_bytes("ARROW:extension:metadata"),
            [255, 0, 107],
            text_bytes("empty"),
        ]
        var values: List[List[UInt8]] = [
            text_bytes("external.opaque"),
            [0, 254, 128, 255],
            [128, 255, 0],
            [],
        ]
        var found = False
        for candidate in range(4):
            if key == keys[candidate]:
                assert_true(not seen[candidate])
                assert_equal(value, values[candidate])
                seen[candidate] = True
                found = True
        assert_true(found)
    array_release(array)
    schema_release(schema)
    if not check_cells:
        return
    var payload: List[Int64] = [3, 999, 1, 2]
    assert_equal(len(series), len(rows))
    for row in range(len(rows)):
        assert_equal(
            series.get(row).is_null(), rows[row] < 0 or rows[row] % 4 == 1
        )
        if rows[row] >= 0 and rows[row] % 4 != 1:
            assert_equal(series.get(row).int64(), payload[rows[row] % 4])


def test_arrow_opaque_metadata_agrees_by_binary_pairs_across_row_operations_and_rejects_conflicts() raises:
    var source = opaque_series()
    var reordered = opaque_series(True)
    assert_opaque(source, [0, 1, 2, 3])
    assert_opaque(source.renamed("renamed"), [0, 1, 2, 3])
    assert_opaque(source.slice(1, 3), [1, 2, 3])
    assert_opaque(source.take([3, 1, 0, 3]), [3, 1, 0, 3])
    assert_opaque(source.take_or_null([2, -1, 0]), [2, -1, 0])
    var chunked = Series._from_chunks([source.copy(), reordered.copy()])
    assert_opaque(chunked, [0, 1, 2, 3, 0, 1, 2, 3])
    assert_opaque(chunked.rechunk(), [0, 1, 2, 3, 0, 1, 2, 3])
    var frame = DataFrame([source.copy()])
    assert_opaque(frame.select(col("x").alias("y")).column("y"), [0, 1, 2, 3])
    assert_opaque(frame.filter(col("x") > 1).column("x"), [0, 3])
    assert_opaque(frame.sort("x", nulls_last=True).column("x"), [2, 3, 0, 1])
    assert_opaque(
        concat([frame.copy(), DataFrame([reordered.copy()])]).column("x"),
        [0, 1, 2, 3, 0, 1, 2, 3],
    )
    assert_opaque(
        concat(
            [frame.copy(), DataFrame([Series("y", Column[Int64]([7]))])],
            how="diagonal",
        ).column("x"),
        [0, 1, 2, 3, -1],
    )
    for conflict in [
        opaque_series(different=True),
        Series("x", Column[Int64]([3, 999, 1, 2])),
    ]:
        with assert_raises(contains="Conflicting Arrow field metadata"):
            _ = concat([frame.copy(), DataFrame([conflict.copy()])])
    # Changing the storage dtype invalidates unknown extension metadata.
    var retagged = source.with_dtype(DataType.DATE)
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(retagged, array, schema)
    assert_equal(schema.metadata, 0)
    array_release(array)
    schema_release(schema)


def test_arrow_boolean_and_string_metadata_survive_actual_parallel_gathers() raises:
    from dataframe.gather import take_parallel

    for kind in range(2):
        var producer = ForeignArrow(1)
        producer.nodes[0].describe("b" if kind == 0 else "U", "x", 4, 0, 1)
        producer.nodes[0].buffer([13])
        if kind == 0:
            producer.nodes[0].buffer([11])
        else:
            producer.nodes[0].buffer(integers([0, 1, 7, 8, 9], 8))
            producer.nodes[0].buffer(text_bytes("aPOISONbc"))
        producer.nodes[0].metadata = edge_metadata()
        producer.seal()
        var source = import_arrow_series(
            producer.nodes[0].array, producer.nodes[0].schema
        )
        producer.assert_released()
        producer.poison()
        var rows = List[Int]()
        for row in range(131073):
            rows.append((row * 3) % 4)
        var setting = WorkerSetting(4)
        var actual = take_parallel([source.copy()], rows.copy(), 4)[0].copy()
        _ = setting^
        assert_opaque(actual, rows, False)
        assert_equal(len(actual), len(rows))
        for row in range(len(rows)):
            var index = rows[row]
            assert_equal(actual.get(row).is_null(), index == 1)
            if index != 1:
                if kind == 0:
                    assert_equal(actual.get(row).bool(), index != 2)
                else:
                    var words: List[String] = ["a", "POISON", "b", "c"]
                    assert_equal(actual.get(row).string(), words[index])
        assert_opaque(
            Series._from_chunks([source.copy(), source.copy()]).rechunk(),
            [],
            False,
        )


def test_arrow_dispatch_sized_parent_slices_preserve_order_nulls_and_recursive_error_release() raises:
    comptime N = 1000005
    for workers in [1, 4]:
        var setting = WorkerSetting(workers)
        assert_true((N - 5) * 2 >= _ARROW_IMPORT_MIN_CELLS)
        assert_equal(
            _arrow_import_workers(N - 5, 2),
            1 if workers == 1 or CompilationTarget.is_macos() else 2,
        )
        for fail in range(2):
            var producer = ForeignArrow(3)
            producer.nodes[0].describe("+s", "", N - 5, 3)
            producer.nodes[0].buffer([])
            producer.nodes[0].children = [1, 2]
            var values = List[Int128](capacity=N)
            var other = List[Int128](capacity=N)
            var validity = List[UInt8](length=(N + 7) // 8, fill=0)
            for row in range(N):
                values.append(Int128(row - 17))
                other.append(Int128(3 * row + 1))
                if row % 7 != 0:
                    validity[row // 8] |= UInt8(1 << (row % 8))
            producer.nodes[1].describe("l", "a", N, 0, -1)
            producer.nodes[1].buffer(validity^)
            producer.nodes[1].buffer(integers(values^, 8, 1), 1)
            producer.nodes[2].describe("e" if fail else "l", "b", N)
            producer.nodes[2].buffer([])
            producer.nodes[2].buffer(integers(other^, 8, 1), 1)
            producer.seal()
            if fail:
                with assert_raises(contains="Unsupported Arrow format 'e'"):
                    _ = import_arrow(
                        producer.nodes[0].array, producer.nodes[0].schema
                    )
                producer.assert_released()
            else:
                var result = import_arrow(
                    producer.nodes[0].array, producer.nodes[0].schema
                )
                producer.assert_released()
                producer.poison()
                assert_equal(result.height(), N - 5)
                for row in range(result.height()):
                    var original = row + 3
                    assert_equal(
                        result.item(row, "a").is_null(), original % 7 == 0
                    )
                    if original % 7 != 0:
                        assert_equal(
                            result.item(row, "a").int64(), Int64(original - 17)
                        )
                    assert_equal(
                        result.item(row, "b").int64(), Int64(3 * original + 1)
                    )
        _ = setting^
