"""Foreign fixed-width payload ownership, alias isolation and Arrow release."""
# contract-group: storage.borrowed_arrow
from std.ffi import external_call
from std.collections import Optional
from std.memory import ArcPointer, Pointer
from std.testing import assert_equal, assert_true
from dataframe import (
    BoolColumn,
    StringColumn,
    ListColumn,
    StructColumn,
    Column,
    DataFrame,
    DataType,
    Series,
    ArrowArray,
    ArrowSchema,
    import_arrow,
    import_arrow_series,
    export_arrow_series,
    export_arrow,
    col,
)
from dataframe.io.arrow import (
    _at,
    _read_c_string,
    _buffer,
    _export_frame,
    _export_series,
    _c_string,
    _release_imported,
    _import_arrow_with_pool,
)
from dataframe.io.ipc import _open_scan
from dataframe.io.parquet import _stream_next, _stream_schema, _release_stream
from dataframe.parallel import Pool
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.arrow_foreign import integers, ForeignArrow, text_bytes

from tests.support.typed_storage_reference import (
    ForeignPages,
    make_foreign_pages,
    assert_foreign_unmapped,
    assert_storage_series,
    positional_rows,
)
from tests.support.selection_types_reference import selection_column


def imported_after_producer_scope(counter: Int) raises -> Series:
    var source = Series("x", Column[Int64]([10, 20, 30, 40]))
    var array = ArrowArray()
    var schema = ArrowSchema()
    _export_series(source, array, schema, counter)
    var original = _buffer(array, 1)
    var imported = import_arrow_series(array, schema, copy=False)
    assert_equal(array.release, 0)
    assert_equal(schema.release, 0)
    assert_equal(_at[Int](counter)[], 0)
    assert_equal(Int(imported.int64().unsafe_values()), original)
    return imported.slice(1, 2)


def exercise_last_view(counter: Int) raises:
    var view = imported_after_producer_scope(counter)
    assert_equal(view.get(0).int64(), Int64(20))
    assert_equal(view.get(1).int64(), Int64(30))
    assert_equal(_at[Int](counter)[], 0)
    var grown = view.int64()
    grown._append_column(Column[Int64]([99]))
    assert_equal(
        grown.to_list(), List[Int64]([Int64(20), Int64(30), Int64(99)])
    )
    assert_equal(view.get(1).int64(), Int64(30))
    assert_equal(_at[Int](counter)[], 0)


def test_borrowed_arrow_last_view_releases_once_after_source_and_mutation_copy() raises:
    var counter = ArcPointer(Int(0))
    exercise_last_view(Int(counter.ptr()))
    assert_equal(counter[], 1)


def exercise_frame(counter: Int, copy: Bool) raises:
    var source = DataFrame(
        [
            Series("x", Column[Int64]([1, 2, 3], [True, False, True])),
            Series("y", Column[Float64]([1.5, 2.5, 3.5])),
        ]
    )
    var array = ArrowArray()
    var schema = ArrowSchema()
    _export_frame(source, array, schema, counter)
    ref child = _at[ArrowArray](_at[Int](array.children)[])[]
    var address = _buffer(child, 1)
    var frame = import_arrow(array, schema, copy=copy)
    assert_equal(Int(frame["x"].int64().unsafe_values()) == address, not copy)
    assert_equal(_at[Int](counter)[], 3 if copy else 0)
    assert_true(frame.item(1, "x").is_null())
    assert_equal(frame.item(2, "y").float64(), Float64(3.5))
    var selected = frame.filter(col("x") > 1)
    assert_equal(selected.item(0, "x").int64(), Int64(3))
    var output = ArrowArray()
    var output_schema = ArrowSchema()
    export_arrow_series(frame["x"], output, output_schema)
    if not copy:
        assert_equal(_buffer(output, 1), address)
    _release_imported(output, output_schema)


def test_borrowed_arrow_frame_default_copy_and_retained_export_obey_release_counts() raises:
    for copy in [False, True]:
        var counter = ArcPointer(Int(0))
        exercise_frame(Int(counter.ptr()), copy)
        assert_equal(counter[], 3)


def test_borrowed_arrow_mapped_payload_outlives_reader_and_keeps_buffer_address() raises:
    var stream = _open_scan(
        "tests/fixtures/ipc/core-uncompressed.arrow", ["x"], True, True, 0, 3
    )
    var array = ArrowArray()
    var schema = ArrowSchema()
    assert_equal(_stream_next(stream, array), Int32(0))
    assert_equal(_stream_schema(stream, schema), Int32(0))
    ref child = _at[ArrowArray](_at[Int](array.children)[])[]
    var address = _buffer(child, 1)
    var frame = import_arrow(array, schema, copy=False)
    _release_stream(stream)
    assert_equal(Int(frame["x"].int64().unsafe_values()), address)
    assert_equal(frame.item(2, "x").int64(), Int64(2))
    var tail = frame.slice(1, 2)
    assert_equal(Int(tail["x"].int64().unsafe_values()), address + 8)
    assert_equal(tail.item(1, "x").int64(), Int64(2))


def exercise_error(counter: Int, parallel: Bool) raises:
    var rows = 131072 if parallel else 4
    var source = DataFrame(
        [
            Series("x", Column[Int64](List[Int64](length=rows, fill=7))),
            Series("y", Column[Int64](List[Int64](length=rows, fill=8))),
        ]
    )
    var array = ArrowArray()
    var schema = ArrowSchema()
    _export_frame(source, array, schema, counter)
    var invalid = _c_string("invalid-foreign-format")
    _at[ArrowSchema](_at[Int](schema.children + 8)[])[].format = Int(
        invalid.unsafe_ptr()
    )
    var pool = Pool(2)
    var failed = False
    try:
        _ = _import_arrow_with_pool(array, schema, pool, copy=False)
    except error:
        assert_true("Unsupported Arrow format" in String(error))
        failed = True
    assert_true(failed)
    assert_equal(array.release, 0)
    assert_equal(schema.release, 0)
    _ = invalid^


def test_borrowed_arrow_partial_and_parallel_import_errors_release_parent_once() raises:
    var setting = WorkerSetting(2)
    for parallel in [False, True]:
        var counter = ArcPointer(Int(0))
        exercise_error(Int(counter.ptr()), parallel)
        assert_equal(counter[], 3)
    _ = setting^


def test_borrowed_arrow_fixed_width_temporal_decimal_and_slice_values_match_copied_import() raises:
    var path = "tests/fixtures/ipc/typed.arrow"
    var stream = _open_scan(path, List[String](), False, True, 0, -1)
    while True:
        var array = ArrowArray()
        var schema = ArrowSchema()
        assert_equal(_stream_next(stream, array), Int32(0))
        if array.release == 0:
            break
        assert_equal(_stream_schema(stream, schema), Int32(0))
        var imported = import_arrow(array, schema, copy=False)
        assert_equal(imported["i"].dtype(), DataType.INT8)
        assert_equal(
            imported["narrow_precision_wide_storage"].dtype(),
            DataType.decimal(9, 2, 128),
        )
        var exported = ArrowArray()
        var exported_schema = ArrowSchema()

        export_arrow(imported, exported, exported_schema)
        var copied = import_arrow(exported, exported_schema)
        assert_true(copied.equals(imported))
    _release_stream(stream)


def test_borrowed_arrow_unaligned_payload_uses_owned_copy_without_mutating_producer() raises:
    var counter = ArcPointer(Int(0))
    var source = Series("x", Column[Int64]([0, 0]))
    var array = ArrowArray()
    var schema = ArrowSchema()
    _export_series(source, array, schema, Int(counter.ptr()))
    var raw: List[UInt8] = [0]
    raw.extend(Span(integers([-7, 99], 8)))
    var address = Int(raw.unsafe_ptr()) + 1
    _at[Int](array.buffers)[unsafe_offset=1] = address
    var imported = import_arrow_series(array, schema, copy=False)
    assert_equal(imported.get(0).int64(), Int64(-7))
    assert_equal(imported.get(1).int64(), Int64(99))
    assert_true(Int(imported.int64().unsafe_values()) != address)
    assert_true(not imported.int64()._data.is_borrowed())
    assert_equal(counter[], 1)
    assert_equal(raw[1], UInt8(249))
    _ = raw^


def paged_bitmap_after_source_scope(
    mut foreign: ForeignPages, kind: Int, n: Int, offset: Int, mode: Int
) raises -> Series:
    var source = selection_column(kind, n, offset, mode)
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(source, array, schema)
    make_foreign_pages(array, schema, foreign)
    for i in range(len(foreign.pages)):
        assert_equal(
            external_call["mprotect", Int32](
                foreign.pages[i], foreign.lengths[i], Int32(1)
            ),
            Int32(0),
        )
    var values = _buffer(array, 1)
    var validity = _buffer(array, 0)
    var has_validity = validity != 0 and (
        array.null_count != 0 or offset % 8 != 0
    )
    var result = import_arrow_series(array, schema, copy=False)
    assert_equal(foreign.calls, 0)
    if kind == 10:
        var column = result.bool()
        assert_equal(Int(column.unsafe_values()), values + offset // 8)
        assert_equal(column.validity_offset(), offset % 8)
        if has_validity:
            assert_equal(Int(column.unsafe_validity()), validity + offset // 8)
            assert_true(column._bits.is_borrowed())
    else:
        var column = result.int64()
        assert_equal(Int(column.unsafe_values()), values + offset * 8)
        assert_equal(column.validity_offset(), offset % 8)
        if has_validity:
            assert_equal(Int(column.unsafe_validity()), validity + offset // 8)
            assert_true(column._bits.is_borrowed())
    assert_storage_series(result, kind, positional_rows(n, offset), mode)
    return result.slice(0, n)


def exercise_paged_bitmap(
    mut foreign: ForeignPages, kind: Int, n: Int, offset: Int, mode: Int
) raises:
    var view = paged_bitmap_after_source_scope(foreign, kind, n, offset, mode)
    assert_equal(foreign.calls, 0)
    assert_storage_series(view, kind, positional_rows(n, offset), mode)
    if kind == 10:
        var changed = view.bool()
        changed._reserve_rows(n + 2, 0)
        changed._append_column(BoolColumn([True, False], [False, True]))
        assert_equal(len(changed), n + 2)
        assert_true(changed.is_null(n))
        assert_equal(changed.value(n + 1), False)
    else:
        var changed = view.int64()
        changed._reserve_rows(n + 2, 0)
        changed._append_column(Column[Int64]([123, 456], [False, True]))
        assert_equal(len(changed), n + 2)
        assert_true(changed.is_null(n))
        assert_equal(changed.value(n + 1), Int64(456))
    assert_storage_series(view, kind, positional_rows(n, offset), mode)
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(view, array, schema)
    var copy = import_arrow_series(array, schema)
    assert_storage_series(copy, kind, positional_rows(n, offset), mode)
    assert_equal(foreign.calls, 0)
    _ = view^


def test_borrowed_arrow_bitmaps_and_booleans_share_readonly_pages_through_views_and_copy_on_write() raises:
    for kind in [3, 10]:
        for n in [1, 9, 65]:
            for offset in [0, 1, 7, 9, 17]:
                for mode in range(3):
                    var foreign = ForeignPages()
                    exercise_paged_bitmap(foreign, kind, n, offset, mode)
                    assert_foreign_unmapped(foreign)


def paged_layout_after_source_scope(
    mut foreign: ForeignPages, kind: Int, n: Int, offset: Int, mode: Int
) raises -> Series:
    var source = selection_column(kind, n, offset, mode)
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(source, array, schema)
    make_foreign_pages(array, schema, foreign)
    for i in range(len(foreign.pages)):
        assert_equal(
            external_call["mprotect", Int32](
                foreign.pages[i], foreign.lengths[i], Int32(1)
            ),
            Int32(0),
        )
    var offsets = _buffer(array, 1)
    var bytes = _buffer(array, 2) if kind == 11 or kind == 12 else 0
    var result = import_arrow_series(array, schema, copy=False)
    assert_equal(foreign.calls, 0)
    if kind == 11 or kind == 12:
        ref column = result._data[StringColumn]
        assert_true(not column._is_view_storage())
        assert_equal(
            Int(column._offsets[].unsafe_ptr()),
            offsets + 8 * (offset - offset % 8),
        )
        if len(column._bytes[]) > 0:
            assert_equal(Int(column._bytes[].unsafe_ptr()), bytes)
            assert_true(column._bytes.is_borrowed())
    elif kind == 26:
        ref column = result._data[ListColumn]
        assert_equal(
            Int(column._offsets[].unsafe_ptr()),
            offsets + 8 * (offset - offset % 8),
        )
        assert_true(column._offsets.is_borrowed())
        var child = column.child()
        var child_array = _at[ArrowArray](_at[Int](array.children)[])
        if len(child) > 0:
            assert_equal(
                Int(child.int64().unsafe_values()),
                _buffer(child_array[], 1) + Int(child_array[].offset) * 8,
            )
    elif kind == 27:
        var child_array = _at[ArrowArray](_at[Int](array.children)[])
        var child = result._data[StructColumn].field(0)
        assert_equal(
            Int(child.int64().unsafe_values()),
            _buffer(child_array[], 1)
            + (Int(child_array[].offset) + Int(array.offset)) * 8,
        )
    else:
        ref column = result._data[Column[UInt32]]
        assert_equal(Int(column.unsafe_values()), offsets + offset * 4)
        assert_true(column._data.is_borrowed())
        var dictionary_array = _at[ArrowArray](array.dictionary)
        var dictionary = result.dtype().dictionary()
        assert_equal(
            Int(dictionary[].bytes.unsafe_ptr()), _buffer(dictionary_array[], 2)
        )
        assert_equal(
            Int(dictionary[].offsets.unsafe_ptr()),
            _buffer(dictionary_array[], 1),
        )
    assert_storage_series(result, kind, positional_rows(n, offset), mode)
    return result.slice(0, n)


def exercise_paged_layout(
    mut foreign: ForeignPages, kind: Int, n: Int, offset: Int, mode: Int
) raises:
    var view = paged_layout_after_source_scope(foreign, kind, n, offset, mode)
    assert_equal(foreign.calls, 0)
    assert_storage_series(view, kind, positional_rows(n, offset), mode)
    var grown = view.copy()
    grown._append_series(selection_column(kind, 9, 101, mode))
    var expected = positional_rows(n, offset)
    expected.extend(positional_rows(9, 101))
    assert_storage_series(grown, kind, expected, mode)
    assert_storage_series(view, kind, positional_rows(n, offset), mode)
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(view, array, schema)
    var copied = import_arrow_series(array, schema)
    assert_storage_series(copied, kind, positional_rows(n, offset), mode)
    assert_equal(foreign.calls, 0)
    _ = view^


def test_borrowed_arrow_strings_binary_and_nested_buffers_share_readonly_pages_and_retain_children() raises:
    for kind in [11, 12, 13, 26, 27]:
        for n in [1, 9, 65]:
            for offset in [0, 1, 7, 9, 17]:
                for mode in range(3):
                    var foreign = ForeignPages()
                    exercise_paged_layout(foreign, kind, n, offset, mode)
                    assert_foreign_unmapped(foreign)


def exercise_small_offsets(mut foreign: ForeignArrow) raises:
    ref node = foreign.nodes[0]
    var bytes = _buffer(node.array, 2)
    var offsets = _buffer(node.array, 1)
    var imported = import_arrow_series(node.array, node.schema, copy=False)
    assert_equal(imported.get(0).string(), "bcd")
    assert_equal(imported.get(1).string(), "ef")
    ref column = imported._data[StringColumn]
    assert_equal(Int(column._bytes[].unsafe_ptr()), bytes)
    assert_true(column._bytes.is_borrowed())
    assert_true(not column._offsets.is_borrowed())
    assert_true(Int(column._offsets[].unsafe_ptr()) != offsets)
    assert_equal(foreign.nodes[0].array_releases, 0)
    _ = imported^


def test_borrowed_arrow_small_offsets_widen_without_copying_string_payloads() raises:
    var foreign = ForeignArrow(1)
    foreign.nodes[0].describe("u", "s", 2, 1)
    foreign.nodes[0].buffer(List[UInt8]())
    foreign.nodes[0].buffer(integers([0, 1, 4, 6], 4))
    foreign.nodes[0].buffer(text_bytes("abcdef"))
    foreign.seal()
    exercise_small_offsets(foreign)
    assert_equal(foreign.nodes[0].array_releases, 1)
    assert_equal(foreign.nodes[0].schema_releases, 1)


def test_borrowed_arrow_native_view_descriptors_and_external_bytes_outlive_ipc_reader() raises:
    for copy in [False, True]:
        var stream = _open_scan(
            "tests/fixtures/ipc/views.arrow", List[String](), False, True, 0, -1
        )
        var array = ArrowArray()
        var schema = ArrowSchema()
        assert_equal(_stream_next(stream, array), Int32(0))
        assert_equal(_stream_schema(stream, schema), Int32(0))
        ref text_array = _at[ArrowArray](_at[Int](array.children)[])[]
        ref text_schema = _at[ArrowSchema](_at[Int](schema.children)[])[]
        assert_equal(_read_c_string(text_schema.format), "vu")
        var descriptors = _buffer(text_array, 1)
        var bytes = _buffer(text_array, 2)
        var frame = import_arrow(array, schema, copy=copy)
        _release_stream(stream)
        ref column = frame["text"]._data[StringColumn]
        assert_true(column._is_view_storage())
        var storage = column._view_storage_unchecked()
        assert_equal(
            Int(storage._views[].unsafe_ptr()) == descriptors, not copy
        )
        assert_equal(
            Int(storage._buffers[][0][].unsafe_ptr()) == bytes, not copy
        )
        assert_equal(frame.item(0, "text").string(), "long external string")
        assert_equal(frame.item(2, "text").string(), "é")
        assert_true(frame.item(1, "text").is_null())
        var expected: List[UInt8] = [255, 0]
        assert_equal(frame.item(0, "binary").bytes(), expected)
        assert_true(frame.item(1, "binary").is_null())
        var exported = ArrowArray()
        var exported_schema = ArrowSchema()
        export_arrow(frame, exported, exported_schema)
        var roundtrip = import_arrow(exported, exported_schema)
        assert_true(roundtrip.equals(frame))


def exercise_undefined_null_view(mut foreign: ForeignArrow) raises:
    ref node = foreign.nodes[0]
    var original = _buffer(node.array, 1)
    var result = import_arrow_series(node.array, node.schema, copy=False)
    assert_equal(result.get(0).string(), "a")
    assert_true(result.get(1).is_null())
    var storage = result._data[StringColumn]._view_storage_unchecked()
    assert_true(Int(storage._views[].unsafe_ptr()) != original)
    assert_equal(storage._view_unchecked(1).length, UInt32(0))


def test_borrowed_arrow_undefined_null_view_descriptor_is_sanitized_without_reading_invalid_bytes() raises:
    var foreign = ForeignArrow(1)
    foreign.nodes[0].describe("vu", "s", 2, 0, 1)
    foreign.nodes[0].buffer(List[UInt8]([UInt8(1)]))
    var descriptors = integers([1, 97, 0, 0], 4)
    descriptors.extend(integers([-1, -1, -1, -1], 4))
    foreign.nodes[0].buffer(descriptors^)
    foreign.nodes[0].buffer(List[UInt8]())
    foreign.seal()
    exercise_undefined_null_view(foreign)
    assert_equal(foreign.nodes[0].array_releases, 1)


def exercise_foreign_struct_offset(mut foreign: ForeignArrow) raises:
    var result = import_arrow_series(
        foreign.nodes[0].array, foreign.nodes[0].schema, copy=False
    )
    assert_equal(len(result), 2)
    assert_true(not result.get(0).is_null())
    assert_true(result.get(1).is_null())
    ref column = result._data[StructColumn]
    assert_equal(column._offset, 2)
    assert_true(column._bits.is_borrowed())
    assert_equal(column.field(0).get(0).int64(), Int64(5))
    assert_equal(column.field(0).get(1).int64(), Int64(6))
    assert_equal(column.field(1).get(0).string(), "d")
    assert_equal(column.field(1).get(1).string(), "e")
    assert_equal(foreign.nodes[0].array_releases, 0)
    _ = result^


def test_borrowed_arrow_struct_parent_and_child_offsets_compose_without_copying() raises:
    var foreign = ForeignArrow(3)
    foreign.nodes[0].describe("+s", "record", 2, 2, 1)
    foreign.nodes[0].buffer(List[UInt8]([UInt8(4)]))
    foreign.nodes[0].children = [1, 2]
    foreign.nodes[1].describe("l", "number", 5, 3)
    foreign.nodes[1].buffer(List[UInt8]())
    foreign.nodes[1].buffer(integers([0, 1, 2, 3, 4, 5, 6, 7], 8))
    foreign.nodes[2].describe("u", "label", 5, 1)
    foreign.nodes[2].buffer(List[UInt8]())
    foreign.nodes[2].buffer(integers([0, 1, 2, 3, 4, 5, 6], 4))
    foreign.nodes[2].buffer(text_bytes("abcdef"))
    foreign.seal()
    exercise_foreign_struct_offset(foreign)
    for node in foreign.nodes:
        assert_equal(node.array_releases, 1)
        assert_equal(node.schema_releases, 1)
