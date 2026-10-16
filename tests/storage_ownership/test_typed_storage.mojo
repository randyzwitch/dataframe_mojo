"""Typed views, aliases, child owners, native blocks, and foreign lifetimes.

Cell expectations are independent positional literals. Sharing and release are
observed separately with Arc identities, actual destructors, and unmapped pages.
"""
# contract-group: storage.views
# contract-group: storage.isolation
# contract-group: storage.chunks
from std.ffi import external_call
from std.memory import ArcPointer, Pointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    AnyValue,
    ArrowArray,
    ArrowSchema,
    Column,
    DataFrame,
    DataType,
    ListColumn,
    Series,
    StringColumn,
    StructColumn,
    import_arrow_series,
)
from dataframe.io.arrow import _export_series
from dataframe.string_view import StringViewBuilder
from tests.support.lifecycle_stream import array_release, schema_release, at
from tests.support.selection_types_reference import (
    SELECTION_TYPES,
    assert_selected_frame,
    assert_selection_cell,
    selection_column,
    selection_dtype,
    selection_frame,
    selection_text,
    selection_valid,
)
from tests.support.typed_storage_reference import (
    TypedStorageOwner,
    ForeignPages,
    assert_export_source_buffers,
    assert_foreign_unmapped,
    assert_storage_series,
    exported_node_count,
    exportable_storage,
    make_foreign_pages,
    positional_rows,
    storage_identity,
    storage_offset,
    storage_root_count,
)


def escaped_typed_window(
    kind: Int,
    n: Int,
    offset: Int,
    mode: Int,
    destroyed: Int,
    mut before: Int,
    mut stamp: List[Int],
) raises -> Series:
    var owner = TypedStorageOwner(
        selection_column(kind, offset + n + 2, 0, mode), destroyed
    )
    stamp = storage_identity(owner.series)
    var window = owner.series.slice(offset, n).slice(0, n)
    assert_equal(storage_identity(window), stamp)
    assert_equal(storage_offset(window), offset)
    before = storage_root_count(window)
    assert_equal(storage_identity(owner.series), stamp)
    return window^


def test_every_typed_window_outlives_actual_owner_and_preserves_nested_bitmap_offsets() raises:
    for kind in range(SELECTION_TYPES):
        for n in [0, 1, 7, 8, 9, 63, 64, 65]:
            for offset in [0, 1, 7]:
                for mode in range(3):
                    var destroyed = 0
                    var before = 0
                    var stamp = List[Int]()
                    var view = escaped_typed_window(
                        kind,
                        n,
                        offset,
                        mode,
                        Int(Pointer(to=destroyed)),
                        before,
                        stamp,
                    )
                    assert_equal(destroyed, 1)
                    assert_equal(storage_root_count(view), before - 1)
                    assert_equal(storage_identity(view), stamp)
                    assert_storage_series(
                        view, kind, positional_rows(n, offset), mode
                    )
                    var end = view.slice(n, 0)
                    assert_equal(storage_identity(end), stamp)
                    assert_storage_series(end, kind, List[Int](), mode)
                    if n > 0:
                        var nested = view.slice(1, n - 1).slice(0, n - 1)
                        assert_equal(storage_identity(nested), stamp)
                        assert_equal(storage_offset(nested), offset + 1)
                        assert_storage_series(
                            nested,
                            kind,
                            positional_rows(n - 1, offset + 1),
                            mode,
                        )


def escaped_typed_cell(
    kind: Int, row: Int, mode: Int, destroyed: Int
) raises -> AnyValue:
    var owner = TypedStorageOwner(
        selection_column(kind, 1, row, mode), destroyed
    )
    return owner.series.get(0)


def test_owned_cells_keep_binary_geometry_and_nested_children_after_series_destruction() raises:
    for kind in range(SELECTION_TYPES):
        for row in [1, 4, 7]:
            for mode in range(3):
                var destroyed = 0
                var cell = escaped_typed_cell(
                    kind, row, mode, Int(Pointer(to=destroyed))
                )
                assert_equal(destroyed, 1)
                assert_selection_cell(cell, kind, row, mode)


def test_all_typed_mutable_appends_detach_and_public_append_reuses_source_owners() raises:
    for kind in range(SELECTION_TYPES):
        for n in [0, 1, 7, 8, 9, 63, 64, 65]:
            for offset in [0, 1, 7]:
                for mode in range(3):
                    var source = selection_column(kind, n, offset, mode)
                    var alias = source.copy()
                    var window = source.slice(min(1, n), max(0, n - 1))
                    var stamp = storage_identity(source)
                    var right = selection_column(kind, 9, 101, mode)
                    var right_stamp = storage_identity(right)
                    var original_rows = positional_rows(n, offset)
                    var right_rows = positional_rows(9, 101)
                    var joined_rows = original_rows.copy()
                    joined_rows.extend(right_rows.copy())
                    var appended = source.append(right)
                    var parts = appended.chunks()
                    assert_equal(len(parts), 2 if n > 0 else 1)
                    assert_equal(
                        storage_identity(parts[len(parts) - 1]), right_stamp
                    )
                    if n > 0:
                        assert_equal(storage_identity(parts[0]), stamp)
                    var mutating = source.copy()
                    assert_equal(storage_identity(mutating), stamp)
                    mutating._append_series(right)
                    assert_true(
                        storage_identity(mutating)[0] != stamp[0],
                        "mutable append must detach its shared owner",
                    )
                    assert_storage_series(mutating, kind, joined_rows, mode)
                    assert_storage_series(appended, kind, joined_rows, mode)
                    assert_storage_series(alias, kind, original_rows, mode)
                    assert_storage_series(source, kind, original_rows, mode)
                    assert_storage_series(
                        window,
                        kind,
                        positional_rows(max(0, n - 1), offset + min(1, n)),
                        mode,
                    )
                    assert_storage_series(right, kind, right_rows, mode)
                    assert_equal(storage_identity(source), stamp)
                    assert_equal(storage_identity(alias), stamp)
                    assert_equal(storage_identity(window), stamp)
                    assert_equal(storage_identity(right), right_stamp)
                    # The rhs aliases the exact destination buffers before the
                    # mutation: copy-on-write must also make self-append safe.
                    var same = mutating.copy()
                    var doubled = joined_rows.copy()
                    doubled.extend(joined_rows.copy())
                    mutating._append_series(same)
                    assert_storage_series(mutating, kind, doubled, mode)
                    assert_storage_series(same, kind, joined_rows, mode)


def test_typed_chunks_empty_boundaries_cross_windows_and_rechunk_cache_keep_aliases() raises:
    for kind in range(SELECTION_TYPES):
        for n in [0, 1, 9, 65]:
            for mode in range(3):
                var source = selection_column(kind, n, 7, mode)
                var cut = min(n, 1 + kind % 13)
                var parts: List[Series] = [
                    source.slice(0, 0),
                    source.slice(0, cut),
                    source.slice(cut, 0),
                    source.slice(cut, n - cut),
                    source.slice(n, 0),
                ]
                var chunked = Series._from_chunks(parts)
                var alias = chunked.copy()
                var first = chunked.rechunk()
                var second = alias.rechunk()
                assert_equal(
                    storage_identity(first),
                    storage_identity(second),
                    "rechunk cache must reuse the exact typed backing owners",
                )
                var rows = positional_rows(n, 7)
                assert_storage_series(chunked, kind, rows, mode)
                assert_storage_series(first, kind, rows, mode)
                assert_storage_series(alias, kind, rows, mode)
                var start = min(3, n)
                assert_storage_series(
                    chunked.slice(start, n - start),
                    kind,
                    positional_rows(n - start, 7 + start),
                    mode,
                )
                first._append_series(selection_column(kind, 9, 101, mode))
                var joined = rows.copy()
                joined.extend(positional_rows(9, 101))
                assert_storage_series(first, kind, joined, mode)
                assert_storage_series(second, kind, rows, mode)
                assert_storage_series(alias.rechunk(), kind, rows, mode)
                assert_storage_series(source, kind, rows, mode)
    # Boundaries deliberately differ between all 31 dataframe columns.
    for mode in range(3):
        var owner = selection_frame(65, 7, mode, True)
        assert_selected_frame(owner.slice(3, 57), positional_rows(57, 10), mode)
        assert_selected_frame(owner.rechunk(), positional_rows(65, 7), mode)
        assert_selected_frame(owner, positional_rows(65, 7), mode)


def native_text(row: Int) -> String:
    if row % 5 == 0:
        return ""
    if row % 5 == 1:
        return "123456789012"
    if row % 5 == 2:
        return "1234567890123"
    if row % 5 == 4:
        return "λ\x00"
    var text = String("")
    for _ in range(5000):
        text += "0123456789abcdef"
    return text + "-é-" + String(row)


def native_valid(row: Int) -> Bool:
    return row % 7 != 2


def native_series(n: Int, offset: Int) raises -> Series:
    var builder = StringViewBuilder(n + offset)
    for row in range(n + offset):
        var text = native_text(row)
        if native_valid(row):
            builder.append(text)
        else:
            builder.append_null()
    return Series("native", StringColumn(builder^.finish())).slice(offset, n)


def assert_native(actual: Series, rows: List[Int]) raises:
    assert_true(actual.dtype() == DataType.STRING)
    assert_equal(len(actual), len(rows))
    for i in range(len(rows)):
        var row = rows[i]
        assert_equal(actual.get(i).is_null(), not native_valid(row))
        if native_valid(row):
            assert_equal(actual.get(i).string(), native_text(row))


def escaped_native_window(destroyed: Int) raises -> Series:
    var owner = TypedStorageOwner(native_series(21, 0), destroyed)
    assert_true(
        owner.series._data[StringColumn]
        ._view_storage_unchecked()
        .buffer_count()
        >= 2
    )
    return owner.series.slice(1, 17)


def escaped_native_owned_string(destroyed: Int) raises -> String:
    var owner = TypedStorageOwner(native_series(17, 1), destroyed)
    return owner.series.get(2).string()


def test_borrowed_native_strings_keep_inline_external_blocks_with_owner_and_append_aliases() raises:
    var destroyed = 0
    var view = escaped_native_window(Int(Pointer(to=destroyed)))
    assert_equal(destroyed, 1)
    assert_native(view, positional_rows(17, 1))
    ref column = view._data[StringColumn]
    var storage = column._view_storage_unchecked()
    var inline_seen = False
    var external_seen = False
    for i in range(len(view)):
        if not native_valid(i + 1):
            continue
        var descriptor = storage.view(column._offset + i)
        var before = 0
        if not descriptor.is_inline():
            before = Int(
                storage._buffers[][Int(descriptor.buffer_index)].count()
            )
        var borrowed = column._get(i)
        assert_equal(String(borrowed), native_text(i + 1))
        var expected_pointer: Int
        if descriptor.is_inline():
            inline_seen = True
            expected_pointer = Int(
                storage._views[]
                .unsafe_ptr()
                .unsafe_offset(column._offset + i)
                .unsafe_bitcast[UInt8]()
                .unsafe_offset(4)
            )
        else:
            external_seen = True
            ref block = storage._buffers[][Int(descriptor.buffer_index)]
            expected_pointer = Int(
                block[].unsafe_ptr().unsafe_offset(Int(descriptor.offset))
            )
            assert_equal(
                Int(block.count()),
                before,
                "borrowed read must retain no extra block owner",
            )
        assert_equal(Int(borrowed.as_bytes().unsafe_ptr()), expected_pointer)
        var alias = view.copy()
        alias._append_series(native_series(3, 101))
        assert_equal(String(borrowed), native_text(i + 1))
        assert_equal(
            Int(column._get(i).as_bytes().unsafe_ptr()), expected_pointer
        )
    assert_true(inline_seen and external_seen)
    assert_native(view, positional_rows(17, 1))
    # Public owned strings survive even the final column owner disappearing.
    var owned_destroyed = 0
    var owned_text = escaped_native_owned_string(
        Int(Pointer(to=owned_destroyed))
    )
    assert_equal(owned_destroyed, 1)
    assert_equal(owned_text, native_text(3))


def escaped_native_rechunk(
    destroyed: Int,
    mut rows: List[Int],
    mut pointers: List[Int],
    mut block_ids: List[Int],
) raises -> Series:
    var left = TypedStorageOwner(native_series(17, 1), destroyed)
    var right = TypedStorageOwner(native_series(17, 3), destroyed)
    var left_storage = left.series._data[StringColumn]._view_storage_unchecked()
    var right_storage = right.series._data[
        StringColumn
    ]._view_storage_unchecked()
    for block in left_storage._buffers[]:
        block_ids.append(Int(block.ptr()))
    for block in right_storage._buffers[]:
        assert_true(Int(block.ptr()) not in block_ids)
        block_ids.append(Int(block.ptr()))
    var parts: List[Series] = [
        left.series.slice(0, 7),
        right.series.slice(0, 0),
        right.series.slice(0, 17),
        left.series.slice(7, 10),
        left.series.slice(0, 7),
    ]
    var chunked = Series._from_chunks(parts)
    var merged = chunked.rechunk()
    var cached = chunked.rechunk()
    assert_equal(storage_identity(merged), storage_identity(cached))
    for part in parts:
        for i in range(len(part)):
            var row = storage_offset(part) + i
            rows.append(row)
            pointers.append(
                Int(
                    part._data[StringColumn]._get(i).as_bytes().unsafe_ptr()
                ) if native_valid(row)
                and row % 5 == 3 else 0
            )
    var merged_storage = merged._data[StringColumn]._view_storage_unchecked()
    assert_equal(
        merged_storage.buffer_count(),
        len(block_ids),
        "rechunk must deduplicate shared byte blocks and remap independent sources",
    )
    for block in merged_storage._buffers[]:
        assert_true(Int(block.ptr()) in block_ids)
    assert_native(left.series, positional_rows(17, 1))
    assert_native(right.series, positional_rows(17, 3))
    return merged^


def test_native_rechunk_remaps_independent_blocks_and_retains_them_after_both_owners_die() raises:
    var destroyed = 0
    var rows = List[Int]()
    var pointers = List[Int]()
    var block_ids = List[Int]()
    var merged = escaped_native_rechunk(
        Int(Pointer(to=destroyed)), rows, pointers, block_ids
    )
    assert_equal(destroyed, 2)
    assert_native(merged, rows)
    ref column = merged._data[StringColumn]
    var storage = column._view_storage_unchecked()
    assert_equal(storage.buffer_count(), len(block_ids))
    for i in range(len(rows)):
        if pointers[i] != 0:
            assert_equal(
                Int(column._get(i).as_bytes().unsafe_ptr()),
                pointers[i],
                "external payload bytes must be reused across descriptor remapping",
            )
    var indices: List[Int] = [16, 0, 7, 16, 33, 40, 2]
    var gathered = merged.take(indices)
    var gathered_rows = List[Int]()
    for i in indices:
        gathered_rows.append(rows[i])
    assert_native(gathered, gathered_rows)
    var gathered_storage = gathered._data[
        StringColumn
    ]._view_storage_unchecked()
    assert_equal(
        Int(gathered_storage.buffers_arc().ptr()),
        Int(storage.buffers_arc().ptr()),
    )


def escaped_nested_native(structure: Bool, destroyed: Int) raises -> Series:
    var child = TypedStorageOwner(native_series(17, 1), destroyed)
    if not structure:
        var parent = TypedStorageOwner(
            Series(
                "list",
                ListColumn(
                    [0, 3, 5, 9, 17], child.series.copy(), [UInt8(0b1101)]
                ),
            ),
            destroyed,
        )
        return parent.series.slice(1, 3).slice(0, 3)
    var number = TypedStorageOwner(
        selection_column(7, 17, 1, 1).renamed("number"), destroyed
    )
    var bits: List[UInt8] = [0b01101101, 0b11011011, 0]
    var parent = TypedStorageOwner(
        Series(
            "struct",
            StructColumn([child.series.copy(), number.series.copy()], bits^),
        ),
        destroyed,
    )
    return parent.series.slice(1, 9).slice(1, 7)


def assert_nested_native(escaped: Series, structure: Bool) raises:
    if not structure:
        assert_true(escaped.get(0).is_null())
        assert_native(escaped.get(1).list(), positional_rows(4, 6))
        assert_native(escaped.get(2).list(), positional_rows(8, 10))
    else:
        for i in range(7):
            var global_row = i + 3
            var value = escaped.get(i)
            var parent_valid = (global_row - 1) % 3 != 1
            assert_equal(value.is_null(), not parent_valid)
            if parent_valid:
                assert_selection_cell(
                    value.struct_field("number"), 7, global_row, 1
                )
                var text = value.struct_field("native")
                assert_equal(text.is_null(), not native_valid(global_row))
                if native_valid(global_row):
                    assert_equal(text.string(), native_text(global_row))


def test_nested_native_children_outlive_source_child_and_parent_owners() raises:
    for structure in [False, True]:
        var destroyed = 0
        var escaped = escaped_nested_native(
            structure, Int(Pointer(to=destroyed))
        )
        assert_equal(destroyed, 3 if structure else 2)
        assert_nested_native(escaped, structure)


def export_from_destroyed_owner(
    kind: Int,
    n: Int,
    offset: Int,
    mode: Int,
    mut array: ArrowArray,
    mut schema: ArrowSchema,
    releases: Int,
    destroyed: Int,
) raises:
    var owner = TypedStorageOwner(
        exportable_storage(kind, n, offset, mode), destroyed
    )
    var before = storage_root_count(owner.series)
    _export_series(owner.series, array, schema, releases)
    assert_equal(
        storage_root_count(owner.series),
        before + 1,
        "export must retain its actual typed source owner",
    )
    assert_export_source_buffers(owner.series, array)


def test_every_typed_foreign_buffer_is_released_once_and_imported_aliases_outlive_unmapped_pages() raises:
    for kind in range(SELECTION_TYPES):
        for n in [0, 1, 9, 65]:
            for offset in [0, 1, 7]:
                for mode in range(3):
                    var releases = 0
                    var destroyed = 0
                    var array = ArrowArray()
                    var schema = ArrowSchema()
                    export_from_destroyed_owner(
                        kind,
                        n,
                        offset,
                        mode,
                        array,
                        schema,
                        Int(Pointer(to=releases)),
                        Int(Pointer(to=destroyed)),
                    )
                    assert_equal(destroyed, 1)
                    assert_equal(releases, 0)
                    var foreign = ForeignPages()
                    make_foreign_pages(array, schema, foreign)
                    if n > 0:
                        assert_true(len(foreign.pages) > 0)
                    var imported = import_arrow_series(array, schema)
                    assert_equal(array.release, 0)
                    assert_equal(schema.release, 0)
                    assert_equal(releases, exported_node_count(kind))
                    assert_foreign_unmapped(foreign)
                    var rows = positional_rows(n, offset)
                    assert_storage_series(imported, kind, rows, mode)
                    var alias = imported.copy()
                    var start = min(1, n)
                    var window = imported.slice(start, n - start)
                    imported._append_series(
                        selection_column(kind, 9, 101, mode)
                    )
                    assert_storage_series(alias, kind, rows, mode)
                    assert_storage_series(
                        window,
                        kind,
                        positional_rows(n - start, offset + start),
                        mode,
                    )
                    rows.extend(positional_rows(9, 101))
                    assert_storage_series(imported, kind, rows, mode)
                    var chunks = alias.append(window)
                    var combined = positional_rows(n, offset)
                    combined.extend(positional_rows(n - start, offset + start))
                    assert_storage_series(
                        chunks.rechunk(), kind, combined, mode
                    )
                    array_release(array)
                    schema_release(schema)
                    assert_equal(foreign.calls, 1)
                    assert_equal(releases, exported_node_count(kind))


def test_foreign_import_errors_release_recursive_buffers_and_unmap_pages_once() raises:
    for kind in [3, 11, 13, 16, 22, 26, 27, 28]:
        var releases = 0
        var destroyed = 0
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_from_destroyed_owner(
            kind,
            65,
            7,
            1,
            array,
            schema,
            Int(Pointer(to=releases)),
            Int(Pointer(to=destroyed)),
        )
        assert_equal(destroyed, 1)
        var foreign = ForeignPages()
        make_foreign_pages(array, schema, foreign)
        var unsupported: List[UInt8] = [120, 0]
        schema.format = Int(unsupported.unsafe_ptr())
        with assert_raises():
            _ = import_arrow_series(array, schema)
        assert_foreign_unmapped(foreign)
        assert_equal(releases, exported_node_count(kind))
        assert_equal(array.release, 0)
        assert_equal(schema.release, 0)
        array_release(array)
        schema_release(schema)
        assert_equal(foreign.calls, 1)


def export_native_shape(
    shape: Int,
    mut array: ArrowArray,
    mut schema: ArrowSchema,
    releases: Int,
    destroyed: Int,
) raises:
    var source = native_series(17, 1) if shape == 0 else escaped_nested_native(
        shape == 2, destroyed
    )
    var owner = TypedStorageOwner(source^, destroyed)
    _export_series(owner.series, array, schema, releases)


def test_native_string_and_nested_foreign_exports_own_materialized_adapters_after_multi_page_release() raises:
    for shape in range(3):
        var releases = 0
        var destroyed = 0
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_native_shape(
            shape,
            array,
            schema,
            Int(Pointer(to=releases)),
            Int(Pointer(to=destroyed)),
        )
        assert_equal(destroyed, 1 if shape == 0 else 3 if shape == 1 else 4)
        assert_equal(releases, 0)
        var foreign = ForeignPages()
        make_foreign_pages(array, schema, foreign)
        var multi_page = False
        for length in foreign.lengths:
            multi_page = multi_page or length > Int(
                external_call["getpagesize", Int32]()
            )
        assert_true(
            multi_page,
            "native external payload fixture must span multiple mapped pages",
        )
        var imported = import_arrow_series(array, schema)
        assert_equal(releases, shape + 1)
        assert_foreign_unmapped(foreign)
        if shape == 0:
            assert_native(imported, positional_rows(17, 1))
        else:
            assert_nested_native(imported, shape == 2)


# expected-failure: Arrow date32 export narrows hidden null Int64.MAX payloads
# expected-failure: and raises instead of preserving valid visible dates plus
# expected-failure: nulls.
# expected-diagnostic: date export must ignore hidden null payloads
def test_date_export_does_not_narrow_hidden_null_payloads() raises:
    var releases = 0
    var array = ArrowArray()
    var schema = ArrowSchema()
    var source = selection_column(17, 9, 7, 1)
    try:
        _export_series(source, array, schema, Int(Pointer(to=releases)))
        var imported = import_arrow_series(array, schema)
        assert_storage_series(imported, 17, positional_rows(9, 7), 1)
    except error:
        array_release(array)
        schema_release(schema)
        if "date outside the Arrow date32 range" in String(error):
            raise Error("date export must ignore hidden null payloads")
        raise error


def test_validity_lists_pack_to_the_same_bits_at_every_length() raises:
    # Lengths around whole bytes and eight-lane groups, with patterns
    # that set and clear each lane position: every row's validity and the
    # null count read back as the list gave them.
    for n in [0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 129, 300]:
        for pattern in range(4):
            var values = List[Int64]()
            var valid = List[Bool]()
            var nulls = 0
            for i in range(n):
                values.append(Int64(i))
                var keep = (
                    i % 3
                    != 1 if pattern
                    == 0 else (i * 7) % 11
                    < 5 if pattern
                    == 1 else i % 8
                    == pattern if pattern
                    == 2 else i
                    != n - 1
                )
                valid.append(keep)
                nulls += Int(not keep)
            var column = Column[Int64](values^, valid.copy())
            assert_equal(column.null_count(), nulls)
            for i in range(n):
                assert_equal(column.is_valid(i), valid[i])
