"""Typed storage observations and real foreign pages, separate from cell oracles.

Identity/count checks observe backing owners. Arrow fixture plumbing replaces
exported buffers with independent mmap pages; release really unmaps those pages.
Expected cells remain the positional scalar literals in selection_types_reference.
"""
from std.ffi import external_call
from std.memory import Pointer, unsafe_memcpy
from std.sys import CompilationTarget
from std.testing import assert_equal, assert_true
from dataframe import (
    ArrowArray,
    ArrowSchema,
    BoolColumn,
    Column,
    DataType,
    ListColumn,
    Series,
    StringColumn,
    StructColumn,
)
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.io.arrow import _read_c_string
from tests.support.lifecycle_stream import ArrayRelease, array_release, at
from tests.support.mapped_pages import assert_mapped_page, assert_unmapped_page
from tests.support.selection_types_reference import (
    assert_selection_cell,
    selection_column,
    selection_dtype,
    selection_tick,
    selection_valid,
)


struct TypedStorageOwner(Deinitable, Movable):
    var series: Series
    var destroyed: Int

    def __init__(out self, var series: Series, destroyed: Int):
        self.series = series^
        self.destroyed = destroyed

    def __deinit__(deinit self):
        # Drop the actual source before recording completion, rather than just
        # incrementing a scope-exit counter while its buffers remain retained.
        _ = self.series^
        at[Int](self.destroyed)[] += 1


def storage_identity(series: Series) raises -> List[Int]:
    assert_true(not series.is_chunked())
    comptime for i in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[i]
        if series._data.isa[Column[Scalar[D]]]():
            ref column = series._data[Column[Scalar[D]]]
            return [Int(column._data.ptr()), Int(column._bits.ptr())]
    if series._data.isa[Column[Int128]]():
        ref column = series._data[Column[Int128]]
        return [Int(column._data.ptr()), Int(column._bits.ptr())]
    if series._data.isa[BoolColumn]():
        ref column = series._data[BoolColumn]
        return [Int(column._data.ptr()), Int(column._bits.ptr())]
    if series._data.isa[ListColumn]():
        ref column = series._data[ListColumn]
        var result: List[Int] = [
            Int(column._offsets.ptr()),
            Int(column._bits.ptr()),
            Int(column._child.ptr()),
        ]
        result.extend(storage_identity(column.child()))
        return result^
    if series._data.isa[StructColumn]():
        ref column = series._data[StructColumn]
        var result: List[Int] = [
            Int(column._fields.ptr()),
            Int(column._bits.ptr()),
        ]
        for i in range(column.field_count()):
            result.extend(storage_identity(column.field(i)))
        return result^
    ref column = series._data[StringColumn]
    if column._is_view_storage():
        var storage = column._view_storage_unchecked()
        var result: List[Int] = [
            Int(storage.views_arc().ptr()),
            Int(storage.buffers_arc().ptr()),
            Int(storage.validity_arc().ptr()),
        ]
        for buffer in storage._buffers[]:
            result.append(Int(buffer.ptr()))
        return result^
    return [
        Int(column._bytes.ptr()),
        Int(column._offsets.ptr()),
        Int(column._bits.ptr()),
    ]


def storage_root_count(series: Series) -> Int:
    comptime for i in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[i]
        if series._data.isa[Column[Scalar[D]]]():
            return Int(series._data[Column[Scalar[D]]]._data.count())
    if series._data.isa[Column[Int128]]():
        return Int(series._data[Column[Int128]]._data.count())
    if series._data.isa[BoolColumn]():
        return Int(series._data[BoolColumn]._data.count())
    if series._data.isa[ListColumn]():
        return Int(series._data[ListColumn]._offsets.count())
    if series._data.isa[StructColumn]():
        return Int(series._data[StructColumn]._fields.count())
    return Int(series._data[StringColumn]._bytes.count())


def storage_offset(series: Series) -> Int:
    comptime for i in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[i]
        if series._data.isa[Column[Scalar[D]]]():
            return series._data[Column[Scalar[D]]]._offset
    if series._data.isa[Column[Int128]]():
        return series._data[Column[Int128]]._offset
    if series._data.isa[BoolColumn]():
        return series._data[BoolColumn]._offset
    if series._data.isa[ListColumn]():
        return series._data[ListColumn]._offset
    if series._data.isa[StructColumn]():
        return series._data[StructColumn]._offset
    return series._data[StringColumn]._offset


def positional_rows(n: Int, offset: Int) -> List[Int]:
    var result = List[Int]()
    for row in range(offset, offset + n):
        result.append(row)
    return result^


def assert_storage_series(
    actual: Series, kind: Int, rows: List[Int], mode: Int
) raises:
    assert_equal(actual.name(), "x")
    assert_true(actual.dtype() == selection_dtype(kind))
    assert_equal(len(actual), len(rows))
    for i in range(len(rows)):
        assert_selection_cell(actual.get(i), kind, rows[i], mode)


def exportable_storage(
    kind: Int, n: Int, offset: Int, mode: Int
) raises -> Series:
    if kind != 17:
        return selection_column(kind, n, offset, mode)
    # date32 conversion is a separate contract. Neutral null payloads keep
    # this lifecycle fixture focused on ownership; its poison has its own test.
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(offset + n):
        var present = selection_valid(row, mode)
        values.append(selection_tick(kind, row) if present else Int64(0))
        valid.append(present)
    return (
        Series("x", Column[Int64](values^, valid^))
        .with_dtype(DataType.DATE)
        .slice(offset, n)
    )


def exported_node_count(kind: Int) -> Int:
    return 2 if kind == 13 or kind == 26 else 3 if kind == 27 else 1


def assert_export_source_buffers(series: Series, array: ArrowArray) raises:
    var buffers = at[Int](array.buffers)
    if series.dtype() == DataType.DATE:
        assert_equal(array.offset, Int64(0))
        if len(series) > 0:
            assert_true(
                buffers[unsafe_offset=1]
                != Int(series._data[Column[Int64]]._data[].unsafe_ptr()),
                "date32 conversion must own a separate narrowed buffer",
            )
        return
    comptime for i in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[i]
        if series._data.isa[Column[Scalar[D]]]():
            ref column = series._data[Column[Scalar[D]]]
            assert_equal(array.offset, Int64(column._offset))
            assert_equal(
                buffers[unsafe_offset=1], Int(column._data[].unsafe_ptr())
            )
            return
    if series._data.isa[Column[Int128]]():
        ref column = series._data[Column[Int128]]
        assert_equal(array.offset, Int64(column._offset))
        assert_equal(buffers[unsafe_offset=1], Int(column._data[].unsafe_ptr()))
    elif series._data.isa[BoolColumn]():
        ref column = series._data[BoolColumn]
        assert_equal(array.offset, Int64(column._offset))
        assert_equal(buffers[unsafe_offset=1], Int(column._data[].unsafe_ptr()))
    elif series._data.isa[ListColumn]():
        ref column = series._data[ListColumn]
        assert_equal(array.offset, Int64(column._offset))
        assert_equal(
            buffers[unsafe_offset=1], Int(column._offsets[].unsafe_ptr())
        )
        assert_export_source_buffers(
            column.child(), at[ArrowArray](at[Int](array.children)[])[]
        )
    elif series._data.isa[StructColumn]():
        ref column = series._data[StructColumn]
        assert_equal(array.offset, Int64(0))
        for i in range(column.field_count()):
            assert_export_source_buffers(
                column.field(i),
                at[ArrowArray](at[Int](array.children)[unsafe_offset=i])[],
            )
    else:
        ref column = series._data[StringColumn]
        assert_true(not column._is_view_storage())
        assert_equal(array.offset, Int64(column._offset))
        assert_equal(
            buffers[unsafe_offset=1], Int(column._offsets[].unsafe_ptr())
        )
        assert_equal(
            buffers[unsafe_offset=2], Int(column._bytes[].unsafe_ptr())
        )


struct ForeignPages(Deinitable, Movable):
    var pages: List[Int]
    var lengths: List[Int]
    var original_release: Int
    var original_private: Int
    var calls: Int
    var unmapped: Int
    var failed_unmaps: Int

    def __init__(out self):
        self.pages = List[Int]()
        self.lengths = List[Int]()
        self.original_release = 0
        self.original_private = 0
        self.calls = 0
        self.unmapped = 0
        self.failed_unmaps = 0

    def __deinit__(deinit self):
        # Cleanup if constructing the foreign fixture itself failed before
        # its release callback was installed or invoked.
        if self.calls == 0:
            for i in range(len(self.pages)):
                _ = external_call["munmap", Int32](
                    self.pages[i], self.lengths[i]
                )


def foreign_page(
    mut state: ForeignPages, original: Int, bytes: Int
) raises -> Int:
    if original == 0 or bytes == 0:
        return original
    var page = Int(external_call["getpagesize", Int32]())
    var length = ((bytes + page - 1) // page) * page
    # MAP_ANON is 0x1000 on Darwin and 0x20 on Linux; MAP_PRIVATE is
    # 0x2 on both.
    comptime MAP_FLAGS = 0x1002 if CompilationTarget.is_macos() else 0x22
    var address = external_call["mmap", Int](
        0, length, Int32(3), Int32(MAP_FLAGS), Int32(-1), Int64(0)
    )
    assert_true(address != -1, "foreign fixture mmap must succeed")
    unsafe_memcpy(dest=at[UInt8](address), src=at[UInt8](original), count=bytes)
    state.pages.append(address)
    state.lengths.append(length)
    return address


def replace_foreign_buffers(
    mut array: ArrowArray, schema: ArrowSchema, mut state: ForeignPages
) raises:
    var format = _read_c_string(schema.format)
    var rows = Int(array.offset + array.length)
    var buffers = at[Int](array.buffers)
    if array.n_buffers > 0:
        buffers[unsafe_offset=0] = foreign_page(
            state, buffers[unsafe_offset=0], (rows + 7) // 8
        )
    if format == "U" or format == "Z" or format == "+L":
        var address = buffers[unsafe_offset=1]
        var bytes = Int(at[Int64](address)[unsafe_offset=rows])
        buffers[unsafe_offset=1] = foreign_page(state, address, (rows + 1) * 8)
        if format != "+L":
            buffers[unsafe_offset=2] = foreign_page(
                state, buffers[unsafe_offset=2], bytes
            )
    elif format != "+s":
        var width = 8
        if format == "c" or format == "C":
            width = 1
        elif format == "s" or format == "S":
            width = 2
        elif format == "i" or format == "I" or format == "f" or format == "tdD":
            width = 4
        elif format.startswith("d:"):
            var fields = String(format[byte = 2 : format.byte_length()]).split(
                ","
            )
            width = Int(String(fields[2])) // 8 if len(fields) == 3 else 16
        var bytes = (rows + 7) // 8 if format == "b" else rows * width
        buffers[unsafe_offset=1] = foreign_page(
            state, buffers[unsafe_offset=1], bytes
        )
    for i in range(Int(array.n_children)):
        replace_foreign_buffers(
            at[ArrowArray](at[Int](array.children)[unsafe_offset=i])[],
            at[ArrowSchema](at[Int](schema.children)[unsafe_offset=i])[],
            state,
        )
    if array.dictionary != 0:
        replace_foreign_buffers(
            at[ArrowArray](array.dictionary)[],
            at[ArrowSchema](schema.dictionary)[],
            state,
        )


def release_foreign_pages(array: Pointer[ArrowArray, MutAnyOrigin]) abi("C"):
    ref state = at[ForeignPages](array[].private_data)[]
    state.calls += 1
    array[].release = state.original_release
    array[].private_data = state.original_private
    array_release(array[])
    for i in range(len(state.pages)):
        var result = external_call["munmap", Int32](
            state.pages[i], state.lengths[i]
        )
        state.unmapped += Int(result == 0)
        state.failed_unmaps += Int(result != 0)
    array[].release = 0


def make_foreign_pages(
    mut array: ArrowArray, schema: ArrowSchema, mut state: ForeignPages
) raises:
    replace_foreign_buffers(array, schema, state)
    state.original_release = array.release
    state.original_private = array.private_data
    var release: ArrayRelease = release_foreign_pages
    array.release = Pointer(to=release).unsafe_bitcast[Int]()[]
    array.private_data = Int(Pointer(to=state))
    var page = Int(external_call["getpagesize", Int32]())
    for i in range(len(state.pages)):
        for address in [
            state.pages[i],
            state.pages[i] + state.lengths[i] - page,
        ]:
            assert_mapped_page(address)


def assert_foreign_unmapped(state: ForeignPages) raises:
    assert_equal(state.calls, 1)
    assert_equal(state.unmapped, len(state.pages))
    assert_equal(state.failed_unmaps, 0)
    var page = Int(external_call["getpagesize", Int32]())
    for i in range(len(state.pages)):
        for address in [
            state.pages[i],
            state.pages[i] + state.lengths[i] - page,
        ]:
            assert_unmapped_page(address)
