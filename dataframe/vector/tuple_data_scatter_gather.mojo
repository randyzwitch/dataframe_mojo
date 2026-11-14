"""Write chunk values into row layouts and gather rows back into vectors.

Fixed-width values occupy their row fields. Validity bits identify nulls.
Short strings store their bytes inline; long strings and nested values use
the collection's heap. Dispatch on the physical type selects the matching
scatter or gather routine.

Nested output is assembled in buffers for each level, then exposed through
an at-rest column that the chunk's vector can read. String views retain
the collection's owners so referenced bytes outlive the output vectors.
"""
from std.collections import Dict, Optional
from std.memory import ArcPointer, Pointer, unsafe_memcpy, unsafe_memset
from std.sys import size_of

from dataframe.dtype import DataType
from dataframe.series import Series
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column
from dataframe.storage.nested import ListColumn, StructColumn
from dataframe.storage.buffer import _Buffer
from dataframe.storage.string import StringBuilder, StringColumn
from dataframe.storage.string_view import StringView, StringViewStorage
from dataframe.vector.data_chunk import (
    ColumnArena,
    DataChunk,
    _typed,
    flat_over,
)
from dataframe.vector.tuple_data_layout import (
    TupleDataLayout,
    constant_size,
    physical_of_dtype,
    validity_bytes,
)
from dataframe.vector.tuple_data_states import TupleDataChunkState
from dataframe.vector.types import (
    PHYSICAL_BOOL,
    PHYSICAL_LIST,
    PHYSICAL_STRUCT,
    PHYSICAL_VARCHAR,
    STANDARD_VECTOR_SIZE,
    STRING_INLINE_BYTES,
    SelectionVector,
    StringT,
    UnifiedVectorFormat,
    ValidityMask,
    physical_of,
    physical_size,
    set_invalid,
    string_data,
)

comptime _FIXED = [
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.float32,
    DType.float64,
    DType.int128,
]

# A string's slot within a list's heap data: its 4-byte length, the bytes
# following in the heap.
comptime _WITHIN_LIST_STRING_SIZE = 4


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


@always_inline
def _load[D: DType](address: Int) -> Scalar[D]:
    return Pointer[Scalar[D], MutAnyOrigin](
        unsafe_from_address=address
    ).unsafe_load[alignment=1]()


@always_inline
def _store[D: DType](address: Int, value: Scalar[D]):
    Pointer[Scalar[D], MutAnyOrigin](unsafe_from_address=address).unsafe_store[
        alignment=1
    ](value)


@always_inline
def _ints(address: Int) -> Pointer[Int, MutAnyOrigin]:
    return Pointer[Int, MutAnyOrigin](unsafe_from_address=address)


@always_inline
def _bytes(address: Int) -> Pointer[UInt8, MutAnyOrigin]:
    return Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address)


@always_inline
def _fill_ones(address: Int, count: Int):
    unsafe_memset(
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address), 255, count
    )


@always_inline
def _copy(dest: Int, src: Int, count: Int):
    unsafe_memcpy(dest=_bytes(dest), src=_bytes(src), count=count)


@always_inline
def _move_block[W: Int](dest: Int, src: Int):
    """Copy `W` bytes from `src` to `dest`, unaligned."""
    var value = _bytes(src).unsafe_load[width=W, alignment=1]()
    _bytes(dest).unsafe_store[alignment=1](value)


@always_inline
def _copy_string(dest: Int, src: Int, count: Int):
    """Copy a string's `count` bytes (more than 12, the longest inlined)
    in whole blocks. A length that is not a multiple of the block is
    covered by one more block ending at the last byte, overlapping the
    previous one, so no byte is copied one at a time and nothing outside
    [src, src + count) or [dest, dest + count) is touched."""
    if count < 16:
        # 13 to 15 bytes: two overlapping 8-byte words.
        _move_block[8](dest, src)
        _move_block[8](dest + count - 8, src + count - 8)
        return
    if count <= 32:
        _move_block[16](dest, src)
        _move_block[16](dest + count - 16, src + count - 16)
        return
    var at = 0
    while at + 32 <= count:
        _move_block[32](dest + at, src + at)
        at += 32
    if at < count:
        _move_block[32](dest + count - 32, src + count - 32)


@always_inline
def _bit_valid(bits: Int, index: Int) -> Bool:
    """Bit `index` of a validity mask written here (offset 0)."""
    return (_bytes(bits)[unsafe_offset=index >> 3] >> UInt8(index & 7)) & 1 == 1


@always_inline
def _entry_offset(entries: Int, index: Int) -> Int:
    """list_entry_t.offset of entry `index`."""
    return Int(_load[DType.uint64](entries + 16 * index))


@always_inline
def _entry_length(entries: Int, index: Int) -> Int:
    """list_entry_t.length of entry `index`."""
    return Int(_load[DType.uint64](entries + 16 * index + 8))


@always_inline
def _null_value[D: DType]() -> Scalar[D]:
    """The value stored for a null: the type's minimum, NaN for floats. A fixed
    value keeps a row's bytes the same for every null; validity still marks it
    null."""
    comptime if D.is_floating_point():
        var zero = Scalar[D](0)
        return zero / zero
    else:
        return Scalar[D].MIN


# ---------------------------------------------------------------------------
# Heap sizes
# ---------------------------------------------------------------------------


def compute_heap_sizes(
    mut state: TupleDataChunkState,
    append_sel: SelectionVector,
    append_count: Int,
):
    """Each appended row's heap bytes (known before any is written, so the heap
    is allocated once), into `state.heap_sizes`."""
    var sizes = _ints(state.sizes())
    for i in range(append_count):
        sizes[unsafe_offset=i] = 0
    for k in range(len(state.column_ids)):
        _heap_sizes(state, state.column_ids[k], append_sel, append_count)


def _heap_sizes(
    mut state: TupleDataChunkState,
    f: Int,
    append_sel: SelectionVector,
    append_count: Int,
):
    var physical = state.vector_data[f].physical
    var source = state.vector_data[f].unified
    var sizes = _ints(state.sizes())
    if physical == PHYSICAL_VARCHAR:
        # Only non-inlined strings are stored in the heap, and none of a
        # referenced vector's.
        if state.vector_data[f].referenced:
            return
        for i in range(append_count):
            var at = source.sel.get_index(append_sel.get_index(i))
            if source.validity.row_is_valid(at):
                var length = Int(_load[DType.uint32](source.data + 16 * at))
                if length > STRING_INLINE_BYTES:
                    sizes[unsafe_offset=i] += length
    elif physical == PHYSICAL_STRUCT:
        var children = state.vector_data[f].children.copy()
        for c in range(len(children)):
            _heap_sizes(state, children[c], append_sel, append_count)
    elif physical == PHYSICAL_LIST:
        # Lists are stored entirely in the heap.
        for i in range(append_count):
            var at = source.sel.get_index(append_sel.get_index(i))
            if source.validity.row_is_valid(at):
                sizes[unsafe_offset=i] += 8
        var child = state.vector_data[f].children[0]
        _within_heap_sizes(state, child, append_sel, append_count, source)


def _within_heap_sizes(
    mut state: TupleDataChunkState,
    f: Int,
    append_sel: SelectionVector,
    append_count: Int,
    list_data: UnifiedVectorFormat,
):
    """Heap bytes of a list's child values, per appended row."""
    var physical = state.vector_data[f].physical
    var source = state.vector_data[f].unified
    var sizes = _ints(state.sizes())
    if physical == PHYSICAL_LIST:
        _collection_within_heap_sizes(
            state, f, append_sel, append_count, list_data
        )
        return
    for i in range(append_count):
        var list_idx = list_data.sel.get_index(append_sel.get_index(i))
        if not list_data.validity.row_is_valid(list_idx):
            continue
        var length = _entry_length(list_data.data, list_idx)
        if length == 0:
            continue
        # A validity mask, then the values.
        var size = validity_bytes(length)
        if physical == PHYSICAL_VARCHAR:
            size += length * _WITHIN_LIST_STRING_SIZE
            var offset = _entry_offset(list_data.data, list_idx)
            for c in range(length):
                var at = source.sel.get_index(offset + c)
                if source.validity.row_is_valid(at):
                    size += Int(_load[DType.uint32](source.data + 16 * at))
        elif physical != PHYSICAL_STRUCT:
            size += length * physical_size(physical)
        sizes[unsafe_offset=i] += size
    if physical == PHYSICAL_STRUCT:
        var children = state.vector_data[f].children.copy()
        for c in range(len(children)):
            _within_heap_sizes(
                state, children[c], append_sel, append_count, list_data
            )


def _collection_within_heap_sizes(
    mut state: TupleDataChunkState,
    f: Int,
    append_sel: SelectionVector,
    append_count: Int,
    list_data: UnifiedVectorFormat,
):
    """Heap bytes of a list in a list. The child list's entries of each appended
    row are combined into one entry, and the grandchild is sliced to read them
    in order."""
    var child_list = state.vector_data[f].unified
    var sizes = _ints(state.sizes())

    # The actual child list size (may differ from the child's size if the
    # child is a dictionary or constant).
    var sum_of_sizes = 0
    for i in range(append_count):
        var list_idx = list_data.sel.get_index(append_sel.get_index(i))
        if not list_data.validity.row_is_valid(list_idx):
            continue
        var offset = _entry_offset(list_data.data, list_idx)
        var length = _entry_length(list_data.data, list_idx)
        for c in range(length):
            var at = child_list.sel.get_index(offset + c)
            if child_list.validity.row_is_valid(at):
                sum_of_sizes += _entry_length(child_list.data, at)
    var child_count = max(sum_of_sizes, state.vector_data[f].list_size)

    var child_f = state.vector_data[f].children[0]
    var combined_sel = List[UInt32](length=max(child_count, 1), fill=0)
    var sel_at = combined_sel.unsafe_ptr()
    var combined_entries = List[UInt64](length=2 * STANDARD_VECTOR_SIZE, fill=0)
    var entries_at = combined_entries.unsafe_ptr()
    var combined_validity = List[UInt8](
        length=STANDARD_VECTOR_SIZE // 8, fill=255
    )
    var validity_at = Int(combined_validity.unsafe_ptr())

    var combined_offset = 0
    for i in range(append_count):
        var append_idx = append_sel.get_index(i)
        var list_idx = list_data.sel.get_index(append_idx)
        if not list_data.validity.row_is_valid(list_idx):
            set_invalid(validity_at, append_idx)
            continue
        var offset = _entry_offset(list_data.data, list_idx)
        var length = _entry_length(list_data.data, list_idx)
        # The validity mask and the child list lengths.
        sizes[unsafe_offset=i] += validity_bytes(length) + length * 8
        var child_list_size = 0
        for c in range(length):
            var at = child_list.sel.get_index(offset + c)
            if not child_list.validity.row_is_valid(at):
                continue
            var child_offset = _entry_offset(child_list.data, at)
            var child_length = _entry_length(child_list.data, at)
            for v in range(child_length):
                sel_at[
                    unsafe_offset=combined_offset + child_list_size + v
                ] = UInt32(child_offset + v)
            child_list_size += child_length
        entries_at[unsafe_offset=2 * append_idx] = UInt64(combined_offset)
        entries_at[unsafe_offset=2 * append_idx + 1] = UInt64(child_list_size)
        combined_offset += child_list_size

    _apply_slice_recursive(state, child_f, Int(sel_at), child_count)
    _ = combined_sel^

    ref child = state.vector_data[child_f]
    child.combined_entries = combined_entries^
    child.combined_validity = combined_validity^
    child.combined = UnifiedVectorFormat(
        SelectionVector(),
        Int(child.combined_entries.unsafe_ptr()),
        ValidityMask(Int(child.combined_validity.unsafe_ptr()), 0),
        PHYSICAL_LIST,
    )
    child.has_combined = True
    var combined = child.combined
    _within_heap_sizes(state, child_f, append_sel, append_count, combined)


def _apply_slice_recursive(
    mut state: TupleDataChunkState, f: Int, combined_sel: Int, count: Int
):
    """The format reads its original selection through `combined_sel`, so a
    nested child is read through one selection; a struct's fields too."""
    ref format = state.vector_data[f]
    format.sliced_sel = List[UInt32](unsafe_uninit_length=max(count, 1))
    var out = format.sliced_sel.unsafe_ptr()
    var combined = Pointer[UInt32, MutAnyOrigin](
        unsafe_from_address=combined_sel
    )
    for i in range(count):
        out[unsafe_offset=i] = UInt32(
            format.original_sel.get_index(Int(combined[unsafe_offset=i]))
        )
    format.unified.sel = SelectionVector(Int(out))
    if format.physical == PHYSICAL_STRUCT:
        var children = format.children.copy()
        var sliced = Int(out)
        for c in range(len(children)):
            _apply_slice_recursive(state, children[c], sliced, count)


# ---------------------------------------------------------------------------
# Scatter
# ---------------------------------------------------------------------------


def scatter(
    layout: TupleDataLayout,
    state: TupleDataChunkState,
    append_sel: SelectionVector,
    append_count: Int,
):
    """The appended rows' values into the rows at `state.row_locations`, their
    heap data at the heap pointers."""
    var rows = _ints(state.rows())
    ref top = layout.nodes[][0]
    if top.can_have_null:
        # Every value valid until a null clears its bit.
        var width = validity_bytes(top.column_count())
        for i in range(append_count):
            _fill_ones(rows[unsafe_offset=i], width)
    if not top.all_constant:
        var sizes = _ints(state.sizes())
        for i in range(append_count):
            _store[DType.uint64](
                rows[unsafe_offset=i] + top.heap_size_offset,
                UInt64(sizes[unsafe_offset=i]),
            )
    for k in range(len(state.column_ids)):
        var col = state.column_ids[k]
        _scatter(
            layout, 0, col, state, col, append_sel, append_count, Int(rows)
        )


def _scatter(
    layout: TupleDataLayout,
    node: Int,
    col: Int,
    state: TupleDataChunkState,
    f: Int,
    append_sel: SelectionVector,
    append_count: Int,
    rows: Int,
):
    """The scatter function of column `col` of layout `node`, from format
    `f`, into the rows at `rows`."""
    var physical = layout.nodes[][node].physicals[col]
    var offset = layout.nodes[][node].offsets[col]
    var source = state.vector_data[f].unified
    if physical == PHYSICAL_VARCHAR:
        _scatter_strings(
            source,
            append_sel,
            append_count,
            rows,
            0 if state.vector_data[f].referenced else state.heaps(),
            offset,
            col,
        )
    elif physical == PHYSICAL_STRUCT:
        _scatter_struct(
            layout, node, col, state, f, append_sel, append_count, rows
        )
    elif physical == PHYSICAL_LIST:
        _scatter_list(
            layout, node, col, state, f, append_sel, append_count, rows
        )
    elif physical == PHYSICAL_BOOL:
        _scatter_fixed[DType.uint8](
            source, append_sel, append_count, rows, offset, col
        )
    else:
        comptime for k in range(len(_FIXED)):
            comptime D = _FIXED[k]
            if physical == physical_of[D]():
                _scatter_fixed[D](
                    source, append_sel, append_count, rows, offset, col
                )


def _scatter_fixed[
    D: DType
](
    source: UnifiedVectorFormat,
    append_sel: SelectionVector,
    append_count: Int,
    rows: Int,
    offset_in_row: Int,
    col: Int,
):
    """Scatter fixed-width values. Specialize the loop with no append
    selection, source selection or null checks, avoiding branches in the
    common contiguous all-valid case."""
    if (
        not append_sel.is_set()
        and not source.sel.is_set()
        and source.validity.all_valid()
    ):
        _scatter_fixed_loop[D, True](
            source, append_sel, append_count, rows, offset_in_row, col
        )
    else:
        _scatter_fixed_loop[D, False](
            source, append_sel, append_count, rows, offset_in_row, col
        )


@always_inline
def _scatter_fixed_loop[
    D: DType, PLAIN: Bool
](
    source: UnifiedVectorFormat,
    append_sel: SelectionVector,
    append_count: Int,
    rows: Int,
    offset_in_row: Int,
    col: Int,
):
    var data = source.values[D]()
    var targets = _ints(rows)
    var entry = col >> 3
    var bit = UInt8(col & 7)
    var null_value = _null_value[D]()
    for i in range(append_count):
        var row = targets[unsafe_offset=i]
        comptime if PLAIN:
            _store[D](row + offset_in_row, data[unsafe_offset=i])
        else:
            var at = source.sel.get_index(append_sel.get_index(i))
            if source.validity.row_is_valid(at):
                _store[D](row + offset_in_row, data[unsafe_offset=at])
            else:
                _store[D](row + offset_in_row, null_value)
                _bytes(row)[unsafe_offset=entry] &= ~(UInt8(1) << bit)


def _scatter_strings(
    source: UnifiedVectorFormat,
    append_sel: SelectionVector,
    append_count: Int,
    rows: Int,
    heaps: Int,
    offset_in_row: Int,
    col: Int,
):
    """Store strings: an inlined string as it stands; a longer one's bytes into
    the heap and its header and heap pointer into the row. A null stores the
    fixed one-byte string "\\x80", inlined, so it needs no heap. `heaps` 0:
    the source's bytes outlive the rows, which point at them."""
    var targets = _ints(rows)
    var heap_at = _ints(heaps)
    var entry = col >> 3
    var bit = UInt8(col & 7)
    for i in range(append_count):
        var row = targets[unsafe_offset=i]
        var at = source.sel.get_index(append_sel.get_index(i))
        var target = row + offset_in_row
        if not source.validity.row_is_valid(at):
            _store[DType.uint64](target, UInt64(1) | (UInt64(0x80) << 32))
            _store[DType.uint64](target + 8, 0)
            _bytes(row)[unsafe_offset=entry] &= ~(UInt8(1) << bit)
            continue
        var value = source.data + 16 * at
        var length = Int(_load[DType.uint32](value))
        _store[DType.uint64](target, _load[DType.uint64](value))
        if length <= STRING_INLINE_BYTES or heaps == 0:
            _store[DType.uint64](target + 8, _load[DType.uint64](value + 8))
        else:
            var heap = heap_at[unsafe_offset=i]
            _copy_string(heap, Int(_load[DType.uint64](value + 8)), length)
            _store[DType.uint64](target + 8, UInt64(heap))
            heap_at[unsafe_offset=i] = heap + length


def _scatter_struct(
    layout: TupleDataLayout,
    node: Int,
    col: Int,
    state: TupleDataChunkState,
    f: Int,
    append_sel: SelectionVector,
    append_count: Int,
    rows: Int,
):
    """The struct's validity in this row, then its fields into its nested
    row."""
    var source = state.vector_data[f].unified
    var targets = _ints(rows)
    var entry = col >> 3
    var bit = UInt8(col & 7)
    if not source.validity.all_valid():
        for i in range(append_count):
            var at = source.sel.get_index(append_sel.get_index(i))
            if not source.validity.row_is_valid(at):
                _bytes(targets[unsafe_offset=i])[unsafe_offset=entry] &= ~(
                    UInt8(1) << bit
                )
    var offset = layout.nodes[][node].offsets[col]
    var struct_rows = List[Int](unsafe_uninit_length=max(append_count, 1))
    var struct_at = struct_rows.unsafe_ptr()
    for i in range(append_count):
        struct_at[unsafe_offset=i] = targets[unsafe_offset=i] + offset
    var child_node = layout.nodes[][node].struct_layouts[col]
    var width = validity_bytes(layout.nodes[][child_node].column_count())
    for i in range(append_count):
        _fill_ones(struct_at[unsafe_offset=i], width)
    ref children = state.vector_data[f].children
    for c in range(len(children)):
        _scatter(
            layout,
            child_node,
            c,
            state,
            children[c],
            append_sel,
            append_count,
            Int(struct_at),
        )
    _ = struct_rows^


def _scatter_list(
    layout: TupleDataLayout,
    node: Int,
    col: Int,
    state: TupleDataChunkState,
    f: Int,
    append_sel: SelectionVector,
    append_count: Int,
    rows: Int,
):
    """The heap pointer in the row and the length at the start of the list's
    heap data, then the child."""
    var source = state.vector_data[f].unified
    var targets = _ints(rows)
    var heap_at = _ints(state.heaps())
    var entry = col >> 3
    var bit = UInt8(col & 7)
    var offset = layout.nodes[][node].offsets[col]
    for i in range(append_count):
        var row = targets[unsafe_offset=i]
        var at = source.sel.get_index(append_sel.get_index(i))
        if source.validity.row_is_valid(at):
            var heap = heap_at[unsafe_offset=i]
            _store[DType.uint64](row + offset, UInt64(heap))
            _store[DType.uint64](heap, UInt64(_entry_length(source.data, at)))
            heap_at[unsafe_offset=i] = heap + 8
        else:
            _bytes(row)[unsafe_offset=entry] &= ~(UInt8(1) << bit)
    _scatter_within(
        state,
        state.vector_data[f].children[0],
        append_sel,
        append_count,
        source,
    )


def _scatter_within(
    state: TupleDataChunkState,
    f: Int,
    append_sel: SelectionVector,
    append_count: Int,
    list_data: UnifiedVectorFormat,
):
    """The within-collection scatter functions: per appended row's list, a
    validity mask then the values, in the heap."""
    var physical = state.vector_data[f].physical
    var source = state.vector_data[f].unified
    var heap_at = _ints(state.heaps())
    var size = _WITHIN_LIST_STRING_SIZE if physical == PHYSICAL_VARCHAR else (
        8 if physical == PHYSICAL_LIST else physical_size(physical)
    )
    for i in range(append_count):
        var list_idx = list_data.sel.get_index(append_sel.get_index(i))
        if not list_data.validity.row_is_valid(list_idx):
            continue
        var length = _entry_length(list_data.data, list_idx)
        if length == 0:
            continue
        var list_offset = _entry_offset(list_data.data, list_idx)
        var heap = heap_at[unsafe_offset=i]
        var mask = heap
        _fill_ones(mask, validity_bytes(length))
        heap += validity_bytes(length)
        if physical == PHYSICAL_STRUCT:
            # A struct inside a list: the struct validity; the fields follow,
            # each its own mask and values.
            for c in range(length):
                var at = source.sel.get_index(list_offset + c)
                if not source.validity.row_is_valid(at):
                    set_invalid(mask, c)
            heap_at[unsafe_offset=i] = heap
            continue
        var data_location = heap
        heap += length * size
        if physical == PHYSICAL_VARCHAR:
            for c in range(length):
                var at = source.sel.get_index(list_offset + c)
                if source.validity.row_is_valid(at):
                    var value = source.data + 16 * at
                    var bytes = Int(_load[DType.uint32](value))
                    _store[DType.uint32](
                        data_location + c * size, UInt32(bytes)
                    )
                    _copy(heap, string_data(value), bytes)
                    heap += bytes
                else:
                    set_invalid(mask, c)
        elif physical == PHYSICAL_LIST:
            # A list inside a list: the child lists' lengths.
            for c in range(length):
                var at = source.sel.get_index(list_offset + c)
                if source.validity.row_is_valid(at):
                    _store[DType.uint64](
                        data_location + c * 8,
                        UInt64(_entry_length(source.data, at)),
                    )
                else:
                    set_invalid(mask, c)
        else:
            _within_fixed_values(
                physical, source, list_offset, length, data_location, mask
            )
        heap_at[unsafe_offset=i] = heap
    if physical == PHYSICAL_STRUCT:
        ref children = state.vector_data[f].children
        for c in range(len(children)):
            _scatter_within(
                state, children[c], append_sel, append_count, list_data
            )
    elif physical == PHYSICAL_LIST:
        var child = state.vector_data[f].children[0]
        _scatter_within(
            state,
            child,
            append_sel,
            append_count,
            state.vector_data[child].combined,
        )


@always_inline
def _within_fixed_values(
    physical: Int,
    source: UnifiedVectorFormat,
    list_offset: Int,
    length: Int,
    data_location: Int,
    mask: Int,
):
    """The value loop for fixed-width child values of a list."""
    if physical == PHYSICAL_BOOL:
        _within_fixed_loop[DType.uint8](
            source, list_offset, length, data_location, mask
        )
        return
    comptime for k in range(len(_FIXED)):
        comptime D = _FIXED[k]
        if physical == physical_of[D]():
            _within_fixed_loop[D](
                source, list_offset, length, data_location, mask
            )


@always_inline
def _within_fixed_loop[
    D: DType
](
    source: UnifiedVectorFormat,
    list_offset: Int,
    length: Int,
    data_location: Int,
    mask: Int,
):
    var data = source.values[D]()
    comptime size = size_of[Scalar[D]]()
    for c in range(length):
        var at = source.sel.get_index(list_offset + c)
        if source.validity.row_is_valid(at):
            _store[D](data_location + c * size, data[unsafe_offset=at])
        else:
            set_invalid(mask, c)


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------


def gather_column(
    layout: TupleDataLayout,
    rows: Int,
    scan_sel: SelectionVector,
    scan_count: Int,
    column_id: Int,
    mut result: DataChunk,
    out_column: Int,
    target_sel: SelectionVector,
    mut arena: ColumnArena,
) raises:
    """One column of the rows `scan_sel` picks from the row pointers at `rows`
    into vector `out_column` of `result`, at the positions `target_sel`
    gives."""
    ref top = layout.nodes[][0]
    var physical = top.physicals[column_id]
    if physical != PHYSICAL_LIST and physical != PHYSICAL_STRUCT:
        var vector = result.writable(out_column, physical)
        var bits = 0
        if top.can_have_null:
            bits = result.validity_bits(out_column)
        _gather_fixed_dispatch(
            physical,
            rows,
            scan_sel,
            scan_count,
            top.offsets[column_id],
            column_id,
            top.can_have_null,
            vector.data,
            bits,
            target_sel,
        )
        return
    var target_count = scan_count
    if target_sel.is_set():
        target_count = 0
        for i in range(scan_count):
            target_count = max(target_count, target_sel.get_index(i) + 1)
    var nodes = List[_GatherNode]()
    var root = _make_nodes(nodes, top.types[column_id])
    _reserve(nodes, root, target_count)
    _gather(
        layout,
        0,
        column_id,
        rows,
        scan_sel,
        scan_count,
        nodes,
        root,
        target_sel,
    )
    var series = _node_series(nodes, root, target_count, "")
    var handle = arena.add(series)
    result.data[out_column] = flat_over(arena, handle, 0)


struct TupleDataHeaps(Copyable, Movable):
    """A collection's heap blocks in address order, each a buffer that
    keeps its block alive. A string column gathered from the rows
    references their strings in place: string_t's pointer becomes the
    block and offset of a Utf8View descriptor, and the column holds the
    blocks, so the bytes stay valid after the collection is gone without
    being copied."""

    var starts: List[Int]
    var ends: List[Int]
    var buffers: List[_Buffer[UInt8]]
    var bytes: Int

    def __init__(out self):
        self.starts = List[Int]()
        self.ends = List[Int]()
        self.buffers = List[_Buffer[UInt8]]()
        self.bytes = 0

    def __init__(out self, blocks: List[ArcPointer[List[UInt8]]]):
        """The heaps `blocks`, put in address order."""
        self.starts = List[Int](capacity=len(blocks))
        self.ends = List[Int](capacity=len(blocks))
        self.buffers = List[_Buffer[UInt8]](capacity=len(blocks))
        self.bytes = 0
        var at = Dict[Int, Int]()
        for b in range(len(blocks)):
            var address = Int(blocks[b][].unsafe_ptr())
            at[address] = b
            self.starts.append(address)
        sort(self.starts)
        for b in range(len(self.starts)):
            ref block = blocks[at.get(self.starts[b], 0)]
            self.ends.append(self.starts[b] + len(block[]))
            self.buffers.append(_Buffer[UInt8](block))
            self.bytes += len(block[])

    def add_buffers(mut self, extra: List[_Buffer[UInt8]]):
        """Add `extra`, buffers that rows reference strings in (a
        referenced vector's bytes), kept in address order; a buffer
        already held, or an empty one, is skipped."""
        var added = False
        for b in range(len(extra)):
            if len(extra[b]) == 0:
                continue
            var address = Int(extra[b].unsafe_ptr())
            var held = False
            for s in range(len(self.starts)):
                held = held or self.starts[s] == address
            if held:
                continue
            self.starts.append(address)
            self.ends.append(address + len(extra[b]))
            self.buffers.append(extra[b].copy())
            self.bytes += len(extra[b])
            added = True
        if not added:
            return
        # Insertion sort by start address: few buffers.
        for i in range(1, len(self.starts)):
            var j = i
            while j > 0 and self.starts[j - 1] > self.starts[j]:
                self.starts.swap_elements(j - 1, j)
                self.ends.swap_elements(j - 1, j)
                self.buffers.swap_elements(j - 1, j)
                j -= 1

    @always_inline
    def find(self, address: Int, hint: Int) -> Int:
        """The block holding `address`; `hint`, the last one found, is
        checked first (rows gathered in order share blocks)."""
        if self.starts[hint] <= address and address < self.ends[hint]:
            return hint
        var low = 0
        var high = len(self.starts) - 1
        while low < high:
            var mid = (low + high + 1) >> 1
            if self.starts[mid] <= address:
                low = mid
            else:
                high = mid - 1
        return low


def _views_over_heaps(
    heaps: TupleDataHeaps,
    var views: List[StringView],
    count: Int,
    var bits: List[UInt8],
) -> StringColumn:
    """The `count` gathered string_t values in `views` as Utf8View
    descriptors, in place: an inlined string is already one, a longer
    one's pointer becomes its heap block and offset. `bits`: the validity,
    empty when every row is valid."""
    return _views_at(heaps, views^, count, 0, len(views), bits^)


def _views_at(
    heaps: TupleDataHeaps,
    var views: List[StringView],
    count: Int,
    slots: Int,
    size: Int,
    var bits: List[UInt8],
) -> StringColumn:
    """`_views_over_heaps` for the `count` entries at the positions
    `slots` lists (0: the first `count`) of `size` views."""
    var at = Int(views.unsafe_ptr())
    var slot_at = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=slots)
    var hint = 0
    for i in range(count):
        var entry = at + 16 * (
            Int(slot_at[unsafe_offset=i]) if slots != 0 else i
        )
        if _load[DType.uint32](entry) > UInt32(STRING_INLINE_BYTES):
            var address = Int(_load[DType.uint64](entry + 8))
            hint = heaps.find(address, hint)
            _store[DType.uint32](entry + 8, UInt32(hint))
            _store[DType.uint32](
                entry + 12, UInt32(address - heaps.starts[hint])
            )
    return StringColumn(
        StringViewStorage(
            views^, heaps.buffers.copy(), bits^, size, heaps.bytes
        )
    )


def gather_rows_series(
    layout: TupleDataLayout,
    rows: Int,
    count: Int,
    column_id: Int,
    var name: String,
    dtype: DataType,
    heaps: TupleDataHeaps,
) raises -> Series:
    """The `count` rows of column `column_id` at the row pointers at `rows`,
    a new at-rest column typed `dtype` of `count` rows (for an
    operator reading columns rather than vectors; no 2,048 limit, the row
    pointers are the only limit). A string column references `heaps`, the
    collection's heap blocks, without copying its bytes."""
    ref top = layout.nodes[][0]
    var physical = top.physicals[column_id]
    if physical == PHYSICAL_LIST or physical == PHYSICAL_STRUCT:
        var chunk = DataChunk(1)
        var arena = ColumnArena()
        gather_column(
            layout,
            rows,
            SelectionVector(),
            count,
            column_id,
            chunk,
            0,
            SelectionVector(),
            arena,
        )
        return _typed(arena.columns[chunk.data[0].source].renamed(name^), dtype)
    var bits = List[UInt8](length=(count + 7) // 8 + 1, fill=255)
    var bits_at = Int(bits.unsafe_ptr())
    var valid_of = bits_at if top.can_have_null else 0
    if physical == PHYSICAL_VARCHAR:
        var views = List[StringView](unsafe_uninit_length=max(count, 1))
        _gather_fixed_dispatch(
            physical,
            rows,
            SelectionVector(),
            count,
            top.offsets[column_id],
            column_id,
            top.can_have_null,
            Int(views.unsafe_ptr()),
            valid_of,
            SelectionVector(),
        )
        if not _any_invalid(bits_at, count):
            bits = List[UInt8]()
        return _typed(
            Series(name^, _views_over_heaps(heaps, views^, count, bits^)),
            dtype,
        )
    if physical == PHYSICAL_BOOL:
        var buffer = List[UInt8](unsafe_uninit_length=max(count, 1))
        _gather_fixed_dispatch(
            physical,
            rows,
            SelectionVector(),
            count,
            top.offsets[column_id],
            column_id,
            top.can_have_null,
            Int(buffer.unsafe_ptr()),
            valid_of,
            SelectionVector(),
        )
        var values = List[Bool](capacity=count)
        for i in range(count):
            values.append(buffer[i] != 0)
        var valid = List[Bool](capacity=count)
        var any_null = False
        for i in range(count):
            var ok = _bit_valid(bits_at, i)
            any_null = any_null or not ok
            valid.append(ok)
        # `bits` is read only through its address: without this it dies
        # before the gather and `buffer` reuses its memory.
        _ = bits^
        if any_null:
            return _typed(Series(name^, BoolColumn(values, valid)), dtype)
        return _typed(Series(name^, BoolColumn(values)), dtype)
    comptime for k in range(len(_FIXED)):
        comptime D = _FIXED[k]
        if physical == physical_of[D]():
            var values = List[Scalar[D]](unsafe_uninit_length=count)
            _gather_fixed_dispatch(
                physical,
                rows,
                SelectionVector(),
                count,
                top.offsets[column_id],
                column_id,
                top.can_have_null,
                Int(values.unsafe_ptr()),
                valid_of,
                SelectionVector(),
            )
            if top.can_have_null:
                if _any_invalid(bits_at, count):
                    return _typed(
                        Series(
                            name^,
                            Column[Scalar[D]](values=values^, bits=bits^),
                        ),
                        dtype,
                    )
            return _typed(Series(name^, Column[Scalar[D]](values^)), dtype)
    raise Error("tuple data: no gather for this column type")


def _any_invalid(bits: Int, count: Int) -> Bool:
    """Whether any of the first `count` bits at `bits` is clear."""
    var at = _bytes(bits)
    for b in range(count // 8):
        if at[unsafe_offset=b] != 255:
            return True
    for i in range(count & ~7, count):
        if not _bit_valid(bits, i):
            return True
    return False


def pack_bytes(bytes: Int, size: Int) -> List[UInt8]:
    """The `size` bytes at `bytes`, each 0 or 1, as bits (eight to a byte,
    byte i's value at bit i)."""
    var bits = List[UInt8](length=(size + 7) // 8, fill=0)
    var target = bits.unsafe_ptr()
    var whole = size // 8
    for w in range(whole):
        # Each byte is 0 or 1: the multiply gathers the low bit of each of
        # the eight bytes into the top byte, byte i at bit i.
        var eight = _load[DType.uint64](bytes + 8 * w)
        target[unsafe_offset=w] = UInt8((eight * 0x0102040810204080) >> 56)
    for i in range(8 * whole, size):
        if _bytes(bytes)[unsafe_offset=i] != 0:
            target[unsafe_offset=i >> 3] |= UInt8(1) << UInt8(i & 7)
    return bits^


struct NumberedGather(Movable):
    """Column `column_id` of a collection's rows gathered into a column of
    `size` entries, each row at the entry its number names, by tasks that
    each gather their own rows. No two rows share a number, so no two
    tasks write the same entry, and every write is a whole entry: a
    value, or a validity byte (the bits are packed once every task is
    done). Strings reference the collection's heaps."""

    var column_id: Int
    var physical: Int
    var offset: Int
    var nullable: Bool
    var size: Int
    var name: String
    var dtype: DataType
    # A fixed-width column's values, written in place.
    var values: Optional[Series]
    # Strings: each entry's string_t (the vector layout) and Utf8View
    # descriptor; Booleans: a byte per entry.
    var decoded: List[UInt8]
    var views: List[StringView]
    # A validity byte per entry, and per task whether its rows held a null.
    var valid: List[UInt8]
    var nulls: List[UInt8]
    var target: Int

    def __init__(
        out self,
        layout: TupleDataLayout,
        column_id: Int,
        var name: String,
        dtype: DataType,
        size: Int,
        tasks: Int,
        every_entry: Bool,
    ) raises:
        """The column's storage, uninitialized where `every_entry` (every
        entry gets a row); otherwise the entries no row fills read as
        valid and empty."""
        ref top = layout.nodes[][0]
        self.column_id = column_id
        self.physical = top.physicals[column_id]
        self.offset = top.offsets[column_id]
        self.nullable = top.can_have_null
        self.size = size
        self.name = name^
        self.dtype = dtype
        self.values = None
        self.decoded = List[UInt8]()
        self.views = List[StringView]()
        self.valid = List[UInt8]()
        self.nulls = List[UInt8](length=tasks, fill=0)
        self.target = 0
        if self.nullable:
            if every_entry:
                self.valid = List[UInt8](unsafe_uninit_length=size)
            else:
                self.valid = List[UInt8](length=size, fill=1)
        if self.physical == PHYSICAL_VARCHAR:
            if every_entry:
                self.decoded = List[UInt8](unsafe_uninit_length=16 * size)
                self.views = List[StringView](unsafe_uninit_length=size)
            else:
                self.decoded = List[UInt8](length=16 * size, fill=0)
                self.views = List[StringView](
                    length=size, fill=StringView(0, 0, 0, 0)
                )
            self.target = Int(self.views.unsafe_ptr())
            return
        if self.physical == PHYSICAL_BOOL:
            if every_entry:
                self.decoded = List[UInt8](unsafe_uninit_length=size)
            else:
                self.decoded = List[UInt8](length=size, fill=0)
            self.target = Int(self.decoded.unsafe_ptr())
            return
        comptime for k in range(len(_FIXED)):
            comptime D = _FIXED[k]
            if self.physical == physical_of[D]():
                var column = Column[Scalar[D]](
                    List[Scalar[D]](unsafe_uninit_length=size)
                )
                self.target = Int(column._ptr())
                self.values = Series(self.name, column^)
                return
        raise Error("tuple data: no numbered gather for this column type")

    def gather(
        self,
        rows: Int,
        numbers: Int,
        count: Int,
        task: Int,
        heaps: TupleDataHeaps,
    ):
        """Gather the `count` rows at the pointers at `rows`, row i to
        entry `numbers[i]` (UInt32s); `task` is the gathering task."""
        var sources = _ints(rows)
        var number_at = Pointer[UInt32, MutAnyOrigin](
            unsafe_from_address=numbers
        )
        var valid = Int(self.valid.unsafe_ptr())
        var entry = self.column_id >> 3
        var bit = UInt8(self.column_id & 7)
        var any_null = False
        if self.physical == PHYSICAL_VARCHAR:
            var decoded = Int(self.decoded.unsafe_ptr())
            var hint = 0
            for i in range(count):
                var row = sources[unsafe_offset=i]
                var at = Int(number_at[unsafe_offset=i])
                var low = _load[DType.uint64](row + self.offset)
                var high = _load[DType.uint64](row + self.offset + 8)
                if self.nullable:
                    var ok = (_bytes(row)[unsafe_offset=entry] >> bit) & 1
                    _bytes(valid)[unsafe_offset=at] = ok
                    if ok == 0:
                        any_null = True
                        low = 0
                        high = 0
                _store[DType.uint64](decoded + 16 * at, low)
                _store[DType.uint64](decoded + 16 * at + 8, high)
                var view = self.target + 16 * at
                _store[DType.uint64](view, low)
                if UInt32(low & 0xFFFFFFFF) > UInt32(STRING_INLINE_BYTES):
                    # A longer string's pointer becomes its heap block and
                    # offset.
                    var address = Int(high)
                    hint = heaps.find(address, hint)
                    _store[DType.uint32](view + 8, UInt32(hint))
                    _store[DType.uint32](
                        view + 12, UInt32(address - heaps.starts[hint])
                    )
                else:
                    _store[DType.uint64](view + 8, high)
        elif self.physical == PHYSICAL_BOOL:
            for i in range(count):
                var row = sources[unsafe_offset=i]
                var at = Int(number_at[unsafe_offset=i])
                var value = UInt8(1) if _bytes(row)[
                    unsafe_offset=self.offset
                ] != 0 else UInt8(0)
                if self.nullable:
                    var ok = (_bytes(row)[unsafe_offset=entry] >> bit) & 1
                    _bytes(valid)[unsafe_offset=at] = ok
                    if ok == 0:
                        any_null = True
                        value = 0
                _bytes(self.target)[unsafe_offset=at] = value
        else:
            comptime for k in range(len(_FIXED)):
                comptime D = _FIXED[k]
                if self.physical == physical_of[D]():
                    var out = Pointer[Scalar[D], MutAnyOrigin](
                        unsafe_from_address=self.target
                    )
                    for i in range(count):
                        var row = sources[unsafe_offset=i]
                        var at = Int(number_at[unsafe_offset=i])
                        var value = _load[D](row + self.offset)
                        if self.nullable:
                            var ok = (
                                _bytes(row)[unsafe_offset=entry] >> bit
                            ) & 1
                            _bytes(valid)[unsafe_offset=at] = ok
                            if ok == 0:
                                any_null = True
                                # A null's row value is the type's minimum
                                # (`_null_value`), which a reader that
                                # checks decimal precision on every value
                                # it widens would reject.
                                value = 0
                        out[unsafe_offset=at] = value
        if any_null:
            _bytes(Int(self.nulls.unsafe_ptr()))[unsafe_offset=task] = 1

    def finish(
        mut self, heaps: TupleDataHeaps, mut decoded: List[UInt8]
    ) raises -> Series:
        """The gathered column, once every task is done; `decoded` is set
        to its values in the vector layout (strings' string_t, Booleans'
        bytes; empty for other columns)."""
        var any_null = False
        for t in range(len(self.nulls)):
            any_null = any_null or self.nulls[t] != 0
        var bits = List[UInt8]()
        if any_null:
            bits = pack_bytes(Int(self.valid.unsafe_ptr()), self.size)
        self.valid = List[UInt8]()
        if self.physical == PHYSICAL_VARCHAR:
            swap(decoded, self.decoded)
            var views = List[StringView]()
            swap(views, self.views)
            return _typed(
                Series(
                    self.name,
                    StringColumn(
                        StringViewStorage(
                            views^,
                            heaps.buffers.copy(),
                            bits^,
                            self.size,
                            heaps.bytes,
                        )
                    ),
                ),
                self.dtype,
            )
        if self.physical == PHYSICAL_BOOL:
            var packed = pack_bytes(Int(self.decoded.unsafe_ptr()), self.size)
            swap(decoded, self.decoded)
            if not any_null:
                bits = List[UInt8](length=(self.size + 7) // 8, fill=255)
            return _typed(
                Series(
                    self.name,
                    BoolColumn(values=packed^, bits=bits^, length=self.size),
                ),
                self.dtype,
            )
        decoded = List[UInt8]()
        var values = self.values.take()
        if any_null:
            comptime for k in range(len(_FIXED)):
                comptime D = _FIXED[k]
                if self.physical == physical_of[D]():
                    ref column = values._data[Column[Scalar[D]]]
                    return _typed(
                        Series(
                            self.name,
                            Column[Scalar[D]]._from_buffers(
                                column._data, _Buffer(bits^), 0, self.size
                            ),
                        ),
                        self.dtype,
                    )
        return _typed(values^, self.dtype)


def _gather_fixed_dispatch(
    physical: Int,
    rows: Int,
    scan_sel: SelectionVector,
    scan_count: Int,
    offset_in_row: Int,
    col: Int,
    can_have_null: Bool,
    target: Int,
    bits: Int,
    target_sel: SelectionVector,
):
    if physical == PHYSICAL_BOOL:
        _gather_fixed[DType.uint8](
            rows,
            scan_sel,
            scan_count,
            offset_in_row,
            col,
            can_have_null,
            target,
            bits,
            target_sel,
        )
        return
    if physical == PHYSICAL_VARCHAR:
        # A string_t row entry is the vector's string_t: inlined, or
        # pointing at its bytes in the heap.
        _gather_fixed[DType.int128](
            rows,
            scan_sel,
            scan_count,
            offset_in_row,
            col,
            can_have_null,
            target,
            bits,
            target_sel,
        )
        return
    comptime for k in range(len(_FIXED)):
        comptime D = _FIXED[k]
        if physical == physical_of[D]():
            _gather_fixed[D](
                rows,
                scan_sel,
                scan_count,
                offset_in_row,
                col,
                can_have_null,
                target,
                bits,
                target_sel,
            )


def _gather_fixed[
    D: DType
](
    rows: Int,
    scan_sel: SelectionVector,
    scan_count: Int,
    offset_in_row: Int,
    col: Int,
    can_have_null: Bool,
    target: Int,
    bits: Int,
    target_sel: SelectionVector,
):
    """Gather fixed-width values. As with scatter, the loop without selections
    (and, here, without nulls) is the one specialized."""
    if not scan_sel.is_set() and not target_sel.is_set():
        if can_have_null:
            _gather_fixed_loop[D, True, True](
                rows,
                scan_sel,
                scan_count,
                offset_in_row,
                col,
                target,
                bits,
                target_sel,
            )
        else:
            _gather_fixed_loop[D, True, False](
                rows,
                scan_sel,
                scan_count,
                offset_in_row,
                col,
                target,
                bits,
                target_sel,
            )
    else:
        _gather_fixed_loop[D, False, True](
            rows,
            scan_sel,
            scan_count,
            offset_in_row,
            col,
            target,
            bits,
            target_sel,
        )


@always_inline
def _gather_fixed_loop[
    D: DType, PLAIN: Bool, NULLS: Bool
](
    rows: Int,
    scan_sel: SelectionVector,
    scan_count: Int,
    offset_in_row: Int,
    col: Int,
    target: Int,
    bits: Int,
    target_sel: SelectionVector,
):
    var sources = _ints(rows)
    var out = Pointer[Scalar[D], MutAnyOrigin](unsafe_from_address=target)
    var entry = col >> 3
    var bit = UInt8(col & 7)
    for i in range(scan_count):
        var source_idx: Int
        var target_idx: Int
        comptime if PLAIN:
            source_idx = i
            target_idx = i
        else:
            source_idx = scan_sel.get_index(i)
            target_idx = target_sel.get_index(i)
        var row = sources[unsafe_offset=source_idx]
        out[unsafe_offset=target_idx] = _load[D](row + offset_in_row)
        comptime if NULLS:
            if bits != 0 and (
                (_bytes(row)[unsafe_offset=entry] >> bit) & 1 == 0
            ):
                set_invalid(bits, target_idx)
                # A null's row value is the type's minimum (`_null_value`),
                # which a reader that checks decimal precision on every value it
                # widens would reject.
                out[unsafe_offset=target_idx] = 0


struct _GatherNode(Copyable, Movable):
    """The vector a nested column's level is gathered into: values (or list
    entries), a validity mask, and for a list the child's size."""

    var dtype: DataType
    var physical: Int
    var width: Int
    var values: List[UInt8]
    var bits: List[UInt8]
    var entries: List[UInt64]
    var list_size: Int
    var capacity: Int
    var children: List[Int]

    def __init__(out self, dtype: DataType, physical: Int):
        self.dtype = dtype
        self.physical = physical
        self.width = 0
        if physical == PHYSICAL_VARCHAR:
            self.width = 16
        elif physical != PHYSICAL_LIST and physical != PHYSICAL_STRUCT:
            self.width = physical_size(physical)
        self.values = List[UInt8]()
        self.bits = List[UInt8]()
        self.entries = List[UInt64]()
        self.list_size = 0
        self.capacity = 0
        self.children = List[Int]()

    def reserve(mut self, count: Int):
        """Room for `count` rows, new rows valid; capacity at least doubles, so
        repeated appends cost amortized constant time."""
        if count <= self.capacity:
            return
        var capacity = max(count, 2 * self.capacity)
        self.values.resize(capacity * self.width, 0)
        self.bits.resize((capacity + 7) // 8, 255)
        if self.physical == PHYSICAL_LIST:
            self.entries.resize(2 * capacity, 0)
        self.capacity = capacity


def _make_nodes(mut nodes: List[_GatherNode], dtype: DataType) raises -> Int:
    var index = len(nodes)
    var physical = physical_of_dtype(dtype)
    nodes.append(_GatherNode(dtype, physical))
    if physical == PHYSICAL_STRUCT:
        var children = List[Int]()
        for field in dtype.field_dtypes():
            children.append(_make_nodes(nodes, field))
        nodes[index].children = children^
    elif physical == PHYSICAL_LIST:
        var child = _make_nodes(nodes, dtype.inner())
        nodes[index].children = [child]
    return index


def _reserve(mut nodes: List[_GatherNode], index: Int, count: Int):
    """Reserve a level and, for a struct, its fields (row-aligned)."""
    nodes[index].reserve(count)
    if nodes[index].physical == PHYSICAL_STRUCT:
        var children = nodes[index].children.copy()
        for c in range(len(children)):
            _reserve(nodes, children[c], count)


def _gather(
    layout: TupleDataLayout,
    node: Int,
    col: Int,
    rows: Int,
    scan_sel: SelectionVector,
    scan_count: Int,
    mut nodes: List[_GatherNode],
    target: Int,
    target_sel: SelectionVector,
) raises:
    """The gather function of column `col` of layout `node` into level
    `target`."""
    ref layout_node = layout.nodes[][node]
    var physical = layout_node.physicals[col]
    var offset = layout_node.offsets[col]
    var can_have_null = layout_node.can_have_null
    if physical == PHYSICAL_STRUCT:
        _gather_struct(
            layout,
            node,
            col,
            rows,
            scan_sel,
            scan_count,
            nodes,
            target,
            target_sel,
        )
    elif physical == PHYSICAL_LIST:
        _gather_list(
            layout,
            node,
            col,
            rows,
            scan_sel,
            scan_count,
            nodes,
            target,
            target_sel,
        )
    else:
        _gather_fixed_dispatch(
            physical,
            rows,
            scan_sel,
            scan_count,
            offset,
            col,
            can_have_null,
            Int(nodes[target].values.unsafe_ptr()),
            Int(nodes[target].bits.unsafe_ptr()),
            target_sel,
        )


def _gather_struct(
    layout: TupleDataLayout,
    node: Int,
    col: Int,
    rows: Int,
    scan_sel: SelectionVector,
    scan_count: Int,
    mut nodes: List[_GatherNode],
    target: Int,
    target_sel: SelectionVector,
) raises:
    """The struct's validity, then its fields from the nested rows."""
    ref layout_node = layout.nodes[][node]
    var offset = layout_node.offsets[col]
    var can_have_null = layout_node.can_have_null
    var sources = _ints(rows)
    var bits = Int(nodes[target].bits.unsafe_ptr())
    var entry = col >> 3
    var bit = UInt8(col & 7)
    # The nested row pointers sit where the row pointers do (by source
    # index), so the fields read them through the same selection.
    var source_count = scan_count
    if scan_sel.is_set():
        source_count = 0
        for i in range(scan_count):
            source_count = max(source_count, scan_sel.get_index(i) + 1)
    var struct_rows = List[Int](length=max(source_count, 1), fill=0)
    var struct_at = struct_rows.unsafe_ptr()
    for i in range(scan_count):
        var source_idx = scan_sel.get_index(i)
        var row = sources[unsafe_offset=source_idx]
        if can_have_null and (
            (_bytes(row)[unsafe_offset=entry] >> bit) & 1 == 0
        ):
            set_invalid(bits, target_sel.get_index(i))
        struct_at[unsafe_offset=source_idx] = row + offset
    var child_node = layout_node.struct_layouts[col]
    var children = nodes[target].children.copy()
    for c in range(len(children)):
        _gather(
            layout,
            child_node,
            c,
            Int(struct_at),
            scan_sel,
            scan_count,
            nodes,
            children[c],
            target_sel,
        )
    _ = struct_rows^


def _gather_list(
    layout: TupleDataLayout,
    node: Int,
    col: Int,
    rows: Int,
    scan_sel: SelectionVector,
    scan_count: Int,
    mut nodes: List[_GatherNode],
    target: Int,
    target_sel: SelectionVector,
) raises:
    """List entries from the lengths in the heap, then the child from the heap
    data after them."""
    ref layout_node = layout.nodes[][node]
    var offset = layout_node.offsets[col]
    var can_have_null = layout_node.can_have_null
    var sources = _ints(rows)
    var entry = col >> 3
    var bit = UInt8(col & 7)
    var heaps = List[Int](length=max(scan_count, 1), fill=0)
    var heap_at = heaps.unsafe_ptr()
    var entries = nodes[target].entries.unsafe_ptr()
    var bits = Int(nodes[target].bits.unsafe_ptr())
    var list_size_before = nodes[target].list_size
    var target_offset = list_size_before
    for i in range(scan_count):
        var row = sources[unsafe_offset=scan_sel.get_index(i)]
        var target_idx = target_sel.get_index(i)
        if not can_have_null or (
            (_bytes(row)[unsafe_offset=entry] >> bit) & 1 == 1
        ):
            var heap = Int(_load[DType.uint64](row + offset))
            var length = Int(_load[DType.uint64](heap))
            heap_at[unsafe_offset=i] = heap + 8
            entries[unsafe_offset=2 * target_idx] = UInt64(target_offset)
            entries[unsafe_offset=2 * target_idx + 1] = UInt64(length)
            target_offset += length
        else:
            set_invalid(bits, target_idx)
    nodes[target].list_size = target_offset
    var child = nodes[target].children[0]
    _reserve(nodes, child, target_offset)
    _gather_within(
        nodes,
        child,
        Int(heap_at),
        list_size_before,
        scan_count,
        target_sel,
        Int(nodes[target].entries.unsafe_ptr()),
        bits,
    )
    _ = heaps^


def _gather_within(
    mut nodes: List[_GatherNode],
    target: Int,
    heaps: Int,
    list_size_before: Int,
    scan_count: Int,
    target_sel: SelectionVector,
    list_entries: Int,
    list_bits: Int,
) raises:
    """The within-collection gather functions: per scanned row's list, the
    mask and values from the heap into the child level, from
    `list_size_before` on."""
    var physical = nodes[target].physical
    var heap_at = _ints(heaps)
    var bits = Int(nodes[target].bits.unsafe_ptr())
    var values = Int(nodes[target].values.unsafe_ptr())
    var width = nodes[target].width
    var target_offset = list_size_before
    if physical == PHYSICAL_LIST:
        _collection_within_gather(
            nodes,
            target,
            heaps,
            list_size_before,
            scan_count,
            target_sel,
            list_entries,
            list_bits,
        )
        return
    for i in range(scan_count):
        var target_idx = target_sel.get_index(i)
        if not _bit_valid(list_bits, target_idx):
            continue
        var length = _entry_length(list_entries, target_idx)
        if length == 0:
            continue
        var heap = heap_at[unsafe_offset=i]
        var mask = heap
        heap += validity_bytes(length)
        if physical == PHYSICAL_STRUCT:
            for c in range(length):
                if not _bit_valid(mask, c):
                    set_invalid(bits, target_offset + c)
        elif physical == PHYSICAL_VARCHAR:
            var data_location = heap
            heap += length * _WITHIN_LIST_STRING_SIZE
            for c in range(length):
                if _bit_valid(mask, c):
                    var size = Int(
                        _load[DType.uint32](
                            data_location + c * _WITHIN_LIST_STRING_SIZE
                        )
                    )
                    Pointer[StringT, MutAnyOrigin](
                        unsafe_from_address=values + 16 * (target_offset + c)
                    )[] = StringT.make(heap, size)
                    heap += size
                else:
                    set_invalid(bits, target_offset + c)
        else:
            var data_location = heap
            heap += length * width
            for c in range(length):
                if _bit_valid(mask, c):
                    _copy(
                        values + width * (target_offset + c),
                        data_location + c * width,
                        width,
                    )
                else:
                    set_invalid(bits, target_offset + c)
        heap_at[unsafe_offset=i] = heap
        target_offset += length
    if physical == PHYSICAL_STRUCT:
        var children = nodes[target].children.copy()
        for c in range(len(children)):
            _gather_within(
                nodes,
                children[c],
                heaps,
                list_size_before,
                scan_count,
                target_sel,
                list_entries,
                list_bits,
            )


def _collection_within_gather(
    mut nodes: List[_GatherNode],
    target: Int,
    heaps: Int,
    list_size_before: Int,
    scan_count: Int,
    target_sel: SelectionVector,
    list_entries: Int,
    list_bits: Int,
) raises:
    """A list in a list. The child lists' entries come from their lengths in the
    heap; one combined entry per scanned row lets the grandchild gather the same
    way."""
    var heap_at = _ints(heaps)
    var target_count = 0
    for i in range(scan_count):
        target_count = max(target_count, target_sel.get_index(i) + 1)
    var combined = List[UInt64](length=2 * max(target_count, 1), fill=0)
    var combined_at = combined.unsafe_ptr()
    var entries = nodes[target].entries.unsafe_ptr()
    var bits = Int(nodes[target].bits.unsafe_ptr())
    var child_list_size_before = nodes[target].list_size
    var target_offset = list_size_before
    var target_child_offset = child_list_size_before
    for i in range(scan_count):
        var target_idx = target_sel.get_index(i)
        if not _bit_valid(list_bits, target_idx):
            continue
        combined_at[unsafe_offset=2 * target_idx] = UInt64(target_child_offset)
        var length = _entry_length(list_entries, target_idx)
        if length == 0:
            combined_at[unsafe_offset=2 * target_idx + 1] = 0
            continue
        var heap = heap_at[unsafe_offset=i]
        var mask = heap
        heap += validity_bytes(length)
        var data_location = heap
        heap += length * 8
        for c in range(length):
            var e = target_offset + c
            if _bit_valid(mask, c):
                var child_length = Int(
                    _load[DType.uint64](data_location + c * 8)
                )
                entries[unsafe_offset=2 * e] = UInt64(target_child_offset)
                entries[unsafe_offset=2 * e + 1] = UInt64(child_length)
                target_child_offset += child_length
            else:
                set_invalid(bits, e)
        combined_at[unsafe_offset=2 * target_idx + 1] = (
            UInt64(target_child_offset)
            - combined_at[unsafe_offset=2 * target_idx]
        )
        heap_at[unsafe_offset=i] = heap
        target_offset += length
    nodes[target].list_size = target_child_offset
    var child = nodes[target].children[0]
    _reserve(nodes, child, target_child_offset)
    _gather_within(
        nodes,
        child,
        heaps,
        child_list_size_before,
        scan_count,
        target_sel,
        Int(combined_at),
        list_bits,
    )
    _ = combined^


def _node_series(
    nodes: List[_GatherNode], index: Int, length: Int, var name: String
) raises -> Series:
    """The `length` rows of a gathered level as an at-rest column."""
    ref node = nodes[index]
    var bits = Int(node.bits.unsafe_ptr())
    var valid = List[Bool](capacity=length)
    var any_null = False
    for i in range(length):
        var ok = _bit_valid(bits, i)
        any_null = any_null or not ok
        valid.append(ok)
    var packed = List[UInt8]()
    if any_null:
        packed = List[UInt8](length=(length + 7) // 8, fill=0)
        _copy(Int(packed.unsafe_ptr()), bits, (length + 7) // 8)
    var physical = node.physical
    if physical == PHYSICAL_STRUCT:
        var names = node.dtype.field_names()
        var fields = List[Series]()
        for c in range(len(node.children)):
            fields.append(
                _node_series(nodes, node.children[c], length, names[c])
            )
        return Series(name^, StructColumn(fields^, packed^, length=length))
    if physical == PHYSICAL_LIST:
        # Entries are in row order unless a target selection reordered
        # them; then the child is taken in row order.
        var offsets = List[Int64](capacity=length + 1)
        offsets.append(0)
        var running = 0
        var in_order = True
        var entries = node.entries.unsafe_ptr()
        for i in range(length):
            if valid[i]:
                if Int(entries[unsafe_offset=2 * i]) != running:
                    in_order = False
                running += Int(entries[unsafe_offset=2 * i + 1])
            offsets.append(Int64(running))
        var child = _node_series(
            nodes, node.children[0], node.list_size, "item"
        )
        if not in_order:
            var picked = List[Int](capacity=running)
            for i in range(length):
                if valid[i]:
                    var start = Int(entries[unsafe_offset=2 * i])
                    for k in range(Int(entries[unsafe_offset=2 * i + 1])):
                        picked.append(start + k)
            child = child.take(picked)
        return Series(name^, ListColumn(offsets^, child^, packed^))
    var result: Series
    if physical == PHYSICAL_VARCHAR:
        var builder = StringBuilder(length)
        var values = Int(node.values.unsafe_ptr())
        for i in range(length):
            if not valid[i]:
                builder.append_null()
                continue
            var entry = values + 16 * i
            var size = Int(_load[DType.uint32](entry))
            builder.append(
                StringSlice[ImmutAnyOrigin](
                    unsafe_from_utf8=Span[UInt8, ImmutAnyOrigin](
                        unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                            unsafe_from_address=string_data(entry)
                        ),
                        length=size,
                    )
                )
            )
        result = Series(name^, builder^.finish())
    elif physical == PHYSICAL_BOOL:
        var values = List[Bool](capacity=length)
        for i in range(length):
            values.append(node.values[i] != 0)
        if any_null:
            result = Series(name^, BoolColumn(values, valid))
        else:
            result = Series(name^, BoolColumn(values))
    else:
        result = _fixed_series(node, length, valid, any_null, name^)
    if result.dtype() != node.dtype:
        return result.with_dtype(node.dtype)
    return result^


def _fixed_series(
    node: _GatherNode,
    length: Int,
    valid: List[Bool],
    any_null: Bool,
    var name: String,
) raises -> Series:
    comptime for k in range(len(_FIXED)):
        comptime D = _FIXED[k]
        if node.physical == physical_of[D]():
            var values = List[Scalar[D]](unsafe_uninit_length=length)
            if length > 0:
                _copy(
                    Int(values.unsafe_ptr()),
                    Int(node.values.unsafe_ptr()),
                    length * size_of[Scalar[D]](),
                )
            if any_null:
                return Series(name^, Column[Scalar[D]](values^, valid))
            return Series(name^, Column[Scalar[D]](values^))
    raise Error("tuple data: no column for this physical type")
