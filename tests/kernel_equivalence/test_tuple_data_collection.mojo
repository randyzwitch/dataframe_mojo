"""Round-trip every supported column type through fixed-layout row storage,
checking payloads, validity, nested children and collection ownership
independently."""
# contract-group: kernel.tuple_data
from std.testing import assert_equal, assert_true

from dataframe import Column, Series, StringColumn
from dataframe.dtype import DataType
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import _pack_bits
from dataframe.storage.nested import ListColumn, StructColumn
from dataframe.vector.data_chunk import (
    ColumnArena,
    DataChunk,
    from_frame_window,
    to_series,
)
from dataframe.vector.tuple_data_collection import TupleDataCollection
from dataframe.vector.tuple_data_layout import TupleDataLayout
from dataframe.vector.tuple_data_scatter_gather import gather_rows_series
from dataframe.vector.tuple_data_states import (
    TupleDataAppendState,
    TupleDataChunkState,
    TupleDataScanState,
)
from dataframe.vector.types import STANDARD_VECTOR_SIZE, SelectionVector

# Rows: three chunks, several row blocks (rows are ~150 bytes, ~1,700 to
# a block) and heap blocks.
comptime _ROWS = 5000
# One string longer than a heap block (TUPLE_DATA_BLOCK_SIZE).
comptime _HUGE_ROW = 3000
comptime _HUGE_BYTES = 300000


def _fixed[D: DType](name: String, n: Int, null_every: Int) raises -> Series:
    var values = List[Scalar[D]](capacity=n)
    var valid = List[Bool](capacity=n)
    for i in range(n):
        comptime if D.is_floating_point():
            values.append(Scalar[D](i) * Scalar[D](0.25) - Scalar[D](100))
        elif D == DType.int128:
            values.append(
                (Scalar[D](i) - 4000) * Scalar[D](100000000000000000000)
            )
        else:
            values.append(Scalar[D](i % 97) * Scalar[D](3))
        valid.append(i % null_every != 1)
    return Series(name, Column[Scalar[D]](values^, valid))


def _text(i: Int) -> String:
    if i == _HUGE_ROW:
        var huge = String("")
        for _ in range(_HUGE_BYTES // 10):
            huge += "0123456789"
        return huge^
    var kind = i % 5
    if kind == 0:
        return ""
    if kind == 1:
        return "s" + String(i)
    if kind == 2:
        return "a string longer than twelve bytes " + String(i)
    if kind == 3:
        return "héllo wörld ünïcode " + String(i)
    return "exactly12byt"


def _strings(name: String, n: Int, null_every: Int) raises -> Series:
    var values = List[String](capacity=n)
    var valid = List[Bool](capacity=n)
    for i in range(n):
        values.append(_text(i))
        valid.append(i % null_every != 3)
    return Series(name, StringColumn(values, valid))


def _bools(name: String, n: Int) raises -> Series:
    var values = List[Bool](capacity=n)
    var valid = List[Bool](capacity=n)
    for i in range(n):
        values.append(i % 3 == 0)
        valid.append(i % 8 != 2)
    return Series(name, BoolColumn(values, valid))


def _list_of(
    name: String,
    n: Int,
    var child: Series,
    lengths: List[Int],
    null_every: Int,
) raises -> Series:
    """A list column: row i has lengths[i] elements of `child`, in order;
    null every `null_every` rows (keeping their elements, hidden)."""
    var offsets = List[Int64](capacity=n + 1)
    offsets.append(0)
    var valid = List[Bool](capacity=n)
    var total = 0
    for i in range(n):
        total += lengths[i]
        offsets.append(Int64(total))
        valid.append(i % null_every != 4)
    return Series(name, ListColumn(offsets^, child^, _pack_bits(valid)))


def _lengths(n: Int, modulo: Int) -> List[Int]:
    var out = List[Int](capacity=n)
    for i in range(n):
        out.append((i * 7 + 3) % modulo)
    return out^


def _total(lengths: List[Int]) -> Int:
    var total = 0
    for x in lengths:
        total += x
    return total


def _input(n: Int) raises -> List[Series]:
    var columns = List[Series]()
    columns.append(_fixed[DType.int8]("i8", n, 5))
    columns.append(_fixed[DType.int16]("i16", n, 6))
    columns.append(_fixed[DType.int32]("i32", n, 7))
    columns.append(_fixed[DType.int64]("i64", n, 8))
    columns.append(_fixed[DType.uint8]("u8", n, 9))
    columns.append(_fixed[DType.uint16]("u16", n, 10))
    columns.append(_fixed[DType.uint32]("u32", n, 11))
    columns.append(_fixed[DType.uint64]("u64", n, 12))
    columns.append(_fixed[DType.float32]("f32", n, 13))
    columns.append(_fixed[DType.float64]("f64", n, 14))
    columns.append(_bools("b", n))
    columns.append(
        _fixed[DType.int128]("d128", n, 4).with_dtype(DataType.decimal(38, 2))
    )
    columns.append(_fixed[DType.int64]("day", n, 15).with_dtype(DataType.DATE))
    columns.append(_strings("s", n, 7))

    # struct{a: int64, b: string, c: list<int32>}, the struct null every
    # 11 rows.
    var c_lengths = _lengths(n, 4)
    var c = _list_of(
        "c", n, _fixed[DType.int32]("item", _total(c_lengths), 5), c_lengths, 6
    )
    var fields = List[Series]()
    fields.append(_fixed[DType.int64]("a", n, 5))
    fields.append(_strings("b", n, 4))
    fields.append(c^)
    var struct_valid = List[Bool](capacity=n)
    for i in range(n):
        struct_valid.append(i % 11 != 7)
    columns.append(
        Series("st", StructColumn(fields^, _pack_bits(struct_valid)))
    )

    # list<string>
    var ls_lengths = _lengths(n, 5)
    columns.append(
        _list_of(
            "ls",
            n,
            _strings("item", _total(ls_lengths), 6),
            ls_lengths,
            9,
        )
    )

    # list<list<int64>>
    var outer_lengths = _lengths(n, 4)
    var inner_count = _total(outer_lengths)
    var inner_lengths = _lengths(inner_count, 3)
    var inner = _list_of(
        "item",
        inner_count,
        _fixed[DType.int64]("item", _total(inner_lengths), 4),
        inner_lengths,
        5,
    )
    columns.append(_list_of("ll", n, inner^, outer_lengths, 13))

    # list<struct{x: float64, y: string}>
    var lst_lengths = _lengths(n, 3)
    var lst_count = _total(lst_lengths)
    var lst_fields = List[Series]()
    lst_fields.append(_fixed[DType.float64]("x", lst_count, 3))
    lst_fields.append(_strings("y", lst_count, 5))
    var lst_valid = List[Bool](capacity=lst_count)
    for i in range(lst_count):
        lst_valid.append(i % 7 != 5)
    var lst_child = Series(
        "item", StructColumn(lst_fields^, _pack_bits(lst_valid))
    )
    columns.append(_list_of("lst", n, lst_child^, lst_lengths, 8))

    # struct{inner: struct{z: int16}, w: bool}
    var z = List[Series]()
    z.append(_fixed[DType.int16]("z", n, 3))
    var inner_valid = List[Bool](capacity=n)
    for i in range(n):
        inner_valid.append(i % 10 != 1)
    var outer = List[Series]()
    outer.append(Series("inner", StructColumn(z^, _pack_bits(inner_valid))))
    outer.append(_bools("w", n))
    columns.append(Series("ss", StructColumn(outer^)))
    return columns^


def _types(columns: List[Series]) -> List[DataType]:
    var out = List[DataType]()
    for c in columns:
        out.append(c.dtype())
    return out^


def _append_all(
    mut rows: TupleDataCollection,
    columns: List[Series],
    start: Int,
    stop: Int,
) raises:
    """Append rows [start, stop) a standard vector at a time."""
    var arena = ColumnArena()
    var handles = List[Int]()
    for c in columns:
        handles.append(arena.add(c))
    var state = TupleDataAppendState()
    rows.initialize_append(state)
    var chunk = DataChunk(len(columns))
    var offset = start
    while offset < stop:
        var count = min(STANDARD_VECTOR_SIZE, stop - offset)
        chunk.reset()
        from_frame_window(chunk, arena, handles, offset, count)
        rows.append(state, chunk, arena)
        offset += count


def _scan_all(
    rows: TupleDataCollection, columns: List[Series]
) raises -> List[Series]:
    """Every row, scanned a chunk at a time, as one column per column."""
    var arena = ColumnArena()
    var state = TupleDataScanState()
    rows.initialize_scan(state)
    var parts = List[List[Series]]()
    for _ in range(len(columns)):
        parts.append(List[Series]())
    var chunk = DataChunk(len(columns))
    while rows.scan(state, chunk, arena):
        for c in range(len(columns)):
            parts[c].append(
                to_series(
                    chunk, c, arena, columns[c].name(), columns[c].dtype()
                )
            )
    var out = List[Series]()
    for c in range(len(columns)):
        out.append(Series._from_chunks(parts[c]).rechunk())
    return out^


def _assert_columns(got: List[Series], expected: List[Series]) raises:
    assert_equal(len(got), len(expected))
    for c in range(len(expected)):
        assert_equal(len(got[c]), len(expected[c]), expected[c].name())
        assert_true(
            got[c].dtype() == expected[c].dtype(),
            expected[c].name() + ": " + got[c].dtype().name(),
        )
        assert_true(got[c].equals(expected[c]), expected[c].name())


def test_layout_offsets_follow_duckdb() raises:
    # [validity 1][heap size 8][int32 4][string_t 16]
    var types = List[DataType]()
    types.append(DataType.INT32)
    types.append(DataType.STRING)
    var layout = TupleDataLayout(types)
    assert_equal(layout.data_offset(), 1)
    assert_equal(layout.heap_size_offset(), 1)
    assert_equal(layout.offsets()[0], 9)
    assert_equal(layout.offsets()[1], 13)
    assert_equal(layout.row_width(), 29)
    assert_true(not layout.all_constant())

    # A struct is a nested row in place: [validity 1][[validity 1][int8 1]].
    var names = List[String]()
    names.append("a")
    var fields = List[DataType]()
    fields.append(DataType.INT8)
    var nested = List[DataType]()
    nested.append(DataType.struct(names, fields))
    var struct_layout = TupleDataLayout(nested)
    assert_true(struct_layout.all_constant())
    assert_equal(struct_layout.offsets()[0], 1)
    assert_equal(struct_layout.row_width(), 3)

    # Aggregate states follow the columns, 8-byte aligned.
    var sizes = List[Int]()
    sizes.append(16)
    var with_states = TupleDataLayout(nested, True, sizes)
    assert_equal(with_states.aggr_offset(), 8)
    assert_equal(with_states.row_width(), 24)


def test_every_type_round_trips() raises:
    var columns = _input(_ROWS)
    var rows = TupleDataCollection(TupleDataLayout(_types(columns)))
    _append_all(rows, columns, 0, _ROWS)
    assert_equal(rows.count(), _ROWS)
    assert_equal(
        rows.chunk_count(),
        (_ROWS + STANDARD_VECTOR_SIZE - 1) // STANDARD_VECTOR_SIZE,
    )
    var blocks = 0
    var heap_blocks = 0
    for s in range(len(rows.segments)):
        blocks += len(rows.segments[s].allocator.row_blocks)
        heap_blocks += len(rows.segments[s].allocator.heap_blocks)
    assert_true(blocks > 1, "rows span several row blocks")
    assert_true(heap_blocks > 1, "heap data spans several heap blocks")
    _assert_columns(_scan_all(rows, columns), columns)


def test_dictionary_chunks_and_append_selections() raises:
    var n = 2500
    var columns = _input(n)
    var rows = TupleDataCollection(TupleDataLayout(_types(columns)))
    var arena = ColumnArena()
    var handles = List[Int]()
    for c in columns:
        handles.append(arena.add(c))
    var state = TupleDataAppendState()
    rows.initialize_append(state)

    # Each standard vector read backwards (a dictionary over the flat
    # window), then only its odd positions appended.
    var expected_rows = List[Int]()
    var flat = DataChunk(len(columns))
    var sliced = DataChunk(len(columns))
    var reverse = List[UInt32](length=STANDARD_VECTOR_SIZE, fill=0)
    var odd = List[UInt32](length=STANDARD_VECTOR_SIZE, fill=0)
    var offset = 0
    while offset < n:
        var count = min(STANDARD_VECTOR_SIZE, n - offset)
        flat.reset()
        from_frame_window(flat, arena, handles, offset, count)
        for i in range(count):
            reverse[i] = UInt32(count - 1 - i)
        sliced.reset()
        sliced.slice(flat, SelectionVector(Int(reverse.unsafe_ptr())), count)
        var picked = 0
        for i in range(1, count, 2):
            odd[picked] = UInt32(i)
            expected_rows.append(offset + count - 1 - i)
            picked += 1
        rows.append(
            state,
            sliced,
            arena,
            SelectionVector(Int(odd.unsafe_ptr())),
            picked,
        )
        offset += count
    var expected = List[Series]()
    for c in columns:
        expected.append(c.take(expected_rows))
    _assert_columns(_scan_all(rows, columns), expected)
    _ = reverse^
    _ = odd^


def test_gather_through_row_pointers() raises:
    var n = 2 * STANDARD_VECTOR_SIZE
    var columns = _input(n)
    var rows = TupleDataCollection(TupleDataLayout(_types(columns)))
    _append_all(rows, columns, 0, n)

    # Chunk 1's rows: every third one, scanned backwards, written to the
    # result in the opposite order.
    var state = TupleDataChunkState()
    var count = rows.fetch_chunk(state, 1, False)
    assert_equal(count, STANDARD_VECTOR_SIZE)
    var scan = List[UInt32]()
    var i = count - 1
    while i >= 0:
        scan.append(UInt32(i))
        i -= 3
    var picked = len(scan)
    var target = List[UInt32]()
    var expected_rows = List[Int](length=picked, fill=0)
    for k in range(picked):
        target.append(UInt32(picked - 1 - k))
        expected_rows[picked - 1 - k] = STANDARD_VECTOR_SIZE + Int(scan[k])
    var all_columns = List[Int]()
    for c in range(len(columns)):
        all_columns.append(c)
    var arena = ColumnArena()
    var chunk = DataChunk(len(columns))
    rows.gather(
        state.rows(),
        SelectionVector(Int(scan.unsafe_ptr())),
        picked,
        all_columns,
        chunk,
        SelectionVector(Int(target.unsafe_ptr())),
        arena,
    )
    chunk.count = picked
    for c in range(len(columns)):
        var got = to_series(
            chunk, c, arena, columns[c].name(), columns[c].dtype()
        )
        assert_true(
            got.equals(columns[c].take(expected_rows)), columns[c].name()
        )
    _ = scan^
    _ = target^
    _ = state^
    _ = rows^


def test_boolean_payload_through_row_pointers() raises:
    # GatherRHS for a Boolean column: `false` stays false, null stays
    # null, with and without nulls in the column.
    var n = 3 * STANDARD_VECTOR_SIZE
    var plain_values = List[Bool](capacity=n)
    for i in range(n):
        plain_values.append(i % 3 == 0)
    var columns = List[Series]()
    columns.append(_bools("nulls", n))
    columns.append(Series("plain", BoolColumn(plain_values)))
    var rows = TupleDataCollection(TupleDataLayout(_types(columns)))
    _append_all(rows, columns, 0, n)
    for c in range(rows.chunk_count()):
        var state = TupleDataChunkState()
        var count = rows.fetch_chunk(state, c, False)
        var picked = List[Int](capacity=count)
        for i in range(count):
            picked.append(c * STANDARD_VECTOR_SIZE + i)
        for k in range(len(columns)):
            var got = gather_rows_series(
                rows.layout,
                Int(state.rows()),
                count,
                k,
                columns[k].name(),
                columns[k].dtype(),
            )
            assert_true(got.equals(columns[k].take(picked)), columns[k].name())
        _ = state^
    _ = rows^


def test_combined_collections_keep_their_rows() raises:
    var n = 3500
    var columns = _input(n)
    var layout = TupleDataLayout(_types(columns))
    var first = TupleDataCollection(layout)
    var second = TupleDataCollection(layout)
    _append_all(first, columns, 0, 2100)
    _append_all(second, columns, 2100, n)
    first.combine(second)
    assert_equal(first.count(), n)
    assert_equal(second.count(), 0)
    _assert_columns(_scan_all(first, columns), columns)
