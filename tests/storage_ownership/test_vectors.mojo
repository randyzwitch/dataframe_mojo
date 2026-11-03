"""Vectors over at-rest columns: flat windows without copies, dictionary
slices that merge, constants, sequences, the unified format and chunk
reuse (DuckDB's vector, data chunk and vector cache semantics)."""
# contract-group: storage.vectors
from std.memory import Pointer
from std.testing import assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, Series, StringColumn
from dataframe.dtype import DataType
from dataframe.storage.boolean import BoolColumn
from dataframe.vector.types import (
    CONSTANT_VECTOR,
    DICTIONARY_VECTOR,
    FLAT_VECTOR,
    PHYSICAL_INT64,
    PHYSICAL_VARCHAR,
    STANDARD_VECTOR_SIZE,
    SelectionVector,
    StringT,
    Vector,
    string_compare,
    string_data,
    string_equals,
)
from dataframe.vector.data_chunk import (
    ColumnArena,
    DataChunk,
    flat_over,
    from_frame_window,
    to_series,
)


def _ints(n: Int, null_every: Int) raises -> Series:
    var values = List[Int64]()
    var valid = List[Bool]()
    for i in range(n):
        values.append(Int64(10 * i))
        valid.append(null_every == 0 or i % null_every != 0)
    return Series("x", Column[Int64](values^, valid))


def _texts(n: Int) raises -> Series:
    var values = List[String]()
    var valid = List[Bool]()
    for i in range(n):
        # Short (inline) and long (pointer) strings, and nulls.
        values.append(
            "s" + String(i) if i % 2 == 0 else "a longer string " + String(i)
        )
        valid.append(i % 7 != 3)
    return Series("s", StringColumn(values, valid))


def _int_at(format_data: Int, row: Int) -> Int64:
    return Pointer[Int64, MutAnyOrigin](unsafe_from_address=format_data)[
        unsafe_offset=row
    ]


def test_flat_window_reads_storage_in_place() raises:
    var arena = ColumnArena()
    var series = _ints(5000, 5).slice(7, 4000)
    var handle = arena.add(series)
    var chunk = DataChunk()
    from_frame_window(chunk, arena, [handle], 100, STANDARD_VECTOR_SIZE)
    assert_equal(chunk.data[0].vector_type, FLAT_VECTOR)
    assert_equal(chunk.data[0].physical, PHYSICAL_INT64)
    # The window's values are the column's own buffer, not a copy.
    var values = Int(arena.columns[handle]._data[Column[Int64]]._ptr())
    var buffer = arena.columns[handle]._data[Column[Int64]]._data.ptr()
    assert_equal(chunk.data[0].data, values + 8 * 100)
    var format = chunk.unified(0, arena)
    assert_false(format.sel.is_set())
    for i in range(STANDARD_VECTOR_SIZE):
        var row = 107 + i
        var row_valid = row % 5 != 0
        assert_equal(format.validity.row_is_valid(i), row_valid)
        if row_valid:
            assert_equal(_int_at(format.data, i), Int64(10 * row))
    var back = to_series(chunk, 0, arena, "y", DataType.INT64)
    assert_equal(len(back), STANDARD_VECTOR_SIZE)
    assert_equal(back.name(), "y")
    assert_true(back._data[Column[Int64]]._data.ptr() == buffer)
    _ = arena^


def test_dictionary_slices_merge_and_gather_on_the_way_out() raises:
    var arena = ColumnArena()
    var handle = arena.add(_ints(3000, 4))
    var source = DataChunk()
    from_frame_window(source, arena, [handle], 1000, 1000)
    # Keep every third row, then every other kept row.
    var first = DataChunk()
    var sel = first.new_selection()
    var kept = 0
    for i in range(1000):
        if i % 3 == 0:
            sel.set_index(kept, i)
            kept += 1
    first.slice(source, sel, kept)
    assert_equal(first.data[0].vector_type, DICTIONARY_VECTOR)
    var second = DataChunk()
    var sel2 = second.new_selection()
    var kept2 = 0
    for i in range(kept):
        if i % 2 == 1:
            sel2.set_index(kept2, i)
            kept2 += 1
    second.slice(first, sel2, kept2)
    # Not a dictionary of a dictionary: one merged selection over the
    # flat child.
    assert_equal(second.data[0].vector_type, DICTIONARY_VECTOR)
    assert_equal(second.data[0].data, source.data[0].data)
    var format = second.unified(0, arena)
    for i in range(kept2):
        var row = 1000 + 3 * (2 * i + 1)
        var child = format.sel.get_index(i)
        assert_equal(format.validity.row_is_valid(child), row % 4 != 0)
        if row % 4 != 0:
            assert_equal(_int_at(format.data, child), Int64(10 * row))
    var back = to_series(second, 0, arena, "x", DataType.INT64)
    assert_equal(len(back), kept2)
    for i in range(kept2):
        var row = 1000 + 3 * (2 * i + 1)
        assert_equal(back.get(i).is_null(), row % 4 == 0)
        if row % 4 != 0:
            assert_equal(back.get(i).int64(), Int64(10 * row))


def test_strings_decode_to_string_t_at_the_rows_read() raises:
    var arena = ColumnArena()
    var handle = arena.add(_texts(600))
    var source = DataChunk()
    from_frame_window(source, arena, [handle], 50, 500)
    assert_equal(source.data[0].physical, PHYSICAL_VARCHAR)
    assert_true(source.data[0].encoded())
    var sliced = DataChunk()
    var sel = sliced.new_selection()
    for i in range(100):
        sel.set_index(i, 499 - 5 * i)
    sliced.slice(source, sel, 100)
    var format = sliced.unified(0, arena)
    for i in range(100):
        var row = 50 + 499 - 5 * i
        var at = format.sel.get_index(i)
        assert_equal(format.validity.row_is_valid(at), row % 7 != 3)
        if row % 7 == 3:
            continue
        var expected = "s" + String(
            row
        ) if row % 2 == 0 else "a longer string " + String(row)
        var entry = format.data + 16 * at
        var s = Pointer[StringT, MutAnyOrigin](unsafe_from_address=entry)[]
        assert_equal(Int(s.length), expected.byte_length())
        assert_equal(s.is_inlined(), expected.byte_length() <= 12)
        var text = StringSlice[ImmutAnyOrigin](
            unsafe_from_utf8=Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                    unsafe_from_address=string_data(entry)
                ),
                length=Int(s.length),
            )
        )
        assert_equal(String(text), expected)
    var back = to_series(sliced, 0, arena, "s", DataType.STRING)
    for i in range(100):
        var row = 50 + 499 - 5 * i
        assert_equal(back.get(i).is_null(), row % 7 == 3)


def test_string_t_compares_by_bytes() raises:
    var words: List[String] = [
        "",
        "a",
        "ab",
        "abcd",
        "abcde",
        "abcdefghijkl",
        "abcdefghijklm",
        "abcdefghijklmn",
        "abd",
        "b",
        "zzzzzzzzzzzzzzzz",
    ]
    var entries = List[StringT]()
    for w in words:
        entries.append(StringT.make(Int(w.unsafe_ptr()), w.byte_length()))
    var base = Int(entries.unsafe_ptr())
    for i in range(len(words)):
        for j in range(len(words)):
            var c = string_compare(base + 16 * i, base + 16 * j)
            var expected = 0 if words[i] == words[j] else (
                -1 if words[i] < words[j] else 1
            )
            assert_equal(c < 0, expected < 0)
            assert_equal(c > 0, expected > 0)
            assert_equal(string_equals(base + 16 * i, base + 16 * j), i == j)
    _ = entries^
    _ = words^


def test_constant_and_sequence_vectors() raises:
    var arena = ColumnArena()
    var chunk = DataChunk(2)
    var value = List[Int64](length=1, fill=42)
    chunk.data[0] = Vector.constant(
        PHYSICAL_INT64, Int(value.unsafe_ptr()), True
    )
    chunk.data[1] = Vector.sequence(5, 3)
    chunk.count = 300
    var c = chunk.unified(0, arena)
    var s = chunk.unified(1, arena)
    for i in range(300):
        assert_equal(_int_at(c.data, c.sel.get_index(i)), 42)
        assert_equal(_int_at(s.data, s.sel.get_index(i)), Int64(5 + 3 * i))
    var sliced = DataChunk()
    var sel = sliced.new_selection()
    for i in range(10):
        sel.set_index(i, 29 * i)
    sliced.slice(chunk, sel, 10)
    assert_equal(sliced.data[0].vector_type, CONSTANT_VECTOR)
    var flat = sliced.unified(1, arena)
    for i in range(10):
        assert_equal(
            _int_at(flat.data, flat.sel.get_index(i)), Int64(5 + 87 * i)
        )
    var null = Vector.constant(PHYSICAL_INT64, Int(value.unsafe_ptr()), False)
    assert_false(null.validity.row_is_valid(0))
    var back = to_series(sliced, 0, arena, "c", DataType.INT64)
    assert_equal(len(back), 10)
    assert_equal(back.get(9).int64(), 42)
    _ = value^


def test_booleans_decode_and_reset_reuses_buffers() raises:
    var arena = ColumnArena()
    var flags = List[Bool]()
    for i in range(3000):
        flags.append(i % 3 == 1)
    var handle = arena.add(Series("b", BoolColumn(flags)))
    var chunk = DataChunk()
    from_frame_window(chunk, arena, [handle], 9, 2048)
    var format = chunk.unified(0, arena)
    var bytes = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=format.data)
    for i in range(2048):
        assert_equal(bytes[unsafe_offset=i] != 0, (9 + i) % 3 == 1)
    var out = DataChunk(1)
    var first = out.writable(0, PHYSICAL_INT64).data
    var sel = out.new_selection()
    out.reset()
    out.set_columns(1)
    # Reset keeps every buffer: the same cache and selection come back.
    assert_equal(out.writable(0, PHYSICAL_INT64).data, first)
    assert_equal(out.new_selection().data, sel.data)
