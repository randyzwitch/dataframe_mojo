"""Bulk nested reassembly retains slices, nulls, child dictionaries and aliases."""
# contract-group: storage.chunks
from std.testing import assert_equal, assert_true
from dataframe import Column, DataType, ListColumn, StructColumn, Series
from dataframe.column import _pack_bits
from tests.support.coverage_compare import assert_value


def _source() raises -> Series:
    var offsets: List[Int64] = [0]
    var values = List[Int64]()
    var element_valid = List[Bool]()
    var list_valid = List[Bool]()
    var text = List[String]()
    var text_valid = List[Bool]()
    var parent_valid = List[Bool]()
    for row in range(73):
        for item in range(row % 4):
            values.append(Int64(row * 10 + item))
            element_valid.append(item != 2)
        offsets.append(Int64(len(values)))
        list_valid.append(row % 5 != 1)
        text.append("é row " + String(row))
        text_valid.append(row % 7 != 3)
        parent_valid.append(row % 11 != 5)
    var children: List[Series] = [
        Series(
            "items",
            ListColumn(
                offsets^,
                Series("item", Column[Int64](values^, element_valid)),
                _pack_bits(list_valid),
            ),
        ),
        Series("text", Column[String](text^, text_valid)),
    ]
    return Series("nested", StructColumn(children^, _pack_bits(parent_valid)))


def test_bulk_nested_rechunk_keeps_unaligned_slices_empty_chunks_and_cache_aliases() raises:
    var source = _source()
    var alias = source.copy()
    var starts: List[Int] = [4, 3, 23, 42, 1, 63]
    var lengths: List[Int] = [7, 0, 5, 13, 1, 9]
    var parts = List[Series]()
    var rows = List[Int]()
    for i in range(len(starts)):
        parts.append(source.slice(starts[i], lengths[i]))
        for row in range(starts[i], starts[i] + lengths[i]):
            rows.append(row)
    var chunked = Series._from_chunks(parts)
    var first = chunked.rechunk()
    var second = chunked.copy().rechunk()
    assert_true(
        first._data[StructColumn]._fields.ptr()
        == second._data[StructColumn]._fields.ptr()
    )
    assert_equal(len(first), len(rows))
    for row in range(len(rows)):
        assert_value(first.get(row), source.get(rows[row]))
    for row in range(len(source)):
        assert_value(source.get(row), alias.get(row))
    var items = first.struct_column().field("items")
    var list_parts = List[Series]()
    for i in range(len(starts)):
        list_parts.append(
            alias.struct_column().field("items").slice(starts[i], lengths[i])
        )
    var lists = Series._from_chunks(list_parts).rechunk()
    for row in range(len(rows)):
        assert_value(lists.get(row), items.get(row))


def _categorical_nested() raises -> Series:
    var parts = List[Series]()
    for batch in range(9):
        var a = String("z") if batch % 2 == 0 else String("é")
        var b = String("é") if batch % 2 == 0 else String("z")
        var values = Series(
            "item",
            Column[String](
                [a.copy(), b.copy(), String(""), a.copy()],
                [True, True, False, True],
            ),
        ).cast(DataType.CATEGORICAL)
        var offsets: List[Int64] = [0, 1, 3, 4]
        var lists = Series(
            "items",
            ListColumn(offsets^, values^, _pack_bits([True, False, True])),
        )
        var label = Series(
            "label", Column[String]([b.copy(), a.copy(), b.copy()])
        ).cast(DataType.CATEGORICAL)
        parts.append(Series("nested", StructColumn([lists^, label^])))
    return Series._from_chunks(parts).rechunk()


def test_bulk_nested_rechunk_unifies_different_child_dictionaries_after_owners_die() raises:
    var result = _categorical_nested()
    assert_equal(len(result), 27)
    var lists = result.struct_column().field("items")
    assert_true(lists.list_column().child().dtype().is_categorical())
    assert_equal(len(lists.list_column().child().dtype().dictionary()[]), 2)
    for batch in range(9):
        var a = String("z") if batch % 2 == 0 else String("é")
        var b = String("é") if batch % 2 == 0 else String("z")
        assert_equal(lists.get(3 * batch).list().get(0).string(), a)
        assert_true(lists.get(3 * batch + 1).is_null())
        assert_equal(lists.get(3 * batch + 2).list().get(0).string(), a)
        assert_equal(result.get(3 * batch).struct_field("label").string(), b)
        assert_equal(
            result.get(3 * batch + 1).struct_field("label").string(), a
        )


def test_bulk_empty_struct_rechunk_retains_length_and_unaligned_validity() raises:
    var valid: List[Bool] = [
        True,
        False,
        True,
        True,
        False,
        True,
        False,
        True,
        True,
    ]
    var empty_fields = List[Series]()
    var source = Series(
        "empty",
        StructColumn(empty_fields^, _pack_bits(valid), length=len(valid)),
    )
    var result = Series._from_chunks(
        [source.slice(1, 3), source.slice(4, 5)]
    ).rechunk()
    assert_equal(len(result), 8)
    assert_equal(result.struct_column().field_count(), 0)
    for i in range(8):
        assert_equal(result.get(i).is_null(), not valid[i + 1])


def test_bulk_many_distinct_categorical_child_dictionaries() raises:
    var parts = List[Series]()
    for batch in range(64):
        var values: List[String] = [
            String(batch) + "_a",
            "shared",
            String(batch) + "_b",
        ]
        var child = Series("item", Column[String](values^)).cast(
            DataType.CATEGORICAL
        )
        var offsets: List[Int64] = [0, 3]
        parts.append(Series("items", ListColumn(offsets^, child^)))
    var result = Series._from_chunks(parts).rechunk()
    assert_equal(len(result.list_column().child().dtype().dictionary()[]), 129)
    for batch in range(64):
        var row = result.get(batch).list()
        assert_equal(row.get(0).string(), String(batch) + "_a")
        assert_equal(row.get(1).string(), "shared")
        assert_equal(row.get(2).string(), String(batch) + "_b")
