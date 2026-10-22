"""Candidate hashes never replace complete typed tuple equality."""
# contract-group: kernel.order_keys
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataType, Series, StringColumn
from dataframe.dtype import CategoricalDictionary
from dataframe.hashing import column_codes
from dataframe.partition import encode_bucket, _KeyView, _KEY_VIEW, _KEY_STRING
from dataframe.string_view import StringViewBuilder
from tests.support.key_kernel_reference import KeyCell, expected_key_ids
from tests.support.selection_types_reference import (
    selection_column,
    selection_key,
)


def check_bucket(
    keys: List[Series], rows: List[Int], cells: List[List[KeyCell]]
) raises:
    var hashes = List[UInt64](length=len(rows), fill=42)
    var ids = List[Int](length=len(rows) + 13, fill=-91)
    var firsts: List[Int] = [777, 888]
    encode_bucket(keys, Span(hashes), Span(rows), ids, firsts)
    var selected = List[List[KeyCell]]()
    for row in rows:
        selected.append(cells[row].copy())
    var expected = expected_key_ids(selected, True)
    assert_equal(ids, expected[0])
    assert_equal(len(firsts), len(expected[1]))
    for group in range(len(firsts)):
        assert_equal(firsts[group], rows[expected[1][group]])


def test_colliding_bucket_hashes_confirm_all_fixed_binary_dictionary_temporal_and_composite_cells() raises:
    for n in [0, 1, 9, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                for kind in range(29):
                    if kind == 26 or kind == 27:
                        continue
                    var key = selection_column(kind, n, offset, mode)
                    var second = selection_column(10, n, offset, (mode + 1) % 3)
                    var cells = List[List[KeyCell]]()
                    for row in range(n):
                        cells.append(
                            [
                                selection_key(kind, row + offset, mode),
                                selection_key(10, row + offset, (mode + 1) % 3),
                            ]
                        )
                    var rows = List[Int]()
                    for row in range(n - 1, -1, -1):
                        rows.append(row)
                        if row % 3 == 0:
                            rows.append(row)
                    check_bucket([key.copy(), second.copy()], rows, cells)


def test_bucket_string_storage_inline_external_growth_and_repeated_selection_compare_complete_bytes() raises:
    var n = 2203
    var values = List[String]()
    var valid = List[Bool]()
    var cells = List[List[KeyCell]]()
    var builder = StringViewBuilder(n + 7)
    for physical in range(n + 7):
        var code = (physical * 37) % 1103
        var text = "abcd" + String(code)
        if code % 3 == 1:
            text = "abcdefghijkl\x00é/" + String(code)
        elif code % 3 == 2:
            text = String("x") * 9000 + "/" + String(code)
        var present = physical % 17 != 2
        values.append(text.copy())
        valid.append(present)
        if present:
            builder.append(StringSlice(text))
        else:
            builder.append_null()
        if physical >= 7:
            cells.append([KeyCell(2, present, text=text.copy())])
    var native = Series("s", StringColumn(builder^.finish())).slice(7, n)
    var arrow = Series("s", StringColumn(values.copy(), valid.copy())).slice(
        7, n
    )
    assert_equal(_KeyView(native).kind, _KEY_VIEW)
    assert_equal(_KeyView(arrow).kind, _KEY_STRING)
    var rows = List[Int]()
    for row in range(n - 1, -1, -1):
        rows.append(row)
        if row % 11 == 0:
            rows.append(row)
    check_bucket([native.copy()], rows, cells)
    check_bucket([arrow.copy()], rows, cells)
    # The offsets-only all-valid path bypasses _KeyView entirely.
    var direct = Series("s", StringColumn(values.copy())).slice(7, n)
    var direct_cells = List[List[KeyCell]]()
    for row in range(n):
        direct_cells.append([KeyCell(2, text=values[row + 7].copy())])
    assert_equal(direct.null_count(), 0)
    assert_equal(len(direct._data[StringColumn]._bits[]), 0)
    check_bucket([direct.copy()], rows, direct_cells)
    var compound = List[List[KeyCell]]()
    for row in range(n):
        compound.append([cells[row][0].copy(), direct_cells[row][0].copy()])
    check_bucket([native.copy(), direct.copy()], rows, compound)


def category_codes(domain: Int, poison_visible: Bool = False) raises -> Series:
    var dictionary = CategoricalDictionary()
    for label in range(domain):
        dictionary.append("label/" + String(label))
    var values: List[UInt32] = [
        0,
        UInt32(domain - 1),
        UInt32.MAX,
        0,
        UInt32(domain - 1),
        1,
        0,
    ]
    var valid: List[Bool] = [True, True, poison_visible, True, True, True, True]
    return (
        Series("c", Column[UInt32](values^, valid))
        .with_dtype(DataType.categorical(dictionary^))
        .slice(1, 5)
    )


def test_categorical_column_codes_lookup_boundary_null_poison_and_reused_output_have_literal_ids() raises:
    for domain in [2, 4096, 4097]:
        var source = category_codes(domain)
        assert_equal(len(source.dtype().dictionary()[]), domain)
        assert_equal(domain <= max(len(source), 4096), domain <= 4096)
        for chunked in [False, True]:
            var input = source.copy()
            if chunked:
                input = Series._from_chunks(
                    [source.slice(0, 2), source.slice(2, 3)]
                )
            var codes = List[Int](length=19, fill=991)
            var nulls = List[Bool](length=5, fill=False)
            assert_equal(
                column_codes(input, codes, nulls), 2 if domain == 2 else 3
            )
            var expected: List[Int] = [0, -1, 1, 0, 0 if domain == 2 else 2]
            assert_equal(codes, expected)
            assert_equal(nulls, [False, True, False, False, False])
        var empty_codes = List[Int](length=19, fill=991)
        var empty_nulls = List[Bool]()
        assert_equal(column_codes(source.head(0), empty_codes, empty_nulls), 0)
        assert_equal(len(empty_codes), 0)
    with assert_raises():
        var codes = List[Int]()
        var nulls = List[Bool](length=5, fill=False)
        _ = column_codes(category_codes(2, True), codes, nulls)
