"""A string hashes the same whether its column stores offsets or Arrow
views, so joins and group-bys match keys across storages: up to eight bytes
read in one load from inside the view, nine to twelve inline, longer ones
from a buffer; with nulls, Unicode, NUL bytes, a sliced view column, and as
the second key after an integer."""
from std.testing import assert_equal

from dataframe import Column, Series, StringColumn
from dataframe.partition import Partitioner
from dataframe.string_view import StringViewBuilder


def text_of(row: Int) -> String:
    var length = row % 41
    var text = String()
    var pieces: List[String] = ["a", "é", "\x00", "東", "z", "9"]
    while len(text.as_bytes()) < length:
        text += pieces[(row + len(text.as_bytes())) % len(pieces)]
    return text^


def columns(rows: Int) raises -> Tuple[Series, Series, List[Bool]]:
    var values = List[String]()
    var valid = List[Bool]()
    var builder = StringViewBuilder(rows)
    for row in range(rows):
        values.append(text_of(row))
        valid.append(row % 13 != 6)
        if valid[row]:
            builder.append(StringSlice(values[row]))
        else:
            builder.append_null()
    return (
        Series("s", StringColumn(values^, valid.copy())),
        Series("s", StringColumn(builder^.finish())),
        valid^,
    )


def hashes(keys: List[Series]) raises -> List[UInt64]:
    return Partitioner(keys, 3).into_hashes()


def test_offsets_and_views_hash_every_length_alike() raises:
    var rows = 2_000
    var built = columns(rows)
    var offsets = built[0].copy()
    var views = built[1].copy()
    assert_equal(hashes([offsets.copy()]), hashes([views.copy()]))
    # A view column sliced past its first rows.
    assert_equal(
        hashes([offsets.slice(37, 1_500)]), hashes([views.slice(37, 1_500)])
    )
    # As the second key, combined with what the first wrote.
    var numbers = List[Int64]()
    for row in range(rows):
        numbers.append(Int64(row % 7))
    var first = Series("n", Column[Int64](numbers^))
    assert_equal(
        hashes([first.copy(), offsets.copy()]),
        hashes([first.copy(), views.copy()]),
    )
