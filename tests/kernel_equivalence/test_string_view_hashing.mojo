"""A string hashes the same whether its column stores offsets or Arrow
views, so joins and group-bys match keys across storages: up to eight bytes
read in one load from inside the view, nine to twelve inline, longer ones
from a buffer; with nulls, Unicode, NUL bytes, a sliced view column, and as
the second key after an integer."""
# contract-group: kernel.order_keys
from std.testing import assert_equal

from dataframe import Column, Series, StringColumn
from dataframe.join_hash import _string_probe_hash
from dataframe.partition import Partitioner
from dataframe.string_bytes import _Bytes
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
        Series("s", StringColumn(values, valid.copy())),
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


def test_string_probe_hashes_match_build_hashes_at_offsets_views_and_byte_buffer_tails() raises:
    var built = columns(2_000)
    for native in [False, True]:
        var source = built[1].copy() if native else built[0].copy()
        for window in [(0, 2_000), (37, 1_500), (1_990, 10)]:
            var key = source.slice(window[0], window[1])
            var expected = hashes([key.copy()])
            ref strings = key._data[StringColumn]
            var bytes = _Bytes(strings)
            for row in range(len(key)):
                if strings._valid(row):
                    assert_equal(_string_probe_hash(bytes, row), expected[row])
    # All short-key lengths, with short valid strings ending at the byte
    # buffer's boundary. An eight-byte read there would cross the buffer.
    var values = List[String]()
    for length in range(18):
        var value = String()
        for _ in range(length):
            value += "x"
        values.append(value^)
    values.extend(["λ\x00東", "\x00", "x", ""])
    var flags = List[Bool](length=len(values), fill=True)
    for native in [False, True]:
        var builder = StringViewBuilder(len(values))
        for value in values:
            builder.append(StringSlice(value))
        var key = Series(
            "s",
            StringColumn(builder^.finish()) if native else StringColumn(
                values.copy(), flags.copy()
            ),
        )
        var expected = hashes([key.copy()])
        var bytes = _Bytes(key._data[StringColumn])
        for row in range(len(key)):
            assert_equal(_string_probe_hash(bytes, row), expected[row])
    # The offsets buffer can contain no bytes at all when every value is
    # empty. A zero-length span must not cause a speculative word read.
    for native in [False, True]:
        var builder = StringViewBuilder(3)
        for _ in range(3):
            builder.append(StringSlice(""))
        var key = Series(
            "s",
            StringColumn(builder^.finish()) if native else StringColumn(
                List[String](length=3, fill=""),
                List[Bool](length=3, fill=True),
            ),
        )
        var expected = hashes([key.copy()])
        var bytes = _Bytes(key._data[StringColumn])
        for row in range(len(key)):
            assert_equal(_string_probe_hash(bytes, row), expected[row])
