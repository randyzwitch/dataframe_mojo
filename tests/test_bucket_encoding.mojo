"""Grouping a hash bucket from precomputed hashes (#334), and taking a few
rows of a chunked column without rechunking it.

`encode_bucket` trusts hashes only to find candidates: every match is
confirmed by comparing the rows' keys. Forcing every hash to one value
makes each probe collide, so the ids must still equal `encode_rows`."""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import BoolColumn, Column, Series, StringColumn
from dataframe.hashing import encode_rows
from dataframe.partition import encode_bucket
from dataframe.string_view import StringViewBuilder


def keys_with_duplicates() raises -> List[Series]:
    var texts = List[String]()
    var valid = List[Bool]()
    var floats = List[Float64]()
    var ints = List[Int64]()
    for i in range(300):
        # Long (over 12 bytes) and short strings, nulls, NaN and -0.0.
        texts.append(
            "a-much-longer-key-" + String(i % 17) if i % 3
            == 0 else String(i % 11)
        )
        valid.append(i % 29 != 5)
        var f = Float64(i % 7)
        if i % 13 == 0:
            f = Float64(0) / Float64(0)
        elif i % 7 == 0:
            f = -0.0 if i % 2 == 0 else 0.0
        floats.append(f)
        ints.append(Int64(i % 5))
    return [
        Series("s", StringColumn(texts, valid)),
        Series("f", Column[Float64](floats^)),
        Series("i", Column[Int64](ints^)),
    ]


def check(keys: List[Series], label: String) raises:
    var rows = List[Int]()
    for i in range(len(keys[0])):
        rows.append(i)
    var colliding = List[UInt64](length=len(rows), fill=42)
    var ids = List[Int]()
    var firsts = List[Int]()
    encode_bucket(keys, Span(colliding), Span(rows), ids, firsts)
    var expected = encode_rows(keys, nulls_equal=True)
    assert_equal(len(firsts), expected.count(), label)
    for i in range(len(rows)):
        assert_equal(ids[i], expected.ids[i], label + " row " + String(i))
    for g in range(len(firsts)):
        assert_equal(firsts[g], expected.representatives[g], label)


def test_colliding_hashes_still_group_exactly() raises:
    var keys = keys_with_duplicates()
    check([keys[0].copy()], "string")
    check([keys[1].copy()], "float")
    check([keys[0].copy(), keys[1].copy(), keys[2].copy()], "composite")
    # A subset of rows in another order: ids follow first occurrence.
    var rows: List[Int] = [7, 3, 7, 250, 3, 18, 250]
    var hashes = List[UInt64](length=len(rows), fill=0)
    var ids = List[Int]()
    var firsts = List[Int]()
    encode_bucket([keys[2].copy()], Span(hashes), Span(rows), ids, firsts)
    # i % 5: 7 -> 2, 3 -> 3, 250 -> 0, 18 -> 3.
    assert_equal(ids, [0, 1, 0, 2, 1, 1, 2])
    assert_equal(firsts, [7, 3, 250])


def test_take_from_chunks_without_rechunking() raises:
    var parts = List[Series]()
    for c in range(4):
        var texts = List[String]()
        for i in range(100):
            texts.append("c" + String(c) + "r" + String(i))
        parts.append(Series("s", StringColumn(texts)))
    var chunked = Series._from_chunks(parts^)
    assert_true(chunked.is_chunked())
    # Few rows, out of order, repeated, across chunk boundaries.
    var picked = chunked.take([399, 0, 100, 99, 250, 0])
    var expected: List[String] = [
        "c3r99",
        "c0r0",
        "c1r0",
        "c0r99",
        "c2r50",
        "c0r0",
    ]
    for i in range(len(expected)):
        assert_equal(picked.get(i).string(), expected[i])
    assert_equal(len(chunked.take(List[Int]())), 0)
    with assert_raises(contains="out of bounds"):
        _ = chunked.take([400])


def test_composite_string_collisions_offsets_and_null_fallback() raises:
    var words: List[String] = [
        "",
        "a",
        "abcdefg",
        "abcdefgh",
        "abcdefghi",
        "abcdefghijklmno",
        "abcdefghijklmnop",
        "abcdefghijklmnopq",
        "a\x00b",
        "é雪",
        "same-long-prefix-first",
        "same-long-prefix-second",
    ]
    var columns = List[Series]()
    for k in range(3):
        var texts = List[String]()
        for i in range(317):
            texts.append(words[(i // (k + 1)) % len(words)])
        columns.append(Series("s" + String(k), StringColumn(texts)))
    var sliced = List[Series]()
    for column in columns:
        sliced.append(column.slice(7, 300))
    check(sliced, "composite strings with slice offsets")
    check([sliced[2].copy(), sliced[0].copy()], "reordered string keys")
    # Changing any component must reject even a deliberately equal hash.
    # A null-containing component routes the whole comparison to the generic
    # implementation and preserves equality of nulls.
    var texts = List[String]()
    var valid = List[Bool]()
    for i in range(300):
        texts.append(words[i % len(words)])
        valid.append(i % 11 != 0)
    sliced[1] = Series("nullable", StringColumn(texts, valid))
    check(sliced, "composite strings with nulls")


def test_composite_strings_grow_table_and_follow_selected_row_order() raises:
    var left = List[String]()
    var right = List[String]()
    for i in range(2200):
        left.append("constant")
        right.append("long-common-prefix-" + String(i % 1100))
    var keys: List[Series] = [
        Series("a", StringColumn(left)),
        Series("b", StringColumn(right)),
    ]
    check(keys, "composite string collisions through table growth")
    var rows: List[Int] = [2199, 0, 1099, 1100, 43, 1143, 2199]
    var hashes = List[UInt64](length=len(rows), fill=0)
    var ids = List[Int]()
    var firsts = List[Int]()
    encode_bucket(keys, Span(hashes), Span(rows), ids, firsts)
    assert_equal(ids, [0, 1, 0, 1, 2, 2, 0])
    assert_equal(firsts, [2199, 0, 43])


def test_composite_view_string_fallback_with_collisions() raises:
    var texts = List[String]()
    var builder = StringViewBuilder()
    for i in range(137):
        var text = "a-long-common-prefix-" + String(i % 13)
        texts.append(text)
        builder.append(StringSlice(text))
    var offsets = Series("offsets", StringColumn(texts))
    var views = Series("views", StringColumn(builder^.finish()))
    check([offsets.slice(3, 130), views.slice(3, 130)], "mixed view storage")
    check([views.slice(3, 130), offsets.slice(3, 130)], "reversed view storage")
    # Inline (through 12 bytes) and long values sharing the first four
    # bytes, values that differ only after the prefix, nulls, and enough
    # long bytes to span more than one buffer.
    var words: List[String] = [
        "",
        "abcd",
        "abcdefghijkl",
        "abcdefghijklm",
        "abcdefghijklmn",
        "abcdXfghijklmn",
        "abcd" + String("z") * 9000,
        "abcd" + String("z") * 8999 + "y",
    ]
    var mixed = StringViewBuilder()
    var plain = List[String]()
    var valid = List[Bool]()
    for i in range(400):
        var word = words[(i * 7) % len(words)]
        if i % 17 == 3:
            mixed.append_null()
            plain.append("")
            valid.append(False)
        else:
            mixed.append(StringSlice(word))
            plain.append(word)
            valid.append(True)
    var viewed = Series("viewed", StringColumn(mixed^.finish()))
    var offsets_nulls = Series("plain", StringColumn(plain, valid))
    check([viewed.copy()], "view strings with nulls")
    check([viewed.slice(5, 390), offsets_nulls.slice(5, 390)], "view nulls")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_every_key_dtype_compares_through_its_view() raises:
    """Composite keys of every fixed width, floats of both widths,
    decimals, bools and strings, each with nulls and taken from a slice
    (#486): a row's key is compared through a typed view against the
    group's key kept beside the bucket's table, and the ids equal
    `encode_rows` under colliding hashes."""
    var n = 400
    var valid = List[Bool](capacity=n)
    var bytes8 = List[UInt8](capacity=n)
    var shorts = List[Int16](capacity=n)
    var words = List[UInt32](capacity=n)
    var longs = List[Int64](capacity=n)
    var singles = List[Float32](capacity=n)
    var doubles = List[Float64](capacity=n)
    var wide = List[Int128](capacity=n)
    var flags = List[Bool](capacity=n)
    var texts = List[String](capacity=n)
    for i in range(n):
        valid.append(i % 23 != 4)
        bytes8.append(UInt8(i % 3))
        shorts.append(Int16(-(i % 4)))
        words.append(UInt32(i % 2))
        longs.append(Int64(1) << 40 if i % 5 == 0 else Int64(i % 2))
        var f = Float32(i % 3)
        if i % 9 == 0:
            f = Float32(0) / Float32(0)
        elif i % 6 == 0:
            f = -0.0 if i % 12 == 0 else 0.0
        singles.append(f)
        doubles.append(Float64(i % 4) if i % 8 else Float64(0) / Float64(0))
        wide.append(Int128(i % 3) << 70 if i % 2 == 0 else Int128(i % 3))
        flags.append(i % 7 < 3)
        texts.append("t" + String(i % 5))
    var keys: List[Series] = [
        Series("u8", Column[UInt8](bytes8^, valid.copy())),
        Series("i16", Column[Int16](shorts^, valid.copy())),
        Series("u32", Column[UInt32](words^, valid.copy())),
        Series("i64", Column[Int64](longs^, valid.copy())),
        Series("f32", Column[Float32](singles^, valid.copy())),
        Series("f64", Column[Float64](doubles^, valid.copy())),
        Series("i128", Column[Int128](wide^, valid.copy())),
        Series("b", BoolColumn(flags^, valid.copy())),
        Series("s", StringColumn(texts^, valid.copy())),
    ]
    for key in keys:
        check([key.copy()], key.name())
        check([key.slice(13, 350)], key.name() + " sliced")
    check(keys, "all dtypes")
    var sliced = List[Series]()
    for key in keys:
        sliced.append(key.slice(13, 350))
    check(sliced, "all dtypes sliced")
    check([keys[6].copy(), keys[3].copy()], "i128 then i64")
    check([keys[8].slice(1, 398), keys[4].slice(1, 398)], "string then f32")
