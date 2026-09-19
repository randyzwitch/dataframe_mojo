"""Arrow large_utf8 string columns: layout, windows, and bulk operations."""
from std.testing import TestSuite, assert_equal, assert_true, assert_false
from dataframe import (
    Column,
    DataFrame,
    Series,
    StringBuilder,
    StringColumn,
    col,
    concat,
    lit,
)


def sample() raises -> StringColumn:
    # Empty strings, nulls, ASCII, and 2-, 3-, and 4-byte UTF-8.
    return StringColumn(
        ["a", "", "héllo", "", "日本", "🔥x", "zz", ""],
        [True, True, True, False, True, True, False, True],
    )


def expected(column: StringColumn) raises -> List[String]:
    var out = List[String]()
    for i in range(len(column)):
        out.append("<null>" if column.is_null(i) else column.value(i))
    return out^


def test_layout_is_arrow_large_utf8() raises:
    var column = sample()
    assert_equal(len(column._offsets[]), len(column) + 1)
    assert_equal(column._offsets[][0], Int64(0))
    assert_equal(Int(column._offsets[][len(column)]), len(column._bytes[]))
    for i in range(len(column)):
        assert_true(column._offsets[][i] <= column._offsets[][i + 1])
    # Nulls occupy no bytes; "héllo" is 6 bytes, "日本" 6, "🔥x" 5.
    assert_equal(column._byte_length(2), 6)
    assert_equal(column._byte_length(3), 0)
    assert_equal(column._byte_length(4), 6)
    assert_equal(column._byte_length(5), 5)
    assert_equal(column._bits[][0], UInt8(0b10110111))


def test_empty_strings_are_not_nulls() raises:
    var column = sample()
    assert_false(column.is_null(1))
    assert_equal(column.value(1), "")
    assert_true(column.is_null(3))
    assert_equal(column.null_count(), 2)
    var s = Series("s", column.copy())
    assert_equal(s.null_count(), 2)
    assert_true(s.get(1).is_null() == False)
    assert_equal(s.get(1).string(), "")
    assert_true(s.get(3).is_null())
    var frame = DataFrame([s.copy()])
    var nulls = frame.select(col("s").is_null().alias("n")).column("n").bool()
    assert_false(nulls.value(1))
    assert_true(nulls.value(3))
    assert_equal(frame.select(col("s").n_unique()).item(0, "s").int64(), 6)


def test_take_duplicates_and_nulls() raises:
    var column = sample()
    var taken = column.take([5, 5, 3, 0, 4, 5, 1])
    assert_equal(
        expected(taken),
        ["🔥x", "🔥x", "<null>", "a", "日本", "🔥x", ""],
    )
    var outer = column.take_or_null([-1, 2, -1, 7], "")
    assert_equal(expected(outer), ["<null>", "héllo", "<null>", ""])


def test_slices_share_buffers_at_every_boundary() raises:
    var column = sample()
    var all = expected(column)
    for start in range(len(column) + 1):
        for length in range(len(column) - start + 1):
            var window = column.slice(start, length)
            assert_true(window._shares_buffers_with(column))
            assert_equal(len(window), length)
            for i in range(length):
                assert_equal(expected(window)[i], all[start + i])
            var compact = window._compact()
            assert_false(compact._shares_buffers_with(column))
            assert_equal(compact._offsets[][0], Int64(0))
            assert_equal(expected(compact), expected(window))


def test_append_windows_copy_on_write() raises:
    var column = sample()
    var before = expected(column)
    for start in range(len(column)):
        for extra in range(len(column) - start + 1):
            var left = column.slice(start, len(column) - start)
            var right = column.slice(len(column) - extra, extra)
            var joined = left.copy()
            joined._append_column(right)
            var want = expected(left)
            for value in expected(right):
                want.append(value)
            assert_equal(expected(joined), want)
            assert_equal(expected(column), before)
            assert_equal(expected(left), List[String](before[start:]))
    # Series-level append and frame concat go through the same path.
    var a = Series("s", column.slice(1, 3))
    var b = Series("s", column.slice(4, 4))
    assert_equal(len(a.append(b)), 7)
    assert_equal(a.append(b).get(6).string(), "")
    var frame = DataFrame([Series("s", column.copy())])
    var stacked = concat([frame.slice(5, 3), frame.slice(0, 3)])
    assert_equal(stacked.item(0, "s").string(), "🔥x")
    assert_equal(stacked.item(4, "s").string(), "")


def test_broadcast_and_nulls() raises:
    var one = StringColumn(["日"])
    var wide = one._broadcast(4)
    assert_equal(expected(wide), ["日", "日", "日", "日"])
    assert_equal(len(wide._bytes[]), 12)
    var null = StringColumn([""], [False])._broadcast(3)
    assert_equal(null.null_count(), 3)
    var empty = StringColumn._nulls(0)
    assert_equal(len(empty), 0)
    assert_equal(len(empty._offsets[]), 1)
    var nothing = StringColumn(List[String]())
    nothing._append_column(sample().slice(2, 3))
    assert_equal(expected(nothing), ["héllo", "<null>", "日本"])


def test_builder_pop_restores_state() raises:
    var builder = StringBuilder()
    for i in range(9):
        if i % 3 == 0:
            builder.append_null()
        else:
            builder.append(String("v") + String(i))
    builder._pop()
    builder._pop()
    builder.append("🔥")
    var column = builder^.finish()
    assert_equal(len(column), 8)
    assert_equal(
        expected(column),
        ["<null>", "v1", "v2", "<null>", "v4", "v5", "<null>", "🔥"],
    )
    assert_equal(Int(column._offsets[][8]), len(column._bytes[]))


def test_list_backed_constructor_converts() raises:
    var s = Series("s", Column[String](["x", "yy", ""], [True, False, True]))
    assert_true(s.string().is_null(1))
    assert_equal(s.string().value(0), "x")
    assert_equal(s.string()._to_list(), ["x", "", ""])


def test_kernels_on_multibyte_content() raises:
    var frame = DataFrame([Series("s", sample())])
    var result = frame.select_exprs(
        [
            col("s").str().len_chars().alias("chars"),
            col("s").str().len_bytes().alias("bytes"),
            col("s").str().to_uppercase().alias("upper"),
            (col("s") < lit("b")).alias("lt"),
            (col("s") >= lit("日")).alias("ge"),
        ]
    )
    assert_equal(result.item(2, "chars").int64(), 5)
    assert_equal(result.item(2, "bytes").int64(), 6)
    assert_equal(result.item(2, "upper").string(), "HÉLLO")
    assert_true(result.item(3, "upper").is_null())
    assert_true(result.item(0, "lt").bool())
    assert_true(result.item(4, "ge").bool())
    assert_true(result.item(5, "ge").bool())  # U+1F525 sorts after U+65E5
    assert_false(result.item(2, "ge").bool())
    var sorted = frame.sort(["s"])
    assert_equal(sorted.item(0, "s").string(), "")
    assert_equal(sorted.item(2, "s").string(), "a")
    assert_equal(sorted.item(5, "s").string(), "🔥x")
    assert_true(sorted.item(6, "s").is_null())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
