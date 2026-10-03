"""Hash-partitioned grouping must equal the serial path at every key dtype,
null pattern and cardinality, in order when asked and as a set otherwise."""
from std.collections import Dict
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    Expr,
    Series,
    StringColumn,
    col,
    concat,
)
from dataframe.parallel import MIN_ROWS_PER_WORKER, worker_count
from dataframe.partition import (
    _hash_bytes,
    _hash_column,
    low_cardinality,
    small_key_product,
)
from dataframe.string_view import StringViewBuilder

comptime ROWS = 200_000


def test_string_hash_agrees_across_storage_and_slices() raises:
    var texts: List[String] = [
        "",
        "a",
        "ab",
        "abc",
        "abcd",
        "abcde",
        "abcdef",
        "abcdefg",
        "abcdefgh",
        "abcdefghi",
        "é",
        "",
        "abcdefgh",
    ]
    var valid: List[Bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
    ]
    # Cross inline-view and every eight-byte tail boundary with exact bytes.
    for size in range(9, 81):
        var text = String()
        for i in range(size):
            text += "a" if i % 3 else "\x00"
        texts.append(text^)
        valid.append(True)
    texts.append("多字节é共同前缀-tail")
    valid.append(True)
    var contiguous = Series("s", StringColumn(texts.copy(), valid.copy()))
    var builder = StringViewBuilder(len(texts))
    for i in range(len(texts)):
        if valid[i]:
            builder.append(StringSlice(texts[i]))
        else:
            builder.append_null()
    var view = Series("s", StringColumn(builder^.finish()))
    for offset in [0, 1, 5]:
        var count = len(texts) - offset
        var contiguous_slice = contiguous.slice(offset, count)
        var view_slice = view.slice(offset, count)
        var left = List[UInt64](length=count, fill=0)
        var right = List[UInt64](length=count, fill=0)
        _hash_column(contiguous_slice, 0, count, Int(left.unsafe_ptr()), True)
        _hash_column(view_slice, 0, count, Int(right.unsafe_ptr()), True)
        for i in range(count):
            assert_equal(left[i], right[i])


def test_word_hash_unaligned_spans_and_tail_boundaries() raises:
    # DuckDB block-mixer vectors, before this project's column finalizer.
    var lengths: List[Int] = [
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        23,
        24,
        25,
        31,
        32,
        33,
        63,
        64,
        65,
    ]
    var expected: List[UInt64] = [
        0x4EB2B2F4045FEE6E,
        0xDF7813878559DD45,
        0x9479384728E1A0C8,
        0x1498AF9B3C02397F,
        0xC3F38FCDBA43DAFA,
        0x7BA1BFD411625971,
        0xF6E78FD4EE1DCF34,
        0x847F657550CADE0B,
        0x132CDE3DDE0E5C27,
        0x16B92FDF0BCA1DF5,
        0xF5029816BBE3C848,
        0x2D998608B12EE554,
        0x2124460EC4E77A2A,
        0x67242CD4FC968359,
        0x0A998BE8D76643B5,
        0x36FF0DAD0DB27AC6,
        0xED2F5D98A9AE675D,
        0x9AE22B5143CF9FA9,
    ]
    for offset in range(8):
        for at in range(len(lengths)):
            var size = lengths[at]
            var bytes = List[UInt8](length=offset + size, fill=0xFF)
            for i in range(size):
                bytes[offset + i] = UInt8((i * 37 + 11) % 256)
            var window = Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=bytes.unsafe_ptr()
                .unsafe_offset(offset)
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                length=size,
            )
            assert_equal(_hash_bytes(window), expected[at])
            # Neither prefix bytes nor allocation alignment are key bytes.
            for i in range(offset):
                bytes[i] = 0
            assert_equal(_hash_bytes(window), expected[at])
            bytes[offset + size - 1] ^= 1
            assert_true(_hash_bytes(window) != expected[at])


def c_string(text: String) -> List[UInt8]:
    var bytes = List[UInt8]()
    bytes.extend(text.as_bytes())
    bytes.append(0)
    return bytes^


def set_threads(n: Int):
    var name = c_string("DATAFRAME_THREADS")
    var value = c_string(String(n))
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


def frame(rows: Int, cardinality: Int) raises -> DataFrame:
    """Keys of every dtype with nulls, NaN and -0.0, plus two value columns."""
    var i64 = List[Int64](capacity=rows)
    var i32 = List[Int32](capacity=rows)
    var u8 = List[UInt8](capacity=rows)
    var f64 = List[Float64](capacity=rows)
    var b = List[Bool](capacity=rows)
    var s = List[String](capacity=rows)
    var key_valid = List[Bool](capacity=rows)
    var v = List[Float64](capacity=rows)
    var n = List[Int64](capacity=rows)
    var v_valid = List[Bool](capacity=rows)
    var state = UInt64(7)
    for i in range(rows):
        state = state * 6364136223846793005 + 1442695040888963407
        var r = Int((state >> 33) % UInt64(cardinality))
        i64.append(Int64(r) - Int64(cardinality // 2))
        i32.append(Int32(r % 1000) - 500)
        u8.append(UInt8(r % 256))
        # Every fourth float key is NaN and every fifth is a signed zero, so
        # the hash must agree with encode_rows on both.
        if r % 4 == 3:
            f64.append(Float64(0) / Float64(0))
        elif r % 5 == 0:
            f64.append(-0.0 if i % 2 == 0 else 0.0)
        else:
            f64.append(Float64(r) / 8)
        b.append(r % 3 == 0)
        s.append("key_" + String(r))
        key_valid.append(i % 11 != 3)
        v.append(Float64(i % 977) / 4)
        n.append(Int64(i % 13) - 6)
        v_valid.append(i % 7 != 5)
    return DataFrame(
        [
            Series("i64", Column[Int64](i64^, key_valid.copy())),
            Series("i32", Column[Int32](i32^, key_valid.copy())),
            Series("u8", Column[UInt8](u8^, key_valid.copy())),
            Series("f64", Column[Float64](f64^, key_valid.copy())),
            Series("b", Column[Bool](b^, key_valid.copy())),
            Series("s", Column[String](s^, key_valid^)),
            Series("v", Column[Float64](v^, v_valid^)),
            Series("n", Column[Int64](n^)),
        ]
    )


def aggregates() -> List[Expr]:
    return [
        col("v").sum().alias("sum"),
        col("v").count().alias("count"),
        col("v").min().alias("min"),
        col("v").max().alias("max"),
        col("v").mean().alias("mean"),
        col("n").sum().alias("nsum"),
    ]


def check_keys(df: DataFrame, keys: List[String]) raises:
    set_threads(1)
    assert_equal(worker_count(df.height()), 1)
    var serial = df.group_by(keys, maintain_order=True).agg(aggregates())
    set_threads(32)
    assert_true(worker_count(df.height()) > 1, "partitioned path not taken")
    var ordered = df.group_by(keys, maintain_order=True).agg(aggregates())
    assert_true(serial.equals(ordered), "ordered partitioned result differs")
    var unordered = df.group_by(keys).agg(aggregates())
    assert_equal(unordered.height(), serial.height())
    var by = keys.copy()
    by.append("sum")
    by.append("count")
    assert_true(
        serial.sort(by).equals(unordered.sort(by)),
        "unordered partitioned result differs as a set",
    )
    set_threads(32)


def test_every_key_dtype_low_cardinality() raises:
    var df = frame(ROWS, 16)
    for key in ["i64", "i32", "u8", "f64", "b", "s"]:
        check_keys(df, [key])


def test_every_key_dtype_high_cardinality() raises:
    var df = frame(ROWS, ROWS // 10)
    for key in ["i64", "i32", "f64", "s"]:
        check_keys(df, [key])


def test_parallel_string_groups_across_misaligned_chunks() raises:
    var df = frame(2 * MIN_ROWS_PER_WORKER, 16)
    var key = df.column("s")
    var pieces = Series._from_chunks(
        [
            key.slice(0, 17001),
            key.slice(17001, 60000),
            key.slice(77001, len(key) - 77001),
        ]
    )
    check_keys(df.with_column(pieces^), ["s"])


def test_composite_keys() raises:
    var df = frame(ROWS, 300)
    check_keys(df, ["i64", "s"])
    check_keys(df, ["b", "f64", "u8"])
    check_keys(df, ["s", "i32", "b"])


def test_just_above_the_threshold_and_skewed() raises:
    var small = frame(2 * MIN_ROWS_PER_WORKER, 40)
    check_keys(small, ["s"])
    # Half the rows share one key.
    var rows = ROWS
    var k = List[Int64](capacity=rows)
    var v = List[Float64](capacity=rows)
    for i in range(rows):
        k.append(0 if i % 2 == 0 else Int64(i % 1000))
        v.append(Float64(i % 100))
    var skewed = DataFrame(
        [Series("k", Column[Int64](k^)), Series("v", Column[Float64](v^))]
    )
    set_threads(1)
    var serial = skewed.group_by("k", maintain_order=True).agg(
        col("v").sum().alias("sum")
    )
    set_threads(32)
    var parallel = skewed.group_by("k", maintain_order=True).agg(
        col("v").sum().alias("sum")
    )
    assert_true(serial.equals(parallel))
    assert_equal(parallel.item(0, "k").int64(), Int64(0))


def test_group_indices_numbering_matches_ordered_output() raises:
    var df = frame(ROWS, 200)
    set_threads(32)
    var ordered = df.group_by("s", maintain_order=True).agg(
        col("v").count().alias("count")
    )
    var groups = df.group_indices("s")
    assert_equal(groups.count(), ordered.height())
    var sizes = groups.sizes()
    var total = 0
    for size in sizes:
        total += size
    assert_equal(total, df.height())
    for g in range(groups.count()):
        var expected = df.item(groups.representative(g), "s")
        var actual = ordered.item(g, "s")
        if expected.is_null():
            assert_true(actual.is_null())
        else:
            assert_equal(expected.string(), actual.string())


def test_all_null_keys_and_single_group() raises:
    var rows = ROWS
    var nulls = List[Int64](length=rows, fill=1)
    var valid = List[Bool](length=rows, fill=False)
    var v = List[Float64](capacity=rows)
    for i in range(rows):
        v.append(Float64(i))
    var df = DataFrame(
        [
            Series("k", Column[Int64](nulls^, valid^)),
            Series("v", Column[Float64](v^)),
        ]
    )
    set_threads(32)
    var result = df.group_by("k").agg(col("v").sum().alias("sum"))
    assert_equal(result.height(), 1)
    assert_true(result.item(0, "k").is_null())
    assert_equal(
        result.item(0, "sum").float64(), Float64(rows) * Float64(rows - 1) / 2
    )


def test_worker_range_aggregation_matches_row_by_row_totals() raises:
    """Any list of reductions on a low-cardinality key, with nulls in both
    the key and the values, agrees with totals computed row by row."""
    var n = 4 * MIN_ROWS_PER_WORKER + 17
    var keys = List[Int64](capacity=n)
    var key_valid = List[Bool](capacity=n)
    var xs = List[Float64](capacity=n)
    var x_valid = List[Bool](capacity=n)
    var ns = List[Int64](capacity=n)
    # Slot 7 is the null key.
    var sums = List[Float64](length=8, fill=0)
    var counts = List[Int64](length=8, fill=0)
    var lows = List[Int64](length=8, fill=Int64.MAX)
    var highs = List[Int64](length=8, fill=Int64.MIN)
    for i in range(n):
        var slot = 7 if i % 11 == 0 else i % 7
        keys.append(Int64(slot))
        key_valid.append(slot != 7)
        var value = Float64(i % 101) - 50
        xs.append(value)
        x_valid.append(i % 13 != 0)
        ns.append(Int64(i % 997) - 400)
        if i % 13 != 0:
            sums[slot] += value
            counts[slot] += 1
        lows[slot] = min(lows[slot], ns[i])
        highs[slot] = max(highs[slot], ns[i])
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys^, key_valid^)),
            Series("x", Column[Float64](xs^, x_valid^)),
            Series("n", Column[Int64](ns^)),
        ]
    )
    assert_true(worker_count(n) > 1)
    var one: List[Expr] = [col("x").sum().alias("s")]
    var three: List[Expr] = [
        col("x").sum().alias("s"),
        col("x").count().alias("c"),
        col("x").mean().alias("m"),
    ]
    var extremes: List[Expr] = [
        col("n").min().alias("lo"),
        col("n").max().alias("hi"),
    ]
    var lists = List[List[Expr]]()
    lists.append(one^)
    lists.append(three^)
    lists.append(extremes^)
    for exprs in lists:
        var result = frame.group_by("k").agg(exprs)
        assert_equal(result.height(), 8)
        # Groups keep first-occurrence order: the null key comes first.
        assert_true(result.item(0, "k").is_null())
        for row in range(result.height()):
            var cell = result.item(row, "k")
            var slot = 7 if cell.is_null() else Int(cell.int64())
            if "s" in result.columns():
                assert_equal(result.item(row, "s").float64(), sums[slot])
            if "c" in result.columns():
                assert_equal(result.item(row, "c").int64(), counts[slot])
                assert_equal(
                    result.item(row, "m").float64(),
                    sums[slot] / Float64(counts[slot]),
                )
            if "lo" in result.columns():
                assert_equal(result.item(row, "lo").int64(), lows[slot])
                assert_equal(result.item(row, "hi").int64(), highs[slot])


def test_few_values_per_key_group_by_ranges() raises:
    """Two string keys of 60 and 70 values form 4,200 pairs: a sample of
    rows looks mostly distinct, but each key's sample shows its few values,
    so the pairs group by worker ranges. Totals, first-occurrence order and
    the null key must match a row-by-row reference, over chunked keys."""
    var n = 4 * MIN_ROWS_PER_WORKER + 17
    var a = List[String](capacity=n)
    var a_valid = List[Bool](capacity=n)
    var b = List[String](capacity=n)
    var v = List[Int64](capacity=n)
    var totals = Dict[String, Int64]()
    var order = List[String]()
    for i in range(n):
        var x = (i * 7919) % 60
        var y = (i * 104729) % 70
        a.append("left" + String(x))
        a_valid.append(i % 97 != 5)
        b.append("right" + String(y))
        v.append(Int64(i % 1000))
        var pair = (a[i] if a_valid[i] else String("<null>")) + "|" + b[i]
        if pair not in totals:
            totals[pair] = 0
            order.append(pair)
        totals[pair] += v[i]
    var whole = DataFrame(
        [
            Series("a", StringColumn(a, a_valid)),
            Series("b", Column[String](b^)),
            Series("v", Column[Int64](v^)),
        ]
    )
    var half = n // 2
    var frame = concat([whole.slice(0, half), whole.slice(half, n - half)])
    var keys: List[Series] = [frame.column("a"), frame.column("b")]
    assert_true(small_key_product(keys))
    assert_true(not small_key_product([frame.column("v"), frame.column("b")]))
    var result = frame.group_by(["a", "b"]).agg([col("v").sum().alias("s")])
    assert_equal(result.height(), len(order))
    for row in range(result.height()):
        var cell = result.item(row, "a")
        var pair = (
            (String("<null>") if cell.is_null() else cell.string())
            + "|"
            + result.item(row, "b").string()
        )
        assert_equal(pair, order[row])
        assert_equal(result.item(row, "s").int64(), totals[pair])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_exact_hash_keys_group_like_their_values() raises:
    """A single numeric key without nulls is grouped by its bijective hash
    alone (no row comparison). Float keys still fold -0.0 into 0.0 and
    every NaN into one group, and integer keys of every width agree with
    a row-by-row count, at a cardinality that takes the partitioned path."""
    var n = 200_000
    var floats = List[Float64](capacity=n)
    var ints = List[Int32](capacity=n)
    var nan = Float64(0) / Float64(0)
    for i in range(n):
        var v = Float64((i * 7919) % 50_000)
        if i % 1000 == 1:
            v = nan
        elif i % 1000 == 2:
            v = -0.0
        elif i % 1000 == 3:
            v = 0.0
        floats.append(v)
        ints.append(Int32((i * 104729) % 60_000 - 30_000))
    var frame = DataFrame(
        [
            Series("f", Column[Float64](floats^)),
            Series("i", Column[Int32](ints^)),
        ]
    )
    var by_float = frame.group_by("f").agg([col("i").len().alias("c")])
    var nans = 0
    var zeros = 0
    var total = 0
    for row in range(by_float.height()):
        var key = by_float.item(row, "f").float64()
        var count = Int(by_float.item(row, "c").int64())
        total += count
        if key != key:
            nans += 1
            assert_equal(count, n // 1000)
        elif key == 0:
            zeros += 1
    assert_equal(nans, 1)
    assert_equal(zeros, 1)
    assert_equal(total, n)
    var by_int = frame.group_by("i").agg([col("f").len().alias("c")])
    var counts = Dict[Int, Int]()
    for i in range(n):
        var k = Int(frame.item(i, "i").int32())
        counts[k] = counts.get(k, 0) + 1
    assert_equal(by_int.height(), len(counts))
    for row in range(by_int.height()):
        var k = Int(by_int.item(row, "i").int32())
        assert_equal(Int(by_int.item(row, "c").int64()), counts[k])


def test_chunked_cardinality_sample_matches_rechunked() raises:
    var rows = 8192
    for cardinality in [16, 1000]:
        var strings = List[String](capacity=rows)
        var numbers = List[Int64](capacity=rows)
        var valid = List[Bool](capacity=rows)
        for i in range(rows):
            strings.append("key_" + String(i % cardinality))
            numbers.append(Int64((i * 17) % cardinality))
            valid.append(i % 13 != 4)
        var string_key = Series("s", Column[String](strings^, valid.copy()))
        var number_key = Series("n", Column[Int64](numbers^, valid^))
        var chunked_strings = Series._from_chunks(
            [
                string_key.slice(0, 1371),
                string_key.slice(1371, 3629),
                string_key.slice(5000, rows - 5000),
            ]
        )
        var chunked_numbers = Series._from_chunks(
            [
                number_key.slice(0, 701),
                number_key.slice(701, 2299),
                number_key.slice(3000, rows - 3000),
            ]
        )
        assert_equal(
            low_cardinality([chunked_strings.copy()]),
            low_cardinality([string_key.copy()]),
        )
        assert_equal(
            low_cardinality([chunked_strings.copy(), chunked_numbers.copy()]),
            low_cardinality([string_key.copy(), number_key.copy()]),
        )


def test_skewed_small_domain_uses_whole_encoding() raises:
    # A hot key with a small remaining domain is cheaper without scattering.
    # The same hot key plus unique trailing values still needs partitioning.
    var small_values = List[Int64](capacity=100_000)
    var unique_values = List[Int64](capacity=100_000)
    for i in range(100_000):
        small_values.append(Int64(0 if i % 2 == 0 else i % 1000))
        unique_values.append(Int64(0 if i % 2 == 0 else i))
    var small = Series("small", Column[Int64](small_values^))
    var unique = Series("unique", Column[Int64](unique_values^))
    assert_true(low_cardinality([small.copy()]))
    assert_true(not low_cardinality([unique.copy()]))
    var small_chunks = Series._from_chunks(
        [small.slice(0, 37_001), small.slice(37_001, 62_999)]
    )
    var unique_chunks = Series._from_chunks(
        [unique.slice(0, 37_001), unique.slice(37_001, 62_999)]
    )
    assert_true(low_cardinality([small_chunks^]))
    assert_true(not low_cardinality([unique_chunks^]))


def test_fused_low_cardinality_sum_count_matches_serial() raises:
    var source = frame(ROWS, 16)
    var n_values = source.column("n").int64().to_list()
    var n_valid = List[Bool](capacity=ROWS)
    for i in range(ROWS):
        n_valid.append(i % 5 != 2)
    source = source.with_column(Series("n", Column[Int64](n_values^, n_valid^)))
    for chunked in [False, True]:
        var df = source.copy()
        if chunked:
            for name in ["i64", "v", "n"]:
                var column = df.column(name)
                var split = 37_001 if name != "v" else 61_003
                df = df.with_column(
                    Series._from_chunks(
                        [
                            column.slice(0, split),
                            column.slice(split, ROWS - split),
                        ]
                    )
                )
        var expressions: List[Expr] = [
            col("v").sum().alias("sum"),
            col("n").count().alias("count"),
        ]
        for reversed in [False, True]:
            if reversed:
                expressions.reverse()
            set_threads(1)
            var serial = df.group_by("i64", maintain_order=True).agg(
                expressions
            )
            set_threads(32)
            var parallel = df.group_by("i64", maintain_order=True).agg(
                expressions
            )
            assert_true(
                serial.equals(parallel),
                "fused low-cardinality result differs",
            )
            var unordered = df.group_by("i64").agg(expressions)
            assert_true(
                serial.equals(unordered),
                "fused low-cardinality order differs",
            )
    # All-valid COUNT needs no payload read, regardless of its dtype.
    for counted_name in ["b", "s"]:
        var generic_count: List[Expr] = [
            col("v").sum().alias("sum"),
            col(counted_name).count().alias("count"),
        ]
        set_threads(1)
        var serial_generic = source.group_by("i64").agg(generic_count)
        set_threads(32)
        assert_true(
            serial_generic.equals(source.group_by("i64").agg(generic_count)),
            "all-valid generic count differs",
        )
    # Counting a nullable Float64 column keeps the general reduction path.
    var float_count: List[Expr] = [
        col("v").sum().alias("sum"),
        col("v").count().alias("count"),
    ]
    set_threads(1)
    var serial_float_count = source.group_by("i64").agg(float_count)
    set_threads(32)
    assert_true(
        serial_float_count.equals(source.group_by("i64").agg(float_count)),
        "float count fallback differs",
    )
    set_threads(32)


def test_direct_numeric_sum_count_matches_serial_with_nulls() raises:
    var df = frame(ROWS, ROWS // 10)
    var n_values = df.column("n").int64().to_list()
    var n_valid = List[Bool](capacity=ROWS)
    for i in range(ROWS):
        n_valid.append(i % 5 != 2)
    df = df.with_column(Series("n", Column[Int64](n_values^, n_valid^)))
    var key = df.column("i64")
    var value = df.column("v")
    df = df.with_column(
        Series._from_chunks(
            [key.slice(0, 37_001), key.slice(37_001, ROWS - 37_001)]
        )
    ).with_column(
        Series._from_chunks(
            [value.slice(0, 61_003), value.slice(61_003, ROWS - 61_003)]
        )
    )
    var expressions: List[Expr] = [
        col("v").sum().alias("sum"),
        col("n").count().alias("count"),
    ]
    for reversed in [False, True]:
        if reversed:
            expressions.reverse()
        set_threads(1)
        var serial = df.group_by("i64", maintain_order=True).agg(expressions)
        set_threads(32)
        var parallel = df.group_by("i64", maintain_order=True).agg(expressions)
        assert_true(serial.equals(parallel), "direct grouped result differs")
        var unordered = df.group_by("i64").agg(expressions)
        assert_true(
            serial.sort("i64").equals(unordered.sort("i64")),
            "direct unordered grouped result differs",
        )
    set_threads(32)


def test_direct_numeric_sum_count_reads_aligned_chunks() raises:
    var df = frame(ROWS, ROWS // 10)
    var n_values = df.column("n").int64().to_list()
    var n_valid = List[Bool](capacity=ROWS)
    for i in range(ROWS):
        n_valid.append(i % 5 != 2)
    df = df.with_column(Series("n", Column[Int64](n_values^, n_valid^)))
    for name in ["i64", "v", "n"]:
        var source = df.column(name)
        df = df.with_column(
            Series._from_chunks(
                [
                    source.slice(0, 37_001),
                    source.slice(37_001, ROWS - 37_001),
                ]
            )
        )
    var expressions: List[Expr] = [
        col("v").sum().alias("sum"),
        col("n").count().alias("count"),
    ]
    set_threads(1)
    var serial = df.group_by("i64", maintain_order=True).agg(expressions)
    set_threads(32)
    var parallel = df.group_by("i64", maintain_order=True).agg(expressions)
    assert_true(
        serial.equals(parallel), "aligned direct grouped result differs"
    )
    var unordered = df.group_by("i64").agg(expressions)
    assert_true(
        serial.sort("i64").equals(unordered.sort("i64")),
        "aligned direct unordered grouped result differs",
    )


def test_indexed_reductions_match_serial() raises:
    """Plain reductions on the partitioned path read values at their
    source rows (`indexed_reduce.mojo`): NaN in min and max, Int64 with
    nulls under every reduction, len, and values split into chunks."""
    var df = frame(ROWS, ROWS // 10)
    var n_values = df.column("n").int64().to_list()
    var n_valid = List[Bool](capacity=ROWS)
    var w = List[Float64](capacity=ROWS)
    for i in range(ROWS):
        n_valid.append(i % 5 != 2)
        w.append(Float64(0) / Float64(0) if i % 9 == 4 else Float64(i % 31))
    df = df.with_column(Series("n", Column[Int64](n_values^, n_valid^)))
    df = df.with_column(Series("w", Column[Float64](w^)))
    var source = df.column("n")
    df = df.with_column(
        Series._from_chunks(
            [source.slice(0, 37_001), source.slice(37_001, ROWS - 37_001)]
        )
    )
    var expressions: List[Expr] = [
        col("n").sum().alias("nsum"),
        col("n").mean().alias("nmean"),
        col("n").min().alias("nmin"),
        col("n").max().alias("nmax"),
        col("n").count().alias("ncount"),
        col("w").min().alias("wmin"),
        col("w").max().alias("wmax"),
        col("w").count().alias("wcount"),
        col("v").len().alias("len"),
    ]
    var key_sets: List[List[String]] = [["s"], ["s", "i32"]]
    for keys in key_sets:
        set_threads(1)
        var serial = df.group_by(keys, maintain_order=True).agg(expressions)
        set_threads(32)
        var parallel = df.group_by(keys, maintain_order=True).agg(expressions)
        assert_true(serial.equals(parallel), "indexed grouped result differs")
