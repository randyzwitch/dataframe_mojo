"""Hash-partitioned grouping must equal the serial path at every key dtype,
null pattern and cardinality, in order when asked and as a set otherwise."""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, Expr, Series, col
from dataframe.parallel import MIN_ROWS_PER_WORKER, worker_count
from dataframe.partition import low_cardinality

comptime ROWS = 200_000


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


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
