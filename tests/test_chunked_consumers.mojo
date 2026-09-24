"""Differential coverage for consumers of multi-array Series values."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe.gather import (
    _take_sorted_chunked_partitioned,
    take_parallel,
    take_sorted_chunked,
)
from dataframe import (
    Column,
    DataFrame,
    Expr,
    Series,
    col,
    concat_str,
    lit,
    to_csv_string,
)


def contiguous() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "i",
                Column[Int64](
                    [4, 1, 7, 2, 6, 3, 5, 8],
                    [True, False, True, True, True, False, True, True],
                ),
            ),
            Series(
                "f",
                Column[Float64](
                    [1.5, 2.5, -3.0, 4.0, 5.5, 6.0, -7.0, 8.5],
                    [True, True, False, True, True, True, False, True],
                ),
            ),
            Series(
                "ok",
                Column[Bool](
                    [True, False, True, False, True, True, False, True],
                    [True, False, True, True, True, True, False, True],
                ),
            ),
            Series(
                "key",
                Column[String](
                    ["b", "a", "b", "a", "c", "a", "c", "b"],
                    [True, True, True, False, True, True, True, True],
                ),
            ),
            Series(
                "s",
                Column[String](
                    [
                        "one",
                        "two",
                        "THREE",
                        "four",
                        "é",
                        "six",
                        "seven",
                        "eight",
                    ],
                    [True, True, False, True, True, False, True, True],
                ),
            ),
        ]
    )


def chunked() raises -> DataFrame:
    # Each column breaks at different rows. Nulls land both within chunks and
    # immediately adjacent to a boundary, so a consumer cannot use chunk zero
    # as if it held the whole series.
    var i = Series._from_chunks(
        [
            Series("i", Column[Int64]([4, 1, 7], [True, False, True])),
            Series("i", Column[Int64]([2, 6], [True, True])),
            Series("i", Column[Int64]([3, 5, 8], [False, True, True])),
        ]
    )
    var f = Series._from_chunks(
        [
            Series("f", Column[Float64]([1.5, 2.5], [True, True])),
            Series(
                "f",
                Column[Float64](
                    [-3.0, 4.0, 5.5, 6.0], [False, True, True, True]
                ),
            ),
            Series("f", Column[Float64]([-7.0, 8.5], [False, True])),
        ]
    )
    var ok = Series._from_chunks(
        [
            Series(
                "ok",
                Column[Bool](
                    [True, False, True, False], [True, False, True, True]
                ),
            ),
            Series("ok", Column[Bool]([True], [True])),
            Series(
                "ok", Column[Bool]([True, False, True], [True, False, True])
            ),
        ]
    )
    var key = Series._from_chunks(
        [
            Series("key", Column[String](["b", "a"], [True, True])),
            Series("key", Column[String](["b", "a", "c"], [True, False, True])),
            Series("key", Column[String](["a", "c", "b"], [True, True, True])),
        ]
    )
    var s = Series._from_chunks(
        [
            Series("s", Column[String](["one"], [True])),
            Series(
                "s",
                Column[String](["two", "THREE", "four"], [True, False, True]),
            ),
            Series(
                "s",
                Column[String](
                    ["é", "six", "seven", "eight"], [True, False, True, True]
                ),
            ),
        ]
    )
    return DataFrame([i^, f^, ok^, key^, s^])


def assert_same(actual: DataFrame, expected: DataFrame, label: String) raises:
    assert_true(
        actual.equals(expected),
        label
        + "\nactual:\n"
        + String(actual)
        + "\nexpected:\n"
        + String(expected),
    )


def test_chunked_reductions_elementwise_filter_and_cast() raises:
    var whole = contiguous()
    var parts = chunked()
    assert_true(parts.equals(whole))
    assert_true(parts.column("i").is_chunked())
    assert_equal(parts.column("i").n_chunks(), 3)
    var first = parts.column("i").chunks()[0].copy()

    var reductions = List[Expr]()
    reductions.append(col("i").sum().alias("i_sum"))
    reductions.append(col("f").mean().alias("f_mean"))
    reductions.append(col("ok").count().alias("ok_count"))
    reductions.append(col("s").n_unique().alias("s_unique"))
    assert_same(
        parts.select_exprs(reductions, batch_size=2),
        whole.select_exprs(reductions, batch_size=2),
        "reductions",
    )

    var expressions = List[Expr]()
    expressions.append((col("i") + lit(Int64(10))).alias("i_plus"))
    expressions.append((col("f") * lit(Float64(2))).alias("f_twice"))
    expressions.append(col("ok").is_not_null().alias("ok_present"))
    assert_same(
        parts.select_exprs(expressions, batch_size=2),
        whole.select_exprs(expressions, batch_size=2),
        "elementwise",
    )
    assert_same(
        parts.filter(col("i") > lit(Int64(3)), batch_size=2),
        whole.filter(col("i") > lit(Int64(3)), batch_size=2),
        "filter",
    )
    assert_same(
        parts.cast({"i": "string", "f": "int64"}, strict=False),
        whole.cast({"i": "string", "f": "int64"}, strict=False),
        "cast",
    )

    # Read-only reductions and expression evaluation retain every original
    # chunk and still share the original Arrow buffer.
    _ = parts.column("i").sum()
    _ = parts.select((col("i") + lit(Int64(1))).alias("out"), batch_size=2)
    assert_equal(len(parts.column("i")), 8)
    assert_equal(parts.column("i").get(7).int64(), 8)
    assert_true(
        parts.column("i")
        .chunks()[0]
        .int64()
        ._shares_buffers_with(first.int64())
    )


def test_chunked_strings_hash_sort_display_and_gather() raises:
    var whole = contiguous()
    var parts = chunked()

    var string_expr = concat_str(
        [col("key"), lit(String("/")), col("s").str().to_uppercase()]
    ).alias("text")
    assert_same(
        parts.select(string_expr, batch_size=2),
        whole.select(string_expr, batch_size=2),
        "strings",
    )
    assert_same(
        parts.take([7, 0, 3, 6, 1]), whole.take([7, 0, 3, 6, 1]), "gather"
    )
    assert_same(parts.slice(1, 6), whole.slice(1, 6), "sliced frame")
    assert_same(parts.sort(["key", "i"]), whole.sort(["key", "i"]), "sort")

    var grouped = List[Expr]()
    grouped.append(col("i").sum().alias("total"))
    grouped.append(col("s").count().alias("n"))
    assert_same(
        parts.group_by(["key", "ok"], maintain_order=True).agg(
            grouped, batch_size=2
        ),
        whole.group_by(["key", "ok"], maintain_order=True).agg(
            grouped, batch_size=2
        ),
        "row hashing and grouping",
    )
    assert_same(
        parts.group_by("key", maintain_order=True).agg(grouped, batch_size=2),
        whole.group_by("key", maintain_order=True).agg(grouped, batch_size=2),
        "ordered single-key grouping",
    )
    assert_same(
        parts.group_by("key").len(),
        whole.group_by("key").len(),
        "chunked group lengths",
    )
    assert_equal(String(parts), String(whole))
    assert_equal(
        parts.column("s").to_string(max_rows=4),
        whole.column("s").to_string(max_rows=4),
    )
    assert_equal(to_csv_string(parts), to_csv_string(whole))


def test_parallel_gather_rechunks_misaligned_inputs() raises:
    var source = chunked()
    var rows: List[Int] = [7, 0, 3, 6, 1]
    var gathered = take_parallel(source._columns, rows.copy(), 3)
    assert_same(
        DataFrame(gathered^),
        contiguous().take(rows),
        "parallel gather from misaligned chunks",
    )


def test_large_chunked_filter_with_misaligned_columns_and_null_mask() raises:
    var ids = List[Int64]()
    var labels = List[String]()
    var chosen = List[Bool]()
    var valid = List[Bool]()
    for i in range(1024):
        ids.append(Int64(i))
        labels.append(String("key_", i % 7))
        chosen.append(i % 3 == 0)
        valid.append(i % 17 != 0)
    var id = Series("id", Column[Int64](ids^))
    var label = Series("label", Column[String](labels^))
    var whole = DataFrame([id.copy(), label.copy()])
    var parts = DataFrame(
        [
            Series._from_chunks([id.slice(0, 512), id.slice(512, 512)]),
            Series._from_chunks([label.slice(0, 600), label.slice(600, 424)]),
        ]
    )
    var mask = Column[Bool](chosen^, valid^)
    var actual = parts.filter(mask)
    assert_same(actual, whole.filter(mask), "large chunked filter")
    assert_true(actual.column("id").is_chunked())
    var reject_all = Column[Bool](List[Bool](length=1024, fill=False))
    assert_same(
        parts.filter(reject_all), whole.filter(reject_all), "empty filter"
    )


def test_filter_reuses_fully_selected_chunks() raises:
    var numbers = List[Int64]()
    var labels = List[String]()
    var keep = List[Bool]()
    for i in range(1536):
        numbers.append(Int64(i))
        labels.append(String("item_", i))
        keep.append(i >= 512 and i < 1100)
    var id = Series("id", Column[Int64](numbers^))
    var label = Series("label", Column[String](labels^))
    var whole = DataFrame([id.copy(), label.copy()])
    var parts = DataFrame(
        [
            Series._from_chunks(
                [id.slice(0, 512), id.slice(512, 512), id.slice(1024, 512)]
            ),
            Series._from_chunks(
                [
                    label.slice(0, 512),
                    label.slice(512, 512),
                    label.slice(1024, 512),
                ]
            ),
        ]
    )
    var mask = Column[Bool](keep^)
    assert_same(parts.filter(mask), whole.filter(mask), "whole chunk filter")


def test_partitioned_sorted_chunk_gather_preserves_order_and_nulls() raises:
    var numbers = List[Int64]()
    var labels = List[String]()
    var valid = List[Bool]()
    for i in range(1024):
        numbers.append(Int64(i))
        labels.append("key_" + String(i % 11))
        valid.append(i % 17 != 0)
    var id = Series("id", Column[Int64](numbers^, valid.copy()))
    var label = Series("label", Column[String](labels^, valid^))
    var whole = DataFrame([id.copy(), label.copy()])
    var id_parts = List[Series]()
    var label_parts = List[Series]()
    for c in range(16):
        id_parts.append(id.slice(c * 64, 64))
        var first = c * 63
        label_parts.append(label.slice(first, 79 if c == 15 else 63))
    var parts = DataFrame(
        [
            Series._from_chunks(id_parts^),
            Series._from_chunks(label_parts^),
        ]
    )
    var rows = List[Int]()
    for i in range(1024):
        if i % 3 == 0 or (256 <= i and i < 320) or i == 1023:
            rows.append(i)
    var expected = whole.take(rows.copy())
    var actual = DataFrame(
        _take_sorted_chunked_partitioned(parts._columns, rows^, 4)
    )
    assert_same(actual, expected, "partitioned sorted chunk gather")
    assert_true(actual.column("id").is_chunked())
    assert_true(actual.column("label").is_chunked())

    # The first chunk has exactly 64 selected positions and matching
    # endpoints, but row 0 repeats and row 1 is absent. It cannot be shared
    # as a whole chunk merely because its endpoints match.
    var repeated = List[Int]([0, 0])
    for i in range(2, 64):
        repeated.append(i)
    repeated.append(64)
    repeated.append(64)
    repeated.append(1023)
    var repeat_expected = whole.take(repeated.copy())
    var repeat_serial = DataFrame(
        take_sorted_chunked(parts._columns, repeated.copy(), 4, True)
    )
    var repeat_partitioned = DataFrame(
        _take_sorted_chunked_partitioned(parts._columns, repeated^, 4, True)
    )
    assert_same(repeat_serial, repeat_expected, "ordered repeated chunk gather")
    assert_same(
        repeat_partitioned,
        repeat_expected,
        "partitioned ordered repeated chunk gather",
    )


def test_parallel_expression_output_stays_chunked_and_reduces() raises:
    var values = List[Float64]()
    var valid = List[Bool]()
    var expected = 0.0
    for i in range(131072):
        var value = Float64(i % 31)
        var is_valid = i % 17 != 0
        values.append(value)
        valid.append(is_valid)
        if is_valid:
            expected += value + 2.0
    var frame = DataFrame([Series("x", Column[Float64](values^, valid^))])
    var result = frame.with_columns((col("x") + lit(Float64(2))).alias("out"))
    assert_true(result.column("out").is_chunked())
    assert_equal(result.column("out").n_chunks(), 2)
    assert_equal(result.column("out").get(65536).float64(), 4.0)
    assert_equal(result.select(col("out").sum()).item().float64(), expected)


def test_direct_float_filter_matches_boolean_mask() raises:
    var nan = Float64(0) / Float64(0)
    var x = Series._from_chunks(
        [
            Series("x", Column[Float64]([nan, -2.0], [True, True])),
            Series("x", Column[Float64]([-0.0, 0.0], [True, False])),
            Series("x", Column[Float64]([2.0, 3.0], [True, True])),
        ]
    )
    var frame = DataFrame(
        [x^, Series("row", Column[Int64]([0, 1, 2, 3, 4, 5]))]
    )
    var predicates = List[Expr]()
    predicates.append(col("x") > lit(Float64(0)))
    predicates.append(col("x") < lit(Float64(0)))
    predicates.append(col("x") >= lit(Float64(0)))
    predicates.append(col("x") <= lit(Float64(0)))
    predicates.append(col("x") == lit(Float64(0)))
    predicates.append(col("x") != lit(Float64(0)))
    for predicate in predicates:
        var mask = frame.select(predicate.alias("mask")).column("mask").bool()
        assert_same(
            frame.filter(predicate), frame.filter(mask), "Float64 filter"
        )
    with assert_raises():
        _ = frame.filter(predicates[0], batch_size=0)


def test_direct_float_filter_crosses_parallel_chunk_boundaries() raises:
    var parts = List[Series]()
    for chunk in range(132):
        var values = List[Float64]()
        var valid = List[Bool]()
        for i in range(1024):
            values.append(Float64((chunk + i) % 31) - 15.0)
            valid.append((chunk + i) % 13 != 0)
        parts.append(Series("x", Column[Float64](values^, valid^)))
    var frame = DataFrame([Series._from_chunks(parts^)])
    var predicate = col("x") > lit(Float64(0))
    var mask = frame.select(predicate.alias("mask")).column("mask").bool()
    assert_same(
        frame.filter(predicate), frame.filter(mask), "parallel Float64 filter"
    )


def test_direct_float_filter_aligned_columns_and_nulls() raises:
    var xs = List[Series]()
    var ys = List[Series]()
    var labels = List[Series]()
    for chunk in range(4):
        xs.append(
            Series(
                "x",
                Column[Float64](
                    [-2.0, 1.0, 3.0],
                    [True, chunk != 1, True],
                ),
            )
        )
        ys.append(
            Series(
                "y",
                Column[Int64](
                    [
                        Int64(chunk * 3),
                        Int64(chunk * 3 + 1),
                        Int64(chunk * 3 + 2),
                    ],
                    [True, True, chunk != 2],
                ),
            )
        )
        labels.append(
            Series(
                "label",
                Column[String](
                    ["low", "middle", "high"], [True, chunk != 3, True]
                ),
            )
        )
    var frame = DataFrame(
        [
            Series._from_chunks(xs^),
            Series._from_chunks(ys^),
            Series._from_chunks(labels^),
        ]
    )
    var predicate = col("x") > lit(Float64(0))
    var mask = frame.select(predicate.alias("mask")).column("mask").bool()
    assert_same(frame.filter(predicate), frame.filter(mask), "aligned chunks")
    var none = col("x") > lit(Float64(100))
    var empty_mask = frame.select(none.alias("mask")).column("mask").bool()
    assert_same(frame.filter(none), frame.filter(empty_mask), "empty chunks")


def test_parallel_float_sum_across_many_nullable_chunks() raises:
    # Worker partitions can start and end inside different physical chunks.
    var parts = List[Series]()
    var expected = 0
    for chunk in range(70):
        var values = List[Float64](length=4096, fill=1.0)
        var valid = List[Bool](length=4096, fill=True)
        for i in range(4096):
            valid[i] = (chunk + i) % 7 != 0
            expected += Int(valid[i])
        parts.append(Series("x", Column[Float64](values^, valid^)))
    parts.append(
        Series(
            "x",
            Column[Float64]([1.0, 1.0, 1.0], [False, True, True]),
        )
    )
    expected += 2
    var frame = DataFrame([Series._from_chunks(parts^)])
    assert_true(frame.column("x").n_chunks() == 71)
    assert_equal(
        frame.select(col("x").sum()).item().float64(), Float64(expected)
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
