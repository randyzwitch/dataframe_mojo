"""Expressions over data chunks against the eager evaluator: comparisons of
integers, floats, decimals, dates and strings, Kleene AND / OR, NOT, null
tests, IN lists, fill_null, abs, round, date parts, slices, string
predicates and CASE WHEN, arithmetic and casts in lanes, and nodes left to
the at-rest evaluator, over flat chunks and over dictionaries through a
selection, with nulls. A filter's selection and a projection's column equal
filtering and projecting the frame eagerly."""
# contract-group: expr.vectors
from std.testing import assert_equal, assert_true

from dataframe import Column, DataFrame, DataType, Expr, Series, StringColumn
from dataframe import col, concat_str, lit, when
from dataframe.vector.data_chunk import (
    ColumnArena,
    DataChunk,
    from_frame_window,
    to_series,
)
from dataframe.vector.expression import VectorExpression
from dataframe.vector.types import SelectionVector
from tests.support.coverage_compare import assert_frame

comptime ROWS = 5000


def frame() raises -> DataFrame:
    var a = List[Int64]()
    var a_valid = List[Bool]()
    var b = List[Int32]()
    var f = List[Float64]()
    var day = List[Int64]()
    var s = List[String]()
    var s_valid = List[Bool]()
    var price = List[String]()
    for row in range(ROWS):
        a.append(Int64((row * 37) % 101 - 50))
        a_valid.append(row % 13 != 4)
        b.append(Int32(row % 17))
        f.append(Float64(row % 211) / 4.0 - 10.0)
        day.append(Int64(row % 400))
        s.append(
            "k"
            + String(row % 9) if row % 3 else "a longer key "
            + String(row % 5)
        )
        s_valid.append(row % 11 != 6)
        price.append(String(row % 997) + "." + String(row % 89 + 10))
    return DataFrame(
        [
            Series("a", Column[Int64](a^, a_valid)),
            Series("b", Column[Int32](b^)),
            Series("f", Column[Float64](f^)),
            Series("day", Column[Int64](day^)).cast(DataType.DATE),
            Series("s", StringColumn(s, s_valid)),
            Series("price", Column[String](price^)).cast(
                DataType.decimal(7, 2)
            ),
        ]
    )


def predicates() raises -> List[Expr]:
    return [
        col("a") > lit(Int64(3)),
        (col("a") <= col("b").cast(DataType.INT64)) | col("a").is_null(),
        (col("f") >= lit(0.5)) & (col("s") != lit("k4")),
        (col("s") == lit("a longer key 2")) | (col("b") == lit(Int32(3))),
        ~(col("s") < lit("k5")),
        col("price") > lit("500.00").cast(DataType.decimal(7, 2)),
        col("day") < lit("1970-03-01").cast(DataType.DATE),
        col("s").str().starts_with("k1") | col("a").is_not_null(),
        col("s").is_null() & (col("a") > lit(Int64(0))),
        col("s").is_in(["k1", "a longer key 3", "absent"]),
        (lit(Int64(-10)) <= col("a")) & (col("a") < lit(Int64(10))),
        col("f").is_between(lit(-2.0), lit(3.0), closed="none")
        | col("a").is_between(lit(Int64(20)), lit(Int64(30))),
        (
            (col("price") >= lit("100.00").cast(DataType.decimal(7, 2)))
            & (col("price") <= lit("300.00").cast(DataType.decimal(7, 2)))
        )
        | (lit(5.0) < col("f"))
        | (col("b") == lit(Int32(2))),
        (col("b") > lit(Int32(3)))
        & ((col("a") != lit(Int64(7))) | (col("f") < lit(0.0))),
        col("b").is_in([lit(Int32(1)), lit(Int32(4)), lit(Int32(16))])
        | col("a").is_in([lit(Int64(-50)), lit(Int64(7))]),
        col("s").str().like("%key%2")
        | col("s").str().like("k_")
        | col("s").str().like("a%"),
        col("s").str().slice(2, 6) == lit("onger "),
        when(col("b") > lit(Int32(8))).then(col("s")).otherwise(lit("k3"))
        == lit("k3"),
        col("s").str().starts_with("a long")
        | col("s").str().ends_with("4")
        | col("s").str().contains("ey 1", literal=True),
        when(col("a") > lit(Int64(0)))
        .then(col("f") > lit(1.0))
        .otherwise(col("s") == lit("k2")),
        (col("a") > lit(Int64(5))).fill_null(lit(True)),
        col("a").abs() > lit(Int64(40)),
    ]


def projections() raises -> List[Expr]:
    return [
        (col("a") * lit(Int64(3)) - col("b").cast(DataType.INT64)).alias("p"),
        (col("f") * col("f") + lit(1.0)).alias("q"),
        (col("price") * col("price")).alias("r"),
        (col("price") + lit("0.5").cast(DataType.decimal(3, 1))).alias("t"),
        col("price").cast(DataType.FLOAT64).alias("u"),
        # Booleans cast to 1 / 0 in lanes, a null staying null.
        (col("a") > col("b").cast(DataType.INT64))
        .cast(DataType.INT64)
        .alias("bi"),
        (col("a").is_not_null() & (col("f") < lit(0.0)))
        .cast(DataType.FLOAT64)
        .alias("bf"),
        (col("a") > col("b").cast(DataType.INT64)).alias("v"),
        col("s").str().len_bytes().alias("w"),
        col("s").alias("x"),
        when(col("a") > lit(Int64(10)))
        .then(col("a") * lit(Int64(2)))
        .when(col("b") == lit(Int32(3)))
        .then(lit(Int64(-1)))
        .otherwise(col("a"))
        .alias("z"),
        when(col("f") < lit(0.0)).then(col("price")).alias("zz"),
        when(col("a") > lit(Int64(20)))
        .then(col("s"))
        .when(col("b") == lit(Int32(3)))
        .then(lit("three"))
        .otherwise(col("s").str().slice(1, 1))
        .alias("ws"),
        concat_str([col("s"), lit("-"), col("s").str().slice(0, 2)], "/").alias(
            "ca"
        ),
        concat_str([col("s"), col("s")], "", ignore_nulls=True).alias("cb"),
        lit(Int64(7)).alias("y"),
        col("a").fill_null(lit(Int64(5))).abs().alias("fa"),
        col("a").fill_null(col("b").cast(DataType.INT64)).alias("fb"),
        col("f").abs().alias("fc"),
        col("b").abs().alias("fd"),
        (lit("100.00").cast(DataType.decimal(7, 2)) - col("price"))
        .abs()
        .alias("fe"),
        (col("f") / lit(3.0)).round(2).alias("ra"),
        (col("f") / lit(3.0)).floor().alias("rb"),
        col("f").ceil().alias("rc"),
        col("day").dt().year().alias("da"),
        (
            col("day").dt().month() * lit(Int64(100)) + col("day").dt().day()
        ).alias("db"),
        col("s").str().slice(2, 5).alias("sa"),
        col("s").str().slice(-3).alias("sb"),
        col("s").str().slice(0, 1).alias("sc"),
    ]


def chunk_of(
    source: DataFrame,
    mut arena: ColumnArena,
    offset: Int,
    count: Int,
    keep_every: Int,
) raises -> DataChunk:
    """Rows [offset, offset + count) as a chunk; with `keep_every` above 1,
    a chunk of dictionaries through every `keep_every`th row."""
    var handles = List[Int]()
    for c in source._columns:
        handles.append(arena.add(c))
    var flat = DataChunk()
    from_frame_window(flat, arena, handles, offset, count)
    if keep_every <= 1:
        return flat^
    var sliced = DataChunk()
    var sel = sliced.new_selection()
    var kept = 0
    for i in range(count):
        if i % keep_every == 0:
            sel.set_index(kept, i)
            kept += 1
    sliced.slice(flat, sel, kept)
    _ = flat^
    return sliced^


def expected_rows(
    source: DataFrame, offset: Int, count: Int, keep_every: Int
) raises -> DataFrame:
    var window = source.slice(offset, count)
    if keep_every <= 1:
        return window^
    var rows = List[Int]()
    for i in range(count):
        if i % keep_every == 0:
            rows.append(i)
    return window.take(rows)


def test_filters_select_what_the_eager_filter_keeps() raises:
    var source = frame()
    var schema = source.clear()._columns.copy()
    for keep_every in [1, 3]:
        for offset in [0, 61, 2900]:
            var arena = ColumnArena()
            var chunk = chunk_of(source, arena, offset, 2048, keep_every)
            var rows = expected_rows(source, offset, 2048, keep_every)
            for p in predicates():
                var compiled = VectorExpression(p, schema)
                var out = DataChunk()
                var target = out.new_selection()
                var kept = compiled.select(
                    chunk, arena, SelectionVector(), chunk.count, target
                )
                var expected = rows.filter(p)
                assert_equal(kept, expected.height())
                var picked = List[Int]()
                for k in range(kept):
                    picked.append(target.get_index(k))
                assert_frame(rows.take(picked), expected)
                # The selection's entries are `out`'s.
                _ = out^


def test_projections_equal_the_eager_columns() raises:
    var source = frame()
    var schema = source.clear()._columns.copy()
    for keep_every in [1, 4]:
        for offset in [0, 777]:
            var arena = ColumnArena()
            var chunk = chunk_of(source, arena, offset, 2048, keep_every)
            var rows = expected_rows(source, offset, 2048, keep_every)
            for e in projections():
                var compiled = VectorExpression(e, schema)
                var out = DataChunk(1)
                out.count = chunk.count
                compiled.write(chunk, arena, out, 0)
                # Every row's value, a constant's broadcast (with_columns).
                var expected = rows.with_columns([e.copy()]).select([e._name])
                var got = to_series(
                    out,
                    0,
                    arena,
                    expected._columns[0].name(),
                    expected._columns[0].dtype(),
                )
                assert_frame(DataFrame([got^]), expected)
                # A constant's value is the compiled expression's.
                _ = compiled^


def test_a_dropped_rows_error_is_not_raised() raises:
    # A decimal product past its precision on rows the selection drops.
    var values = List[Int64]()
    for row in range(2048):
        values.append(Int64(999_999_999) if row % 2 else Int64(row % 7))
    var source = DataFrame(
        [Series("big", Column[Int64](values^)).cast(DataType.decimal(9, 0, 32))]
    )
    var schema = source.clear()._columns.copy()
    var arena = ColumnArena()
    var chunk = chunk_of(source, arena, 0, 2048, 2)
    var compiled = VectorExpression(
        (col("big") * col("big") * col("big")).alias("p"), schema
    )
    var out = DataChunk(1)
    out.count = chunk.count
    compiled.write(chunk, arena, out, 0)
    var got = to_series(out, 0, arena, "p", compiled.root_dtype())
    assert_equal(len(got), 1024)
    assert_true(got.null_count() == 0)


def test_abs_of_the_minimum_raises_as_the_column_kernel() raises:
    var values = List[Int64]()
    for row in range(100):
        values.append(Int64.MIN if row == 57 else Int64(row - 50))
    var source = DataFrame([Series("m", Column[Int64](values^))])
    var schema = source.clear()._columns.copy()
    var arena = ColumnArena()
    var chunk = chunk_of(source, arena, 0, 100, 1)
    var compiled = VectorExpression(col("m").abs().alias("p"), schema)
    var out = DataChunk(1)
    out.count = chunk.count
    var message = String()
    try:
        compiled.write(chunk, arena, out, 0)
    except e:
        message = String(e)
    assert_equal(message, "Int64 abs overflow")


def test_adaptive_conjunctions_keep_the_eager_rows() raises:
    # Nested AND / OR nodes reorder their children while one compiled
    # expression runs over many chunks (AdaptiveFilter: 5 warmup calls,
    # then a trial swap every 20 observed for 10): every chunk's rows stay
    # the eager filter's, in row order.
    var source = frame()
    var schema = source.clear()._columns.copy()
    var nested: List[Expr] = [
        (col("b") > lit(Int32(3)))
        & ((col("a") != lit(Int64(7))) | (col("f") < lit(0.0)))
        & (col("s") != lit("k4")),
        (
            (col("a") > lit(Int64(10)))
            & (col("b") < lit(Int32(12)))
            & col("s").is_not_null()
        )
        | (col("f") < lit(-5.0))
        | col("s").str().starts_with("a long"),
    ]
    for p in nested:
        var compiled = VectorExpression(p, schema)
        for call in range(120):
            var offset = (call * 37) % 2900
            var keep_every = 1 + call % 3
            var arena = ColumnArena()
            var chunk = chunk_of(source, arena, offset, 2048, keep_every)
            var rows = expected_rows(source, offset, 2048, keep_every)
            var out = DataChunk()
            var target = out.new_selection()
            var kept = compiled.select(
                chunk, arena, SelectionVector(), chunk.count, target
            )
            var expected = rows.filter(p)
            assert_equal(kept, expected.height())
            var picked = List[Int]()
            for k in range(kept):
                picked.append(target.get_index(k))
            assert_frame(rows.take(picked), expected)
            _ = out^


def test_literal_search_keeps_utf8_nuls_empty_patterns_and_like_boundaries() raises:
    var source = DataFrame(
        [
            Series(
                "s",
                StringColumn(
                    [
                        "",
                        "é🙂tail",
                        "a\x00b",
                        "ababa",
                        "prefix-middle-tail",
                        "aééz",
                        "tail",
                        "ignored",
                    ],
                    [True, True, True, True, True, True, True, False],
                ),
            )
        ]
    )
    var patterns = List[Expr]()
    for needle in ["", "é", "🙂", "\x00", "aba", "absent"]:
        patterns.append(col("s").str().contains(needle, literal=True))
    for pattern in [
        "%",
        "%é%",
        "%🙂%tail",
        "a%ba%ba",
        "%tail%tail",
        "prefix%middle%tail",
        "a%é%é%z",
        "%\x00%b",
    ]:
        patterns.append(col("s").str().like(pattern))
    for keep_every in [1, 2]:
        var arena = ColumnArena()
        var chunk = chunk_of(source, arena, 0, source.height(), keep_every)
        var rows = expected_rows(source, 0, source.height(), keep_every)
        for p in patterns:
            var compiled = VectorExpression(p, source.clear()._columns)
            var out = DataChunk()
            var target = out.new_selection()
            var kept = compiled.select(
                chunk, arena, SelectionVector(), chunk.count, target
            )
            var picked = List[Int]()
            for k in range(kept):
                picked.append(target.get_index(k))
            assert_frame(rows.take(picked), rows.filter(p))
            _ = out^
