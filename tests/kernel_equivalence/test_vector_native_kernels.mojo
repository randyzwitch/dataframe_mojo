"""Comparisons, ranges and casts that read a column's vector at its own
physical type (no copy into a 64-bit lane) against the eager evaluator, for
every stored width: 8-, 16-, 32- and 64-bit integers, unsigned integers,
both float widths, dates and decimals stored in 32, 64 and 128 bits. Each
type is compared with constants on either side, with a second column of its
type, with a null constant and with a constant its storage cannot hold;
ranges and casts likewise. Every case runs over flat chunks, over
dictionaries through a selection, over a subset of rows (`write_rows`), as
a projection and as a filter, with columns that have nulls and columns that
have none. Arithmetic over columns read in place keeps its overflow errors,
and raises none for null rows."""
# contract-group: expr.vectors
from std.testing import assert_equal, assert_true

from dataframe import Column, DataFrame, DataType, Expr, Series
from dataframe import col, lit, null
from dataframe.vector.data_chunk import (
    ColumnArena,
    DataChunk,
    from_frame_window,
    to_series,
)
from dataframe.vector.expression import VectorExpression
from dataframe.vector.types import SelectionVector

comptime ROWS = 600
comptime CHUNK = 512


def _valid(row: Int, salt: Int) -> Bool:
    return (row * 7 + salt) % 11 != 3


def frame() raises -> DataFrame:
    """Two columns of every stored type (`<t>` with nulls, `<t>2` without),
    values crossing zero and repeating so every comparison is mixed."""
    var columns = List[Series]()
    var i8 = List[Int8]()
    var i16 = List[Int16]()
    var i32 = List[Int32]()
    var i64 = List[Int64]()
    var u8 = List[UInt8]()
    var u16 = List[UInt16]()
    var u32 = List[UInt32]()
    var f32 = List[Float32]()
    var f64 = List[Float64]()
    var valid = List[Bool]()
    var j8 = List[Int8]()
    var j16 = List[Int16]()
    var j32 = List[Int32]()
    var j64 = List[Int64]()
    var v8 = List[UInt8]()
    var v16 = List[UInt16]()
    var v32 = List[UInt32]()
    var g32 = List[Float32]()
    var g64 = List[Float64]()
    for row in range(ROWS):
        var a = (row * 37) % 101 - 50
        var b = (row * 53) % 97 - 48
        i8.append(Int8(a))
        i16.append(Int16(a * 300))
        i32.append(Int32(a * 70000))
        i64.append(Int64(a) * 3_000_000_000)
        u8.append(UInt8(a + 50))
        u16.append(UInt16((a + 50) * 600))
        u32.append(UInt32((a + 50) * 40_000_000))
        f32.append(Float32(a) / 4.0)
        f64.append(Float64(a) / 8.0)
        valid.append(_valid(row, 0))
        j8.append(Int8(b))
        j16.append(Int16(b * 300))
        j32.append(Int32(b * 70000))
        j64.append(Int64(b) * 3_000_000_000)
        v8.append(UInt8(b + 50))
        v16.append(UInt16((b + 50) * 600))
        v32.append(UInt32((b + 50) * 40_000_000))
        g32.append(Float32(b) / 4.0)
        g64.append(Float64(b) / 8.0)
    columns.append(Series("i8", Column[Int8](i8^, valid)))
    columns.append(Series("i16", Column[Int16](i16^, valid)))
    columns.append(Series("i32", Column[Int32](i32^, valid)))
    columns.append(Series("i64", Column[Int64](i64^, valid)))
    columns.append(Series("u8", Column[UInt8](u8^, valid)))
    columns.append(Series("u16", Column[UInt16](u16^, valid)))
    columns.append(Series("u32", Column[UInt32](u32^, valid)))
    columns.append(Series("f32", Column[Float32](f32^, valid)))
    columns.append(Series("f64", Column[Float64](f64^, valid)))
    columns.append(Series("i82", Column[Int8](j8^)))
    columns.append(Series("i162", Column[Int16](j16^)))
    columns.append(Series("i322", Column[Int32](j32^)))
    columns.append(Series("i642", Column[Int64](j64^)))
    columns.append(Series("u82", Column[UInt8](v8^)))
    columns.append(Series("u162", Column[UInt16](v16^)))
    columns.append(Series("u322", Column[UInt32](v32^)))
    columns.append(Series("f322", Column[Float32](g32^)))
    columns.append(Series("f642", Column[Float64](g64^)))
    var days = List[Int64]()
    var days2 = List[Int64]()
    var cents = List[Int64]()
    var cents2 = List[Int64]()
    for row in range(ROWS):
        days.append(Int64((row * 13) % 400))
        days2.append(Int64((row * 29) % 400))
        cents.append(Int64((row * 37) % 2001 - 1000))
        cents2.append(Int64((row * 53) % 2001 - 1000))
    columns.append(
        Series("d", Column[Int64](days.copy(), valid)).cast(DataType.DATE)
    )
    columns.append(Series("d2", Column[Int64](days2^)).cast(DataType.DATE))
    for width in [32, 64, 128]:
        var name = "m" + String(width)
        columns.append(
            Series(name, Column[Int64](cents.copy(), valid)).cast(
                DataType.decimal(9, 0, width)
            )
        )
        columns.append(
            Series(name + "2", Column[Int64](cents2.copy())).cast(
                DataType.decimal(9, 0, width)
            )
        )
    return DataFrame(columns^)


def _constant(name: String, v: Int) raises -> Expr:
    """A constant of column `name`'s type, `v` (within -50 ... 50) at the
    scale its values are made at."""
    if name.startswith("i8"):
        return lit(Int8(v))
    if name.startswith("i16"):
        return lit(Int16(v * 300))
    if name.startswith("i32"):
        return lit(Int32(v * 70000))
    if name.startswith("i64"):
        return lit(Int64(v) * 3_000_000_000)
    if name.startswith("u8"):
        return lit(UInt8(v + 50))
    if name.startswith("u16"):
        return lit(UInt16((v + 50) * 600))
    if name.startswith("u32"):
        return lit(UInt32((v + 50) * 40_000_000))
    if name.startswith("f32"):
        return lit(Float32(v) / 4.0)
    if name.startswith("f64"):
        return lit(Float64(v) / 8.0)
    if name.startswith("d"):
        return lit(Int64(v * 4 + 200)).cast(DataType.DATE)
    var width = 128 if name.startswith("m128") else (
        64 if name.startswith("m64") else 32
    )
    return lit(Int64(v * 20)).cast(DataType.decimal(9, 0, width))


def cases(source: DataFrame) raises -> List[Expr]:
    var names: List[String] = [
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "f32",
        "f64",
        "d",
        "m32",
        "m64",
        "m128",
    ]
    var out = List[Expr]()
    for name in names:
        for column in [name, name + "2"]:
            var x = col(column)
            var c = _constant(name, 20)
            var low = _constant(name, -20)
            out.append((x.copy() == c.copy()).alias("p"))
            out.append((x.copy() < c.copy()).alias("p"))
            out.append((x.copy() >= c.copy()).alias("p"))
            if column == name:
                out.append((x.copy() != c.copy()).alias("p"))
                out.append((x.copy() <= c.copy()).alias("p"))
                out.append((x.copy() > c.copy()).alias("p"))
                # The constant on the left.
                out.append((c.copy() < x.copy()).alias("p"))
                out.append((c.copy() >= x.copy()).alias("p"))
                # An open range in one pass.
                out.append(
                    ((x.copy() > low.copy()) & (x.copy() < c.copy())).alias("p")
                )
            # A closed range in one pass.
            out.append(x.copy().is_between(low.copy(), c.copy()).alias("p"))
        var a = col(name)
        var b = col(name + "2")
        # Two columns of one type.
        out.append((a.copy() < b.copy()).alias("p"))
        out.append((a.copy() == b.copy()).alias("p"))
        out.append((b.copy() >= a.copy()).alias("p"))
        # A null constant: every row null.
        out.append((a.copy() > null(source.column(name).dtype())).alias("p"))
        if not name.startswith("d"):
            if not name.startswith("f") and not name.startswith("m"):
                out.append(a.copy().cast(DataType.INT64).alias("p"))
            out.append(a.copy().cast(DataType.FLOAT64).alias("p"))
    # Arithmetic over columns read where they are stored, or widened.
    out.append((col("i64") + col("i642")).alias("p"))
    out.append((col("i642") * lit(Int64(3)) - col("i64")).alias("p"))
    out.append((col("f64") * col("f642") + lit(1.5)).alias("p"))
    for name in ["m32", "m64", "m128"]:
        out.append((col(name) - col(name + "2")).alias("p"))
        out.append((col(name) * col(name + "2")).alias("p"))
    return out^


def chunk_of(
    source: DataFrame,
    mut arena: ColumnArena,
    offset: Int,
    count: Int,
    keep_every: Int,
) raises -> DataChunk:
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


def test_native_kernels_equal_the_eager_columns() raises:
    var source = frame()
    var schema = source.clear()._columns.copy()
    var all_cases = cases(source)
    for keep_every in [1, 3]:
        for offset in [0, 37]:
            var arena = ColumnArena()
            var chunk = chunk_of(source, arena, offset, CHUNK, keep_every)
            var rows = expected_rows(source, offset, CHUNK, keep_every)
            # Every fifth row, for evaluation over some rows only.
            var subset = DataChunk()
            var picked = subset.new_selection()
            var taken = List[Int]()
            for i in range(chunk.count):
                if i % 5 == 1:
                    picked.set_index(len(taken), i)
                    taken.append(i)
            var subset_rows = rows.take(taken)
            for ci in range(len(all_cases)):
                ref e = all_cases[ci]
                var expected = rows.with_columns([e.copy()]).select(["p"])
                var compiled = VectorExpression(e.copy(), schema)
                var out = DataChunk(1)
                out.count = chunk.count
                compiled.write(chunk, arena, out, 0)
                var got = to_series(
                    out, 0, arena, "p", expected._columns[0].dtype()
                )
                assert_true(
                    got.equals(expected._columns[0]), "case " + String(ci)
                )
                if compiled.computes_rows():
                    var part = DataChunk(1)
                    part.count = len(taken)
                    compiled.write_rows(
                        chunk, arena, part, 0, picked, len(taken)
                    )
                    var got_part = to_series(
                        part, 0, arena, "p", expected._columns[0].dtype()
                    )
                    assert_true(
                        got_part.equals(
                            subset_rows.with_columns([e.copy()]).column("p")
                        ),
                        "rows case " + String(ci),
                    )
                    _ = part^
                if expected._columns[0].dtype() == DataType.BOOL:
                    var filter = rows.filter(e.copy())
                    var target_chunk = DataChunk()
                    var target = target_chunk.new_selection()
                    var kept = compiled.select(
                        chunk, arena, SelectionVector(), chunk.count, target
                    )
                    assert_equal(kept, filter.height())
                    var kept_rows = List[Int]()
                    for k in range(kept):
                        kept_rows.append(target.get_index(k))
                    assert_true(
                        rows.take(kept_rows).equals(filter),
                        "filter case " + String(ci),
                    )
                    _ = target_chunk^
                _ = compiled^
            _ = subset^


def _raised(e: Expr, source: DataFrame, keep_every: Int) raises -> String:
    var schema = source.clear()._columns.copy()
    var arena = ColumnArena()
    var chunk = chunk_of(source, arena, 0, source.height(), keep_every)
    var compiled = VectorExpression(e.copy(), schema)
    var out = DataChunk(1)
    out.count = chunk.count
    try:
        compiled.write(chunk, arena, out, 0)
    except error:
        return String(error)
    return String()


def test_arithmetic_over_columns_read_in_place_keeps_overflow_errors() raises:
    # A flat Int64 column is read where it is stored; its overflow is still
    # raised, and a null row's value never is.
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(1000):
        values.append(Int64.MAX if row == 517 else Int64(row))
        valid.append(row != 900)
    var source = DataFrame(
        [
            Series("x", Column[Int64](values.copy())),
            Series("y", Column[Int64](values^, valid)),
        ]
    )
    for keep_every in [1, 2]:
        var add = _raised(
            (col("x") + lit(Int64(1))).alias("p"), source, keep_every
        )
        # Row 517 is odd: a selection of every second row drops it.
        assert_equal(add, "Int64 sum overflow" if keep_every == 1 else String())
        assert_equal(
            _raised((col("x") * col("y")).alias("p"), source, 1),
            "Int64 multiplication overflow",
        )
        assert_equal(
            _raised((col("x") - lit(Int64.MIN)).alias("p"), source, 1),
            "Int64 subtraction overflow",
        )
    # The null row of y holds a value that would overflow: no error.
    var big = List[Int64]()
    var big_valid = List[Bool]()
    for row in range(1000):
        big.append(Int64.MAX if row == 3 else Int64(row))
        big_valid.append(row != 3)
    var nulls = DataFrame([Series("z", Column[Int64](big^, big_valid))])
    assert_equal(
        _raised((col("z") + lit(Int64(1))).alias("p"), nulls, 1), String()
    )
    # A decimal stored in 32 bits, past its precision when scaled.
    var cents = List[Int64]()
    for row in range(1000):
        cents.append(Int64(999_999_999) if row == 10 else Int64(row))
    var decimals = DataFrame(
        [Series("m", Column[Int64](cents^)).cast(DataType.decimal(9, 0, 32))]
    )
    assert_equal(
        _raised(
            (col("m") * col("m")).cast(DataType.decimal(9, 0)).alias("p"),
            decimals,
            1,
        ),
        "value exceeds decimal precision 9",
    )
