"""A reduction's state updated straight from a chunk's vectors equals the
same state updated from the chunk's rows as a column: flat windows and
dictionary slices, nulls, every number width, decimals, Booleans and
strings; and a buffer of input rows filled from vectors holds the rows a
column would."""
# contract-group: reduce.global
from std.collections import Optional
from std.math import nan
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series, StringColumn, col
from dataframe.dtype import DataType
from dataframe.expr import (
    ALL,
    ANY,
    CORR,
    COUNT,
    COV,
    KURTOSIS,
    MAX,
    MEAN,
    MEDIAN,
    MIN,
    MODE,
    NULL_COUNT,
    PRODUCT,
    QUANTILE,
    SKEW,
    STD,
    SUM,
    VAR,
)
from dataframe.storage.boolean import BoolColumn
from dataframe.aggregation.function import _OrderedBuffer, reducer_input
from dataframe.aggregation.reducer import Reducer, reads_vectors
from dataframe.vector.data_chunk import (
    ColumnArena,
    DataChunk,
    from_frame_window,
    storage_physical,
    to_series,
)
from tests.support.coverage_compare import assert_value

comptime _ROWS = 5000


def _valid(i: Int, every: Int) -> Bool:
    return i % every != 3


def _columns() raises -> List[Series]:
    var i8 = List[Int8]()
    var u64 = List[UInt64]()
    var f32 = List[Float32]()
    var f64 = List[Float64]()
    var d64 = List[Int64]()
    var d128 = List[Int128]()
    var small = List[Int16]()
    var near = List[Float64]()
    var flags = List[Bool]()
    var texts = List[String]()
    var v11 = List[Bool]()
    var v13 = List[Bool]()
    var v17 = List[Bool]()
    var v19 = List[Bool]()
    for i in range(_ROWS):
        i8.append(Int8((i * 7) % 200 - 100))
        u64.append(UInt64.MAX - UInt64(i % 97) * 1_000_000_007)
        f32.append(Float32(i % 101) * 0.25 - 3.0)
        f64.append(
            nan[DType.float64]() if i % 29
            == 5 else Float64(i % 113) * 1.5 - 40.0
        )
        d64.append(Int64((i * 31) % 10_000) - 5_000)
        d128.append(Int128(i % 89) * Int128(10) ** 20 + Int128(i))
        small.append(Int16(-1 if i % 997 == 0 else (2 if i % 1301 == 0 else 1)))
        near.append(1.0 + Float64(i % 7 - 3) * 1e-4)
        flags.append(i % 5 < 2)
        texts.append(
            "v" + String(i % 41) if i % 2
            == 0 else "a longer value " + String(i % 23)
        )
        v11.append(_valid(i, 11))
        v13.append(_valid(i, 13))
        v17.append(_valid(i, 17))
        v19.append(_valid(i, 19))
    var result = List[Series]()
    result.append(Series("i8", Column[Int8](i8^, v11)))
    result.append(Series("u64", Column[UInt64](u64^, v13)))
    result.append(Series("f32", Column[Float32](f32^, v17)))
    result.append(Series("f64", Column[Float64](f64^, v19)))
    result.append(
        Series("d64", Column[Int64](d64^, v11)).with_dtype(
            DataType.decimal(12, 2, 64)
        )
    )
    result.append(
        Series("d128", Column[Int128](d128^, v13)).with_dtype(
            DataType.decimal(30, 4)
        )
    )
    result.append(Series("small", Column[Int16](small^, v17)))
    result.append(Series("near", Column[Float64](near^, v19)))
    result.append(Series("flags", BoolColumn(flags, v13)))
    result.append(Series("texts", StringColumn(texts, v11)))
    return result^


def _reducer(op: Int, dtype: DataType) -> Reducer:
    return Reducer(
        op,
        dtype.physical(),
        1,
        0,
        1,
        0.3 if op == QUANTILE else 0.0,
        "linear" if op == QUANTILE else "pearson",
        Optional(dtype),
    )


def _chunks(
    mut arena: ColumnArena, handles: List[Int], dictionary: Bool
) raises -> List[DataChunk]:
    """The rows as flat windows, or as dictionary slices that read every
    third row of each window backwards."""
    var chunks = List[DataChunk]()
    var at = 0
    while at < _ROWS:
        var count = min(2048, _ROWS - at)
        var window = DataChunk()
        from_frame_window(window, arena, handles, at, count)
        if not dictionary:
            chunks.append(window^)
        else:
            var sliced = DataChunk()
            var sel = sliced.new_selection()
            var kept = 0
            var r = count - 1
            while r >= 0:
                sel.set_index(kept, r)
                kept += 1
                r -= 3
            sliced.slice(window, sel, kept)
            chunks.append(sliced^)
        at += count
    return chunks^


def _assert_same(a: Reducer, b: Reducer) raises:
    var left = a.finish()
    var right = b.finish()
    assert_equal(len(left), len(right))
    for r in range(len(right)):
        assert_value(left.get(r), right.get(r))


def test_reducer_vector_update_matches_column_update() raises:
    var columns = _columns()
    var ops: List[Int] = [
        COUNT,
        NULL_COUNT,
        SUM,
        MEAN,
        MIN,
        MAX,
        STD,
        VAR,
        SKEW,
        KURTOSIS,
        MEDIAN,
        QUANTILE,
        PRODUCT,
        MODE,
        ANY,
        ALL,
    ]
    var checked = 0
    for dictionary in [False, True]:
        for c in range(len(columns)):
            var name = columns[c].name()
            var dtype = columns[c].dtype()
            var arena = ColumnArena()
            var handle = arena.add(columns[c])
            var chunks = _chunks(arena, [handle], dictionary)
            for op in ops:
                if not reads_vectors(op, dtype):
                    continue
                # Products only of the columns whose products stay in range.
                # A sum of the UInt64 column leaves 64 bits.
                if op == SUM and name == "u64":
                    continue
                if op == PRODUCT and not (
                    name == "small" or name == "near" or name == "flags"
                ):
                    continue
                var by_column = _reducer(op, dtype)
                var by_vector = _reducer(op, dtype)
                for k in range(len(chunks)):
                    var column = to_series(chunks[k], 0, arena, "", dtype)
                    by_column.update(
                        reducer_input(by_column, column), 0, False, List[Int]()
                    )
                    by_vector.update_vector(
                        chunks[k].unified(0, arena), chunks[k].count, dtype
                    )
                _assert_same(by_vector, by_column)
                checked += 1
            _ = arena^
    assert_true(checked > 100)


def test_pair_vector_update_matches_column_update() raises:
    var columns = _columns()
    var pairs: List[Tuple[Int, Int]] = [(3, 0), (4, 1), (2, 5), (7, 6)]
    for dictionary in [False, True]:
        for pair in pairs:
            var x = columns[pair[0]].copy()
            var y = columns[pair[1]].copy()
            var arena = ColumnArena()
            var handles: List[Int] = [arena.add(x), arena.add(y)]
            var chunks = _chunks(arena, handles, dictionary)
            for method in ["pearson", "spearman", "cov"]:
                var op = COV if method == "cov" else CORR
                var text = "pearson" if method == "cov" else method
                var by_column = Reducer(
                    op, DataType.FLOAT64, 1, 0, 1, 0.0, text, None
                )
                var by_vector = Reducer(
                    op, DataType.FLOAT64, 1, 0, 1, 0.0, text, None
                )
                for k in range(len(chunks)):
                    var left = to_series(chunks[k], 0, arena, "", x.dtype())
                    var right = to_series(chunks[k], 1, arena, "", y.dtype())
                    by_column.update_pair(left, right, 0, False, List[Int]())
                    by_vector.update_pair_vector(
                        chunks[k].unified(0, arena),
                        chunks[k].unified(1, arena),
                        chunks[k].count,
                        x.dtype(),
                        y.dtype(),
                    )
                _assert_same(by_vector, by_column)
            _ = arena^


def test_ordered_buffer_from_vectors_keeps_the_rows_in_order() raises:
    var columns = _columns()
    for dictionary in [False, True]:
        for c in range(len(columns)):
            var dtype = columns[c].dtype()
            var arena = ColumnArena()
            var handle = arena.add(columns[c])
            var chunks = _chunks(arena, [handle], dictionary)
            var storage = storage_physical(columns[c])
            var by_column = _OrderedBuffer()
            var by_vector = _OrderedBuffer()
            var sequence = 0
            for k in range(len(chunks)):
                var n = chunks[k].count
                var column = to_series(chunks[k], 0, arena, "", dtype)
                by_column.add(column, List[Int](length=n, fill=0), sequence, 1)
                # Every other chunk of a fixed-width column from its vector,
                # the rest as columns: either way the rows stay in order.
                if k % 2 == 1 or not by_vector.add_vector(
                    chunks[k].unified(0, arena), n, dtype, storage, sequence
                ):
                    by_vector.add(
                        column, List[Int](length=n, fill=0), sequence, 1
                    )
                sequence += n
            by_column.prepare()
            by_vector.prepare()
            var left = by_vector.values.value().copy()
            var right = by_column.values.value().copy()
            assert_true(left.dtype() == right.dtype())
            assert_equal(len(left), len(right))
            for r in range(len(right)):
                assert_value(left.get(r), right.get(r))
            assert_equal(by_vector.sequences, by_column.sequences)
            assert_equal(by_vector.rows, by_column.rows)
            _ = arena^


def test_global_reductions_by_address_read_vectors() raises:
    """Reductions whose states the global sink holds by address, over
    several chunks, against the same states fed the whole column."""
    var columns = _columns()
    var frame = DataFrame(columns.copy())
    var modes = frame.lazy().select(col("i8").mode().alias("m")).collect()
    var mode = _reducer(MODE, columns[0].dtype())
    mode.update(columns[0], 0, False, List[Int]())
    # The state's result is the modes as one list; select() returns a row
    # per mode.
    var expected = mode.finish().get(0).list()
    assert_equal(modes.height(), len(expected))
    for r in range(len(expected)):
        assert_equal(String(modes.item(r, "m")), String(expected.get(r)))
    var result = (
        frame.lazy()
        .select_exprs(
            [
                col("small").product().alias("product_small"),
                col("d64").quantile(0.3).alias("quantile_d64"),
                col("u64").arg_max().alias("arg_max_u64"),
                col("f64").arg_min().alias("arg_min_f64"),
            ]
        )
        .collect()
    )
    var product = _reducer(PRODUCT, columns[6].dtype())
    product.update(columns[6], 0, False, List[Int]())
    assert_equal(
        result.item(0, "product_small").int64(),
        product.finish().get(0).int64(),
    )
    var quantile = _reducer(QUANTILE, columns[4].dtype())
    quantile.update(columns[4], 0, False, List[Int]())
    assert_equal(
        result.item(0, "quantile_d64").float64(),
        quantile.finish().get(0).float64(),
    )
    # The first largest UInt64 and the first smallest float (NaN only
    # counts when nothing else is valid), by position.
    var best = -1
    for r in range(_ROWS):
        if columns[1].get(r).is_null():
            continue
        if (
            best < 0
            or columns[1].get(r).uint64() > columns[1].get(best).uint64()
        ):
            best = r
    assert_equal(result.item(0, "arg_max_u64").uint32(), UInt32(best))
    best = -1
    for r in range(_ROWS):
        var value = columns[3].get(r)
        if value.is_null() or value.float64() != value.float64():
            continue
        if best < 0 or value.float64() < columns[3].get(best).float64():
            best = r
    assert_equal(result.item(0, "arg_min_f64").uint32(), UInt32(best))
