"""Ordinary comparisons preserve byte order, nulls, ownership and dispatch."""
# contract-group: expr.comparison_chunks
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.memory import ArcPointer
from dataframe import (
    BoolColumn,
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    col,
    lit,
    when,
)
from dataframe.expr import EQ, NE, LT, LE, GT, GE, null
from dataframe.dtype import CategoricalDictionary
from dataframe.storage.buffer import _Buffer
from dataframe.storage.string import StringCodes
from dataframe.kernels.binary import binary
from dataframe.string_view import StringViewBuilder
from dataframe.parallel import worker_count
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import ScopedEnvironment


def _labels() -> List[String]:
    # Explicit UTF8 byte ordering; duplicate abcd has the same independent rank.
    return [
        "",
        "a",
        "a\0",
        "a\0b",
        "abcd",
        "abcd",
        "abcdefghijk",
        "abcdefghijkl",
        "abcdefghijklm",
        "abcdefghijklmnop",
        "abcdefghijklmnopq",
        "b",
        "é",
        "é",
        "中",
        "🙂",
    ]


def _rank(index: Int) -> Int:
    return index if index < 5 else index - 1


def _master(
    views: Bool, n: Int, phase: Int, dense: Bool, coded: Bool
) raises -> Series:
    var labels = _labels()
    var values = List[String]()
    var valid = List[Bool]()
    var codes = List[UInt32]()
    var builder = StringViewBuilder()
    for row in range(n):
        var value = labels[(row + phase) % 16]
        var present = dense or row % (5 if phase != 0 else 7) != 0
        values.append(value)
        valid.append(present)
        codes.append(UInt32((row + phase) % 16))
        if views:
            if present:
                builder.append(StringSlice(value))
            else:
                builder.append_null()
    var column = StringColumn(values) if dense else StringColumn(values, valid)
    if views:
        column = StringColumn(builder^.finish())
    if coded:
        var dictionary = CategoricalDictionary()
        for value in labels:
            dictionary.append(value)
        column._codes = ArcPointer(
            StringCodes(_Buffer(codes^), DataType.categorical(dictionary^))
        )
    return Series("a" if phase == 0 else "b", column^)


def _source(
    mode: Int,
    n: Int,
    first: Int,
    phase: Int,
    dense: Bool = False,
    coded: Bool = False,
) raises -> Series:
    var offsets = _master(False, n + first + 1, phase, dense, coded)
    var views = _master(True, n + first + 1, phase, dense, coded)
    var parts = List[Series]()
    parts.append(offsets.slice(0, 0))
    var at = 0
    var part = 0
    var sizes: List[Int] = [3, 7, 1, 11, n] if phase == 0 else [1, 2, 5, 9, n]
    for size in sizes:
        var count = min(size, n - at)
        if mode == 1 or (mode == 2 and part % 2 != 0):
            parts.append(views.slice(first + at, count))
        else:
            parts.append(offsets.slice(first + at, count))
        at += count
        part += 1
    return Series._from_chunks(parts)


def _expected[
    op: Int
](first: Int, n: Int, shape: Int, dense: Bool) raises -> Series:
    var values = List[Bool]()
    var valid = List[Bool]()
    for row in range(first, first + n):
        var present = shape != 3 and shape != 4
        if not dense:
            if shape != 1:
                present = present and row % 7 != 0
            if shape != 2:
                present = present and row % 5 != 0
        var a = 4 if shape == 1 else _rank(row % 16)
        var b = 4 if shape == 2 else _rank((row + 1) % 16)
        var hit: Bool
        comptime if op == EQ:
            hit = a == b
        elif op == NE:
            hit = a != b
        elif op == LT:
            hit = a < b
        elif op == LE:
            hit = a <= b
        elif op == GT:
            hit = a > b
        else:
            hit = a >= b
        values.append(present and hit)
        valid.append(present)
    return Series("result", BoolColumn(values, valid))


def _check[
    op: Int
](
    mode: Int,
    first: Int,
    n: Int,
    shape: Int,
    dense: Bool = False,
    mask: List[Bool] = List[Bool](),
) raises:
    var a = _source(mode, n, first, 0, dense)
    var b = _source(mode, n, first, 1, dense)
    if shape == 1:
        a = Series("a", StringColumn(["abcd"]))
    elif shape == 2:
        b = Series("b", StringColumn(["abcd"]))
    elif shape == 3:
        a = Series.full_null("a", DataType.STRING, 1)
    elif shape == 4:
        b = Series.full_null("b", DataType.STRING, 1)
    var actual = binary[op](a, b, mask)
    assert_equal(actual.name(), "")
    assert_true(actual.dtype() == DataType.BOOL)
    assert_false(Bool(actual._field_metadata))
    for chunk in actual.chunks():
        ref column = chunk._data[BoolColumn]
        for row in range(len(column)):
            if not column._valid(row):
                assert_false(column._get(row))
    assert_frame(
        DataFrame([actual.renamed("result")]),
        DataFrame([_expected[op](first, n, shape, dense)]),
    )
    for source in [a.copy(), b.copy()]:
        if source.is_chunked():
            assert_equal(
                source._chunked.value()[].cache[].state.load(), Int64(0)
            )


def test_string_comparisons_all_operators_scalar_directions_views_and_bitless_chunks() raises:
    comptime ops = (EQ, NE, LT, LE, GT, GE)
    comptime for k in range(len(ops)):
        for mode in range(3):
            for shape in range(3):
                _check[ops[k]](mode, 13, 41, shape)
                _check[ops[k]](mode, 7, 9, shape, True)
        # Direct observation masks do not change String comparison results.
        _check[ops[k]](2, 1, 41, 0, False, List[Bool](length=41, fill=False))
        _check[ops[k]](2, 1, 41, 0, False, [True, False, True])


def test_string_comparison_null_scalars_empty_lengths_and_error_priority() raises:
    comptime ops = (EQ, NE, LT, LE, GT, GE)
    comptime for k in range(len(ops)):
        for shape in range(5):
            _check[ops[k]](2, 0, 0, shape)
        _check[ops[k]](2, 13, 41, 3)
        _check[ops[k]](2, 13, 41, 4)
        var empty = _source(2, 0, 0, 0)
        var two = _source(2, 2, 1, 1)
        with assert_raises(contains="Incompatible expression lengths"):
            _ = binary[ops[k]](empty, two)
        with assert_raises(contains="Incompatible expression lengths"):
            _ = binary[ops[k]](two, empty)
        with assert_raises(contains="Incompatible expression lengths"):
            _ = binary[ops[k]](_source(2, 17, 13, 0), _source(2, 15, 13, 1))


def _owned_results() raises -> List[Series]:
    var results = List[Series]()
    var a = _source(2, 41, 13, 0)
    var b = _source(1, 41, 13, 1)
    comptime ops = (EQ, NE, LT, LE, GT, GE)
    comptime for k in range(len(ops)):
        results.append(binary[ops[k]](a, b))
    return results^


def test_string_comparison_owned_results_and_each_published_cache_are_preserved() raises:
    var results = _owned_results()
    comptime ops = (EQ, NE, LT, LE, GT, GE)
    comptime for k in range(len(ops)):
        assert_frame(
            DataFrame([results[k].renamed("result")]),
            DataFrame([_expected[ops[k]](13, 41, 0, False)]),
        )
        for ready in [1, 2, 3]:
            var a = _source(2, 41, 13, 0)
            var b = _source(1, 41, 13, 1)
            if ready & 1 != 0:
                var flat = a.rechunk()
                _ = flat^
            if ready & 2 != 0:
                var flat = b.rechunk()
                _ = flat^
            var actual = binary[ops[k]](a, b)
            assert_frame(
                DataFrame([actual.renamed("result")]),
                DataFrame([_expected[ops[k]](13, 41, 0, False)]),
            )
            assert_equal(
                a._chunked.value()[].cache[].state.load(),
                Int64(2 if ready & 1 != 0 else 0),
            )
            assert_equal(
                b._chunked.value()[].cache[].state.load(),
                Int64(2 if ready & 2 != 0 else 0),
            )


def _forged_string_bool(name: String, values: List[Bool]) -> Series:
    var source = Series(name, BoolColumn(values))
    return Series(name, source._data.copy(), DataType.STRING)


def test_string_comparison_public_dictionary_and_physical_storage_controls() raises:
    comptime ops = (EQ, NE, LT, LE, GT, GE)
    comptime for k in range(len(ops)):
        for categorical in [False, True]:
            var a = _source(2, 257, 13, 0, False, True)
            var b = _source(1, 257, 13, 1, False, True)
            if categorical:
                a = a.cast(DataType.CATEGORICAL)
                b = b.cast(DataType.CATEGORICAL)
            var frame = DataFrame([a.renamed("a"), b.renamed("b")])
            assert_frame(
                frame.select(
                    col("a")._binary(col("b"), ops[k]).alias("result")
                ),
                DataFrame([_expected[ops[k]](13, 257, 0, False)]),
            )
            assert_frame(
                frame.select(
                    col("a")._binary(lit("abcd"), ops[k]).alias("result")
                ),
                DataFrame([_expected[ops[k]](13, 257, 2, False)]),
            )
    # Logical tags do not authorize interpreting actual Bool storage as String.
    var a = Series._from_chunks(
        [
            _forged_string_bool("a", [False, True]),
            _forged_string_bool("a", [False, True]),
        ]
    )
    var b = Series._from_chunks(
        [
            _forged_string_bool("b", [False]),
            _forged_string_bool("b", [False, True, True]),
        ]
    )
    var actual = binary[LT](a, b)
    assert_frame(
        DataFrame([actual.renamed("result")]),
        DataFrame([Series("result", BoolColumn([False, False, True, False]))]),
    )
    assert_equal(a._chunked.value()[].cache[].state.load(), Int64(2))
    assert_equal(b._chunked.value()[].cache[].state.load(), Int64(2))
    for n in [0, 2]:
        var frame = DataFrame(
            [Series("a", StringColumn(List[String](length=n, fill="a")))]
        )
        with assert_raises(contains="requires matching dtypes"):
            _ = frame.select(
                when(lit(False))
                .then(col("a") < lit(Int64(1)))
                .otherwise(lit(False))
            )


def test_string_comparison_parallel_workers_own_complete_global_bitmap_bytes() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var n = 4 * 65536 + 21
    assert_equal(worker_count(n), 4)
    _check[LT](2, 7, n, 0)
    _check[NE](2, 13, n, 2)
    _ = workers^
