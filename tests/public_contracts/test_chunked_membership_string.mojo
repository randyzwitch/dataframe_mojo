"""String membership keeps literal equality, owned chunks and packed row order."""
# contract-group: expr.membership
# contract-group: expr.chunk_windows
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    BoolColumn,
    StringColumn,
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    lit,
    when,
)
from dataframe.expr import (
    STR_IS_IN,
    STR_STARTS_WITH,
    STR_ENDS_WITH,
    STR_CONTAINS,
    _node,
    null,
    pack_literals,
)
from dataframe.text.kernels import string_op
from dataframe.string_view import StringViewBuilder
from dataframe.parallel import worker_count
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import ScopedEnvironment


def _seed() -> List[String]:
    return [
        "",
        "abc",
        "漢字",
        "é",
        "é",
        "x:y",
        "a,b",
        "a\0b",
        "a.*[",
        "0123456789abcde",
        "0123456789abcdef",
        "0123456789abcdefg",
        "long_external_payload_abcdefghijklmnopqrstuvwxyz",
        "miss",
        "abc",
        "hidden",
    ]


def _master(views: Bool, n: Int) raises -> Series:
    var seed = _seed()
    var values = List[String]()
    var valid = List[Bool]()
    var builder = StringViewBuilder()
    for row in range(n):
        var present = row % 7 != 0
        values.append(seed[row % 16])
        valid.append(present)
        if views:
            if present:
                builder.append(StringSlice(seed[row % 16]))
            else:
                builder.append_null()
    if views:
        return Series("s", StringColumn(builder^.finish()))
    return Series("s", StringColumn(values, valid))


def _source(mode: Int, n: Int, first: Int) raises -> Series:
    var offsets = _master(False, n + first + 1)
    var views = _master(True, n + first + 1)
    var parts = List[Series]()
    parts.append(views.slice(0, 0) if mode == 1 else offsets.slice(0, 0))
    var at = 0
    var part = 0
    for size in [1, 2, 5, 9, n]:
        var count = min(size, n - at)
        if mode == 1 or (mode == 2 and part % 2 != 0):
            parts.append(views.slice(first + at, count))
        else:
            parts.append(offsets.slice(first + at, count))
        at += count
        part += 1
    return Series._from_chunks(parts)


def _wanted(kind: Int) -> List[String]:
    if kind == 0:
        return List[String]()
    if kind == 3:
        return List[String](length=9, fill="abc")
    var values: List[String] = [
        "",
        "abc",
        "漢字",
        "é",
        "x:y",
        "a,b",
        "a\0b",
        "a.*[",
    ]
    if kind == 2:
        values.append("0123456789abcdef")
    return values^


def _expected(kind: Int, first: Int, n: Int) raises -> Series:
    var hits: List[Bool] = [
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    if kind == 0:
        hits = List[Bool](length=16, fill=False)
    elif kind == 2:
        hits[10] = True
    elif kind == 3:
        hits = [
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
        ]
    elif kind == 4:
        hits = [
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    var values = List[Bool]()
    var valid = List[Bool]()
    for row in range(first, first + n):
        var present = row % 7 != 0
        values.append(present and hits[row % 16])
        valid.append(present)
    return Series("result", BoolColumn(values, valid))


def _check(source: Series, kind: Int, first: Int, cold: Bool = True) raises:
    var actual = string_op(
        _node(STR_IS_IN, text=pack_literals(_wanted(kind))), source
    )
    assert_equal(actual.name(), "")
    assert_true(actual.dtype() == DataType.BOOL)
    assert_false(Bool(actual._field_metadata))
    assert_frame(
        DataFrame([actual.renamed("result")]),
        DataFrame([_expected(kind, first, len(source))]),
    )
    for chunk in actual.chunks():
        ref column = chunk._data[BoolColumn]
        for row in range(len(column)):
            if not column._valid(row):
                assert_false(column._get(row))
    if cold and source.is_chunked():
        assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))


def test_string_membership_eight_nine_and_duplicate_tables_keep_complete_bytes_and_tails() raises:
    for mode in range(3):
        for first in [0, 1, 7, 13]:
            for n in [0, 1, 7, 8, 9, 41, 257]:
                var source = _source(mode, n, first)
                for kind in range(4):
                    _check(source, kind, first)


def _owned_answers() raises -> List[Series]:
    var source = _source(2, 41, 13)
    var results = List[Series]()
    for kind in range(4):
        results.append(
            string_op(
                _node(STR_IS_IN, text=pack_literals(_wanted(kind))), source
            )
        )
    assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))
    return results^


def test_string_membership_results_outlive_sources_and_lookup_owners_and_reuse_ready_cache() raises:
    var results = _owned_answers()
    for kind in range(4):
        assert_frame(
            DataFrame([results[kind].renamed("result")]),
            DataFrame([_expected(kind, 13, 41)]),
        )
    var source = _source(2, 41, 13)
    var merged = source.rechunk()
    assert_false(merged.is_chunked())
    for kind in range(4):
        _check(source, kind, 13, False)
        assert_equal(source._chunked.value()[].cache[].state.load(), Int64(2))


def test_string_membership_public_null_empty_categorical_and_inactive_binding_controls() raises:
    var source = _source(2, 41, 13)
    var categorical = source.cast(DataType.CATEGORICAL)
    for field in [source.copy(), categorical.copy()]:
        var frame = DataFrame([field.copy()])
        assert_frame(
            frame.select(col("s").is_in(_wanted(2)).alias("result")),
            DataFrame([_expected(2, 13, 41)]),
        )
        assert_frame(
            frame.select(col("s").is_in(List[String]()).alias("result")),
            DataFrame([_expected(0, 13, 41)]),
        )
        var candidates = List[Expr]()
        for value in _wanted(2):
            candidates.append(lit(value))
        candidates.append(null(DataType.STRING))
        assert_frame(
            frame.select(col("s").is_in(candidates).alias("result")),
            DataFrame([_expected(2, 13, 41)]),
        )
        assert_frame(
            frame.select(
                col("s").is_in([null(DataType.STRING)]).alias("result")
            ),
            DataFrame([_expected(0, 13, 41)]),
        )
        assert_frame(
            frame.select(
                when(lit(False))
                .then(col("s").is_in(_wanted(2)))
                .otherwise(lit(True))
                .alias("result")
            ),
            DataFrame(
                [Series("result", BoolColumn(List[Bool](length=41, fill=True)))]
            ),
        )
    for n in [0, 2]:
        var integers = DataFrame(
            [Series("i", Column[Int64](List[Int64](length=n, fill=1)))]
        )
        with assert_raises(contains="eq requires matching dtypes"):
            _ = integers.select(col("i").is_in(["abc"]))
        with assert_raises(contains="eq requires matching dtypes"):
            _ = integers.select(
                when(lit(False))
                .then(col("i").is_in(["abc"]))
                .otherwise(lit(False))
            )


def test_string_membership_parallel_global_bytes_and_existing_literal_dispatch_are_preserved() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var n = 4 * 65536 + 21
    assert_equal(worker_count(n), 4)
    var source = _source(2, n, 7)
    _check(source, 1, 7)
    _check(source, 2, 7)
    comptime ops = (STR_STARTS_WITH, STR_ENDS_WITH, STR_CONTAINS)
    comptime for k in range(len(ops)):
        var actual = string_op(_node(ops[k], text="é", integer=0), source)
        assert_frame(
            DataFrame([actual.renamed("result")]),
            DataFrame([_expected(4, 7, n)]),
        )
        assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))
    _ = workers^
