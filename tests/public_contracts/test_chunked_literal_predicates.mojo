"""Literal predicates retain packed row order across physical string windows."""
# contract-group: expr.chunk_windows
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    BoolColumn,
    StringColumn,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    lit,
    when,
)
from dataframe.expr import STR_STARTS_WITH, STR_ENDS_WITH, STR_CONTAINS, _node
from dataframe.text.kernels import string_op
from dataframe.text.regex_adapter import compile_pattern
from dataframe.parallel import worker_count
from dataframe.string_view import StringViewBuilder
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import ScopedEnvironment


def _master(views: Bool, n: Int) raises -> Series:
    var seed: List[String] = [
        "",
        "ab",
        "abx",
        "xab",
        "xaby",
        "ab-hidden",
        "a.*[",
        "é\x00漢",
        "abab",
        "a",
        "long_external_payload_ab",
        "ab_long_external_payload",
        "miss",
        "hidden",
        "漢ab",
        "ab漢",
    ]
    var values = List[String]()
    var valid = List[Bool]()
    var builder = StringViewBuilder()
    for row in range(n):
        var present = row % 16 != 5 and row % 16 != 13
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


def _source(mode: Int, n: Int, offset: Int) raises -> Series:
    var offsets = _master(False, n + offset + 1)
    var views = _master(True, n + offset + 1)
    var parts = List[Series]()
    var at = offset
    var part = 0
    for size in [1, 1, 1, 1, 1, 1, 1, 1, 0, n - 8]:
        if mode == 1 or (mode == 2 and part % 2 != 0):
            parts.append(views.slice(at, size))
        else:
            parts.append(offsets.slice(at, size))
        at += size
        part += 1
    return Series._from_chunks(parts)


def _expected(op: Int, pattern: String, offset: Int, n: Int) raises -> Series:
    var flags = List[Bool](length=16, fill=False)
    if pattern == "":
        flags = List[Bool](length=16, fill=True)
    elif pattern == "a.*[":
        flags = [
            False,
            False,
            False,
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
        ]
    elif pattern == "é\x00漢":
        flags = [
            False,
            False,
            False,
            False,
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
        ]
    else:
        assert_equal(pattern, "ab")
        if op == STR_STARTS_WITH:
            flags = [
                False,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
            ]
        elif op == STR_ENDS_WITH:
            flags = [
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
            ]
        else:
            assert_equal(op, STR_CONTAINS)
            flags = [
                False,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
            ]
    var values = List[Bool]()
    var valid = List[Bool]()
    for row in range(n):
        var phase = (row + offset) % 16
        var present = phase != 5 and phase != 13
        values.append(present and flags[phase])
        valid.append(present)
    return Series("result", BoolColumn(values, valid))


def _cold(source: Series) raises:
    if source.is_chunked():
        assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))


def _check(
    source: Series, op: Int, pattern: String, offset: Int, cold: Bool = True
) raises:
    var actual = string_op(_node(op, text=pattern, integer=0), source)
    assert_equal(actual.name(), "")
    assert_true(actual.dtype() == DataType.BOOL)
    assert_false(Bool(actual._field_metadata))
    assert_frame(
        DataFrame([actual.renamed("result")]),
        DataFrame([_expected(op, pattern, offset, len(source))]),
    )
    for chunk in actual.chunks():
        ref bits = chunk._data[BoolColumn]
        for row in range(len(bits)):
            if not bits._valid(row):
                assert_false(bits._get(row))
    # Independent flat controls never materialize the timed/tested source.
    var flat = _master(False, len(source) + offset + 1).slice(
        offset, len(source)
    )
    var control = string_op(_node(op, text=pattern, integer=0), flat)
    assert_frame(
        DataFrame([actual.renamed("result")]),
        DataFrame([control.renamed("result")]),
    )
    if cold:
        _cold(source)


def test_literal_predicates_sliced_mixed_chunks_have_complete_literal_answers() raises:
    comptime ops = (STR_STARTS_WITH, STR_ENDS_WITH, STR_CONTAINS)
    for offset in range(8):
        var source = _source(offset % 3, 41, offset)
        comptime for k in range(len(ops)):
            _check(source, ops[k], "ab", offset)
            _check(source, ops[k], "", offset)
    var source = _source(2, 41, 3)
    comptime for k in range(len(ops)):
        _check(source, ops[k], "a.*[", 3)
        _check(source, ops[k], "é\x00漢", 3)
    var frame = DataFrame([source.copy()])
    var expressions: List[Expr] = [
        col("s").str().starts_with("ab").alias("prefix"),
        col("s").str().ends_with("ab").alias("suffix"),
        col("s").str().contains("ab", literal=True).alias("inside"),
    ]
    var expected = DataFrame(
        [
            _expected(STR_STARTS_WITH, "ab", 3, 41).renamed("prefix"),
            _expected(STR_ENDS_WITH, "ab", 3, 41).renamed("suffix"),
            _expected(STR_CONTAINS, "ab", 3, 41).renamed("inside"),
        ]
    )
    assert_frame(frame.select_exprs(expressions, batch_size=3), expected)
    assert_frame(
        frame.lazy()
        .select_exprs(expressions)
        .collect(engine="cpu", batch_size=3),
        expected,
    )
    _cold(source)


def test_literal_predicates_global_packed_bytes_are_disjoint_across_four_workers() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "4")
    comptime n = 4 * 65536 + 7
    assert_equal(worker_count(n), 4)
    comptime ops = (STR_STARTS_WITH, STR_ENDS_WITH, STR_CONTAINS)
    for mode in [0, 2]:
        var source = _source(mode, n, 5)
        comptime for k in range(len(ops)):
            _check(source, ops[k], "ab", 5)
    # Keep the scoped setting alive through every assertion and worker call.
    _ = workers^


def _owned_answers() raises -> List[Series]:
    var source = _source(2, 41, 5)
    var result = List[Series]()
    result.append(string_op(_node(STR_STARTS_WITH, text="ab"), source))
    result.append(string_op(_node(STR_ENDS_WITH, text="ab"), source))
    result.append(string_op(_node(STR_CONTAINS, text="ab", integer=0), source))
    _cold(source)
    return result^


def test_literal_predicate_outputs_outlive_sources_and_ready_cache_stays_ready() raises:
    var outputs = _owned_answers()
    comptime ops = (STR_STARTS_WITH, STR_ENDS_WITH, STR_CONTAINS)
    comptime for k in range(len(ops)):
        assert_frame(
            DataFrame([outputs[k].renamed("result")]),
            DataFrame([_expected(ops[k], "ab", 5, 41)]),
        )
    var source = _source(2, 41, 5)
    var merged = source.rechunk()
    assert_false(merged.is_chunked())
    assert_equal(source._chunked.value()[].cache[].state.load(), Int64(2))
    comptime for k in range(len(ops)):
        _check(source, ops[k], "ab", 5, False)
        assert_equal(source._chunked.value()[].cache[].state.load(), Int64(2))


def test_literal_predicate_dispatch_keeps_compiled_regex_and_fallback_semantics() raises:
    var source = _source(2, 41, 3)
    var compiled = compile_pattern("^never$", False)
    var literal = string_op(
        _node(STR_CONTAINS, text="a.*[", integer=0), source, compiled^
    )
    assert_frame(
        DataFrame([literal.renamed("result")]),
        DataFrame([_expected(STR_CONTAINS, "a.*[", 3, 41)]),
    )
    _cold(source)
    var regex = string_op(_node(STR_CONTAINS, text="^ab", integer=1), source)
    assert_frame(
        DataFrame([regex.renamed("result")]),
        DataFrame([_expected(STR_STARTS_WITH, "ab", 3, 41)]),
    )
    assert_equal(source._chunked.value()[].cache[].state.load(), Int64(2))
    var empty = _master(False, 0)
    with assert_raises():
        _ = string_op(_node(STR_CONTAINS, text="[", integer=1), empty)
    var nulls = string_op(_node(STR_CONTAINS, text="[", integer=2), source)
    assert_frame(
        DataFrame([nulls.renamed("result")]),
        DataFrame(
            [
                Series(
                    "result",
                    BoolColumn(
                        List[Bool](length=41, fill=False),
                        List[Bool](length=41, fill=False),
                    ),
                )
            ]
        ),
    )
    for frame in [DataFrame([source.copy()]), DataFrame([empty.copy()])]:
        with assert_raises():
            _ = frame.select_exprs([col("s").str().contains("[")])
        with assert_raises():
            _ = frame.select_exprs(
                [
                    when(lit(False))
                    .then(col("s").str().contains("["))
                    .otherwise(lit(False))
                ]
            )


def test_literal_predicates_empty_allnull_and_public_binary_boundaries() raises:
    comptime ops = (STR_STARTS_WITH, STR_ENDS_WITH, STR_CONTAINS)
    var empty = _master(False, 0)
    var nulls = Series._from_chunks(
        [
            Series("s", StringColumn(["ab", "hidden"], [False, False])),
            Series("s", StringColumn(["ab"], [False])),
        ]
    )
    comptime for k in range(len(ops)):
        var result = string_op(_node(ops[k], text=""), empty)
        assert_frame(
            DataFrame([result.renamed("result")]),
            DataFrame([Series("result", BoolColumn([]))]),
        )
        var allnull = string_op(_node(ops[k], text=""), nulls)
        assert_frame(
            DataFrame([allnull.renamed("result")]),
            DataFrame(
                [
                    Series(
                        "result",
                        BoolColumn(
                            [False, False, False], [False, False, False]
                        ),
                    )
                ]
            ),
        )
        for row in range(3):
            assert_false(allnull.bool()._get(row))
        _cold(nulls)
    var binary = Series.binary("b", [[UInt8(97), 98]], [True])
    for selected in [binary.copy(), binary.slice(0, 0)]:
        with assert_raises(contains="string expression"):
            _ = DataFrame([selected.copy()]).select_exprs(
                [col("b").str().starts_with("ab")]
            )
