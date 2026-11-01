"""Automaton text kernels consume cold String windows without merging them."""
# contract-group: expr.chunk_windows
from std.testing import assert_true, assert_raises
from dataframe import (
    BoolColumn,
    DataFrame,
    Series,
    StringColumn,
    col,
    lit,
    when,
)
from dataframe.expr import (
    STR_CONTAINS_ANY,
    STR_REPLACE_MANY,
    _node,
    pack_literals,
)
from dataframe.string_view import StringViewBuilder
from dataframe.text.functions import text_function
from tests.support.coverage_compare import assert_frame


def _validity() -> List[Bool]:
    return [True, True, True, True, True, True, True, True, True, False]


def _master(views: Bool, offset: Int) raises -> Series:
    var seed: List[String] = [
        "abcd",
        "é\x00漢",
        "ABC",
        "",
        "ababa",
        "a.*[b]$",
        "prefix_payload_abcdefghijklmnopqrstuvwxyz_suffix",
        "é",
        "É",
        "hidden",
    ]
    var values = List[String](length=offset, fill="before")
    var valid = List[Bool](length=offset, fill=True)
    for i in range(len(seed)):
        values.append(seed[i])
        valid.append(i != 9)
    values.append("after")
    valid.append(True)
    if views:
        var builder = StringViewBuilder()
        for i in range(len(values)):
            if valid[i]:
                builder.append(StringSlice(values[i]))
            else:
                builder.append_null()
        return Series("a", StringColumn(builder^.finish()))
    return Series("a", StringColumn(values, valid))


def _source(mode: Int, offset: Int) raises -> Series:
    var offsets = _master(False, offset)
    var views = _master(True, offset)
    var parts = List[Series]()
    var at = offset
    var part = 0
    for size in [0, 1, 2, 0, 3, 1, 3, 0]:
        if mode == 1 or (mode == 2 and part % 2 != 0):
            parts.append(views.slice(at, size))
        else:
            parts.append(offsets.slice(at, size))
        at += size
        part += 1
    return Series._from_chunks(parts)


def _unmerged(source: Series):
    assert_true(source.is_chunked())
    assert_true(source._chunked.value()[].cache[].state.load() == 0)


def _contains(
    source: Series, patterns: List[String], fold: Bool, whole: Bool
) raises -> Series:
    if whole:
        return (
            DataFrame([source.copy()])
            .select(
                col("a")
                .str()
                .contains_any(patterns, ascii_case_insensitive=fold)
                .alias("result")
            )
            .column("result")
        )
    var node = _node(
        STR_CONTAINS_ANY, text=pack_literals(patterns), integer=Int64(fold)
    )
    return text_function(node, source).renamed("result")


def _replace(
    source: Series,
    patterns: List[String],
    replacements: List[String],
    leftmost: Bool,
    whole: Bool,
) raises -> Series:
    if whole:
        return (
            DataFrame([source.copy()])
            .select(
                col("a")
                .str()
                .replace_many(patterns, replacements, leftmost=leftmost)
                .alias("result")
            )
            .column("result")
        )
    var node = _node(
        STR_REPLACE_MANY,
        text=pack_literals(patterns),
        text2=pack_literals(replacements),
        integer=2 if leftmost else 0,
    )
    return text_function(node, source).renamed("result")


def test_chunked_contains_matches_literal_bytes_ascii_fold_and_nulls() raises:
    for mode in range(3):
        for offset in range(8):
            for variant in range(3):
                for whole in [False, True]:
                    var source = _source(mode, offset)
                    var patterns: List[String] = ["abc", "漢"]
                    var flags: List[Bool] = [
                        True,
                        True,
                        variant == 1,
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                        False,
                    ]
                    if variant == 2:
                        patterns = ["É"]
                        flags = [
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
                    var result = _contains(
                        source, patterns, variant != 0, whole
                    )
                    assert_frame(
                        DataFrame([result^]),
                        DataFrame(
                            [Series("result", BoolColumn(flags, _validity()))]
                        ),
                    )
                    if not whole:
                        _unmerged(source)


def test_chunked_replace_preserves_match_order_and_empty_utf8_advance() raises:
    for mode in range(3):
        for offset in range(8):
            for whole in [False, True]:
                for leftmost in [False, True]:
                    var source = _source(mode, offset)
                    var expected: List[String] = [
                        "a1d",
                        "e\x00漢",
                        "ABC",
                        "",
                        "ababa",
                        "a.*[b]$",
                        "prefix_payload_a1defghijklmnopqrstuvwxyz_suffix",
                        "e",
                        "É",
                        "",
                    ]
                    if leftmost:
                        expected[0] = "2"
                        expected[
                            6
                        ] = "prefix_payload_2efghijklmnopqrstuvwxyz_suffix"
                    var result = _replace(
                        source,
                        ["bc", "abcd", "é"],
                        ["1", "2", "e"],
                        leftmost,
                        whole,
                    )
                    assert_frame(
                        DataFrame([result^]),
                        DataFrame(
                            [
                                Series(
                                    "result",
                                    StringColumn(expected, _validity()),
                                )
                            ]
                        ),
                    )
                    if not whole:
                        _unmerged(source)
                var source = _source(mode, offset).slice(0, 5)
                var result = _replace(source, [""], ["-"], False, whole)
                assert_frame(
                    DataFrame([result^]),
                    DataFrame(
                        [
                            Series(
                                "result",
                                StringColumn(
                                    [
                                        "-a-b-c-d-",
                                        "-é-\x00-漢-",
                                        "-A-B-C-",
                                        "-",
                                        "-a-b-a-b-a-",
                                    ]
                                ),
                            )
                        ]
                    ),
                )
                if not whole:
                    _unmerged(source)


def _cached_answer(contains: Bool) raises -> Series:
    var source = _source(2, 7)
    var values = List[String](length=10, fill="zzz")
    values[1] = "abc"
    var cached = Series("a", StringColumn(values))
    source._chunked.value()[].cache[].merged = cached._data.copy()
    source._chunked.value()[].cache[].state.store(2)
    var result = _contains(
        source, ["abc"], False, False
    ) if contains else _replace(source, ["abc"], ["#"], False, False)
    assert_true(source._chunked.value()[].cache[].state.load() == 2)
    return result^


def test_published_cache_values_and_validity_survive_source_destruction() raises:
    var flags = List[Bool](length=10, fill=False)
    flags[1] = True
    assert_frame(
        DataFrame([_cached_answer(True)]),
        DataFrame([Series("result", BoolColumn(flags))]),
    )
    var strings = List[String](length=10, fill="zzz")
    strings[1] = "#"
    assert_frame(
        DataFrame([_cached_answer(False)]),
        DataFrame([Series("result", StringColumn(strings))]),
    )


def test_automaton_argument_errors_precede_empty_and_inactive_rows() raises:
    var frame = DataFrame([_source(2, 3)])
    for source in [frame.copy(), frame.clear()]:
        with assert_raises():
            _ = source.select(
                when(lit(False))
                .then(col("a").str().contains_any(List[String]()))
                .otherwise(lit(True))
            )
        with assert_raises():
            _ = source.select(
                when(lit(False))
                .then(
                    col("a").str().replace_many(List[String](), List[String]())
                )
                .otherwise(lit("unused"))
            )
        with assert_raises():
            _ = source.select(col("a").str().replace_many(["a"], ["x", "y"]))
