"""Regular expressions in the `.str()` namespace (#219): contains,
count_matches, find, extract, extract_all, replace, replace_all and
split(literal=False).

Expected values for the same input are written out as literals. Each
function runs over offsets, a sliced window, chunks and string views, with
nulls, eagerly and through LazyFrame, in select, filter and group_by().agg.
The regex engine is also checked against a larger corpus.
"""
# contract-group: expr.regex
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    Series,
    StringColumn,
    Expr,
    col,
    columns,
    DataType,
    StructColumn,
    ArrowArray,
    ArrowSchema,
    export_arrow_series,
    import_arrow_series,
    all_horizontal,
)
from dataframe.string_view import StringViewBuilder
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.coverage_compare import assert_frame


def _values() -> List[String]:
    return [
        "id-12 and id-7",
        "",
        "Straße 9",
        "no digits",
        "",
        "a.b a.b",
    ]


def _valid() -> List[Bool]:
    return [True, False, True, True, True, True]


def _source(layout: Int, padding: String = "") raises -> DataFrame:
    var values = _values()
    for i in range(len(values)):
        values[i] = padding + values[i]
    var valid = _valid()
    var groups: List[Int64] = [1, 1, 2, 2, 2, 3]
    if layout == 3:
        var builder = StringViewBuilder(len(values))
        for i in range(len(values)):
            if valid[i]:
                builder.append(values[i])
            else:
                builder.append_null()
        return DataFrame(
            [
                Series("g", Column[Int64](groups.copy())),
                Series("s", StringColumn(builder^.finish())),
            ]
        )
    var frame = DataFrame(
        [
            Series("g", Column[Int64](groups.copy())),
            Series("s", Column[String](values.copy(), valid.copy())),
        ]
    )
    if layout == 1:
        return offset_layout(frame)
    if layout == 2:
        return chunk_layout(frame)
    return frame^


def _strings(
    name: String, values: List[String], valid: List[Bool]
) raises -> Series:
    return Series(name, Column[String](values.copy(), valid.copy()))


def _expressions() -> List[Expr]:
    var s = col("s").str()
    return [
        s.contains("\\d+").alias("c"),
        s.contains("a.b", literal=True).alias("cl"),
        s.count_matches("\\d").alias("n"),
        s.find("\\d").alias("f"),
        s.extract("([a-z]+)-(\\d+)", 2).alias("x"),
        s.extract_all("\\d+").list().join("|").alias("xa"),
        s.extract_all("\\d+").list().len().alias("xn"),
        s.replace("(\\w+)-(\\d+)", "$2:$1").alias("r"),
        s.replace_all("\\d", "#").alias("ra"),
        s.replace_all("a.b", "X", literal=True).alias("rl"),
        s.split("\\s+", literal=False).list().join("|").alias("sp"),
        s.contains("(", strict=False).alias("bad"),
    ]


def _expected() raises -> DataFrame:
    var valid = _valid()
    var none: List[Bool] = List[Bool](length=6, fill=False)
    var found: List[Bool] = [True, False, True, False, False, False]
    var extracted: List[Bool] = [True, False, False, False, False, False]
    var r: List[String] = [
        "12:id and id-7",
        "",
        "Straße 9",
        "no digits",
        "",
        "a.b a.b",
    ]
    var ra: List[String] = [
        "id-## and id-#",
        "",
        "Straße #",
        "no digits",
        "",
        "a.b a.b",
    ]
    var rl: List[String] = [
        "id-12 and id-7",
        "",
        "Straße 9",
        "no digits",
        "",
        "X X",
    ]
    var sp: List[String] = [
        "id-12|and|id-7",
        "",
        "Straße|9",
        "no|digits",
        "",
        "a.b|a.b",
    ]
    return DataFrame(
        [
            Series(
                "c",
                Column[Bool]([True, False, True, False, False, False], valid),
            ),
            Series(
                "cl",
                Column[Bool]([False, False, False, False, False, True], valid),
            ),
            Series("n", Column[UInt32]([3, 0, 1, 0, 0, 0], valid)),
            Series("f", Column[UInt32]([3, 0, 8, 0, 0, 0], found)),
            _strings("x", ["12", "", "", "", "", ""], extracted),
            _strings("xa", ["12|7", "", "9", "", "", ""], valid),
            Series("xn", Column[Int64]([2, 0, 1, 0, 0, 0], valid)),
            _strings("r", r, valid),
            _strings("ra", ra, valid),
            _strings("rl", rl, valid),
            _strings("sp", sp, valid),
            Series("bad", Column[Bool](List[Bool](length=6, fill=False), none)),
        ]
    )


def test_regex_functions_match_polars_eager_and_lazy() raises:
    var expected = _expected()
    for layout in range(4):
        var source = _source(layout)
        for batch_size in [1, 2, 8192]:
            assert_frame(
                source.select_exprs(_expressions(), batch_size=batch_size),
                expected,
            )
            assert_frame(
                source.lazy()
                .select_exprs(_expressions())
                .collect(batch_size=batch_size),
                expected,
            )
        assert_frame(
            source.clear().select_exprs(_expressions()), expected.clear()
        )


def test_regex_filter_with_columns_and_group_by() raises:
    for layout in range(4):
        var source = _source(layout)
        var lazy = source.lazy()
        var kept = lazy.filter(col("s").str().contains("^\\w+ \\d$")).collect()
        assert_equal(kept.height(), 1)
        assert_equal(kept.item(0, "s").string(), "Straße 9")
        var eager = source.filter(col("s").str().contains("^\\w+ \\d$"))
        assert_equal(eager.height(), 1)
        var added = lazy.with_columns(
            [col("s").str().extract("(\\d+)$", 1).alias("tail")]
        ).collect()
        assert_equal(added.item(0, "tail").string(), "7")
        assert_true(added.item(3, "tail").is_null())
        var grouped = (
            lazy.group_by("g")
            .agg(
                [
                    col("s").str().count_matches("\\d").sum().alias("d"),
                    col("s")
                    .str()
                    .contains("(?i)STRASSE|straße")
                    .any()
                    .alias("any"),
                ]
            )
            .sort("g")
            .collect()
        )
        assert_equal(grouped.height(), 3)
        assert_equal(grouped.item(0, "d").uint32(), 3)
        assert_equal(grouped.item(1, "d").uint32(), 1)
        assert_equal(grouped.item(2, "d").uint32(), 0)
        assert_true(not grouped.item(0, "any").bool())
        assert_true(grouped.item(1, "any").bool())


def test_regex_literal_rules_follow_polars() raises:
    var frame = DataFrame(
        [Series("s", Column[String](["a1b22c", "price $5", "abc"]))]
    )
    var s = col("s").str()
    var out = frame.select_exprs(
        [
            # No ASCII punctuation: replaced as text, `$1` kept as is.
            s.replace("b", "$1").alias("plain"),
            # With punctuation the value is a template.
            s.replace("(\\d)", "<$1>").alias("group"),
            s.replace_all("(\\d)", "${1}x").alias("braced"),
            s.replace_all("$", "!", literal=True).alias("dollar"),
            s.split("\\d+", literal=False, inclusive=True)
            .list()
            .join("|")
            .alias("inclusive"),
            s.split("", literal=False).list().join("|").alias("chars"),
            s.count_matches("$", literal=True).alias("dollars"),
        ]
    )
    assert_equal(out.item(0, "plain").string(), "a1$122c")
    assert_equal(out.item(0, "group").string(), "a<1>b22c")
    assert_equal(out.item(0, "braced").string(), "a1xb2x2xc")
    assert_equal(out.item(1, "dollar").string(), "price !5")
    assert_equal(out.item(0, "inclusive").string(), "a1|b22|c")
    assert_equal(out.item(2, "chars").string(), "a|b|c")
    assert_equal(out.item(1, "dollars").uint32(), 1)


def test_invalid_regex_patterns_raise() raises:
    var frame = DataFrame([Series("s", Column[String](["abc"]))])
    for source in [frame.copy(), frame.clear()]:
        for invalid in [
            col("s").str().contains("a(b"),
            col("s").str().contains("(?=a)"),
            col("s").str().extract("[a", 1),
            col("s").str().extract_all("a{2,1}"),
            col("s").str().replace("(a)\\1", "x"),
            col("s").str().replace("a.", "x", n=2),
            col("s").str().split("(", literal=False),
            col("s").str().find("*"),
        ]:
            with assert_raises():
                _ = source.select(invalid)


def test_regex_capture_struct_and_escaping_preserve_layouts_nulls_and_schema() raises:
    var pattern = "(?P<head>\\w+)(?:-(\\d+))?"
    var expected = DataFrame(
        [
            Series(
                "captures",
                StructColumn(
                    [
                        _strings(
                            "head",
                            ["id", "", "Straße", "no", "", "a"],
                            [True, False, True, True, False, True],
                        ),
                        _strings(
                            "2",
                            ["12", "", "", "", "", ""],
                            [True, False, False, False, False, False],
                        ),
                    ],
                    [UInt8(61)],
                ),
            ),
            _strings(
                "escaped",
                [
                    "id\\-12 and id\\-7",
                    "",
                    "Straße 9",
                    "no digits",
                    "",
                    "a\\.b a\\.b",
                ],
                _valid(),
            ),
        ]
    )
    var exprs: List[Expr] = [
        col("s").str().extract_groups(pattern).alias("captures"),
        col("s").str().escape_regex().alias("escaped"),
    ]
    for layout in range(4):
        var source = _source(layout)
        assert_frame(source.select_exprs(exprs, batch_size=1), expected)
        for optimize in [False, True]:
            for streaming in [False, True]:
                var result = (
                    source.lazy()
                    .select_exprs(exprs)
                    .collect(
                        optimize=optimize, streaming=streaming, batch_size=1
                    )
                )
                assert_frame(result, expected)
        assert_frame(source.clear().select_exprs(exprs), expected.clear())
        var fields = source.select(
            col("s").str().extract_groups(pattern)
        ).unnest("s")
        assert_equal(fields.columns(), [String("head"), "2"])
        assert_equal(fields.item(0, "2").string(), "12")
        assert_true(fields.item(1, "head").is_null())
        assert_true(fields.item(4, "head").is_null())
        # A field access composes directly with the new struct operation.
        var length = source.select(
            col("s")
            .str()
            .extract_groups(pattern)
            .field("head")
            .str()
            .len_chars()
        )
        assert_equal(length.item(2, "head").int64(), 6)


def test_regex_without_captures_returns_empty_structs_with_row_count() raises:
    var expr = col("s").str().extract_groups("x")
    for layout in range(4):
        var source = _source(layout)
        for streaming in [False, True]:
            var result = (
                source.lazy()
                .select(expr)
                .collect(streaming=streaming, batch_size=1)
            )
            var series = result.column("s")
            assert_equal(len(series), source.height())
            assert_true(series.dtype().is_struct())
            assert_equal(series.dtype().field_count(), 0)
            assert_equal(series.null_count(), 0)
            var taken = series.take([5, 1, 3])
            assert_equal(len(taken), 3)
            assert_equal(len(taken.slice(1, 1)), 1)
            assert_equal(len(series.take_or_null([1, -1, 4])), 3)
            assert_true(series.take_or_null([1, -1, 4]).get(1).is_null())
        var empty = source.clear().select(expr)
        assert_equal(empty.height(), 0)
        assert_equal(empty.column("s").dtype().field_count(), 0)

        # Empty capture structs retain their explicit length at the Arrow
        # boundary and when flattened/restored as grouping keys.
        var values = source.select(expr).column("s")
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_arrow_series(values, array, schema)
        var imported = import_arrow_series(array, schema)
        assert_equal(len(imported), source.height())
        assert_equal(imported.dtype().field_count(), 0)
        var grouped = (
            source.with_columns(expr.alias("empty"))
            .group_by("empty")
            .agg(col("g").sum())
        )
        assert_equal(grouped.height(), 1)
        assert_equal(grouped.item(0, "g").int64(), 11)


def test_regex_selectors_expand_in_schema_order_and_compose_with_expressions() raises:
    var source = DataFrame(
        [
            Series("cost_b", Column[Int64]([3, 5])),
            Series("id", Column[Int64]([1, 2])),
            Series("cost_a", Column[Int64]([7, 11])),
            Series("α", Column[Int64]([13, 17])),
            Series("x\x1f.y", Column[Int64]([19, 23])),
        ]
    )
    var expr = (col("^cost_.*$") + 1).name_suffix("_plus")
    var expected = DataFrame(
        [
            Series("cost_b_plus", Column[Int64]([4, 6])),
            Series("cost_a_plus", Column[Int64]([8, 12])),
        ]
    )
    assert_frame(source.select(expr), expected)
    for optimize in [False, True]:
        for streaming in [False, True]:
            assert_frame(
                source.lazy()
                .select(expr)
                .collect(optimize=optimize, streaming=streaming, batch_size=1),
                expected,
            )
    assert_equal(
        source.select(columns("cost")).columns(), [String("cost_b"), "cost_a"]
    )
    assert_equal(source.select(columns("\\p{Greek}")).columns(), [String("α")])
    assert_equal(
        source.select(col(["id", "^cost_.*$"])).columns(),
        [String("cost_b"), "id", "cost_a"],
    )
    assert_equal(source.select(columns("\\.")).columns(), [String("x\x1f.y")])
    assert_equal(source.select(col("^nothing$")).width(), 0)
    assert_equal(
        source.select(col("^cost_.*$").sum()).item(0, "cost_b").int64(), 8
    )
    var kept = (
        source.lazy().filter(all_horizontal([col("^cost_.*$") > 4])).collect()
    )
    assert_equal(kept.height(), 1)
    assert_equal(kept.item(0, "id").int64(), 2)
    for input in [source.copy(), source.clear()]:
        with assert_raises():
            _ = input.select(col("^[$"))
        with assert_raises():
            _ = input.select(columns("("))
        with assert_raises():
            _ = input.select(col("absent"))


def test_new_regex_apis_validate_even_for_empty_or_unobserved_inputs() raises:
    var source = _source(0)
    for input in [source.copy(), source.clear()]:
        with assert_raises():
            _ = input.select(col("s").str().extract_groups("("))
        with assert_raises():
            _ = input.select(col("g").str().extract_groups("(a)"))
        with assert_raises():
            _ = input.select(col("g").str().escape_regex())
        with assert_raises():
            _ = input.select(col("s").str().contains("(?-u)."))
    # All metacharacters, control bytes and Unicode round-trip literally.
    var text = String("a.b*c+(d)|[e]{f}^$#&-~\\\t\nα😀")
    var frame = DataFrame([Series("s", Column[String]([text.copy()]))])
    var escaped = (
        frame.select(col("s").str().escape_regex()).item(0, "s").string()
    )
    assert_equal(
        frame.filter(col("s").str().contains("^" + escaped + "$")).height(), 1
    )


def test_literal_alternatives_on_nullable_string_layouts() raises:
    # Polars results, including a multibyte alternative and captures. The
    # same expression must work with offsets, chunks and string views.
    var pattern = "(id|Straße|a\\.b|never|zebra)"
    var padding = String("!") * 96
    var valid = _valid()
    var found: List[Bool] = [True, False, True, False, False, True]
    var expected = DataFrame(
        [
            Series("c", Column[Bool](found.copy(), valid.copy())),
            Series("n", Column[UInt32]([2, 0, 1, 0, 0, 2], valid.copy())),
            _strings("x", ["id", "", "Straße", "", "", "a.b"], found),
            _strings(
                "r",
                [
                    padding + "id!-12 and id!-7",
                    "",
                    padding + "Straße! 9",
                    padding + "no digits",
                    padding,
                    padding + "a.b! a.b!",
                ],
                valid,
            ),
        ]
    )
    for layout in range(4):
        var source = _source(layout, padding)
        var expressions: List[Expr] = [
            col("s").str().contains(pattern).alias("c"),
            col("s").str().count_matches(pattern).alias("n"),
            col("s").str().extract(pattern, 1).alias("x"),
            col("s").str().replace_all(pattern, "$1!").alias("r"),
        ]
        assert_frame(source.select_exprs(expressions), expected)
        assert_frame(
            source.lazy().select_exprs(expressions).collect(), expected
        )
