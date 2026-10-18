"""Unicode strings, literal patterns and arbitrary binary cells.

Expected strings/bytes are literal semantic examples. Layouts deliberately
include UTF8View inline and external descriptors, offsets and chunk boundaries.
"""
# contract-group: expr.text_binary
# contract-group: types.cast
# contract-group: reference.anchors
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    Series,
    StringColumn,
    ListColumn,
    Expr,
    col,
    lit,
    concat_str,
    when,
)
from dataframe.string_view import StringViewBuilder
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.coverage_compare import assert_frame
from tests.support.text_binary_reference import like_reference


def _source(
    values: List[String], valid: List[Bool], layout: Int
) raises -> DataFrame:
    if layout == 3:
        var builder = StringViewBuilder(len(values))
        for i in range(len(values)):
            if valid[i]:
                builder.append(values[i])
            else:
                builder.append_null()
        var storage = builder^.finish()
        # Prove that long examples really take the external descriptor path.
        for i in range(len(values)):
            if valid[i] and values[i].byte_length() > 12:
                assert_true(not storage.view(i).is_inline())
        return DataFrame([Series("s", StringColumn(storage^))])
    var frame = DataFrame(
        [Series("s", Column[String](values.copy(), valid.copy()))]
    )
    if layout == 1:
        return offset_layout(frame)
    if layout == 2:
        return chunk_layout(frame)
    return frame^


def _validity(n: Int, mode: Int) -> List[Bool]:
    var result = List[Bool]()
    for row in range(n):
        result.append(mode == 0 or (mode == 1 and row % 3 != 1))
    return result^


def _bits(valid: List[Bool]) -> List[UInt8]:
    var result = List[UInt8](length=(len(valid) + 7) // 8, fill=0)
    for row in range(len(valid)):
        if valid[row]:
            result[row // 8] |= UInt8(1) << UInt8(row % 8)
    return result^


def _strings(
    name: String, values: List[String], valid: List[Bool]
) raises -> Series:
    return Series(name, Column[String](values.copy(), valid.copy()))


def test_unicode_lengths_slices_reverse_and_physical_string_layouts() raises:
    var values: List[String] = [
        "",
        "aé中🙂",
        "e\u0301",
        "a\x00b",
        "abcdefghijklmnopé",
    ]
    var chars: List[Int64] = [0, 4, 2, 3, 17]
    var bytes: List[Int64] = [0, 10, 3, 3, 18]
    var reversed: List[String] = [
        "",
        "🙂中éa",
        "\u0301e",
        "b\x00a",
        "éponmlkjihgfedcba",
    ]
    var middle: List[String] = ["", "é中", "\u0301", "\x00b", "bc"]
    var last: List[String] = ["", "中🙂", "e\u0301", "\x00b", "pé"]
    var head: List[String] = ["", "aé", "e\u0301", "a\x00", "ab"]
    for mode in range(3):
        var valid = _validity(len(values), mode)
        var expected = DataFrame(
            [
                Series("chars", Column[Int64](chars.copy(), valid.copy())),
                Series("bytes", Column[Int64](bytes.copy(), valid.copy())),
                _strings("reverse", reversed, valid),
                _strings("middle", middle, valid),
                _strings("last", last, valid),
                _strings("head", head, valid),
                _strings("clipped", values, valid),
                _strings(
                    "past", List[String](length=len(values), fill=""), valid
                ),
            ]
        )
        for layout in range(4):
            var source = _source(values, valid, layout)
            var expressions: List[Expr] = [
                col("s").str().len_chars().alias("chars"),
                col("s").str().len_bytes().alias("bytes"),
                col("s").str().reverse().alias("reverse"),
                col("s").str().slice(1, 2).alias("middle"),
                col("s").str().tail(2).alias("last"),
                col("s").str().head(2).alias("head"),
                col("s").str().slice(-100).alias("clipped"),
                col("s").str().slice(100).alias("past"),
            ]
            assert_frame(source.select_exprs(expressions), expected)
            assert_frame(
                source.clear().select_exprs(expressions), expected.clear()
            )
            assert_frame(source, DataFrame([_strings("s", values, valid)]))


def test_literal_search_replace_and_empty_pattern_rules() raises:
    var values: List[String] = [
        "a.a.a",
        "é🙂é",
        "",
        "a\x00a",
        "abcdefghijklmnopé",
    ]
    var first: List[String] = [
        "aXa.a",
        "é🙂é",
        "",
        "a\x00a",
        "abcdefghijklmnopé",
    ]
    var all: List[String] = ["aXaXa", "é🙂é", "", "a\x00a", "abcdefghijklmnopé"]
    var prefixed: List[String] = [
        "!a.a.a",
        "!é🙂é",
        "!",
        "!a\x00a",
        "!abcdefghijklmnopé",
    ]
    # As in Polars, an empty pattern matches at every code point boundary.
    var everywhere: List[String] = [
        "!a!.!a!.!a!",
        "!é!🙂!é!",
        "!",
        "!a!\x00!a!",
        "!a!b!c!d!e!f!g!h!i!j!k!l!m!n!o!p!é!",
    ]
    for mode in range(3):
        var valid = _validity(len(values), mode)
        var expected = DataFrame(
            [
                Series(
                    "contains",
                    Column[Bool]([True, False, False, False, False], valid),
                ),
                Series(
                    "starts",
                    Column[Bool]([False, True, False, False, False], valid),
                ),
                Series(
                    "ends",
                    Column[Bool]([False, True, False, False, True], valid),
                ),
                Series(
                    "empty",
                    Column[Bool](List[Bool](length=5, fill=True), valid),
                ),
                Series(
                    "nul",
                    Column[Bool]([False, False, False, True, False], valid),
                ),
                _strings("first", first, valid),
                _strings("all", all, valid),
                _strings("insert", prefixed, valid),
                _strings("everywhere", everywhere, valid),
            ]
        )
        var expressions: List[Expr] = [
            col("s").str().contains(".", literal=True).alias("contains"),
            col("s").str().starts_with("é").alias("starts"),
            col("s").str().ends_with("é").alias("ends"),
            col("s").str().contains("").alias("empty"),
            col("s").str().contains("\x00").alias("nul"),
            col("s").str().replace(".", "X", literal=True).alias("first"),
            col("s").str().replace_all(".", "X", literal=True).alias("all"),
            col("s").str().replace("", "!").alias("insert"),
            col("s").str().replace_all("", "!").alias("everywhere"),
        ]
        for layout in range(4):
            var source = _source(values, valid, layout)
            assert_frame(source.select_exprs(expressions), expected)
            assert_frame(
                source.clear().select_exprs(expressions), expected.clear()
            )
    # Replacement is nonoverlapping, with literal regex punctuation and Unicode.
    var source = _source(["aaaa", "[a].*", "ééé"], [True, True, True], 0)
    assert_frame(
        source.select_exprs(
            [
                col("s").str().replace_all("aa", "X").alias("aa"),
                col("s").str().replace_all("é", "🙂").alias("unicode"),
            ]
        ),
        DataFrame(
            [
                _strings("aa", ["XX", "[a].*", "ééé"], [True, True, True]),
                _strings(
                    "unicode", ["aaaa", "[a].*", "🙂🙂🙂"], [True, True, True]
                ),
            ]
        ),
    )


def test_sql_like_matches_codepoints_with_independent_pattern_relation() raises:
    # Literal anchors verify the independent DP matcher, including no escape.
    assert_true(like_reference("é", "_"))
    assert_true(not like_reference("e\u0301", "_"))
    assert_true(like_reference("", "%"))
    assert_true(not like_reference("", "_"))
    assert_true(like_reference("a\\XYZ", "a\\%"))
    assert_true(not like_reference("a%", "a\\%"))
    var values: List[String] = [
        "",
        "é",
        "e\u0301",
        "aé🙂b",
        "aaab",
        "a%b",
        "a\\XYZ",
        "[a]",
        "a\x00b",
        "abcdefghijklmnopé",
    ]
    var patterns: List[String] = [
        "",
        "%",
        "_",
        "__",
        "%é%",
        "a%b",
        "a_b",
        "a%%_b",
        "a\\%",
        "[a]",
        "%\x00%",
        "%é",
    ]
    for mode in range(3):
        var valid = _validity(len(values), mode)
        for layout in range(4):
            var source = _source(values, valid, layout)
            for pattern in patterns:
                var expected = List[Bool]()
                for value in values:
                    expected.append(like_reference(value, pattern))
                assert_frame(
                    source.select(col("s").str().like(pattern)),
                    DataFrame([Series("s", Column[Bool](expected^, valid))]),
                )
            assert_equal(
                source.clear().select(col("s").str().like("%")).shape(), (0, 1)
            )


def test_case_trimming_padding_and_sign_preserving_zero_fill() raises:
    var values: List[String] = [
        "AbÉΩ",
        "é中🙂",
        "  \tAb\n",
        "\u00a0Ab\u00a0",
        "éxé",
        "",
    ]
    var upper: List[String] = [
        "ABÉΩ",
        "É中🙂",
        "  \tAB\n",
        "\u00a0AB\u00a0",
        "ÉXÉ",
        "",
    ]
    var lower: List[String] = [
        "abéω",
        "é中🙂",
        "  \tab\n",
        "\u00a0ab\u00a0",
        "éxé",
        "",
    ]
    var strip: List[String] = ["AbÉΩ", "é中🙂", "Ab", "\u00a0Ab\u00a0", "éxé", ""]
    var left: List[String] = [
        "AbÉΩ",
        "é中🙂",
        "Ab\n",
        "\u00a0Ab\u00a0",
        "éxé",
        "",
    ]
    var right: List[String] = [
        "AbÉΩ",
        "é中🙂",
        "  \tAb",
        "\u00a0Ab\u00a0",
        "éxé",
        "",
    ]
    var custom: List[String] = ["AbÉΩ", "中🙂", "  \tAb\n", "Ab", "x", ""]
    for mode in range(3):
        var valid = _validity(len(values), mode)
        var expected = DataFrame(
            [
                _strings("upper", upper, valid),
                _strings("lower", lower, valid),
                _strings("strip", strip, valid),
                _strings("left", left, valid),
                _strings("right", right, valid),
                _strings("custom", custom, valid),
            ]
        )
        var expressions: List[Expr] = [
            col("s").str().to_uppercase().alias("upper"),
            col("s").str().to_lowercase().alias("lower"),
            col("s").str().strip_chars().alias("strip"),
            col("s").str().strip_chars_start().alias("left"),
            col("s").str().strip_chars_end().alias("right"),
            col("s").str().strip_chars("é\u00a0").alias("custom"),
        ]
        for layout in range(4):
            var source = _source(values, valid, layout)
            assert_frame(source.select_exprs(expressions), expected)
            assert_frame(
                source.clear().select_exprs(expressions), expected.clear()
            )
    var padded_values: List[String] = [
        "-7",
        "+8",
        "é",
        "",
        "abcdef",
        "null poison",
    ]
    for layout in range(4):
        var source = _source(
            padded_values, [True, True, True, True, True, False], layout
        )
        var valid: List[Bool] = [True, True, True, True, True, False]
        assert_frame(
            source.select_exprs(
                [
                    col("s").str().pad_start(5, "🙂").alias("left"),
                    col("s").str().pad_end(5, "é").alias("right"),
                    col("s").str().zfill(5).alias("zero"),
                    col("s").str().pad_start(0).alias("identity"),
                ]
            ),
            DataFrame(
                [
                    _strings(
                        "left",
                        ["🙂🙂🙂-7", "🙂🙂🙂+8", "🙂🙂🙂🙂é", "🙂🙂🙂🙂🙂", "abcdef", ""],
                        valid,
                    ),
                    _strings(
                        "right",
                        ["-7ééé", "+8ééé", "ééééé", "ééééé", "abcdef", ""],
                        valid,
                    ),
                    _strings(
                        "zero",
                        ["-0007", "+0008", "0000é", "00000", "abcdef", ""],
                        valid,
                    ),
                    _strings("identity", padded_values, valid),
                ]
            ),
        )


def test_concat_str_broadcast_and_split_preserve_empty_and_null_parts() raises:
    var values: List[String] = [
        "",
        "::a::",
        "a::::b",
        "é::🙂",
        "abcdefghijklmnopé",
    ]
    var pieces: List[String] = [
        "",
        "",
        "a",
        "",
        "a",
        "",
        "b",
        "é",
        "🙂",
        "abcdefghijklmnopé",
    ]
    var inclusive: List[String] = [
        "",
        "::",
        "a::",
        "",
        "a::",
        "::",
        "b",
        "é::",
        "🙂",
        "abcdefghijklmnopé",
    ]
    for mode in range(3):
        var valid = _validity(len(values), mode)
        # Canonical offsets permit null list rows to retain hidden child spans;
        # the comparator must use parent validity before reading those children.
        var expected = DataFrame(
            [
                Series(
                    "parts",
                    ListColumn(
                        [0, 1, 4, 7, 9, 10],
                        Series("item", Column[String](pieces.copy())),
                        _bits(valid),
                    ),
                ),
                Series(
                    "inclusive",
                    ListColumn(
                        [0, 1, 4, 7, 9, 10],
                        Series("item", Column[String](inclusive.copy())),
                        _bits(valid),
                    ),
                ),
                _strings(
                    "concat",
                    [
                        "<>|",
                        "<>|::a::",
                        "<>|a::::b",
                        "<>|é::🙂",
                        "<>|abcdefghijklmnopé",
                    ],
                    valid,
                ),
            ]
        )
        var expressions: List[Expr] = [
            col("s").str().split("::").alias("parts"),
            col("s").str().split("::", inclusive=True).alias("inclusive"),
            concat_str([lit("<>"), col("s")], separator="|").alias("concat"),
        ]
        for layout in range(4):
            var source = _source(values, valid, layout)
            assert_frame(source.select_exprs(expressions), expected)
            assert_frame(
                source.clear().select_exprs(expressions), expected.clear()
            )
    var source = DataFrame(
        [
            Series(
                "a", Column[String](["a", "poison", "c"], [True, False, True])
            ),
            Series(
                "b", Column[String](["x", "y", "poison"], [True, True, False])
            ),
        ]
    )
    assert_frame(
        source.select(concat_str([col("a"), col("b")], separator="é")),
        DataFrame([_strings("a", ["aéx", "", ""], [True, False, False])]),
    )
    assert_equal(
        source.select(concat_str([lit("é"), lit("🙂")], separator="|"))
        .item()
        .string(),
        "é|🙂",
    )


def test_binary_comparison_and_cast_keep_arbitrary_bytes_distinct_from_text() raises:
    var values: List[List[UInt8]] = [
        [],
        [0],
        [97],
        [195, 169],
        [255],
        [255, 0],
        [255, 0, 122],
    ]
    var valid: List[Bool] = [True, True, True, True, True, True, False]
    var source = DataFrame([Series.binary("b", values, valid)])
    for layout in range(3):
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        var out = frame.select_exprs(
            [
                (col("b") == "a").alias("eq"),
                (col("b") < "é").alias("lt"),
                col("b").fill_null(lit("é").cast("binary")).alias("filled"),
                when(col("b") == "a")
                .then(col("b"))
                .otherwise(col("b"))
                .alias("identity"),
            ]
        )
        var filled: List[List[UInt8]] = [
            [],
            [0],
            [97],
            [195, 169],
            [255],
            [255, 0],
            [195, 169],
        ]
        assert_frame(
            out,
            DataFrame(
                [
                    Series(
                        "eq",
                        Column[Bool](
                            [False, False, True, False, False, False, False],
                            valid,
                        ),
                    ),
                    Series(
                        "lt",
                        Column[Bool](
                            [True, True, True, False, False, False, False],
                            valid,
                        ),
                    ),
                    Series.binary("filled", filled),
                    Series.binary("identity", values, valid),
                ]
            ),
        )
        assert_frame(
            frame.clear().select_exprs(
                [(col("b") == "a").alias("eq"), (col("b") < "é").alias("lt")]
            ),
            out.select(["eq", "lt"]).clear(),
        )
        with assert_raises():
            _ = frame.select(col("b").str().len_bytes())
        with assert_raises():
            _ = frame.vstack(DataFrame([Series("b", Column[String](["a"]))]))
    var texts: List[String] = ["", "a\x00é🙂", "abcdefghijklmnopé"]
    var utf8: List[List[UInt8]] = [
        [],
        [97, 0, 195, 169, 240, 159, 153, 130],
        [
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            195,
            169,
        ],
    ]
    for layout in range(4):
        var frame = _source(texts, [True, True, False], layout)
        assert_frame(
            frame.select(col("s").cast("binary")),
            DataFrame([Series.binary("s", utf8, [True, True, False])]),
        )
        assert_frame(
            frame.select(col("s").cast("binary").cast("string")), frame
        )


def test_binary_utf8_cast_validates_every_row_and_ignores_null_payloads() raises:
    var bytes: List[List[UInt8]] = [
        [255],
        [192, 175],
        [237, 160, 128],
        [244, 144, 128, 128],
        [226, 130],
        [128],
        [0, 97],
        [240, 159, 153, 130],
        [],
        [244, 143, 191, 191],
    ]
    var expected = DataFrame(
        [
            _strings(
                "b",
                ["", "", "", "", "", "", "\x00a", "🙂", "", "\U0010ffff"],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                ],
            )
        ]
    )
    var source = DataFrame([Series.binary("b", bytes)])
    for layout in range(3):
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        assert_frame(
            frame.select(col("b").cast("string", strict=False)), expected
        )
        for bad in range(6):
            with assert_raises():
                _ = frame.take([bad]).select(col("b").cast("string"))
    var poison = DataFrame([Series.binary("b", [[255], [97]], [False, True])])
    assert_frame(
        poison.select(col("b").cast("string")),
        DataFrame([_strings("b", ["", "a"], [False, True])]),
    )


def test_string_binding_and_argument_errors() raises:
    var frame = DataFrame(
        [Series("i", Column[Int64]([1])), Series("s", Column[String](["é"]))]
    )
    for source in [frame.copy(), frame.clear()]:
        for invalid in [
            col("i").str().len_chars(),
            col("i").str().like("%"),
            col("i").str().split(","),
            col("s").str().slice(0, -2),
            col("s").str().pad_start(-1),
            col("s").str().pad_end(2, ""),
            col("s").str().pad_start(2, "ab"),
            concat_str([col("s"), col("i")]),
            col("missing").str().reverse(),
        ]:
            with assert_raises():
                _ = source.select(invalid)
    # Separator validation belongs to the row kernel; binding errors above
    # are independent of whether any rows are present.
    with assert_raises():
        _ = frame.select(col("s").str().split(""))
    with assert_raises():
        _ = concat_str([])
