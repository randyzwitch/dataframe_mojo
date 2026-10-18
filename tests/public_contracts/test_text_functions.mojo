"""String functions that need no regular expressions (#798): titlecase,
prefix and suffix stripping, find, count_matches, contains_any,
replace_many, hex and base64, to_integer, split_exact/splitn,
concat_str(ignore_nulls) and str.join.

Expected values are Polars 1.44 results for the same input, written out as
literals. Every function runs over offsets, a sliced window, chunks and
string views (rows past 12 bytes reference a buffer), with and without
nulls, eagerly and through LazyFrame.
"""
# contract-group: expr.text_binary
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    Expr,
    col,
    lit,
    concat_str,
)
from dataframe.accelerator.dictionaries import lower_dictionary_operations
from dataframe.accelerator.rows import lower_rows
from dataframe.string_view import StringViewBuilder
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.coverage_compare import assert_frame


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
        return DataFrame([Series("s", StringColumn(builder^.finish()))])
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


def _both(valid: List[Bool], present: List[Bool]) -> List[Bool]:
    var result = List[Bool]()
    for i in range(len(valid)):
        result.append(valid[i] and present[i])
    return result^


def _strings(
    name: String, values: List[String], valid: List[Bool]
) raises -> Series:
    return Series(name, Column[String](values.copy(), valid.copy()))


def _counts(
    name: String, values: List[UInt32], valid: List[Bool]
) raises -> Series:
    return Series(name, Column[UInt32](values.copy(), valid.copy()))


def _values() -> List[String]:
    return [
        "hello wORLD",
        "élan VITAL-x",
        "",
        "SKU-12SKU-",
        "a_b c'd 1st",
        "ǆungla ßtraße σς",
        "abcdefghijklmnopé SKU-",
        "  two  spaces",
    ]


def _expressions() -> List[Expr]:
    var s = col("s").str()
    return [
        s.to_titlecase().alias("title"),
        s.strip_prefix("SKU-").alias("prefix"),
        s.strip_suffix("SKU-").alias("suffix"),
        s.strip_prefix("abcd").alias("long"),
        s.find("l", literal=True).alias("find"),
        s.find("").alias("find_empty"),
        s.count_matches("SKU").alias("count"),
        s.count_matches("").alias("count_empty"),
        s.contains_any(["sku", "WORLD"], ascii_case_insensitive=True).alias(
            "any"
        ),
        s.replace_many(["l", "SKU", "é"], ["L", "#", "e"]).alias("many"),
        s.replace_many(["bc", "abcd", "SKU-", "1"], "_", leftmost=True).alias(
            "leftmost"
        ),
        s.replace_many(["bc", "abcd", "SKU-", "1"], "_").alias("standard"),
        s.encode("hex").alias("hex"),
        s.encode("base64").alias("base64"),
        s.split_exact(" ", 1).field("field_1").alias("exact1"),
        s.split_exact(" ", 1).field("field_0").alias("exact0"),
        s.split_exact(" ", 1, inclusive=True)
        .field("field_1")
        .alias("inclusive"),
        s.split_exact("", 2).field("field_2").alias("chars"),
        s.splitn(" ", 2).field("field_1").alias("rest"),
        s.splitn("", 2).field("field_1").alias("rest_chars"),
    ]


def _expected(valid: List[Bool]) raises -> DataFrame:
    var title: List[String] = [
        "Hello World",
        "Élan Vital-X",
        "",
        "Sku-12Sku-",
        "A_B C'D 1St",
        "Ǆungla SStraße Σς",
        "Abcdefghijklmnopé Sku-",
        "  Two  Spaces",
    ]
    var prefix: List[String] = [
        "hello wORLD",
        "élan VITAL-x",
        "",
        "12SKU-",
        "a_b c'd 1st",
        "ǆungla ßtraße σς",
        "abcdefghijklmnopé SKU-",
        "  two  spaces",
    ]
    var suffix: List[String] = [
        "hello wORLD",
        "élan VITAL-x",
        "",
        "SKU-12",
        "a_b c'd 1st",
        "ǆungla ßtraße σς",
        "abcdefghijklmnopé ",
        "  two  spaces",
    ]
    var long: List[String] = [
        "hello wORLD",
        "élan VITAL-x",
        "",
        "SKU-12SKU-",
        "a_b c'd 1st",
        "ǆungla ßtraße σς",
        "efghijklmnopé SKU-",
        "  two  spaces",
    ]
    var many: List[String] = [
        "heLLo wORLD",
        "eLan VITAL-x",
        "",
        "#-12#-",
        "a_b c'd 1st",
        "ǆungLa ßtraße σς",
        "abcdefghijkLmnope #-",
        "  two  spaces",
    ]
    var leftmost: List[String] = [
        "hello wORLD",
        "élan VITAL-x",
        "",
        "__2_",
        "a_b c'd _st",
        "ǆungla ßtraße σς",
        "_efghijklmnopé _",
        "  two  spaces",
    ]
    var standard: List[String] = [
        "hello wORLD",
        "élan VITAL-x",
        "",
        "__2_",
        "a_b c'd _st",
        "ǆungla ßtraße σς",
        "a_defghijklmnopé _",
        "  two  spaces",
    ]
    var hex: List[String] = [
        "68656c6c6f20774f524c44",
        "c3a96c616e20564954414c2d78",
        "",
        "534b552d3132534b552d",
        "615f622063276420317374",
        "c786756e676c6120c39f747261c39f6520cf83cf82",
        "6162636465666768696a6b6c6d6e6f70c3a920534b552d",
        "202074776f2020737061636573",
    ]
    var base64: List[String] = [
        "aGVsbG8gd09STEQ=",
        "w6lsYW4gVklUQUwteA==",
        "",
        "U0tVLTEyU0tVLQ==",
        "YV9iIGMnZCAxc3Q=",
        "x4Z1bmdsYSDDn3RyYcOfZSDPg8+C",
        "YWJjZGVmZ2hpamtsbW5vcMOpIFNLVS0=",
        "ICB0d28gIHNwYWNlcw==",
    ]
    var split_present: List[Bool] = [
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    var chars_present: List[Bool] = [
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
    ]
    var found: List[Bool] = [True, True, False, False, False, True, True, False]
    return DataFrame(
        [
            _strings("title", title, valid),
            _strings("prefix", prefix, valid),
            _strings("suffix", suffix, valid),
            _strings("long", long, valid),
            _counts("find", [2, 2, 0, 0, 0, 5, 11, 0], _both(valid, found)),
            _counts("find_empty", List[UInt32](length=8, fill=0), valid),
            _counts("count", [0, 0, 0, 2, 0, 0, 1, 0], valid),
            _counts("count_empty", [12, 13, 1, 11, 12, 17, 23, 14], valid),
            Series(
                "any",
                Column[Bool](
                    [True, False, False, True, False, False, True, False],
                    valid.copy(),
                ),
            ),
            _strings("many", many, valid),
            _strings("leftmost", leftmost, valid),
            _strings("standard", standard, valid),
            _strings("hex", hex, valid),
            _strings("base64", base64, valid),
            _strings(
                "exact1",
                ["wORLD", "VITAL-x", "", "", "c'd", "ßtraße", "SKU-", ""],
                _both(valid, split_present),
            ),
            _strings(
                "exact0",
                [
                    "hello",
                    "élan",
                    "",
                    "SKU-12SKU-",
                    "a_b",
                    "ǆungla",
                    "abcdefghijklmnopé",
                    "",
                ],
                valid,
            ),
            _strings(
                "inclusive",
                ["wORLD", "VITAL-x", "", "", "c'd ", "ßtraße ", "SKU-", " "],
                _both(valid, split_present),
            ),
            _strings(
                "chars",
                ["l", "a", "", "U", "b", "n", "c", "t"],
                _both(valid, chars_present),
            ),
            _strings(
                "rest",
                [
                    "wORLD",
                    "VITAL-x",
                    "",
                    "",
                    "c'd 1st",
                    "ßtraße σς",
                    "SKU-",
                    " two  spaces",
                ],
                _both(valid, split_present),
            ),
            _strings(
                "rest_chars",
                [
                    "ello wORLD",
                    "lan VITAL-x",
                    "",
                    "KU-12SKU-",
                    "_b c'd 1st",
                    "ungla ßtraße σς",
                    "bcdefghijklmnopé SKU-",
                    " two  spaces",
                ],
                _both(valid, chars_present),
            ),
        ]
    )


def test_text_functions_match_polars_on_every_layout() raises:
    var values = _values()
    for mode in range(3):
        var valid = _validity(len(values), mode)
        var expected = _expected(valid)
        for layout in range(4):
            var source = _source(values, valid, layout)
            assert_frame(source.select_exprs(_expressions()), expected)
            assert_frame(
                source.clear().select_exprs(_expressions()), expected.clear()
            )
            # The same through LazyFrame, streamed and materialized, in
            # batches that split the rows.
            for streaming in [True, False]:
                for batch_size in [65536, 3]:
                    assert_frame(
                        source.lazy()
                        .select_exprs(_expressions())
                        .collect(streaming=streaming, batch_size=batch_size),
                        expected,
                    )
            # The source keeps its own rows.
            assert_frame(source, DataFrame([_strings("s", values, valid)]))


def test_categorical_input_and_filters() raises:
    # A categorical runs the functions once per dictionary value; the
    # results equal those on the strings.
    var values = _values()
    var rows = List[String]()
    var valid = List[Bool]()
    for i in range(40):
        rows.append(values[i % len(values)])
        valid.append(i % 7 != 3)
    var strings = DataFrame(
        [Series("s", Column[String](rows.copy(), valid.copy()))]
    )
    var categorical = DataFrame(
        [strings.column("s").cast(DataType.CATEGORICAL)]
    )
    assert_frame(
        categorical.select_exprs(_expressions()),
        strings.select_exprs(_expressions()),
    )
    assert_frame(
        categorical.lazy().select_exprs(_expressions()).collect(),
        strings.select_exprs(_expressions()),
    )
    # In a filter, eager and lazy.
    var predicate = (
        col("s").str().contains_any(["SKU", "wor"])
        | col("s").str().find("l", literal=True).is_null()
    )
    var kept = strings.filter(predicate.copy())
    assert_frame(
        categorical.filter(predicate.copy()).select_exprs(
            [col("s").cast(DataType.STRING)]
        ),
        kept,
    )
    assert_frame(strings.lazy().filter(predicate.copy()).collect(), kept)
    assert_true(kept.height() > 0)
    for row in range(kept.height()):
        if kept.item(row, "s").is_null():
            continue
        var text = kept.item(row, "s").string()
        assert_true("SKU" in text or "wor" in text or "l" not in text)


def test_split_struct_dtype_fields_and_null_rows() raises:
    var frame = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    ["a b c", "poison", "", "x"], [True, False, True, True]
                ),
            )
        ]
    )
    var exact = frame.select(col("s").str().split_exact(" ", 1))
    assert_true(
        exact.column("s").dtype()
        == DataType.struct(
            ["field_0", "field_1"], [DataType.STRING, DataType.STRING]
        )
    )
    # A null string gives a valid struct whose fields are null, as in
    # Polars.
    assert_true(not exact.item(1, "s").is_null())
    var fields = frame.select_exprs(
        [
            col("s").str().split_exact(" ", 1).field("field_0").alias("a"),
            col("s").str().split_exact(" ", 1).field("field_1").alias("b"),
            col("s").str().splitn(" ", 1).field("field_0").alias("whole"),
            col("s").str().split_exact(" ", 0).field("field_0").alias("first"),
        ]
    )
    assert_frame(
        fields,
        DataFrame(
            [
                _strings("a", ["a", "", "", "x"], [True, False, True, True]),
                _strings("b", ["b", "", "", ""], [True, False, False, False]),
                _strings(
                    "whole", ["a b c", "", "", "x"], [True, False, True, True]
                ),
                _strings(
                    "first", ["a", "", "", "x"], [True, False, True, True]
                ),
            ]
        ),
    )
    var schema = (
        frame.lazy().select(col("s").str().splitn(" ", 3)).collect_schema()
    )
    var three = DataType.struct(
        ["field_0", "field_1", "field_2"],
        [DataType.STRING, DataType.STRING, DataType.STRING],
    )
    assert_equal(schema, ["s: " + three.name()])


def test_titlecase_unicode_rules() raises:
    # Letters after any non-alphabetic character (Unicode `Alphabetic`, so
    # not after a CJK character or before a combining mark) start a word;
    # a capital sigma ending a word lowercases to final sigma; uppercase
    # mappings can grow ("ﬁ" to "FI").
    var frame = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    [
                        "ΣΑΣ. ǅ",
                        "中a 1st",
                        "x\u0301yz",
                        "ﬁx",
                        "ΑΣ",
                        "Σ",
                        "ÉCOLE d'ÉTÉ",
                    ]
                ),
            )
        ]
    )
    assert_frame(
        frame.select(col("s").str().to_titlecase()),
        DataFrame(
            [
                Series(
                    "s",
                    Column[String](
                        [
                            "Σας. Ǆ",
                            "中a 1St",
                            "X\u0301Yz",
                            "FIx",
                            "Ας",
                            "Σ",
                            "École D'Été",
                        ]
                    ),
                )
            ]
        ),
    )


def test_replace_many_and_contains_any_edges() raises:
    var frame = DataFrame(
        [Series("s", Column[String](["é a", "aé", "abcd", "xabcd", "ABC"]))]
    )
    var s = col("s").str()
    assert_frame(
        frame.select_exprs(
            [
                # An empty pattern matches between code points; under the
                # default semantics it is the first match everywhere.
                s.replace_many([""], ["-"]).alias("empty"),
                s.replace_many(["", "a"], ["-", "A"]).alias("empty_first"),
                # Leftmost-first: "a" before "" where both start; no empty
                # match right where the previous match ended.
                s.replace_many(["a", ""], ["A", "-"], leftmost=True).alias(
                    "leftmost_empty"
                ),
                s.replace_many(["bc", "abcd"], ["1", "2"]).alias("standard"),
                s.replace_many(["bc", "abcd"], ["1", "2"], leftmost=True).alias(
                    "leftmost"
                ),
                s.replace_many(["abc", "b"], ["1", "2"]).alias("inner"),
                s.replace_many(
                    ["b", "C"], ["_", "!"], ascii_case_insensitive=True
                ).alias("folded"),
                s.contains_any(["É"], ascii_case_insensitive=True).alias(
                    "unicode_case"
                ),
                s.contains_any([""]).alias("any_empty"),
                s.contains_any(["abc"]).alias("case"),
            ]
        ),
        DataFrame(
            [
                Series(
                    "empty",
                    Column[String](
                        [
                            "-é- -a-",
                            "-a-é-",
                            "-a-b-c-d-",
                            "-x-a-b-c-d-",
                            "-A-B-C-",
                        ]
                    ),
                ),
                Series(
                    "empty_first",
                    Column[String](
                        [
                            "-é- -a-",
                            "-a-é-",
                            "-a-b-c-d-",
                            "-x-a-b-c-d-",
                            "-A-B-C-",
                        ]
                    ),
                ),
                Series(
                    "leftmost_empty",
                    Column[String](
                        ["-é- A", "Aé-", "Ab-c-d-", "-xAb-c-d-", "-A-B-C-"]
                    ),
                ),
                Series(
                    "standard",
                    Column[String](["é a", "aé", "a1d", "xa1d", "ABC"]),
                ),
                Series(
                    "leftmost", Column[String](["é a", "aé", "2", "x2", "ABC"])
                ),
                Series(
                    "inner",
                    Column[String](["é a", "aé", "a2cd", "xa2cd", "ABC"]),
                ),
                Series(
                    "folded",
                    Column[String](["é a", "aé", "a_!d", "xa_!d", "A_!"]),
                ),
                Series(
                    "unicode_case",
                    Column[Bool]([False, False, False, False, False]),
                ),
                Series(
                    "any_empty", Column[Bool]([True, True, True, True, True])
                ),
                Series("case", Column[Bool]([False, False, True, True, False])),
            ]
        ),
    )


def test_decode_and_to_integer_strict_or_null() raises:
    var frame = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    ["1F", "ff", "zz", "poison", "-10", "0x1f", "+7", ""],
                    [True, True, True, False, True, True, True, True],
                ),
            )
        ]
    )
    var lenient = frame.select(col("s").str().to_integer(base=16, strict=False))
    assert_frame(
        lenient,
        DataFrame(
            [
                Series(
                    "s",
                    Column[Int64](
                        [31, 255, 0, 0, -16, 0, 7, 0],
                        [True, True, False, False, True, False, True, False],
                    ),
                )
            ]
        ),
    )
    with assert_raises(contains="strict integer parsing failed"):
        _ = frame.select(col("s").str().to_integer(base=16))
    var edges = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    [
                        "7fffffffffffffff",
                        "-8000000000000000",
                        "8000000000000000",
                        "-",
                        "1_0",
                        " 1",
                    ]
                ),
            )
        ]
    )
    assert_frame(
        edges.select(col("s").str().to_integer(base=16, strict=False)),
        DataFrame(
            [
                Series(
                    "s",
                    Column[Int64](
                        [Int64.MAX, Int64.MIN, 0, 0, 0, 0],
                        [True, True, False, False, False, False],
                    ),
                )
            ]
        ),
    )
    assert_equal(
        edges.head(1)
        .select(lit("zz").str().to_integer(base=36))
        .item()
        .int64(),
        1295,
    )
    # Decoding: hex in either case, canonical padded base64; invalid text
    # raises when strict and is null otherwise.
    var hex = DataFrame(
        [Series("s", Column[String](["4A4b", "4", "zz", ""]))]
    ).select(col("s").str().decode("hex", strict=False))
    assert_true(hex.column("s").dtype() == DataType.BINARY)
    assert_frame(
        hex.select(col("s").cast(DataType.STRING)),
        DataFrame(
            [_strings("s", ["JK", "", "", ""], [True, False, False, True])]
        ),
    )
    var encoded = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    ["YWI=", "YWI", "YR==", "", "Y===", "YQ==", "YW I="]
                ),
            )
        ]
    )
    assert_frame(
        encoded.select(
            col("s").str().decode("base64", strict=False).cast(DataType.STRING)
        ),
        DataFrame(
            [
                _strings(
                    "s",
                    ["ab", "", "", "", "", "a", ""],
                    [True, False, False, True, False, True, False],
                )
            ]
        ),
    )
    for encoding in ["hex", "base64"]:
        with assert_raises(contains="encoding found"):
            _ = encoded.select(col("s").str().decode(encoding))
        # Encoding then decoding gives back the bytes.
        var values = _values()
        var source = DataFrame(
            [_strings("s", values, List[Bool](length=len(values), fill=True))]
        )
        assert_frame(
            source.select(
                col("s")
                .str()
                .encode(encoding)
                .str()
                .decode(encoding)
                .cast(DataType.STRING)
            ),
            source,
        )


def test_concat_str_ignore_nulls_and_str_join() raises:
    var frame = DataFrame(
        [
            Series(
                "a",
                Column[String](
                    ["a b", "poison", "aaaa", "x"], [True, False, True, True]
                ),
            ),
            Series(
                "b",
                Column[String](
                    ["p", "q", "poison", "poison"], [True, True, False, False]
                ),
            ),
            Series("g", Column[Int64]([1, 1, 2, 2])),
        ]
    )
    var concat = frame.select_exprs(
        [
            concat_str([col("a"), col("b")], separator="+", ignore_nulls=True),
            concat_str([col("b")], separator="+", ignore_nulls=True).alias(
                "one"
            ),
            concat_str(
                [col("b"), lit("é"), col("a")], separator="-", ignore_nulls=True
            ).alias("three"),
            concat_str([col("a"), col("b")], separator="+").alias("strict"),
        ]
    )
    assert_frame(
        concat,
        DataFrame(
            [
                _strings(
                    "a", ["a b+p", "q", "aaaa", "x"], [True, True, True, True]
                ),
                _strings("one", ["p", "q", "", ""], [True, True, True, True]),
                _strings(
                    "three",
                    ["p-é-a b", "q-é", "é-aaaa", "é-x"],
                    [True, True, True, True],
                ),
                _strings(
                    "strict", ["a b+p", "", "", ""], [True, False, False, False]
                ),
            ]
        ),
    )
    assert_frame(
        frame.lazy()
        .select_exprs(
            [concat_str([col("a"), col("b")], separator="+", ignore_nulls=True)]
        )
        .collect(),
        concat.select(["a"]),
    )
    # str.join aggregates: over the frame, per group, eagerly and lazily.
    var whole = frame.select_exprs(
        [
            col("a").str().join("-"),
            col("a").str().join("-", ignore_nulls=False).alias("strict"),
        ]
    )
    assert_frame(
        whole,
        DataFrame(
            [
                _strings("a", ["a b-aaaa-x"], [True]),
                _strings("strict", [""], [False]),
            ]
        ),
    )
    # No values join to "".
    assert_frame(
        frame.clear().select(col("a").str().join("-")),
        DataFrame([_strings("a", [""], [True])]),
    )
    var expressions: List[Expr] = [
        col("a").str().join(","),
        col("b").str().join(",").alias("b"),
        col("b").str().join(",", ignore_nulls=False).alias("strict"),
        col("a").str().to_titlecase().str().join("|").alias("titles"),
    ]
    var expected = DataFrame(
        [
            Series("g", Column[Int64]([1, 2])),
            _strings("a", ["a b", "aaaa,x"], [True, True]),
            _strings("b", ["p,q", ""], [True, True]),
            _strings("strict", ["p,q", ""], [True, False]),
            _strings("titles", ["A B", "Aaaa|X"], [True, True]),
        ]
    )
    assert_frame(
        frame.group_by("g", maintain_order=True).agg(expressions), expected
    )
    assert_frame(
        frame.lazy()
        .group_by("g", maintain_order=True)
        .agg(expressions)
        .collect(),
        expected,
    )


def test_text_function_argument_errors() raises:
    var frame = DataFrame(
        [Series("i", Column[Int64]([1])), Series("s", Column[String](["é"]))]
    )
    for source in [frame.copy(), frame.clear()]:
        for invalid in [
            col("i").str().to_titlecase(),
            col("i").str().strip_prefix("a"),
            col("i").str().split_exact(" ", 1),
            # Invalid regular expressions.
            col("s").str().find("a(b"),
            col("s").str().count_matches("[a"),
            col("s").str().contains_any(List[String]()),
            col("s").str().replace_many(List[String](), List[String]()),
            col("s").str().replace_many(["a"], ["x", "y"]),
            col("s").str().encode("utf8"),
            col("s").str().decode("base32"),
            col("s").str().to_integer(base=1),
            col("s").str().to_integer(base=37),
            col("s").str().splitn(" ", 0),
            col("s").str().split_exact(" ", -1),
            concat_str([col("s"), col("i")], ignore_nulls=True),
        ]:
            with assert_raises():
                _ = source.select(invalid)
    # With literal=True the same text is matched as is.
    assert_true(
        frame.select(col("s").str().find("a.b", literal=True)).item().is_null()
    )


def test_accelerator_lowering_rejects_text_functions() raises:
    # The row lowering knows none of these operations, and the dictionary
    # lowering's concat tables cannot skip nulls: each raises its ordinary
    # unsupported error, so the query runs on the CPU.
    var frame = DataFrame(
        [
            Series("s", Column[String](["a b", "c"])),
            Series("t", Column[String](["x", "y"])),
        ]
    )
    var queries = List[Expr]()
    for expression in _expressions():
        queries.append(expression.copy())
    queries.append(col("s").str().decode("hex", strict=False))
    queries.append(col("s").str().to_integer(strict=False))
    queries.append(
        concat_str([col("s"), col("t")], ignore_nulls=True).alias("c")
    )
    for expression in queries:
        var query = frame.lazy().select(expression.copy())
        var failed = False
        try:
            _ = lower_rows(query)
        except error:
            assert_true(String(error).find("unsupported") >= 0, String(error))
            failed = True
        assert_true(failed, "text functions must not lower to rows")
    var categorical = DataFrame(
        [
            frame.column("s").cast(DataType.CATEGORICAL),
            frame.column("t").cast(DataType.CATEGORICAL),
        ]
    )
    with assert_raises(contains="unsupported"):
        _ = lower_dictionary_operations(
            categorical.lazy().select(
                concat_str([col("s"), col("t")], ignore_nulls=True)
            )
        )
