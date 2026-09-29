"""Binary columns (#120): arbitrary bytes in the string layout, tagged
DataType.BINARY so they are never read as text. Values include bytes that
are not UTF-8 (0xFF, an encoded lone surrogate, embedded NUL), which must
survive every operation unchanged. Expected behaviour follows Polars."""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import (
    ArrowArray,
    ArrowSchema,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    concat,
    export_arrow_series,
    import_arrow_series,
    to_csv_string,
    when,
)
from dataframe.arrow import _at, _leak, _read_c_string, _reclaim


def raw() -> List[List[UInt8]]:
    return [
        [0xFF, 0x00, 0x7A],  # not UTF-8, with a NUL
        [0x61, 0x62, 0x63],  # "abc"
        [],
        [0xED, 0xA0, 0x80],  # a UTF-16 surrogate encoded as UTF-8: invalid
        [0x61, 0x62, 0x63],
        [0x7A],  # "z"
    ]


def rows(indices: List[Int]) -> List[List[UInt8]]:
    var all = raw()
    var out = List[List[UInt8]]()
    for i in indices:
        out.append(all[i].copy())
    return out^


def valid() -> List[Bool]:
    return [True, True, True, True, True, False]


def sample() raises -> DataFrame:
    return DataFrame(
        [
            Series.binary("b", raw(), valid()),
            Series("n", StringColumn(["x", "y", "x", "y", "x", "y"])),
        ]
    )


def bytes_of(series: Series) raises -> List[List[UInt8]]:
    var out = List[List[UInt8]]()
    for i in range(len(series)):
        var value = series.get(i)
        out.append(List[UInt8]() if value.is_null() else value.bytes())
    return out^


def assert_binary(series: Series, expected: List[List[UInt8]]) raises:
    assert_true(series.dtype() == DataType.BINARY, series.dtype().name())
    assert_equal(len(series), len(expected))
    var got = bytes_of(series)
    for i in range(len(expected)):
        assert_equal(got[i], expected[i], "row " + String(i))


def test_dtype_and_values() raises:
    assert_equal(DataType.BINARY.name(), "binary")
    assert_true(DataType.parse("binary") == DataType.BINARY)
    assert_true(DataType.BINARY != DataType.STRING)
    assert_true(DataType.BINARY.physical() == DataType.STRING)
    var b = sample().column("b")
    assert_binary(b.slice(0, 5), rows([0, 1, 2, 3, 4]))
    assert_true(b.get(5).is_null())
    assert_equal(b.null_count(), 1)
    assert_equal(String(b.get(0)), 'b"\\xff\\x00z"')
    assert_equal(String(b.get(2)), 'b""')
    assert_true(b.get(1) == b.get(4))
    assert_true(not (b.get(0) == b.get(1)))
    with assert_raises(contains="Expected string value"):
        _ = b.get(1).string()
    # Binary is never retagged as text: that would skip the UTF-8 check.
    with assert_raises(contains="cast it instead"):
        _ = b.with_dtype(DataType.STRING)


def test_display_escapes_bytes() raises:
    var text = String(sample().select(col("b")))
    assert_true("binary" in text, text)
    assert_true('b"\\xff\\x00z"' in text, text)
    assert_true('b"\\xed\\xa0\\x80"' in text, text)
    assert_true('b"abc"' in text, text)


def test_row_operations_keep_bytes() raises:
    var frame = sample()
    var b = frame.column("b")
    assert_binary(b.take([3, 0, 0]), rows([3, 0, 0]))
    var picked = frame.filter(col("n") == "x").column("b")
    assert_binary(picked, rows([0, 2, 4]))
    var stacked = concat([frame.slice(0, 2), frame.slice(3, 2)]).column("b")
    assert_binary(stacked, rows([0, 1, 3, 4]))
    # Sorting compares bytes: b"" < b"abc" < b"\xed..." < b"\xff...".
    var sorted = frame.slice(0, 5).sort("b").column("b")
    assert_binary(sorted, rows([2, 1, 4, 3, 0]))
    var chunked = Series._from_chunks([b.slice(0, 3), b.slice(3, 3)])
    assert_true(chunked.dtype() == DataType.BINARY)
    assert_binary(chunked.rechunk().slice(0, 5), rows([0, 1, 2, 3, 4]))
    var distinct = frame.select(["b"]).unique(maintain_order=True).column("b")
    assert_equal(len(distinct), 5)
    assert_true(distinct.dtype() == DataType.BINARY)


def test_reductions() raises:
    var out = sample().select_exprs(
        [
            col("b").min().alias("min"),
            col("b").max().alias("max"),
            col("b").first().alias("first"),
            col("b").n_unique().alias("unique"),
            col("b").count().alias("count"),
        ]
    )
    assert_binary(out.column("min"), rows([2]))
    assert_binary(out.column("max"), rows([0]))
    assert_binary(out.column("first"), rows([0]))
    assert_equal(String(out.item(0, "unique")), "5")
    assert_equal(String(out.item(0, "count")), "5")
    with assert_raises():
        _ = sample().select(col("b").sum())


def test_grouping_and_joins_match_string_keys() raises:
    # Where the bytes are valid UTF-8, a binary key groups and joins exactly
    # as the same text in a string key does.
    var words: List[String] = ["a", "bb", "a", "", "bb", "a"]
    var strings = DataFrame(
        [
            Series("k", StringColumn(words)),
            Series("v", StringColumn(["1", "2", "3", "4", "5", "6"])),
        ]
    )
    var binary = strings.with_columns(col("k").cast("binary"))
    assert_true(binary.column("k").dtype() == DataType.BINARY)
    var by_text = strings.group_by("k", maintain_order=True).agg(
        col("v").count().alias("c")
    )
    var by_bytes = binary.group_by("k", maintain_order=True).agg(
        col("v").count().alias("c")
    )
    assert_true(by_bytes.column("k").dtype() == DataType.BINARY)
    assert_true(by_bytes.with_columns(col("k").cast("string")).equals(by_text))
    var right_text = DataFrame(
        [
            Series("k", StringColumn(["bb", "a", "zz"])),
            Series("r", StringColumn(["B", "A", "Z"])),
        ]
    )
    var right_bytes = right_text.with_columns(col("k").cast("binary"))
    var joined_text = strings.join(right_text, "k", "inner")
    var joined_bytes = binary.join(right_bytes, "k", "inner")
    assert_true(
        joined_bytes.with_columns(col("k").cast("string")).equals(joined_text)
    )
    with assert_raises(contains="Join key dtypes differ"):
        _ = binary.join(right_text, "k", "inner")


def test_grouped_windows_and_payloads_keep_the_tag() raises:
    var frame = sample()
    var grouped = frame.group_by("n", maintain_order=True).agg(
        [
            col("b").first().alias("first"),
            col("b").min().alias("min"),
            col("b").max().alias("max"),
            col("b").implode().alias("all"),
        ]
    )
    assert_binary(grouped.column("first"), rows([0, 1]))
    assert_binary(grouped.column("min"), rows([2, 1]))
    assert_binary(grouped.column("max"), rows([0, 3]))
    assert_true(grouped.column("all").dtype() == DataType.list(DataType.BINARY))
    assert_binary(frame.select(col("b").mode()).column("b"), rows([1]))
    var counts = frame.select(col("b").value_counts(sort=True))
    var first = counts.column("b").get(0)
    assert_true(String(first).startswith('{b: b"abc"'), String(first))
    var over = frame.select(col("b").first().over("n")).column("b")
    assert_binary(over.slice(0, 2), rows([0, 1]))
    # A join carries a binary payload through its gathers.
    var right = DataFrame(
        [
            Series("n", StringColumn(["y", "x"])),
            Series.binary("p", rows([3, 0])),
        ]
    )
    var joined = frame.join(right, "n", "left")
    assert_binary(joined.column("p"), rows([0, 3, 0, 3, 0, 3]))
    var filled = frame.select(col("b").fill_null(col("b").first())).column("b")
    assert_binary(filled, rows([0, 1, 2, 3, 4, 0]))
    var chosen = frame.select(
        when(col("n") == "x").then(col("b")).otherwise(col("b").last())
    ).column("b")
    assert_true(chosen.dtype() == DataType.BINARY)
    var lazy = frame.lazy().filter(col("n") == "y").select(col("b")).collect()
    assert_binary(lazy.column("b").slice(0, 2), rows([1, 3]))


def test_large_chunked_frames_match_string_results() raises:
    """Row counts that take the parallel and chunk-local paths: every result
    on binary keys equals the string result on the same text."""
    var parts = List[Series]()
    var values = List[Series]()
    for c in range(3):
        var texts = List[String]()
        var valid = List[Bool]()
        var numbers = List[String]()
        for i in range(1000):
            texts.append("k" + String((c * 1000 + i) * 7919 % 97))
            valid.append(i % 11 != 3)
            numbers.append(String(c * 1000 + i))
        parts.append(Series("k", StringColumn(texts, valid)))
        values.append(Series("v", StringColumn(numbers)))
    var binary_parts = List[Series]()
    for part in parts:
        binary_parts.append(part.cast("binary"))
    var text = DataFrame(
        [Series._from_chunks(parts^), Series._from_chunks(values.copy())]
    )
    var bytes = DataFrame(
        [Series._from_chunks(binary_parts^), Series._from_chunks(values^)]
    )
    assert_true(bytes.column("k").dtype() == DataType.BINARY)
    assert_true(bytes.column("k").is_chunked())

    def as_text(frame: DataFrame) raises -> DataFrame:
        return frame.with_columns(col("k").cast("string"))

    assert_true(
        as_text(bytes.filter(col("k") > "k5")).equals(
            text.filter(col("k") > "k5")
        )
    )
    assert_true(as_text(bytes.sort(["k", "v"])).equals(text.sort(["k", "v"])))
    assert_true(
        as_text(
            bytes.group_by("k", maintain_order=True).agg(
                col("v").count().alias("c")
            )
        ).equals(
            text.group_by("k", maintain_order=True).agg(
                col("v").count().alias("c")
            )
        )
    )
    var right_text = text.slice(500, 1500).unique(["k"], maintain_order=True)
    var right_bytes = right_text.with_columns(col("k").cast("binary"))
    var joined = bytes.join(right_bytes, "k", "inner")
    assert_true(joined.column("k").dtype() == DataType.BINARY)
    assert_true(as_text(joined).equals(text.join(right_text, "k", "inner")))
    assert_equal(
        bytes.select(["k"]).unique().height(),
        text.select(["k"]).unique().height(),
    )


def test_comparisons() raises:
    var frame = sample()
    var eq = frame.select(col("b") == "abc").column("b")
    assert_equal(String(eq.get(1)), "true")
    assert_equal(String(eq.get(0)), "false")
    assert_true(eq.get(5).is_null())
    var lt = frame.select(col("b") < "b").column("b")
    assert_equal(String(lt.get(2)), "true")
    assert_equal(String(lt.get(0)), "false")
    var same = frame.select(col("b") == col("b")).column("b")
    assert_equal(String(same.get(0)), "true")
    # Text operations need text.
    with assert_raises(contains="string"):
        _ = frame.select(col("b").str().len_bytes())


def test_casts() raises:
    var frame = sample()
    # Binary to string checks UTF-8: strict raises, otherwise null.
    with assert_raises(contains="not valid UTF-8"):
        _ = frame.select(col("b").cast("string"))
    var text = frame.select(col("b").cast("string", strict=False)).column("b")
    assert_true(text.dtype() == DataType.STRING)
    assert_true(text.get(0).is_null())
    assert_equal(text.get(1).string(), "abc")
    assert_equal(text.get(2).string(), "")
    assert_true(text.get(3).is_null())
    # Anything that casts to string casts to binary as its bytes.
    var numbers = DataFrame(
        [Series("i", StringColumn(["12", "-3"])).cast("int64")]
    ).select(col("i").cast("binary"))
    assert_binary(numbers.column("i"), [[0x31, 0x32], [0x2D, 0x33]])
    var accented = DataFrame([Series("s", StringColumn(["hé"]))]).select(
        col("s").cast("binary")
    )
    assert_binary(accented.column("s"), [[0x68, 0xC3, 0xA9]])
    with assert_raises(contains="cannot cast binary to int64"):
        _ = frame.select(col("b").cast("int64"))


def test_describe_and_csv() raises:
    var described = sample().select(col("b")).describe()
    var stats = described.column("b")
    assert_equal(stats.get(0).string(), "5")  # count
    var text = String(described)
    assert_true('b""' in text and 'b"\\xff\\x00z"' in text, text)
    with assert_raises(contains="CSV cannot hold binary columns"):
        _ = to_csv_string(sample())
    with assert_raises(contains="CSV cannot read binary columns"):
        _ = CsvSchema([CsvField("b", DataType.BINARY, True, "")])


def test_arrow_round_trip() raises:
    var series = sample().column("b")
    var array = _leak(ArrowArray())
    var schema = _leak(ArrowSchema())
    export_arrow_series(
        series, _at[ArrowArray](array)[], _at[ArrowSchema](schema)[]
    )
    assert_equal(_read_c_string(_at[ArrowSchema](schema)[].format), "Z")
    var back = import_arrow_series(array, schema)
    _ = _reclaim[ArrowArray](array)
    _ = _reclaim[ArrowSchema](schema)
    assert_true(back.equals(series))
    assert_binary(back.slice(0, 5), rows([0, 1, 2, 3, 4]))
    assert_true(back.get(5).is_null())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
