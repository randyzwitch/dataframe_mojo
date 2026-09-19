"""Deterministic, bounded text rendering of frames and series."""
from std.testing import TestSuite, assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, Series, col, lit


def small() raises -> DataFrame:
    var nan = Float64(0) / Float64(0)
    return DataFrame(
        [
            Series("id", Column[Int64]([1, -2, 3])),
            Series("x", Column[Float64]([1.5, nan, -0.0], [True, True, False])),
            Series(
                "ok", Column[Bool]([True, False, True], [True, False, True])
            ),
            Series("name", Column[String](["héllo", "", " pad"])),
        ]
    )


def numbered(rows: Int) raises -> DataFrame:
    var values = List[Int64]()
    for i in range(rows):
        values.append(Int64(i))
    return DataFrame([Series("n", Column[Int64](values^))])


def test_frame_layout() raises:
    var expected = String(
        "shape: (3, 4)\n",
        "┌─────┬──────┬──────┬────────┐\n",
        "│ id  ┆ x    ┆ ok   ┆ name   │\n",
        "│ --- ┆ ---  ┆ ---  ┆ ---    │\n",
        "│ i64 ┆ f64  ┆ bool ┆ str    │\n",
        "╞═════╪══════╪══════╪════════╡\n",
        "│ 1   ┆ 1.5  ┆ true ┆ héllo  │\n",
        '│ -2  ┆ nan  ┆ null ┆ ""     │\n',
        '│ 3   ┆ null ┆ true ┆ " pad" │\n',
        "└─────┴──────┴──────┴────────┘",
    )
    assert_equal(String(small()), expected)
    assert_equal(small().to_string(), expected)


def test_empty_shapes() raises:
    assert_equal(String(DataFrame([], height=3)), "shape: (3, 0)\n┌┐\n└┘")
    var cleared = String(small().clear())
    assert_true(cleared.startswith("shape: (0, 4)\n"))
    assert_true(cleared.endswith("╡\n└─────┴─────┴──────┴──────┘"))


def test_row_truncation() raises:
    var text = numbered(25).to_string(max_rows=4)
    var expected = String(
        "shape: (25, 1)\n",
        "┌─────┐\n",
        "│ n   │\n",
        "│ --- │\n",
        "│ i64 │\n",
        "╞═════╡\n",
        "│ 0   │\n",
        "│ 1   │\n",
        "│ …   │\n",
        "│ 23  │\n",
        "│ 24  │\n",
        "└─────┘",
    )
    assert_equal(text, expected)
    # Odd limits show one extra row at the front.
    var odd = numbered(25).to_string(max_rows=3)
    assert_true("│ 1   │" in odd)
    assert_false("│ 23  │" in odd)
    assert_true("│ 24  │" in odd)
    # Exactly at the limit: no ellipsis. Negative: unlimited.
    assert_false("…" in numbered(10).to_string(max_rows=10))
    assert_true("│ 999 │" in numbered(1000).to_string(max_rows=-1))
    assert_true("│ …   │" in String(numbered(11)))


def test_column_truncation() raises:
    var columns = List[Series]()
    for i in range(6):
        columns.append(Series("c" + String(i), Column[Int64]([Int64(i)])))
    var text = DataFrame(columns^).to_string(max_columns=4)
    assert_true("│ c0  ┆ c1  ┆ …   ┆ c4  ┆ c5  │" in text)
    assert_true("│ 0   ┆ 1   ┆ …   ┆ 4   ┆ 5   │" in text)
    assert_true(text.startswith("shape: (1, 6)"))


def test_strings_and_quoting() raises:
    var frame = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    ["abcdefghij", "null", "tab\there", "x", "日本語テキスト"],
                    [True, True, True, False, True],
                ),
            )
        ]
    )
    var text = frame.to_string(max_string_length=5)
    assert_true("│ abcd… │" in text)
    assert_true('│ "nul… │' in text)
    assert_true('│ "tab… │' in text)
    assert_true("│ null  │" in text)
    assert_true("│ 日本語テ… │" in text)
    var wide = frame.to_string()
    assert_true('│ "null"      │' in wide)
    assert_true('│ "tab\\there" │' in wide)
    # Alignment counts code points; East Asian display width is not computed.
    assert_true("│ 日本語テキスト     │" in wide)


def test_series_rendering() raises:
    var series = Series(
        "v", Column[Float64]([1, 2.25, 1e300], [True, False, True])
    )
    assert_equal(
        String(series),
        "shape: (3,)\nSeries: v [f64]\n[\n\t1.0\n\tnull\n\t1e+300\n]",
    )
    var long = numbered(30).column("n").to_string(max_rows=2)
    assert_equal(long, "shape: (30,)\nSeries: n [i64]\n[\n\t0\n\t…\n\t29\n]")
    assert_equal(
        String(Series("", Column[Bool]([]))),
        'shape: (0,)\nSeries: "" [bool]\n[\n]',
    )


def test_glimpse() raises:
    var text = small().glimpse()
    var expected = String(
        "Rows: 3\n",
        "Columns: 4\n",
        "$ id   <i64> 1, -2, 3\n",
        "$ x    <f64> 1.5, nan, null\n",
        "$ ok   <bool> true, null, true\n",
        '$ name <str> héllo, "", " pad"',
    )
    assert_equal(text, expected)
    var narrow = numbered(100000).glimpse(max_width=30)
    assert_equal(
        narrow, "Rows: 100000\nColumns: 1\n$ n <i64> 0, 1, 2, 3, 4, 5, …"
    )
    for line in narrow.split("\n"):
        assert_true(len(line.codepoints()) <= 30)


def test_rendering_is_independent_of_batch_size() raises:
    var frame = numbered(40)
    var a = frame.with_columns(
        (col("n") * lit(Int64(3))).alias("m"), batch_size=1
    )
    var b = frame.with_columns(
        (col("n") * lit(Int64(3))).alias("m"), batch_size=7
    )
    assert_equal(String(a), String(b))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
