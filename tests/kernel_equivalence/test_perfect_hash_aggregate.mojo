"""Perfect hash group-by (integer keys over a small span) against the same
grouping by string copies of the keys, which takes the hashed routes."""
# contract-group: kernel.reduce
from std.testing import assert_equal, assert_false, assert_true
from dataframe import AnyValue, Column, DataFrame, DataType, Expr, Series, col
from dataframe.categorical import encode
from dataframe.perfect_aggregate import perfect_layout


def _frame(n: Int, span: Int, chunked: Bool) raises -> DataFrame:
    """Int64 key `a` (every 11th row null, values from -3), Int32 key `b`
    over [-span/2, span/2) with every 13th row null, Int64 `v` and Float64
    `f` with scattered nulls, and string copies `sa` and `sb` of the keys."""
    var a = List[Int64]()
    var a_valid = List[Bool]()
    var b = List[Int32]()
    var b_valid = List[Bool]()
    var v = List[Int64]()
    var v_valid = List[Bool]()
    var f = List[Float64]()
    var f_valid = List[Bool]()
    for row in range(n):
        a.append(Int64((row * 7 + row // 3) % 9 - 3))
        a_valid.append(row % 11 != 4)
        b.append(Int32((row * 5 + row // 7) % span - span // 2))
        b_valid.append(row % 13 != 6)
        v.append(Int64(row % 97) - 40)
        v_valid.append(row % 17 != 3)
        # Quarters are exact in binary, so sums agree in any order.
        f.append(Float64(row % 23) / 4 - 2)
        f_valid.append(row % 19 != 5)
    var columns: List[Series] = [
        Series("a", Column[Int64](a^, a_valid)),
        Series("b", Column[Int32](b^, b_valid)),
        Series("v", Column[Int64](v^, v_valid)),
        Series("f", Column[Float64](f^, f_valid)),
    ]
    columns.append(columns[0].cast(DataType.STRING).renamed("sa"))
    columns.append(columns[1].cast(DataType.STRING).renamed("sb"))
    if chunked and n > 2:
        var cut = n // 3
        for c in range(len(columns)):
            columns[c] = Series._from_chunks(
                [columns[c].slice(0, cut), columns[c].slice(cut, n - cut)]
            )
    return DataFrame(columns^)


def _aggregates() -> List[Expr]:
    return [
        col("v").sum().alias("sum"),
        col("v").mean().alias("mean"),
        col("f").sum().alias("fsum"),
        col("f").mean().alias("fmean"),
        col("v").count().alias("count"),
        col("v").len().alias("len"),
        col("v").min().alias("min"),
        col("f").max().alias("max"),
        col("v").first().alias("first"),
        col("f").last().alias("last"),
        col("f").std().alias("std"),
        col("v").n_unique().alias("distinct"),
    ]


def _assert_close(actual: AnyValue, expected: AnyValue) raises:
    assert_true(actual.dtype() == expected.dtype())
    assert_equal(actual.is_null(), expected.is_null())
    if expected.is_null():
        return
    if expected.dtype() == DataType.FLOAT64:
        var x = actual.numeric[DType.float64]()
        var y = expected.numeric[DType.float64]()
        assert_true(abs(x - y) <= 1e-9 * max(Float64(1), abs(y)))
    else:
        assert_true(actual == expected)


def test_perfect_hash_group_by_matches_string_keys() raises:
    # 140K rows take two workers (`parallel.worker_count`), whose slots
    # merge; once, as the string grouping is slow in a test build.
    for n in [0, 1, 9, 65, 3000, 140_000]:
        for chunked in [False, True]:
            var large = n > 100_000
            if large and not chunked:
                continue
            var frame = _frame(n, 40, chunked)
            var key_sets: List[List[String]] = [["a", "b"]]
            if not large:
                key_sets = [["a"], ["b"], ["a", "b"]]
            for keys in key_sets:
                var strings = List[String]()
                for key in keys:
                    strings.append("s" + key)
                var actual = frame.group_by(keys, maintain_order=True).agg(
                    _aggregates()
                )
                var expected = frame.group_by(strings, maintain_order=True).agg(
                    _aggregates()
                )
                assert_equal(actual.height(), expected.height())
                for k in range(len(keys)):
                    assert_true(
                        actual.column(keys[k]).dtype()
                        == frame.column(keys[k]).dtype()
                    )
                    var as_text = DataFrame(
                        [actual.column(keys[k]).cast(DataType.STRING)]
                    )
                    for row in range(actual.height()):
                        _assert_close(
                            as_text.item(row, keys[k]),
                            expected.item(row, strings[k]),
                        )
                for name in expected.columns():
                    if name in strings:
                        continue
                    for row in range(expected.height()):
                        _assert_close(
                            actual.item(row, name), expected.item(row, name)
                        )


def test_perfect_hash_layout_follows_key_span() raises:
    # 100 values and a null need 7 bits; two such keys need 14, more than
    # DuckDB's 12-bit perfect_ht_threshold.
    var narrow = _frame(500, 100, False)
    var one = perfect_layout([narrow.column("b")])
    assert_true(Bool(one))
    assert_equal(one.value().bits, 7)
    assert_false(Bool(perfect_layout([narrow.column("b"), narrow.column("b")])))
    # A key spanning more than 2^12 values falls back, and still groups.
    var wide = _frame(20_000, 10_000, False)
    assert_false(Bool(perfect_layout([wide.column("b")])))
    var grouped = wide.group_by("b", maintain_order=True).agg(
        [col("v").sum().alias("sum")]
    )
    var expected = wide.group_by("sb", maintain_order=True).agg(
        [col("v").sum().alias("sum")]
    )
    assert_equal(grouped.height(), expected.height())
    for row in range(expected.height()):
        _assert_close(grouped.item(row, "sum"), expected.item(row, "sum"))
    # A float or string key never qualifies.
    assert_false(Bool(perfect_layout([narrow.column("f")])))
    assert_false(Bool(perfect_layout([narrow.column("sa")])))


def test_categorical_keys_group_by_their_codes() raises:
    # A categorical with a small dictionary (what a dictionary-coded string
    # column's codes are) gives each row its slot as an integer key does; a
    # null code lands in slot 0. Results match grouping by the strings.
    for n in [0, 1, 9, 3000]:
        var frame = _frame(n, 40, False)
        var columns = frame._columns.copy()
        columns.append(encode(frame.column("sa")).renamed("ca"))
        columns.append(encode(frame.column("sb")).renamed("cb"))
        var coded = DataFrame(columns^)
        if n > 0:
            assert_true(Bool(perfect_layout([coded.column("ca")])))
            assert_true(
                Bool(perfect_layout([coded.column("ca"), coded.column("cb")]))
            )
        var key_sets: List[List[String]] = [["a"], ["b"], ["a", "b"]]
        for keys in key_sets:
            var categories = List[String]()
            var strings = List[String]()
            for key in keys:
                categories.append("c" + key)
                strings.append("s" + key)
            var actual = coded.group_by(categories, maintain_order=True).agg(
                _aggregates()
            )
            var expected = coded.group_by(strings, maintain_order=True).agg(
                _aggregates()
            )
            assert_equal(actual.height(), expected.height())
            for k in range(len(keys)):
                var as_text = DataFrame(
                    [actual.column(categories[k]).cast(DataType.STRING)]
                )
                for row in range(actual.height()):
                    _assert_close(
                        as_text.item(row, categories[k]),
                        expected.item(row, strings[k]),
                    )
            for name in expected.columns():
                if name in strings:
                    continue
                for row in range(expected.height()):
                    _assert_close(
                        actual.item(row, name), expected.item(row, name)
                    )
