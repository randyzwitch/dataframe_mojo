"""Arrow string views are the in-memory layout of text: Parquet text arrives
as views (dictionary-coded text with its codes still attached), and joining
views with offset-stored strings, by appending in place or by rechunking
chunks, gives views. The kernels read both layouts through one accessor and must agree:
equality and inequality against a literal (values sharing a length and first
four bytes, inline and long), membership, prefixes, sorting, single-key
joins (inner, semi, anti) and grouping, with nulls."""
from std.testing import assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    Series,
    StringColumn,
    col,
    lit,
    read_parquet,
)
from dataframe.string_view import StringViewBuilder
from tests.support.parquet_runtime import Observer

comptime ROWS = 3_000


def is_views(series: Series) -> Bool:
    for part in series.chunks():
        if not part._data[StringColumn]._is_view_storage():
            return False
    return True


def test_parquet_text_arrives_as_views_with_its_codes() raises:
    var observer = Observer()
    for name in ["plain", "dictionary"]:
        var frame = read_parquet(
            "tests/fixtures/parquet/" + String(name) + ".parquet"
        )
        assert_true(is_views(frame.column("s")), name + " text is not views")
        assert_equal(frame.item(0, "s").string(), "red")
        assert_equal(frame.item(2, "s").string(), "blue")
    var coded = read_parquet("tests/fixtures/parquet/dictionary.parquet")
    assert_true(Bool(coded.column("s")._dictionary_codes()))
    _ = coded^
    observer.assert_balanced()


def word(row: Int) -> String:
    """Values that share lengths and first four bytes with each other,
    inline (up to twelve bytes) and long, empty and Unicode."""
    var kind = row % 7
    if kind == 0:
        return "abcd" + String(row % 5)
    if kind == 1:
        return "abcd-long-shared-prefix-" + String(row % 9)
    if kind == 2:
        return ""
    if kind == 3:
        return "é" + String(row % 4)
    if kind == 4:
        return "abcdefghijk" + String(row % 3)
    if kind == 5:
        return "xyz"
    return "abcd-long-shared-prefix-" + String((row + 3) % 9)


def strings(views: Bool, salt: Int) raises -> Series:
    var values = List[String]()
    var valid = List[Bool]()
    for row in range(ROWS):
        values.append(word(row * salt))
        valid.append((row * salt) % 23 != 9)
    if not views:
        return Series("s", StringColumn(values^, valid^))
    var builder = StringViewBuilder(ROWS)
    for row in range(ROWS):
        if valid[row]:
            builder.append(StringSlice(values[row]))
        else:
            builder.append_null()
    return Series("s", StringColumn(builder^.finish()))


def frame(views: Bool, salt: Int) raises -> DataFrame:
    var ids = List[Int64]()
    for row in range(ROWS):
        ids.append(Int64(row))
    return DataFrame([strings(views, salt), Series("id", Column[Int64](ids^))])


def test_mixed_appends_and_rechunks_give_views() raises:
    var offsets = strings(False, 1)
    var views = strings(True, 3)
    for pair in [(False, True), (True, False)]:
        var first = views.copy() if pair[0] else offsets.copy()
        var second = views.copy() if pair[1] else offsets.copy()
        # Batched results are assembled by appending in place.
        var appended = first.copy()
        appended._append_series(second)
        assert_true(is_views(appended), "an append did not give views")
        var chunked = Series._from_chunks([first.copy(), second.copy()])
        var merged = chunked.rechunk()
        assert_true(is_views(merged), "a rechunk did not give views")
        for r in [0, 1, 5, ROWS - 1, ROWS, ROWS + 7, 2 * ROWS - 1]:
            var expected = first.get(r) if r < ROWS else second.get(r - ROWS)
            assert_true(appended.get(r) == expected)
            assert_true(merged.get(r) == expected)


def ids(result: DataFrame) raises -> List[Int]:
    var out = List[Int]()
    for r in range(result.height()):
        out.append(Int(result.item(r, "id").int64()))
    return out^


def test_kernels_agree_on_offsets_and_views() raises:
    var offsets = frame(False, 1)
    var views = frame(True, 1)
    var literals: List[String] = [
        "abcd3",
        "abcd-long-shared-prefix-4",
        "",
        "abcdefghijk1",
        "nothing",
    ]
    for literal in literals:
        for ne in [False, True]:
            var predicate = (col("s") != lit(literal)) if ne else (
                col("s") == lit(literal)
            )
            var a = ids(offsets.filter(predicate))
            assert_equal(a, ids(views.filter(predicate)))
            # The model: valid rows equal (or not) to the literal.
            var expected = List[Int]()
            for row in range(ROWS):
                if row % 23 == 9:
                    continue
                if (word(row) == literal) != ne:
                    expected.append(row)
            assert_equal(a, expected)
    var members = col("s").is_in(["xyz", "abcd-long-shared-prefix-2", ""])
    assert_equal(ids(offsets.filter(members)), ids(views.filter(members)))
    var prefix = col("s").str().starts_with("abcd-long")
    assert_equal(ids(offsets.filter(prefix)), ids(views.filter(prefix)))
    assert_equal(ids(offsets.sort(["s", "id"])), ids(views.sort(["s", "id"])))
    for build_views in [False, True]:
        var build = frame(build_views, 3).select_exprs(
            [col("s"), col("id").alias("build_id")]
        )
        for how in ["inner", "semi", "anti"]:
            var a = offsets.join(build, on="s", how=how)
            var b = views.join(build, on="s", how=how)
            assert_equal(a.height(), b.height(), how)
            var sort_by: List[String] = ["id", "build_id"] if how == (
                "inner"
            ) else ["id"]
            assert_equal(ids(a.sort(sort_by)), ids(b.sort(sort_by)), how)
    var grouped_a = offsets.group_by("s").agg([col("id").sum().alias("t")])
    var grouped_b = views.group_by("s").agg([col("id").sum().alias("t")])
    assert_equal(grouped_a.height(), grouped_b.height())
    var sums = Dict[String, Int64]()
    for r in range(grouped_a.height()):
        var key = grouped_a.item(r, "s")
        sums["null" if key.is_null() else "v:" + key.string()] = grouped_a.item(
            r, "t"
        ).int64()
    for r in range(grouped_b.height()):
        var key = grouped_b.item(r, "s")
        var name = "null" if key.is_null() else "v:" + key.string()
        assert_equal(grouped_b.item(r, "t").int64(), sums[name])
