"""String search and slices over raw bytes (#374): `starts_with`,
`ends_with`, `contains`, SQL `like` and `slice` must equal a row-by-row
reference on `String`s, for offsets and view storage, nulls, non-ASCII
text, empty needles and patterns, and enough rows for several workers.
"""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import DataFrame, Expr, Series, StringColumn, col
from dataframe.string_view import StringViewBuilder


def set_threads(n: Int):
    var name = String("DATAFRAME_THREADS")
    var value = String(n)
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


def pool() -> List[String]:
    return [
        "",
        "a",
        "special requests",
        "carefully special final requests sleep",
        "requests are special",
        "ünïcode spécial requëst",
        "abcabcabc",
        "MEDIUM POLISHED TIN",
        "PROMO BURNISHED COPPER",
        "13-555-0101",
        "a_b%c",
    ]


def values(rows: Int) -> List[String]:
    var p = pool()
    var out = List[String](capacity=rows)
    for i in range(rows):
        out.append(p[(i * 7 + i // 3) % len(p)])
    return out^


def codepoints(text: String) -> List[String]:
    var out = List[String]()
    for c in text.codepoints():
        out.append(String(c))
    return out^


def like_ref(text: List[String], t: Int, pattern: List[String], p: Int) -> Bool:
    if p == len(pattern):
        return t == len(text)
    if pattern[p] == "%":
        for k in range(t, len(text) + 1):
            if like_ref(text, k, pattern, p + 1):
                return True
        return False
    if t == len(text):
        return False
    if pattern[p] == "_" or pattern[p] == text[t]:
        return like_ref(text, t + 1, pattern, p + 1)
    return False


def slice_ref(text: String, offset: Int, length: Int) -> String:
    var parts = codepoints(text)
    var n = len(parts)
    var start = offset if offset >= 0 else max(n + offset, 0)
    start = min(start, n)
    var end = n if length < 0 else min(start + length, n)
    var out = String()
    for i in range(start, end):
        out += parts[i]
    return out^


def frame(rows: Int, views: Bool) raises -> DataFrame:
    var texts = values(rows)
    var valid = List[Bool](capacity=rows)
    for i in range(rows):
        valid.append(i % 13 != 5)
    if not views:
        return DataFrame([Series("s", StringColumn(texts, valid))])
    var builder = StringViewBuilder()
    for i in range(rows):
        if valid[i]:
            builder.append(StringSlice(texts[i]))
        else:
            builder.append_null()
    return DataFrame([Series("s", StringColumn(builder^.finish()))])


def check_bool(data: DataFrame, e: Expr, which: String, arg: String) raises:
    var got = data.select_exprs([e.copy().alias("r")]).column("r")
    var column = data.column("s")
    for i in range(data.height()):
        var cell = column.get(i)
        var label = which + "('" + arg + "') row " + String(i)
        if cell.is_null():
            assert_true(got.get(i).is_null(), label)
            continue
        var text = cell.string()
        var want: Bool
        if which == "starts_with":
            want = text.startswith(arg)
        elif which == "ends_with":
            want = text.endswith(arg)
        elif which == "contains":
            want = arg in text
        else:
            want = like_ref(codepoints(text), 0, codepoints(arg), 0)
        assert_equal(got.get(i).bool(), want, label)


def check_all(data: DataFrame) raises:
    var needles: List[String] = [
        "",
        "special",
        "requests",
        "s",
        "ü",
        "abcabc",
        "zzz",
    ]
    for needle in needles:
        check_bool(
            data, col("s").str().starts_with(needle), "starts_with", needle
        )
        check_bool(data, col("s").str().ends_with(needle), "ends_with", needle)
        check_bool(data, col("s").str().contains(needle), "contains", needle)
    var patterns: List[String] = [
        "%special%requests%",
        "special%",
        "%requests",
        "%",
        "",
        "a",
        "_",
        "%_c%",
        "MEDIUM POLISHED%",
        "%abc%abc",
        "13-%",
        "_n_code%",
        "%sp_cial%",
    ]
    for pattern in patterns:
        check_bool(data, col("s").str().like(pattern), "like", pattern)
    var slices: List[Tuple[Int, Int]] = [
        (0, 2),
        (0, 20),
        (3, -1),
        (-3, -1),
        (-30, 4),
        (40, 2),
        (2, 0),
    ]
    for s in slices:
        var got = data.select_exprs(
            [col("s").str().slice(s[0], s[1]).alias("r")]
        ).column("r")
        var column = data.column("s")
        for i in range(data.height()):
            var cell = column.get(i)
            if cell.is_null():
                assert_true(got.get(i).is_null())
            else:
                assert_equal(
                    got.get(i).string(), slice_ref(cell.string(), s[0], s[1])
                )


def test_offsets_storage() raises:
    set_threads(8)
    check_all(frame(2000, False))


def test_view_storage() raises:
    set_threads(8)
    check_all(frame(2000, True))


def test_many_rows_on_every_worker() raises:
    set_threads(8)
    var data = frame(150_000, False)
    check_bool(
        data, col("s").str().contains("requests"), "contains", "requests"
    )
    check_bool(
        data,
        col("s").str().like("%special%requests%"),
        "like",
        "%special%requests%",
    )
    var got = data.select_exprs([col("s").str().slice(0, 2).alias("r")]).column(
        "r"
    )
    for i in range(0, 150_000, 997):
        var cell = data.column("s").get(i)
        if not cell.is_null():
            assert_equal(got.get(i).string(), slice_ref(cell.string(), 0, 2))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
