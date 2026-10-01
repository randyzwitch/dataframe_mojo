"""String comparisons with a literal and string `is_in` run over raw bytes on
every worker (#373). Each result must equal a row-by-row comparison of
`String`s: for every operator, the literal on either side, nulls, empty and
non-ASCII strings, view storage, sliced columns whose validity starts
mid-byte, and enough rows for several workers.
"""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    Expr,
    Series,
    StringColumn,
    col,
    lit,
    null,
)
from dataframe.string_view import StringViewBuilder


def set_threads(n: Int):
    var name = String("DATAFRAME_THREADS")
    var value = String(n)
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


struct Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self, bound: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(bound))


def words() -> List[String]:
    return [
        "",
        "a",
        "ab",
        "abc",
        "Brand#23",
        "Brand#2",
        "Brand#230",
        "MED BOX",
        "MED BOXES",
        "ünïcode",
        "unicode",
        "zzzzzzzzzzzzzzzzzz",
        "zzzzzzzzzzzzzzzzzy",
        "eightchr",
        "eightchrs",
    ]


def frame(rows: Int, seed: UInt64) raises -> DataFrame:
    var rng = Lcg(seed)
    var pool = words()
    var values = List[String](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for _ in range(rows):
        values.append(pool[rng.next(len(pool))])
        valid.append(rng.next(13) != 0)
    return DataFrame([Series("s", StringColumn(values, valid))])


def expected(
    data: DataFrame, op: String, literal: String, left: Bool
) raises -> List[Int]:
    """Per row: 1 true, 0 false, -1 null, by comparing Strings."""
    var out = List[Int]()
    var column = data.column("s")
    for i in range(data.height()):
        var cell = column.get(i)
        if cell.is_null():
            out.append(-1)
            continue
        var a = cell.string()
        var x = literal if left else a
        var y = a if left else literal
        var result: Bool
        if op == "==":
            result = x == y
        elif op == "!=":
            result = x != y
        elif op == "<":
            result = x < y
        elif op == "<=":
            result = x <= y
        elif op == ">":
            result = x > y
        else:
            result = x >= y
        out.append(1 if result else 0)
    return out^


def check(data: DataFrame, e: Expr, want: List[Int], label: String) raises:
    var got = data.select_exprs([e.copy().alias("r")]).column("r")
    assert_equal(len(got), len(want), label)
    for i in range(len(want)):
        var cell = got.get(i)
        if want[i] < 0:
            assert_true(cell.is_null(), label + " row " + String(i))
        else:
            assert_true(not cell.is_null(), label + " row " + String(i))
            assert_equal(cell.bool(), want[i] == 1, label + " row " + String(i))


def compare(e: Expr, op: String, literal: String, left: Bool) -> Expr:
    var x = lit(literal) if left else e.copy()
    var y = e.copy() if left else lit(literal)
    if op == "==":
        return x == y
    if op == "!=":
        return x != y
    if op == "<":
        return x < y
    if op == "<=":
        return x <= y
    if op == ">":
        return x > y
    return x >= y


def check_all(data: DataFrame, label: String) raises:
    var ops: List[String] = ["==", "!=", "<", "<=", ">", ">="]
    var literals: List[String] = [
        "Brand#23",
        "",
        "zzzzzzzzzzzzzzzzzz",
        "ü",
        "eightchr",
    ]
    for op in ops:
        for literal in literals:
            for left in [False, True]:
                check(
                    data,
                    compare(col("s"), op, literal, left),
                    expected(data, op, literal, left),
                    label
                    + " "
                    + op
                    + " '"
                    + literal
                    + "'"
                    + (" left" if left else ""),
                )
    var pool = words()
    for count in [0, 2, 7, 25]:
        var values = List[String]()
        for k in range(count):
            values.append(
                pool[(k * 4) % len(pool)] + ("" if k < len(pool) else "x")
            )
        var want = List[Int]()
        var column = data.column("s")
        for i in range(data.height()):
            var cell = column.get(i)
            if cell.is_null():
                want.append(-1)
                continue
            var found = False
            for v in values:
                found = found or cell.string() == v
            want.append(1 if found else 0)
        check(
            data,
            col("s").is_in(values),
            want,
            label + " is_in " + String(count),
        )
    # A null among the values never matches.
    var with_null: List[Expr] = [lit("abc"), null("string")]
    var want = expected(data, "==", "abc", False)
    check(data, col("s").is_in(with_null), want, label + " is_in with null")


def test_small_and_sliced() raises:
    set_threads(8)
    var data = frame(1000, 3)
    check_all(data, "small")
    # A window starting mid-byte: validity must be re-aligned to row 0.
    check_all(data.slice(5, 600), "sliced")


def test_many_rows_on_every_worker() raises:
    set_threads(8)
    check_all(frame(150_000, 7), "parallel")
    set_threads(1)
    check_all(frame(150_000, 7), "serial")
    set_threads(8)


def test_view_storage() raises:
    set_threads(8)
    var plain = frame(70_000, 11)
    var builder = StringViewBuilder()
    var column = plain.column("s")
    for i in range(plain.height()):
        var cell = column.get(i)
        if cell.is_null():
            builder.append_null()
        else:
            builder.append(StringSlice(cell.string()))
    var views = DataFrame([Series("s", StringColumn(builder^.finish()))])
    check_all(views, "views")


def test_numbers_keep_their_is_in() raises:
    var data = DataFrame([Series("x", Column[Int64]([1, 2, 3, 4]))])
    var out = data.select_exprs(
        [col("x").is_in([lit(Int64(2)), lit(Int64(4))])]
    )
    assert_equal(out.column("x").get(1).bool(), True)
    assert_equal(out.column("x").get(0).bool(), False)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
