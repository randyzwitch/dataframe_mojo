"""Filters a column stored in ascending order answers by binary search:
every answer must equal a reference that never takes that path, whatever
the data's order, nulls, chunks, operators and constants."""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, Expr, Series, col, lit, concat


def ordered(n: Int, step: Int, offset: Int = 0) -> List[Int64]:
    var values = List[Int64](capacity=n)
    for i in range(n):
        values.append(Int64(offset + (i * step) // 7))
    return values^


def frame(keys: List[Int64], valid: List[Bool]) raises -> DataFrame:
    var n = len(keys)
    var second = List[Int64](capacity=n)
    var other = List[Int64](capacity=n)
    for i in range(n):
        # A second key ordered within each run of the first.
        second.append(Int64(i % 7))
        other.append(Int64((i * 7919) % 13))
    return DataFrame(
        [
            Series("k", Column[Int64](keys.copy(), valid.copy())),
            Series("s", Column[Int64](second^)),
            Series("o", Column[Int64](other^)),
        ]
    )


def check(frame: DataFrame, predicate: Expr, label: String) raises:
    """The filter against a reference that never reaches the rewrite: the
    predicate evaluated as a column, then a filter on that column."""
    var want = (
        frame.with_columns([predicate.alias("keep")])
        .filter(col("keep"))
        .drop(["keep"])
    )
    assert_true(frame.filter(predicate).equals(want), label)
    var lazy = frame.lazy().filter(predicate).collect()
    assert_true(lazy.equals(want), label + " (lazy)")


def predicates() -> List[Expr]:
    var all = List[Expr]()
    all.append(col("k") == lit(Int64(40)))
    all.append(col("k") == lit(Int64(-5)))
    all.append(col("k") < lit(Int64(40)))
    all.append(col("k") <= lit(Int64(40)))
    all.append(col("k") > lit(Int64(10_000)))
    all.append(lit(Int64(40)) <= col("k"))
    all.append((col("k") >= lit(Int64(30))) & (col("k") <= lit(Int64(90))))
    all.append(
        (col("k") >= lit(Int64(30)))
        & (col("k") <= lit(Int64(90)))
        & (col("s") == lit(Int64(3)))
    )
    all.append(
        (col("k") == lit(Int64(50)))
        & (col("s") >= lit(Int64(2)))
        & (col("s") < lit(Int64(5)))
        & (col("o") != lit(Int64(4)))
    )
    all.append((col("k") > lit(Int64(90))) & (col("k") < lit(Int64(30))))
    all.append(col("k") == 40)
    all.append((col("k") >= lit(Int64.MIN)) & (col("k") < lit(Int64.MIN)))
    return all^


def test_ordered_unordered_and_null_keys() raises:
    var n = 50_000
    var valid = List[Bool](length=n, fill=True)
    var sorted = frame(ordered(n, 13), valid)
    var shuffled_keys = ordered(n, 13)
    for i in range(0, n - 1, 97):
        var t = shuffled_keys[i]
        shuffled_keys[i] = shuffled_keys[i + 1] + 1
        shuffled_keys[i + 1] = t
    var unsorted = frame(shuffled_keys, valid)
    var nulls = List[Bool](length=n, fill=True)
    for i in range(0, n, 1013):
        nulls[i] = False
    var with_nulls = frame(ordered(n, 13), nulls)
    var ps = predicates()
    for k in range(len(ps)):
        check(sorted, ps[k], "sorted " + String(k))
        check(unsorted, ps[k], "unsorted " + String(k))
        check(with_nulls, ps[k], "nulls " + String(k))
        check(sorted.slice(777, 30_000), ps[k], "slice " + String(k))


def test_chunks_and_rebuilt_buffers() raises:
    """Chunks each ascending, in order or not across their boundaries, and
    a column whose order was found before a concat rebuilt it."""
    var n = 20_000
    var valid = List[Bool](length=n, fill=True)
    var a = frame(ordered(n, 13), valid)
    var b = frame(ordered(n, 13, 400), valid)
    var c = frame(ordered(n, 13, 100), valid)
    var ascending = concat([a.copy(), b.copy()])
    var crossing = concat([b.copy(), c.copy()])
    var ps = predicates()
    for k in range(len(ps)):
        check(ascending, ps[k], "ascending chunks " + String(k))
        check(crossing, ps[k], "descending boundary " + String(k))
    # The order cell of `a` is filled by now; a frame built from it by
    # appending another must not inherit it.
    var appended = concat([a.copy(), c.copy()]).rechunk()
    for k in range(len(ps)):
        check(appended, ps[k], "rebuilt " + String(k))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
