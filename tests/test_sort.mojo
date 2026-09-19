"""Multi-column sorting, arg_sort, and top_k/bottom_k against references."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import Column, DataFrame, Series


struct Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self, bound: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(bound))


def random_frame(rows: Int, seed: UInt64) raises -> DataFrame:
    var rng = Lcg(seed)
    var nan = Float64(0) / Float64(0)
    var ints = List[Int64]()
    var iv = List[Bool]()
    var floats = List[Float64]()
    var fv = List[Bool]()
    var strings = List[String]()
    var sv = List[Bool]()
    var bools = List[Bool]()
    var bv = List[Bool]()
    var words: List[String] = ["", "a", "ab", "b", "Z", "é", "a "]
    for _ in range(rows):
        ints.append(Int64(rng.next(5)) - 2)
        iv.append(rng.next(6) != 0)
        var pick = rng.next(8)
        floats.append(
            nan if pick == 0 else (-0.0 if pick == 1 else Float64(pick) / 2 - 1)
        )
        fv.append(rng.next(7) != 0)
        strings.append(words[rng.next(len(words))])
        sv.append(rng.next(5) != 0)
        bools.append(rng.next(2) == 0)
        bv.append(rng.next(4) != 0)
    return DataFrame(
        [
            Series("i", Column[Int64](ints^, iv^)),
            Series("f", Column[Float64](floats^, fv^)),
            Series("s", Column[String](strings^, sv^)),
            Series("b", Column[Bool](bools^, bv^)),
        ]
    ).with_row_index()


def reference_order(
    frame: DataFrame,
    by: List[String],
    descending: List[Bool],
    nulls_last: List[Bool],
) raises -> List[Int]:
    """Stable insertion sort using the original per-comparison comparator."""
    var columns = List[Series]()
    for name in by:
        columns.append(frame.column(name))
    var order = List[Int]()
    for row in range(frame.height()):
        var position = len(order)
        while position > 0:
            var previous = order[position - 1]
            var before = False
            for k in range(len(by)):
                ref column = columns[k]
                if column._less(row, previous, descending[k], nulls_last[k]):
                    before = True
                    break
                if column._less(previous, row, descending[k], nulls_last[k]):
                    break
            if not before:
                break
            position -= 1
        order.insert(position, row)
    return order^


def test_single_key_matches_reference_comparator() raises:
    var frame = random_frame(200, 7)
    var names: List[String] = ["i", "f", "s", "b", "index"]
    for name in names:
        for d in range(2):
            for last in range(2):
                var column = frame.column(name)
                assert_equal(
                    column.argsort(d == 1, last == 1),
                    column._argsort_reference(d == 1, last == 1),
                    msg=name,
                )


def test_multi_key_matches_reference() raises:
    for seed in range(1, 4):
        var frame = random_frame(40, UInt64(seed))
        var keys: List[List[String]] = [
            ["i", "f"],
            ["s", "b", "i"],
            ["b", "s"],
            ["f", "s", "b", "i"],
        ]
        for by in keys:
            for pattern in range(1 << len(by)):
                var descending = List[Bool]()
                var nulls_last = List[Bool]()
                for k in range(len(by)):
                    descending.append((pattern >> k) & 1 == 1)
                    nulls_last.append((pattern >> k) & 1 == 0 or k == 0)
                var expected = reference_order(
                    frame, by, descending, nulls_last
                )
                var actual = frame.arg_sort(
                    by, descending=descending, nulls_last=nulls_last
                )
                assert_equal(actual, expected)
                var sorted = frame.sort(
                    by, descending=descending, nulls_last=nulls_last
                )
                assert_true(sorted.equals(frame.take(expected)))


def test_uniform_direction_overload_and_stability() raises:
    var frame = DataFrame(
        [
            Series("k", Column[Int64]([2, 1, 2, 1, 2])),
            Series("tag", Column[String](["a", "b", "c", "d", "e"])),
        ]
    )
    var ascending = frame.sort(["k"])
    assert_equal(
        ascending.column("tag").string()._to_list(),
        [String("b"), "d", "a", "c", "e"],
    )
    var descending = frame.sort(["k"], descending=True)
    assert_equal(
        descending.column("tag").string()._to_list(),
        [String("a"), "c", "e", "b", "d"],
    )
    assert_true(frame.sort("k").equals(ascending))
    assert_equal(
        frame.sort(["tag", "k"], True).column("tag").string().value(0), "e"
    )
    # Sorting an already sorted frame is the identity.
    assert_true(ascending.sort(["k"]).equals(ascending))
    assert_equal(frame.clear().sort(["k"]).height(), 0)


def test_top_and_bottom_k_equal_sorted_heads() raises:
    var frame = random_frame(150, 42)
    var keys: List[List[String]] = [["f"], ["s", "i"], ["b", "f", "index"]]
    for by in keys:
        var n = len(by)
        var top_order = frame.sort(
            by,
            descending=List[Bool](length=n, fill=True),
            nulls_last=List[Bool](length=n, fill=True),
        )
        var bottom_order = frame.sort(by)
        for k in [0, 1, 2, 7, 64, 150, 400]:
            assert_true(frame.top_k(k, by).equals(top_order.head(k)))
            assert_true(frame.bottom_k(k, by).equals(bottom_order.head(k)))
    assert_equal(frame.top_k(3, "index").column("index").int64().value(0), 149)
    with assert_raises(contains="k must be nonnegative"):
        _ = frame.top_k(-1, "f")


def test_sort_errors() raises:
    var frame = random_frame(5, 1)
    with assert_raises(contains="at least one column"):
        _ = frame.sort(List[String]())
    with assert_raises(contains="one entry per sort column"):
        _ = frame.sort(["i", "f"], descending=[True], nulls_last=[True, True])
    with assert_raises(contains="Unknown column"):
        _ = frame.sort(["i", "missing"])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
