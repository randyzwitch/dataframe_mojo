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
        ascending.column("tag").string().to_list(),
        [String("b"), "d", "a", "c", "e"],
    )
    var descending = frame.sort(["k"], descending=True)
    assert_equal(
        descending.column("tag").string().to_list(),
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


def test_dense_rank_edges_match_reference() raises:
    """Column shapes where the rank pass has no values to sort, or nothing to
    distinguish. Dense ranks are built by sorting (value, row) pairs, so the
    cases that carry no pairs, or only equal ones, are the ones where a rank
    count can go wrong without any comparison being wrong."""
    var nan = Float64(0) / Float64(0)
    var frames = List[DataFrame]()
    # Every value null: no pairs at all.
    frames.append(
        DataFrame(
            [
                Series(
                    "i",
                    Column[Int64](
                        List[Int64](length=6, fill=0),
                        List[Bool](length=6, fill=False),
                    ),
                ),
                Series("f", Column[Float64](List[Float64](length=6, fill=1.5))),
            ]
        ).with_row_index()
    )
    # One distinct value repeated: every pair compares equal.
    frames.append(
        DataFrame(
            [
                Series("i", Column[Int64](List[Int64](length=6, fill=7))),
                Series("f", Column[Float64](List[Float64](length=6, fill=0.5))),
            ]
        ).with_row_index()
    )
    # NaN and null together, so both reserved ranks are in play at once.
    frames.append(
        DataFrame(
            [
                Series(
                    "i",
                    Column[Int64](
                        [1, 1, 2, 2, 3, 3],
                        [True, False, True, False, True, False],
                    ),
                ),
                Series(
                    "f",
                    Column[Float64](
                        [nan, 0.0, nan, -0.0, 1.0, nan],
                        [True, True, False, True, True, True],
                    ),
                ),
            ]
        ).with_row_index()
    )
    # A single row, and an empty frame.
    frames.append(
        DataFrame(
            [
                Series("i", Column[Int64]([5])),
                Series("f", Column[Float64]([2.5])),
            ]
        ).with_row_index()
    )
    frames.append(
        DataFrame(
            [
                Series("i", Column[Int64](List[Int64]())),
                Series("f", Column[Float64](List[Float64]())),
            ]
        ).with_row_index()
    )

    var keys: List[List[String]] = [["i"], ["f"], ["i", "f"], ["f", "i"]]
    for f in range(len(frames)):
        ref frame = frames[f]
        for by in keys:
            for d in range(2):
                for last in range(2):
                    var descending = List[Bool](length=len(by), fill=d == 1)
                    var nulls_last = List[Bool](length=len(by), fill=last == 1)
                    assert_equal(
                        frame.arg_sort(
                            by, descending=descending, nulls_last=nulls_last
                        ),
                        reference_order(frame, by, descending, nulls_last),
                        msg="frame " + String(f),
                    )


def test_medium_low_cardinality_first_key_matches_stable_order() raises:
    var keys = List[Int64]()
    var values = List[Float64]()
    var valid = List[Bool]()
    for i in range(8192):
        keys.append(Int64((i * 7) % 16))
        values.append(Float64((i * 13) % 32))
        valid.append(i % 11 != 0)
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("x", Column[Float64](values^, valid^)),
        ]
    )
    var expected = List[Int]()
    for key in range(16):
        for value in range(32):
            for row in range(frame.height()):
                if (
                    (row * 7) % 16 == key
                    and row % 11 != 0
                    and (row * 13) % 32 == value
                ):
                    expected.append(row)
        for row in range(frame.height()):
            if (row * 7) % 16 == key and row % 11 == 0:
                expected.append(row)
    assert_equal(frame.arg_sort(["k", "x"]), expected)


def test_large_bucket_radix_preserves_order_and_stability() raises:
    # Above 200k rows, low-cardinality first keys take the radix route.
    # Check every output position and source row without another sort.
    var count = 200_001
    var keys = List[Int64](capacity=count)
    var floats = List[Float64](capacity=count)
    var valid = List[Bool](capacity=count)
    var third = List[Int64](capacity=count)
    for i in range(count):
        keys.append(Int64((i * 17) % 16))
        floats.append(Float64((i * 131) % 1009) / 7.0 - 70.0)
        valid.append(i % 19 != 0)
        third.append(Int64((i * 23) % 41) - 20)
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("x", Column[Float64](floats^, valid^)),
            Series("z", Column[Int64](third^)),
        ]
    )
    var pairs: List[List[String]] = [["k", "x"], ["k", "x", "z"]]
    for by in pairs:
        var descending = List[Bool](length=len(by), fill=False)
        var nulls_last = List[Bool](length=len(by), fill=True)
        descending[1] = True
        nulls_last[1] = False
        var order = frame.arg_sort(
            by, descending=descending, nulls_last=nulls_last
        )
        if len(order) != count:
            raise Error("large radix sort changed row count")
        var seen = List[Bool](length=count, fill=False)
        var columns = List[Series]()
        for name in by:
            columns.append(frame.column(name))
        for position in range(count):
            var row = order[position]
            if row < 0 or row >= count or seen[row]:
                raise Error("large radix sort lost or duplicated a row")
            seen[row] = True
            if position == 0:
                continue
            var previous = order[position - 1]
            var strictly_before = False
            for k in range(len(by)):
                if columns[k]._less(
                    previous, row, descending[k], nulls_last[k]
                ):
                    strictly_before = True
                    break
                if columns[k]._less(
                    row, previous, descending[k], nulls_last[k]
                ):
                    raise Error("large radix sort put keys out of order")
            if not strictly_before and previous > row:
                raise Error("large radix sort broke stable row order")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
