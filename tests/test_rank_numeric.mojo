"""The numeric rank path (#330) against a plain reference ranking, on
inputs large enough to take the parallel paths: one partition of 70,000
rows (sorted in parallel runs and merged) and many small ones, with ties,
NaN, -0.0, nulls and both directions. The reference is a comparison sort
with Polars' rules stated directly: NaN above every number, -0.0 equal to
0.0, ties in row order."""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, Series, col
from dataframe.rank import rank_numeric


struct Rng:
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int(self.state >> 33)


def before(a: Float64, b: Float64, descending: Bool) -> Bool:
    """Whether a sorts strictly before b."""
    var a_nan = a != a
    var b_nan = b != b
    if a_nan or b_nan:
        return (b_nan and not a_nan) if not descending else (
            a_nan and not b_nan
        )
    return a < b if not descending else a > b


def reference(
    values: List[Float64],
    valid: List[Bool],
    ids: List[Int],
    method: String,
    descending: Bool,
) -> List[Float64]:
    """Ranks as Float64 (NaN marks null) by sorting each partition."""
    var n = len(values)
    var out = List[Float64](length=n, fill=Float64(0) / Float64(0))
    var count = 1
    for id in ids:
        count = max(count, id + 1)
    var groups = List[List[Int]](length=count, fill=List[Int]())
    for i in range(n):
        if valid[i]:
            groups[ids[i] if len(ids) > 0 else 0].append(i)
    for g in range(count):
        var rows = groups[g].copy()

        def less(a: Int, b: Int) {imm values, imm descending} -> Bool:
            if before(values[a], values[b], descending):
                return True
            if before(values[b], values[a], descending):
                return False
            return a < b

        sort(rows, less)
        var start = 0
        var dense = 0
        while start < len(rows):
            var end = start + 1
            while end < len(rows) and not before(
                values[rows[start]], values[rows[end]], descending
            ):
                end += 1
            dense += 1
            for p in range(start, end):
                var rank: Float64
                if method == "ordinal":
                    rank = Float64(p + 1)
                elif method == "min":
                    rank = Float64(start + 1)
                elif method == "max":
                    rank = Float64(end)
                elif method == "dense":
                    rank = Float64(dense)
                else:
                    rank = Float64(start + 1 + end) / 2
                out[rows[p]] = rank
            start = end
    return out^


def check(
    values: List[Float64], valid: List[Bool], ids: List[Int], label: String
) raises:
    var input = Series("v", Column[Float64](values.copy(), valid.copy()))
    for method in ["ordinal", "min", "max", "dense", "average"]:
        for descending in [False, True]:
            var got = (
                rank_numeric(input, ids, method, descending).value().copy()
            )
            var want = reference(values, valid, ids, method, descending)
            var name = label + " " + method + (" desc" if descending else "")
            for i in range(len(want)):
                var value = got.get(i)
                if want[i] != want[i]:
                    assert_true(value.is_null(), name + " row " + String(i))
                    continue
                var rank = value.float64() if method == "average" else Float64(
                    value.int64()
                )
                assert_equal(rank, want[i], name + " row " + String(i))


def test_matches_polars_on_nan_and_signed_zero() raises:
    # Polars 1.44: NaN is the largest value, so it ranks first descending.
    var v: List[Float64] = [1, Float64(0) / 0, -0.0, 0.0, 2, Float64(0) / 0, 1]
    var frame = DataFrame([Series("v", Column[Float64](v^))])
    var up = frame.select(col("v").rank("ordinal")).column("v")
    var down = frame.select(col("v").rank("ordinal", descending=True)).column(
        "v"
    )
    var expected_up: List[Int64] = [3, 6, 1, 2, 5, 7, 4]
    var expected_down: List[Int64] = [4, 1, 6, 7, 3, 2, 5]
    for i in range(7):
        assert_equal(up.get(i).int64(), expected_up[i])
        assert_equal(down.get(i).int64(), expected_down[i])


def floats(
    seed: UInt64, n: Int, groups: Int
) -> Tuple[List[Float64], List[Bool], List[Int]]:
    var rng = Rng(seed)
    var values = List[Float64](capacity=n)
    var valid = List[Bool](capacity=n)
    var ids = List[Int](capacity=n)
    for _ in range(n):
        var r = rng.next() % 1000
        var v = Float64(rng.next() % 5000) / 8 - 300
        if r == 0:
            v = Float64(0) / Float64(0)
        elif r == 1:
            v = -0.0
        elif r == 2:
            v = 0.0
        values.append(v)
        valid.append(r != 3 and r != 4)
        ids.append(rng.next() % groups)
    return (values^, valid^, ids^)


def test_one_large_partition() raises:
    # Over _LARGE rows: sorted in parallel runs, merged, ranked in parallel.
    var data = floats(3, 70_000, 1)
    check(data[0], data[1], List[Int](), "global")


def test_many_partitions_both_strategies() raises:
    # Counted and bucketed per worker (few partitions), then one sort with
    # the partition in the key (more partitions than rows allow).
    var few = floats(5, 70_000, 300)
    check(few[0], few[1], few[2], "bucketed")
    # Two workers need 131,072 rows; with more partition ids than twice
    # the rows (sparse ids, many partitions empty) per-worker counts cost
    # too much and the one-sort path runs.
    var many = floats(7, 140_000, 420_000)
    check(many[0], many[1], many[2], "one sort")


def test_edge_partitions() raises:
    # Partitions of one row, an all-null partition, and an empty input.
    var values: List[Float64] = [5, 1, 2, 2, 7, 3]
    var valid: List[Bool] = [True, False, True, True, False, True]
    var ids: List[Int] = [0, 1, 2, 2, 1, 3]
    check(values, valid, ids, "edges")
    check(List[Float64](), List[Bool](), List[Int](), "empty")


def test_integers_and_a_window() raises:
    var rng = Rng(9)
    var ints = List[Int64](capacity=20_000)
    for _ in range(20_000):
        ints.append(Int64(rng.next() % 300) - 150)
    var column = Column[Int64](ints^).slice(777, 15_000)
    var got = rank_numeric(Series("v", column.copy()), List[Int](), "min", True)
    var values = List[Float64](capacity=15_000)
    for i in range(15_000):
        values.append(Float64(column._get(i)))
    var want = reference(
        values, List[Bool](length=15_000, fill=True), List[Int](), "min", True
    )
    for i in range(15_000):
        assert_equal(Float64(got.value().get(i).int64()), want[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
