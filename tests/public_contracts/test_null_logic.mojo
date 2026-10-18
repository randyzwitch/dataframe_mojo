"""Truth tables and selected-branch evaluation, with poison hidden under nulls."""
# contract-group: expr.arithmetic
# contract-group: expr.null_logic
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit, when, coalesce, null
from tests.support.coverage_generators import chunk_layout


def test_boolean_truth_tables_for_every_validity_pair() raises:
    # -1 means null. Tables enumerate left-major (false, true, null).
    var states: List[Int] = [0, 1, -1]
    var and_table: List[Int] = [0, 0, 0, 0, 1, -1, 0, -1, -1]
    var or_table: List[Int] = [0, 1, -1, 1, 1, 1, -1, 1, -1]
    var xor_table: List[Int] = [0, 1, -1, 1, 0, -1, -1, -1, -1]
    var a = List[Bool]()
    var b = List[Bool]()
    var av = List[Bool]()
    var bv = List[Bool]()
    for left in states:
        for right in states:
            a.append(left != 0)
            b.append(right != 0)
            av.append(left >= 0)
            bv.append(right >= 0)
    var input = DataFrame(
        [Series("a", Column[Bool](a^, av^)), Series("b", Column[Bool](b^, bv^))]
    )
    for layout in range(2):
        var frame = input.copy() if layout == 0 else chunk_layout(input)
        var out = frame.select_exprs(
            [
                (col("a") & col("b")).alias("and"),
                (col("a") | col("b")).alias("or"),
                (col("a") ^ col("b")).alias("xor"),
                (~col("a")).alias("not"),
                coalesce([col("a"), col("b"), lit(False)]).alias("coalesce"),
            ]
        )
        assert_equal(out.columns(), ["and", "or", "xor", "not", "coalesce"])
        for row in range(9):
            var expected: List[Int] = [
                and_table[row],
                or_table[row],
                xor_table[row],
                -1 if states[row // 3] < 0 else 1 - states[row // 3],
                states[row // 3] if states[row // 3]
                >= 0 else states[row % 3] if states[row % 3]
                >= 0 else 0,
            ]
            for c in range(len(expected)):
                var cell = out.item(row, out.columns()[c])
                assert_equal(cell.is_null(), expected[c] < 0)
                if expected[c] >= 0:
                    assert_equal(cell.bool(), expected[c] == 1)
        assert_equal(frame.filter(col("a") & col("b")).height(), 1)
        assert_equal(frame.filter(col("a") | col("b")).height(), 5)


def test_nan_null_and_conditional_short_circuit_are_distinct() raises:
    var nan = Float64(0) / Float64(0)
    var frame = DataFrame(
        [
            Series(
                "f",
                Column[Float64]([nan, nan, 0, -0.0], [True, False, True, True]),
            ),
            Series(
                "i",
                Column[Int64](
                    [Int64.MAX, Int64.MAX, -2, 2], [False, True, True, True]
                ),
            ),
        ]
    )
    var out = frame.select_exprs(
        [
            col("f").is_null().alias("null"),
            col("f").is_nan().alias("nan"),
            col("f").fill_null(7.0).alias("filled"),
            when(col("i") < 0)
            .then(col("i") + lit(Int64.MAX))
            .otherwise(col("i"))
            .alias("safe"),
            coalesce([col("i"), lit(Int64(3))]).alias("coalesced"),
        ]
    )
    for row in range(4):
        assert_equal(out.item(row, "null").bool(), row == 1)
        assert_equal(out.item(row, "nan").is_null(), row == 1)
        if row != 1:
            assert_equal(out.item(row, "nan").bool(), row == 0)
    assert_equal(out.item(1, "filled").float64(), Float64(7))
    assert_true(
        out.item(0, "filled").float64() != out.item(0, "filled").float64()
    )
    assert_true(out.item(0, "safe").is_null())
    assert_equal(out.item(1, "safe").int64(), Int64.MAX)
    assert_equal(out.item(2, "safe").int64(), Int64.MAX - 2)
    assert_equal(out.item(3, "safe").int64(), Int64(2))
    assert_equal(out.item(0, "coalesced").int64(), Int64(3))
    # A null payload must not overflow; selecting a valid poison does.
    var nulls = frame.take([0])
    assert_true(nulls.select(col("i") + 1).item().is_null())
    with assert_raises():
        _ = frame.take([1]).select(col("i") + 1)
    for empty in [frame.copy(), frame.clear()]:
        with assert_raises():
            _ = empty.select(col("i") & lit(True))
        with assert_raises():
            _ = empty.select(
                when(lit(True)).then(col("missing")).otherwise(col("i"))
            )


def test_conditionals_broadcast_fallthrough_and_binding_errors() raises:
    var frame = DataFrame(
        [
            Series("b", Column[Bool]([True, False, True], [True, True, False])),
            Series(
                "i", Column[Int64]([10, 20, Int64.MAX], [True, True, False])
            ),
        ]
    )
    var out = frame.select_exprs(
        [
            when(col("b")).then(1).otherwise(2).alias("branch"),
            when(col("b"))
            .then(1)
            .when(lit(True))
            .then(3)
            .otherwise(4)
            .alias("chain"),
            when(col("b")).then("yes").alias("implicit"),
            coalesce([null("int64"), col("i"), lit(Int64(7))]).alias(
                "coalesced"
            ),
            col("i").sum().alias("total"),
            lit(Int64(9)).alias("scalar"),
        ]
    )
    assert_equal(out.height(), 3)
    var branches: List[Int64] = [1, 2, 2]
    var chains: List[Int64] = [1, 3, 3]
    var coalesced: List[Int64] = [10, 20, 7]
    for row in range(3):
        assert_equal(out.item(row, "branch").int64(), branches[row])
        assert_equal(out.item(row, "chain").int64(), chains[row])
        assert_equal(out.item(row, "implicit").is_null(), row != 0)
        assert_equal(out.item(row, "coalesced").int64(), coalesced[row])
        assert_equal(out.item(row, "total").int64(), Int64(30))
        assert_equal(out.item(row, "scalar").int64(), Int64(9))
    assert_equal(out.item(0, "implicit").string(), "yes")
    assert_equal(frame.select(lit(Int64(7))).shape(), (1, 1))
    assert_equal(
        frame.clear()
        .select_exprs([col("i"), lit(Int64(7)).alias("scalar")])
        .shape(),
        (0, 2),
    )
    var siblings = frame.with_columns(
        [(col("i") + 1).alias("i"), (col("i") * 2).alias("double")]
    )
    assert_equal(siblings.item(0, "i").int64(), Int64(11))
    assert_equal(siblings.item(0, "double").int64(), Int64(20))
    assert_true(siblings.item(2, "double").is_null())
    with assert_raises():
        _ = frame.column("i") + Series("other", Column[Int64]([1, 2]))
    for source in [frame.copy(), frame.clear()]:
        with assert_raises():
            _ = source.select_exprs(
                [col("i").alias("duplicate"), col("b").alias("duplicate")]
            )
        with assert_raises():
            _ = source.select(col("i").sum().sum())
        with assert_raises():
            _ = source.select(
                when(col("b")).then(lit(Int64(1))).otherwise(lit("wrong type"))
            )


def test_null_other_rows_keep_their_branch_under_the_predicate() raises:
    # `when(p).then(x)` without otherwise over a window that starts
    # mid-byte, in batches that do not end on a byte: each row is x where
    # p holds and both are valid, else null; eager, lazy and summed by key.
    var xs = List[Float64]()
    var x_valid = List[Bool]()
    var ps = List[Int64]()
    var p_valid = List[Bool]()
    var ks = List[Int64]()
    for row in range(5_000):
        xs.append(Float64(row % 250))
        x_valid.append(row % 11 != 3)
        ps.append(Int64(row * 37 % 1_000))
        p_valid.append(row % 7 != 2)
        ks.append(Int64(row % 5))
    var frame = DataFrame(
        [
            Series("x", Column[Float64](xs.copy(), x_valid.copy())),
            Series("p", Column[Int64](ps.copy(), p_valid.copy())),
            Series("k", Column[Int64](ks.copy())),
        ]
    ).slice(3, 4_000)
    var branch = when(col("p") > lit(Int64(500))).then(col("x")).end()
    var masked = branch.alias("w")
    var sums = List[Float64](length=5, fill=0)
    var kept = List[Bool]()
    for r in range(4_000):
        var row = r + 3
        var keep = x_valid[row] and p_valid[row] and ps[row] > 500
        kept.append(keep)
        if keep:
            sums[row % 5] += xs[row]
    var eager = frame.select_exprs([masked.copy()], batch_size=100)
    var lazy = frame.lazy().select_exprs([masked.copy()]).collect()
    for result in [eager.copy(), lazy.copy()]:
        assert_equal(result.height(), 4_000)
        for r in range(4_000):
            assert_equal(result.item(r, "w").is_null(), not kept[r])
            if kept[r]:
                assert_equal(result.item(r, "w").float64(), xs[r + 3])
    var grouped = (
        frame.lazy()
        .group_by(["k"])
        .agg([branch.sum().alias("s")])
        .sort(["k"])
        .collect()
    )
    assert_equal(grouped.height(), 5)
    for g in range(5):
        var key = Int(grouped.item(g, "k").int64())
        assert_equal(grouped.item(g, "s").float64(), sums[key])
