"""Decimal arithmetic and sums check precision with a limit read once per
column: exact results, grouped and ungrouped, with and without nulls, and
the same error as before when a result exceeds the precision.
"""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import Column, DataFrame, DataType, Series, col


def money(values: List[Int128], valid: List[Bool]) raises -> Series:
    return Series("x", Column[Int128](values.copy(), valid.copy())).with_dtype(
        DataType.decimal(15, 2)
    )


def raw(frame: DataFrame, name: String, row: Int) raises -> Int128:
    """The unscaled value of one decimal cell."""
    return frame.column(name)._data[Column[Int128]]._get(row)


def test_sums_match_row_by_row_totals() raises:
    var n = 5000
    for with_nulls in [False, True]:
        var values = List[Int128](capacity=n)
        var valid = List[Bool](capacity=n)
        var keys = List[Int64](capacity=n)
        var totals = List[Int128](length=7, fill=0)
        var whole = Int128(0)
        for i in range(n):
            var v = Int128((i * 7919) % 100_000 - 50_000)
            var present = not with_nulls or i % 11 != 3
            values.append(v)
            valid.append(present)
            keys.append(Int64(i % 7))
            if present:
                totals[i % 7] += v
                whole += v
        var frame = DataFrame(
            [money(values, valid), Series("k", Column[Int64](keys^))]
        )
        var grouped = frame.group_by("k", maintain_order=True).agg(
            [col("x").sum().alias("s")]
        )
        for g in range(7):
            assert_equal(raw(grouped, "s", g), totals[g])
        var total = frame.select_exprs([col("x").sum().alias("s")])
        assert_equal(raw(total, "s", 0), whole)


def test_arithmetic_is_exact_and_checks_precision() raises:
    var values: List[Int128] = [12345, -250, 99, 0]
    var valid: List[Bool] = [True, True, True, False]
    var frame = DataFrame([money(values, valid)])
    var out = frame.select_exprs(
        [
            (col("x") + col("x")).alias("add"),
            (col("x") - col("x")).alias("sub"),
            (col("x") * col("x")).alias("mul"),
        ]
    )
    assert_equal(raw(out, "add", 0), Int128(24690))
    assert_equal(raw(out, "sub", 1), Int128(0))
    assert_true(out.item(3, "mul").is_null())
    # 123.45 * 123.45 = 15239.9025 at the product's scale.
    var scale = out.column("mul").dtype().scale()
    var expected = Int128(152399025)
    for _ in range(4 - scale):
        expected = expected // 10
    assert_equal(raw(out, "mul", 0), expected)
    # A sum at precision 38 that leaves it raises, as before.
    var big = Int128(6) * Int128(10) ** 37
    var wide = DataFrame(
        [
            Series("x", Column[Int128]([big, big])).with_dtype(
                DataType.decimal(38, 0)
            )
        ]
    )
    with assert_raises(contains="exceeds decimal precision"):
        _ = wide.select_exprs([col("x").sum()])
    with assert_raises(contains="exceeds decimal precision"):
        _ = wide.select_exprs([col("x") + col("x")])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
