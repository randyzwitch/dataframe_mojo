"""Contract tests, including nulls, overflow, joins, and empty shapes."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import (
    Column,
    Series,
    DataFrame,
    Expr,
    col,
    lit,
)


def ints(var name: String, var values: List[Int64]) -> Series:
    return Series(name^, Column[Int64](values^))


def strings(var name: String, var values: List[String]) -> Series:
    return Series(name^, Column[String](values^))


def test_validity_crosses_bitmap_boundaries() raises:
    var values = List[Int64]()
    var valid = List[Bool]()
    for i in range(19):
        values.append(Int64(i))
        valid.append(i % 3 != 0)
    var column = Column[Int64](values^, valid)
    assert_equal(len(column), 19)
    assert_equal(column.null_count(), 7)
    for i in range(19):
        assert_equal(column.is_null(i), i % 3 == 0)
        if i % 3 != 0:
            assert_equal(column.value(i), Int64(i))
    var selected = column.take([18, 17, 0, 8, 17])
    assert_equal(selected.null_count(), 2)
    assert_equal(selected.value(1), Int64(17))
    assert_equal(selected.value(3), Int64(8))


def test_column_validation_and_null_access() raises:
    var column = Column[Int64]([5, 99], [True, False])
    with assert_raises():
        _ = Column[Int64]([1], [])
    with assert_raises():
        _ = column.value(1)
    with assert_raises():
        _ = column.value(-1)
    with assert_raises():
        _ = column.is_null(2)
    with assert_raises():
        _ = column.take([-1])
    with assert_raises():
        _ = column.take([2])
    with assert_raises():
        _ = column.take_or_null([-2], Int64(0))
    var missing = Column[String]([]).take_or_null([-1, -1], String(""))
    assert_equal(len(missing), 2)
    assert_equal(missing.null_count(), 2)


def test_schema_and_constructor_validation() raises:
    var frame = DataFrame([ints("n", [1, 2]), strings("label", ["a", "b"])])
    assert_equal(frame.height(), 2)
    assert_equal(frame.width(), 2)
    var schema = frame.schema()
    assert_equal(schema[0].name, "n")
    assert_equal(schema[0].dtype, "int64")
    assert_equal(schema[1].dtype, "string")
    with assert_raises():
        _ = DataFrame([ints("a", [1]), ints("b", [1, 2])])
    with assert_raises():
        _ = DataFrame([ints("a", [1]), ints("a", [2])])
    with assert_raises():
        _ = DataFrame([ints("a", [1])], height=2)
    with assert_raises():
        _ = DataFrame([], height=-2)
    with assert_raises():
        _ = frame.column("missing")
    with assert_raises():
        _ = frame.column("n").float64()
    with assert_raises():
        _ = frame.column("label").int64()


def test_projection_take_and_zero_column_shape() raises:
    var frame = DataFrame(
        [ints("n", [10, 20, 30]), strings("label", ["a", "b", "c"])]
    )
    var selected = frame.select(["label", "n"])
    assert_equal(selected.schema()[0].name, "label")
    var rows = selected.take([2, 0, 2])
    assert_equal(rows.column("n").int64().value(0), Int64(30))
    assert_equal(rows.column("label").string().value(1), "a")
    var zero = frame.select([])
    assert_equal(zero.height(), 3)
    assert_equal(zero.width(), 0)
    assert_equal(zero.take([1, 1]).height(), 2)
    assert_equal(zero.filter(Column[Bool]([False, True, False])).height(), 1)
    assert_equal(frame.take([]).height(), 0)
    assert_equal(frame.take([]).width(), 2)
    with assert_raises():
        _ = zero.take([3])
    with assert_raises():
        _ = frame.select(["n", "n"])
    with assert_raises():
        _ = frame.select(["missing"])


def test_filter_drops_null_mask_and_preserves_order() raises:
    var frame = DataFrame([ints("n", [4, 3, 2, 1])])
    var mask = Column[Bool](
        [True, True, False, True], [True, False, True, True]
    )
    var result = frame.filter(mask)
    assert_equal(result.height(), 2)
    assert_equal(result.column("n").int64().value(0), Int64(4))
    assert_equal(result.column("n").int64().value(1), Int64(1))
    with assert_raises():
        _ = frame.filter(Column[Bool]([True]))
    assert_equal(
        frame.filter(Column[Bool]([False, False, False, False])).height(), 0
    )


def test_with_column_replacement_and_independence() raises:
    var frame = DataFrame([ints("n", [1, 2])])
    var replaced = frame.with_column(ints("n", [9, 8]))
    assert_equal(replaced.width(), 1)
    assert_equal(replaced.column("n").int64().value(0), Int64(9))
    assert_equal(frame.column("n").int64().value(0), Int64(1))
    var added = frame.with_column(strings("s", ["x", "y"]))
    assert_equal(added.width(), 2)
    assert_equal(frame.width(), 1)
    with assert_raises():
        _ = frame.with_column(ints("bad", [1]))
    var empty_columns = DataFrame([], height=2)
    assert_equal(empty_columns.with_column(ints("n", [1, 2])).height(), 2)


def one(frame: DataFrame, expr: Expr) raises -> Series:
    return frame.select(expr.alias("r")).column("r")


def test_numeric_expressions_propagate_null_and_preserve_int_precision() raises:
    var frame = DataFrame(
        [
            Series(
                "i",
                Column[Int64]([9007199254740993, 2, 999], [True, True, False]),
            ),
            Series("f", Column[Float64]([2.5, 123, -1.0], [True, False, True])),
        ]
    )
    assert_equal(
        one(frame, col("i").sum()).int64().value(0), Int64(9007199254740995)
    )
    var mask = one(frame, col("i") > lit(Int64(2))).bool()
    assert_true(mask.value(0))
    assert_false(mask.value(1))
    assert_true(mask.is_null(2))
    var scaled = one(frame, col("f") * lit(Float64(2)))
    assert_equal(scaled.float64().value(0), Float64(5))
    assert_equal(scaled.float64().value(2), Float64(-2))
    assert_true(scaled.float64().is_null(1))
    assert_equal(
        one(frame, (col("f") * lit(Float64(2))).sum()).float64().value(0),
        Float64(3),
    )
    assert_equal(one(frame, col("f").sum()).float64().value(0), Float64(1.5))


def test_empty_and_all_null_reductions() raises:
    var empty = DataFrame(
        [
            Series("i", Column[Int64]([])),
            Series("f", Column[Float64]([])),
        ]
    )
    var nulls = DataFrame(
        [
            Series("i", Column[Int64]([99], [False])),
            Series("f", Column[Float64]([99], [False])),
        ]
    )
    for frame in [empty.copy(), nulls.copy()]:
        assert_true(one(frame, col("i").sum(min_count=1)).int64().is_null(0))
        assert_true(one(frame, col("f").sum(min_count=1)).float64().is_null(0))
        assert_equal(one(frame, col("i").sum()).int64().value(0), Int64(0))
    var zero = DataFrame([Series("i", Column[Int64]([0]))])
    assert_equal(
        one(zero, col("i").sum(min_count=1)).int64().value(0), Int64(0)
    )


def test_checked_sum_both_overflow_directions() raises:
    comptime MAX = Int64(9223372036854775807)
    comptime MIN = Int64(-9223372036854775807) - 1
    with assert_raises(contains="overflow"):
        _ = DataFrame([ints("v", [MAX, 1])]).select(col("v").sum())
    with assert_raises(contains="overflow"):
        _ = DataFrame([ints("v", [MIN, -1])]).select(col("v").sum())
    assert_equal(
        one(DataFrame([ints("v", [MIN, MAX])]), col("v").sum())
        .int64()
        .value(0),
        Int64(-1),
    )
    # Totals are exact, so an intermediate overflow that cancels is fine.
    assert_equal(
        one(DataFrame([ints("v", [MAX, 1, -1])]), col("v").sum())
        .int64()
        .value(0),
        MAX,
    )
    # Invalid payloads cannot cause overflow.
    var masked = DataFrame(
        [Series("v", Column[Int64]([MAX, 1], [True, False]))]
    )
    assert_equal(one(masked, col("v").sum()).int64().value(0), MAX)


def test_grouped_int_sum_null_keys_and_all_null_values() raises:
    var frame = DataFrame(
        [
            Series(
                "key",
                Column[String](
                    ["b", "unused", "a", "b", "unused", "empty"],
                    [True, False, True, True, False, True],
                ),
            ),
            Series(
                "value",
                Column[Int64](
                    [10, 2, 9007199254740993, 5, 3, 999],
                    [True, True, True, True, True, False],
                ),
            ),
        ]
    )
    var result = frame.group_by("key", maintain_order=True).agg(
        col("value").sum(min_count=1).alias("total")
    )
    assert_equal(result.height(), 4)
    var keys = result.column("key").string()
    var totals = result.column("total").int64()
    assert_equal(keys.value(0), "b")
    assert_true(keys.is_null(1))
    assert_equal(keys.value(2), "a")
    assert_equal(totals.value(0), Int64(15))
    assert_equal(totals.value(1), Int64(5))
    assert_equal(totals.value(2), Int64(9007199254740993))
    assert_true(totals.is_null(3))


def test_grouped_float_sum_and_empty_input() raises:
    var frame = DataFrame(
        [
            strings("k", ["a", "b", "a", "c"]),
            Series(
                "v",
                Column[Float64]([1.5, 99, 2.5, 0], [True, False, True, True]),
            ),
        ]
    )
    var grouped = frame.group_by("k", maintain_order=True).agg(
        col("v").sum(min_count=1).alias("sum")
    )
    var totals = grouped.column("sum").float64()
    assert_equal(totals.value(0), Float64(4))
    assert_true(totals.is_null(1))
    assert_equal(totals.value(2), Float64(0))
    var empty = frame.take([]).group_by("k").agg(col("v").sum().alias("sum"))
    assert_equal(empty.height(), 0)
    assert_equal(empty.width(), 2)
    assert_equal(empty.column("sum").dtype(), "float64")


def test_grouped_sum_validation_and_overflow() raises:
    var frame = DataFrame(
        [strings("k", ["a", "a"]), ints("v", [9223372036854775807, 1])]
    )
    with assert_raises(contains="overflow"):
        _ = frame.group_by("k").agg(col("v").sum())
    with assert_raises(contains="collides with grouping key"):
        _ = frame.group_by("k").agg(col("v").sum().alias("k"))
    with assert_raises(contains="sum requires a numeric expression"):
        _ = frame.group_by("v").agg(col("k").sum())
    with assert_raises(contains="Unknown column"):
        _ = frame.group_by("missing").agg(col("v").sum())


def test_sort_is_stable_in_both_directions_and_places_nulls() raises:
    var frame = DataFrame(
        [
            Series(
                "key",
                Column[Int64](
                    [2, 1, 2, 999, 1], [True, True, True, False, True]
                ),
            ),
            ints("id", [0, 1, 2, 3, 4]),
        ]
    )
    var ascending = frame.sort("key").column("id").int64()
    var descending = frame.sort("key", True).column("id").int64()
    var first = frame.sort("key", False, False).column("id").int64()
    var expected_asc: List[Int64] = [1, 4, 0, 2, 3]
    var expected_desc: List[Int64] = [0, 2, 1, 4, 3]
    var expected_first: List[Int64] = [3, 1, 4, 0, 2]
    for i in range(5):
        assert_equal(ascending.value(i), expected_asc[i])
        assert_equal(descending.value(i), expected_desc[i])
        assert_equal(first.value(i), expected_first[i])
    assert_equal(frame.take([]).sort("key").height(), 0)


def test_sort_strings_bool_and_nan() raises:
    var frame = DataFrame(
        [
            strings("s", ["b", "a", "b"]),
            Series("b", Column[Bool]([True, False, True])),
            ints("id", [0, 1, 2]),
        ]
    )
    assert_equal(frame.sort("s").column("id").int64().value(0), Int64(1))
    assert_equal(frame.sort("b").column("id").int64().value(0), Int64(1))
    assert_true(frame.column("b").bool().value(0))
    var nan = Float64("nan")
    var floats = DataFrame(
        [
            Series(
                "f",
                Column[Float64](
                    [nan, 3, 999, -1, nan], [True, True, False, True, True]
                ),
            ),
            ints("id", [0, 1, 2, 3, 4]),
        ]
    )
    var sorted = floats.sort("f").column("id").int64()
    var expected: List[Int64] = [3, 1, 0, 4, 2]
    for i in range(5):
        assert_equal(sorted.value(i), expected[i])
    assert_equal(floats.sort("f", True).column("id").int64().value(0), Int64(1))
    assert_equal(
        floats.sort("f", True, False).column("id").int64().value(0), Int64(2)
    )
    var total = one(floats, col("f").sum()).float64().value(0)
    assert_true(total != total)


def test_join_duplicate_keys_produce_all_pairs_in_order() raises:
    var left = DataFrame(
        [strings("k", ["b", "a", "a", "c"]), ints("id", [0, 1, 2, 3])]
    )
    var right = DataFrame(
        [strings("k", ["a", "b", "a"]), ints("id", [10, 20, 30])]
    )
    var result = left.join(right, "k")
    assert_equal(result.height(), 5)
    assert_equal(result.width(), 3)
    var l = result.column("id").int64()
    var r = result.column("id_right").int64()
    var expected_l: List[Int64] = [0, 1, 1, 2, 2]
    var expected_r: List[Int64] = [20, 10, 30, 10, 30]
    for i in range(5):
        assert_equal(l.value(i), expected_l[i])
        assert_equal(r.value(i), expected_r[i])
    assert_equal(left.height(), 4)
    assert_equal(right.height(), 3)


def test_left_join_unmatched_and_null_keys_never_match() raises:
    var left = DataFrame(
        [
            Series(
                "k", Column[String](["a", "x", "unused"], [True, True, False])
            ),
            ints("id", [0, 1, 2]),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[String](["a", "unused"], [True, False])),
            Series("f", Column[Float64]([2.5, 9])),
            Series("b", Column[Bool]([True, False])),
            strings("s", ["yes", "no"]),
        ]
    )
    var result = left.join(right, "k", "left")
    assert_equal(result.height(), 3)
    assert_equal(result.column("f").float64().value(0), Float64(2.5))
    assert_true(result.column("b").bool().value(0))
    assert_equal(result.column("s").string().value(0), "yes")
    for i in range(1, 3):
        assert_true(result.column("f").float64().is_null(i))
        assert_true(result.column("b").bool().is_null(i))
        assert_true(result.column("s").string().is_null(i))
    assert_equal(left.join(right, "k").height(), 1)
    assert_true(result.column("k").string().is_null(2))


def test_join_empty_inputs_and_name_collisions() raises:
    var left = DataFrame([strings("k", ["a"]), ints("v", [1])])
    var right = DataFrame([strings("k", []), ints("v", [])])
    assert_equal(left.join(right, "k").height(), 0)
    var outer = left.join(right, "k", "left")
    assert_equal(outer.height(), 1)
    assert_true(outer.column("v_right").int64().is_null(0))
    assert_equal(left.take([]).join(right, "k", "left").height(), 0)
    with assert_raises(contains="Join how must be"):
        _ = left.join(right, "k", "outer")
    with assert_raises():
        _ = left.join(right, "k", "inner", "")
    var collision = left.with_column(ints("v_right", [2]))
    with assert_raises():
        _ = collision.join(right, "k")
    with assert_raises(contains="Unknown column"):
        _ = left.join(right, "missing")
    var right_collision = DataFrame(
        [strings("k", ["a"]), ints("v", [1]), ints("v_right", [2])]
    )
    with assert_raises():
        _ = left.join(right_collision, "k")


def test_sort_many_rows_matches_order_and_stability() raises:
    var values = List[Int64]()
    for i in range(257):
        values.append(Int64((i * 37) % 19))
    var series = Series("key", Column[Int64](values.copy()))
    var order = series.argsort()
    assert_equal(len(order), 257)
    var seen = List[Bool](length=257, fill=False)
    for i in range(257):
        assert_false(seen[order[i]])
        seen[order[i]] = True
        if i > 0:
            assert_true(values[order[i - 1]] <= values[order[i]])
            if values[order[i - 1]] == values[order[i]]:
                assert_true(order[i - 1] < order[i])


def test_sales_pipeline() raises:
    var sales = DataFrame(
        [
            strings("region", ["east", "west", "east", "west", "north"]),
            Series(
                "amount",
                Column[Float64](
                    [100, 80, -20, 120, 999], [True, True, True, True, False]
                ),
            ),
        ]
    )
    var result = (
        sales.filter(col("amount") > lit(Float64(0)))
        .with_columns((col("amount") * lit(Float64(0.9))).alias("net"))
        .group_by("region", maintain_order=True)
        .agg(col("net").sum().alias("revenue"))
    )
    assert_equal(result.height(), 2)
    assert_equal(result.column("region").string().value(0), "east")
    assert_equal(result.column("revenue").float64().value(0), Float64(90))
    assert_equal(result.column("region").string().value(1), "west")
    assert_equal(result.column("revenue").float64().value(1), Float64(180))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
