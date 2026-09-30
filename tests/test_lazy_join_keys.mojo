"""Lazy joins on differently named keys (#329): left_on/right_on pairs, as
in TPC-H's `o_custkey = c_custkey`. Results match the eager join; each
input reads only its own keys and the columns used above the join; filters
move to the side that owns their columns."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from dataframe import (
    Column,
    DataFrame,
    Series,
    StringColumn,
    col,
    lit,
    scan_csv,
    write_csv,
)

comptime ORDERS_CSV = "/tmp/dataframe_mojo_lazy_join_keys_orders.csv"
comptime CUSTOMER_CSV = "/tmp/dataframe_mojo_lazy_join_keys_customer.csv"


def tables() raises -> Tuple[DataFrame, DataFrame]:
    var orders = DataFrame(
        [
            Series("o_orderkey", Column[Int64]([1, 2, 3, 4, 5, 6, 7, 8])),
            Series(
                "o_custkey",
                Column[Int64](
                    [10, 20, 10, 30, 40, 20, 10, 0],
                    [True, True, True, True, True, True, True, False],
                ),
            ),
            Series(
                "o_total",
                Column[Float64]([5.0, 7.5, 1.0, 9.0, 2.0, 3.0, 4.0, 6.0]),
            ),
            Series(
                "o_note",
                StringColumn(["a", "b", "c", "d", "e", "f", "g", "h"]),
            ),
        ]
    )
    var customer = DataFrame(
        [
            Series("c_custkey", Column[Int64]([20, 10, 30, 50])),
            Series("c_name", StringColumn(["bo", "al", "cy", "di"])),
            Series("c_nation", Column[Int64]([1, 2, 1, 3])),
            Series("c_pad", StringColumn(["x", "y", "z", "w"])),
        ]
    )
    return (orders^, customer^)


def test_every_join_type_matches_eager() raises:
    var t = tables()
    var hows: List[String] = ["inner", "left", "right", "full", "semi", "anti"]
    for how in hows:
        for coalesce in [True, False]:
            var lazy = (
                t[0]
                .lazy()
                .join(
                    t[1].lazy(),
                    left_on=["o_custkey"],
                    right_on=["c_custkey"],
                    how=how,
                    coalesce=coalesce,
                )
                .collect()
            )
            var eager = t[0].join(
                t[1],
                left_on=["o_custkey"],
                right_on=["c_custkey"],
                how=how,
                coalesce=coalesce,
            )
            assert_equal(lazy.columns(), eager.columns(), how)
            assert_true(lazy.equals(eager), how)


def test_multiple_key_pairs() raises:
    var t = tables()
    var left = t[0].with_columns((col("o_custkey") % 20).alias("o_bucket"))
    var right = t[1].with_columns((col("c_custkey") % 20).alias("c_bucket"))
    var lazy = (
        left.lazy()
        .join(
            right.lazy(),
            left_on=["o_custkey", "o_bucket"],
            right_on=["c_custkey", "c_bucket"],
        )
        .collect()
    )
    var eager = left.join(
        right,
        left_on=["o_custkey", "o_bucket"],
        right_on=["c_custkey", "c_bucket"],
    )
    assert_true(lazy.equals(eager))


def test_each_side_reads_only_its_keys_and_used_columns() raises:
    var t = tables()
    var query = (
        t[0]
        .lazy()
        .join(t[1].lazy(), left_on=["o_custkey"], right_on=["c_custkey"])
        .group_by("c_name")
        .agg([col("o_total").sum()])
    )
    var plan = query.explain()
    assert_true("JOIN inner on o_custkey = c_custkey" in plan, plan)
    assert_true("[project o_custkey, o_total]" in plan, plan)
    assert_true("[project c_custkey, c_name]" in plan, plan)
    assert_false("o_note" in plan or "c_pad" in plan, plan)
    var eager = (
        t[0]
        .join(t[1], left_on=["o_custkey"], right_on=["c_custkey"])
        .group_by("c_name")
        .agg([col("o_total").sum()])
    )
    assert_true(
        query.collect().sort("c_name").equals(eager.sort("c_name")), plan
    )
    # The schema comes from the plan without running it.
    var schema = (
        t[0]
        .lazy()
        .join(t[1].lazy(), left_on=["o_custkey"], right_on=["c_custkey"])
        .collect_schema()
    )
    assert_equal(
        schema,
        [
            "o_orderkey: int64",
            "o_custkey: int64",
            "o_total: float64",
            "o_note: string",
            "c_name: string",
            "c_nation: int64",
            "c_pad: string",
        ],
    )


def test_filters_move_to_the_side_that_owns_them() raises:
    var t = tables()
    var query = (
        t[0]
        .lazy()
        .join(t[1].lazy(), left_on=["o_custkey"], right_on=["c_custkey"])
        .filter(col("c_nation") == lit(Int64(1)))
        .filter(col("o_custkey") > lit(Int64(10)))
        .select(["o_orderkey", "c_name"])
    )
    var plan = query.explain()
    # Both filters sit below the join, one on each input.
    var join_at = plan.find("JOIN")
    assert_true(join_at >= 0 and plan.find("FILTER") > join_at, plan)
    var eager = (
        t[0]
        .join(t[1], left_on=["o_custkey"], right_on=["c_custkey"])
        .filter(col("c_nation") == lit(Int64(1)))
        .filter(col("o_custkey") > lit(Int64(10)))
        .select(["o_orderkey", "c_name"])
    )
    assert_true(query.collect().equals(eager), plan)
    # A coalesced right key is not an output column: filtering on it
    # fails as it does eagerly, rather than quietly moving to the right.
    with assert_raises():
        _ = (
            t[0]
            .lazy()
            .join(t[1].lazy(), left_on=["o_custkey"], right_on=["c_custkey"])
            .filter(col("c_custkey") > lit(Int64(10)))
            .collect()
        )


def test_streaming_and_csv_scans() raises:
    var t = tables()
    var eager = t[0].join(
        t[1], left_on=["o_custkey"], right_on=["c_custkey"], how="left"
    )
    var streamed = (
        t[0]
        .lazy()
        .join(
            t[1].lazy(),
            left_on=["o_custkey"],
            right_on=["c_custkey"],
            how="left",
        )
        .collect(batch_size=3)
    )
    assert_true(streamed.equals(eager))
    write_csv(t[0], ORDERS_CSV)
    write_csv(t[1], CUSTOMER_CSV)
    var query = (
        scan_csv(ORDERS_CSV)
        .join(
            scan_csv(CUSTOMER_CSV),
            left_on=["o_custkey"],
            right_on=["c_custkey"],
        )
        .select(["o_total", "c_name"])
    )
    var plan = query.explain()
    assert_true("[project o_custkey, o_total]" in plan, plan)
    assert_true("[project c_custkey, c_name]" in plan, plan)
    var expected = (
        t[0]
        .join(t[1], left_on=["o_custkey"], right_on=["c_custkey"])
        .select(["o_total", "c_name"])
    )
    assert_true(query.collect().equals(expected))


def test_sort_with_a_direction_per_key() raises:
    var t = tables()
    var lazy = (
        t[0]
        .lazy()
        .sort(
            ["o_custkey", "o_total"],
            descending=[False, True],
            nulls_last=[False, True],
        )
        .collect()
    )
    var eager = t[0].sort(
        ["o_custkey", "o_total"],
        descending=[False, True],
        nulls_last=[False, True],
    )
    assert_true(lazy.equals(eager))
    with assert_raises(contains="one descending and nulls_last flag"):
        _ = (
            t[0]
            .lazy()
            .sort(["o_custkey"], descending=[True, False], nulls_last=[True])
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
