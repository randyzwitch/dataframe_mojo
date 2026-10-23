"""An inner join's inequality between its sides, written as a filter above
it, bounds the large side by the small side's least or greatest value
for <, <=, > and >= comparisons: the large side then holds
only rows under that bound, and every result equals the unoptimized
plan's (TPC-DS q72's stock on hand below the quantity ordered)."""
# contract-group: planner.relational
from std.testing import assert_true
from dataframe import Column, DataFrame, LazyFrame, Series, col, lit
from dataframe.lazy import JOIN, SCAN_FRAME
from tests.support.coverage_compare import assert_frame

comptime STOCK = 8192


def _stock() raises -> LazyFrame:
    var ids = List[Int64]()
    var items = List[Int64]()
    var days = List[Int64]()
    var on_hand = List[Int64]()
    for row in range(STOCK):
        ids.append(Int64(row))
        items.append(Int64(row % 64))
        days.append(Int64(row % 14))
        on_hand.append(Int64((row * 7919) % 1000))
    var stock = DataFrame(
        [
            Series("stock_id", Column[Int64](ids^)),
            Series("stock_item", Column[Int64](items^)),
            Series("stock_day", Column[Int64](days^)),
            Series("on_hand", Column[Int64](on_hand^)),
        ]
    )
    var day_keys = List[Int64]()
    var weeks = List[Int64]()
    for day in range(14):
        day_keys.append(Int64(day))
        weeks.append(Int64(day // 7))
    var calendar = DataFrame(
        [
            Series("day", Column[Int64](day_keys^)),
            Series("stock_week", Column[Int64](weeks^)),
        ]
    )
    return stock.lazy().join(
        calendar.lazy(), left_on=["stock_day"], right_on=["day"]
    )


def _orders() raises -> LazyFrame:
    var ids = List[Int64]()
    var items = List[Int64]()
    var weeks = List[Int64]()
    var quantities = List[Int64]()
    for row in range(40):
        ids.append(Int64(row))
        items.append(Int64((row * 5) % 64))
        weeks.append(Int64(row % 2))
        quantities.append(Int64(1 + (row * 13) % 50))
    return (
        DataFrame(
            [
                Series("order_id", Column[Int64](ids^)),
                Series("item", Column[Int64](items^)),
                Series("week", Column[Int64](weeks^)),
                Series("quantity", Column[Int64](quantities^)),
            ]
        )
        .lazy()
        .filter(col("quantity") > lit(Int64(0)))
    )


def _short(small_bound: Bool) raises -> LazyFrame:
    var joined = _orders().join(
        _stock(),
        left_on=["item", "week"],
        right_on=["stock_item", "stock_week"],
    )
    if small_bound:
        # The small side on the left of the comparison: flipped.
        joined = joined.filter(col("quantity") > col("on_hand"))
    else:
        joined = joined.filter(col("on_hand") < col("quantity"))
    return joined.select(["order_id", "stock_id", "on_hand", "quantity"])


def test_an_inequality_bounds_the_large_side_by_the_small_sides_extreme() raises:
    for flipped in [False, True]:
        var query = _short(flipped)
        var plan = query._optimized()
        plan._push_join_key_sets(False, 29)
        var bounded = False
        for node in plan._nodes:
            if node.kind != JOIN or len(node.names) != 2:
                continue
            ref right = plan._nodes[node.right]
            assert_true(right.kind == SCAN_FRAME)
            ref stock = plan._frames[right.offset]
            # The largest quantity ordered is 50.
            var most = stock.column("on_hand").max().int64()
            assert_true(most < 50, "the stock was not bounded")
            bounded = True
        assert_true(bounded)
        var sort_keys: List[String] = ["order_id", "stock_id"]
        assert_frame(
            query.collect().sort(sort_keys),
            query.collect(optimize=False).sort(sort_keys),
        )
