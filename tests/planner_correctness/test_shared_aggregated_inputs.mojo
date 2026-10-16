"""A subplan read whole in several places runs once and every reader reads
its result, as DuckDB materializes a CTE read twice and Polars caches a
repeated subplan: a join several aggregations read (a rollup written as
UNION ALL), and a subquery several joins build on. Copies with a filter of
their own above them stay apart, since each filter would have narrowed its
copy. Either way the result is the scalar model's, with the optimizer on
or off."""
# contract-group: planner.relational
from std.testing import assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    Series,
    col,
    lit,
)
from dataframe.value import AnyValue
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 40_000
comptime ITEMS = 300
comptime STORES = 12


def integer(value: AnyValue) raises -> Int128:
    if value.dtype() == DataType.INT32:
        return Int128(value.int32())
    return Int128(value.int64())


def item_of(row: Int) -> Int:
    return (row * 7919) % (ITEMS + 20)


def store_of(row: Int) -> Int:
    return (row * 104729) % STORES


def quantity(row: Int) -> Int64:
    return Int64(row % 97)


def joins(row: Int) -> Bool:
    # Items past ITEMS have no item row; odd stores have no store row.
    return item_of(row) < ITEMS and store_of(row) % 2 == 0


def item_name(item: Int) -> String:
    return "item identifier " + String(item % 41)


def state_of(store: Int) -> String:
    return "state " + String(store % 3)


def sales() raises -> DataFrame:
    var items = List[Int64]()
    var stores = List[Int64]()
    var quantities = List[Int64]()
    for row in range(ROWS):
        items.append(Int64(item_of(row)))
        stores.append(Int64(store_of(row)))
        quantities.append(quantity(row))
    return DataFrame(
        [
            Series("ss_item", Column[Int64](items^)),
            Series("ss_store", Column[Int64](stores^)),
            Series("qty", Column[Int64](quantities^)),
        ]
    )


def items() raises -> DataFrame:
    var keys = List[Int64]()
    var names = List[String]()
    for item in range(ITEMS):
        keys.append(Int64(item))
        names.append(item_name(item))
    return DataFrame(
        [
            Series("i_item", Column[Int64](keys^)),
            Series("i_name", Column[String](names^)),
        ]
    )


def stores() raises -> DataFrame:
    var keys = List[Int64]()
    var states = List[String]()
    for store in range(0, STORES, 2):
        keys.append(Int64(store))
        states.append(state_of(store))
    return DataFrame(
        [
            Series("s_store", Column[Int64](keys^)),
            Series("s_state", Column[String](states^)),
        ]
    )


def joined() raises -> LazyFrame:
    return (
        sales()
        .lazy()
        .join(items().lazy(), left_on=["ss_item"], right_on=["i_item"])
        .join(stores().lazy(), left_on=["ss_store"], right_on=["s_store"])
    )


def totals() -> List[Expr]:
    return [col("qty").sum().alias("total"), col("qty").count().alias("rows")]


def rollup(with_filters: Bool) raises -> LazyFrame:
    """By name and state, by name, and over every row, as one frame."""
    var source = joined()
    var by_both = source.copy()
    var by_name = source.copy()
    var overall = source.copy()
    if with_filters:
        by_both = by_both.filter(col("qty") >= lit(Int64(10)))
        by_name = by_name.filter(col("qty") < lit(Int64(50)))
    var shown: List[Expr] = [
        col("i_name"),
        col("s_state"),
        col("total"),
        col("rows"),
    ]
    var first = (
        by_both.group_by(["i_name", "s_state"])
        .agg(totals())
        .select_exprs(shown)
    )
    var second = (
        by_name.group_by(["i_name"])
        .agg(totals())
        .with_columns([lit("all").alias("s_state")])
        .select_exprs(shown)
    )
    var third = (
        overall.select_exprs(totals())
        .with_columns([lit("all").alias("i_name"), lit("all").alias("s_state")])
        .select_exprs(shown)
    )
    return first.concat(second).concat(third)


def add(mut expected: Dict[String, List[Int128]], key: String, row: Int) raises:
    if key not in expected:
        expected[key] = [Int128(0), Int128(0)]
    expected[key][0] += Int128(quantity(row))
    expected[key][1] += 1


def assert_rollup(actual: DataFrame, with_filters: Bool) raises:
    var expected = Dict[String, List[Int128]]()
    for row in range(ROWS):
        if not joins(row):
            continue
        var name = item_name(item_of(row))
        var q = quantity(row)
        if not with_filters or q >= 10:
            add(expected, "1|" + name + "|" + state_of(store_of(row)), row)
        if not with_filters or q < 50:
            add(expected, "2|" + name + "|all", row)
        add(expected, "3|all|all", row)
    assert_equal(actual.height(), len(expected))
    for r in range(actual.height()):
        var name = actual.item(r, "i_name").string()
        var state = actual.item(r, "s_state").string()
        var level = "3" if name == "all" else ("2" if state == "all" else "1")
        var key = level + "|" + name + "|" + state
        assert_true(key in expected, "unexpected row " + key)
        assert_equal(integer(actual.item(r, "total")), expected[key][0])
        assert_equal(integer(actual.item(r, "rows")), expected[key][1])


def test_a_join_every_aggregation_reads_whole_runs_once() raises:
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    assert_rollup(rollup(False).collect(), False)
    assert_true(
        "dataframe-path: lazy.shared_subplan\n" in observed.text(),
        "the repeated join was not shared",
    )
    _ = observed^
    assert_rollup(rollup(False).collect(optimize=False), False)
    _ = trace^


def test_copies_under_their_own_filters_stay_apart() raises:
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    assert_rollup(rollup(True).collect(), True)
    assert_equal(
        "dataframe-path: lazy.shared_subplan\n" in observed.text(), False
    )
    _ = observed^
    assert_rollup(rollup(True).collect(optimize=False), True)
    _ = trace^


def chosen_items(item_frame: DataFrame, wanted: DataFrame) raises -> LazyFrame:
    """A subquery two joins build on: the items whose name is one of a
    list, found by a semi join (as TPC-DS q83 picks its dates)."""
    return (
        item_frame.lazy()
        .join(
            wanted.lazy(), left_on=["i_name"], right_on=["w_name"], how="semi"
        )
        .select(["i_item"])
    )


def chosen(item: Int) -> Bool:
    return (item % 41) % 7 == 0


def wanted_names() raises -> DataFrame:
    var names = List[String]()
    for k in range(0, 41, 7):
        names.append("item identifier " + String(k))
    return DataFrame([Series("w_name", Column[String](names^))])


def channel(
    low: Bool, sale_frame: DataFrame, item_frame: DataFrame, wanted: DataFrame
) raises -> LazyFrame:
    var rows = sale_frame.lazy()
    if low:
        rows = rows.filter(col("qty") < lit(Int64(48)))
    else:
        rows = rows.filter(col("qty") >= lit(Int64(48)))
    return rows.join(
        chosen_items(item_frame, wanted),
        left_on=["ss_item"],
        right_on=["i_item"],
    ).select(["ss_item", "qty"])


def test_a_subquery_two_joins_build_on_runs_once() raises:
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    # One copy of each table, as a query scans: the two subqueries read the
    # same buffers.
    var sale_frame = sales()
    var item_frame = items()
    var wanted = wanted_names()
    for optimize in [True, False]:
        var observed = StrategyTrace()
        var actual = (
            channel(True, sale_frame, item_frame, wanted)
            .concat(channel(False, sale_frame, item_frame, wanted))
            .collect(optimize=optimize)
        )
        if optimize:
            assert_true(
                "dataframe-path: lazy.shared_subplan\n" in observed.text(),
                "the repeated build side was not shared",
            )
        _ = observed^
        var expected = 0
        var total = Int128(0)
        for row in range(ROWS):
            var item = item_of(row)
            if item < ITEMS and chosen(item):
                expected += 1
                total += Int128(quantity(row))
        assert_equal(actual.height(), expected)
        var seen = Int128(0)
        for r in range(actual.height()):
            seen += integer(actual.item(r, "qty"))
            var item = Int(integer(actual.item(r, "ss_item")))
            assert_true(chosen(item))
        assert_equal(seen, total)
    _ = trace^
