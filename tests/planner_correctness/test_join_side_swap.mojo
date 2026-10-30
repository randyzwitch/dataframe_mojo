"""An inner join whose executed right side costs more to hash, in rows times
columns, than its left can is run with the sides swapped: the left is hashed
and the right streams through it. A join promises no row order, so the swap applies whatever
consumes the join, an ordered group-by included. The swap must not change
the result, the join's column names (the left key's own name), its key type,
or which rows join; it is declined where a shared non-key name would be
suffixed."""
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
from tests.support.aggregate_table_reference import integer
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime DIMENSIONS = 500


def num(value: AnyValue) raises -> Int64:
    return Int64(integer(value))


comptime FACT_ROWS = 60_000
comptime FACT_KEYS = 600


def name_valid(d: Int) -> Bool:
    return d % 50 != 3


def name_of(d: Int) -> String:
    return "dimension name number " + String(d % 40)


def dimension() raises -> DataFrame:
    var keys = List[Int32]()
    var names = List[String]()
    var valid = List[Bool]()
    var classes = List[Int64]()
    for d in range(DIMENSIONS):
        keys.append(Int32(d))
        names.append(name_of(d))
        valid.append(name_valid(d))
        classes.append(Int64(d % 7))
    return DataFrame(
        [
            Series("d", Column[Int32](keys^)),
            Series("name", Column[String](names^, valid)),
            Series("grp", Column[Int64](classes^)),
        ]
    )


def fact_key(row: Int) -> Int:
    # Keys DIMENSIONS..FACT_KEYS-1 have no dimension row.
    return (row * 7919) % FACT_KEYS


def fact_value(row: Int) -> Int64:
    return Int64(row % 101)


def kept(row: Int) -> Bool:
    return fact_value(row) >= 3


def facts(shared_name: Bool = False) raises -> DataFrame:
    var keys = List[Int32]()
    var values = List[Int64]()
    for row in range(FACT_ROWS):
        keys.append(Int32(fact_key(row)))
        values.append(fact_value(row))
    var columns: List[Series] = [
        Series("fd", Column[Int32](keys^)),
        Series("v", Column[Int64](values^)),
    ]
    if shared_name:
        var classes = List[Int64]()
        for row in range(FACT_ROWS):
            classes.append(Int64(row % 3))
        columns.append(Series("grp", Column[Int64](classes^)))
    return DataFrame(columns^)


def sums() -> List[Expr]:
    return [
        col("v").sum().alias("total"),
        col("v").count().alias("rows"),
    ]


def joined(shared_name: Bool = False) raises -> LazyFrame:
    return (
        dimension()
        .lazy()
        .join(
            facts(shared_name).lazy().filter(col("v") >= lit(Int64(3))),
            left_on=["d"],
            right_on=["fd"],
        )
    )


def expected_by_name_and_group() raises -> Dict[String, List[Int64]]:
    """Scalar totals and row counts by (name, grp) over the joined rows."""
    var out = Dict[String, List[Int64]]()
    for row in range(FACT_ROWS):
        var d = fact_key(row)
        if d >= DIMENSIONS or not kept(row):
            continue
        var key = (name_of(d) if name_valid(d) else String("<null>")) + "|"
        key += String(d % 7)
        if key not in out:
            out[key] = [Int64(0), Int64(0)]
        out[key][0] += fact_value(row)
        out[key][1] += 1
    return out^


def assert_by_name_and_group(actual: DataFrame) raises:
    var expected = expected_by_name_and_group()
    assert_equal(actual.height(), len(expected))
    for r in range(actual.height()):
        var name = actual.item(r, "name")
        var key = (String("<null>") if name.is_null() else name.string()) + "|"
        key += String(num(actual.item(r, "grp")))
        assert_true(key in expected, "unexpected group " + key)
        assert_equal(num(actual.item(r, "total")), expected[key][0])
        assert_equal(num(actual.item(r, "rows")), expected[key][1])


def swapped(paths: String) -> Bool:
    return "dataframe-path: lazy.swap_join_sides\n" in paths


def test_larger_built_right_side_is_swapped_with_the_same_groups_either_optimization() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    for optimize in [False, True]:
        var observed = StrategyTrace()
        var actual = (
            joined()
            .group_by(["name", "grp"])
            .agg(sums())
            .collect(optimize=optimize)
        )
        assert_true(swapped(observed.text()), "the join was not swapped")
        _ = observed^
        assert_by_name_and_group(actual)
    _ = trace^
    _ = workers^


def test_a_built_side_of_more_rows_but_fewer_cells_is_not_swapped() raises:
    """Rows times columns decides, as the planner's build side and
    DuckDB's build/probe side optimizer count it: 640 two-column facts
    (about 620 kept) build cheaper than 500 three-column dimension rows,
    so the join keeps its sides (TPC-DS q12 restarted its pipeline to
    flip such a join)."""
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    comptime ROWS = 640
    var keys = List[Int32]()
    var values = List[Int64]()
    for row in range(ROWS):
        keys.append(Int32(fact_key(row)))
        values.append(fact_value(row))
    var few = DataFrame(
        [
            Series("fd", Column[Int32](keys^)),
            Series("v", Column[Int64](values^)),
        ]
    )
    var totals = List[Int64](length=7, fill=0)
    var counts = List[Int64](length=7, fill=0)
    var kept_rows = 0
    for row in range(ROWS):
        if not kept(row):
            continue
        kept_rows += 1
        var d = fact_key(row)
        if d < DIMENSIONS:
            totals[d % 7] += fact_value(row)
            counts[d % 7] += 1
    assert_true(kept_rows > DIMENSIONS and 2 * kept_rows < 3 * DIMENSIONS)
    var observed = StrategyTrace()
    var actual = (
        dimension()
        .lazy()
        .join(
            few.lazy().filter(col("v") >= lit(Int64(3))),
            left_on=["d"],
            right_on=["fd"],
        )
        .group_by("grp")
        .agg(
            [
                col("v").sum().alias("total"),
                col("v").count().alias("rows"),
                # Reads the name: the dimension keeps its three columns.
                col("name").max().alias("last_name"),
            ]
        )
        .collect()
    )
    assert_true(not swapped(observed.text()), "the join was swapped")
    _ = observed^
    var present = 0
    for g in range(7):
        if counts[g] > 0:
            present += 1
    assert_equal(actual.height(), present)
    for r in range(actual.height()):
        var g = Int(num(actual.item(r, "grp")))
        assert_equal(num(actual.item(r, "total")), totals[g])
        assert_equal(num(actual.item(r, "rows")), counts[g])
    _ = trace^
    _ = workers^


def test_swapped_join_keeps_the_left_key_name_and_type() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    # The name keeps the left wider than its key alone, which the planner
    # would otherwise already probe from the larger side.
    var actual = joined().group_by(["d", "name"]).agg(sums()).collect()
    assert_true(swapped(observed.text()), "the join was not swapped")
    _ = observed^
    assert_equal(actual.columns(), ["d", "name", "total", "rows"])
    assert_true(actual.column("d").dtype() == DataType.INT32)
    var totals = List[Int64](length=DIMENSIONS, fill=0)
    var counts = List[Int64](length=DIMENSIONS, fill=0)
    var present = 0
    for row in range(FACT_ROWS):
        var d = fact_key(row)
        if d < DIMENSIONS and kept(row):
            if counts[d] == 0:
                present += 1
            totals[d] += fact_value(row)
            counts[d] += 1
    assert_equal(actual.height(), present)
    for r in range(actual.height()):
        var d = Int(num(actual.item(r, "d")))
        assert_equal(num(actual.item(r, "total")), totals[d])
        assert_equal(num(actual.item(r, "rows")), counts[d])
    _ = trace^
    _ = workers^


def test_ordered_group_bys_and_distinct_counts_swap_and_renamed_columns_keep_the_join_as_written() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    # An ordered group-by keeps its groups in first-occurrence order of
    # the rows it reads, but the join promises no order: it swaps, and
    # the groups are the same.
    var observed = StrategyTrace()
    var ordered = (
        joined()
        .group_by(["name", "grp"], maintain_order=True)
        .agg(sums())
        .collect()
    )
    assert_equal(swapped(observed.text()), True)
    _ = observed^
    assert_by_name_and_group(ordered)
    # A distinct count depends on each group's values only: the join
    # swaps, and the streamed batches are collected for one eager group-by
    # rather than merged as per-batch sets.
    observed = StrategyTrace()
    var distinct = (
        joined()
        .group_by(["grp"])
        .agg([col("v").n_unique().alias("values")])
        .collect()
    )
    assert_equal(swapped(observed.text()), True)
    _ = observed^
    var values = List[List[Int64]](length=7, fill=List[Int64]())
    for row in range(FACT_ROWS):
        var d = fact_key(row)
        if (
            d < DIMENSIONS
            and kept(row)
            and fact_value(row) not in values[d % 7]
        ):
            values[d % 7].append(fact_value(row))
    assert_equal(distinct.height(), 7)
    for r in range(distinct.height()):
        var g = Int(num(distinct.item(r, "grp")))
        assert_equal(num(distinct.item(r, "values")), Int64(len(values[g])))
    # A non-key name on both sides: the join suffixes the right one.
    observed = StrategyTrace()
    var suffixed = (
        joined(True).group_by(["grp", "grp_right"]).agg(sums()).collect()
    )
    assert_equal(swapped(observed.text()), False)
    _ = observed^
    var totals = Dict[String, Int64]()
    for row in range(FACT_ROWS):
        var d = fact_key(row)
        if d >= DIMENSIONS or not kept(row):
            continue
        var key = String(d % 7) + "|" + String(row % 3)
        totals[key] = totals.get(key, Int64(0)) + fact_value(row)
    assert_equal(suffixed.height(), len(totals))
    for r in range(suffixed.height()):
        var key = (
            String(num(suffixed.item(r, "grp")))
            + "|"
            + String(num(suffixed.item(r, "grp_right")))
        )
        assert_equal(num(suffixed.item(r, "total")), totals[key])
    _ = trace^
    _ = workers^
