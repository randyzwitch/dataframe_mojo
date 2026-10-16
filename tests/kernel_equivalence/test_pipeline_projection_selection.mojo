"""A pipeline's select of bare columns (renamed, reordered, some dropped)
keeps the morsel's selection instead of gathering the kept rows, and later
steps read through it: a filter (composing a mask with a mask, and narrowing
a join's row indices), a join and the grouped sink must give the scalar
model's groups, for dense and sparse selections and several worker counts."""
# contract-group: table.join
from std.testing import assert_equal, assert_true

from dataframe import Column, DataFrame, Series, col, lit
from tests.support.aggregate_table_reference import integer
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 60_000
comptime DIMENSIONS = 500


def fact() raises -> DataFrame:
    var k = List[Int64]()
    var g = List[Int64]()
    var x = List[Int64]()
    var x_valid = List[Bool]()
    var noise = List[Int64]()
    for row in range(ROWS):
        k.append(Int64((row * 7919) % (DIMENSIONS + 40)))
        g.append(Int64(row % 23))
        x.append(Int64(row % 101 - 50))
        x_valid.append(row % 17 != 3)
        noise.append(Int64(row))
    return DataFrame(
        [
            Series("k", Column[Int64](k^)),
            Series("g", Column[Int64](g^)),
            Series("x", Column[Int64](x^, x_valid^)),
            Series("noise", Column[Int64](noise^)),
        ]
    )


def dimension() raises -> DataFrame:
    var sk = List[Int64]()
    var band = List[Int64]()
    for d in range(DIMENSIONS):
        sk.append(Int64(d))
        band.append(Int64(d % 7))
    return DataFrame(
        [
            Series("d_sk", Column[Int64](sk^)),
            Series("band", Column[Int64](band^)),
        ]
    )


def kept(row: Int, keep: Int) -> Bool:
    """The first filter: every row whose position modulo 10 is below
    `keep` (dense at 6, sparse at 1)."""
    return row % 10 < keep


def check(keep: Int) raises:
    var observed = StrategyTrace()
    var result = (
        fact()
        .lazy()
        .filter(col("noise") % lit(Int64(10)) < lit(Int64(keep)))
        .select_exprs(
            [col("x").alias("value"), col("k").alias("key"), col("g")]
        )
        .join(dimension().lazy(), left_on=["key"], right_on=["d_sk"])
        .filter(col("band") != lit(Int64(3)))
        .group_by(["g", "band"])
        .agg(
            [
                col("value").sum(min_count=1).alias("total"),
                col("value").len().alias("rows"),
            ]
        )
        .collect()
    )
    assert_true(
        "dataframe-path: pipeline.projection_selection\n" in observed.text(),
        "the select did not keep the selection",
    )
    _ = observed^
    var totals = Dict[String, Int64]()
    var counts = Dict[String, Int]()
    var valid = Dict[String, Bool]()
    for row in range(ROWS):
        if not kept(row, keep):
            continue
        var d = (row * 7919) % (DIMENSIONS + 40)
        if d >= DIMENSIONS or d % 7 == 3:
            continue
        var key = String(row % 23) + "|" + String(d % 7)
        counts[key] = counts.get(key, 0) + 1
        if row % 17 != 3:
            totals[key] = totals.get(key, 0) + Int64(row % 101 - 50)
            valid[key] = True
    assert_equal(result.height(), len(counts))
    for r in range(result.height()):
        var key = (
            String(integer(result.item(r, "g")))
            + "|"
            + String(integer(result.item(r, "band")))
        )
        assert_true(key in counts, "unexpected group " + key)
        assert_equal(Int(integer(result.item(r, "rows"))), counts[key])
        if valid.get(key, False):
            assert_equal(Int64(integer(result.item(r, "total"))), totals[key])
        else:
            assert_true(result.item(r, "total").is_null())


def test_projections_keep_dense_and_sparse_selections() raises:
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    for threads in [1, 3, 8]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
        check(6)
        check(1)
        _ = workers^
    _ = trace^
