"""A streamed join whose build has no rows. An inner, semi or right join
then passes no probe row, so the stream keeps its operators and sink and
reads nothing from its source. Each query is compared with the same query
over a probe that a filter empties first, a different route to the same
empty input: grouped and ungrouped sinks, joins above and below the empty
build (left, right and full ones that emit build rows), and two empty
builds in one stream, on one and eight workers."""
# contract-group: table.join
from dataframe import Column, DataFrame, LazyFrame, Series, col, lit
from tests.support.coverage_compare import assert_frame_unordered
from tests.support.lifecycle_jobs import WorkerSetting


def probe() raises -> DataFrame:
    var k = List[Int64]()
    var d = List[Int64]()
    var x = List[Int64]()
    for i in range(5_000):
        k.append(Int64(i % 97))
        d.append(Int64(i % 13))
        x.append(Int64(i))
    return DataFrame(
        [
            Series("k", Column[Int64](k^)),
            Series("d", Column[Int64](d^)),
            Series("x", Column[Int64](x^)),
        ]
    )


def dims() raises -> DataFrame:
    var d = List[Int64]()
    var name = List[Int64]()
    for i in range(20):
        d.append(Int64(i))
        name.append(Int64(100 + i))
    return DataFrame(
        [
            Series("dk", Column[Int64](d^)),
            Series("dname", Column[Int64](name^)),
        ]
    )


def empty_build() raises -> DataFrame:
    return DataFrame(
        [
            Series("bk", Column[Int64](List[Int64]())),
            Series("bv", Column[Int64](List[Int64]())),
        ]
    )


def emptied(source: LazyFrame) -> LazyFrame:
    """The same rows' columns with every row filtered out."""
    return source.filter(col("x") < lit(Int64(-1)))


def check(built: LazyFrame, expected: LazyFrame) raises:
    assert_frame_unordered(built.collect(), expected.collect())


def queries(workers: Int) raises:
    var setting = WorkerSetting(workers)
    var source = probe().lazy()
    var nothing = empty_build().lazy()
    var dimension = dims().lazy()
    # Grouped sink, a left join above the empty inner build.
    check(
        source.join(nothing, left_on=["k"], right_on=["bk"])
        .join(dimension, left_on=["d"], right_on=["dk"], how="left")
        .group_by(["k"])
        .agg([col("x").sum().alias("s")]),
        emptied(source)
        .join(nothing, left_on=["k"], right_on=["bk"], how="left")
        .join(dimension, left_on=["d"], right_on=["dk"], how="left")
        .group_by(["k"])
        .agg([col("x").sum().alias("s")]),
    )
    # Ungrouped sink: one row of reductions over no rows.
    check(
        source.join(
            nothing, left_on=["k"], right_on=["bk"], how="semi"
        ).select_exprs(
            [col("x").sum().alias("s"), col("x").count().alias("n")]
        ),
        emptied(source).select_exprs(
            [col("x").sum().alias("s"), col("x").count().alias("n")]
        ),
    )
    # Joins above the empty build that emit their unmatched build rows.
    for how in ["right", "full"]:
        check(
            source.join(nothing, left_on=["k"], right_on=["bk"]).join(
                dimension, left_on=["d"], right_on=["dk"], how=how
            ),
            emptied(source)
            .join(nothing, left_on=["k"], right_on=["bk"], how="left")
            .join(dimension, left_on=["d"], right_on=["dk"], how=how),
        )
    # A join below the empty build, and a second empty build.
    check(
        source.join(dimension, left_on=["d"], right_on=["dk"])
        .join(nothing, left_on=["k"], right_on=["bk"])
        .join(
            nothing.select_exprs(
                [col("bk").alias("ck"), col("bv").alias("cv")]
            ),
            left_on=["x"],
            right_on=["ck"],
        )
        .select(["k", "x", "dname", "bv", "cv"]),
        emptied(source)
        .join(dimension, left_on=["d"], right_on=["dk"])
        .join(nothing, left_on=["k"], right_on=["bk"], how="left")
        .join(
            nothing.select_exprs(
                [col("bk").alias("ck"), col("bv").alias("cv")]
            ),
            left_on=["x"],
            right_on=["ck"],
            how="left",
        )
        .select(["k", "x", "dname", "bv", "cv"]),
    )
    _ = setting^


def test_empty_build_streams_match_an_emptied_probe() raises:
    queries(1)
    queries(8)
