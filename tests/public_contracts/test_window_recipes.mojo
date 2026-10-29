"""SQL-style ranking recipes use existing expressions and preserve input rows."""
# contract-group: expr.ordered_over
# The 140 literal cells below were independently checked with DuckDB SQL:
# row_number, ntile, percent_rank, cume_dist, partitioned by g; ranking x
# in both directions with NULLS FIRST/LAST, and row positions ordered by t.
from dataframe import Column, DataFrame, Expr, Series, col, lit, when
from std.testing import assert_raises
from tests.support.coverage_compare import assert_frame


def _ntile(buckets: Int) raises -> Expr:
    if buckets < 1:
        raise Error("bucket count must be positive")
    var one = lit(Int64(1))
    var zero = lit(Int64(0))
    var n = col("x").len().over("g")
    var row = col("x").is_null().cum_count().over("g", order_by="t") - one
    var q = n // lit(Int64(buckets))
    var extra = n % lit(Int64(buckets))
    var boundary = (q + one) * extra
    var safe_q = when(q > zero).then(q).otherwise(one)
    return (
        when(row < boundary)
        .then(row // (q + one) + one)
        .otherwise((row - boundary) // safe_q + extra + one)
    )


def _sql_rank(method: String, descending: Bool, nulls_first: Bool) -> Expr:
    var ranked = col("x").rank(method, descending=descending).over("g")
    var missing = col("x").is_null()
    var nulls = missing.cast("int64").sum().over("g")
    if nulls_first:
        var null_rank = nulls.copy() if method == "max" else lit(Int64(1))
        return when(missing).then(null_rank).otherwise(ranked + nulls)
    var null_rank = col("x").len().over("g") if method == "max" else col(
        "x"
    ).count().over("g") + lit(Int64(1))
    return when(missing).then(null_rank).otherwise(ranked)


def _percent_rank(descending: Bool, nulls_first: Bool) -> Expr:
    var n = col("x").len().over("g")
    var one = lit(Int64(1))
    var denominator = when(n > one).then(n - one).otherwise(one)
    return (_sql_rank("min", descending, nulls_first) - one).cast(
        "float64"
    ) / denominator.cast("float64")


def _cume_dist(descending: Bool, nulls_first: Bool) -> Expr:
    return _sql_rank("max", descending, nulls_first).cast("float64") / col(
        "x"
    ).len().over("g").cast("float64")


def test_sql_ranking_recipes_with_ties_nulls_and_shuffled_partitions() raises:
    var frame = DataFrame(
        [
            Series("g", Column[Int64]([1, 2, 1, 3, 2, 1, 4, 4, 1, 1])),
            Series("t", Column[Int64]([3, 1, 0, 0, 0, 4, 1, 0, 2, 1])),
            Series(
                "x",
                Column[Int64](
                    [10, 20, 10, 0, 20, 0, 0, 0, 5, 5],
                    [
                        True,
                        True,
                        True,
                        False,
                        True,
                        False,
                        False,
                        False,
                        True,
                        True,
                    ],
                ),
            ),
        ]
    )
    var expressions: List[Expr] = [
        col("x").is_null().cum_count().over("g").alias("input_row"),
        col("x")
        .is_null()
        .cum_count()
        .over("g", order_by="t")
        .alias("ordered_row"),
        _ntile(1).alias("tile1"),
        _ntile(2).alias("tile2"),
        _ntile(3).alias("tile3"),
        _ntile(20).alias("tile20"),
        _percent_rank(False, False).alias("p_asc_last"),
        _cume_dist(False, False).alias("c_asc_last"),
        _percent_rank(False, True).alias("p_asc_first"),
        _cume_dist(False, True).alias("c_asc_first"),
        _percent_rank(True, False).alias("p_desc_last"),
        _cume_dist(True, False).alias("c_desc_last"),
        _percent_rank(True, True).alias("p_desc_first"),
        _cume_dist(True, True).alias("c_desc_first"),
    ]
    var expected = DataFrame(
        [
            Series("input_row", Column[Int64]([1, 1, 2, 1, 2, 3, 1, 2, 4, 5])),
            Series(
                "ordered_row", Column[Int64]([4, 2, 1, 1, 1, 5, 2, 1, 3, 2])
            ),
            Series("tile1", Column[Int64]([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])),
            Series("tile2", Column[Int64]([2, 2, 1, 1, 1, 2, 2, 1, 1, 1])),
            Series("tile3", Column[Int64]([2, 2, 1, 1, 1, 3, 2, 1, 2, 1])),
            Series("tile20", Column[Int64]([4, 2, 1, 1, 1, 5, 2, 1, 3, 2])),
            Series(
                "p_asc_last",
                Column[Float64]([0.5, 0, 0.5, 0, 0, 1, 0, 0, 0, 0]),
            ),
            Series(
                "c_asc_last",
                Column[Float64]([0.8, 1, 0.8, 1, 1, 1, 1, 1, 0.4, 0.4]),
            ),
            Series(
                "p_asc_first",
                Column[Float64]([0.75, 0, 0.75, 0, 0, 0, 0, 0, 0.25, 0.25]),
            ),
            Series(
                "c_asc_first",
                Column[Float64]([1, 1, 1, 1, 1, 0.2, 1, 1, 0.6, 0.6]),
            ),
            Series(
                "p_desc_last",
                Column[Float64]([0, 0, 0, 0, 0, 1, 0, 0, 0.5, 0.5]),
            ),
            Series(
                "c_desc_last",
                Column[Float64]([0.4, 1, 0.4, 1, 1, 1, 1, 1, 0.8, 0.8]),
            ),
            Series(
                "p_desc_first",
                Column[Float64]([0.25, 0, 0.25, 0, 0, 0, 0, 0, 0.75, 0.75]),
            ),
            Series(
                "c_desc_first",
                Column[Float64]([0.6, 1, 0.6, 1, 1, 0.2, 1, 1, 1, 1]),
            ),
        ]
    )
    assert_frame(frame.select_exprs(expressions), expected)
    for optimize in [False, True]:
        for streaming in [False, True]:
            assert_frame(
                frame.lazy()
                .select_exprs(expressions)
                .collect(
                    engine="cpu",
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=1,
                ),
                expected,
            )
    assert_frame(frame.clear().select_exprs(expressions), expected.clear())
    with assert_raises(contains="positive"):
        _ = _ntile(0)
    with assert_raises(contains="positive"):
        _ = _ntile(-1)


def test_ntile_gives_the_first_remainder_buckets_one_extra_row() raises:
    # Six rows / four buckets must have sizes 2, 2, 1, 1. Multiplying a
    # row fraction by four instead gives 2, 1, 2, 1 and is not SQL NTILE.
    # DuckDB 2.0.0 independently gives [4, 1, 2, 1, 3, 2] in input order:
    # SELECT ntile(4) OVER (PARTITION BY g ORDER BY t) FROM items ORDER BY id.
    # The t permutation also checks scattering back to original row order.
    var frame = DataFrame(
        [
            Series("g", Column[Int64]([1, 1, 1, 1, 1, 1])),
            Series("t", Column[Int64]([5, 0, 3, 1, 4, 2])),
            Series(
                "x",
                Column[Int64](
                    [0, 0, 0, 0, 0, 0],
                    [False, False, False, False, False, False],
                ),
            ),
        ]
    )
    var expressions: List[Expr] = [_ntile(4).alias("bucket")]
    var expected = DataFrame(
        [
            Series("bucket", Column[Int64]([4, 1, 2, 1, 3, 2])),
        ]
    )
    assert_frame(frame.select_exprs(expressions), expected)
    for optimize in [False, True]:
        for streaming in [False, True]:
            assert_frame(
                frame.lazy()
                .select_exprs(expressions)
                .collect(
                    engine="cpu",
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=1,
                ),
                expected,
            )
    assert_frame(frame.clear().select_exprs(expressions), expected.clear())
