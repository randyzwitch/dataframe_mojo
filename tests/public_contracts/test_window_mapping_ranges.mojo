"""Frame-height expressions and window mappings have explicit row axes."""
# contract-group: expr.ordered_over
from std.testing import assert_equal, assert_raises
from dataframe import (
    Column,
    DataFrame,
    Expr,
    ListColumn,
    Series,
    col,
    int_range,
    len as row_len,
    lit,
    null,
)
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import chunk_layout, offset_layout


def _frame() raises -> DataFrame:
    return DataFrame(
        [
            Series("g", Column[Int64]([2, 1, 2, 1, 3, 3])),
            Series("t", Column[Int64]([2, 1, 0, 0, 0, 1])),
            Series(
                "x",
                Column[Int64](
                    [10, 20, 30, 40, 0, 60],
                    [True, True, True, True, False, True],
                ),
            ),
        ]
    )


def _check(
    frame: DataFrame, expressions: List[Expr], expected: DataFrame
) raises:
    for source in [frame.copy(), chunk_layout(frame), offset_layout(frame)]:
        assert_frame(source.select_exprs(expressions), expected)
        for optimize in [False, True]:
            for streaming in [False, True]:
                assert_frame(
                    source.lazy()
                    .select_exprs(expressions)
                    .collect(
                        engine="cpu",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=1,
                    ),
                    expected,
                )


def test_free_height_and_range_work_without_payload_columns() raises:
    var frame = DataFrame(List[Series](), height=3)
    var count: List[Expr] = [row_len().alias("n")]
    var expected = DataFrame([Series("n", Column[Int64]([3]))])
    assert_frame(frame.select_exprs(count), expected)
    assert_frame(
        frame.lazy().select_exprs(count).collect(engine="cpu"), expected
    )
    var rows: List[Expr] = [int_range(row_len()).alias("i")]
    var indices = DataFrame([Series("i", Column[Int64]([0, 1, 2]))])
    assert_frame(frame.select_exprs(rows), indices)
    assert_frame(
        frame.lazy().select_exprs(rows).collect(engine="cpu", batch_size=1),
        indices,
    )
    var empty = frame.clear()
    assert_frame(
        empty.select_exprs(count), DataFrame([Series("n", Column[Int64]([0]))])
    )
    assert_frame(empty.select_exprs(rows), indices.clear())
    assert_frame(
        empty.lazy().select_exprs(rows).collect(engine="cpu"), indices.clear()
    )


def test_free_range_restarts_per_group_and_order_and_includes_null_rows() raises:
    var frame = _frame()
    var expressions: List[Expr] = [
        row_len().over("g").alias("n"),
        int_range(row_len()).over("g").alias("input"),
        int_range(0, row_len()).over("g", order_by="t").alias("ordered"),
    ]
    _check(
        frame,
        expressions,
        DataFrame(
            [
                Series("n", Column[Int64]([2, 2, 2, 2, 2, 2])),
                Series("input", Column[Int64]([0, 0, 1, 1, 0, 1])),
                Series("ordered", Column[Int64]([1, 1, 0, 0, 0, 1])),
            ]
        ),
    )
    var grouped = frame.group_by("g").agg(row_len().alias("n"))
    assert_frame(
        grouped,
        DataFrame(
            [
                Series("g", Column[Int64]([2, 1, 3])),
                Series("n", Column[Int64]([2, 2, 2])),
            ]
        ),
    )
    assert_frame(
        frame.lazy()
        .group_by("g")
        .agg(row_len().alias("n"))
        .collect(engine="cpu"),
        grouped,
    )
    var lists = frame.group_by("g").agg(int_range(row_len()).alias("indices"))
    assert_frame(
        lists,
        DataFrame(
            [
                Series("g", Column[Int64]([2, 1, 3])),
                Series(
                    "indices",
                    ListColumn(
                        [Int64(0), 2, 4, 6],
                        Series("item", Column[Int64]([0, 1, 0, 1, 0, 1])),
                    ),
                ),
            ]
        ),
    )


def test_range_exclusive_bounds_steps_empty_and_checked_scalar_inputs() raises:
    var empty = DataFrame(List[Series]())
    assert_frame(
        empty.select(int_range(-2, 5, step=2).alias("i")),
        DataFrame([Series("i", Column[Int64]([-2, 0, 2, 4]))]),
    )
    assert_frame(
        empty.select(int_range(4, -3, step=-2).alias("i")),
        DataFrame([Series("i", Column[Int64]([4, 2, 0, -2]))]),
    )
    assert_frame(
        empty.select(int_range(0, 3, step=-1).alias("i")),
        DataFrame([Series("i", Column[Int64](List[Int64]()))]),
    )
    assert_frame(
        empty.select(
            int_range(lit(Int64.MAX - 1), lit(Int64.MAX), step=2).alias("i")
        ),
        DataFrame([Series("i", Column[Int64]([Int64.MAX - 1]))]),
    )
    assert_frame(
        empty.select(
            int_range(lit(Int64.MIN + 1), lit(Int64.MIN), step=-2).alias("i")
        ),
        DataFrame([Series("i", Column[Int64]([Int64.MIN + 1]))]),
    )
    with assert_raises(contains="nonzero"):
        _ = int_range(0, 3, step=0)
    with assert_raises(contains="non-null"):
        _ = empty.select(int_range(null("int64"), 3))
    with assert_raises(contains="integer"):
        _ = empty.select(int_range(0.5, 3))
    with assert_raises(contains="scalar or aggregate"):
        _ = _frame().select(int_range(col("x")))


def test_join_mapping_lists_and_aggregate_scalars_restore_original_rows() raises:
    var frame = _frame()
    var expected_lists = DataFrame(
        [
            Series(
                "j",
                ListColumn(
                    [Int64(0), 2, 4, 6, 8, 10, 12],
                    Series(
                        "item",
                        Column[Int64](
                            [30, 10, 40, 20, 30, 10, 40, 20, 0, 60, 0, 60],
                            [
                                True,
                                True,
                                True,
                                True,
                                True,
                                True,
                                True,
                                True,
                                False,
                                True,
                                False,
                                True,
                            ],
                        ),
                    ),
                ),
            ),
        ]
    )
    _check(
        frame,
        [col("x").over("g", order_by="t", mapping_strategy="join").alias("j")],
        expected_lists,
    )
    _check(
        frame,
        [
            col("x")
            .implode()
            .over("g", order_by="t", mapping_strategy="join")
            .alias("j")
        ],
        expected_lists,
    )
    _check(
        frame,
        [col("x").sum().over("g", mapping_strategy="join").alias("s")],
        DataFrame([Series("s", Column[Int64]([40, 60, 40, 60, 60, 60]))]),
    )
    assert_frame(
        frame.clear().select(
            col("x").over("g", mapping_strategy="join").alias("j")
        ),
        expected_lists.clear(),
    )


def test_explode_mapping_uses_partition_appearance_then_local_order() raises:
    var frame = _frame()
    _check(
        frame,
        [
            col("x")
            .over("g", order_by="t", mapping_strategy="explode")
            .alias("e")
        ],
        DataFrame(
            [
                Series(
                    "e",
                    Column[Int64](
                        [30, 10, 40, 20, 0, 60],
                        [True, True, True, True, False, True],
                    ),
                )
            ]
        ),
    )
    _check(
        frame,
        [col("x").sum().over("g", mapping_strategy="explode").alias("e")],
        DataFrame([Series("e", Column[Int64]([40, 60, 60]))]),
    )
    _check(
        frame,
        [
            col("x")
            .filter(col("x") > 25)
            .over("g", order_by="t", mapping_strategy="explode")
            .alias("e")
        ],
        DataFrame([Series("e", Column[Int64]([30, 40, 60]))]),
    )
    _check(
        frame,
        [lit(Int64(7)).over("g", mapping_strategy="explode").alias("e")],
        DataFrame([Series("e", Column[Int64]([7, 7, 7]))]),
    )
    assert_frame(
        frame.select(
            col("x")
            .sum()
            .over(order_by="t", mapping_strategy="explode")
            .alias("e")
        ),
        DataFrame([Series("e", Column[Int64]([160]))]),
    )
    assert_frame(
        frame.clear().select(
            col("x").sum().over("g", mapping_strategy="explode").alias("e")
        ),
        DataFrame([Series("e", Column[Int64](List[Int64]()))]),
    )
    with assert_raises(contains="length"):
        _ = frame.select_exprs(
            [col("g"), col("x").sum().over("g", mapping_strategy="explode")]
        )
    with assert_raises(contains="mapping_strategy"):
        _ = frame.select(col("x").over("g", mapping_strategy="invalid"))
    with assert_raises(contains="different length"):
        _ = frame.select(col("x").head(1).over("g"))


def test_range_aggregate_bounds_use_whole_input_before_generating_rows() raises:
    var frame = DataFrame([Series("x", Column[Int64]([2, 3]))])
    var expected = DataFrame([Series("i", Column[Int64]([0, 1, 2, 3, 4]))])
    _check(frame, [int_range(col("x").sum()).alias("i")], expected)
    var expected_count = DataFrame([Series("n", Column[Int64]([5]))])
    _check(frame, [int_range(col("x").sum()).len().alias("n")], expected_count)


def test_range_positions_do_not_restart_in_streaming_batches() raises:
    var frame = _frame()
    var expressions: List[Expr] = [
        col("x"),
        int_range(lit(Int64(6))).alias("i"),
    ]
    var expected = DataFrame(
        [frame.column("x"), Series("i", Column[Int64]([0, 1, 2, 3, 4, 5]))]
    )
    _check(frame, expressions, expected)
    for optimize in [False, True]:
        for streaming in [False, True]:
            var result = (
                frame.lazy()
                .with_columns(int_range(lit(Int64(6))).alias("i"))
                .collect(
                    engine="cpu",
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=2,
                )
            )
            assert_frame(result.select_exprs(expressions), expected)
    _check(
        frame,
        [int_range(lit(Int64(6))).sum().alias("s")],
        DataFrame([Series("s", Column[Int64]([15]))]),
    )
