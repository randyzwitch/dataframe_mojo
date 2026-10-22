"""Implicit same-scale Decimal literal casts in original grouped operands."""
# contract-group: execution.backends
# contract-group: execution.apple_parity
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.binding import BoundExpr, bind, ROWS
from dataframe_metal.accelerator import decimal_group_feed
from dataframe_metal.accelerator.grouped_feeds import has_checked_grouped_feeds
from dataframe_metal.accelerator.constants import fold_literal_text_parsing
from dataframe_metal.accelerator.joins import lower_join_dag
from dataframe_metal.accelerator.rows import lower_rows
from dataframe_metal.accelerator.gpu.runtime import capabilities
from dataframe_metal.accelerator.gpu.program import Program
from dataframe_metal.metal import MetalRuntime
from tests.execution_modes.test_backend_metal import require_metal
from tests.support.coverage_generators import offset_layout, chunk_layout


def source(nullable: Bool = False) raises -> DataFrame:
    return DataFrame(
        [
            Series("g", Column[Int32]([0, 1, 0, 1])),
            Series("before", Column[Bool]([True, False, True, False])),
            Series(
                "price",
                Column[Int32](
                    [1000, 2000, 3000, 4000], [True, True, not nullable, True]
                ),
            ).with_dtype(DataType.decimal(7, 2, 32)),
            Series(
                "refund",
                Column[Int32]([100, 200, 0, 500], [True, True, False, True]),
            ).with_dtype(DataType.decimal(7, 2, 32)),
        ]
    )


def expressions() raises -> List[Expr]:
    # The public like(frame, "price", "0") helper retains the actual input
    # width; Parquet narrows these precision7 money columns to32bits.
    var zero = lit("0").cast(DataType.decimal(7, 2, 32))
    var net = col("price") - col("refund").fill_null(zero)
    return [
        when(col("before"))
        .then(net)
        .otherwise(zero)
        .sum()
        .alias("before_total"),
        when(~col("before"))
        .then(net)
        .otherwise(zero)
        .sum()
        .alias("after_total"),
    ]


def test_decimal_same_scale_selection_literals_and_cast_refusals() raises:
    var frame = source()
    var original = (
        frame.lazy().group_by("g", maintain_order=True).agg(expressions())
    )
    # Public eager dispatch must reach the same original mode7 proof as
    # metadata lowering, rather than the legacy text/List feed adapter.
    assert_true(not has_checked_grouped_feeds(original, eager_feed=True))
    var query = fold_literal_text_parsing(original)
    assert_true(not has_checked_grouped_feeds(query, eager_feed=True))
    var failed_source = original.copy()
    failed_source._nodes[0].source_failure = "original source failure"
    assert_true(has_checked_grouped_feeds(failed_source, eager_feed=True))
    var mixed = expressions()
    var foreign = (
        col("g")
        .cast(DataType.INT64)
        .fill_null(lit("bad").cast(DataType.INT64))
        .sum()
        .alias("foreign")
    )
    var foreign_bound = bind(foreign, frame._columns)
    assert_equal(
        foreign_bound.shapes[
            foreign_bound.expr._nodes[len(foreign_bound.expr._nodes) - 1].left
        ],
        ROWS,
    )
    mixed.append(foreign.copy())
    var invalid_literal = frame.lazy().group_by("g").agg(mixed)
    assert_true(has_checked_grouped_feeds(invalid_literal, eager_feed=True))
    var bounds = List[BoundExpr]()
    for expression in query._nodes[1].exprs:
        bounds.append(bind(expression, frame._columns))
    assert_true(decimal_group_feed.eligible(bounds, frame._columns, ["g"]))
    var dag = lower_join_dag(query, capabilities())
    var proved = 0
    for task in dag.tasks:
        for node in task.query._nodes:
            if node.text == decimal_group_feed.MATERIALIZATION:
                proved += 1
                var plan = lower_rows(
                    task.query, capabilities(), terminal_boundary=True
                )
                assert_equal(plan.ordered_mode, 7)
                _ = Program(plan)
    assert_equal(proved, 1)
    var literal = fold_literal_text_parsing(
        frame.lazy().select(lit("9.99").cast(DataType.decimal(3, 2)))
    )
    var value = bind(literal._nodes[1].exprs[0], frame._columns)
    assert_true(
        decimal_group_feed._selection_branch(
            value, 0, DataType.decimal(3, 2, 32)
        )
    )
    assert_true(
        not decimal_group_feed._selection_branch(
            value, 0, DataType.decimal(3, 1, 32)
        )
    )
    var outside = value.copy()
    outside.expr._nodes[0].integer = 1000
    assert_true(
        not decimal_group_feed._selection_branch(
            outside, 0, DataType.decimal(3, 2, 32)
        )
    )
    var forged_high = value.copy()
    forged_high.expr._nodes[0].text2 = "1"
    assert_true(
        not decimal_group_feed._selection_branch(
            forged_high, 0, DataType.decimal(3, 2, 32)
        )
    )
    var forged32 = value.copy()
    forged32.dtypes[0] = DataType.decimal(3, 2, 32)
    forged32.expr._nodes[0].integer = 3000000000
    assert_true(
        not decimal_group_feed._selection_branch(
            forged32, 0, DataType.decimal(10, 2, 64)
        )
    )
    var forged64 = value.copy()
    forged64.dtypes[0] = DataType.decimal(3, 2, 64)
    forged64.expr._nodes[0].text2 = "1"
    assert_true(
        not decimal_group_feed._selection_branch(
            forged64, 0, DataType.decimal(4, 2, 64)
        )
    )
    # A raw source column is not a constant, and its declared precision is
    # not a checked value bound. CPU's unchosen common-type cast still fails.
    var raw = DataFrame(
        [
            Series("g", Column[Int32]([0])),
            Series("p", Column[Bool]([False])),
            Series("a", Column[Int64]([10000])).with_dtype(
                DataType.decimal(3, 2, 64)
            ),
            Series("b", Column[Int32]([0])).with_dtype(
                DataType.decimal(4, 2, 32)
            ),
        ]
    )
    var unsafe = when(col("p")).then(col("a")).otherwise(col("b"))
    var unsafe_bound = bind(unsafe, raw._columns)
    assert_true(not decimal_group_feed.operand(unsafe_bound)[0])
    with assert_raises(contains="cast from decimal"):
        _ = (
            raw.lazy()
            .group_by("g")
            .agg(unsafe.sum())
            .collect(engine="cpu", optimize=False, streaming=False)
        )
    var unsafe_fill = bind(col("a").fill_null(col("b")), raw._columns)
    assert_true(not decimal_group_feed.operand(unsafe_fill)[0])


def literal(actual: DataFrame, nullable: Bool) raises:
    assert_equal(actual.column("g").int32().to_list(), [Int32(0), Int32(1)])
    assert_equal(actual.column("before_total").dtype(), DataType.decimal(38, 2))
    assert_equal(actual.column("after_total").dtype(), DataType.decimal(38, 2))
    assert_equal(
        actual.column("before_total")._decimal_raw(0),
        Int128(900 if nullable else 3900),
    )
    assert_equal(actual.column("before_total")._decimal_raw(1), Int128(0))
    assert_equal(actual.column("after_total")._decimal_raw(0), Int128(0))
    assert_equal(actual.column("after_total")._decimal_raw(1), Int128(5300))


# requires: metal
def test_mojo_gpu_decimal_same_scale_selection_nulls_masks_and_layouts() raises:
    require_metal()
    for nullable in [False, True]:
        var frame = source(nullable)
        for layout in range(3):
            var input = (
                frame.copy() if layout
                == 0 else offset_layout(frame) if layout
                == 1 else chunk_layout(frame)
            )
            var query = (
                input.lazy()
                .group_by("g", maintain_order=True)
                .agg(expressions())
            )
            for streaming in [False, True]:
                literal(
                    query.collect(
                        engine="cpu",
                        optimize=False,
                        streaming=streaming,
                        batch_size=1,
                    ),
                    nullable,
                )
                literal(
                    query.collect(
                        accelerator=MetalRuntime(),
                        optimize=False,
                        streaming=streaming,
                        batch_size=1,
                    ),
                    nullable,
                )


# requires: metal
def test_mojo_gpu_decimal_same_scale_fill_observes_both_children_under_when() raises:
    require_metal()
    var frame = DataFrame(
        [
            Series("g", Column[Int32]([0])),
            Series("p", Column[Bool]([False])),
            Series("a", Column[Int32]([1])).with_dtype(
                DataType.decimal(3, 2, 32)
            ),
            Series("b", Column[Int32]([2])).with_dtype(
                DataType.decimal(3, 2, 32)
            ),
            Series("c", Column[Int32]([10000])).with_dtype(
                DataType.decimal(3, 2, 32)
            ),
        ]
    )
    var fill = (col("a") + col("b")).fill_null(col("c") + col("b"))
    var choice = (
        when(col("p"))
        .then(fill)
        .otherwise(lit("0").cast(DataType.decimal(3, 2)))
    )
    for streaming in [False, True]:
        var masked = frame.lazy().group_by("g").agg(choice.sum().alias("v"))
        var cpu = masked.collect(
            engine="cpu", optimize=False, streaming=streaming, batch_size=1
        )
        var gpu = masked.collect(
            accelerator=MetalRuntime(),
            optimize=False,
            streaming=streaming,
            batch_size=1,
        )
        assert_equal(cpu.column("v")._decimal_raw(0), Int128(0))
        assert_equal(gpu.column("v")._decimal_raw(0), Int128(0))
        var selected = (
            frame.with_column(Series("p", Column[Bool]([True])))
            .lazy()
            .group_by("g")
            .agg(choice.sum(min_count=999).alias("v"))
            .head(0)
        )
        with assert_raises(contains="decimal precision"):
            _ = selected.collect(
                engine="cpu", optimize=False, streaming=streaming, batch_size=1
            )
        with assert_raises(contains="decimal precision"):
            _ = selected.collect(
                accelerator=MetalRuntime(),
                optimize=False,
                streaming=streaming,
                batch_size=1,
            )
