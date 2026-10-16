"""CUT normalization binds metadata without observing invalid edge controls."""
# contract-group: expr.cut
# contract-group: public.apple_parity
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit
from dataframe.accelerator.cuts import (
    has_cut_operations,
    has_cut_materializations,
    lower_cuts,
    control_failure,
    staging_bytes,
)
from dataframe.accelerator.metadata import schema_frame
from dataframe.expr import CUT, QCUT, CAST, POW, ADD, _node, _append_shifted
from dataframe.parse import parse_int64
from dataframe.binding import bind, ROWS
from dataframe.accelerator.constants import has_literal_text_parsing


def _ordered_scalar_payload(
    frame: DataFrame, value: Expr
) raises -> Tuple[Expr, List[String]]:
    var query = lower_cuts(
        frame.lazy().select_exprs(
            [
                value.copy().alias("checked"),
                col("x").cut([0], ["lower", "upper"]).alias("bin"),
            ]
        )
    )
    for operation in query._nodes:
        if (
            len(operation.names2) >= 3
            and operation.names2[1] == "ordered_scalar_v1"
        ):
            assert_equal(len(operation.exprs), 1)
            assert_equal(
                len(operation.names2), len(operation.exprs[0]._nodes) + 3
            )
            assert_true(Bool(operation.names2[2]))
            return (operation.exprs[0].copy(), operation.names2.copy())
    raise Error("Ordered CUT scalar producer is missing")


def _check_scalar_occurrence_ranks(frame: DataFrame) raises:
    var bad_cast = lit(Int64(128)).cast(DataType.INT8).cast(DataType.INT64)
    var bad_power = lit(Int64(2)).pow(lit(Int64(-1)))
    # Original storage puts the right subtree FIRST. CPU visits the left
    # subtree first, so ranking raw node indices would report the wrong fault.
    var nodes = bad_power._nodes.copy()
    var right = len(nodes) - 1
    _append_shifted(nodes, bad_cast._nodes, len(nodes))
    var left = len(nodes) - 1
    nodes.append(_node(ADD, left, right))
    var expression = Expr(nodes^, "checked")
    var scalar = _ordered_scalar_payload(frame, expression)
    var cast_rank = Int64(0)
    var power_rank = Int64(0)
    for index in range(len(scalar[0]._nodes)):
        ref node = scalar[0]._nodes[index]
        if node.op == CAST and node.dtypes[0].value() == DataType.INT8:
            cast_rank = parse_int64(scalar[1][index + 3])
        elif node.op == POW:
            power_rank = parse_int64(scalar[1][index + 3])
    assert_true(cast_rank > 0 and power_rank > cast_rank)
    # Shared checked children need separate occurrence ranks, not one rank
    # for a reused original node. The traversal clones both operand visits.
    nodes = bad_cast._nodes.copy()
    var root = len(nodes) - 1
    nodes.append(_node(ADD, root, root))
    expression = Expr(nodes^, "checked")
    scalar = _ordered_scalar_payload(frame, expression)
    var previous = Int64(0)
    var visits = 0
    for index in range(len(scalar[0]._nodes)):
        ref node = scalar[0]._nodes[index]
        if node.op == CAST and node.dtypes[0].value() == DataType.INT8:
            var rank = parse_int64(scalar[1][index + 3])
            assert_true(rank > previous)
            previous = rank
            visits += 1
    assert_equal(visits, 2)
    # A text lookup would replace the AST and needs a separate rank mapping.
    # Plain String literals remain admitted (covered by the GPU sibling test).
    with assert_raises(contains="string lookup rank remapping"):
        _ = lower_cuts(
            frame.lazy().select_exprs(
                [
                    lit("abc").str().to_uppercase().alias("text"),
                    col("x").cut([0], ["lower", "upper"]).alias("bin"),
                ]
            )
        )


def test_cut_lowering_preserves_string_struct_schema_and_observed_producers() raises:
    var frame = DataFrame([Series("x", Column[Int64]([0, 1, 2]))])
    for packed in [False, True]:
        var query = frame.lazy().select(
            col("x").cut([0, 1], include_breaks=packed).alias("out")
        )
        assert_true(has_cut_operations(query))
        var lowered = lower_cuts(query)
        assert_true(not has_cut_operations(lowered))
        assert_true(has_cut_materializations(lowered))
        var expected = schema_frame(query, len(query._nodes) - 1)
        var actual = schema_frame(lowered, len(lowered._nodes) - 1)
        assert_equal(actual.columns(), expected.columns())
        assert_equal(actual.dtypes(), expected.dtypes())
        assert_equal(actual.height(), 0)


def test_cut_lowering_defers_edge_order_but_keeps_binder_failure() raises:
    var frame = DataFrame([Series("x", Column[Int64]([Int64.MAX]))])
    var lowered = lower_cuts(frame.lazy().select((col("x") + 1).cut([1, 1])))
    assert_equal(
        control_failure(lowered),
        "breaks must be unique and strictly increasing",
    )
    assert_equal(schema_frame(lowered, len(lowered._nodes) - 1).height(), 0)
    var marked = 0
    for operation in lowered._nodes:
        if operation.source_failure:
            marked += 1
            assert_equal(operation.text, "metal_window_materialization")
    assert_equal(marked, 1)
    with assert_raises(contains="labels must have one more"):
        _ = lower_cuts(
            frame.lazy().select((col("x") + 1).cut([1, 1], ["wrong"]))
        )
    var dynamic = lower_cuts(
        frame.lazy().select(col("x").qcut([0.5]).alias("bin"))
    )
    assert_true(not has_cut_operations(dynamic))
    assert_true(has_cut_materializations(dynamic))
    assert_equal(control_failure(dynamic), "")
    var dynamic_schema = schema_frame(dynamic, len(dynamic._nodes) - 1)
    assert_equal(dynamic_schema.columns(), ["bin"])
    assert_equal(dynamic_schema.dtypes(), [DataType.STRING])
    assert_equal(dynamic_schema.height(), 0)


def test_cut_lowering_checked_scalar_metadata_uses_empty_literal_anchor() raises:
    var frame = DataFrame([Series("x", Column[Int64]([0]))])
    var lowered = lower_cuts(
        frame.lazy().select_exprs(
            [
                lit(Int64(128)).cast(DataType.INT8).alias("checked"),
                col("x").sum().alias("total"),
                col("x").cut([1, 1]).alias("bin"),
            ]
        )
    )
    # Schema discovery must not evaluate the earlier unsafe scalar cast, or
    # prematurely raise the later invalid CUT controls.
    var schema = schema_frame(lowered, len(lowered._nodes) - 1)
    assert_equal(schema.height(), 0)
    assert_equal(
        schema.dtypes(), [DataType.INT8, DataType.INT64, DataType.STRING]
    )
    assert_equal(staging_bytes(lowered), 2)
    # The unsafe CAST operand must itself be ROWS-shaped. Protecting only the
    # final result with WHEN would leave eager constant folding and the strict
    # scalar CAST admission guard active before the earlier/later CUT boundary.
    assert_true(not has_literal_text_parsing(lowered))
    var checked_casts = 0
    for operation in lowered._nodes:
        if operation.left < 0:
            continue
        var source = schema_frame(lowered, operation.left)
        for expression in operation.exprs:
            var bound = bind(expression, source._columns)
            for index in range(len(expression._nodes)):
                ref node = expression._nodes[index]
                if node.op == CAST and bound.dtypes[index] == DataType.INT8:
                    checked_casts += 1
                    assert_equal(bound.shapes[node.left], ROWS)
                    assert_equal(bound.shapes[index], ROWS)
    assert_equal(checked_casts, 1)
    _check_scalar_occurrence_ranks(frame)
    assert_equal(
        control_failure(lowered),
        "breaks must be unique and strictly increasing",
    )


def test_cut_lowering_literal_window_input_rejects_before_any_column_access() raises:
    for rows in [0, 2]:
        var ordinary = DataFrame(
            [Series("x", Column[Int64](List[Int64](length=rows, fill=0)))]
        )
        var no_columns = DataFrame(List[Series](), height=rows)
        for source in [ordinary.copy(), no_columns.copy()]:
            for sibling in [False, True]:
                var expressions: List[Expr] = [
                    lit(Float64(1)).cut([0]).alias("bin")
                ]
                if sibling:
                    expressions.insert(0, lit(Int64(7)).alias("scalar"))
                var query = source.lazy().select_exprs(expressions)
                with assert_raises(
                    contains="Window operations require a row-valued input"
                ):
                    _ = lower_cuts(query)
