"""Original numeric operand certificates and scalar-literal ROWS lifting."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    lit,
    when,
)
from dataframe.binding import bind, ROWS, SCALAR
from dataframe.expr import COL, LIT_INT, LIT_FLOAT, LIT_BOOL, LIT_NULL
from dataframe.accelerator.constants import has_literal_text_parsing
from dataframe.accelerator.partition_operands import (
    certificate,
    lift_literals,
    prepare,
)


# requires: mojo_gpu
def test_partition_operand_metadata_scalar_faults_masks_and_numeric_anchor() raises:
    var source = DataFrame(
        [
            Series("key", StringColumn(["a", "b", "a"])),
            Series("x", Column[Int64]([0, 1, 2], [True, False, True])),
        ]
    )
    var cast = lit(Int64(128)).cast(DataType.INT8).cast(DataType.INT64)
    var power = lit(Int64(2)).pow(lit(Int64(-1)))
    for expression in [
        (col("x") + cast.copy() + power.copy()).alias("cast_first"),
        (col("x") + power.copy() + cast.copy()).alias("power_first"),
        when(lit(False))
        .then(col("x") + cast.copy())
        .otherwise(lit(Int64(7)))
        .alias("masked"),
    ]:
        var original = bind(expression, source._columns)
        var lifted = lift_literals(original, source)
        assert_equal(lifted.shape(), ROWS)
        for at in range(len(lifted.shapes)):
            # Broadcast wrappers make checked parents row-valued; their
            # immutable literal leaves correctly remain scalar.
            var op = lifted.expr._nodes[at].op
            assert_equal(
                lifted.shapes[at],
                SCALAR if op
                in [LIT_INT, LIT_FLOAT, LIT_BOOL, LIT_NULL] else ROWS,
            )
            if lifted.expr._nodes[at].op == COL:
                assert_equal(lifted.expr._nodes[at].text, "x")
        # The untouched scalar CAST would be foldable. Its new ROWS dependency
        # prevents an eager constant folder from crossing the original mask.
        assert_false(
            has_literal_text_parsing(source.lazy().select(lifted.expr))
        )
        var proof = certificate(expression, source)
        var plan = prepare(expression, source, 3, proof, source_proved=True)
        assert_equal(plan.source.width(), 1)
        assert_equal(plan.source._columns[0].name(), "x")
        assert_equal(plan.outputs[0].dtype, DataType.FLOAT64)
        assert_equal(len(plan.steps), 1)
        assert_equal(len(plan.ordered_ranks[]), len(plan.literals))
        var previous = UInt32(0)
        for at in range(plan.steps[0].nodes):
            assert_true(plan.ordered_ranks[][at] > previous)
            previous = plan.ordered_ranks[][at]
        for at in range(plan.steps[0].nodes, len(plan.ordered_ranks[])):
            assert_equal(plan.ordered_ranks[][at], UInt32(0))
        with assert_raises(contains="source certificate is missing"):
            _ = prepare(expression, source, 3, proof)
        with assert_raises(contains="AST or logical metadata changed"):
            _ = prepare(
                expression, source, 3, proof + "changed", source_proved=True
            )
    var masked = source.select_exprs(
        [
            when(lit(False))
            .then(col("x") + cast.copy())
            .otherwise(lit(Int64(7)))
            .alias("masked"),
        ]
    )
    for row in range(3):
        assert_equal(masked.item(row, "masked").int64(), Int64(7))
