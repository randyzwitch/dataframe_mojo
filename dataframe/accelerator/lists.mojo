"""Terminal fixed-width IMPLODE packaging over real GPU operand outputs.

Only final list construction occurs on the host. Primitive operands, their
masks and their checked errors are evaluated by the existing resident executor.
Resident variable-length List consumers remain a separate storage layer.
"""
from std.collections import Dict
from dataframe import DataFrame, DataType, Expr, ListColumn, Series
from dataframe.binding import bind, ROWS
from dataframe.expr import IMPLODE, subtree, col
from dataframe.lazy import (
    LazyFrame,
    SELECT,
    WITH_COLUMNS,
    SLICE,
    PlanNode,
    _plan_node,
)
from .aggregates import _in_memory_sources
from .metadata import schema_frame
from .memory import checked_mul
from .selectors import _live_nodes

comptime IMPLODE_MATERIALIZATION = "metal_implode_materialization"


@fieldwise_init
struct ImplodeOutput(Copyable):
    var prototype: Series
    var child_name: String


@fieldwise_init
struct ImplodeOutputs(Movable):
    var query: LazyFrame
    var schema: DataFrame
    var outputs: List[ImplodeOutput]
    var slices: List[PlanNode]
    var assembly_bytes: Int


def has_implode(query: LazyFrame) -> Bool:
    var live = _live_nodes(query)
    for index in range(len(query._nodes)):
        if live[index]:
            for expression in query._nodes[index].exprs:
                for node in expression._nodes:
                    if node.op == IMPLODE:
                        return True
    return False


def has_implode_materializations(query: LazyFrame) -> Bool:
    for operation in query._nodes:
        if operation.text == IMPLODE_MATERIALIZATION:
            return True
    return False


def _fixed(dtype: DataType) -> Bool:
    return (
        not dtype.is_nested()
        and dtype != DataType.STRING
        and not dtype._is_bytes()
        and not dtype.is_categorical()
        and (
            dtype.is_numeric() or dtype == DataType.BOOL or dtype.is_temporal()
        )
    )


def lower_terminal_implode(query: LazyFrame) raises -> ImplodeOutputs:
    """Peel one/multiple terminal row-shaped primitive IMPLODE roots.

    Trailing slices apply to the one-row List result AFTER all operand errors
    have been observed; in particular head(0) never prunes a faulting operand.
    Original metadata binds first, in CPU projection order, before rewriting.
    """
    _in_memory_sources(query)
    var original = query.copy()
    original._reorder(len(original._nodes) - 1)
    original._compact_scan_slots()
    var root = len(original._nodes) - 1
    var reverse_slices = List[PlanNode]()
    while original._nodes[root].kind == SLICE:
        var trailing = original._nodes[root].copy()
        if trailing.length < -1 and not trailing.source_failure:
            trailing.source_failure = "Slice length must be nonnegative"
        reverse_slices.append(trailing^)
        root = original._nodes[root].left
    if original._nodes[root].kind != SELECT:
        raise Error("Metal unsupported [implode]: terminal SELECT required")
    # Bind all original projections before touching trailing slice controls.
    # Empty metadata evaluation retains the one-row List prototype but cannot
    # let a later invalid slice outrank actual checked operand work.
    var expected = schema_frame(original, root).clear()
    var input = schema_frame(original, original._nodes[root].left)
    var live = _live_nodes(original)
    for index in range(root):
        if live[index]:
            for expression in original._nodes[index].exprs:
                for node in expression._nodes:
                    if node.op == IMPLODE:
                        raise Error(
                            "Metal unsupported [implode]: intermediate lists"
                        )
    var used = Dict[String, Bool]()
    for name in input.columns():
        used[name] = True
    var expressions = List[Expr]()
    var outputs = List[ImplodeOutput]()
    for index in range(len(original._nodes[root].exprs)):
        var expression = original._nodes[root].exprs[index].copy()
        var bound = bind(expression, input._columns)
        var top = len(bound.expr._nodes) - 1
        ref node = bound.expr._nodes[top]
        if node.op != IMPLODE or node.left < 0:
            raise Error(
                "Metal unsupported [implode]: only IMPLODE output roots"
            )
        if bound.shapes[node.left] != ROWS:
            raise Error(
                "Metal unsupported [implode]: row-shaped operands required"
            )
        if not _fixed(bound.dtypes[node.left]):
            raise Error(
                "Metal unsupported [implode]: fixed-width primitive operands required"
            )
        for child in range(top):
            if bound.expr._nodes[child].op == IMPLODE:
                raise Error("Metal unsupported [implode]: nested lists")
        var name = "__dataframe_implode_" + String(index)
        while name in used:
            name += "_"
        used[name] = True
        expressions.append(subtree(bound.expr, node.left).alias(name))
        outputs.append(ImplodeOutput(expected._columns[index].copy(), name^))
    if not len(outputs):
        raise Error("Metal unsupported [implode]: no list output")
    var flattened = original.copy()
    flattened._reorder(original._nodes[root].left)
    flattened._compact_scan_slots()
    var producer = len(flattened._nodes) - 1
    var final_columns = List[Expr]()
    for index in range(len(expressions)):
        # CPU prepares each original expression's states and complete operand
        # before moving to the next original projection. A separate observed
        # producer prevents later aggregate/OVER leaves from running first.
        flattened._nodes.append(
            _plan_node(
                WITH_COLUMNS,
                producer,
                exprs=[expressions[index].copy()],
                text=IMPLODE_MATERIALIZATION,
            )
        )
        producer = len(flattened._nodes) - 1
        final_columns.append(col(outputs[index].child_name))
    flattened._nodes.append(_plan_node(SELECT, producer, exprs=final_columns^))
    var schema = schema_frame(flattened, len(flattened._nodes) - 1)
    for index in range(len(outputs)):
        if (
            schema._columns[index].dtype()
            != outputs[index].prototype.dtype().inner()
        ):
            raise Error("Metal implode operand schema differs")
    var slices = List[PlanNode](capacity=len(reverse_slices))
    for reverse in range(len(reverse_slices)):
        slices.append(reverse_slices[len(reverse_slices) - 1 - reverse].copy())
    # Exactly two requested Int64 offsets per list. Child payload owners are
    # shared from the exported primitive table, already in the inner peak.
    return ImplodeOutputs(
        flattened^, expected^, outputs^, slices^, checked_mul(len(outputs), 16)
    )


def observe_trailing_failures(plan: ImplodeOutputs) raises:
    for slice in plan.slices:
        if slice.source_failure != "":
            raise Error(slice.source_failure)


def assemble(flat: DataFrame, plan: ImplodeOutputs) raises -> DataFrame:
    """Wrap primitive child owners; preserve child nulls and source order."""
    observe_trailing_failures(plan)
    if flat.width() != len(plan.outputs):
        raise Error("Metal implode exported width differs")
    var columns = List[Series](capacity=flat.width())
    for index in range(flat.width()):
        ref output = plan.outputs[index]
        var prototype = output.prototype.list_column().child()
        var child = flat._columns[index].renamed("item")
        if (
            child.dtype() != prototype.dtype()
            or flat._columns[index].name() != output.child_name
        ):
            raise Error("Metal implode exported child schema differs")
        child._field_metadata = prototype._field_metadata
        var offsets: List[Int64] = [0, Int64(flat.height())]
        var column = Series(
            output.prototype.name(), ListColumn(offsets^, child^)
        )
        column._field_metadata = output.prototype._field_metadata
        columns.append(column^)
    var result = DataFrame(columns^, height=1)
    for slice in plan.slices:
        result = result.slice(slice.offset, slice.length)
    return result^
