"""Native scalar reductions over batches produced by the CPU schedule."""
from dataframe.dtype import DataType
from dataframe.expr import COL, COUNT, MAX, MEAN, MIN, SUM
from dataframe.planning.context import QueryContext
from dataframe.planning.model import SCAN_FRAME, SELECT
from .multi_plan import ReductionOutput
from .filter_plan import supported_dtype


@fieldwise_init
struct SelectionField(Copyable, Movable):
    var column: Int
    var dtype: DataType
    var needs_sum: Bool
    var needs_extrema: Bool


@fieldwise_init
struct SelectionReduction(Copyable, Movable):
    var fields: List[SelectionField]
    var outputs: List[ReductionOutput]


def plan_selection(region: QueryContext) raises -> Optional[SelectionReduction]:
    if (
        len(region._nodes) != 2
        or region._nodes[0].kind != SCAN_FRAME
        or region._nodes[1].kind != SELECT
        or region._nodes[1].left != 0
    ):
        return None
    ref root = region._nodes[1]
    if len(root.exprs) == 0:
        return None
    ref schema = region._frames[region._nodes[0].source_slot()]
    var fields = List[SelectionField]()
    var outputs = List[ReductionOutput]()
    var names = List[String]()
    for expression in root.exprs:
        if len(expression._nodes) != 2 or expression._nodes[0].op != COL:
            return None
        ref aggregate = expression._nodes[1]
        if aggregate.left != 0 or (
            aggregate.op != SUM
            and aggregate.op != COUNT
            and aggregate.op != MEAN
            and aggregate.op != MIN
            and aggregate.op != MAX
        ):
            return None
        if expression._name in names:
            return None
        names.append(expression._name.copy())
        var column = schema._index(expression._nodes[0].text)
        var dtype = schema._columns[column].dtype()
        if not supported_dtype(dtype):
            return None
        var field = -1
        for i in range(len(fields)):
            if fields[i].column == column:
                field = i
        if field < 0:
            field = len(fields)
            fields.append(SelectionField(column, dtype, False, False))
        fields[field].needs_sum |= aggregate.op == SUM or aggregate.op == MEAN
        fields[field].needs_extrema |= (
            aggregate.op == MIN or aggregate.op == MAX
        )
        outputs.append(
            ReductionOutput(
                field,
                expression._name.copy(),
                aggregate.op,
                aggregate.min_count,
            )
        )
    return SelectionReduction(fields^, outputs^)
