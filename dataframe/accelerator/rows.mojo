"""CPU-only lowering for bounded resident accelerator row-expression pipelines.

Expressions use the shared binder and retain their logical dtypes. Programs
are device descriptors, not a second public expression API. Rejections happen
before a provider context or device allocation is created.
"""
from std.memory import bitcast
from .capabilities import RowCapabilities
from dataframe.binding import bind, BoundExpr, ROWS, op_name
from dataframe.bool_column import BoolColumn
from dataframe.column import Column
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.expr import (
    Expr,
    COL,
    CAST,
    LIT_FLOAT,
    LIT_INT,
    LIT_BOOL,
    LIT_NULL,
    ADD,
    SUB,
    MUL,
    GT,
    LT,
    GE,
    LE,
    EQ,
    NE,
    AND,
    OR,
    XOR,
    NOT,
    IS_NULL,
    IS_NOT_NULL,
    FILL_NULL,
    NEG,
    SUM,
    COUNT,
    MEAN,
    LEN,
)
from dataframe.frame import DataFrame
from dataframe.lazy import (
    LazyFrame,
    SCAN_FRAME,
    SELECT,
    WITH_COLUMNS,
    FILTER,
    SLICE,
    DROP,
)
from dataframe.series import Series

comptime ROW_MAX_NODES = 64
comptime ROW_MAX_SLOTS = 64
comptime ROW_MAX_STEPS = 64
comptime ROW_CODE_WORDS = 4


@fieldwise_init
struct RowStep(Copyable):
    var start: Int
    var nodes: Int
    var slot: Int
    var filter: Bool
    var gather_start: Int
    var gather_count: Int


@fieldwise_init
struct RowOutput(Copyable):
    var name: String
    var dtype: DataType
    var slot: Int
    var reduction: Int
    var min_count: Int


@fieldwise_init
struct RowPlan(Movable):
    var source: DataFrame
    var dtype: DataType
    var code: List[Int64]
    var literals: List[Float64]
    var node_dtypes: List[DataType]
    var slot_dtypes: List[DataType]
    var gathers: List[Int64]
    var steps: List[RowStep]
    var outputs: List[RowOutput]
    var slots: Int
    var root: Int
    var limit: Int
    var reductions: Bool


def _empty(name: String, dtype: DataType) raises -> Series:
    comptime for i in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[i]
        if dtype == DataType.of(D):
            return Series(name, Column[Scalar[D]](List[Scalar[D]]()))
    if dtype == DataType.BOOL:
        return Series(name, BoolColumn(List[Bool]()))
    raise Error("Unsupported row schema dtype")


def _program(
    bound: BoundExpr,
    count: Int,
    slots: List[Int],
    dtype: DataType,
    mut code: List[Int64],
    mut literals: List[Float64],
    mut node_dtypes: List[DataType],
    capabilities: RowCapabilities,
) raises:
    if count < 1 or count > ROW_MAX_NODES:
        capabilities.reject(
            "plan", "expression must contain 1 to 64 row-local nodes"
        )
    for i in range(count):
        ref node = bound.expr._nodes[i]
        var op = node.op
        capabilities.require_dtype(bound.dtypes[i])
        if (
            not capabilities.mixed_types
            and bound.dtypes[i] != dtype
            and bound.dtypes[i] != DataType.BOOL
        ):
            capabilities.reject(
                "plan", "mixed numeric dtypes or unsupported intermediate dtype"
            )
        if not (
            op == COL
            or (op == CAST and capabilities.casts)
            or op == LIT_INT
            or op == LIT_FLOAT
            or op == LIT_BOOL
            or op == LIT_NULL
            or op == ADD
            or op == SUB
            or op == MUL
            or op == NEG
            or op == GT
            or op == LT
            or op == GE
            or op == LE
            or op == EQ
            or op == NE
            or op == AND
            or op == OR
            or op == XOR
            or op == NOT
            or op == IS_NULL
            or op == IS_NOT_NULL
            or op == FILL_NULL
        ):
            capabilities.reject(
                "plan", "resident expression operation " + op_name(op)
            )
        if op == ADD or op == SUB or op == MUL or op == NEG:
            capabilities.require_arithmetic(bound.dtypes[i])
        if (
            bound.dtypes[i].is_integer()
            and (op == ADD or op == SUB or op == MUL or op == NEG)
            and bound.shapes[i] != ROWS
        ):
            capabilities.reject(
                "plan", "scalar integer arithmetic requires CPU execution"
            )
        if op == CAST and node.integer and bound.shapes[i] != ROWS:
            capabilities.reject(
                "plan", "scalar strict casts require CPU execution"
            )
        var source = slots[bound.sources[i]] if op == COL else (
            Int(node.integer) if op == CAST else -1
        )
        node_dtypes.append(bound.dtypes[i])
        code.append(Int64(op))
        code.append(Int64(node.left))
        code.append(Int64(node.right))
        code.append(Int64(source))
        # Integer payloads travel as raw 64-bit words, never through a float
        # conversion (Int64 literals beyond 2**53 must remain exact).
        literals.append(
            bitcast[DType.float64](node.integer) if op
            == LIT_INT else (
                Float64(node.integer) if op == LIT_BOOL else node.floating
            )
        )


def lower_rows(
    query: LazyFrame, capabilities: RowCapabilities = RowCapabilities()
) raises -> RowPlan:
    """Bind the entire resident region before GPU submission or source I/O."""
    var root = len(query._nodes) - 1
    if root < 0:
        capabilities.reject("plan", "empty plan")
    var cursor = root
    var limit = -1
    if query._nodes[cursor].kind == SLICE:
        if query._nodes[cursor].offset != 0:
            capabilities.reject(
                "plan", "only final head is supported in a resident region"
            )
        limit = query._nodes[cursor].length
        cursor = query._nodes[cursor].left
    var chain = List[Int]()
    while cursor >= 0 and query._nodes[cursor].kind != SCAN_FRAME:
        var kind = query._nodes[cursor].kind
        if (
            kind != SELECT
            and kind != WITH_COLUMNS
            and kind != FILTER
            and kind != DROP
        ):
            capabilities.reject(
                "plan",
                "resident region requires an in-memory scan and row-local steps",
            )
        chain.append(cursor)
        cursor = query._nodes[cursor].left
        if len(chain) > ROW_MAX_STEPS:
            capabilities.reject(
                "plan", "resident region supports at most 64 plan steps"
            )
    if cursor < 0:
        capabilities.reject(
            "plan", "resident region requires an in-memory scan"
        )
    # Reuse core projection analysis after validating the resident chain.
    # This only narrows the source: no expression evaluation or row movement.
    var projected_query = query.copy()
    projected_query._push_projections()
    var frame = query._frames[query._nodes[cursor].offset].copy()
    if len(projected_query._nodes[cursor].names):
        frame = frame.select(projected_query._nodes[cursor].names)
    if frame.width() < 1 or frame.width() > ROW_MAX_SLOTS:
        capabilities.reject(
            "plan", "resident region requires 1 to 64 source columns"
        )
    var dtype = DataType.BOOL
    var schema = frame._columns.copy()
    var slots = List[Int]()
    for i in range(len(schema)):
        var type = schema[i].dtype()
        capabilities.require_dtype(type)
        if schema[i].is_chunked():
            capabilities.reject(
                "plan", "chunked input; rechunk before accelerator execution"
            )
        if type != DataType.BOOL:
            if (
                not capabilities.mixed_types
                and dtype != DataType.BOOL
                and dtype != type
            ):
                capabilities.reject(
                    "plan",
                    "resident source numeric columns must share one dtype",
                )
            if dtype == DataType.BOOL:
                dtype = type
        slots.append(i)
    if dtype == DataType.BOOL:
        # Internal physical representation only; public columns remain Bool.
        dtype = DataType.FLOAT32
    var code = List[Int64]()
    var literals = List[Float64]()
    var node_dtypes = List[DataType]()
    var slot_dtypes = List[DataType]()
    for column in schema:
        slot_dtypes.append(column.dtype())
    var gathers = List[Int64]()
    var steps = List[RowStep]()
    var outputs = List[RowOutput]()
    var next_slot = len(schema)
    var reduced = False
    for k in range(len(chain) - 1, -1, -1):
        ref node = query._nodes[chain[k]]
        if reduced:
            capabilities.reject(
                "plan", "only final head may follow resident reductions"
            )
        if node.kind == DROP:
            var keep_schema = List[Series]()
            var keep_slots = List[Int]()
            for name in node.names:
                var found = False
                for column in schema:
                    found |= column.name() == name
                if not found:
                    raise Error("Unknown column: " + name)
            for i in range(len(schema)):
                if schema[i].name() not in node.names:
                    keep_schema.append(schema[i].copy())
                    keep_slots.append(slots[i])
            schema = keep_schema^
            slots = keep_slots^
            continue
        var projected = List[Series]()
        var projected_slots = List[Int]()
        var aggregate_outputs = List[RowOutput]()
        var has_rows = False
        var has_aggregate = False
        var projection_names = List[String]()
        for expr in node.exprs:
            if node.kind != FILTER:
                if expr._name in projection_names:
                    capabilities.reject(
                        "plan", "duplicate output name: " + expr._name
                    )
                projection_names.append(expr._name)
            var bound = bind(expr, schema)
            for part_index in range(len(bound.expr._nodes)):
                if bound.dtypes[part_index].is_integer():
                    ref part = bound.expr._nodes[part_index]
                    if (
                        part.op == ADD
                        or part.op == SUB
                        or part.op == MUL
                        or part.op == NEG
                    ):
                        # CPU predicate/projection/head pushdown can remove
                        # checked work. Keep the first integer subset at an
                        # observable terminal boundary, with no final head.
                        if k != 0 or node.kind == FILTER or limit >= 0:
                            capabilities.reject(
                                "plan",
                                "checked integer arithmetic requires terminal projections without head",
                            )
                    if part.op == SUM and limit >= 0:
                        capabilities.reject(
                            "plan",
                            "integer sum with final head requires CPU execution",
                        )
            for part in bound.expr._nodes:
                if (
                    part.op == CAST
                    and part.integer
                    and (k != 0 or node.kind == FILTER or limit >= 0)
                ):
                    capabilities.reject(
                        "plan",
                        "strict casts require terminal projections without head",
                    )
            var count = len(bound.expr._nodes)
            var result_dtype = bound.dtypes[count - 1]
            var reduction = -1
            var min_count = 0
            var row_shape = bound.shape() == ROWS
            ref last = bound.expr._nodes[count - 1]
            if (
                last.op == SUM
                or last.op == COUNT
                or last.op == MEAN
                or last.op == LEN
            ):
                if node.kind != SELECT or bound.shapes[last.left] != ROWS:
                    capabilities.reject(
                        "plan",
                        "reductions require a row expression in final select",
                    )
                reduction = last.op
                capabilities.require_reduction(
                    bound.dtypes[last.left], reduction, frame.height()
                )
                min_count = last.min_count
                count -= 1
                if last.left != count - 1:
                    capabilities.reject(
                        "plan",
                        "reduction must consume the complete row expression",
                    )
                if (
                    (reduction == SUM or reduction == MEAN)
                    and bound.dtypes[last.left] != dtype
                    and not capabilities.mixed_types
                ):
                    capabilities.reject(
                        "plan",
                        "resident numeric reduction requires the common numeric input dtype",
                    )
                has_aggregate = True
            elif bound.aggregated[len(bound.aggregated) - 1]:
                capabilities.reject(
                    "plan", "unsupported reduction or aggregate expression"
                )
            else:
                has_rows |= row_shape
            if node.kind == FILTER and result_dtype != DataType.BOOL:
                capabilities.reject("plan", "filter predicate must be Boolean")
            if next_slot >= ROW_MAX_SLOTS:
                capabilities.reject(
                    "plan", "resident region supports at most 64 value slots"
                )
            var start = len(literals)
            _program(
                bound,
                count,
                slots,
                dtype,
                code,
                literals,
                node_dtypes,
                capabilities,
            )
            slot_dtypes.append(bound.dtypes[count - 1])
            var gather_start = len(gathers)
            if node.kind == FILTER:
                for slot in slots:
                    gathers.append(Int64(slot))
            steps.append(
                RowStep(
                    start,
                    count,
                    next_slot,
                    node.kind == FILTER,
                    gather_start,
                    len(slots),
                )
            )
            if reduction >= 0:
                aggregate_outputs.append(
                    RowOutput(
                        expr._name,
                        result_dtype,
                        next_slot,
                        reduction,
                        min_count,
                    )
                )
            else:
                projected.append(_empty(expr._name, result_dtype))
                projected_slots.append(next_slot)
            next_slot += 1
        if node.kind == FILTER:
            if len(node.exprs) != 1:
                capabilities.reject("plan", "filter requires one predicate")
            continue
        if has_aggregate:
            if len(aggregate_outputs) != len(node.exprs):
                capabilities.reject(
                    "plan", "cannot mix row and reduction projections"
                )
            outputs = aggregate_outputs^
            reduced = True
        elif node.kind == SELECT:
            if not has_rows:
                capabilities.reject(
                    "plan",
                    "scalar-only select is not a resident row projection",
                )
            schema = projected^
            slots = projected_slots^
        else:
            for i in range(len(projected)):
                var found = -1
                for j in range(len(schema)):
                    if schema[j].name() == projected[i].name():
                        found = j
                if found < 0:
                    schema.append(projected[i].copy())
                    slots.append(projected_slots[i])
                else:
                    schema[found] = projected[i].copy()
                    slots[found] = projected_slots[i]
        var names = List[String]()
        if reduced:
            for output in outputs:
                if output.name in names:
                    capabilities.reject(
                        "plan", "duplicate output name: " + output.name
                    )
                names.append(output.name)
        else:
            for column in schema:
                if column.name() in names:
                    capabilities.reject(
                        "plan", "duplicate output name: " + column.name()
                    )
                names.append(column.name())
    if not reduced:
        for i in range(len(schema)):
            outputs.append(
                RowOutput(schema[i].name(), schema[i].dtype(), slots[i], -1, 0)
            )
    if len(outputs) == 0:
        capabilities.reject(
            "plan", "resident region requires at least one output column"
        )
    return RowPlan(
        frame^,
        dtype,
        code^,
        literals^,
        node_dtypes^,
        slot_dtypes^,
        gathers^,
        steps^,
        outputs^,
        next_slot,
        root,
        limit,
        reduced,
    )
