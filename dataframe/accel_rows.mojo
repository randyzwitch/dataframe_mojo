"""CPU-only lowering for bounded resident NVIDIA row-expression pipelines.

Expressions use the shared binder and retain their logical dtypes. Programs
are device descriptors, not a second public expression API. Rejections happen
before a provider context or device allocation is created.
"""
from .binding import bind, BoundExpr, ROWS, op_name
from .bool_column import BoolColumn
from .column import Column
from .dtype import DataType
from .expr import (
    Expr,
    COL,
    LIT_FLOAT,
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
)
from .frame import DataFrame
from .lazy import (
    LazyFrame,
    SCAN_FRAME,
    SELECT,
    WITH_COLUMNS,
    FILTER,
    SLICE,
    DROP,
)
from .series import Series

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
    var gathers: List[Int64]
    var steps: List[RowStep]
    var outputs: List[RowOutput]
    var slots: Int
    var root: Int
    var limit: Int
    var reductions: Bool


def _reject(reason: String) raises:
    raise Error("NVIDIA unsupported: " + reason)


def _empty(name: String, dtype: DataType) raises -> Series:
    if dtype == DataType.FLOAT32:
        return Series(name, Column[Float32](List[Float32]()))
    if dtype == DataType.FLOAT64:
        return Series(name, Column[Float64](List[Float64]()))
    if dtype == DataType.BOOL:
        return Series(name, BoolColumn(List[Bool]()))
    _reject("resident expressions require Float32/Float64 or Bool")
    return Series(name, Column[Float64](List[Float64]()))


def _program(
    bound: BoundExpr,
    count: Int,
    slots: List[Int],
    dtype: DataType,
    mut code: List[Int64],
    mut literals: List[Float64],
) raises:
    if count < 1 or count > ROW_MAX_NODES:
        _reject("expression must contain 1 to 64 row-local nodes")
    for i in range(count):
        ref node = bound.expr._nodes[i]
        var op = node.op
        if bound.dtypes[i] != dtype and bound.dtypes[i] != DataType.BOOL:
            _reject("mixed numeric dtypes or unsupported intermediate dtype")
        if not (
            op == COL
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
            _reject("resident expression operation " + op_name(op))
        var source = slots[bound.sources[i]] if op == COL else -1
        code.append(Int64(op))
        code.append(Int64(node.left))
        code.append(Int64(node.right))
        code.append(Int64(source))
        literals.append(
            Float64(node.integer) if op == LIT_BOOL else node.floating
        )


def lower_rows(query: LazyFrame) raises -> RowPlan:
    """Bind the entire resident region before GPU submission or source I/O."""
    var root = len(query._nodes) - 1
    if root < 0:
        _reject("empty plan")
    var cursor = root
    var limit = -1
    if query._nodes[cursor].kind == SLICE:
        if query._nodes[cursor].offset != 0:
            _reject("only final head is supported in a resident region")
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
            _reject(
                "resident region requires an in-memory scan and row-local steps"
            )
        chain.append(cursor)
        cursor = query._nodes[cursor].left
        if len(chain) > ROW_MAX_STEPS:
            _reject("resident region supports at most 64 plan steps")
    if cursor < 0:
        _reject("resident region requires an in-memory scan")
    var frame = query._frames[query._nodes[cursor].offset].copy()
    if frame.width() < 1 or frame.width() > ROW_MAX_SLOTS:
        _reject("resident region requires 1 to 64 source columns")
    var dtype = DataType.BOOL
    var schema = frame._columns.copy()
    var slots = List[Int]()
    for i in range(len(schema)):
        var type = schema[i].dtype()
        if (
            type != DataType.FLOAT32
            and type != DataType.FLOAT64
            and type != DataType.BOOL
        ):
            _reject("resident source dtype must be Float32, Float64 or Bool")
        if schema[i].is_chunked():
            _reject("chunked input; rechunk before accelerator execution")
        if type != DataType.BOOL:
            if dtype != DataType.BOOL and dtype != type:
                _reject("resident source numeric columns must share one dtype")
            dtype = type
        slots.append(i)
    if dtype == DataType.BOOL:
        # Internal physical representation only; public columns remain Bool.
        dtype = DataType.FLOAT32
    var code = List[Int64]()
    var literals = List[Float64]()
    var gathers = List[Int64]()
    var steps = List[RowStep]()
    var outputs = List[RowOutput]()
    var next_slot = len(schema)
    var reduced = False
    for k in range(len(chain) - 1, -1, -1):
        ref node = query._nodes[chain[k]]
        if reduced:
            _reject("only final head may follow resident reductions")
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
                    _reject("duplicate output name: " + expr._name)
                projection_names.append(expr._name)
            var bound = bind(expr, schema)
            var count = len(bound.expr._nodes)
            var result_dtype = bound.dtypes[count - 1]
            var reduction = -1
            var min_count = 0
            var row_shape = bound.shape() == ROWS
            ref last = bound.expr._nodes[count - 1]
            if last.op == SUM or last.op == COUNT:
                if node.kind != SELECT or bound.shapes[last.left] != ROWS:
                    _reject(
                        "reductions require a row expression in final select"
                    )
                reduction = last.op
                min_count = last.min_count
                count -= 1
                if last.left != count - 1:
                    _reject(
                        "reduction must consume the complete row expression"
                    )
                if reduction == SUM and result_dtype != dtype:
                    _reject("resident sum currently requires floating input")
                has_aggregate = True
            elif bound.aggregated[len(bound.aggregated) - 1]:
                _reject("unsupported reduction or aggregate expression")
            else:
                has_rows |= row_shape
            if node.kind == FILTER and result_dtype != DataType.BOOL:
                _reject("filter predicate must be Boolean")
            if next_slot >= ROW_MAX_SLOTS:
                _reject("resident region supports at most 64 value slots")
            var start = len(literals)
            _program(bound, count, slots, dtype, code, literals)
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
                _reject("filter requires one predicate")
            continue
        if has_aggregate:
            if len(aggregate_outputs) != len(node.exprs):
                _reject("cannot mix row and reduction projections")
            outputs = aggregate_outputs^
            reduced = True
        elif node.kind == SELECT:
            if not has_rows:
                _reject("scalar-only select is not a resident row projection")
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
                    _reject("duplicate output name: " + output.name)
                names.append(output.name)
        else:
            for column in schema:
                if column.name() in names:
                    _reject("duplicate output name: " + column.name())
                names.append(column.name())
    if not reduced:
        for i in range(len(schema)):
            outputs.append(
                RowOutput(schema[i].name(), schema[i].dtype(), slots[i], -1, 0)
            )
    if len(outputs) == 0:
        _reject("resident region requires at least one output column")
    return RowPlan(
        frame^,
        dtype,
        code^,
        literals^,
        gathers^,
        steps^,
        outputs^,
        next_slot,
        root,
        limit,
        reduced,
    )
