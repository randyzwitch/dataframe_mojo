"""CPU-only lowering for bounded resident accelerator row-expression pipelines.

Expressions use the shared binder and retain their logical dtypes. Programs
are device descriptors, not a second public expression API. Rejections happen
before a provider context or device allocation is created.
"""
from std.memory import bitcast
from std.math import pow, abs
from .capabilities import RowCapabilities
from dataframe.binding import bind, BoundExpr, ROWS, op_name
from dataframe.dtype import DataType
from dataframe.expr import (
    Expr,
    DIV,
    FLOORDIV,
    MOD,
    POW,
    CLIP_LOW,
    CLIP_HIGH,
    FILL_NAN,
    KEEP_NULLS,
    ABS,
    SQRT,
    FLOOR,
    CEIL,
    ROUND,
    IS_NAN,
    IS_NOT_NAN,
    IS_FINITE,
    IS_INFINITE,
    WHEN,
    INT_IS_IN,
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
    MIN,
    MAX,
    COUNT,
    MEAN,
    LEN,
    FIRST,
    LAST,
    ANY,
    ALL,
    NULL_COUNT,
    ARG_MIN,
    ARG_MAX,
    DT_YEAR,
    DT_TIME,
    DT_TRUNCATE,
)
from dataframe.frame import DataFrame
from dataframe.temporal import parse_every, NANOS_PER_DAY
from dataframe.lazy import (
    LazyFrame,
    SCAN_FRAME,
    SELECT,
    WITH_COLUMNS,
    FILTER,
    SLICE,
    DROP,
    SORT,
    AGG,
)
from dataframe.series import Series

comptime ROW_GROUP_HEADER = 1002
comptime ROW_GROUP_KEY = 1003
comptime ROW_GROUP_OUTPUT = 1004
comptime ROW_GROUP_HEADER_LAST = 1005
comptime ROW_SLICE = 1006
comptime ROW_SORT_KEY = 1001
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
    var order: Bool
    var grouped: Bool


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
    return Series.full_null(name, dtype, 0)


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
    var program_start = len(literals)
    var memberships = List[Int]()
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
            or (
                op == FLOORDIV
                and capabilities.float_division
                and bound.dtypes[i] == DataType.FLOAT32
            )
            or (
                op == SQRT
                and capabilities.float_sqrt
                and bound.dtypes[i].is_float()
            )
            or (
                op == ROUND
                and capabilities.float_rounding
                and bound.dtypes[i].is_float()
            )
            or (
                (
                    (op == DIV or op == FLOORDIV)
                    and capabilities.float_division
                    or op == MOD
                    and capabilities.float_modulo
                )
                and bound.dtypes[i] == DataType.FLOAT64
            )
            or (
                capabilities.temporal_fields
                and op >= DT_YEAR
                and op <= DT_TRUNCATE
            )
            or (op == INT_IS_IN and capabilities.extended_integers)
            or (
                capabilities.row_extras
                and (
                    op == FLOORDIV
                    or op == MOD
                    or op == POW
                    or op == CLIP_LOW
                    or op == CLIP_HIGH
                    or op == FILL_NAN
                    or op == KEEP_NULLS
                    or op == ABS
                    or op == FLOOR
                    or op == CEIL
                    or op == ROUND
                    or op == IS_NAN
                    or op == IS_NOT_NAN
                    or op == IS_FINITE
                    or op == IS_INFINITE
                    or op == WHEN
                )
            )
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
        var left_type = (
            bound.dtypes[node.left] if node.left >= 0 else bound.dtypes[i]
        )
        var right_type = (
            bound.dtypes[node.right] if node.right >= 0 else left_type
        )
        var temporal = op >= DT_YEAR and op <= DT_TRUNCATE
        if (
            temporal
            and left_type.is_datetime()
            and left_type.time_zone().byte_length()
        ):
            capabilities.reject(
                "temporal", "time zone field extraction requires CPU execution"
            )
        if (
            not capabilities.float64
            and (
                bound.dtypes[i] == DataType.FLOAT64
                or left_type == DataType.FLOAT64
                or right_type == DataType.FLOAT64
            )
            and not (
                op == COL
                or op == LIT_FLOAT
                or op == LIT_NULL
                or op == IS_NULL
                or op == IS_NOT_NULL
                or op == FILL_NULL
                or op == KEEP_NULLS
                or op == WHEN
                or (
                    capabilities.float64_emulated
                    and (
                        op == ADD
                        or op == SUB
                        or op == MUL
                        or op == NEG
                        or op == DIV
                        or op == FLOORDIV
                        or op == MOD
                        or op == SQRT
                        or op == CAST
                        or op == FLOOR
                        or op == CEIL
                        or op == ROUND
                        or op == ABS
                        or op == GT
                        or op == LT
                        or op == GE
                        or op == LE
                        or op == EQ
                        or op == NE
                        or op == IS_NAN
                        or op == IS_NOT_NAN
                        or op == IS_FINITE
                        or op == IS_INFINITE
                        or op == FILL_NAN
                        or op == CLIP_LOW
                        or op == CLIP_HIGH
                    )
                )
            )
        ):
            capabilities.reject(
                "precision", "Float64 requires CPU execution for this operation"
            )
        var logical = (
            bound.dtypes[i].is_temporal()
            or bound.dtypes[i].is_decimal()
            or left_type.is_temporal()
            or left_type.is_decimal()
            or right_type.is_temporal()
            or right_type.is_decimal()
        )
        if logical and (
            op == SQRT
            or op == ABS
            or op == FLOORDIV
            or op == MOD
            or op == POW
            or op == FLOOR
            or op == CEIL
            or op == ROUND
            or op == ADD
            or op == SUB
            or op == MUL
            or op == NEG
            or op == CAST
        ):
            capabilities.reject(
                "logical", "logical arithmetic and casts require CPU execution"
            )
        if left_type.is_decimal() and (
            op == GT or op == LT or op == GE or op == LE or op == EQ or op == NE
        ):
            if (
                left_type.scale() != right_type.scale()
                or left_type.physical() != right_type.physical()
            ):
                capabilities.reject(
                    "logical",
                    "decimal comparisons require matching scales and storage widths",
                )
        if (
            op == ADD
            or op == SUB
            or op == MUL
            or op == NEG
            or op == ABS
            or op == FLOORDIV
            or op == POW
        ):
            capabilities.require_arithmetic(bound.dtypes[i])
        if (
            bound.dtypes[i].is_integer()
            and (
                op == ADD
                or op == SUB
                or op == MUL
                or op == NEG
                or op == ABS
                or op == FLOORDIV
                or op == POW
            )
            and bound.shapes[i] != ROWS
        ):
            capabilities.reject(
                "plan", "scalar integer arithmetic requires CPU execution"
            )
        if (
            op == CAST
            and node.integer
            and bound.shapes[i] != ROWS
            and not (
                capabilities.float64_emulated and bound.dtypes[i].is_float()
            )
        ):
            capabilities.reject(
                "plan", "scalar strict casts require CPU execution"
            )
        if (
            op == ROUND
            and bound.dtypes[i] == DataType.FLOAT32
            and node.integer != 0
            and not capabilities.float_rounding
        ):
            capabilities.reject(
                "precision",
                "Float32 decimal rounding requires Float64 intermediate arithmetic",
            )
        if (
            (op == FLOORDIV or op == MOD or op == POW)
            and bound.dtypes[i].is_float()
            and not (
                op == FLOORDIV
                and capabilities.float_division
                or op == MOD
                and capabilities.float_modulo
                and bound.dtypes[i] == DataType.FLOAT64
            )
        ):
            capabilities.reject(
                "precision",
                "floating division, modulo, and power require CPU execution",
            )
        var source = slots[bound.sources[i]] if op == COL else (
            Int(node.integer) if op == CAST
            or op == ROUND else (node.extra if op == WHEN else -1)
        )
        if temporal:
            source = 0 if left_type.is_date() else (
                4 if left_type.is_time() else (
                    1 if left_type.per_second()
                    == 1000 else (2 if left_type.per_second() == 1000000 else 3)
                )
            )
        var right = Int64(node.right)
        if op == DT_TRUNCATE:
            var interval = parse_every(node.text)
            var per_tick = NANOS_PER_DAY if left_type.is_date() else (
                Int64(1000000000) // left_type.per_second()
            )
            if (
                interval[0] != 0
                or interval[1] <= 0
                or interval[1] % per_tick != 0
            ):
                capabilities.reject(
                    "temporal",
                    "truncate requires a positive fixed interval aligned to the storage unit",
                )
            right = interval[1] // per_tick
        node_dtypes.append(bound.dtypes[i])
        code.append(Int64(op))
        code.append(Int64(node.left))
        code.append(right)
        code.append(Int64(source))
        # Integer payloads travel as raw 64-bit words, never through a float
        # conversion (Int64 literals beyond 2**53 must remain exact).
        literals.append(
            bitcast[DType.float64](node.integer) if op
            == LIT_INT else (
                Float64(node.integer) if op
                == LIT_BOOL else (
                    pow(Float64(10), Float64(abs(node.integer))) if op == ROUND
                    and capabilities.float_rounding
                    and bound.dtypes[i].is_float()
                    and node.integer != 0 else node.floating
                )
            )
        )
        if op == INT_IS_IN:
            memberships.append(i)
    # Membership values are runtime literal payloads, outside the executable
    # expression's local node range. Changing values reuses the same shader.
    # Preserve exact signed/unsigned 64-bit words without Float64 conversion.
    for i in memberships:
        ref node = bound.expr._nodes[i]
        var entries = node.text.split(",")
        var size = len(entries) if node.text else 0
        code[4 * (program_start + i) + 2] = Int64(len(literals))
        code[4 * (program_start + i) + 3] = Int64(size)
        for j in range(size):
            code.extend([Int64(LIT_INT), Int64(-1), Int64(-1), Int64(-1)])
            literals.append(bitcast[DType.float64](Int64(atol(entries[j]))))
            node_dtypes.append(bound.dtypes[node.left])


def lower_rows(
    query: LazyFrame,
    capabilities: RowCapabilities = RowCapabilities(),
    *,
    terminal_boundary: Bool = True,
) raises -> RowPlan:
    """Bind the entire resident region before GPU submission or source I/O.

    Regions feeding another operator are not observable terminal boundaries:
    that operator can prune rows and change whether checked work is evaluated.
    """
    var root = len(query._nodes) - 1
    if root < 0:
        capabilities.reject("plan", "empty plan")
    var cursor = root
    var limit = -1
    if query._nodes[cursor].kind == SLICE and query._nodes[cursor].offset == 0:
        if query._nodes[cursor].length < -1:
            raise Error("Slice length must be nonnegative")
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
            and (kind != SLICE or not capabilities.slices)
            and (kind != SORT or not capabilities.order)
            and (kind != AGG or not capabilities.grouped)
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
        if node.kind == SLICE:
            if node.length < -1:
                raise Error("Slice length must be nonnegative")
            if next_slot >= ROW_MAX_SLOTS:
                capabilities.reject("plan", "slice exceeds 64 value slots")
            var start = len(literals)
            code.extend(
                [
                    Int64(ROW_SLICE),
                    Int64(start + 1),
                    Int64(start + 2),
                    Int64(-1),
                ]
            )
            literals.append(0)
            node_dtypes.append(DataType.BOOL)
            # Runtime scalar parameters keep shader caching independent of
            # offset and length. These two literal nodes are not evaluated.
            for parameter in List[Int]([node.offset, node.length]):
                code.extend([Int64(LIT_INT), Int64(-1), Int64(-1), Int64(-1)])
                literals.append(bitcast[DType.float64](Int64(parameter)))
                node_dtypes.append(DataType.INT64)
            var gather_start = len(gathers)
            for slot in slots:
                gathers.append(Int64(slot))
            slot_dtypes.append(DataType.BOOL)
            steps.append(
                RowStep(
                    start,
                    1,
                    next_slot,
                    True,
                    gather_start,
                    len(slots),
                    False,
                    False,
                )
            )
            next_slot += 1
            continue
        if node.kind == SORT:
            if len(node.names) == 0:
                continue
            if len(node.flags) != 2 * len(node.names):
                raise Error("sort needs direction and null placement per key")
            var start = len(literals)
            for i in range(len(node.names)):
                var found = -1
                for j in range(len(schema)):
                    if schema[j].name() == node.names[i]:
                        found = j
                if found < 0:
                    raise Error("Unknown column: " + node.names[i])
                capabilities.require_key(schema[found].dtype())
                code.extend(
                    [
                        Int64(ROW_SORT_KEY),
                        Int64(slots[found]),
                        Int64(node.flags[i]),
                        Int64(node.flags[len(node.names) + i]),
                    ]
                )
                literals.append(0)
                node_dtypes.append(schema[found].dtype())
            var gather_start = len(gathers)
            for slot in slots:
                gathers.append(Int64(slot))
            steps.append(
                RowStep(
                    start,
                    len(node.names),
                    0,
                    False,
                    gather_start,
                    len(slots),
                    True,
                    False,
                )
            )
            continue
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
        var group_keys = List[Int]()
        if node.kind == AGG:
            if len(node.names) == 0:
                raise Error("group_by requires at least one key")
            var key_names = List[String]()
            for name in node.names:
                if name in key_names:
                    raise Error("Duplicate group key: " + name)
                key_names.append(name)
                var found = -1
                for j in range(len(schema)):
                    if schema[j].name() == name:
                        found = j
                if found < 0:
                    raise Error("Unknown column: " + name)
                group_keys.append(found)
                capabilities.require_key(schema[found].dtype())
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
                        or part.op == ABS
                        or part.op == FLOORDIV
                        or part.op == POW
                    ):
                        # CPU predicate/projection/head pushdown can remove
                        # checked work. Keep the first integer subset at an
                        # observable terminal boundary, with no final head.
                        if (
                            not terminal_boundary
                            or k != 0
                            or node.kind == FILTER
                            or limit >= 0
                        ):
                            capabilities.reject(
                                "plan",
                                "checked integer arithmetic requires terminal projections without head",
                            )
                    if part.op == SUM and limit >= 0 and node.kind != AGG:
                        capabilities.reject(
                            "plan",
                            "integer sum with final head requires CPU execution",
                        )
            for part_index in range(len(bound.expr._nodes)):
                ref part = bound.expr._nodes[part_index]
                if (
                    part.op == CAST
                    and part.integer
                    and not (
                        capabilities.float64_emulated
                        and bound.dtypes[part_index].is_float()
                    )
                    and (
                        not terminal_boundary
                        or k != 0
                        or node.kind == FILTER
                        or limit >= 0
                    )
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
                last.op == MIN
                or last.op == FIRST
                or last.op == LAST
                or last.op == ANY
                or last.op == ALL
                or last.op == NULL_COUNT
                or last.op == ARG_MIN
                or last.op == ARG_MAX
                or last.op == MAX
                or last.op == SUM
                or last.op == COUNT
                or last.op == MEAN
                or last.op == LEN
            ):
                if (node.kind != SELECT and node.kind != AGG) or bound.shapes[
                    last.left
                ] != ROWS:
                    capabilities.reject(
                        "plan",
                        "reductions require a row expression in final select",
                    )
                reduction = last.op
                capabilities.require_reduction(
                    bound.dtypes[last.left], reduction, frame.height()
                )
                capabilities.require_dtype(result_dtype)
                var checked_sum = (
                    result_dtype == DataType.INT32
                    or result_dtype == DataType.UINT32
                    or (
                        capabilities.bounded_integer_sum
                        and (
                            bound.dtypes[last.left].physical() == DataType.INT64
                            or bound.dtypes[last.left] == DataType.UINT64
                        )
                    )
                )
                if reduction == SUM and checked_sum and not terminal_boundary:
                    capabilities.reject(
                        "plan", "checked sums require a terminal boundary"
                    )
                if (
                    node.kind == AGG
                    and reduction == SUM
                    and checked_sum
                    and (k != 0 or limit >= 0)
                ):
                    capabilities.reject(
                        "plan",
                        "checked grouped sums require terminal aggregation without head",
                    )
                min_count = (
                    Int(last.integer) if reduction == ANY
                    or reduction == ALL else last.min_count
                )
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
                    False,
                    False,
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
        if node.kind == AGG:
            if len(aggregate_outputs) != len(node.exprs):
                capabilities.reject(
                    "plan", "group aggregation requires reductions"
                )
            if next_slot + 3 + len(aggregate_outputs) > ROW_MAX_SLOTS:
                capabilities.reject(
                    "plan", "group aggregation exceeds 64 value slots"
                )
            var index_slot = next_slot
            var start_slot = next_slot + 1
            var first_slot = next_slot + 2
            next_slot += 3
            slot_dtypes.extend([DataType.INT64, DataType.INT64, DataType.INT64])
            var start = len(literals)
            code.extend(
                [
                    Int64(
                        ROW_GROUP_HEADER_LAST if len(node.flags)
                        and node.flags[0] else ROW_GROUP_HEADER
                    ),
                    Int64(len(group_keys)),
                    Int64(start_slot),
                    Int64(first_slot),
                ]
            )
            literals.append(0)
            node_dtypes.append(DataType.INT64)
            var grouped_schema = List[Series]()
            var grouped_slots = List[Int]()
            var group_names = List[String]()
            for key in group_keys:
                code.extend(
                    [
                        Int64(ROW_GROUP_KEY),
                        Int64(slots[key]),
                        Int64(0),
                        Int64(0),
                    ]
                )
                literals.append(0)
                node_dtypes.append(schema[key].dtype())
                grouped_schema.append(
                    _empty(schema[key].name(), schema[key].dtype())
                )
                grouped_slots.append(slots[key])
                group_names.append(schema[key].name())
            for output in aggregate_outputs:
                if output.name in group_names:
                    raise Error("Duplicate group output name: " + output.name)
                group_names.append(output.name)
                code.extend(
                    [
                        Int64(ROW_GROUP_OUTPUT),
                        Int64(output.slot),
                        Int64(next_slot),
                        Int64(output.reduction),
                    ]
                )
                literals.append(bitcast[DType.float64](Int64(output.min_count)))
                node_dtypes.append(output.dtype)
                slot_dtypes.append(output.dtype)
                grouped_schema.append(_empty(output.name, output.dtype))
                grouped_slots.append(next_slot)
                next_slot += 1
            steps.append(
                RowStep(
                    start,
                    1 + len(group_keys) + len(aggregate_outputs),
                    index_slot,
                    False,
                    0,
                    0,
                    False,
                    True,
                )
            )
            if node.maintain_order:
                var order_start = len(literals)
                code.extend(
                    [Int64(ROW_SORT_KEY), Int64(first_slot), Int64(0), Int64(0)]
                )
                literals.append(0)
                node_dtypes.append(DataType.INT64)
                var gather_start = len(gathers)
                for slot in grouped_slots:
                    gathers.append(Int64(slot))
                steps.append(
                    RowStep(
                        order_start,
                        1,
                        0,
                        False,
                        gather_start,
                        len(grouped_slots),
                        True,
                        False,
                    )
                )
            schema = grouped_schema^
            slots = grouped_slots^
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
    if len(steps) > ROW_MAX_STEPS:
        capabilities.reject(
            "plan", "resident region supports at most 64 device steps"
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
