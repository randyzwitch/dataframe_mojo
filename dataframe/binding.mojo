"""Schema-only binding and expression shape analysis before execution."""
from .expr import (
    Expr,
    COL,
    LIT_INT,
    UNTYPED,
    Node,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_STRING,
    ADD,
    SUB,
    MUL,
    GT,
    EQ,
    SUM,
    COUNT,
    NE,
    DIV,
    FLOORDIV,
    MOD,
    POW,
    CLIP_LOW,
    CLIP_HIGH,
    NEG,
    ABS,
    SQRT,
    EXP,
    LOG,
    FLOOR,
    CEIL,
    ROUND,
    LIT_NULL,
    AND,
    OR,
    XOR,
    FILL_NULL,
    FILL_NAN,
    KEEP_NULLS,
    NOT,
    IS_NULL,
    IS_NOT_NULL,
    IS_NAN,
    IS_NOT_NAN,
    IS_FINITE,
    IS_INFINITE,
    ANY,
    ALL,
    NULL_COUNT,
    MIN,
    MAX,
    MEAN,
    FIRST,
    LAST,
    N_UNIQUE,
    STD,
    VAR,
    MEDIAN,
    QUANTILE,
    LEN,
    Node,
    WHEN,
    SELECTOR,
    SEP,
    CUM_SUM,
    CUM_COUNT,
    RANK,
    ROLLING_SUM,
    ROLLING_MEAN,
    ROLLING_MIN,
    ROLLING_MAX,
    FORWARD_FILL,
    BACKWARD_FILL,
    OVER,
    FLOORDIV,
    DT_YEAR,
    DT_DAY,
    DT_NANOSECOND,
    DT_DATE,
    DT_TIME,
    DT_TRUNCATE,
    DT_OFFSET_BY,
    DT_TOTAL,
    DT_STRFTIME,
    DT_STRPTIME,
    is_dt_op,
    is_window,
    STR_CONCAT,
    CAST,
    STR_LEN_CHARS,
    STR_LEN_BYTES,
    STR_STARTS_WITH,
    STR_ENDS_WITH,
    STR_CONTAINS,
    STR_SLICE,
    STR_PAD,
    is_string_op,
    is_nested_op,
    STR_SPLIT,
    STRUCT_FIELD,
    LIST_LEN,
    LIST_GET,
    LIST_CONTAINS,
    LIST_JOIN,
    LIST_SUM,
    LIST_MIN,
    LIST_MAX,
    LIST_MEAN,
    is_conditional,
    is_logical,
    is_binary,
    is_unary,
    is_reduction,
    is_comparison,
)
from .series import Series
from .dtype import DataType, NUMERIC_DTYPES
from .temporal import parse_every

comptime SCALAR = 0
comptime ROWS = 1
comptime AGGREGATE = 2


@fieldwise_init
struct BoundExpr(Copyable):
    var expr: Expr
    var dtypes: List[DataType]
    var shapes: List[Int]
    var aggregated: List[Bool]
    var sources: List[Int]
    # True where the node's whole subtree can run in one fused Float64 kernel.
    var fusible: List[Bool]

    def shape(self) -> Int:
        return self.shapes[len(self.shapes) - 1]


def op_name(op: Int) -> String:
    if op == ADD:
        return "+"
    if op == SUB:
        return "-"
    if op == MUL:
        return "*"
    if op == DIV:
        return "/"
    if op == FLOORDIV:
        return "//"
    if op == MOD:
        return "%"
    if op == POW:
        return "pow"
    if op == GT:
        return ">"
    if op == EQ:
        return "eq"
    if op == NE:
        return "ne"
    if op == CLIP_LOW or op == CLIP_HIGH:
        return "clip"
    if op == NEG:
        return "negation"
    if op == ABS:
        return "abs"
    if op == SQRT:
        return "sqrt"
    if op == EXP:
        return "exp"
    if op == LOG:
        return "log"
    if op == FLOOR:
        return "floor"
    if op == CEIL:
        return "ceil"
    if op == ROUND:
        return "round"
    if op == AND:
        return "and"
    if op == OR:
        return "or"
    if op == XOR:
        return "xor"
    if op == NOT:
        return "not"
    if op == FILL_NULL:
        return "fill_null"
    if op == FILL_NAN:
        return "fill_nan"
    if op == IS_NAN:
        return "is_nan"
    if op == IS_NOT_NAN:
        return "is_not_nan"
    if op == IS_FINITE:
        return "is_finite"
    if op == IS_INFINITE:
        return "is_infinite"
    if op == ANY:
        return "any"
    if op == ALL:
        return "all"
    if op == NULL_COUNT:
        return "null_count"
    if op == MIN:
        return "min"
    if op == MAX:
        return "max"
    if op == MEAN:
        return "mean"
    if op == FIRST:
        return "first"
    if op == LAST:
        return "last"
    if op == N_UNIQUE:
        return "n_unique"
    if op == STD:
        return "std"
    if op == VAR:
        return "var"
    if op == MEDIAN:
        return "median"
    if op == QUANTILE:
        return "quantile"
    if op == LEN:
        return "len"
    if op == SUM:
        return "sum"
    if op == COUNT:
        return "count"
    if is_comparison(op):
        return "comparison"
    return "operation " + String(op)


def _numeric(dtype: DataType) -> Bool:
    return dtype.is_numeric()


def temporal_result(
    op: Int, left: DataType, right: DataType
) raises -> DataType:
    """Result type of + - * // involving a temporal operand."""
    var ok = False
    var result = left
    if op == SUB and left == right and left.is_datetime():
        result = DataType.duration(left.unit())
        ok = True
    elif op == SUB and left == right and left.is_date():
        result = DataType.duration("ms")
        ok = True
    elif op == SUB and left == right and left.is_time():
        result = DataType.duration("ns")
        ok = True
    elif (op == ADD or op == SUB) and left.is_duration() and right == left:
        result = left
        ok = True
    elif (
        (op == ADD or op == SUB) and left.is_datetime() and right.is_duration()
    ):
        ok = left.unit() == right.unit()
        result = left
    elif op == ADD and left.is_duration() and right.is_datetime():
        ok = left.unit() == right.unit()
        result = right
    elif (op == ADD or op == SUB) and left.is_date() and right.is_duration():
        result = DataType.datetime(right.unit())
        ok = True
    elif (
        (op == MUL or op == FLOORDIV)
        and left.is_duration()
        and right == DataType.INT64
    ):
        result = left
        ok = True
    elif op == MUL and left == DataType.INT64 and right.is_duration():
        result = right
        ok = True
    if not ok:
        raise Error(
            op_name(op)
            + " is not defined for "
            + left.name()
            + " and "
            + right.name()
            + "; cast to matching temporal types and units"
        )
    return result


def _binary_dtype(op: Int, left: DataType, right: DataType) raises -> DataType:
    if left.is_nested() or right.is_nested():
        raise Error(
            op_name(op)
            + " does not support list or struct operands, found "
            + left.name()
            + " and "
            + right.name()
        )
    if op == KEEP_NULLS:
        return right
    if (left.is_temporal() or right.is_temporal()) and (
        op == ADD or op == SUB or op == MUL or op == FLOORDIV
    ):
        return temporal_result(op, left, right)
    if op == STR_CONCAT:
        if left != DataType.STRING or right != DataType.STRING:
            raise Error(
                "concat_str requires string operands, found "
                + left.name()
                + " and "
                + right.name()
            )
        return DataType.STRING
    if left != right:
        raise Error(
            op_name(op)
            + " requires matching dtypes, found "
            + left.name()
            + " and "
            + right.name()
            + "; use typed literals"
        )
    if op == EQ or op == NE:
        return DataType.BOOL
    if is_comparison(op):
        return DataType.BOOL
    if is_logical(op):
        if left != DataType.BOOL:
            raise Error(
                op_name(op) + " requires bool operands, found " + left.name()
            )
        return DataType.BOOL
    if op == FILL_NULL:
        return left
    if op == FILL_NAN:
        if not left.is_float():
            raise Error(
                "fill_nan requires float operands, found " + left.name()
            )
        return left
    if not _numeric(left):
        raise Error(
            op_name(op) + " requires numeric operands, found " + left.name()
        )
    if op == DIV:
        # Integer division yields float64; float32 stays float32.
        return (
            DataType.FLOAT32 if left == DataType.FLOAT32 else DataType.FLOAT64
        )
    return left


def _unary_dtype(op: Int, input: DataType) raises -> DataType:
    if (op == NEG or op == ABS) and input.is_duration():
        return input
    if op == IS_NULL or op == IS_NOT_NULL:
        return DataType.BOOL
    if op == NOT:
        if input != DataType.BOOL:
            raise Error("not requires a bool operand, found " + input.name())
        return DataType.BOOL
    if op >= IS_NAN and op <= IS_INFINITE:
        if not input.is_float():
            raise Error(
                op_name(op) + " requires a float operand, found " + input.name()
            )
        return DataType.BOOL
    if not _numeric(input):
        raise Error(
            op_name(op) + " requires a numeric operand, found " + input.name()
        )
    if op == SQRT or op == EXP or op == LOG:
        return (
            DataType.FLOAT32 if input == DataType.FLOAT32 else DataType.FLOAT64
        )
    return input


def _reduction_dtype(node: Node, input: DataType) raises -> DataType:
    var op = node.op
    if input.is_nested():
        raise Error(
            op_name(op)
            + " does not support list or struct expressions yet, found "
            + input.name()
        )
    if op == SUM:
        if input.is_duration():
            return input
        if not _numeric(input):
            raise Error(
                "sum requires a numeric expression, found " + input.name()
            )
        return input.sum_type()
    if op == COUNT or op == NULL_COUNT or op == N_UNIQUE or op == LEN:
        return DataType.INT64
    if op == ANY or op == ALL:
        if input != DataType.BOOL:
            raise Error(
                op_name(op)
                + " requires a bool expression, found "
                + input.name()
            )
        return DataType.BOOL
    if op == MIN or op == MAX or op == FIRST or op == LAST:
        return input
    if op == MEAN or op == STD or op == VAR or op == MEDIAN or op == QUANTILE:
        if not _numeric(input):
            raise Error(
                op_name(op)
                + " requires a numeric expression, found "
                + input.name()
            )
        if (op == STD or op == VAR) and node.integer < 0:
            raise Error("ddof must be nonnegative")
        if op == QUANTILE:
            if not (node.floating >= 0 and node.floating <= 1):
                raise Error("quantile must be between 0 and 1")
            var method = node.text
            if (
                method != "linear"
                and method != "nearest"
                and method != "lower"
                and method != "higher"
                and method != "midpoint"
            ):
                raise Error(
                    "interpolation must be nearest, lower, higher, midpoint,"
                    " or linear"
                )
        return DataType.FLOAT64
    raise Error("Unsupported reduction: " + op_name(op))


def _fusible(expr: Expr, types: List[DataType], fuse: Bool) -> List[Bool]:
    """Float64 columns and literals, and + - * / or comparisons over them."""
    var n = len(expr._nodes)
    var result = List[Bool](length=n, fill=False)
    if not fuse:
        return result^
    for i in range(n):
        ref node = expr._nodes[i]
        if node.op == COL or node.op == LIT_FLOAT:
            result[i] = types[i] == DataType.FLOAT64
        elif (
            node.op == ADD or node.op == SUB or node.op == MUL or node.op == DIV
        ) or is_comparison(node.op):
            result[i] = (
                result[node.left]
                and result[node.right]
                and types[node.left] == DataType.FLOAT64
                and types[node.right] == DataType.FLOAT64
            )
    return result^


def _dt_dtype(node: Node, input: DataType) raises -> DataType:
    var op = node.op
    if op == DT_STRPTIME:
        if input != DataType.STRING:
            raise Error(
                "strptime requires a string expression, found " + input.name()
            )
        var target = DataType.parse(node.text2)
        if not (target.is_date() or target.is_datetime() or target.is_time()):
            raise Error("strptime target must be date, datetime, or time")
        return target
    if op == DT_TOTAL:
        if not input.is_duration():
            raise Error(
                "total requires a duration expression, found " + input.name()
            )
        var unit = node.text
        if (
            unit != "days"
            and unit != "hours"
            and unit != "minutes"
            and unit != "seconds"
            and unit != "milliseconds"
            and unit != "microseconds"
            and unit != "nanoseconds"
        ):
            raise Error("unknown total unit: " + unit)
        return DataType.INT64
    if not (input.is_date() or input.is_datetime() or input.is_time()):
        raise Error(
            "dt operations require a date, datetime, or time expression, found "
            + input.name()
        )
    if op == DT_STRFTIME:
        return DataType.STRING
    if op == DT_DATE:
        if not input.is_datetime():
            raise Error("dt.date requires a datetime expression")
        return DataType.DATE
    if op == DT_TIME:
        if not input.is_datetime():
            raise Error("dt.time requires a datetime expression")
        return DataType.TIME
    if op == DT_TRUNCATE or op == DT_OFFSET_BY:
        _ = parse_every(node.text)
        return input
    if input.is_time() and op <= DT_DAY:
        raise Error("time values have no calendar fields")
    if input.is_date() and op > DT_DAY and op <= DT_NANOSECOND:
        raise Error("date values have no time-of-day fields")
    return DataType.INT64


def _literal_text(node: Node) -> String:
    if node.op == LIT_INT:
        return String(node.integer)
    return String(node.floating)


def _check_adoption(node: Node, target: DataType) raises:
    """Raise unless an untyped literal can take `target` exactly."""
    var kind = "integer" if node.op == LIT_INT else "float"
    var hint = (
        "; write lit(...) with an explicit type, or cast the other operand"
    )
    if not target.is_numeric():
        raise Error(
            kind
            + " literal "
            + _literal_text(node)
            + " cannot be used with "
            + target.name()
            + hint
        )
    if node.op == LIT_FLOAT and not target.is_float():
        raise Error(
            "float literal "
            + _literal_text(node)
            + " cannot adopt "
            + target.name()
            + " (no implicit float-to-integer conversion)"
            + hint
        )
    if node.op == LIT_INT and target.is_integer():
        var value = node.integer.cast[DType.int128]()
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            comptime if D.is_integral():
                if target == DataType.of(D) and (
                    value < Scalar[D].MIN.cast[DType.int128]()
                    or value > Scalar[D].MAX.cast[DType.int128]()
                ):
                    raise Error(
                        "integer literal "
                        + _literal_text(node)
                        + " does not fit "
                        + target.name()
                        + hint
                    )


def _adopt(
    mut nodes: List[Node],
    mut types: List[DataType],
    root: Int,
    target: DataType,
) raises:
    """Give every untyped node under `root` the dtype `target`.

    Literal leaves are range-checked and rewritten to typed literals; an
    integer adopting a float type becomes a float literal (so Float64
    expressions stay fusible).
    """
    if root < 0 or not types[root].is_untyped():
        return
    types[root] = target
    var op = nodes[root].op
    if (op == LIT_INT or op == LIT_FLOAT) and nodes[root].text == UNTYPED:
        _check_adoption(nodes[root], target)
        if op == LIT_INT and target.is_float():
            nodes[root].op = LIT_FLOAT
            nodes[root].floating = Float64(nodes[root].integer)
        nodes[root].text = target.name()
        return
    var left = nodes[root].left
    var right = nodes[root].right
    var extra = nodes[root].extra
    _adopt(nodes, types, left, target)
    _adopt(nodes, types, right, target)
    _adopt(nodes, types, extra, target)


def _default(
    mut nodes: List[Node], mut types: List[DataType], root: Int
) raises:
    """Give an untyped subexpression its standalone type (Int64/Float64)."""
    if root >= 0 and types[root].is_untyped():
        _adopt(nodes, types, root, types[root].default())


def _join_untyped(a: DataType, b: DataType) -> DataType:
    """The untyped result of combining two untyped operands."""
    if a == DataType.UNTYPED_FLOAT or b == DataType.UNTYPED_FLOAT:
        return DataType.UNTYPED_FLOAT
    return DataType.UNTYPED_INT


def _stays_untyped(op: Int) -> Bool:
    """Binary operations whose untyped operands keep the result untyped."""
    return (
        op == ADD
        or op == SUB
        or op == MUL
        or op == FLOORDIV
        or op == MOD
        or op == POW
        or op == CLIP_LOW
        or op == CLIP_HIGH
    )


def _target(other: DataType) -> DataType:
    """What an untyped operand adopts next to `other`: Int64 beside temporal
    types (for duration * n), otherwise `other` itself."""
    return DataType.INT64 if other.is_temporal() else other


def bind(
    expr: Expr, columns: List[Series], fuse: Bool = True
) raises -> BoundExpr:
    if len(expr._nodes) == 0:
        raise Error("An expression must contain at least one node")
    var types = List[DataType]()
    var shapes = List[Int]()
    var aggregated = List[Bool]()
    var sources = List[Int]()
    # Untyped literals are resolved in place, so bind works on a copy.
    var nodes = expr._nodes.copy()
    for i in range(len(nodes)):
        var node = nodes[i].copy()
        # Operations other than binary arithmetic/comparison, negation, and
        # when/then branches give untyped inputs their default types first.
        if not (
            is_binary(node.op) or is_conditional(node.op) or node.op == NEG
        ):
            _default(nodes, types, node.left)
            _default(nodes, types, node.right)
            _default(nodes, types, node.extra)
        var dtype: DataType
        var shape = SCALAR
        var has_aggregate = False
        var source = -1
        if node.op == COL:
            for j in range(len(columns)):
                if columns[j].name() == node.text:
                    source = j
                    break
            if source < 0:
                raise Error("Unknown expression column: " + node.text)
            dtype = columns[source].dtype()
            shape = ROWS
        elif node.op == LIT_INT:
            if node.text == UNTYPED:
                dtype = DataType.UNTYPED_INT
            else:
                dtype = DataType.INT64 if node.text == "" else DataType.parse(
                    node.text
                )
        elif node.op == LIT_FLOAT:
            if node.text == UNTYPED:
                dtype = DataType.UNTYPED_FLOAT
            else:
                dtype = DataType.FLOAT64 if node.text == "" else DataType.parse(
                    node.text
                )
        elif node.op == LIT_BOOL:
            dtype = DataType.BOOL
        elif node.op == LIT_STRING:
            dtype = DataType.STRING
        elif node.op == SELECTOR:
            raise Error(
                "Selectors must be expanded before binding; use select,"
                " with_columns, agg, or filter"
            )
        elif node.op == LIT_NULL:
            if not DataType.is_known(node.text):
                raise Error("Unknown null literal dtype: " + node.text)
            dtype = DataType.parse(node.text)
        elif is_reduction(node.op):
            if node.left < 0 or node.left >= i:
                raise Error("Invalid aggregate input")
            if shapes[node.left] != ROWS or aggregated[node.left]:
                raise Error(
                    "Aggregate input must be row-valued without nested aggregates"
                )
            if node.min_count < 0:
                raise Error("min_count must be nonnegative")
            dtype = _reduction_dtype(node, types[node.left])
            shape = AGGREGATE
            has_aggregate = True
        elif is_conditional(node.op):
            if (
                node.left < 0
                or node.right < 0
                or node.left >= i
                or node.right >= i
                or node.extra >= i
            ):
                raise Error("Invalid conditional expression inputs")
            # An untyped branch adopts the other branch's type.
            var then_type = types[node.right]
            var else_type = types[node.extra] if node.extra >= 0 else then_type
            if then_type.is_nested() or else_type.is_nested():
                raise Error(
                    "when/then/otherwise does not support list or struct"
                    " branches yet"
                )
            if then_type.is_untyped() and not else_type.is_untyped():
                _adopt(nodes, types, node.right, _target(else_type))
            elif else_type.is_untyped() and not then_type.is_untyped():
                _adopt(nodes, types, node.extra, _target(then_type))
            else:
                var joined = _join_untyped(
                    then_type, else_type
                ).default() if then_type.is_untyped() else then_type
                if then_type.is_untyped():
                    _adopt(nodes, types, node.right, joined)
                    _adopt(nodes, types, node.extra, joined)
            if types[node.left] != DataType.BOOL:
                raise Error(
                    "when requires a bool predicate, found "
                    + types[node.left].name()
                )
            dtype = types[node.right]
            var children = List[Int]()
            children.append(node.left)
            children.append(node.right)
            if node.extra >= 0:
                if types[node.extra] != dtype:
                    raise Error(
                        "when/then/otherwise branches require matching dtypes,"
                        " found "
                        + dtype.name()
                        + " and "
                        + types[node.extra].name()
                    )
                children.append(node.extra)
            for child in children:
                has_aggregate = has_aggregate or aggregated[child]
                if shapes[child] == ROWS:
                    shape = ROWS
            if shape != ROWS and has_aggregate:
                shape = AGGREGATE
        elif is_window(node.op):
            if node.left < 0 or node.left >= i:
                raise Error("Invalid window input")
            if shapes[node.left] != ROWS:
                raise Error("Window operations require a row-valued input")
            var input = types[node.left]
            if input.is_nested():
                raise Error(
                    op_name(node.op)
                    + " does not support list or struct expressions, found "
                    + input.name()
                )
            dtype = input
            if node.op == CUM_SUM or node.op == ROLLING_SUM:
                if not _numeric(input) and not input.is_duration():
                    raise Error(
                        "cum_sum and rolling_sum require a numeric expression,"
                        " found " + input.name()
                    )
                dtype = input.sum_type()
            elif node.op == ROLLING_MEAN:
                if not _numeric(input):
                    raise Error(
                        "rolling_mean requires a numeric expression, found "
                        + input.name()
                    )
                dtype = DataType.FLOAT64
            elif node.op == CUM_COUNT:
                dtype = DataType.INT64
            elif node.op == RANK:
                var method = node.text
                if (
                    method != "average"
                    and method != "min"
                    and method != "max"
                    and method != "dense"
                    and method != "ordinal"
                ):
                    raise Error(
                        "rank method must be average, min, max, dense, or"
                        " ordinal"
                    )
                dtype = (
                    DataType.FLOAT64 if method == "average" else DataType.INT64
                )
            if node.op >= ROLLING_SUM and node.op <= ROLLING_MAX:
                if node.integer < 1:
                    raise Error("window_size must be at least 1")
                if node.floating < -1:
                    raise Error("min_samples must be nonnegative")
            if (
                node.op == FORWARD_FILL or node.op == BACKWARD_FILL
            ) and node.integer < -1:
                raise Error("fill limit must be nonnegative")
            shape = ROWS
            has_aggregate = aggregated[node.left]
        elif is_dt_op(node.op):
            if node.left < 0 or node.left >= i:
                raise Error("Invalid dt input")
            dtype = _dt_dtype(node, types[node.left])
            shape = shapes[node.left]
            has_aggregate = aggregated[node.left]
        elif node.op == OVER:
            if node.left < 0 or node.left >= i:
                raise Error("Invalid over input")
            if node.text2.byte_length() == 0:
                raise Error("over requires at least one partition column")
            for part in node.text2.split(SEP):
                var found = False
                for column in columns:
                    found = found or column.name() == String(part)
                if not found:
                    raise Error("Unknown partition column: " + String(part))
            dtype = types[node.left]
            shape = ROWS
        elif node.op == CAST:
            if node.left < 0 or node.left >= i:
                raise Error("Invalid cast input")
            if not DataType.is_known(node.text):
                raise Error("Unknown cast dtype: " + node.text)
            dtype = DataType.parse(node.text)
            shape = shapes[node.left]
            has_aggregate = aggregated[node.left]
        elif is_string_op(node.op):
            if node.left < 0 or node.left >= i:
                raise Error("Invalid string expression input")
            if types[node.left] != DataType.STRING:
                raise Error(
                    "str operations require a string expression, found "
                    + types[node.left].name()
                )
            if node.op == STR_SLICE and node.min_count < -1:
                raise Error("str.slice length must be nonnegative")
            if node.op == STR_PAD:
                if node.min_count < 0:
                    raise Error("pad width must be nonnegative")
                if len(node.text.codepoints()) != 1:
                    raise Error("pad fill_char must be one character")
            if node.op == STR_LEN_CHARS or node.op == STR_LEN_BYTES:
                dtype = DataType.INT64
            elif (
                node.op == STR_STARTS_WITH
                or node.op == STR_ENDS_WITH
                or node.op == STR_CONTAINS
            ):
                dtype = DataType.BOOL
            else:
                dtype = DataType.STRING
            shape = shapes[node.left]
            has_aggregate = aggregated[node.left]
        elif is_nested_op(node.op):
            if node.left < 0 or node.left >= i:
                raise Error("Invalid list expression input")
            var input = types[node.left]
            if node.op == STR_SPLIT:
                if input != DataType.STRING:
                    raise Error(
                        "str.split requires a string expression, found "
                        + input.name()
                    )
                dtype = DataType.list(DataType.STRING)
            elif node.op == STRUCT_FIELD:
                if not input.is_struct():
                    raise Error(
                        "field() requires a struct expression, found "
                        + input.name()
                    )
                dtype = input.field_dtype(input.field_index(node.text))
            else:
                if not input.is_list():
                    raise Error(
                        op_name(node.op)
                        + " requires a list expression, found "
                        + input.name()
                    )
                var inner = input.inner()
                if node.op == LIST_LEN:
                    dtype = DataType.INT64
                elif node.op == LIST_GET:
                    dtype = inner
                elif node.op == LIST_CONTAINS:
                    var wanted = (
                        DataType.STRING if node.min_count == 0 else inner
                    )
                    if node.min_count == 0 and inner != DataType.STRING:
                        raise Error(
                            "list.contains with a string needs a string"
                            " list, found " + input.name()
                        )
                    if node.min_count == 1 and not inner.is_integer():
                        raise Error(
                            "list.contains with an integer needs an integer"
                            " list, found " + input.name()
                        )
                    if node.min_count == 2 and not inner.is_float():
                        raise Error(
                            "list.contains with a float needs a float list,"
                            " found " + input.name()
                        )
                    _ = wanted
                    dtype = DataType.BOOL
                elif node.op == LIST_JOIN:
                    if inner != DataType.STRING:
                        raise Error(
                            "list.join needs a string list, found "
                            + input.name()
                        )
                    dtype = DataType.STRING
                elif node.op == LIST_MEAN:
                    if not inner.is_numeric():
                        raise Error(
                            "list.mean needs a numeric list, found "
                            + input.name()
                        )
                    dtype = DataType.FLOAT64
                else:
                    if not inner.is_numeric():
                        raise Error(
                            op_name(node.op)
                            + " needs a numeric list, found "
                            + input.name()
                        )
                    dtype = inner.sum_type() if node.op == LIST_SUM else inner
            shape = shapes[node.left]
            has_aggregate = aggregated[node.left]
        elif is_unary(node.op):
            if node.left < 0 or node.left >= i:
                raise Error("Invalid unary expression input")
            if node.op == NEG and types[node.left].is_untyped():
                dtype = types[node.left]  # -(untyped) stays untyped
            else:
                _default(nodes, types, node.left)
                dtype = _unary_dtype(node.op, types[node.left])
            shape = shapes[node.left]
            has_aggregate = aggregated[node.left]
        elif is_binary(node.op):
            if (
                node.left < 0
                or node.right < 0
                or node.left >= i
                or node.right >= i
            ):
                raise Error("Invalid binary expression inputs")
            var left = types[node.left]
            var right = types[node.right]
            if left.is_untyped() and right.is_untyped():
                if _stays_untyped(node.op):
                    dtype = _join_untyped(left, right)
                else:
                    var joined = _join_untyped(left, right).default()
                    _adopt(nodes, types, node.left, joined)
                    _adopt(nodes, types, node.right, joined)
                    dtype = _binary_dtype(
                        node.op, types[node.left], types[node.right]
                    )
            else:
                if left.is_untyped():
                    _adopt(nodes, types, node.left, _target(right))
                elif right.is_untyped():
                    _adopt(nodes, types, node.right, _target(left))
                dtype = _binary_dtype(
                    node.op, types[node.left], types[node.right]
                )
            has_aggregate = aggregated[node.left] or aggregated[node.right]
            if shapes[node.left] == ROWS or shapes[node.right] == ROWS:
                shape = ROWS
            elif has_aggregate:
                shape = AGGREGATE
        else:
            raise Error("Unknown expression operation")
        types.append(dtype)
        shapes.append(shape)
        aggregated.append(has_aggregate)
        sources.append(source)
    # A standalone untyped result takes its default type.
    _default(nodes, types, len(nodes) - 1)
    var resolved = Expr(nodes^, expr._name)
    var fusible = _fusible(resolved, types, fuse)
    return BoundExpr(
        resolved^, types^, shapes^, aggregated^, sources^, fusible^
    )
