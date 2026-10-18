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
    ARG_MIN,
    ARG_MAX,
    MODE,
    SKEW,
    KURTOSIS,
    CORR,
    COV,
    VALUE_COUNTS,
    is_pair_reduction,
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
    SELECTOR,
    SEP,
    split_names,
    CUM_SUM,
    CUM_COUNT,
    RANK,
    ROLLING_SUM,
    ROLLING_STD,
    ROLLING_VAR,
    ROLLING_SUM_BY,
    ROLLING_MIN_BY,
    ROLLING_MAX_BY,
    is_rolling_by,
    ROLLING_MEAN,
    ROLLING_MAX,
    FORWARD_FILL,
    BACKWARD_FILL,
    INTERPOLATE,
    INTERPOLATE_BY,
    CUT,
    QCUT,
    OVER,
    FLOORDIV,
    DT_DAY,
    DT_NANOSECOND,
    DT_DATE,
    DT_TIME,
    DT_TRUNCATE,
    DT_OFFSET_BY,
    DT_TOTAL,
    DT_STRFTIME,
    DT_STRPTIME,
    DT_REPLACE_TZ,
    DT_CONVERT_TZ,
    is_dt_op,
    is_window,
    STR_CONCAT,
    CAST,
    STR_LEN_CHARS,
    STR_LEN_BYTES,
    STR_STARTS_WITH,
    STR_ENDS_WITH,
    STR_CONTAINS,
    STR_IS_IN,
    INT_IS_IN,
    STR_LIKE,
    STR_SLICE,
    STR_PAD,
    is_string_op,
    is_text_function,
    STR_TITLECASE,
    STR_STRIP_PREFIX,
    STR_STRIP_SUFFIX,
    STR_FIND,
    STR_COUNT_MATCHES,
    STR_CONTAINS_ANY,
    STR_REPLACE_MANY,
    STR_ENCODE,
    STR_DECODE,
    STR_TO_INTEGER,
    STR_SPLIT_FIELDS,
    is_nested_op,
    IMPLODE,
    STRUCT_PACK,
    struct_pack_children,
    struct_pack_names,
    HORIZONTAL,
    HORIZONTAL_ARGS,
    H_ANY,
    H_ALL,
    horizontal_inputs,
    STR_SPLIT,
    STRUCT_FIELD,
    LIST_LEN,
    LIST_GET,
    LIST_CONTAINS,
    LIST_JOIN,
    LIST_SUM,
    LIST_MEAN,
    is_conditional,
    is_logical,
    is_binary,
    is_unary,
    is_reduction,
    is_comparison,
    is_math,
    ARCTAN2,
    LOG_BASE,
    LOG1P,
    SIGN,
    CBRT,
    TRUNCATE,
    SIN,
    COS,
    TAN,
    ARCSIN,
    ARCCOS,
    ARCTAN,
    SINH,
    COSH,
    TANH,
    ARCSINH,
    ARCCOSH,
    ARCTANH,
    COT,
    DEGREES,
    RADIANS,
    PRODUCT,
    CUM_PROD,
)
from .series import Series
from .dtype import DataType, NUMERIC_DTYPES
from .decimal import common_decimal, decimal_add_type, decimal_mul_type
from .horizontal import (
    horizontal_input,
    horizontal_result,
    horizontal_supertype,
)
from .temporal import (
    ambiguous_code,
    non_existent_code,
    parse_every,
    strptime_target,
)

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


def _text_function_name(op: Int) -> String:
    if op == STR_TITLECASE:
        return "str.to_titlecase"
    if op == STR_STRIP_PREFIX:
        return "str.strip_prefix"
    if op == STR_STRIP_SUFFIX:
        return "str.strip_suffix"
    if op == STR_FIND:
        return "str.find"
    if op == STR_COUNT_MATCHES:
        return "str.count_matches"
    if op == STR_CONTAINS_ANY:
        return "str.contains_any"
    if op == STR_REPLACE_MANY:
        return "str.replace_many"
    if op == STR_ENCODE:
        return "str.encode"
    if op == STR_DECODE:
        return "str.decode"
    if op == STR_TO_INTEGER:
        return "str.to_integer"
    return "str.split_exact"


def op_name(op: Int) -> String:
    if is_text_function(op):
        return _text_function_name(op)
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
    if op == LOG_BASE:
        return "log"
    if op == LOG1P:
        return "log1p"
    if op == SIGN:
        return "sign"
    if op == CBRT:
        return "cbrt"
    if op == TRUNCATE:
        return "truncate"
    if op == SIN:
        return "sin"
    if op == COS:
        return "cos"
    if op == TAN:
        return "tan"
    if op == ARCSIN:
        return "arcsin"
    if op == ARCCOS:
        return "arccos"
    if op == ARCTAN:
        return "arctan"
    if op == SINH:
        return "sinh"
    if op == COSH:
        return "cosh"
    if op == TANH:
        return "tanh"
    if op == ARCSINH:
        return "arcsinh"
    if op == ARCCOSH:
        return "arccosh"
    if op == ARCTANH:
        return "arctanh"
    if op == COT:
        return "cot"
    if op == DEGREES:
        return "degrees"
    if op == RADIANS:
        return "radians"
    if op == ARCTAN2:
        return "arctan2"
    if op == PRODUCT:
        return "product"
    if op == CUM_PROD:
        return "cum_prod"
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
    if op == INTERPOLATE:
        return "interpolate"
    if op == INTERPOLATE_BY:
        return "interpolate_by"
    if op == CUT:
        return "cut"
    if op == QCUT:
        return "qcut"
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
    if op == ARG_MIN:
        return "arg_min"
    if op == ARG_MAX:
        return "arg_max"
    if op == MODE:
        return "mode"
    if op == SKEW:
        return "skew"
    if op == KURTOSIS:
        return "kurtosis"
    if op == CORR:
        return "corr"
    if op == COV:
        return "cov"
    if op == VALUE_COUNTS:
        return "value_counts"
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
    if left.is_decimal() or right.is_decimal():
        if not left.is_decimal() or not right.is_decimal():
            raise Error(op_name(op) + " requires two decimal operands")
        if is_comparison(op):
            return DataType.BOOL
        # DuckDB's decimal arithmetic types: sums and differences at the
        # larger scale with one more integer digit, exact products, and
        # quotients as DOUBLE.
        if op == ADD or op == SUB:
            return decimal_add_type(left, right)
        if op == MUL:
            return decimal_mul_type(left, right)
        if op == DIV:
            return DataType.FLOAT64
        if op == FILL_NULL:
            return common_decimal(left, right)
        raise Error(op_name(op) + " is not supported for decimal operands")
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
    if op == SIGN or op == TRUNCATE:
        return input
    if op == SQRT or op == EXP or op == LOG or is_math(op):
        return (
            DataType.FLOAT32 if input == DataType.FLOAT32 else DataType.FLOAT64
        )
    return input


def _reaches(nodes: List[Node], root: Int, op: Int) -> Bool:
    """Whether the subtree under `root` contains a node with `op`."""
    var pending = List[Int]()
    pending.append(root)
    while len(pending) > 0:
        var index = pending.pop()
        if index < 0:
            continue
        ref node = nodes[index]
        if node.op == op:
            return True
        pending.append(node.left)
        pending.append(node.right)
        pending.append(node.extra)
    return False


def _reduction_dtype(node: Node, input: DataType) raises -> DataType:
    var op = node.op
    if op == IMPLODE:
        return DataType.list(input)
    if input.is_nested():
        raise Error(
            op_name(op)
            + " does not support list or struct expressions yet, found "
            + input.name()
        )
    if op == SUM:
        if input.is_duration():
            return input
        if input == DataType.BOOL:
            # The number of true values, as UInt32 like Polars' sum of a
            # Boolean (its index type), not the Int64 of count and len.
            return DataType.UINT32
        if not _numeric(input):
            raise Error(
                "sum requires a numeric expression, found " + input.name()
            )
        return input.sum_type()
    if op == PRODUCT:
        if not _numeric(input) and input != DataType.BOOL:
            raise Error(
                "product requires a numeric expression, found " + input.name()
            )
        return input.product_type()
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
    if op == ARG_MIN or op == ARG_MAX:
        return DataType.UINT32
    if op == MODE:
        return DataType.list(input)
    if op == VALUE_COUNTS:
        var names = split_names(node.text)
        if names[0] == names[1]:
            raise Error(
                "value_counts count name must differ from the value name: "
                + names[0]
            )
        var types: List[DataType] = [
            input,
            DataType.FLOAT64 if (node.integer & 2) != 0 else DataType.UINT32,
        ]
        return DataType.list(DataType.struct(names, types))
    if op == SKEW or op == KURTOSIS:
        if not _numeric(input):
            raise Error(
                op_name(op)
                + " requires a numeric expression, found "
                + input.name()
            )
        return DataType.FLOAT64
    if op == MEAN or op == STD or op == VAR or op == MEDIAN or op == QUANTILE:
        if op == MEAN and input.is_decimal():
            # Float64, as Polars returns (#341): a mean rarely has the
            # input's scale, and truncating to it lost digits.
            return DataType.FLOAT64
        if op == MEAN and input == DataType.BOOL:
            # The share of true values among the valid ones.
            return DataType.FLOAT64
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
            # A comparison of two leaves (columns or literals) runs faster
            # in the bitmap kernel than through the fused interpreter (#327);
            # comparisons over arithmetic still fuse.
            if (
                is_comparison(node.op)
                and _leaf(expr, node.left)
                and _leaf(expr, node.right)
            ):
                result[i] = False
    return result^


def _leaf(expr: Expr, index: Int) -> Bool:
    var op = expr._nodes[index].op
    return op == COL or op == LIT_FLOAT


def _dt_dtype(node: Node, input: DataType) raises -> DataType:
    var op = node.op
    if op == DT_STRPTIME:
        if input != DataType.STRING:
            raise Error(
                "strptime requires a string expression, found " + input.name()
            )
        if len(node.dtypes) == 0 or not node.dtypes[0]:
            raise Error("Unknown dtype: " + node.text2)
        var target = node.dtypes[0].value()
        if not (target.is_date() or target.is_datetime() or target.is_time()):
            raise Error("strptime target must be date, datetime, or time")
        return strptime_target(target, node.text)
    if op == DT_REPLACE_TZ or op == DT_CONVERT_TZ:
        var name = (
            "replace_time_zone" if op == DT_REPLACE_TZ else "convert_time_zone"
        )
        if not input.is_datetime():
            raise Error(
                name + " requires a datetime expression, found " + input.name()
            )
        if op == DT_CONVERT_TZ and node.text.byte_length() == 0:
            raise Error(
                "convert_time_zone needs a time zone; use"
                " replace_time_zone('') to drop one"
            )
        if op == DT_REPLACE_TZ:
            var options = node.text2.split(",")
            _ = ambiguous_code(String(options[0]))
            _ = non_existent_code(String(options[1]))
        return input.with_time_zone(node.text)
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
        nodes[root].dtypes = [Optional(target)]
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


def _horizontal_target(untyped: DataType, other: DataType) -> DataType:
    """What an untyped input of a horizontal reduction adopts beside a
    typed one: an integer takes a number's type, a float a float's, and
    otherwise the literal keeps its own default (Int64 or Float64) for the
    supertype to combine."""
    if other.is_float() or (
        untyped == DataType.UNTYPED_INT and other.is_integer()
    ):
        return other
    return untyped.default()


def _target(other: DataType) -> DataType:
    """What an untyped operand adopts next to `other`: Int64 beside temporal
    types (for duration * n), otherwise `other` itself."""
    return DataType.INT64 if other.is_temporal() else other


def _packed_count(text: String) -> Int:
    """How many values `pack_literals` packed into text."""
    var bytes = text.as_bytes()
    var count = 0
    var at = 0
    while at < len(bytes):
        var length = 0
        while bytes[at] != 58:
            length = length * 10 + Int(bytes[at]) - 48
            at += 1
        at += 1 + length
        count += 1
    return count


def _has_regex_syntax(pattern: String) -> Bool:
    """Whether a pattern holds a character with a meaning in a regular
    expression, so that matching it literally could differ from Polars."""
    for byte in pattern.as_bytes():
        # \ . + * ? ( ) | [ ] { } ^ $
        if (
            byte == 92
            or byte == 46
            or byte == 43
            or byte == 42
            or byte == 63
            or byte == 40
            or byte == 41
            or byte == 124
            or byte == 91
            or byte == 93
            or byte == 123
            or byte == 125
            or byte == 94
            or byte == 36
        ):
            return True
    return False


def _text_function_dtype(node: Node) raises -> DataType:
    """The result dtype of a text_functions.mojo operation, after checking
    its arguments."""
    var op = node.op
    if op == STR_FIND or op == STR_COUNT_MATCHES:
        if node.integer == 0 and _has_regex_syntax(node.text):
            raise Error(
                op_name(op) + ": regular expressions are not supported; pass"
                " literal=True to match the pattern as text"
            )
        return DataType.UINT32
    if op == STR_CONTAINS_ANY or op == STR_REPLACE_MANY:
        var patterns = _packed_count(node.text)
        if patterns == 0:
            raise Error(op_name(op) + " needs at least one pattern")
        if op == STR_CONTAINS_ANY:
            return DataType.BOOL
        var replacements = _packed_count(node.text2)
        if replacements != 1 and replacements != patterns:
            raise Error(
                "str.replace_many needs one replacement, or one per pattern;"
                " found "
                + String(replacements)
                + " for "
                + String(patterns)
                + " patterns"
            )
        return DataType.STRING
    if op == STR_ENCODE or op == STR_DECODE:
        if node.text != "hex" and node.text != "base64":
            raise Error(
                op_name(op)
                + " encoding must be 'hex' or 'base64', found '"
                + node.text
                + "'"
            )
        return DataType.STRING if op == STR_ENCODE else DataType.BINARY
    if op == STR_TO_INTEGER:
        if node.min_count < 2 or node.min_count > 36:
            raise Error(
                "str.to_integer base must be between 2 and 36, found "
                + String(node.min_count)
            )
        return DataType.INT64
    if op == STR_SPLIT_FIELDS:
        if node.min_count < 1:
            raise Error(
                "str.split_exact needs n >= 0 and str.splitn needs n >= 1"
            )
        var names = List[String]()
        var dtypes = List[DataType]()
        for k in range(node.min_count):
            names.append("field_" + String(k))
            dtypes.append(DataType.STRING)
        return DataType.struct(names, dtypes)
    return DataType.STRING


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
            is_binary(node.op)
            or is_conditional(node.op)
            or node.op == NEG
            or node.op == HORIZONTAL_ARGS
            or node.op == HORIZONTAL
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
            # Expressions read a categorical as its strings (#106); a bare
            # column keeps its type, since evaluate returns it as is.
            if dtype.is_categorical() and len(nodes) > 1:
                dtype = DataType.STRING
            shape = ROWS
        elif node.op == LIT_INT:
            if node.text == UNTYPED:
                dtype = DataType.UNTYPED_INT
            else:
                dtype = node.dtypes[0].value() if len(
                    node.dtypes
                ) else DataType.INT64
        elif node.op == LIT_FLOAT:
            if node.text == UNTYPED:
                dtype = DataType.UNTYPED_FLOAT
            else:
                dtype = node.dtypes[0].value() if len(
                    node.dtypes
                ) else DataType.FLOAT64
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
            if len(node.dtypes) == 0 or not node.dtypes[0]:
                raise Error("Unknown null literal dtype: " + node.text)
            dtype = node.dtypes[0].value()
        elif is_reduction(node.op):
            if node.left < 0 or node.left >= i:
                raise Error("Invalid aggregate input")
            if shapes[node.left] != ROWS or aggregated[node.left]:
                raise Error(
                    "Aggregate input must be row-valued without nested aggregates"
                )
            if node.min_count < 0:
                raise Error("min_count must be nonnegative")
            if is_pair_reduction(node.op):
                if node.right < 0 or node.right >= i:
                    raise Error("Invalid aggregate input")
                if shapes[node.right] != ROWS or aggregated[node.right]:
                    raise Error(
                        "Aggregate input must be row-valued without nested"
                        " aggregates"
                    )
                for side in [node.left, node.right]:
                    if types[side].is_untyped():
                        _default(nodes, types, side)
                    if not _numeric(types[side]):
                        raise Error(
                            op_name(node.op)
                            + " requires numeric expressions, found "
                            + types[side].name()
                        )
                if node.op == CORR and (
                    node.text != "pearson" and node.text != "spearman"
                ):
                    raise Error("corr method must be pearson or spearman")
                if node.op == COV and node.integer < 0:
                    raise Error("ddof must be nonnegative")
                dtype = DataType.FLOAT64
            else:
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
            if (
                node.extra >= 0
                and dtype.is_decimal()
                and types[node.extra].is_decimal()
            ):
                dtype = common_decimal(dtype, types[node.extra])
            var children = List[Int]()
            children.append(node.left)
            children.append(node.right)
            if node.extra >= 0:
                if types[node.extra] != dtype and not (
                    dtype.is_decimal() and types[node.extra].is_decimal()
                ):
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
            if node.op == INTERPOLATE_BY:
                if node.right < 0 or node.right >= i:
                    raise Error("Invalid interpolate_by input")
                if shapes[node.right] != ROWS:
                    raise Error(
                        "interpolate_by requires a row-valued by expression"
                    )
                if not _numeric(types[node.right]):
                    raise Error(
                        "interpolate_by requires a numeric by expression, found "
                        + types[node.right].name()
                    )
            if node.op == INTERPOLATE or node.op == INTERPOLATE_BY:
                if not _numeric(input):
                    raise Error(
                        op_name(node.op)
                        + " requires a numeric expression, found "
                        + input.name()
                    )
                if node.op == INTERPOLATE and (
                    node.text != "linear" and node.text != "nearest"
                ):
                    raise Error("interpolate method must be linear or nearest")
                if node.op == INTERPOLATE_BY or node.text == "linear":
                    dtype = DataType.FLOAT64
            elif node.op == CUT or node.op == QCUT:
                if not _numeric(input):
                    raise Error(
                        op_name(node.op)
                        + " requires a numeric expression, found "
                        + input.name()
                    )
                var edge_count = 0 if node.text.byte_length() == 0 else len(
                    node.text.split(SEP)
                )
                if (
                    node.text2.byte_length() > 0
                    and len(split_names(node.text2)) != edge_count + 1
                ):
                    raise Error("labels must have one more value than breaks")
                if (node.integer & 2) != 0:
                    dtype = DataType.struct(
                        [String("breakpoint"), "category"],
                        [DataType.FLOAT64, DataType.STRING],
                    )
                else:
                    dtype = DataType.STRING
            elif is_rolling_by(node.op):
                if (
                    node.right < 0
                    or node.right >= i
                    or shapes[node.right] != ROWS
                ):
                    raise Error(
                        "rolling_by requires a row-valued by expression"
                    )
                var index_type = types[node.right]
                if not (index_type.is_date() or index_type.is_datetime()):
                    raise Error("rolling_by requires a Date or Datetime index")
                if not _numeric(input):
                    raise Error("rolling_by requires numeric values")
                if node.floating < 0 or node.min_count < 0:
                    raise Error("min_samples and ddof must be nonnegative")
                if node.op == ROLLING_SUM_BY:
                    dtype = input.sum_type()
                elif node.op != ROLLING_MIN_BY and node.op != ROLLING_MAX_BY:
                    dtype = DataType.FLOAT64
            elif node.op == CUM_SUM and input == DataType.BOOL:
                # A running count of true values, UInt32 as in Polars.
                dtype = DataType.UINT32
            elif node.op == CUM_PROD:
                if not _numeric(input) and input != DataType.BOOL:
                    raise Error(
                        "cum_prod requires a numeric expression, found "
                        + input.name()
                    )
                dtype = input.product_type()
            elif node.op == CUM_SUM or node.op == ROLLING_SUM:
                if not _numeric(input) and not input.is_duration():
                    raise Error(
                        "cum_sum and rolling_sum require a numeric expression,"
                        " found " + input.name()
                    )
                dtype = input.sum_type()
            elif (
                node.op == ROLLING_MEAN
                or node.op == ROLLING_STD
                or node.op == ROLLING_VAR
            ):
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
            if (
                (node.op >= ROLLING_SUM and node.op <= ROLLING_MAX)
                or node.op == ROLLING_STD
                or node.op == ROLLING_VAR
            ):
                if node.integer < 1:
                    raise Error("window_size must be at least 1")
                if node.floating < -1:
                    raise Error("min_samples must be nonnegative")
            if (
                node.op == FORWARD_FILL or node.op == BACKWARD_FILL
            ) and node.integer < -1:
                raise Error("fill limit must be nonnegative")
            if (
                node.op == ROLLING_STD or node.op == ROLLING_VAR
            ) and node.min_count < 0:
                raise Error("ddof must be nonnegative")
            if (
                node.op == ROLLING_STD or node.op == ROLLING_VAR
            ) and node.floating > Float64(node.integer):
                raise Error("min_samples cannot exceed window_size")
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
            for part in split_names(node.text2):
                var found = False
                for column in columns:
                    found = found or column.name() == part
                if not found:
                    raise Error("Unknown partition column: " + part)
            if _reaches(nodes, node.left, MODE) or _reaches(
                nodes, node.left, VALUE_COUNTS
            ):
                raise Error(
                    "mode() and value_counts() return a list per group and"
                    " cannot be used in over(); use group_by().agg() instead"
                )
            dtype = types[node.left]
            shape = ROWS
        elif node.op == CAST:
            if node.left < 0 or node.left >= i:
                raise Error("Invalid cast input")
            if len(node.dtypes) == 0 or not node.dtypes[0]:
                raise Error("Unknown cast dtype: " + node.text)
            dtype = node.dtypes[0].value()
            shape = shapes[node.left]
            has_aggregate = aggregated[node.left]
        elif node.op == INT_IS_IN:
            if node.left < 0 or node.left >= i:
                raise Error("Invalid is_in input")
            var wanted = node.dtypes[0].value()
            if types[node.left] != wanted:
                # The same rule and message as the `==` it stands for.
                raise Error(
                    "eq requires matching dtypes, found "
                    + types[node.left].name()
                    + " and "
                    + wanted.name()
                    + "; use typed literals"
                )
            dtype = DataType.BOOL
            shape = shapes[node.left]
            has_aggregate = aggregated[node.left]
        elif is_string_op(node.op):
            if node.left < 0 or node.left >= i:
                raise Error("Invalid string expression input")
            if types[node.left] != DataType.STRING:
                # `is_in` with string values reads as `==` (#373).
                if node.op == STR_IS_IN:
                    raise Error(
                        "eq requires matching dtypes, found "
                        + types[node.left].name()
                        + " and string; use typed literals"
                    )
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
                or node.op == STR_IS_IN
                or node.op == STR_LIKE
            ):
                dtype = DataType.BOOL
            else:
                dtype = DataType.STRING
            shape = shapes[node.left]
            has_aggregate = aggregated[node.left]
        elif node.op == HORIZONTAL_ARGS:
            # Two runs of a horizontal reduction's inputs, typed as their
            # supertype; untyped literals adopt the other side's type.
            if (
                node.left < 0
                or node.right < 0
                or node.left >= i
                or node.right >= i
            ):
                raise Error("Invalid horizontal expression inputs")
            var kind = Int(node.integer)
            var left = types[node.left]
            var right = types[node.right]
            if kind == H_ANY or kind == H_ALL:
                _default(nodes, types, node.left)
                _default(nodes, types, node.right)
                dtype = DataType.BOOL
            elif left.is_untyped() and right.is_untyped():
                dtype = _join_untyped(left, right)
            else:
                if left.is_untyped():
                    _adopt(
                        nodes, types, node.left, _horizontal_target(left, right)
                    )
                elif right.is_untyped():
                    _adopt(
                        nodes,
                        types,
                        node.right,
                        _horizontal_target(right, left),
                    )
                dtype = horizontal_supertype(
                    kind, types[node.left], types[node.right]
                )
            has_aggregate = aggregated[node.left] or aggregated[node.right]
            if shapes[node.left] == ROWS or shapes[node.right] == ROWS:
                shape = ROWS
            elif has_aggregate:
                shape = AGGREGATE
        elif node.op == HORIZONTAL:
            if node.left < 0 or node.left >= i:
                raise Error("Invalid horizontal expression input")
            _default(nodes, types, node.left)
            var kind = Int(node.integer)
            var inputs = horizontal_inputs(nodes, i)
            for input in inputs:
                horizontal_input(kind, types[input])
            dtype = horizontal_result(kind, types[node.left], len(inputs))
            shape = shapes[node.left]
            has_aggregate = aggregated[node.left]
        elif is_text_function(node.op):
            if node.left < 0 or node.left >= i:
                raise Error("Invalid string expression input")
            if types[node.left] != DataType.STRING:
                raise Error(
                    "str operations require a string expression, found "
                    + types[node.left].name()
                )
            dtype = _text_function_dtype(node)
            shape = shapes[node.left]
            has_aggregate = aggregated[node.left]
        elif node.op == STRUCT_PACK:
            var children = struct_pack_children(node)
            var names = struct_pack_names(node)
            if len(children) == 0 or len(children) != len(names):
                raise Error("Invalid struct expression")
            var field_types = List[DataType]()
            var any_rows = False
            for child in children:
                if child < 0 or child >= i:
                    raise Error("Invalid struct expression input")
                _default(nodes, types, child)
                field_types.append(types[child])
                if shapes[child] == ROWS:
                    any_rows = True
                elif shapes[child] == AGGREGATE:
                    shape = AGGREGATE
                has_aggregate = has_aggregate or aggregated[child]
            if any_rows:
                shape = ROWS
            dtype = DataType.struct(names, field_types)
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
        elif node.op == ARCTAN2:
            if (
                node.left < 0
                or node.right < 0
                or node.left >= i
                or node.right >= i
            ):
                raise Error("Invalid arctan2 inputs")
            _default(nodes, types, node.left)
            _default(nodes, types, node.right)
            var y = types[node.left]
            var x = types[node.right]
            if not _numeric(y) or not _numeric(x):
                raise Error(
                    "arctan2 requires numeric operands, found "
                    + y.name()
                    + " and "
                    + x.name()
                )
            # Polars casts x to y's float type, Float64 for any other y.
            dtype = DataType.FLOAT32 if y == DataType.FLOAT32 else (
                DataType.FLOAT64
            )
            has_aggregate = aggregated[node.left] or aggregated[node.right]
            if shapes[node.left] == ROWS or shapes[node.right] == ROWS:
                shape = ROWS
            elif has_aggregate:
                shape = AGGREGATE
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
                # A string literal compared with binary stands for its
                # UTF-8 bytes, as in Polars; both share one storage layout.
                if is_comparison(node.op):
                    if left.is_binary() and nodes[node.right].op == LIT_STRING:
                        types[node.right] = DataType.BINARY
                    elif (
                        right.is_binary() and nodes[node.left].op == LIT_STRING
                    ):
                        types[node.left] = DataType.BINARY
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
