"""Filters pushed into a table scan.

A table filter reads one column. It is a small tree of filter kinds (a
comparison with a constant, IN, IS NULL, IS NOT NULL, AND and OR of filters on
the same column, and a dynamic filter whose constant an operator sets while the
query runs) that `check_statistics` answers against a column's `BaseStatistics`:
whether the filter keeps every row of a vector, none, or some. A vector answered
"none" is never read, and one answered "every" needs no row test. Each filter
also keeps the predicate part it came from, the expression the scan evaluates on
rows the statistics did not rule out.

Constants are taken into the column's own domain once, when the filter is
made (`table_filters`): the integer a decimal, date or integer column
stores, a float, or a string's bytes. A constant the column cannot hold
exactly gives no filter: the part is evaluated as written.
"""
from std.atomic import Atomic
from std.collections import Optional
from std.memory import ArcPointer, Pointer

from dataframe.binding import bind
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.execution import evaluate
from dataframe.expr import (
    AND,
    COL,
    EQ,
    EQ_MISSING,
    Expr,
    GE,
    GT,
    INT_IS_IN,
    IS_NOT_NULL,
    IS_NULL,
    LE,
    LT,
    NE,
    NE_MISSING,
    OR,
    STR_IS_IN,
    node_children,
    subtree,
)
from dataframe.series import Series
from dataframe.storage.column import Column
from dataframe.storage.statistics import (
    BaseStatistics,
    COMPARE_DISTINCT_FROM,
    COMPARE_EQUAL,
    COMPARE_GREATERTHAN,
    COMPARE_GREATERTHANOREQUALTO,
    COMPARE_LESSTHAN,
    COMPARE_LESSTHANOREQUALTO,
    COMPARE_NOTEQUAL,
    COMPARE_NOT_DISTINCT_FROM,
    FILTER_ALWAYS_FALSE,
    FILTER_ALWAYS_TRUE,
    FILTER_FALSE_OR_NULL,
    FILTER_TRUE_OR_NULL,
    FLOAT_STATS,
    NO_PRUNING_POSSIBLE,
    NUMERIC_STATS,
    STRING_STATS,
    check_float_zonemap,
    check_numeric_zonemap,
    check_string_zonemap,
    constants_cover_range,
    flip_comparison,
)
from dataframe.storage.string import StringColumn

# Single-column filter kinds: constant comparison, membership, null checks,
# conjunctions and runtime-updated boundaries.
comptime CONSTANT_COMPARISON = 0
comptime IN_FILTER = 1
comptime IS_NULL_FILTER = 2
comptime IS_NOT_NULL_FILTER = 3
comptime CONJUNCTION_AND = 4
comptime CONJUNCTION_OR = 5
comptime DYNAMIC_FILTER = 6


@fieldwise_init
struct FilterConstants(Copyable, Movable):
    """Constants in a column's domain: `domain` is NUMERIC_STATS (the integers
    the column stores), FLOAT_STATS or STRING_STATS. `integral` marks an integer
    column, whose whole range a list of values can cover (an IN list naming
    every value in [min, max] keeps every row)."""

    var domain: Int
    var integral: Bool
    var ints: List[Int128]
    var floats: List[Float64]
    var strings: List[List[UInt8]]

    @staticmethod
    def of(domain: Int, integral: Bool) -> Self:
        return Self(
            domain,
            integral,
            List[Int128](),
            List[Float64](),
            List[List[UInt8]](),
        )

    def count(self) -> Int:
        if self.domain == NUMERIC_STATS:
            return len(self.ints)
        if self.domain == FLOAT_STATS:
            return len(self.floats)
        return len(self.strings)

    def append(mut self, other: Self):
        self.ints.extend(other.ints.copy())
        self.floats.extend(other.floats.copy())
        self.strings.extend(other.strings.copy())


def column_domain(dtype: DataType) -> Int:
    """The statistics domain a column's values compare in, or -1 when its
    order is not its storage's (categoricals, booleans, nested types)."""
    if dtype == DataType.STRING or dtype == DataType.BINARY:
        return STRING_STATS
    if dtype.is_float():
        return FLOAT_STATS
    if (
        dtype.is_integer()
        or dtype.is_decimal()
        or dtype.is_date()
        or dtype.is_datetime()
        or dtype.is_duration()
        or dtype.is_time()
    ):
        return NUMERIC_STATS
    return -1


def _physical_integer(value: Series) raises -> Optional[Int128]:
    """Row 0 of a one-row integer-stored series, or None when it is null or
    not stored as an integer."""
    var one = value.rechunk() if value.is_chunked() else value.copy()
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        comptime if D.is_integral():
            if one._data.isa[Column[Scalar[D]]]():
                ref column = one._data[Column[Scalar[D]]]
                if len(column) != 1 or not column._valid(0):
                    return None
                return Int128(column._get(0))
    if one._data.isa[Column[Int128]]():
        ref column = one._data[Column[Int128]]
        if len(column) != 1 or not column._valid(0):
            return None
        return column._get(0)
    return None


def _physical_float(value: Series) raises -> Optional[Float64]:
    var one = value.rechunk() if value.is_chunked() else value.copy()
    if one._data.isa[Column[Float64]]():
        ref column = one._data[Column[Float64]]
        if len(column) != 1 or not column._valid(0):
            return None
        return column._get(0)
    if one._data.isa[Column[Float32]]():
        ref column = one._data[Column[Float32]]
        if len(column) != 1 or not column._valid(0):
            return None
        return Float64(column._get(0))
    return None


def _rescale(value: Int128, from_scale: Int, to_scale: Int) -> Optional[Int128]:
    """`value` at `from_scale` decimal places as an integer at `to_scale`,
    when that is exact and fits."""
    if from_scale == to_scale:
        return value
    if from_scale < to_scale:
        var factor = Int128(1)
        for _ in range(to_scale - from_scale):
            factor *= 10
        if value > Int128.MAX // factor or value < Int128.MIN // factor:
            return None
        return value * factor
    var divisor = Int128(1)
    for _ in range(from_scale - to_scale):
        divisor *= 10
    if value % divisor != 0:
        return None
    return value // divisor


def integer_in_domain(
    column: DataType, value: Int128, value_dtype: DataType
) -> Optional[Int128]:
    """An integer-stored constant of `value_dtype` as the integer `column`
    stores for the same value, when exact. Integers and decimals compare by
    number; temporal values only within the same type."""
    if column.is_integer():
        if value_dtype.is_integer():
            return value
        return None
    if column.is_decimal():
        if value_dtype.is_integer():
            return _rescale(value, 0, column.scale())
        if value_dtype.is_decimal():
            return _rescale(value, value_dtype.scale(), column.scale())
        return None
    if value_dtype == column:
        return value
    return None


def constant_in_domain(
    column: DataType, value: Series
) raises -> Optional[FilterConstants]:
    """A one-row constant in the domain of a `column` typed column, or None
    when the column cannot hold it exactly (the comparison is then
    evaluated as written)."""
    var domain = column_domain(column)
    var out = FilterConstants.of(domain, column.is_integer())
    if domain < 0 or len(value) != 1 or value.null_count() != 0:
        return None
    var dtype = value.dtype()
    if domain == STRING_STATS:
        if dtype != column:
            return None
        var one = value.rechunk() if value.is_chunked() else value.copy()
        if not one._data.isa[StringColumn]():
            return None
        var bytes = List[UInt8]()
        bytes.extend(one._data[StringColumn]._row_bytes(0))
        out.strings.append(bytes^)
        return out^
    if domain == FLOAT_STATS:
        if dtype.is_float():
            var found = _physical_float(value)
            if not found:
                return None
            out.floats.append(found.value())
            return out^
        if dtype.is_integer():
            var found = _physical_integer(value)
            if not found:
                return None
            var as_float = Float64(found.value())
            if Int128(as_float) != found.value():
                return None
            out.floats.append(as_float)
            return out^
        return None
    var found = _physical_integer(value)
    if not found:
        return None
    var converted = integer_in_domain(column, found.value(), dtype)
    if not converted:
        return None
    out.ints.append(converted.value())
    return out^


struct DynamicFilterData(Movable):
    """A comparison whose constant an operator sets and tightens while the query
    runs: a TopN's boundary, a join build's key range. Shared by the scan that
    checks it and the operator that sets it. Until it is set it rules nothing
    out. A dynamic filter is optional: the operator that set it still checks
    every row it receives, so the scan only uses it to skip vectors."""

    var lock: Atomic[Int64]
    var initialized: Atomic[Int64]
    var column: DataType
    var op: Int
    var constants: FilterConstants

    def __init__(out self, column: DataType, op: Int):
        self.lock = Atomic[Int64](0)
        self.initialized = Atomic[Int64](0)
        self.column = column
        self.op = op
        self.constants = FilterConstants.of(column_domain(column), False)


def _dynamic(
    data: ArcPointer[DynamicFilterData],
) -> ref[MutAnyOrigin] DynamicFilterData:
    """The shared filter, reached through its address: setters and checks
    on different workers read and change it only under `lock`."""
    return Pointer[DynamicFilterData, MutAnyOrigin](
        unsafe_from_address=Int(Pointer(to=data[]))
    )[]


def _lock(data: ArcPointer[DynamicFilterData]):
    ref lock = _dynamic(data).lock
    while True:
        var expected = Int64(0)
        if lock.compare_exchange(expected, 1):
            return


def _unlock(data: ArcPointer[DynamicFilterData]):
    _dynamic(data).lock.store(0)


def set_dynamic_value(
    data: ArcPointer[DynamicFilterData], op: Int, value: Series
) raises -> Bool:
    """Set the comparison and constant; false when the column cannot hold the
    value exactly, which leaves the filter as it was."""
    var converted = constant_in_domain(_dynamic(data).column, value)
    if not converted:
        return False
    _lock(data)
    _dynamic(data).op = op
    _dynamic(data).constants = converted.take()
    _dynamic(data).initialized.store(1)
    _unlock(data)
    return True


def _dynamic_check(
    data: ArcPointer[DynamicFilterData], stats: BaseStatistics
) -> Int:
    """What the statistics prove about the dynamic filter's current comparison;
    nothing until an operator has set its constant."""
    if _dynamic(data).initialized.load() == 0:
        return NO_PRUNING_POSSIBLE
    # The constant is read in place under the lock rather than copied out:
    # the scan checks it for every vector it reads, not once a row group.
    _lock(data)
    var op = _dynamic(data).op
    var result: Int
    if not stats.can_have_no_null():
        result = (
            FILTER_ALWAYS_TRUE if op
            == COMPARE_DISTINCT_FROM else FILTER_ALWAYS_FALSE
        )
    else:
        result = _zonemap(stats, op, _dynamic(data).constants)
    _unlock(data)
    return result


@fieldwise_init
struct FilterNode(Copyable, Movable):
    var kind: Int
    var op: Int
    var children: List[Int]
    var constants: FilterConstants
    var dynamic: Optional[ArcPointer[DynamicFilterData]]


@fieldwise_init
struct TableFilter(Copyable, Movable):
    """One column's filter: `nodes` with the root last. `source` is the
    index of the predicate part it came from (-1 for a dynamic filter) and
    `expr` that part. An `optional` filter is only checked against
    statistics: whatever it does not rule out
    is not evaluated by the scan."""

    var column: String
    var nodes: List[FilterNode]
    var source: Int
    var expr: Optional[Expr]
    var optional: Bool


struct TableFilterSet(Copyable, Movable):
    """The filters pushed into one scan (`TableFilterSet`)."""

    var filters: List[TableFilter]

    def __init__(out self):
        self.filters = List[TableFilter]()

    def has_filters(self) -> Bool:
        return len(self.filters) > 0

    def push_filter(mut self, var filter: TableFilter):
        self.filters.append(filter^)

    def push_dynamic(
        mut self, column: String, data: ArcPointer[DynamicFilterData]
    ):
        """Add a dynamic filter on `column` (a TopN boundary, a join's key
        filters): an optional filter the scan checks against each vector's
        statistics."""
        var nodes = List[FilterNode]()
        nodes.append(
            FilterNode(
                DYNAMIC_FILTER,
                0,
                List[Int](),
                FilterConstants.of(-1, False),
                Optional(data),
            )
        )
        self.filters.append(TableFilter(column, nodes^, -1, None, True))


def _zonemap(stats: BaseStatistics, op: Int, constants: FilterConstants) -> Int:
    """The numeric or string zone map, when the statistics are in the constants'
    domain."""
    if stats.stats_type != constants.domain:
        return NO_PRUNING_POSSIBLE
    if constants.domain == NUMERIC_STATS:
        return check_numeric_zonemap(stats, op, constants.ints)
    if constants.domain == FLOAT_STATS:
        return check_float_zonemap(stats, op, constants.floats)
    return check_string_zonemap(stats, op, constants.strings)


def _check_comparison(
    stats: BaseStatistics, op: Int, constants: FilterConstants
) -> Int:
    """What the statistics prove about a comparison with a non-null constant."""
    if not stats.can_have_no_null():
        return (
            FILTER_ALWAYS_TRUE if op
            == COMPARE_DISTINCT_FROM else FILTER_ALWAYS_FALSE
        )
    var result = _zonemap(stats, op, constants)
    if stats.can_have_null():
        if result == FILTER_ALWAYS_FALSE and op != COMPARE_DISTINCT_FROM:
            return result
        return NO_PRUNING_POSSIBLE
    return result


def _check_in(stats: BaseStatistics, constants: FilterConstants) -> Int:
    """What the statistics prove about an IN list."""
    if constants.count() == 0:
        return FILTER_ALWAYS_FALSE
    if not stats.can_have_no_null():
        return FILTER_ALWAYS_FALSE
    var result = _zonemap(stats, COMPARE_EQUAL, constants)
    if (
        result == NO_PRUNING_POSSIBLE
        and constants.domain == NUMERIC_STATS
        and constants.integral
        and constants_cover_range(stats, constants.ints)
    ):
        result = FILTER_ALWAYS_TRUE
    if result == FILTER_ALWAYS_TRUE and stats.can_have_null():
        return FILTER_TRUE_OR_NULL
    return result


def _check_node(
    nodes: List[FilterNode], index: Int, stats: BaseStatistics
) -> Int:
    ref node = nodes[index]
    if node.kind == CONSTANT_COMPARISON:
        return _check_comparison(stats, node.op, node.constants)
    if node.kind == IN_FILTER:
        return _check_in(stats, node.constants)
    if node.kind == IS_NULL_FILTER:
        if not stats.can_have_null():
            return FILTER_ALWAYS_FALSE
        if not stats.can_have_no_null():
            return FILTER_ALWAYS_TRUE
        return NO_PRUNING_POSSIBLE
    if node.kind == IS_NOT_NULL_FILTER:
        if not stats.can_have_no_null():
            return FILTER_ALWAYS_FALSE
        if not stats.can_have_null():
            return FILTER_ALWAYS_TRUE
        return NO_PRUNING_POSSIBLE
    if node.kind == DYNAMIC_FILTER:
        return _dynamic_check(node.dynamic.value(), stats)
    if node.kind == CONJUNCTION_AND:
        # AND: always true only when every child is, always false when any child
        # is.
        var result = FILTER_ALWAYS_TRUE
        for child in node.children:
            var answer = _check_node(nodes, child, stats)
            if answer == FILTER_ALWAYS_FALSE or answer == FILTER_FALSE_OR_NULL:
                return answer
            if answer != result:
                result = NO_PRUNING_POSSIBLE
        return result
    # CONJUNCTION_OR
    var result = FILTER_ALWAYS_FALSE
    for child in node.children:
        var answer = _check_node(nodes, child, stats)
        if answer == NO_PRUNING_POSSIBLE:
            return NO_PRUNING_POSSIBLE
        if answer == FILTER_ALWAYS_TRUE:
            return FILTER_ALWAYS_TRUE
        if answer == FILTER_TRUE_OR_NULL:
            result = FILTER_TRUE_OR_NULL
        elif answer == FILTER_FALSE_OR_NULL and result == FILTER_ALWAYS_FALSE:
            result = FILTER_FALSE_OR_NULL
    return result


def check_statistics(filter: TableFilter, stats: BaseStatistics) -> Int:
    """Whether the filter keeps every row the statistics describe, none, or some
    (one of the FILTER_* / NO_PRUNING_POSSIBLE answers)."""
    return _check_node(filter.nodes, len(filter.nodes) - 1, stats)


# --- making filters from predicate parts -----------------------------------


def _comparison(op: Int) -> Int:
    if op == EQ:
        return COMPARE_EQUAL
    if op == NE:
        return COMPARE_NOTEQUAL
    if op == LT:
        return COMPARE_LESSTHAN
    if op == LE:
        return COMPARE_LESSTHANOREQUALTO
    if op == GT:
        return COMPARE_GREATERTHAN
    if op == GE:
        return COMPARE_GREATERTHANOREQUALTO
    if op == EQ_MISSING:
        return COMPARE_NOT_DISTINCT_FROM
    if op == NE_MISSING:
        return COMPARE_DISTINCT_FROM
    return -1


def _is_constant(expr: Expr, index: Int) -> Bool:
    """Whether the subtree at `index` reads no column."""
    var stack: List[Int] = [index]
    while len(stack) > 0:
        var i = stack.pop()
        ref node = expr._nodes[i]
        if node.op == COL:
            return False
        for child in node_children(node):
            stack.append(child)
    return True


def _column_dtype(columns: List[Series], name: String) -> Optional[DataType]:
    for column in columns:
        if column.name() == name:
            return column.dtype()
    return None


struct _Builder(Movable):
    """A filter tree under construction over one column."""

    var column: String
    var nodes: List[FilterNode]

    def __init__(out self):
        self.column = String()
        self.nodes = List[FilterNode]()

    def on(mut self, name: String) -> Bool:
        """Claim `name` as the filter's column; false for a second column."""
        if self.column == "":
            self.column = name
            return True
        return self.column == name

    def add(mut self, var node: FilterNode) -> Int:
        self.nodes.append(node^)
        return len(self.nodes) - 1


def _leaf(
    kind: Int, op: Int, var constants: FilterConstants, children: List[Int]
) -> FilterNode:
    return FilterNode(kind, op, children.copy(), constants^, None)


def _parse_ints(text: String) raises -> List[Int128]:
    var out = List[Int128]()
    for piece in text.split(","):
        out.append(Int128(Int(String(piece))))
    return out^


def _parse_strings(text: String) raises -> List[List[UInt8]]:
    """STR_IS_IN's values: `<byte length>:<bytes>` repeated."""
    var out = List[List[UInt8]]()
    var bytes = text.as_bytes()
    var at = 0
    while at < len(bytes):
        var length = 0
        while bytes[at] != UInt8(ord(":")):
            length = length * 10 + Int(bytes[at] - UInt8(ord("0")))
            at += 1
        at += 1
        var value = List[UInt8](capacity=length)
        value.extend(bytes[at : at + length])
        out.append(value^)
        at += length
    return out^


def _build(
    expr: Expr, index: Int, columns: List[Series], mut builder: _Builder
) raises -> Int:
    """The filter node for the subtree at `index`, or -1 when it is not a
    filter on one column."""
    ref node = expr._nodes[index]
    if node.op == AND or node.op == OR:
        if node.left < 0 or node.right < 0:
            return -1
        var left = _build(expr, node.left, columns, builder)
        if left < 0:
            return -1
        var right = _build(expr, node.right, columns, builder)
        if right < 0:
            return -1
        var kind = CONJUNCTION_AND if node.op == AND else CONJUNCTION_OR
        return builder.add(
            _leaf(kind, 0, FilterConstants.of(-1, False), [left, right])
        )
    if node.op == IS_NULL or node.op == IS_NOT_NULL:
        if node.left < 0 or expr._nodes[node.left].op != COL:
            return -1
        var name = expr._nodes[node.left].text
        if not _column_dtype(columns, name) or not builder.on(name):
            return -1
        var kind = IS_NULL_FILTER if node.op == IS_NULL else IS_NOT_NULL_FILTER
        return builder.add(
            _leaf(kind, 0, FilterConstants.of(-1, False), List[Int]())
        )
    if node.op == INT_IS_IN or node.op == STR_IS_IN:
        if node.left < 0 or expr._nodes[node.left].op != COL:
            return -1
        var name = expr._nodes[node.left].text
        var dtype = _column_dtype(columns, name)
        if not dtype or not builder.on(name):
            return -1
        var domain = column_domain(dtype.value())
        var constants = FilterConstants.of(domain, dtype.value().is_integer())
        if node.op == STR_IS_IN:
            if domain != STRING_STATS:
                return -1
            constants.strings = _parse_strings(node.text)
        else:
            if domain != NUMERIC_STATS or len(node.dtypes) == 0:
                return -1
            if not node.dtypes[0]:
                return -1
            var listed = node.dtypes[0].value()
            for value in _parse_ints(node.text):
                var converted = integer_in_domain(dtype.value(), value, listed)
                if not converted:
                    return -1
                constants.ints.append(converted.value())
        return builder.add(_leaf(IN_FILTER, 0, constants^, List[Int]()))
    var op = _comparison(node.op)
    if op < 0 or node.left < 0 or node.right < 0:
        return -1
    var column: Int
    var constant: Int
    if expr._nodes[node.left].op == COL and _is_constant(expr, node.right):
        column = node.left
        constant = node.right
    elif expr._nodes[node.right].op == COL and _is_constant(expr, node.left):
        column = node.right
        constant = node.left
        op = flip_comparison(op)
    else:
        return -1
    var name = expr._nodes[column].text
    var dtype = _column_dtype(columns, name)
    if not dtype or not builder.on(name):
        return -1
    var none = List[Series]()
    var value = evaluate(bind(subtree(expr, constant), none), none, 1)
    var converted = constant_in_domain(dtype.value(), value)
    if not converted:
        return -1
    return builder.add(
        _leaf(CONSTANT_COMPARISON, op, converted.take(), List[Int]())
    )


def table_filters(parts: List[Expr], columns: List[Series]) -> TableFilterSet:
    """The parts of a conjunction that are filters on one column of `columns`,
    as table filters, which statistics can answer before rows are read; other
    parts stay with the expression evaluated over the scan's rows."""
    var set = TableFilterSet()
    for p in range(len(parts)):
        var builder = _Builder()
        var root: Int
        try:
            root = _build(parts[p], len(parts[p]._nodes) - 1, columns, builder)
        except:
            # A constant that does not evaluate on its own: the part is
            # left to the evaluated expression, which reports any error.
            root = -1
        if root < 0 or root != len(builder.nodes) - 1:
            continue
        set.push_filter(
            TableFilter(
                builder.column.copy(),
                builder.nodes.copy(),
                p,
                Optional(parts[p].copy()),
                False,
            )
        )
    return set^


def range_filter(
    column: String, dtype: DataType, low: Int128, high: Int128
) -> Optional[TableFilter]:
    """An optional filter `low <= column <= high` over an integer-domain
    column: statistics only, the operator that made it checks its rows."""
    if column_domain(dtype) != NUMERIC_STATS:
        return None
    var lower = FilterConstants.of(NUMERIC_STATS, dtype.is_integer())
    lower.ints.append(low)
    var upper = FilterConstants.of(NUMERIC_STATS, dtype.is_integer())
    upper.ints.append(high)
    var nodes = List[FilterNode]()
    nodes.append(
        _leaf(
            CONSTANT_COMPARISON,
            COMPARE_GREATERTHANOREQUALTO,
            lower^,
            List[Int](),
        )
    )
    nodes.append(
        _leaf(
            CONSTANT_COMPARISON, COMPARE_LESSTHANOREQUALTO, upper^, List[Int]()
        )
    )
    nodes.append(
        _leaf(CONJUNCTION_AND, 0, FilterConstants.of(-1, False), [0, 1])
    )
    return TableFilter(column, nodes^, -1, None, True)
