"""Precision-only Date.YEAR plus narrow Decimal row producers.

Only a direct Date column's YEAR extraction is added to the existing narrow
operand proof. Both CPU and GPU use overflow-safe civil_from_days for the full
Int64 Date domain, return Int64 and propagate validity without an error site.
Original input graphs and every reachable downstream consumer must be
exception-infallible. Thus complete producer observations may commute with
streaming batches/workers; no CPU geometry or error ordering is invented.
"""
from dataframe.binding import BoundExpr, bind, ROWS
from dataframe.dtype import DataType
from dataframe.lazy import (
    LazyFrame,
    SELECT,
    WITH_COLUMNS,
    FILTER,
    SORT,
    DROP,
    JOIN,
    AGG,
)
from dataframe.expr import (
    COL,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_NULL,
    LIT_STRING,
    DT_YEAR,
    CAST,
    ADD,
    SUB,
    MUL,
    DIV,
    SUM,
    MEAN,
    COUNT,
    LEN,
    NULL_COUNT,
    WHEN,
    EQ,
    NE,
    GT,
    GE,
    LT,
    LE,
    AND,
    OR,
    XOR,
    NOT,
    IS_NULL,
    IS_NOT_NULL,
    IS_NAN,
    IS_NOT_NAN,
    IS_FINITE,
    IS_INFINITE,
)
from . import decimal_group_feed
from .decimal_prefix import prefix_infallible, _ordinary_key
from .integer_ranges import proven_integer_nodes
from .metadata import schema_frame
from .selectors import _live_nodes

comptime MATERIALIZATION = "metal_narrow_decimal_date_rows"
comptime ORDERED_MODE = 8


def date_year(bound: BoundExpr) -> Bool:
    if bound.shape() != ROWS or len(bound.expr._nodes) != 2:
        return False
    var source = bound.expr._nodes[0].copy()
    var root = bound.expr._nodes[1].copy()
    return (
        source.op == COL
        and source.left < 0
        and source.right < 0
        and source.extra < 0
        and bound.dtypes[0] == DataType.DATE
        and root.op == DT_YEAR
        and root.left == 0
        and root.right < 0
        and root.extra < 0
        and bound.dtypes[1] == DataType.INT64
    )


def eligible(expressions: List[BoundExpr]) -> Bool:
    var year = False
    var checked = False
    for bound in expressions:
        if date_year(bound):
            year = True
        else:
            var proof = decimal_group_feed.operand(bound)
            if not proof[0]:
                return False
            checked |= proof[1]
    return year and checked


def _syntax(op: Int) -> Bool:
    return op in [
        COL,
        LIT_INT,
        LIT_FLOAT,
        LIT_BOOL,
        LIT_NULL,
        LIT_STRING,
        CAST,
        ADD,
        SUB,
        MUL,
        DIV,
        SUM,
        MEAN,
        COUNT,
        LEN,
        NULL_COUNT,
        WHEN,
        EQ,
        NE,
        GT,
        GE,
        LT,
        LE,
        AND,
        OR,
        XOR,
        NOT,
        IS_NULL,
        IS_NOT_NULL,
        IS_NAN,
        IS_NOT_NAN,
        IS_FINITE,
        IS_INFINITE,
    ]


def _infallible(bound: BoundExpr) -> Bool:
    var proven = proven_integer_nodes(bound)
    for at in range(len(bound.expr._nodes)):
        ref node = bound.expr._nodes[at]
        var dtype = bound.dtypes[at]
        if node.op in [COL, LIT_INT, LIT_FLOAT, LIT_BOOL, LIT_NULL, LIT_STRING]:
            continue
        if node.op == CAST:
            if not proven[at]:
                return False
        elif node.op in [ADD, SUB, MUL, DIV]:
            if (
                not (
                    dtype == DataType.FLOAT64
                    and bound.dtypes[node.left] == DataType.FLOAT64
                    and bound.dtypes[node.right] == DataType.FLOAT64
                )
                and not proven[at]
            ):
                return False
        elif node.op in [SUM, MEAN]:
            if node.left < 0:
                return False
            var input = bound.dtypes[node.left]
            if (
                input != DataType.FLOAT64
                and not decimal_group_feed.infallible_sum(
                    input, Int(UInt32.MAX)
                )
            ):
                return False
        elif node.op in [COUNT, LEN, NULL_COUNT]:
            # The resident row bound is UInt32; these return exact Int64.
            if dtype != DataType.INT64:
                return False
        elif node.op == WHEN:
            if (
                node.extra < 0
                or bound.dtypes[node.left] != DataType.BOOL
                or bound.dtypes[node.right] != dtype
                or bound.dtypes[node.extra] != dtype
            ):
                return False
        elif node.op in [EQ, NE, GT, GE, LT, LE]:
            if (
                bound.dtypes[node.left].is_decimal()
                or bound.dtypes[node.right].is_decimal()
            ) and not proven[at]:
                return False
        elif node.op not in [
            AND,
            OR,
            XOR,
            NOT,
            IS_NULL,
            IS_NOT_NULL,
            IS_NAN,
            IS_NOT_NAN,
            IS_FINITE,
            IS_INFINITE,
        ]:
            return False
    return True


def consumers_infallible(query: LazyFrame, producer: Int) raises -> Bool:
    """Prove the original live continuation, including other JOIN arms.

    No head/SLICE, deferred control, window or unknown state can short-circuit
    or precede a late producer precision failure. This is a whole-graph proof,
    not inference from successful zero-row metadata or a normalized scan.
    """
    if producer < 0 or producer >= len(query._nodes):
        return False
    var live = _live_nodes(query)
    var optimize = (
        not query._accelerator_settings
        or query._accelerator_settings.value().optimize
    )
    var streaming = (
        not query._accelerator_settings
        or query._accelerator_settings.value().streaming
    )
    var affected = List[Bool](length=len(query._nodes), fill=False)
    affected[producer] = True
    var grouped = 0
    var projections = 0
    var group_reads = List[String]()
    for index in range(producer + 1, len(query._nodes)):
        ref node = query._nodes[index]
        if node.left >= index or node.right >= index:
            return False
        if not live[index] or not (
            node.left >= 0
            and affected[node.left]
            or node.right >= 0
            and affected[node.right]
        ):
            continue
        affected[index] = True
        # CPU streaming JOIN key ranges can suppress unmatched source rows
        # before WITH_COLUMNS even with optimize=False. Their late precision
        # sites are not necessarily observed, so a complete ROW preflight is
        # not justified for this continuation.
        if streaming and node.kind == JOIN:
            return False
        if optimize:
            # CPU predicate/key-set pushdown and dead-expression pruning can
            # change which producer rows/occurrences are visited. Certify only
            # a single whole GROUP, optional infallible composition projection
            # and ordering, with every original output still consumed. The
            # remaining unoptimized graphs have the proof below.
            if node.kind != AGG and node.kind != SORT and node.kind != SELECT:
                return False
            if node.kind == AGG:
                grouped += 1
                if grouped > 1:
                    return False
                group_reads.extend(node.names.copy())
                for expression in node.exprs:
                    for part in expression._nodes:
                        if part.op == COL:
                            group_reads.append(part.text)
            if node.kind == SELECT:
                projections += 1
                if not grouped or projections > 1:
                    return False
                for expression in node.exprs:
                    for part in expression._nodes:
                        if part.op in [SUM, MEAN, COUNT, LEN, NULL_COUNT]:
                            return False
        if (
            node.kind
            not in [SELECT, WITH_COLUMNS, FILTER, SORT, DROP, JOIN, AGG]
            or node.source_failure
            or node.producer_origin
            or node.producer_region
            or node._executions
        ):
            return False
        # Syntax is checked before schema binding can execute control metadata.
        for expression in node.exprs:
            for part in expression._nodes:
                if not _syntax(part.op):
                    return False
    if optimize and grouped:
        for expression in query._nodes[producer].exprs:
            if expression._name not in group_reads:
                return False
    for index in range(producer + 1, len(query._nodes)):
        if not affected[index]:
            continue
        ref node = query._nodes[index]
        if node.kind == JOIN:
            for child in [node.left, node.right]:
                if child < 0 or child >= index:
                    return False
                if not affected[child] and not prefix_infallible(query, child):
                    return False
            for key in node.names:
                if not _ordinary_key(query, node.left, key):
                    return False
            var right_keys = node.right_keys.copy() if len(
                node.right_keys
            ) else node.names.copy()
            for key in right_keys:
                if not _ordinary_key(query, node.right, key):
                    return False
        if node.kind == SORT or node.kind == AGG:
            for key in node.names:
                if not _ordinary_key(query, node.left, key):
                    return False
        if len(node.exprs):
            var schema = schema_frame(query, node.left)
            if optimize and node.kind == SELECT:
                var reads = List[String]()
                for expression in node.exprs:
                    for part in expression._nodes:
                        if part.op == COL:
                            reads.append(part.text)
                # Aggregate composition preserves all original leaves and
                # grouping keys. A projection dropping an unused state could
                # let CPU pruning suppress an original precision occurrence.
                for name in schema.columns():
                    if name not in reads:
                        return False
            for expression in node.exprs:
                var bound = bind(expression, schema._columns)
                if optimize and node.kind == SELECT and bound.shape() != ROWS:
                    return False
                if not _infallible(bound):
                    return False
    return affected[len(query._nodes) - 1]
