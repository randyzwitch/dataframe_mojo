"""Infallible original input graphs for commuting Decimal producer faults.

A successful GPU input-zero observation is necessary but cannot establish CPU
streaming phase order. This proof walks both original JOIN arms and excludes
all competing value/control/state faults before the producer is certified.
"""
from dataframe import DataType
from dataframe.binding import BoundExpr, bind
from dataframe.lazy import (
    LazyFrame,
    SCAN_FRAME,
    SELECT,
    WITH_COLUMNS,
    FILTER,
    SORT,
    SLICE,
    DROP,
    JOIN,
)
from dataframe.expr import (
    Expr,
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
    NEG,
    ABS,
    FLOORDIV,
    MOD,
    POW,
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
    WHEN,
    IS_NULL,
    IS_NOT_NULL,
    IS_NAN,
    IS_NOT_NAN,
    IS_FINITE,
    IS_INFINITE,
    INT_IS_IN,
)
from .integer_ranges import proven_integer_nodes
from .metadata import schema_frame


def _row_syntax(expression: Expr) -> Bool:
    for node in expression._nodes:
        if node.op not in [
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
            NEG,
            ABS,
            FLOORDIV,
            MOD,
            POW,
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
            WHEN,
            IS_NULL,
            IS_NOT_NULL,
            IS_NAN,
            IS_NOT_NAN,
            IS_FINITE,
            IS_INFINITE,
            INT_IS_IN,
        ]:
            return False
    return True


def _row_infallible(bound: BoundExpr) -> Bool:
    var proven = proven_integer_nodes(bound)
    for at in range(len(bound.expr._nodes)):
        ref node = bound.expr._nodes[at]
        if node.op in [COL, LIT_INT, LIT_FLOAT, LIT_BOOL, LIT_NULL, LIT_STRING]:
            continue
        if node.op == CAST:
            if not proven[at]:
                return False
        elif node.op in [ADD, SUB, MUL, NEG, ABS, FLOORDIV, MOD, POW]:
            if not proven[at]:
                return False
        elif node.op in [EQ, NE, GT, GE, LT, LE]:
            if (
                bound.dtypes[node.left].is_decimal()
                or bound.dtypes[node.right].is_decimal()
            ) and not proven[at]:
                return False
        elif node.op == WHEN:
            if (
                node.extra < 0
                or bound.dtypes[node.left] != DataType.BOOL
                or bound.dtypes[node.right] != bound.dtypes[at]
                or bound.dtypes[node.extra] != bound.dtypes[at]
            ):
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
            INT_IS_IN,
        ]:
            return False
    return True


def _ordinary_key(query: LazyFrame, source: Int, name: String) raises -> Bool:
    var dtype = schema_frame(query, source).column(name).dtype()
    for field in query._nodes[source].text_fields:
        if field.name == name:
            dtype = field.dtype
    return not (
        dtype.is_categorical() or dtype._is_bytes() or dtype.is_nested()
    )


def prefix_infallible(query: LazyFrame, root: Int) raises -> Bool:
    """Inspect the original immutable DAG; never inspect input row values.

    Ordinary coded String comparisons are decoded-label comparisons; existing
    text certificates remain authoritative. Original CAT decoding can fail
    on a raw out-of-range code and is excluded. State/window/control producers
    and unproved arithmetic/casts are excluded even when currently unused.
    """
    if root < 0 or root >= len(query._nodes):
        return False
    var live = List[Bool](length=len(query._nodes), fill=False)
    var pending = List[Int]()
    pending.append(root)
    while len(pending):
        var index = pending.pop()
        if live[index]:
            continue
        live[index] = True
        ref node = query._nodes[index]
        # Syntax first: a deferred control producer must not be schema-executed
        # merely to determine that it cannot prove an infallible prefix.
        if (
            node.kind
            not in [
                SCAN_FRAME,
                SELECT,
                WITH_COLUMNS,
                FILTER,
                SORT,
                SLICE,
                DROP,
                JOIN,
            ]
            or node.source_failure
            or node.producer_origin
            or node.producer_region
            or node._executions
        ):
            return False
        for expression in node.exprs:
            if not _row_syntax(expression):
                return False
        for field in node.text_fields:
            if field.dtype.is_categorical():
                return False
        for child in [node.left, node.right]:
            if child < 0:
                continue
            if child >= index:
                return False
            pending.append(child)
    for index in range(root + 1):
        if not live[index]:
            continue
        ref node = query._nodes[index]
        if node.kind == SCAN_FRAME:
            if node.offset < 0 or node.offset >= len(query._frames):
                return False
            for column in query._frames[node.offset]._columns:
                if len(node.names) and column.name() not in node.names:
                    continue
                var dtype = column.dtype()
                if (
                    dtype.is_categorical()
                    or dtype._is_bytes()
                    or dtype.is_nested()
                ):
                    return False
            continue
        if node.kind == JOIN:
            for key in node.names:
                if not _ordinary_key(query, node.left, key):
                    return False
            var right_keys = node.right_keys.copy() if len(
                node.right_keys
            ) else node.names.copy()
            for key in right_keys:
                if not _ordinary_key(query, node.right, key):
                    return False
            continue
        if node.kind == SORT:
            for key in node.names:
                if not _ordinary_key(query, node.left, key):
                    return False
        if len(node.exprs):
            var schema = schema_frame(query, node.left)
            for expression in node.exprs:
                if not _row_infallible(bind(expression, schema._columns)):
                    return False
    return True
