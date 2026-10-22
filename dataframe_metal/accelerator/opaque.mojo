"""Opaque Binary/Geometry transport to existing ordered GPU UInt32 kernels.

Original schema binding precedes any recoding. All byte decoding is transport;
query comparisons, grouping, sorting and joins execute on resident GPU codes.
"""
from std.memory import ArcPointer
from std.sys import size_of
from dataframe import Column, DataFrame, DataType, Series, StringColumn
from dataframe.binding import bind
from dataframe.string_view import StringView
from dataframe.expr import (
    node_children,
    COL,
    LIT_INT,
    LIT_STRING,
    LIT_NULL,
    WHEN,
    FILL_NULL,
    KEEP_NULLS,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    IS_NULL,
    IS_NOT_NULL,
    MIN,
    MAX,
    FIRST,
    LAST,
    COUNT,
    LEN,
    N_UNIQUE,
    NULL_COUNT,
    ARG_MIN,
    ARG_MAX,
)
from dataframe.lazy import LazyFrame, SCAN_FRAME
from .memory import checked_add, checked_mul
from .metadata import schema_frame
from .selectors import _live_nodes
from dataframe.byte_domain import ByteDomain


@fieldwise_init
struct OpaquePlan(Movable):
    var query: LazyFrame
    var schema: DataFrame
    var dictionary: ArcPointer[ByteDomain]
    var staging_bytes: Int


def has_opaque_columns(query: LazyFrame) -> Bool:
    var live = _live_nodes(query)
    for index in range(len(query._nodes)):
        if not live[index]:
            continue
        ref operation = query._nodes[index]
        if operation.kind == SCAN_FRAME:
            for column in query._frames[operation.offset]._columns:
                if column.dtype()._is_bytes():
                    return True
        for expression in operation.exprs:
            for node in expression._nodes:
                if node.op == LIT_NULL:
                    for dtype in node.dtypes:
                        if dtype and dtype.value()._is_bytes():
                            return True
    return False


def _allowed(op: Int) -> Bool:
    return (
        op == COL
        or op == LIT_STRING
        or op == LIT_NULL
        or op == WHEN
        or op == FILL_NULL
        or op == KEEP_NULLS
        or op == EQ
        or op == NE
        or op == LT
        or op == LE
        or op == GT
        or op == GE
        or op == IS_NULL
        or op == IS_NOT_NULL
        or op == MIN
        or op == MAX
        or op == FIRST
        or op == LAST
        or op == COUNT
        or op == LEN
        or op == N_UNIQUE
        or op == NULL_COUNT
        or op == ARG_MIN
        or op == ARG_MAX
    )


def _literal_values(query: LazyFrame) raises -> List[String]:
    """Check original operand types before erasing them, collecting byte literals."""
    var literals = List[String]()
    for operation in query._nodes:
        for key in operation.window_keys:
            if key.dtype._is_bytes() and not key.string_hash:
                raise Error(
                    "Metal opaque window is missing raw byte hash provenance"
                )
        if not len(operation.exprs):
            continue
        if operation.left < 0:
            raise Error("Metal opaque expression has no input schema")
        var schema = schema_frame(query, operation.left)
        for expression in operation.exprs:
            var bound = bind(expression, schema._columns)
            for index in range(len(bound.expr._nodes)):
                ref node = bound.expr._nodes[index]
                var opaque = bound.dtypes[index]._is_bytes()
                for child in node_children(node):
                    opaque |= child >= 0 and bound.dtypes[child]._is_bytes()
                if opaque and not _allowed(node.op):
                    raise Error(
                        "Metal unsupported [opaque]: byte operation requires a separate implementation"
                    )
                if node.op == LIT_STRING and bound.dtypes[index]._is_bytes():
                    literals.append(node.text)
    return literals^


def _staging_upper(query: LazyFrame, literals: List[String]) raises -> Int:
    """Conservative host domain/lookup/code reservation checked before construction.

    Source duplicates deliberately overestimate unique domain storage. Twelve
    raw-byte copies cover ASCII key copies and transient sorting/decoding;
    Explicit String element capacity plus offsets/hash arrays cover all key
    tables. The domain uses sorted vectors and binary search, not a Dict with
    unspecified capacity/load-factor overhead.
    Encoded source arrays include temporary unpacked validity and bitmaps.
    """
    var payload = 0
    var entries = 0
    var code_bytes = 0
    for frame in query._frames:
        for column in frame._columns:
            if not column.dtype()._is_bytes():
                continue
            var rows = len(column)
            entries = checked_add(entries, rows)
            code_bytes = checked_add(
                code_bytes,
                checked_add(
                    checked_mul(rows, 5),
                    checked_mul(checked_add(rows, 7) // 8, 2),
                ),
            )
            for part in column.chunks():
                ref bytes = part._data[StringColumn]
                for row in range(len(part)):
                    if bytes._valid(row):
                        payload = checked_add(
                            payload, len(bytes._row_bytes(row))
                        )
    for literal in literals:
        payload = checked_add(payload, literal.byte_length())
        entries = checked_add(entries, 1)
    return checked_add(
        checked_add(
            checked_mul(payload, 12),
            checked_mul(entries, 4 * size_of[String]() + 24),
        ),
        checked_add(code_bytes, 8),
    )


def prepare(
    query: LazyFrame,
    *,
    memory_limit_bytes: Int = -1,
    metadata_only: Bool = False,
) raises -> OpaquePlan:
    """Bind original logical metadata, then stage opaque codes once per source."""
    # Compaction is metadata-only and keeps reachable producer order/failures.
    # Bind and stage the SAME compact graph so node/source indices cannot drift.
    var original = query.copy()
    original._reorder(len(original._nodes) - 1)
    original._compact_scan_slots()
    var schema = schema_frame(original, len(original._nodes) - 1).clear()
    var literals = _literal_values(original)
    var staged = original.copy()
    if metadata_only:
        for index in range(len(staged._frames)):
            staged._frames[index] = staged._frames[index].clear()
    var staging = _staging_upper(staged, literals)
    if memory_limit_bytes >= 0 and staging > memory_limit_bytes:
        raise Error("Metal opaque staging exceeds memory budget")
    var sources = List[Series]()
    for frame in staged._frames:
        for column in frame._columns:
            if column.dtype()._is_bytes():
                sources.append(column.copy())
    var dictionary = ByteDomain(sources, literals)
    var key_hashes = ArcPointer(dictionary.hashes.copy())
    for index in range(len(staged._nodes)):
        ref operation = staged._nodes[index]
        for key_index in range(len(operation.window_keys)):
            ref key = operation.window_keys[key_index]
            if key.dtype._is_bytes():
                # Retain the ORIGINAL dtype and byte-hash provenance: never
                # identify opaque codes as CPU categorical-code partitions.
                if not key.string_hash:
                    raise Error(
                        "Metal opaque window is missing raw byte hash provenance"
                    )
                key.values = key_hashes.copy()
        if not len(operation.exprs):
            continue
        var input = schema_frame(original, original._nodes[index].left)
        for expression_index in range(len(operation.exprs)):
            var bound = bind(
                original._nodes[index].exprs[expression_index], input._columns
            )
            ref expression = operation.exprs[expression_index]
            for node_index in range(len(expression._nodes)):
                ref node = expression._nodes[node_index]
                if (
                    node.op == LIT_STRING
                    and bound.dtypes[node_index]._is_bytes()
                ):
                    node.op = LIT_INT
                    node.integer = Int64(dictionary.code(node.text))
                    node.text = "uint32"
                    node.dtypes = [Optional[DataType](DataType.UINT32)]
                elif node.op == LIT_NULL:
                    for type_index in range(len(node.dtypes)):
                        if (
                            node.dtypes[type_index]
                            and node.dtypes[type_index].value()._is_bytes()
                        ):
                            node.dtypes[type_index] = Optional[DataType](
                                DataType.UINT32
                            )
    for index in range(len(staged._frames)):
        var columns = List[Series]()
        for column in staged._frames[index]._columns:
            columns.append(
                dictionary.encode(
                    column
                ) if column.dtype()._is_bytes() else column.copy()
            )
        staged._frames[index] = DataFrame(
            columns^, height=staged._frames[index].height()
        )
    var shared = ArcPointer(dictionary^)
    for index in range(len(staged._nodes)):
        if staged._nodes[index].grouped_keys:
            staged._nodes[index].grouped_byte_domain = shared.copy()
    return OpaquePlan(staged^, schema^, shared^, staging)


def _retained_frame_bytes(frame: DataFrame) raises -> Int:
    var total = 0
    for column in frame._columns:
        var rows = len(column)
        var bitmap = checked_add(rows, 7) // 8
        var dtype = column.dtype()
        if dtype == DataType.STRING or dtype._is_bytes():
            for part in column.chunks():
                ref strings = part._data[StringColumn]
                if strings._is_view_storage():
                    var views = strings._view_storage.value().copy()
                    total = checked_add(
                        total,
                        checked_add(
                            checked_mul(
                                len(views._views[]), size_of[StringView]()
                            ),
                            checked_add(
                                views.total_buffer_bytes(),
                                checked_add(
                                    len(views._bits[]),
                                    checked_mul(
                                        len(views._buffers[]),
                                        size_of[ArcPointer[List[UInt8]]](),
                                    ),
                                ),
                            ),
                        ),
                    )
                else:
                    total = checked_add(
                        total,
                        checked_add(
                            len(strings._bytes[]),
                            checked_add(
                                checked_mul(len(strings._offsets[]), 8),
                                len(strings._bits[]),
                            ),
                        ),
                    )
        elif dtype == DataType.BOOL:
            total = checked_add(total, checked_mul(bitmap, 2))
        elif dtype.is_nested():
            raise Error(
                "Metal opaque restoration requires primitive output fields"
            )
        else:
            var width = (
                dtype.decimal_width() if dtype.is_decimal() else dtype.physical().bit_width()
            )
            total = checked_add(
                total,
                checked_add(checked_mul(rows, max(1, width // 8)), bitmap),
            )
    return total


def restoration_bytes(frame: DataFrame, plan: OpaquePlan) raises -> Int:
    """Retained encoded result plus all simultaneously built typed byte output."""
    if frame.width() != plan.schema.width():
        raise Error("Metal opaque result width differs")
    var total = _retained_frame_bytes(frame)
    for index in range(frame.width()):
        if not plan.schema._columns[index].dtype()._is_bytes():
            continue
        var column = frame._columns[index].rechunk()
        if column.dtype() != DataType.UINT32:
            raise Error("Metal opaque result code dtype differs")
        ref codes = column._data[Column[UInt32]]
        var payload = 0
        for row in range(len(column)):
            if codes._valid(row):
                var code = Int(codes._get(row))
                if code >= len(plan.dictionary[].keys):
                    raise Error(
                        "Metal opaque result code is outside dictionary"
                    )
                payload = checked_add(
                    payload,
                    Int(
                        plan.dictionary[].offsets[code + 1]
                        - plan.dictionary[].offsets[code]
                    ),
                )
        # restore() builds row byte Lists before packing final bytes/offsets.
        # Charge both payload copies, List headers, temporary Bool validity,
        # final offsets and packed validity; primitives stay shared.
        var rows = len(column)
        total = checked_add(
            total,
            checked_add(
                checked_mul(payload, 2),
                checked_add(
                    checked_mul(
                        rows, size_of[List[UInt8]]() + 8 + size_of[Bool]()
                    ),
                    checked_add(8, checked_add(rows, 7) // 8),
                ),
            ),
        )
    return total


def restore(frame: DataFrame, plan: OpaquePlan) raises -> DataFrame:
    var columns = List[Series](capacity=frame.width())
    for index in range(frame.width()):
        var column = (
            plan.dictionary[]
            .restore(
                frame._columns[index], plan.schema._columns[index]
            ) if plan.schema._columns[index]
            .dtype()
            ._is_bytes() else frame._columns[index]
            .copy()
        )
        column._field_metadata = plan.schema._columns[index]._field_metadata
        columns.append(column^)
    return DataFrame(columns^, height=frame.height())
