"""Bounded eager execution; expression nodes dispatch once per column batch.

Reductions are separate passes, not per-row callbacks or per-group dataframes.
The reference reduction schedule is serial; Float64 results may be reassociated
by future parallel implementations. Int64 sums use exact wide states and check final overflow.
"""
from .dtype import DataType, NUMERIC_DTYPES
from .expr import (
    Node,
    COL,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_STRING,
    ADD,
    SUB,
    MUL,
    GT,
    EQ,
    SUM,
    MEAN,
    COUNT,
    MIN,
    MAX,
    LT,
    GE,
    LE,
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
    WHEN,
    STR_CONCAT,
    CAST,
    OVER,
    is_dt_op,
    SEP,
    subtree,
    is_window,
    is_reduction,
    is_string_op,
    is_nested_op,
    IMPLODE,
    STRUCT_PACK,
    struct_pack_children,
    struct_pack_names,
)
from .str_kernels import string_op, concat_strings
from .list_kernels import nested_op
from .nested_column import ListColumn, StructColumn
from .cast import cast_series
from .binding import BoundExpr, bind, ROWS, AGGREGATE, SCALAR
from .hashing import encode_rows
from .window import window_op
from .fusion import fused
from .temporal_kernels import dt_op, temporal_binary
from .bool_column import BoolColumn
from .column import Column, _count_valid
from .string_column import StringColumn, StringBuilder
from .series import Series
from .expr_kernels import binary, unary, choose, fit_mask
from .aggregate import Reducer
from .parallel import Job, partitions, run_jobs, worker_count
from std.memory import ArcPointer
from std.math import isnan


# Rows below which a fused expression copies chunked Float64 sources into
# one contiguous buffer first. Swept on 2026-09-25 with the overall benchmark
# (32 threads, best of 7, two rounds, arithmetic_chain / nullable_compare):
# the copy won at 100k rows (0.68 vs 0.89 ms, 0.42 vs 0.56 ms) and chunk
# windows won from 250k up (1.24 vs 1.39 ms, 0.69 vs 1.09 ms), widening to
# 3-4x at 1M. The old 2,000,000 was chosen without a sweep and cost 1M-row
# expressions 3x.
comptime FUSED_CONTIGUOUS_ROWS = 200_000


def _numeric_literal(node: Node) raises -> Series:
    """A one-row column for an Int/Float literal of its tagged type."""
    if node.text == "":
        if node.op == LIT_INT:
            return Series("", Column[Int64]([node.integer]))
        return Series("", Column[Float64]([node.floating]))
    var dtype = DataType.parse(node.text)
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if dtype == DataType.of(D):
            comptime if D.is_floating_point():
                return Series("", Column[Scalar[D]]([node.floating.cast[D]()]))
            else:
                return Series("", Column[Scalar[D]]([node.integer.cast[D]()]))
    raise Error("Unsupported literal dtype " + node.text)


def _empty(dtype: DataType) raises -> Series:
    return Series.full_null("", dtype, 0)


def _binary_op[
    width: Int
](op: Int, left: Series, right: Series, mask: List[Bool]) raises -> Series:
    """Map a runtime opcode to its compile-time specialized kernel."""
    if op == ADD:
        return binary[ADD, width](left, right, mask)
    if op == SUB:
        return binary[SUB, width](left, right, mask)
    if op == MUL:
        return binary[MUL, width](left, right, mask)
    if op == DIV:
        return binary[DIV, width](left, right, mask)
    if op == FLOORDIV:
        return binary[FLOORDIV, width](left, right, mask)
    if op == MOD:
        return binary[MOD, width](left, right, mask)
    if op == POW:
        return binary[POW, width](left, right, mask)
    if op == CLIP_LOW:
        return binary[CLIP_LOW, width](left, right, mask)
    if op == CLIP_HIGH:
        return binary[CLIP_HIGH, width](left, right, mask)
    if op == GT:
        return binary[GT, width](left, right, mask)
    if op == LT:
        return binary[LT, width](left, right, mask)
    if op == GE:
        return binary[GE, width](left, right, mask)
    if op == LE:
        return binary[LE, width](left, right, mask)
    if op == EQ:
        return binary[EQ, width](left, right, mask)
    if op == NE:
        return binary[NE, width](left, right, mask)
    if op == AND:
        return binary[AND, width](left, right, mask)
    if op == OR:
        return binary[OR, width](left, right, mask)
    if op == XOR:
        return binary[XOR, width](left, right, mask)
    if op == FILL_NULL:
        return binary[FILL_NULL, width](left, right, mask)
    if op == FILL_NAN:
        return binary[FILL_NAN, width](left, right, mask)
    if op == KEEP_NULLS:
        return binary[KEEP_NULLS, width](left, right, mask)
    raise Error("Unsupported binary expression node")


def _unary_op[
    width: Int
](op: Int, input: Series, integer: Int64, mask: List[Bool]) raises -> Series:
    if op == NEG:
        return unary[NEG, width](input, integer, mask)
    if op == ABS:
        return unary[ABS, width](input, integer, mask)
    if op == SQRT:
        return unary[SQRT, width](input, integer, mask)
    if op == EXP:
        return unary[EXP, width](input, integer, mask)
    if op == LOG:
        return unary[LOG, width](input, integer, mask)
    if op == FLOOR:
        return unary[FLOOR, width](input, integer, mask)
    if op == CEIL:
        return unary[CEIL, width](input, integer, mask)
    if op == ROUND:
        return unary[ROUND, width](input, integer, mask)
    if op == NOT:
        return unary[NOT, width](input, integer, mask)
    if op == IS_NULL:
        return unary[IS_NULL, width](input, integer, mask)
    if op == IS_NOT_NULL:
        return unary[IS_NOT_NULL, width](input, integer, mask)
    if op == IS_NAN:
        return unary[IS_NAN, width](input, integer, mask)
    if op == IS_NOT_NAN:
        return unary[IS_NOT_NAN, width](input, integer, mask)
    if op == IS_FINITE:
        return unary[IS_FINITE, width](input, integer, mask)
    if op == IS_INFINITE:
        return unary[IS_INFINITE, width](input, integer, mask)
    raise Error("Unsupported unary expression node")


def _eval[
    width: Int
](
    bound: BoundExpr,
    columns: List[Series],
    aggregates: List[Series],
    index: Int,
    offset: Int,
    length: Int,
    grouped: Bool,
    mask: List[Bool],
) raises -> Series:
    """Evaluate one node for rows [offset, offset + length).

    Returns `length` values, or one value for scalar results that callers
    broadcast. Aggregate nodes read precomputed states instead of recursing.
    `mask` (empty means all rows) marks rows whose result can be observed;
    conditional branches narrow it so unselected rows never raise.
    """
    ref node = bound.expr._nodes[index]
    if node.op == COL:
        return columns[bound.sources[index]].slice(offset, length)
    if node.op == LIT_INT or node.op == LIT_FLOAT:
        return _numeric_literal(node)
    if node.op == LIT_BOOL:
        return Series("", BoolColumn([Bool(node.integer)]))
    if node.op == LIT_STRING:
        return Series("", StringColumn([node.text]))
    if node.op == LIT_NULL:
        return Series.full_null("", DataType.parse(node.text), 1)
    if is_window(node.op) or node.op == OVER:
        return aggregates[index].slice(offset, length)
    if is_reduction(node.op):
        if grouped:
            return aggregates[index].slice(offset, length)
        return aggregates[index].copy()
    if node.op == WHEN:
        return _conditional[width](
            bound, columns, aggregates, index, offset, length, grouped, mask
        )
    if node.op == STRUCT_PACK:
        return _pack_struct[width](
            bound, columns, aggregates, index, offset, length, grouped, mask
        )
    if bound.fusible[index] and node.left >= 0:
        return fused[width](bound, columns, index, offset, length)
    var left = _eval[width](
        bound, columns, aggregates, node.left, offset, length, grouped, mask
    )
    if is_dt_op(node.op):
        return dt_op(node, left, bound.dtypes[node.left])
    if node.op == CAST:
        # Kernel outputs carry physical tags; restore the bound logical type.
        if bound.dtypes[node.left] != left.dtype():
            left = left.with_dtype(bound.dtypes[node.left])
        var observed = fit_mask(mask, len(left))
        # A scalar input is observed if any row is; its offset is not a row.
        return cast_series(
            left,
            DataType.parse(node.text),
            node.integer == 1,
            offset if len(left) == length else 0,
            observed.copy() if len(observed) == len(left) else List[Bool](),
        )
    if is_string_op(node.op):
        return string_op(node, left)
    if is_nested_op(node.op):
        return nested_op(node, left, bound.dtypes[node.left])
    if node.right < 0:
        return _unary_op[width](node.op, left, node.integer, mask)
    var right = _eval[width](
        bound, columns, aggregates, node.right, offset, length, grouped, mask
    )
    if node.op == STR_CONCAT:
        return concat_strings(left, right, node.text)
    var left_type = bound.dtypes[node.left]
    var right_type = bound.dtypes[node.right]
    if (left_type.is_temporal() or right_type.is_temporal()) and (
        node.op == ADD
        or node.op == SUB
        or node.op == MUL
        or node.op == FLOORDIV
    ):
        return temporal_binary(
            node.op, left, right, left_type, right_type, bound.dtypes[index]
        )
    return _binary_op[width](node.op, left, right, mask)


def _pack_struct[
    width: Int
](
    bound: BoundExpr,
    columns: List[Series],
    aggregates: List[Series],
    index: Int,
    offset: Int,
    length: Int,
    grouped: Bool,
    mask: List[Bool],
) raises -> Series:
    """Evaluate every field of a STRUCT_PACK node and pack them; scalar
    fields broadcast to the batch."""
    ref node = bound.expr._nodes[index]
    var shape = bound.shapes[index]
    var size = (
        length if shape == ROWS or (grouped and shape == AGGREGATE) else 1
    )
    var names = struct_pack_names(node)
    var children = struct_pack_children(node)
    var fields = List[Series](capacity=len(children))
    for k in range(len(children)):
        var value = _eval[width](
            bound,
            columns,
            aggregates,
            children[k],
            offset,
            length,
            grouped,
            mask,
        )
        if len(value) != size:
            if len(value) != 1:
                raise Error("struct field length mismatch")
            value = value._broadcast(size)
        # Kernel outputs carry physical tags; restore the bound logical type.
        if value.dtype() != bound.dtypes[children[k]]:
            value = value.with_dtype(bound.dtypes[children[k]])
        fields.append(value.renamed(names[k]))
    return Series("", StructColumn(fields^))


def _implode[
    width: Int
](
    bound: BoundExpr,
    columns: List[Series],
    states: List[Series],
    node: Node,
    height: Int,
    batch_size: Int,
    grouped: Bool,
    groups: List[Int],
    group_count: Int,
) raises -> Series:
    """One list per group (or one list of every row): the input rows in
    input order, nulls included."""
    var input = _full[width](
        bound, columns, states, node.left, height, batch_size, False
    )
    if input.dtype() != bound.dtypes[node.left]:
        input = input.with_dtype(bound.dtypes[node.left])
    if not grouped:
        var offsets = List[Int64](capacity=2)
        offsets.append(0)
        offsets.append(Int64(height))
        return Series("", ListColumn(offsets^, input.renamed("item")))
    var counts = List[Int](length=group_count, fill=0)
    for g in groups:
        counts[g] += 1
    var offsets = List[Int64](capacity=group_count + 1)
    var starts = List[Int](capacity=group_count)
    var total = 0
    offsets.append(0)
    for g in range(group_count):
        starts.append(total)
        total += counts[g]
        offsets.append(Int64(total))
    var order = List[Int](length=height, fill=0)
    for row in range(height):
        var g = groups[row]
        order[starts[g]] = row
        starts[g] += 1
    return Series("", ListColumn(offsets^, input.take(order).renamed("item")))


def _conditional[
    width: Int
](
    bound: BoundExpr,
    columns: List[Series],
    aggregates: List[Series],
    index: Int,
    offset: Int,
    length: Int,
    grouped: Bool,
    mask: List[Bool],
) raises -> Series:
    ref node = bound.expr._nodes[index]
    var shape = bound.shapes[index]
    var size = (
        length if shape == ROWS or (grouped and shape == AGGREGATE) else 1
    )
    var active = fit_mask(mask, size)
    var predicate = _eval[width](
        bound, columns, aggregates, node.left, offset, length, grouped, mask
    )
    var contiguous_predicate = (
        predicate.rechunk() if predicate.is_chunked() else predicate.copy()
    )
    ref flags = contiguous_predicate._data[BoolColumn]
    var selected = List[Bool](capacity=size)
    var then_mask = List[Bool](capacity=size)
    var other_mask = List[Bool](capacity=size)
    for i in range(size):
        var p = 0 if len(flags) == 1 else i
        var take = flags._valid(p) and flags._get(p)
        var observed = len(active) == 0 or active[i]
        selected.append(take)
        then_mask.append(observed and take)
        other_mask.append(observed and not take)
    var then = _eval[width](
        bound,
        columns,
        aggregates,
        node.right,
        offset,
        length,
        grouped,
        then_mask,
    )
    var other: Series
    if node.extra >= 0:
        other = _eval[width](
            bound,
            columns,
            aggregates,
            node.extra,
            offset,
            length,
            grouped,
            other_mask,
        )
    else:
        other = Series.full_null("", bound.dtypes[index], 1)
    return choose(selected, then, other)


def _new_reducer(bound: BoundExpr, node: Node, group_count: Int) -> Reducer:
    return Reducer(
        node.op,
        bound.dtypes[node.left].physical(),
        group_count,
        node.min_count,
        node.integer,
        node.floating,
        node.text,
    )


struct _ReduceJob[width: Int](Job):
    """Reduce rows [start, end) of one reduction input into a local state."""

    var bound: BoundExpr
    var columns: List[Series]
    var states: List[Series]
    var node: Node
    var start: Int
    var end: Int
    var batch_size: Int
    var grouped: Bool
    var groups: ArcPointer[List[Int]]
    var reducer: Reducer

    def __init__(
        out self,
        bound: BoundExpr,
        columns: List[Series],
        states: List[Series],
        node: Node,
        start: Int,
        end: Int,
        batch_size: Int,
        grouped: Bool,
        groups: ArcPointer[List[Int]],
        group_count: Int,
    ):
        self.bound = bound.copy()
        self.columns = columns.copy()
        self.states = states.copy()
        self.node = node.copy()
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.grouped = grouped
        self.groups = groups
        self.reducer = _new_reducer(bound, node, group_count)

    def into_reducer(deinit self) -> Reducer:
        return self.reducer^

    def run(mut self) raises:
        if _direct_grouped_count(
            self.reducer,
            self.bound,
            self.columns,
            self.node,
            self.start,
            self.end,
            self.grouped,
            self.groups[],
        ):
            return
        if _direct_numeric_reduction(
            self.reducer,
            self.bound,
            self.columns,
            self.node,
            self.start,
            self.end,
            self.grouped,
        ):
            return
        for offset in range(self.start, self.end, self.batch_size):
            var chunk = _batch[Self.width](
                self.bound,
                self.columns,
                self.states,
                self.node.left,
                offset,
                min(self.batch_size, self.end - offset),
                False,
            )
            self.reducer.update(chunk, offset, self.grouped, self.groups[])


struct _RowsJob[width: Int](Job):
    """Evaluate rows [start, end) of a row-shaped expression."""

    var bound: BoundExpr
    var columns: List[Series]
    var states: List[Series]
    var root: Int
    var start: Int
    var end: Int
    var batch_size: Int
    var grouped: Bool
    var result: Series

    def __init__(
        out self,
        bound: BoundExpr,
        columns: List[Series],
        states: List[Series],
        root: Int,
        start: Int,
        end: Int,
        batch_size: Int,
        grouped: Bool,
    ) raises:
        self.bound = bound.copy()
        self.columns = columns.copy()
        self.states = states.copy()
        self.root = root
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.grouped = grouped
        self.result = _empty(bound.dtypes[root])
        self.result._reserve_rows(end - start, 0)

    def into_result(deinit self) -> Series:
        return self.result^

    def run(mut self) raises:
        for offset in range(self.start, self.end, self.batch_size):
            var chunk = _batch[Self.width](
                self.bound,
                self.columns,
                self.states,
                self.root,
                offset,
                min(self.batch_size, self.end - offset),
                self.grouped,
            )
            self.result._append_series(chunk)


def _direct_grouped_count_part(
    mut reducer: Reducer,
    part: Series,
    groups: List[Int],
    offset: Int,
    start: Int,
    end: Int,
):
    """Count valid source values by group without constructing row batches."""
    if part.null_count() == 0:
        for i in range(start, end):
            reducer.counts[groups[offset + i]] += 1
        return
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if part._data.isa[Column[Scalar[D]]]():
            ref column = part._data[Column[Scalar[D]]]
            for i in range(start, end):
                if column._valid(i):
                    reducer.counts[groups[offset + i]] += 1
            return
    if part._data.isa[BoolColumn]():
        ref column = part._data[BoolColumn]
        for i in range(start, end):
            if column._valid(i):
                reducer.counts[groups[offset + i]] += 1
        return
    ref column = part._data[StringColumn]
    for i in range(start, end):
        if column._valid(i):
            reducer.counts[groups[offset + i]] += 1


def _direct_grouped_count(
    mut reducer: Reducer,
    bound: BoundExpr,
    columns: List[Series],
    node: Node,
    start: Int,
    end: Int,
    grouped: Bool,
    groups: List[Int],
) -> Bool:
    if (
        not grouped
        or node.op != COUNT
        or bound.expr._nodes[node.left].op != COL
    ):
        return False
    ref source = columns[bound.sources[node.left]]
    var chunk_start = 0
    for part in source.chunks():
        var chunk_end = chunk_start + len(part)
        var lo = max(start, chunk_start)
        var hi = min(end, chunk_end)
        if lo < hi:
            _direct_grouped_count_part(
                reducer,
                part,
                groups,
                chunk_start,
                lo - chunk_start,
                hi - chunk_start,
            )
        chunk_start = chunk_end
        if chunk_start >= end:
            break
    return True


def _direct_float_column_sum(
    mut reducer: Reducer, column: Column[Float64], start: Int, end: Int
):
    """Accumulate a contiguous Float64 interval into one reducer state."""
    var values = column.unsafe_values()
    var total = SIMD[DType.float64, 8](0)
    var count = _count_valid(
        column._bits[], column._offset + start, end - start
    )
    var i = start
    if count == end - start:
        while i + 8 <= end:
            total += values.unsafe_load[width=8](i)
            i += 8
    else:
        var bits = column.unsafe_validity()
        while i + 8 <= end:
            var bit = column._offset + i
            var mask = UInt16(bits.unsafe_load(bit // 8)) >> UInt16(bit % 8)
            if bit % 8 > 0:
                mask |= UInt16(bits.unsafe_load(bit // 8 + 1)) << UInt16(
                    8 - bit % 8
                )
            var valid = (
                SIMD[DType.uint64, 8](
                    UInt64(mask),
                    UInt64(mask),
                    UInt64(mask),
                    UInt64(mask),
                    UInt64(mask),
                    UInt64(mask),
                    UInt64(mask),
                    UInt64(mask),
                )
                & SIMD[DType.uint64, 8](1, 2, 4, 8, 16, 32, 64, 128)
            ).ne(SIMD[DType.uint64, 8](0))
            total += valid.select(
                values.unsafe_load[width=8](i), SIMD[DType.float64, 8](0)
            )
            i += 8
    reducer.float_sums[0].total += total.reduce_add()
    reducer.float_sums[0].count += Int64(count)
    while i < end:
        if column._valid(i):
            reducer.float_sums[0].total += column._get(i)
        i += 1


def _direct_float_sum(
    mut reducer: Reducer,
    bound: BoundExpr,
    columns: List[Series],
    node: Node,
    start: Int,
    end: Int,
    grouped: Bool,
) raises -> Bool:
    """SIMD Float64 sum/mean across intersecting physical chunk intervals."""
    if (
        grouped
        or (node.op != SUM and node.op != MEAN)
        or bound.expr._nodes[node.left].op != COL
    ):
        return False
    ref series = columns[bound.sources[node.left]]
    if series.dtype() != DataType.FLOAT64:
        return False
    if not series.is_chunked():
        _direct_float_column_sum(
            reducer, series._data[Column[Float64]], start, end
        )
        return True
    ref chunks = series._chunked.value()[]
    var first = 0
    var upper = len(chunks.ends)
    while first < upper:
        var mid = (first + upper) // 2
        if chunks.ends[mid] <= start:
            first = mid + 1
        else:
            upper = mid
    for index in range(first, len(chunks.ends)):
        var chunk_start = 0 if index == 0 else chunks.ends[index - 1]
        if chunk_start >= end:
            break
        var chunk_end = chunks.ends[index]
        var lo = max(start, chunk_start)
        var hi = min(end, chunk_end)
        if lo < hi:
            _direct_float_column_sum(
                reducer,
                chunks.arrays[index][Column[Float64]],
                lo - chunk_start,
                hi - chunk_start,
            )
    return True


def _direct_numeric_column[
    D: DType
](
    mut reducer: Reducer,
    column: Column[Scalar[D]],
    op: Int,
    start: Int,
    end: Int,
) -> Bool:
    """Direct ungrouped reduction preserving Reducer's canonical state."""
    if op == COUNT:
        reducer.counts[0] += Int64(
            _count_valid(column._bits[], column._offset + start, end - start)
        )
        return True
    for i in range(start, end):
        if not column._valid(i):
            continue
        var value = column._get(i)
        if op == SUM or op == MEAN:
            comptime if D.is_integral():
                comptime if D == DType.uint64:
                    reducer.int_sums[0].add_wide(value.cast[DType.int128]())
                else:
                    reducer.int_sums[0].add(value.cast[DType.int64]())
            else:
                reducer.float_sums[0].add(value.cast[DType.float64]())
        else:
            var is_max = op == MAX
            comptime if D.is_floating_point():
                var floating = value.cast[DType.float64]()
                if isnan(floating):
                    reducer.nan_seen[0] = True
                elif (
                    not reducer.seen[0]
                    or (is_max and floating > reducer.floats[0])
                    or (not is_max and floating < reducer.floats[0])
                ):
                    reducer.floats[0] = floating
                    reducer.seen[0] = True
            else:
                var integer: Int64
                comptime if D == DType.uint64:
                    integer = (
                        value.cast[DType.uint64]() ^ (UInt64(1) << 63)
                    ).cast[DType.int64]()
                else:
                    integer = value.cast[DType.int64]()
                if (
                    not reducer.seen[0]
                    or (is_max and integer > reducer.ints[0])
                    or (not is_max and integer < reducer.ints[0])
                ):
                    reducer.ints[0] = integer
                    reducer.seen[0] = True
    return True


def _direct_numeric_part(
    mut reducer: Reducer,
    part: Series,
    op: Int,
    start: Int,
    end: Int,
) -> Bool:
    """Dispatch an interval inside one physical numeric array."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if part._data.isa[Column[Scalar[D]]]():
            return _direct_numeric_column[D](
                reducer, part._data[Column[Scalar[D]]], op, start, end
            )
    return False


def _direct_numeric_reduction(
    mut reducer: Reducer,
    bound: BoundExpr,
    columns: List[Series],
    node: Node,
    start: Int,
    end: Int,
    grouped: Bool,
) raises -> Bool:
    if grouped or bound.expr._nodes[node.left].op != COL:
        return False
    if _direct_float_sum(reducer, bound, columns, node, start, end, grouped):
        return True
    if (
        node.op != SUM
        and node.op != MEAN
        and node.op != COUNT
        and node.op != MIN
        and node.op != MAX
    ):
        return False
    ref series = columns[bound.sources[node.left]]
    if not series.is_chunked():
        return _direct_numeric_part(reducer, series, node.op, start, end)
    var chunk_start = 0
    for part in series.chunks():
        var chunk_end = chunk_start + len(part)
        var lo = max(start, chunk_start)
        var hi = min(end, chunk_end)
        if lo < hi and not _direct_numeric_part(
            reducer, part, node.op, lo - chunk_start, hi - chunk_start
        ):
            return False
        chunk_start = chunk_end
        if chunk_start >= end:
            break
    return True


def _reduce[
    width: Int
](
    bound: BoundExpr,
    columns: List[Series],
    states: List[Series],
    node: Node,
    height: Int,
    batch_size: Int,
    grouped: Bool,
    groups: List[Int],
    group_count: Int,
) raises -> Reducer:
    """Reduce every row of `node.left`: one pass, or one worker per row
    partition with states merged in partition order (#6, #8)."""
    var workers = worker_count(height)
    if (
        not grouped
        and (node.op == SUM or node.op == MEAN)
        and bound.expr._nodes[node.left].op == COL
        and columns[bound.sources[node.left]].dtype() == DataType.FLOAT64
    ):
        # SIMD scans saturate memory bandwidth with fewer workers; avoid
        # paying thread startup costs for small contiguous partitions.
        workers = min(workers, min(8, max(1, height // 250_000)))
    if grouped:
        # Each worker holds state for every group and merges it afterwards;
        # with many groups that costs more than it saves, so require an
        # average of at least four rows per group per worker.
        workers = min(workers, max(1, height // max(1, 4 * group_count)))
    if workers <= 1:
        var reducer = _new_reducer(bound, node, group_count)
        if _direct_grouped_count(
            reducer, bound, columns, node, 0, height, grouped, groups
        ):
            return reducer^
        if _direct_numeric_reduction(
            reducer, bound, columns, node, 0, height, grouped
        ):
            return reducer^
        for offset in range(0, height, batch_size):
            var chunk = _batch[width](
                bound,
                columns,
                states,
                node.left,
                offset,
                min(batch_size, height - offset),
                False,
            )
            reducer.update(chunk, offset, grouped, groups)
        return reducer^
    var shared_groups = ArcPointer(groups.copy())
    var bounds = partitions(height, workers, batch_size)
    var jobs = List[_ReduceJob[width]](capacity=workers)
    for w in range(workers):
        jobs.append(
            _ReduceJob[width](
                bound,
                columns,
                states,
                node,
                bounds[w],
                bounds[w + 1],
                batch_size,
                grouped,
                shared_groups,
                group_count,
            )
        )
    run_jobs(jobs)
    var reducer = jobs.pop(0).into_reducer()
    while len(jobs) > 0:
        reducer.merge(jobs.pop(0).into_reducer())
    return reducer^


def _batch[
    width: Int
](
    bound: BoundExpr,
    columns: List[Series],
    aggregates: List[Series],
    target: Int,
    offset: Int,
    length: Int,
    grouped: Bool,
) raises -> Series:
    return _eval[width](
        bound,
        columns,
        aggregates,
        target,
        offset,
        length,
        grouped,
        List[Bool](),
    )


def _full[
    width: Int
](
    bound: BoundExpr,
    columns: List[Series],
    states: List[Series],
    index: Int,
    height: Int,
    batch_size: Int,
    grouped: Bool,
) raises -> Series:
    """Materialize a row-valued node over every row, batch by batch."""
    var result = _empty(bound.dtypes[index])
    for offset in range(0, height, batch_size):
        var chunk = _batch[width](
            bound,
            columns,
            states,
            index,
            offset,
            min(batch_size, height - offset),
            grouped,
        )
        if len(chunk) == 1 and min(batch_size, height - offset) != 1:
            chunk = chunk._broadcast(min(batch_size, height - offset))
        result._append_series(chunk)
    return result^


def _over[
    width: Int
](
    bound: BoundExpr,
    columns: List[Series],
    index: Int,
    height: Int,
    batch_size: Int,
) raises -> Series:
    """Evaluate a node's child per partition and align results to rows."""
    ref node = bound.expr._nodes[index]
    var inner = bind(subtree(bound.expr, node.left), columns)
    if inner.shape() == SCALAR:
        var scalar = evaluate[width](
            inner, columns, height, batch_size=batch_size
        )
        return scalar._broadcast(height)
    var keys = List[Series]()
    for part in node.text2.split(SEP):
        for column in columns:
            if column.name() == String(part):
                keys.append(column.copy())
    var partitions = encode_rows(keys, nulls_equal=True)
    var result = evaluate[width](
        inner,
        columns,
        height,
        batch_size=batch_size,
        grouped=True,
        groups=partitions.ids,
        group_count=partitions.count(),
    )
    if inner.shape() == ROWS:
        return result^
    return result.take(partitions.ids)


# Eight lanes amortize expression-step dispatch across two AVX2 vectors and
# align fused comparisons with their packed output byte.
def evaluate[
    width: Int = 8
](
    bound: BoundExpr,
    columns: List[Series],
    height: Int,
    *,
    batch_size: Int = 1024,
    grouped: Bool = False,
    groups: List[Int] = List[Int](),
    group_count: Int = 1,
) raises -> Series:
    """Evaluate a bound expression; temporary vectors are bounded by batch_size.

    Reductions, window operations, and over() partitions are computed first,
    in node order, and then read per batch. Full output and
    O(groups * aggregates) states are materialized. With grouped=True, an
    aggregate-shaped expression yields one value per group; a row-shaped one
    (inside over()) yields one value per row, with windows restarting and
    aggregates broadcasting per group.
    """
    if batch_size <= 0:
        raise Error("batch_size must be positive")
    if grouped and (len(groups) != height or group_count < 0):
        raise Error("Invalid group mapping")
    var count = len(bound.expr._nodes)
    var root = count - 1
    # Fused kernels read source buffers directly. Small chunked inputs are
    # made contiguous once; large inputs use source chunk windows.
    var prepared_columns = columns.copy()
    var has_fused = False
    for i in range(count):
        if bound.fusible[i] and bound.expr._nodes[i].left >= 0:
            has_fused = True
            break
    # Small frames repay one contiguous copy through lower per-batch overhead;
    # larger ones keep source chunks and use bounded windows in fused().
    if has_fused and height < FUSED_CONTIGUOUS_ROWS:
        for i in range(count):
            if bound.expr._nodes[i].op == COL:
                var source = bound.sources[i]
                if (
                    source >= 0
                    and prepared_columns[source].dtype() == DataType.FLOAT64
                    and prepared_columns[source].is_chunked()
                ):
                    prepared_columns[source] = prepared_columns[
                        source
                    ].rechunk()
    var row_mode = grouped and bound.shape() == ROWS
    # Nodes under over() are evaluated by that over() in its own partitions.
    var inside_over = List[Bool](length=count, fill=False)
    for i in range(count):
        if bound.expr._nodes[i].op != OVER:
            continue
        var reachable = List[Bool](length=count, fill=False)
        reachable[bound.expr._nodes[i].left] = True
        for reverse in range(i):
            var j = i - 1 - reverse
            if not reachable[j]:
                continue
            inside_over[j] = True
            ref child = bound.expr._nodes[j]
            if child.left >= 0:
                reachable[child.left] = True
            if child.right >= 0:
                reachable[child.right] = True
            if child.extra >= 0:
                reachable[child.extra] = True
            if child.op == STRUCT_PACK:
                for field in struct_pack_children(child):
                    reachable[field] = True
    var states = List[Series]()
    for i in range(count):
        states.append(_empty(bound.dtypes[i]))
    for node_index in range(count):
        if inside_over[node_index]:
            continue
        var node = bound.expr._nodes[node_index].copy()
        if node.op == IMPLODE:
            var state = _implode[width](
                bound,
                prepared_columns,
                states,
                node,
                height,
                batch_size,
                grouped,
                groups,
                group_count,
            )
            if row_mode:
                state = state.take(groups)
            states[node_index] = state^
        elif is_reduction(node.op):
            var reducer = _reduce[width](
                bound,
                prepared_columns,
                states,
                node,
                height,
                batch_size,
                grouped,
                groups,
                group_count,
            )
            var state = reducer.finish()
            if row_mode:
                state = state.take(groups)
            states[node_index] = state^
        elif is_window(node.op):
            var input = _full[width](
                bound,
                prepared_columns,
                states,
                node.left,
                height,
                batch_size,
                row_mode,
            )
            states[node_index] = window_op(
                node, input, groups.copy() if grouped else List[Int]()
            )
        elif node.op == OVER:
            states[node_index] = _over[width](
                bound, prepared_columns, node_index, height, batch_size
            )
    var result = _empty(bound.dtypes[root])
    var size = height if bound.shape() == ROWS else (
        group_count if grouped else 1
    )
    var workers = worker_count(size) if bound.shape() == ROWS else 1
    if workers <= 1:
        result._reserve_rows(size, 0)
        for offset in range(0, size, batch_size):
            var chunk = _batch[width](
                bound,
                prepared_columns,
                states,
                root,
                offset,
                min(batch_size, size - offset),
                grouped,
            )
            result._append_series(chunk)
    else:
        # Row-parallel evaluation (#5): each worker evaluates the batches of
        # one contiguous partition into a private series; pieces are joined
        # in partition order. Precomputed states are shared read-only, and a
        # worker error discards every piece (no partial result escapes).
        var bounds = partitions(size, workers, batch_size)
        var jobs = List[_RowsJob[width]](capacity=workers)
        for w in range(workers):
            jobs.append(
                _RowsJob[width](
                    bound,
                    prepared_columns,
                    states,
                    root,
                    bounds[w],
                    bounds[w + 1],
                    batch_size,
                    grouped,
                )
            )
        run_jobs(jobs)
        var parts = List[Series](capacity=len(jobs))
        while len(jobs) > 0:
            parts.append(jobs.pop(0).into_result())
        result = Series._from_chunks(parts^)
    result._name = bound.expr._name
    return result^
