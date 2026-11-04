"""Regions the CPU executor hands to Metal (`LazyFrame._handoff_region`).

A region arrives as a plan over in-memory columns: one scan, the filters
directly above it, and the projection or aggregation the executor offered.
`supports` decides from the plan and the columns' types alone; `execute`
runs an accepted region and returns its columns. A declined region runs
on the CPU in the same query.

Accepted today: `select(col(c).sum())` over filters that compare `c` with
untyped numeric literals (`>`, `>=`, `<`, `<=`, `==`, joined by `&`), where
`c` is Int32, Int64 or Float32. A one-chunk column is read in place (`wrap_native`): Apple silicon shares
memory, so nothing is copied. Float64 and wider work stays on the CPU:
Apple GPUs have no 64-bit floating point, and emulating it would be slower
than the CPU.
"""
from max.gpu.host import DeviceBuffer
from std.utils.numerics import inf, isnan
from std.time import perf_counter_ns

from dataframe.storage.column import Column
from dataframe.dtype import DataType
from dataframe.expr import (
    AND,
    COL,
    EQ,
    Expr,
    GE,
    GT,
    LE,
    LIT_FLOAT,
    LIT_INT,
    LT,
    SUM,
    UNTYPED,
)
from dataframe.frame import DataFrame
from dataframe.lazy import FILTER, LazyFrame, SCAN_FRAME, SELECT
from dataframe.placement import load_costs
from dataframe.series import Series
from .accelerator.faults import data_fault
from .accelerator.gpu.filter_sum import (
    BLOCK_SIZE,
    float32_partials,
    integer_partials,
    split_total,
)
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.typed_storage import (
    NativeColumn,
    upload_native,
    wrap_native,
)

# Blocks per launch; a grid stride covers longer columns.
comptime MAX_PARTIALS = 1024


struct FilterSum(Copyable, Movable):
    """An accepted `select(col(c).sum())` over range filters on `c`."""

    var column: String
    var output: String
    var dtype: DataType
    var min_count: Int
    var filtered: Bool
    # Integer columns: the inclusive range kept.
    var low: Int64
    var high: Int64
    # Float32 columns: the range kept, with bit 0 making `low_float`
    # inclusive and bit 1 `high_float`.
    var low_float: Float32
    var high_float: Float32
    var inclusive: UInt32

    def __init__(out self, column: String, output: String, dtype: DataType):
        self.column = column
        self.output = output
        self.dtype = dtype
        self.min_count = 0
        self.filtered = False
        self.low = Int64.MIN
        self.high = Int64.MAX
        self.low_float = -inf[DType.float32]()
        self.high_float = inf[DType.float32]()
        self.inclusive = 3

    def _narrow(mut self, op: Int, integer: Int64, floating: Float64) -> Bool:
        """Keep only values satisfying `c <op> literal`; False when the
        literal cannot be represented exactly in the column's type."""
        self.filtered = True
        if self.dtype == DataType.FLOAT32:
            var bound = floating.cast[DType.float32]()
            if isnan(bound) or bound.cast[DType.float64]() != floating:
                return False
            if op == GT or op == GE or op == EQ:
                var exclusive = op == GT
                if bound > self.low_float or (
                    bound == self.low_float and exclusive
                ):
                    self.low_float = bound
                    self.inclusive = (self.inclusive & 2) | (
                        UInt32(0) if exclusive else UInt32(1)
                    )
            if op == LT or op == LE or op == EQ:
                var exclusive = op == LT
                if bound < self.high_float or (
                    bound == self.high_float and exclusive
                ):
                    self.high_float = bound
                    self.inclusive = (self.inclusive & 1) | (
                        UInt32(0) if exclusive else UInt32(2)
                    )
            return True
        var lowest = Int64(Int32.MIN) if self.dtype == DataType.INT32 else (
            Int64.MIN
        )
        var highest = Int64(Int32.MAX) if self.dtype == DataType.INT32 else (
            Int64.MAX
        )
        if integer < lowest or integer > highest:
            return False
        if op == GT:
            if integer == highest:
                self.low = highest
                self.high = lowest
                return True
            self.low = max(self.low, integer + 1)
        elif op == GE:
            self.low = max(self.low, integer)
        elif op == LT:
            if integer == lowest:
                self.low = highest
                self.high = lowest
                return True
            self.high = min(self.high, integer - 1)
        elif op == LE:
            self.high = min(self.high, integer)
        else:
            self.low = max(self.low, integer)
            self.high = min(self.high, integer)
        return True


def _predicate(mut plan: FilterSum, predicate: Expr, at: Int) raises -> Bool:
    """Narrow `plan` by the predicate subtree rooted at node `at`."""
    ref node = predicate._nodes[at]
    if node.op == AND:
        return _predicate(plan, predicate, node.left) and _predicate(
            plan, predicate, node.right
        )
    if not (
        node.op == GT
        or node.op == GE
        or node.op == LT
        or node.op == LE
        or node.op == EQ
    ):
        return False
    ref left = predicate._nodes[node.left]
    ref right = predicate._nodes[node.right]
    if left.op != COL or left.text != plan.column:
        return False
    if right.text != UNTYPED:
        return False
    if right.op == LIT_INT:
        if plan.dtype == DataType.FLOAT32:
            return plan._narrow(node.op, 0, Float64(right.integer))
        return plan._narrow(node.op, right.integer, 0)
    if right.op == LIT_FLOAT and plan.dtype == DataType.FLOAT32:
        return plan._narrow(node.op, 0, right.floating)
    return False


def plan_region(region: LazyFrame) raises -> Optional[FilterSum]:
    """The accepted shape of `region`, or None to leave it on the CPU."""
    var nodes = len(region._plan._nodes)
    if nodes < 2 or region._plan._nodes[0].kind != SCAN_FRAME:
        return None
    ref root = region._plan._nodes[nodes - 1]
    if root.kind != SELECT or len(root.exprs) != 1:
        return None
    ref total = root.exprs[0]
    if len(total._nodes) != 2:
        return None
    if total._nodes[0].op != COL or total._nodes[1].op != SUM:
        return None
    var name = total._nodes[0].text
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    if name not in frame.columns():
        return None
    var dtype = frame.column(name).dtype()
    if not (
        dtype == DataType.INT32
        or dtype == DataType.INT64
        or dtype == DataType.FLOAT32
    ):
        return None
    var plan = FilterSum(name, total._name, dtype)
    plan.min_count = total._nodes[1].min_count
    for i in range(1, nodes - 1):
        ref node = region._plan._nodes[i]
        if node.kind != FILTER or len(node.exprs) != 1:
            return None
        if not _predicate(plan, node.exprs[0], len(node.exprs[0]._nodes) - 1):
            return None
    return plan^


def supports(region: LazyFrame) -> Bool:
    """Whether Metal runs `region`; never raises."""
    comptime if not METAL_TARGET:
        return False
    try:
        return Bool(plan_region(region))
    except:
        return False


def operator_key(plan: FilterSum) -> String:
    """The calibration key of an accepted region (`dataframe.placement`)."""
    return (
        "filter_sum:"
        + plan.dtype.name()
        + (":filtered" if plan.filtered else ":plain")
    )


def estimate(region: LazyFrame, rows: Int) -> Tuple[String, Float64]:
    """This region's calibration key and its measured cost on Metal for
    `rows` input rows, in nanoseconds; -1 when it has no measured cost."""
    try:
        var accepted = plan_region(region)
        if not accepted:
            return (String(), Float64(-1))
        var key = operator_key(accepted.value())
        var costs = load_costs("accel")
        var cost = costs.get(key)
        if not cost:
            return (key^, Float64(-1))
        return (key^, cost.value().ns(rows))
    except:
        return (String(), Float64(-1))


struct HandoffTimings(Copyable, Movable):
    """Where one hand-off spent its time, in nanoseconds."""

    var context_ns: Int
    var upload_ns: Int
    var kernel_ns: Int
    var finish_ns: Int

    def __init__(out self):
        self.context_ns = 0
        self.upload_ns = 0
        self.kernel_ns = 0
        self.finish_ns = 0


def _place[
    dtype: DType
](scope: Scope, series: Series, zero_copy: Bool) raises -> NativeColumn[dtype]:
    """The column on the device: its own memory wrapped in place when
    `zero_copy` and the column is one chunk, else a copy."""
    if zero_copy:
        var wrapped = wrap_native[dtype](scope, series, wrap_validity=True)
        if wrapped:
            return wrapped.take()
    return upload_native[dtype](scope, series)


def _place_chunks[
    dtype: DType
](scope: Scope, series: Series, zero_copy: Bool) raises -> List[
    NativeColumn[dtype]
]:
    """Keep native chunks separate, including their own bitmap offsets.

    The caller keeps `series` alive through the batched reduction's final
    synchronization. Explicit copy mode remains a coalesced reference path.
    """
    var columns = List[NativeColumn[dtype]]()
    if zero_copy:
        for part in series.chunks():
            if len(part):
                columns.append(_place[dtype](scope, part, True))
    else:
        columns.append(_place[dtype](scope, series, False))
    return columns^


def _check_ran(counts: List[UInt32]) raises:
    """Raise unless every block wrote its count: a block holds fewer than
    2**31 rows, so the sentinel can only remain where a block never ran."""
    for count in counts:
        if count == UInt32.MAX:
            raise Error(
                "Mojo GPU hand-off kernel did not complete on the device"
            )


def _partials(rows: Int) -> Int:
    """Blocks to launch for `rows` rows: one per BLOCK_SIZE rows, capped
    at MAX_PARTIALS, with a grid stride covering the rest."""
    return max(1, min(MAX_PARTIALS, (rows + BLOCK_SIZE - 1) // BLOCK_SIZE))


def _chunk_partials(rows: Int, total_rows: Int) -> Int:
    """Share the grid budget across chunks instead of multiplying it.

    A small floor keeps a chunk's launch parallel; short chunks still use
    only the blocks their rows need. A single chunk keeps the original grid.
    """
    var share = (rows * MAX_PARTIALS + total_rows - 1) // total_rows
    return min(_partials(rows), max(64, share))


def _enqueue_integer[
    dtype: DType, nullable: Bool, filtered: Bool
](
    plan: FilterSum,
    column: NativeColumn[dtype],
    scope: Scope,
    partials: Int,
    offset: Int,
    high: DeviceBuffer[DType.int64],
    low: DeviceBuffer[DType.uint64],
    counts: DeviceBuffer[DType.uint32],
    empty_bits: DeviceBuffer[DType.uint8],
) raises:
    """Queue one chunk; its owners and output roots survive the final wait."""
    comptime if not METAL_TARGET:
        raise Error("Metal hand-off requires an Apple Metal compilation target")
    else:
        var bits = (
            column.validity().value().copy() if nullable else empty_bits.copy()
        )
        scope.context().enqueue_function[
            integer_partials[dtype, nullable, filtered]
        ](
            column.values(),
            UInt32(column.value_offset),
            bits,
            UInt32(column.validity_bit_offset),
            UInt32(column.length),
            UInt32(partials * BLOCK_SIZE),
            plan.low,
            plan.high,
            high,
            low,
            counts,
            UInt32(offset),
            grid_dim=partials,
            block_dim=BLOCK_SIZE,
        )
        scope.record_launches(1)


def _integer_launch[
    dtype: DType
](
    plan: FilterSum,
    columns: List[NativeColumn[dtype]],
    scope: Scope,
) raises -> Tuple[
    List[Scalar[DType.int64]], List[Scalar[DType.uint64]], List[UInt32]
]:
    """Queue all chunks into disjoint partial ranges; read back once.

    Keep every input, scratch buffer and host readback owner alive until
    synchronization, including when queueing or synchronization raises.
    """
    comptime if not METAL_TARGET:
        raise Error("Metal hand-off requires an Apple Metal compilation target")
    else:
        var ctx = scope.context()
        var total_rows = 0
        for column in columns:
            total_rows += column.length
        var total_parts = 0
        for column in columns:
            total_parts += _chunk_partials(column.length, total_rows)
        var high = ctx.enqueue_create_buffer[DType.int64](total_parts)
        var low = ctx.enqueue_create_buffer[DType.uint64](total_parts)
        var counts = ctx.enqueue_create_buffer[DType.uint32](total_parts)
        var empty_bits = ctx.enqueue_create_buffer[DType.uint8](1)
        var first_parts = List[Scalar[DType.int64]](length=total_parts, fill=0)
        var second_parts = List[Scalar[DType.uint64]](
            length=total_parts, fill=0
        )
        var count_parts = List[UInt32](length=total_parts, fill=0)
        try:
            ctx.enqueue_memset(counts, UInt32.MAX)
            var offset = 0
            for column in columns:
                var partials = _chunk_partials(column.length, total_rows)
                if column.has_validity():
                    if plan.filtered:
                        _enqueue_integer[dtype, True, True](
                            plan,
                            column,
                            scope,
                            partials,
                            offset,
                            high,
                            low,
                            counts,
                            empty_bits,
                        )
                    else:
                        _enqueue_integer[dtype, True, False](
                            plan,
                            column,
                            scope,
                            partials,
                            offset,
                            high,
                            low,
                            counts,
                            empty_bits,
                        )
                elif plan.filtered:
                    _enqueue_integer[dtype, False, True](
                        plan,
                        column,
                        scope,
                        partials,
                        offset,
                        high,
                        low,
                        counts,
                        empty_bits,
                    )
                else:
                    _enqueue_integer[dtype, False, False](
                        plan,
                        column,
                        scope,
                        partials,
                        offset,
                        high,
                        low,
                        counts,
                        empty_bits,
                    )
                offset += partials
            ctx.enqueue_copy(first_parts.unsafe_ptr(), high)
            ctx.enqueue_copy(second_parts.unsafe_ptr(), low)
            ctx.enqueue_copy(count_parts.unsafe_ptr(), counts)
            scope.synchronize()
        except error:
            try:
                scope.synchronize()
            except:
                pass
            raise error^
        _check_ran(count_parts)
        return (first_parts^, second_parts^, count_parts^)


def _integer_sum[
    dtype: DType
](
    plan: FilterSum,
    series: Series,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> Tuple[Int128, Int]:
    """The exact total of the kept values and how many were kept."""
    if len(series) == 0:
        return (Int128(0), 0)
    var started = Int(perf_counter_ns())
    var columns = _place_chunks[dtype](scope, series, zero_copy)
    timings.upload_ns = Int(perf_counter_ns()) - started
    started = Int(perf_counter_ns())
    var parts = _integer_launch[dtype](plan, columns, scope)
    timings.kernel_ns = Int(perf_counter_ns()) - started
    started = Int(perf_counter_ns())
    var total = Int128(0)
    var count = 0
    for p in range(len(parts[0])):
        comptime if split_total[dtype]():
            total += (parts[0][p].cast[DType.int128]() << 32) + parts[1][
                p
            ].cast[DType.int128]()
        else:
            total += parts[0][p].cast[DType.int128]()
        count += Int(parts[2][p])
    _ = columns^
    timings.finish_ns = Int(perf_counter_ns()) - started
    return (total, count)


def _enqueue_float32[
    nullable: Bool, filtered: Bool
](
    plan: FilterSum,
    column: NativeColumn[DType.float32],
    scope: Scope,
    partials: Int,
    offset: Int,
    sums: DeviceBuffer[DType.float32],
    compensations: DeviceBuffer[DType.float32],
    counts: DeviceBuffer[DType.uint32],
    empty_bits: DeviceBuffer[DType.uint8],
) raises:
    """Queue one chunk; its owners and output roots survive the final wait."""
    comptime if not METAL_TARGET:
        raise Error("Metal hand-off requires an Apple Metal compilation target")
    else:
        var bits = (
            column.validity().value().copy() if nullable else empty_bits.copy()
        )
        scope.context().enqueue_function[float32_partials[nullable, filtered]](
            column.values(),
            UInt32(column.value_offset),
            bits,
            UInt32(column.validity_bit_offset),
            UInt32(column.length),
            UInt32(partials * BLOCK_SIZE),
            plan.low_float,
            plan.high_float,
            plan.inclusive,
            sums,
            compensations,
            counts,
            UInt32(offset),
            grid_dim=partials,
            block_dim=BLOCK_SIZE,
        )
        scope.record_launches(1)


def _float32_launch(
    plan: FilterSum,
    columns: List[NativeColumn[DType.float32]],
    scope: Scope,
) raises -> Tuple[
    List[Scalar[DType.float32]], List[Scalar[DType.float32]], List[UInt32]
]:
    """Queue all chunks into disjoint partial ranges; read back once.

    Keep every input, scratch buffer and host readback owner alive until
    synchronization, including when queueing or synchronization raises.
    """
    comptime if not METAL_TARGET:
        raise Error("Metal hand-off requires an Apple Metal compilation target")
    else:
        var ctx = scope.context()
        var total_rows = 0
        for column in columns:
            total_rows += column.length
        var total_parts = 0
        for column in columns:
            total_parts += _chunk_partials(column.length, total_rows)
        var sums = ctx.enqueue_create_buffer[DType.float32](total_parts)
        var compensations = ctx.enqueue_create_buffer[DType.float32](
            total_parts
        )
        var counts = ctx.enqueue_create_buffer[DType.uint32](total_parts)
        var empty_bits = ctx.enqueue_create_buffer[DType.uint8](1)
        var first_parts = List[Scalar[DType.float32]](
            length=total_parts, fill=0
        )
        var second_parts = List[Scalar[DType.float32]](
            length=total_parts, fill=0
        )
        var count_parts = List[UInt32](length=total_parts, fill=0)
        try:
            ctx.enqueue_memset(counts, UInt32.MAX)
            var offset = 0
            for column in columns:
                var partials = _chunk_partials(column.length, total_rows)
                if column.has_validity():
                    if plan.filtered:
                        _enqueue_float32[True, True](
                            plan,
                            column,
                            scope,
                            partials,
                            offset,
                            sums,
                            compensations,
                            counts,
                            empty_bits,
                        )
                    else:
                        _enqueue_float32[True, False](
                            plan,
                            column,
                            scope,
                            partials,
                            offset,
                            sums,
                            compensations,
                            counts,
                            empty_bits,
                        )
                elif plan.filtered:
                    _enqueue_float32[False, True](
                        plan,
                        column,
                        scope,
                        partials,
                        offset,
                        sums,
                        compensations,
                        counts,
                        empty_bits,
                    )
                else:
                    _enqueue_float32[False, False](
                        plan,
                        column,
                        scope,
                        partials,
                        offset,
                        sums,
                        compensations,
                        counts,
                        empty_bits,
                    )
                offset += partials
            ctx.enqueue_copy(first_parts.unsafe_ptr(), sums)
            ctx.enqueue_copy(second_parts.unsafe_ptr(), compensations)
            ctx.enqueue_copy(count_parts.unsafe_ptr(), counts)
            scope.synchronize()
        except error:
            try:
                scope.synchronize()
            except:
                pass
            raise error^
        _check_ran(count_parts)
        return (first_parts^, second_parts^, count_parts^)


def _float32_sum(
    plan: FilterSum,
    series: Series,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> Tuple[Float64, Int]:
    """The compensated total of the kept values and how many were kept."""
    if len(series) == 0:
        return (Float64(0), 0)
    var started = Int(perf_counter_ns())
    var columns = _place_chunks[DType.float32](scope, series, zero_copy)
    timings.upload_ns = Int(perf_counter_ns()) - started
    started = Int(perf_counter_ns())
    var parts = _float32_launch(plan, columns, scope)
    timings.kernel_ns = Int(perf_counter_ns()) - started
    started = Int(perf_counter_ns())
    var total = Float64(0)
    var count = 0
    for p in range(len(parts[0])):
        total += (
            parts[0][p].cast[DType.float64]()
            - parts[1][p].cast[DType.float64]()
        )
        count += Int(parts[2][p])
    _ = columns^
    timings.finish_ns = Int(perf_counter_ns()) - started
    return (total, count)


def execute_timed(
    region: LazyFrame, zero_copy: Bool = True
) raises -> Tuple[DataFrame, HandoffTimings]:
    """Run an accepted region on Metal and report where the time went.
    With `zero_copy` (the default) the GPU reads every native chunk and
    its bitmap in place; without it the column is copied, for comparison."""
    comptime if not METAL_TARGET:
        raise Error("Metal hand-off requires an Apple Metal compilation target")
    else:
        return _execute_on_device(region, zero_copy)


def _execute_on_device(
    region: LazyFrame, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    var accepted = plan_region(region)
    if not accepted:
        raise Error("Metal unsupported [handoff]: region was not accepted")
    var plan = accepted.take()
    var timings = HandoffTimings()
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    var series = frame.column(plan.column)
    if len(series) >= Int(Int32.MAX):
        raise Error("Metal unsupported [handoff]: 2**31 or more rows")
    var started = Int(perf_counter_ns())
    var scope = Scope(service=shared_service())
    _ = scope.context()
    timings.context_ns = Int(perf_counter_ns()) - started
    var output: Series
    if plan.dtype == DataType.FLOAT32:
        var result = _float32_sum(plan, series, scope, zero_copy, timings)
        var valid = result[1] >= plan.min_count
        output = Series(
            plan.output,
            Column[Float32]([result[0].cast[DType.float32]()], [valid]),
        )
    else:
        var result: Tuple[Int128, Int]
        if plan.dtype == DataType.INT32:
            result = _integer_sum[DType.int32](
                plan, series, scope, zero_copy, timings
            )
        else:
            result = _integer_sum[DType.int64](
                plan, series, scope, zero_copy, timings
            )
        var total = result[0]
        var valid = result[1] >= plan.min_count
        if plan.dtype == DataType.INT32:
            if total > Int128(Int32.MAX) or total < Int128(Int32.MIN):
                raise data_fault("int32 expression sum overflow")
            output = Series(
                plan.output,
                Column[Int32]([total.cast[DType.int32]()], [valid]),
            )
        else:
            if total > Int128(Int64.MAX) or total < Int128(Int64.MIN):
                raise data_fault("int64 expression sum overflow")
            output = Series(
                plan.output,
                Column[Int64]([total.cast[DType.int64]()], [valid]),
            )
    scope.mark_success()
    return (DataFrame([output^]), timings^)


def execute(region: LazyFrame) raises -> DataFrame:
    """Run an accepted region on Metal; its columns."""
    return execute_timed(region)[0].copy()
