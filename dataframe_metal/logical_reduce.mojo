"""Word-wise fast path for existing global plain-column logical reductions."""
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, DataType, LazyFrame, Series
from dataframe.expr import COL, NOT, ANY, ALL, LEN, COUNT, NULL_COUNT
from dataframe.lazy import SCAN_FRAME, SELECT
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.reduction_inputs import HandoffTimings, _check_ran
from .accelerator.gpu.erased_input import numeric_views
from .accelerator.gpu.numeric_view import NumericView
from .accelerator.gpu.typed_storage import NativeColumnAny, _NativeReservation
from .accelerator.gpu.logical_words import logical_word_partials
from .accelerator.gpu.filter_sum import BLOCK_SIZE


@fieldwise_init
struct PackedLogicalPlan(Copyable, Movable):
    var inputs: List[String]
    var values: List[Bool]
    var validity: List[Bool]


def plan_packed_logical(
    region: LazyFrame,
) raises -> Optional[PackedLogicalPlan]:
    if len(region._plan._nodes) != 2:
        return None
    ref scan = region._plan._nodes[0]
    ref root = region._plan._nodes[1]
    if scan.kind != SCAN_FRAME or root.kind != SELECT or root.left != 0:
        return None
    ref frame = region._plan._frames[scan.source_slot()]
    var names = List[String]()
    var values = List[Bool]()
    var validity = List[Bool]()
    var outputs = List[String]()
    for expression in root.exprs:
        var size = len(expression._nodes)
        if size != 2 and size != 3:
            return None
        ref column = expression._nodes[0]
        ref aggregate = expression._nodes[size - 1]
        if column.op != COL or column.text not in frame.columns():
            return None
        if (
            aggregate.op not in [ANY, ALL, LEN, COUNT, NULL_COUNT]
            or aggregate.left != size - 2
        ):
            return None
        var dtype = frame.column(column.text).dtype()
        if dtype not in [
            DataType.BOOL,
            DataType.INT32,
            DataType.INT64,
            DataType.FLOAT32,
        ]:
            return None
        if size == 3 and (
            dtype != DataType.BOOL
            or expression._nodes[1].op != NOT
            or expression._nodes[1].left != 0
        ):
            return None
        var needs_value = aggregate.op == ANY or aggregate.op == ALL
        if needs_value and dtype != DataType.BOOL:
            return None
        if expression._name in outputs:
            return None
        outputs.append(expression._name.copy())
        if column.text not in names:
            names.append(column.text.copy())
            values.append(False)
            validity.append(False)
        var slot = names.index(column.text)
        values[slot] = values[slot] or needs_value
        validity[slot] = validity[slot] or aggregate.op != LEN
    if not names:
        return None
    return PackedLogicalPlan(names^, values^, validity^)


def execute_packed_logical(
    region: LazyFrame, plan: PackedLogicalPlan, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    comptime if not METAL_TARGET:
        raise Error("Metal packed logical reductions require an Apple target")
    else:
        ref frame = region._plan._frames[region._plan._nodes[0].source_slot()]
        if frame.height() >= Int(Int32.MAX):
            raise Error(
                "Metal packed logical reductions require fewer than 2**31 rows"
            )
        var timings = HandoffTimings()
        var started = Int(perf_counter_ns())
        var scope = Scope(service=shared_service())
        var ctx = scope.context()
        timings.context_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        var empty = ctx.enqueue_create_buffer[DType.uint8](1)
        var owners = List[NativeColumnAny]()
        var inputs = List[List[NumericView]]()
        var blocks = List[List[Int]]()
        var partials = 0
        for field in range(len(plan.inputs)):
            var views = numeric_views(
                frame,
                plan.inputs[field],
                scope,
                zero_copy,
                empty,
                owners,
                coalesce=True,
            )
            var counts = List[Int]()
            for view in views:
                var work = plan.values[field] or (
                    view.nullable and plan.validity[field]
                )
                var count = min(
                    128, max(1, (view.rows + 8191) // 8192)
                ) if work else 1
                counts.append(count)
                partials += count
            inputs.append(views^)
            blocks.append(counts^)
        timings.upload_ns = Int(perf_counter_ns()) - started
        var reservation = _NativeReservation(scope, partials * 8)
        var result = ctx.enqueue_create_buffer[DType.uint32](
            max(1, partials * 2)
        )
        var host = List[UInt32](length=partials * 2, fill=UInt32.MAX)
        started = Int(perf_counter_ns())
        try:
            var offset = 0
            for field in range(len(inputs)):
                for chunk in range(len(inputs[field])):
                    ref view = inputs[field][chunk]
                    var count = blocks[field][chunk]
                    var flags = UInt32(
                        Int(plan.values[field])
                        | (Int(view.nullable and plan.validity[field]) << 1)
                    )
                    ctx.enqueue_function[logical_word_partials](
                        view.values,
                        view.bits,
                        UInt32(view.value_bit_offset),
                        UInt32(view.bit_offset),
                        UInt32(view.rows),
                        flags,
                        UInt32(count * BLOCK_SIZE),
                        result,
                        UInt32(offset),
                        grid_dim=count,
                        block_dim=BLOCK_SIZE,
                    )
                    scope.record_launches(1)
                    offset += count * 2
            if partials:
                ctx.enqueue_copy(host.unsafe_ptr(), result)
            scope.synchronize()
        except error:
            try:
                scope.synchronize()
            except:
                pass
            _ = (inputs^, owners^, result^, host^, empty^, reservation^)
            raise error^
        _check_ran(host)
        scope.record_download(partials * 8)
        timings.kernel_ns = Int(perf_counter_ns()) - started
        started = Int(perf_counter_ns())
        var valid = List[Int64]()
        var true = List[Int64]()
        var at = 0
        for field in range(len(inputs)):
            var n = Int64(0)
            var t = Int64(0)
            for chunk in range(len(inputs[field])):
                for part in range(blocks[field][chunk]):
                    n += Int64(host[at])
                    t += Int64(host[at + 1])
                    at += 2
            valid.append(n)
            true.append(t)
        var columns = List[Series]()
        var rows = Int64(frame.height())
        for expression in region._plan._nodes[1].exprs:
            var slot = plan.inputs.index(expression._nodes[0].text)
            ref aggregate = expression._nodes[len(expression._nodes) - 1]
            var n = valid[slot]
            var t = (
                n - true[slot] if len(expression._nodes) == 3 else true[slot]
            )
            if aggregate.op == ANY or aggregate.op == ALL:
                var answer = t > 0 if aggregate.op == ANY else t == n
                var decisive = answer if aggregate.op == ANY else not answer
                var present = Bool(aggregate.integer) or decisive or n == rows
                columns.append(
                    Series(
                        expression._name.copy(),
                        Column[Bool]([answer], [present]),
                    )
                )
            else:
                var value = (
                    rows if aggregate.op
                    == LEN else rows - n if aggregate.op
                    == NULL_COUNT else n
                )
                columns.append(
                    Series(expression._name.copy(), Column[Int64]([value]))
                )
        timings.finish_ns = Int(perf_counter_ns()) - started
        scope.mark_success()
        _ = (inputs^, owners^, result^, host^, empty^, reservation^)
        return (DataFrame(columns^), timings^)
