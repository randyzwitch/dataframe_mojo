"""Finish the CPU join workers' integer aggregate states on Metal."""

from max.gpu import block_idx, thread_idx
from std.memory import ArcPointer, Pointer
from std.time import perf_counter_ns

from dataframe.execution_observation import DeviceObservation
from .accelerator.gpu.query_scope import (
    Scope,
    checked_mul,
    checked_add,
    shared_service,
)
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.typed_storage import _NativeReservation


def _merge(
    workers: Pointer[UInt64, UntrackedOrigin[mut=False]],
    totals: Pointer[UInt64, MutUntrackedOrigin],
    fields: UInt32,
    rows: UInt32,
):
    var field = block_idx.x * 32 + thread_idx.x
    if field >= Int(fields):
        return
    var low = UInt64(0)
    var high = UInt64(0)
    var count = UInt64(0)
    for row in range(Int(rows)):
        var at = workers.unsafe_offset(3 * (row * Int(fields) + field))
        var previous = low
        low += at[]
        high += at[unsafe_offset=1] + UInt64(low < previous)
        count += at[unsafe_offset=2]
    var out_state = totals.unsafe_offset(3 * field)
    out_state[] = low
    out_state[unsafe_offset=1] = high
    out_state[unsafe_offset=2] = count


def combine_integrals(
    words: List[UInt64],
    fields: Int,
    observation: Optional[ArcPointer[DeviceObservation]],
) raises -> List[UInt64]:
    comptime if not METAL_TARGET:
        raise Error("Metal integral reductions require Apple silicon")
    else:
        var width = checked_mul(fields, 3)
        if (
            not width
            or not words
            or len(words) % width
            or len(words) > Int(UInt32.MAX)
        ):
            raise Error("Invalid integer aggregate state dimensions")
        var timings = HandoffTimings()
        var start = Int(perf_counter_ns())
        var scope = Scope(service=shared_service(), observation=observation)
        if observation:
            observation.value()[].begin()
        var ctx = scope.context()
        timings.context_ns = Int(perf_counter_ns()) - start
        var reserved = _NativeReservation(
            scope, checked_mul(8, checked_add(len(words), width))
        )
        var answer = List[UInt64](length=width, fill=0)
        var inputs = ctx.enqueue_create_buffer[DType.uint64](len(words))
        var totals = ctx.enqueue_create_buffer[DType.uint64](width)
        try:
            start = Int(perf_counter_ns())
            ctx.enqueue_copy(inputs, words.unsafe_ptr())
            scope.record_upload(len(words) * 8)
            timings.upload_ns = Int(perf_counter_ns()) - start
            start = Int(perf_counter_ns())
            ctx.enqueue_function[_merge](
                inputs,
                totals,
                UInt32(fields),
                UInt32(len(words) // width),
                grid_dim=(fields + 31) // 32,
                block_dim=32,
            )
            scope.record_launches(1)
            ctx.enqueue_copy(answer.unsafe_ptr(), totals)
            scope.record_download(width * 8)
            scope.synchronize()
            timings.kernel_ns = Int(perf_counter_ns()) - start
        except error:
            try:
                scope.synchronize()
            except:
                pass
            if observation:
                observation.value()[].failed(String(error))
            raise error^
        scope.mark_success()
        timings.record(observation)
        if observation:
            observation.value()[].complete()
        _ = reserved^
        return answer^
