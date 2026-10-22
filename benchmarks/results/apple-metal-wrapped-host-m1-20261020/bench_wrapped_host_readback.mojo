"""Compare three Float32 readback routes, including wrapped caller-owned pages.

This probe is intentionally outside the dataframe engine. It measures two
sizes used by the existing HostBuffer control, alternates path order, seeds
source buffers before timed samples, and includes time through owned List
materialization. It is not a GPU bandwidth or full-query benchmark.

Run on Apple Silicon only after review:
    pixi run -e gpu mojo -I . /tmp/wrapped_host_readback_proposal/bench_wrapped_host_readback.mojo
Optional arguments select one row count and repetitions.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import Allocation, Layout, alloc, bitcast
from std.os import abort
from std.sys import argv, size_of
from std.time import perf_counter_ns

comptime _PAGE_BYTES = 16 * 1024
comptime _PAGE_FLOATS = _PAGE_BYTES // size_of[Float32]()
comptime _PAGE_FLOAT_LAYOUT = Layout[
    Float32, alignment=.of_bytes[_PAGE_BYTES]()
]
comptime _PAD_SENTINEL = Float32(-12345.5)


def _expected(i: Int) -> Float32:
    return Float32(i % 257) * Float32(0.125)


def _padded_rows(rows: Int) -> Int:
    return ((rows + _PAGE_FLOATS - 1) // _PAGE_FLOATS) * _PAGE_FLOATS


def _check_list(actual: List[Float32], rows: Int, path: String) raises:
    if len(actual) != rows:
        raise Error(path + " output length mismatch")
    for i in range(rows):
        if bitcast[DType.uint32](actual[i]) != bitcast[DType.uint32](
            _expected(i)
        ):
            raise Error(path + " Float32 value mismatch")


def run_raw_list(
    ctx: DeviceContext,
    source: DeviceBuffer[DType.float32],
    rows: Int,
    rep: Int,
) raises:
    var started = perf_counter_ns()
    var all_in_started = started
    var result = List[Float32](length=rows, fill=0)
    var target_alloc_ns = Int(perf_counter_ns() - started)

    var copy_enqueue_ns: Int
    var transfer_wait_ns: Int
    try:
        started = perf_counter_ns()
        ctx.enqueue_copy(result.unsafe_ptr(), source)
        copy_enqueue_ns = Int(perf_counter_ns() - started)

        started = perf_counter_ns()
        ctx.synchronize()
        transfer_wait_ns = Int(perf_counter_ns() - started)
    except e:
        try:
            ctx.synchronize()
        except:
            abort(
                "raw_list copy drain failed; preserving the host destination "
                "requires process termination"
            )
        _ = result.unsafe_ptr()
        raise e
    var total_ns = Int(perf_counter_ns() - all_in_started)

    _check_list(result, rows, "raw_list")
    print(
        "READBACK_SAMPLE",
        "path",
        "raw_list",
        "rows",
        rows,
        "rep",
        rep,
        "sample_kind",
        "first" if rep == 0 else "warm",
        "logical_bytes",
        4 * rows,
        "transfer_bytes",
        4 * rows,
        "temporary_host_bytes",
        0,
        "owned_result_bytes",
        4 * rows,
        "shared_seed_device_bytes",
        4 * (rows + _padded_rows(rows)),
        "target_alloc_ns",
        target_alloc_ns,
        "wrap_ns",
        0,
        "copy_enqueue_ns",
        copy_enqueue_ns,
        "transfer_wait_ns",
        transfer_wait_ns,
        "owned_list_copy_ns",
        0,
        "release_unmap_wait_ns",
        0,
        "total_owned_ready_ns",
        total_ns,
        "validated",
        True,
    )


def run_host_buffer(
    ctx: DeviceContext,
    source: DeviceBuffer[DType.float32],
    rows: Int,
    rep: Int,
) raises:
    var started = perf_counter_ns()
    var all_in_started = started
    var host = ctx.enqueue_create_host_buffer[DType.float32](rows)
    var result = List[Float32](capacity=rows)
    var target_alloc_ns = Int(perf_counter_ns() - started)

    var copy_enqueue_ns: Int
    var transfer_wait_ns: Int
    try:
        started = perf_counter_ns()
        ctx.enqueue_copy(host, source)
        copy_enqueue_ns = Int(perf_counter_ns() - started)

        started = perf_counter_ns()
        ctx.synchronize()
        transfer_wait_ns = Int(perf_counter_ns() - started)
    except e:
        try:
            ctx.synchronize()
        except:
            abort(
                "HostBuffer copy drain failed; preserving the host destination "
                "requires process termination"
            )
        _ = host.unsafe_ptr()
        raise e

    started = perf_counter_ns()
    var host_ptr = host.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var host_span = Span[Float32, MutAnyOrigin](
        unsafe_ptr=host_ptr, length=rows
    )
    result.extend(host_span)
    var owned_list_copy_ns = Int(perf_counter_ns() - started)
    var total_ns = Int(perf_counter_ns() - all_in_started)

    _check_list(result, rows, "host_buffer")
    print(
        "READBACK_SAMPLE",
        "path",
        "host_buffer",
        "rows",
        rows,
        "rep",
        rep,
        "sample_kind",
        "first" if rep == 0 else "warm",
        "logical_bytes",
        4 * rows,
        "transfer_bytes",
        4 * rows,
        "temporary_host_bytes",
        4 * rows,
        "owned_result_bytes",
        4 * rows,
        "shared_seed_device_bytes",
        4 * (rows + _padded_rows(rows)),
        "target_alloc_ns",
        target_alloc_ns,
        "wrap_ns",
        0,
        "copy_enqueue_ns",
        copy_enqueue_ns,
        "transfer_wait_ns",
        transfer_wait_ns,
        "owned_list_copy_ns",
        owned_list_copy_ns,
        "release_unmap_wait_ns",
        0,
        "total_owned_ready_ns",
        total_ns,
        "validated",
        True,
    )


def _wrapped_copy(
    ctx: DeviceContext,
    padded_source: DeviceBuffer[DType.float32],
    host_ptr: Pointer[Float32, MutAnyOrigin],
    padded_rows: Int,
) raises -> Tuple[Int, Int, Int, Int]:
    # If wrap creation raises, no view is returned and no copy is enqueued.
    # The caller retains the owner and performs a context drain before release.
    var started = perf_counter_ns()
    var wrapped = ctx.wrap_host_memory[DType.float32](host_ptr, padded_rows)
    var wrap_ns = Int(perf_counter_ns() - started)
    var copy_enqueue_ns: Int
    var transfer_wait_ns: Int
    try:
        if ctx.api() == "metal" and wrapped.unsafe_ptr() == host_ptr:
            raise Error(
                "Metal wrapped pointer unexpectedly aliases the host pointer"
            )
        started = perf_counter_ns()
        ctx.enqueue_copy(wrapped, padded_source)
        copy_enqueue_ns = Int(perf_counter_ns() - started)

        started = perf_counter_ns()
        ctx.synchronize()
        transfer_wait_ns = Int(perf_counter_ns() - started)
    except e:
        # Dropping the view queues its release. The caller still owns the host
        # pages and drains the context before propagating this error.
        _ = wrapped^
        raise e

    # Explicitly destroy the wrapped view, then wait for its queued unmap while
    # the managed allocation owner is still live.
    started = perf_counter_ns()
    _ = wrapped^
    try:
        ctx.synchronize()
    except e:
        try:
            ctx.synchronize()
        except:
            pass
        raise e
    var release_unmap_wait_ns = Int(perf_counter_ns() - started)
    return wrap_ns, copy_enqueue_ns, transfer_wait_ns, release_unmap_wait_ns


def run_wrapped(
    ctx: DeviceContext,
    padded_source: DeviceBuffer[DType.float32],
    rows: Int,
    padded_rows: Int,
    rep: Int,
) raises:
    # The owner remains in this outer scope while _wrapped_copy destroys the
    # view and drains the deferred release. Consume it only after all reads.
    var started = perf_counter_ns()
    var all_in_started = started
    var host_owner = alloc(_PAGE_FLOAT_LAYOUT(count=padded_rows)).into_managed()
    var host_ptr = host_owner.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var result = List[Float32](capacity=rows)
    var target_alloc_ns = Int(perf_counter_ns() - started)

    var wrap_ns: Int
    var copy_enqueue_ns: Int
    var transfer_wait_ns: Int
    var release_unmap_wait_ns: Int
    try:
        wrap_ns, copy_enqueue_ns, transfer_wait_ns, release_unmap_wait_ns = (
            _wrapped_copy(ctx, padded_source, host_ptr, padded_rows)
        )
    except e:
        # Includes wrap_host_memory failures, where no wrapper handle exists.
        # The host owner is still live while this pending work is drained.
        var drained = False
        try:
            ctx.synchronize()
            drained = True
            _ = host_owner.unsafe_ptr()
        except:
            pass
        if not drained:
            # On an unrecoverable drain error, preserve the pages rather than
            # risk freeing memory still visible to the device.
            var abandoned = Allocation[
                Float32, alignment=.of_bytes[_PAGE_BYTES]()
            ](host_owner^)
            _ = abandoned^.unsafe_leak()
        raise e

    if host_owner.unsafe_ptr() != host_ptr.unsafe_origin_cast[MutAnyOrigin]():
        raise Error("wrapped host owner pointer changed during readback")

    started = perf_counter_ns()
    var host_span_ptr = host_ptr.unsafe_origin_cast[MutAnyOrigin]()
    var host_span = Span[Float32, MutAnyOrigin](
        unsafe_ptr=host_span_ptr, length=rows
    )
    result.extend(host_span)
    var owned_list_copy_ns = Int(perf_counter_ns() - started)
    var total_ns = Int(perf_counter_ns() - all_in_started)

    _check_list(result, rows, "wrapped")
    for i in range(rows, padded_rows):
        if bitcast[DType.uint32](
            host_ptr.unsafe_offset(i).unsafe_load()
        ) != bitcast[DType.uint32](_PAD_SENTINEL):
            raise Error("wrapped padding sentinel mismatch")

    print(
        "READBACK_SAMPLE",
        "path",
        "wrapped_host",
        "rows",
        rows,
        "rep",
        rep,
        "sample_kind",
        "first" if rep == 0 else "warm",
        "logical_bytes",
        4 * rows,
        "transfer_bytes",
        4 * padded_rows,
        "padding_bytes",
        4 * (padded_rows - rows),
        "temporary_host_bytes",
        4 * padded_rows,
        "owned_result_bytes",
        4 * rows,
        "shared_seed_device_bytes",
        4 * (rows + padded_rows),
        "target_alloc_ns",
        target_alloc_ns,
        "wrap_ns",
        wrap_ns,
        "copy_enqueue_ns",
        copy_enqueue_ns,
        "transfer_wait_ns",
        transfer_wait_ns,
        "owned_list_copy_ns",
        owned_list_copy_ns,
        "release_unmap_wait_ns",
        release_unmap_wait_ns,
        "total_owned_ready_ns",
        total_ns,
        "validated",
        True,
    )
    _ = host_owner^


def _seed_sources(
    ctx: DeviceContext, rows: Int, padded_rows: Int
) raises -> Tuple[DeviceBuffer[DType.float32], DeviceBuffer[DType.float32]]:
    var values = List[Float32](length=rows, fill=0)
    for i in range(rows):
        values[i] = _expected(i)

    var padded = List[Float32](length=padded_rows, fill=_PAD_SENTINEL)
    for i in range(rows):
        padded[i] = _expected(i)

    var source = ctx.enqueue_create_buffer[DType.float32](rows)
    var padded_source = ctx.enqueue_create_buffer[DType.float32](padded_rows)
    try:
        ctx.enqueue_copy(source, values.unsafe_ptr())
        ctx.enqueue_copy(padded_source, padded.unsafe_ptr())
        ctx.synchronize()
    except e:
        try:
            ctx.synchronize()
        except:
            abort(
                "source upload drain failed; preserving host source Lists "
                "requires process termination"
            )
        _ = values[0]
        _ = padded[0]
        raise e
    _ = values[0]
    _ = padded[0]
    return source^, padded_source^


def _run_rows(ctx: DeviceContext, rows: Int, reps: Int) raises:
    var padded_rows = ((rows + _PAGE_FLOATS - 1) // _PAGE_FLOATS) * _PAGE_FLOATS
    var source, padded_source = _seed_sources(ctx, rows, padded_rows)
    for rep in range(reps):
        # Rotate order to balance path-specific first-use and queue effects.
        for slot in range(3):
            var path = (rep + slot) % 3
            if path == 0:
                run_raw_list(ctx, source, rows, rep)
            elif path == 1:
                run_host_buffer(ctx, source, rows, rep)
            else:
                run_wrapped(ctx, padded_source, rows, padded_rows, rep)


def main() raises:
    var args = argv()
    var rows = List[Int]()
    rows.append(4097)
    rows.append(1048576)
    var reps = 5
    if len(args) == 2:
        rows = List[Int]()
        rows.append(Int(String(args[1])))
    elif len(args) == 3:
        rows = List[Int]()
        rows.append(Int(String(args[1])))
        reps = Int(String(args[2]))
    elif len(args) != 1:
        raise Error(
            "Usage: bench_wrapped_host_readback.mojo [rows [repetitions]]"
        )
    if reps < 1:
        raise Error("Repetitions must be positive")
    for i in range(len(rows)):
        if rows[i] < 1:
            raise Error("Rows must be positive")
        if rows[i] > 1048576:
            raise Error("Rows exceed the safe wrapped-host probe bound")

    var context_started = perf_counter_ns()
    var ctx = DeviceContext(api="metal")
    var api = ctx.api()
    var device = ctx.name()
    ctx.synchronize()
    var context_ns = Int(perf_counter_ns() - context_started)
    print(
        "READBACK_CONTEXT",
        "api",
        api,
        "device",
        device,
        "init_ns",
        context_ns,
        "repetitions",
        reps,
        "timed_synchronizations",
        "separate_source_seed_raw_host_wrapped_transfer_and_wrapped_unmap",
    )
    for i in range(len(rows)):
        _run_rows(ctx, rows[i], reps)
