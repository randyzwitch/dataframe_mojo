"""Compare List-pointer and explicit HostBuffer device downloads.

This diagnostic isolates one Float32 payload. A fixed raw-List upload feeds
the device; raw-List and HostBuffer downloads alternate order by repetition to
avoid always charging the first path setup. It times allocation, copy enqueue
and completion separately, then measures HostBuffer-to-owned-List bulk copying
outside device transfer timings. It does not run a GPU kernel,
exercise DeviceBuffer.unsafe_host_ptr(), or measure dataframe query performance.

Run on Apple Silicon with:
    pixi run -e gpu mojo -I providers/metal -I . benchmarks/bench_gpu_host_buffers.mojo
Optional positional arguments select one row count and repetitions. Defaults
are 4097 and 1048576 rows with five repetitions.
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.sys import argv
from std.time import perf_counter_ns


def run_float32(ctx: DeviceContext, rows: Int, reps: Int) raises:
    for rep in range(reps):
        var started = perf_counter_ns()
        var values = List[Float32](length=rows, fill=0)
        var list_readback = List[Float32](length=rows, fill=0)
        var adopted = List[Float32](capacity=rows)
        var list_alloc_ns = Int(perf_counter_ns() - started)

        # Exact binary fractions make the round-trip check bitwise and stable.
        for i in range(rows):
            values[i] = Float32(i % 257) * Float32(0.125)

        started = perf_counter_ns()
        var device_values = ctx.enqueue_create_buffer[DType.float32](rows)
        var device_buffer_enqueue_ns = Int(perf_counter_ns() - started)

        started = perf_counter_ns()
        var host_values = ctx.enqueue_create_host_buffer[DType.float32](rows)
        var host_buffer_enqueue_ns = Int(perf_counter_ns() - started)
        if len(device_values) != rows or len(host_values) != rows:
            raise Error("Host/device allocation length mismatch")
        started = perf_counter_ns()
        ctx.synchronize()
        var allocation_wait_ns = Int(perf_counter_ns() - started)

        # The upload path is held constant: a raw List pointer feeds the device.
        started = perf_counter_ns()
        ctx.enqueue_copy(device_values, values.unsafe_ptr())
        var upload_enqueue_ns = Int(perf_counter_ns() - started)
        started = perf_counter_ns()
        ctx.synchronize()
        var upload_wait_ns = Int(perf_counter_ns() - started)

        # Alternate first position by repetition while measuring both paths.
        var download_order = "list_then_host"
        var list_download_enqueue_ns = 0
        var list_download_wait_ns = 0
        var host_download_enqueue_ns = 0
        var host_download_wait_ns = 0
        if rep % 2 == 0:
            started = perf_counter_ns()
            ctx.enqueue_copy(list_readback.unsafe_ptr(), device_values)
            list_download_enqueue_ns = Int(perf_counter_ns() - started)
            started = perf_counter_ns()
            ctx.synchronize()
            list_download_wait_ns = Int(perf_counter_ns() - started)

            started = perf_counter_ns()
            ctx.enqueue_copy(host_values, device_values)
            host_download_enqueue_ns = Int(perf_counter_ns() - started)
            started = perf_counter_ns()
            ctx.synchronize()
            host_download_wait_ns = Int(perf_counter_ns() - started)
        else:
            download_order = "host_then_list"
            started = perf_counter_ns()
            ctx.enqueue_copy(host_values, device_values)
            host_download_enqueue_ns = Int(perf_counter_ns() - started)
            started = perf_counter_ns()
            ctx.synchronize()
            host_download_wait_ns = Int(perf_counter_ns() - started)

            started = perf_counter_ns()
            ctx.enqueue_copy(list_readback.unsafe_ptr(), device_values)
            list_download_enqueue_ns = Int(perf_counter_ns() - started)
            started = perf_counter_ns()
            ctx.synchronize()
            list_download_wait_ns = Int(perf_counter_ns() - started)

        # Model Column adoption's owned storage: one bulk Span append, timed on
        # its own and not folded into the asynchronous HostBuffer download.
        started = perf_counter_ns()
        var host_ptr = host_values.unsafe_ptr().unsafe_origin_cast[
            MutAnyOrigin
        ]()
        var host_span = Span[Float32, MutAnyOrigin](
            unsafe_ptr=host_ptr, length=rows
        )
        adopted.extend(host_span)
        var host_to_owned_list_ns = Int(perf_counter_ns() - started)

        # Validate exact Float32 bits after all synchronizations and timings.
        for i in range(rows):
            var expected = bitcast[DType.uint32](values[i])
            if (
                bitcast[DType.uint32](list_readback[i]) != expected
                or bitcast[DType.uint32](host_values[i]) != expected
                or bitcast[DType.uint32](adopted[i]) != expected
            ):
                raise Error("Float32 transfer round-trip mismatch")

        print(
            "HOST_BUFFER_SAMPLE",
            "rows",
            rows,
            "rep",
            rep,
            "download_order",
            download_order,
            "sample_kind",
            "first" if rep == 0 else "warm",
            "payload_bytes",
            4 * rows,
            "list_alloc_ns",
            list_alloc_ns,
            "device_buffer_enqueue_ns",
            device_buffer_enqueue_ns,
            "host_buffer_enqueue_ns",
            host_buffer_enqueue_ns,
            "allocation_wait_ns",
            allocation_wait_ns,
            "upload_enqueue_ns",
            upload_enqueue_ns,
            "upload_wait_ns",
            upload_wait_ns,
            "list_download_enqueue_ns",
            list_download_enqueue_ns,
            "list_download_wait_ns",
            list_download_wait_ns,
            "host_buffer_download_enqueue_ns",
            host_download_enqueue_ns,
            "host_buffer_download_wait_ns",
            host_download_wait_ns,
            "host_buffer_to_owned_list_ns",
            host_to_owned_list_ns,
            "validated",
            True,
        )


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
        raise Error("Usage: bench_gpu_host_buffers.mojo [rows [repetitions]]")
    if reps < 1:
        raise Error("Repetitions must be positive")
    var max_rows = Int(Int32.MAX) // 16
    for i in range(len(rows)):
        if rows[i] < 1:
            raise Error("Rows must be positive")
        if rows[i] > max_rows:
            raise Error("Rows exceed the safe HostBuffer probe bound")

    var context_started = perf_counter_ns()
    var ctx = DeviceContext(api="metal")
    var api = ctx.api()
    var device = ctx.name()
    ctx.synchronize()
    var context_ns = Int(perf_counter_ns() - context_started)
    print(
        "HOST_BUFFER_CONTEXT",
        "api",
        api,
        "device",
        device,
        "init_ns",
        context_ns,
        "repetitions",
        reps,
        "timed_synchronizations",
        "separate_upload_list_download_host_buffer_download_waits",
    )
    for i in range(len(rows)):
        run_float32(ctx, rows[i], reps)
