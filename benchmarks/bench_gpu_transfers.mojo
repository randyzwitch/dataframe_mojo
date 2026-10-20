"""Primitive copy timing probe; this is not an end-to-end query or bandwidth claim.

Run on Apple Silicon with:
    pixi run -e gpu mojo -I . benchmarks/bench_gpu_transfers.mojo
Optional positional arguments select one row count and repetitions. The default
runs 4097 and 1048576 rows with five repetitions per layout.
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.sys import argv
from std.time import perf_counter_ns


def run_generic(ctx: DeviceContext, rows: Int, reps: Int) raises:
    for rep in range(reps):
        var started = perf_counter_ns()
        var low = List[UInt64](length=rows, fill=0)
        var high = List[UInt64](length=rows, fill=0)
        var validity = List[UInt8](length=rows, fill=0)
        var errors = List[UInt32](length=rows, fill=0)
        var read_low = List[UInt64](length=rows, fill=0)
        var read_high = List[UInt64](length=rows, fill=0)
        var read_validity = List[UInt8](length=rows, fill=0)
        var read_errors = List[UInt32](length=rows, fill=0)
        var host_alloc_ns = Int(perf_counter_ns() - started)
        for i in range(rows):
            var value = Float32(i % 257) * Float32(0.125)
            low[i] = UInt64(bitcast[DType.uint32](value))
            high[i] = UInt64(0xA5A5_0000_0000_0000) | UInt64(i)
            validity[i] = UInt8(i % 13 != 0)
            errors[i] = UInt32(0x5A5A_0000) | UInt32(i)
        started = perf_counter_ns()
        var device_low = ctx.enqueue_create_buffer[DType.uint64](rows)
        var device_high = ctx.enqueue_create_buffer[DType.uint64](rows)
        var device_validity = ctx.enqueue_create_buffer[DType.uint8](rows)
        var device_errors = ctx.enqueue_create_buffer[DType.uint32](rows)
        var device_alloc_ns = Int(perf_counter_ns() - started)
        started = perf_counter_ns()
        ctx.enqueue_copy(device_low, low.unsafe_ptr())
        ctx.enqueue_copy(device_high, high.unsafe_ptr())
        ctx.enqueue_copy(device_validity, validity.unsafe_ptr())
        ctx.enqueue_copy(device_errors, errors.unsafe_ptr())
        var upload_enqueue_ns = Int(perf_counter_ns() - started)
        started = perf_counter_ns()
        ctx.synchronize()
        var upload_wait_ns = Int(perf_counter_ns() - started)
        started = perf_counter_ns()
        ctx.enqueue_copy(read_low.unsafe_ptr(), device_low)
        ctx.enqueue_copy(read_high.unsafe_ptr(), device_high)
        ctx.enqueue_copy(read_validity.unsafe_ptr(), device_validity)
        ctx.enqueue_copy(read_errors.unsafe_ptr(), device_errors)
        var download_enqueue_ns = Int(perf_counter_ns() - started)
        started = perf_counter_ns()
        ctx.synchronize()
        var download_wait_ns = Int(perf_counter_ns() - started)
        for i in range(rows):
            if (
                read_low[i] != low[i]
                or read_high[i] != high[i]
                or read_validity[i] != validity[i]
                or read_errors[i] != errors[i]
            ):
                raise Error("generic plane round-trip mismatch")
        print(
            "SAMPLE",
            "rows",
            rows,
            "layout",
            "generic_21B_nullable",
            "rep",
            rep,
            "sample_kind",
            "first" if rep == 0 else "warm",
            "payload_bytes",
            21 * rows,
            "host_bytes",
            42 * rows,
            "device_bytes",
            21 * rows,
            "host_alloc_ns",
            host_alloc_ns,
            "device_buffer_enqueue_ns",
            device_alloc_ns,
            "upload_enqueue_ns",
            upload_enqueue_ns,
            "upload_wait_ns",
            upload_wait_ns,
            "download_enqueue_ns",
            download_enqueue_ns,
            "download_wait_ns",
            download_wait_ns,
            "validated",
            True,
        )


def run_float32(ctx: DeviceContext, rows: Int, reps: Int) raises:
    for rep in range(reps):
        var started = perf_counter_ns()
        var values = List[Float32](length=rows, fill=0)
        var read_values = List[Float32](length=rows, fill=0)
        var host_alloc_ns = Int(perf_counter_ns() - started)
        for i in range(rows):
            values[i] = Float32(i % 257) * Float32(0.125)
        started = perf_counter_ns()
        var device_values = ctx.enqueue_create_buffer[DType.float32](rows)
        var device_alloc_ns = Int(perf_counter_ns() - started)
        started = perf_counter_ns()
        ctx.enqueue_copy(device_values, values.unsafe_ptr())
        var upload_enqueue_ns = Int(perf_counter_ns() - started)
        started = perf_counter_ns()
        ctx.synchronize()
        var upload_wait_ns = Int(perf_counter_ns() - started)
        started = perf_counter_ns()
        ctx.enqueue_copy(read_values.unsafe_ptr(), device_values)
        var download_enqueue_ns = Int(perf_counter_ns() - started)
        started = perf_counter_ns()
        ctx.synchronize()
        var download_wait_ns = Int(perf_counter_ns() - started)
        for i in range(rows):
            if bitcast[DType.uint32](read_values[i]) != bitcast[DType.uint32](
                values[i]
            ):
                raise Error("Float32 round-trip mismatch")
        print(
            "SAMPLE",
            "rows",
            rows,
            "layout",
            "float32_4B",
            "rep",
            rep,
            "sample_kind",
            "first" if rep == 0 else "warm",
            "payload_bytes",
            4 * rows,
            "host_bytes",
            8 * rows,
            "device_bytes",
            4 * rows,
            "host_alloc_ns",
            host_alloc_ns,
            "device_buffer_enqueue_ns",
            device_alloc_ns,
            "upload_enqueue_ns",
            upload_enqueue_ns,
            "upload_wait_ns",
            upload_wait_ns,
            "download_enqueue_ns",
            download_enqueue_ns,
            "download_wait_ns",
            download_wait_ns,
            "validated",
            True,
        )


def run_nullable_float32(ctx: DeviceContext, rows: Int, reps: Int) raises:
    var bit_bytes = (rows + 7) // 8
    var payload_bytes = 4 * rows + bit_bytes
    for rep in range(reps):
        var started = perf_counter_ns()
        var values = List[Float32](length=rows, fill=0)
        var validity = List[UInt8](length=bit_bytes, fill=0)
        var read_values = List[Float32](length=rows, fill=0)
        var read_validity = List[UInt8](length=bit_bytes, fill=0)
        var host_alloc_ns = Int(perf_counter_ns() - started)
        for i in range(rows):
            if i % 13 != 0:
                values[i] = Float32(i % 257) * Float32(0.125)
                validity[i // 8] = validity[i // 8] | UInt8(1 << (i % 8))
        started = perf_counter_ns()
        var device_values = ctx.enqueue_create_buffer[DType.float32](rows)
        var device_validity = ctx.enqueue_create_buffer[DType.uint8](bit_bytes)
        var device_alloc_ns = Int(perf_counter_ns() - started)
        started = perf_counter_ns()
        ctx.enqueue_copy(device_values, values.unsafe_ptr())
        ctx.enqueue_copy(device_validity, validity.unsafe_ptr())
        var upload_enqueue_ns = Int(perf_counter_ns() - started)
        started = perf_counter_ns()
        ctx.synchronize()
        var upload_wait_ns = Int(perf_counter_ns() - started)
        started = perf_counter_ns()
        ctx.enqueue_copy(read_values.unsafe_ptr(), device_values)
        ctx.enqueue_copy(read_validity.unsafe_ptr(), device_validity)
        var download_enqueue_ns = Int(perf_counter_ns() - started)
        started = perf_counter_ns()
        ctx.synchronize()
        var download_wait_ns = Int(perf_counter_ns() - started)
        for i in range(rows):
            if bitcast[DType.uint32](read_values[i]) != bitcast[DType.uint32](
                values[i]
            ):
                raise Error("nullable Float32 value round-trip mismatch")
        for i in range(bit_bytes):
            if read_validity[i] != validity[i]:
                raise Error("packed validity round-trip mismatch")
        print(
            "SAMPLE",
            "rows",
            rows,
            "layout",
            "float32_packed_validity",
            "rep",
            rep,
            "sample_kind",
            "first" if rep == 0 else "warm",
            "payload_bytes",
            payload_bytes,
            "host_bytes",
            2 * payload_bytes,
            "device_bytes",
            payload_bytes,
            "host_alloc_ns",
            host_alloc_ns,
            "device_buffer_enqueue_ns",
            device_alloc_ns,
            "upload_enqueue_ns",
            upload_enqueue_ns,
            "upload_wait_ns",
            upload_wait_ns,
            "download_enqueue_ns",
            download_enqueue_ns,
            "download_wait_ns",
            download_wait_ns,
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
        raise Error("Usage: bench_gpu_transfers.mojo [rows [repetitions]]")
    if reps < 1:
        raise Error("Repetitions must be positive")
    var max_rows = Int(Int32.MAX) // 42
    for i in range(len(rows)):
        if rows[i] < 1:
            raise Error("Rows must be positive")
        if rows[i] > max_rows:
            raise Error("Rows exceed the safe transfer payload bound")
    var context_started = perf_counter_ns()
    var ctx = DeviceContext(api="metal")
    var api = ctx.api()
    var device = ctx.name()
    ctx.synchronize()
    var context_ns = Int(perf_counter_ns() - context_started)
    print(
        "TRANSFER_CONTEXT",
        "api",
        api,
        "device",
        device,
        "init_ns",
        context_ns,
        "repetitions",
        reps,
        "timed_synchronizations",
        "separate_upload_and_download_waits",
    )
    print(
        "TRANSFER_COLUMNS",
        "rows layout rep sample_kind payload_bytes host_bytes device_bytes host_alloc_ns device_buffer_enqueue_ns upload_enqueue_ns upload_wait_ns download_enqueue_ns download_wait_ns validated",
    )
    for i in range(len(rows)):
        run_generic(ctx, rows[i], reps)
        run_float32(ctx, rows[i], reps)
        run_nullable_float32(ctx, rows[i], reps)
