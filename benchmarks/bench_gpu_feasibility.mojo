"""NVIDIA mechanism experiment for #528; no library backend/API changes.

Compare the existing lazy CPU API with a handwritten, fused GPU query:
Filter x > 0, multiply in the input dtype by 1.25, then sum/count with
Float64 accumulation and an input-dtype sum result.
The predicate is fused into reduction; this does NOT implement general
filter compaction or expression compilation. All partials stay on device.

Run --check for correctness only; otherwise emit benchmark CSV after checks.
Optional positional arguments: ROWS REPETITIONS (one size, both dtypes).
"""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.sync import barrier
from std.math import abs, isfinite, isnan
from std.memory import AddressSpace, bitcast, stack_allocation
from std.os import getenv
from std.sys import argv
from std.time import monotonic

from dataframe import Column, DataFrame, LazyFrame, Series, col, lit
from dataframe.parallel import worker_count

comptime THREADS = 256
comptime MAX_BLOCKS = 1024


@always_inline
def reduce_pair(pair: SIMD[DType.float64, 2]) -> SIMD[DType.float64, 2]:
    # The pinned nightly's block.sum rejects Float64 warp shuffles.
    # Preserve Float64 accumulation with a simple shared-memory tree.
    # Every thread must participate, including threads beyond input length.
    var shared = stack_allocation[
        THREADS * 2, DType.float64, address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    shared[unsafe_offset=tid * 2] = pair[0]
    shared[unsafe_offset=tid * 2 + 1] = pair[1]
    barrier()
    # Eight stages for this experiment's fixed 256-thread block.
    comptime for step in range(8):
        comptime stride = THREADS >> (step + 1)
        if tid < stride:
            shared[unsafe_offset=tid * 2] += shared[
                unsafe_offset=(tid + stride) * 2
            ]
            shared[unsafe_offset=tid * 2 + 1] += shared[
                unsafe_offset=(tid + stride) * 2 + 1
            ]
        barrier()
    return SIMD[DType.float64, 2](
        shared[unsafe_offset=0], shared[unsafe_offset=1]
    )


def partial_sum[
    D: DType
](
    values: DeviceBuffer[D].device_type,
    bits: DeviceBuffer[DType.uint8].device_type,
    partials: DeviceBuffer[DType.float64].device_type,
    rows: Int64,
    bit_offset: Int64,
    has_bits: Int64,
    blocks: Int64,
):
    var total = Float64(0)
    var count = Float64(0)
    var i = Int(block_idx.x) * THREADS + Int(thread_idx.x)
    while i < Int(rows):
        var valid = True
        if has_bits:
            var b = Int(bit_offset) + i
            valid = (
                bits[unsafe_offset=b // 8] & (UInt8(1) << UInt8(b % 8))
            ) != 0
        # Test validity before loading/evaluating the payload.
        if valid:
            var x = values[unsafe_offset=i]
            if x > 0:
                # Round the product in D before widening, like the CPU API.
                total += (x * Scalar[D](1.25)).cast[DType.float64]()
                count += 1
        i += Int(blocks) * THREADS
    var pair = SIMD[DType.float64, 2](total, count)
    var reduced = reduce_pair(pair)
    if thread_idx.x == 0:
        partials[unsafe_offset=Int(block_idx.x) * 2] = reduced[0]
        partials[unsafe_offset=Int(block_idx.x) * 2 + 1] = reduced[1]


def finish_sum[
    D: DType
](
    partials: DeviceBuffer[DType.float64].device_type,
    output: DeviceBuffer[DType.float64].device_type,
    blocks: Int64,
):
    var pair = SIMD[DType.float64, 2](0)
    var i = Int(thread_idx.x)
    while i < Int(blocks):
        pair[0] += partials[unsafe_offset=i * 2]
        pair[1] += partials[unsafe_offset=i * 2 + 1]
        i += THREADS
    var reduced = reduce_pair(pair)
    if thread_idx.x == 0:
        output[unsafe_offset=0] = reduced[0].cast[D]().cast[DType.float64]()
        output[unsafe_offset=1] = reduced[1]


struct GpuInput[D: DType](Movable):
    """Own both host sources and device buffers until work is synchronized."""

    var ctx: DeviceContext
    var source: Column[Scalar[Self.D]]
    var values: DeviceBuffer[Self.D]
    var bits: DeviceBuffer[DType.uint8]
    var partials: DeviceBuffer[DType.float64]
    var output: DeviceBuffer[DType.float64]
    var blocks: Int
    var has_bits: Bool
    var bit_offset: Int
    var bitmap_bytes: Int

    def __init__(
        out self, ctx: DeviceContext, source: Column[Scalar[Self.D]]
    ) raises:
        self.ctx = ctx
        self.source = source.copy()
        self.blocks = max(
            1, min(MAX_BLOCKS, (len(source) + THREADS - 1) // THREADS)
        )
        # Inspect the stored bitmap, avoiding an O(n) null_count prepass.
        self.has_bits = len(source._bits[]) != 0
        self.bit_offset = source.validity_offset() % 8
        self.bitmap_bytes = (
            self.bit_offset + len(source) + 7
        ) // 8 if self.has_bits and len(source) > 0 else 0
        self.values = ctx.enqueue_create_buffer[Self.D](max(1, len(source)))
        self.bits = ctx.enqueue_create_buffer[DType.uint8](
            max(1, self.bitmap_bytes)
        )
        self.partials = ctx.enqueue_create_buffer[DType.float64](
            self.blocks * 2
        )
        self.output = ctx.enqueue_create_buffer[DType.float64](2)

    def upload(self) raises:
        if len(self.source) > 0:
            self.ctx.enqueue_copy(self.values, self.source.unsafe_values())
        if self.bitmap_bytes > 0:
            self.ctx.enqueue_copy(
                self.bits,
                self.source.unsafe_validity().unsafe_offset(
                    self.source.validity_offset() // 8
                ),
            )

    def enqueue(self) raises:
        self.ctx.enqueue_function[partial_sum[Self.D]](
            self.values,
            self.bits,
            self.partials,
            Int64(len(self.source)),
            Int64(self.bit_offset),
            Int64(self.has_bits),
            Int64(self.blocks),
            grid_dim=self.blocks,
            block_dim=THREADS,
        )
        self.ctx.enqueue_function[finish_sum[Self.D]](
            self.partials,
            self.output,
            Int64(self.blocks),
            grid_dim=1,
            block_dim=THREADS,
        )

    def download(self) raises -> SIMD[DType.float64, 2]:
        var result = List[Float64](length=2, fill=0)
        self.ctx.enqueue_copy(result.unsafe_ptr(), self.output)
        self.ctx.synchronize()
        return SIMD[DType.float64, 2](result[0], result[1])


def query[D: DType](source: Column[Scalar[D]]) raises -> LazyFrame:
    return (
        DataFrame([Series("x", source.copy())])
        .lazy()
        .filter(col("x") > lit(Scalar[D](0)))
        .select_exprs(
            [
                (col("x") * lit(Scalar[D](1.25))).sum().alias("total"),
                col("x").count().alias("count"),
            ]
        )
    )


def cpu_answer[D: DType](frame: DataFrame) raises -> SIMD[DType.float64, 2]:
    var total: Float64
    comptime if D == DType.float32:
        total = Float64(frame.item(0, "total").float32())
    else:
        total = frame.item(0, "total").float64()
    return SIMD[DType.float64, 2](
        total, Float64(frame.item(0, "count").int64())
    )


def scalar_answer[
    D: DType
](source: Column[Scalar[D]]) -> SIMD[DType.float64, 2]:
    var total = Float64(0)
    var count = Float64(0)
    for i in range(len(source)):
        if source.is_valid(i):
            var x = source._get(i)
            if x > 0:
                total += (x * Scalar[D](1.25)).cast[DType.float64]()
                count += 1
    return SIMD[DType.float64, 2](total.cast[D]().cast[DType.float64](), count)


def check(
    actual: SIMD[DType.float64, 2], expected: SIMD[DType.float64, 2]
) raises:
    if actual[1] != expected[1]:
        raise Error("selected row count mismatch")
    var a = actual[0]
    var b = expected[0]
    if a == b or (isnan(a) and isnan(b)):
        return
    if (
        not isfinite(a)
        or not isfinite(b)
        or abs(a - b) > 1e-10 * max(1.0, abs(b))
    ):
        raise Error("sum mismatch: " + String(a) + " vs " + String(b))


def fresh[
    D: DType
](ctx: DeviceContext, source: Column[Scalar[D]]) raises -> SIMD[
    DType.float64, 2
]:
    var gpu = GpuInput[D](ctx, source)
    gpu.upload()
    gpu.enqueue()
    return gpu.download()


def round_trip[D: DType](ctx: DeviceContext, source: Column[Scalar[D]]) raises:
    var gpu = GpuInput[D](ctx, source)
    gpu.upload()
    var values = List[Scalar[D]](length=max(1, len(source)), fill=0)
    var bits = List[UInt8](length=max(1, gpu.bitmap_bytes), fill=0)
    if len(source) > 0:
        ctx.enqueue_copy(values.unsafe_ptr(), gpu.values)
    if gpu.bitmap_bytes > 0:
        ctx.enqueue_copy(bits.unsafe_ptr(), gpu.bits)
    ctx.synchronize()
    for i in range(len(source)):
        comptime U = DType.uint32 if D == DType.float32 else DType.uint64
        if bitcast[U](values[i]) != bitcast[U](source._get(i)):
            raise Error("round-trip payload bits changed")
        var valid = True
        if gpu.has_bits:
            var b = gpu.bit_offset + i
            valid = (bits[b // 8] & (UInt8(1) << UInt8(b % 8))) != 0
        if valid != source.is_valid(i):
            raise Error("round-trip validity changed")


def validate[D: DType](ctx: DeviceContext, source: Column[Scalar[D]]) raises:
    round_trip[D](ctx, source)
    var expected = scalar_answer[D](source)
    check(cpu_answer[D](query[D](source).collect()), expected)
    check(fresh[D](ctx, source), expected)


def make_input[D: DType](rows: Int, variant: Int) raises -> Column[Scalar[D]]:
    # Seeded shuffled values; variants change null density and selectivity.
    var values = List[Scalar[D]](capacity=rows + 19)
    var valid = List[Bool](capacity=rows + 19)
    var rng = UInt64(123456789)
    for i in range(rows + 19):
        rng = rng * 6364136223846793005 + 1442695040888963407
        var x = Float64(Int((rng >> 32) % 2001) - 1000) / 16
        if variant == 2:
            x -= 60
        var present = variant == 0 or i % (3 if variant == 2 else 10) != 0
        values.append(Scalar[D](x) if present else Scalar[D](Float64("nan")))
        valid.append(present)
    if variant == 0:
        return Column[Scalar[D]](values^).slice(0, rows)
    # Start beyond byte 0 and at a non-byte-aligned validity offset.
    return Column[Scalar[D]](values^, valid^).slice(19, rows)


def correctness[D: DType](ctx: DeviceContext) raises:
    for rows in [0, 1, 7, 8, 9, 31, 32, 33, 255, 256, 257, 1025]:
        for variant in range(3):
            validate[D](ctx, make_input[D](rows, variant))
    var values = List[Scalar[D]](
        [
            Scalar[D](-0.0),
            0,
            -1,
            2,
            Scalar[D](Float64("nan")),
            Scalar[D](Float64("inf")),
            Scalar[D](Float64("-inf")),
            0.1,
            0.3,
        ]
    )
    var mixed = Column[Scalar[D]](values.copy())
    for offset in range(len(values) + 1):
        validate[D](ctx, mixed.slice(offset, len(values) - offset))
    var all_null = Column[Scalar[D]](
        values.copy(), List[Bool](length=len(values), fill=False)
    )
    validate[D](ctx, all_null)
    var hidden_inf = Column[Scalar[D]](
        values^, [True, True, True, True, False, False, True, True, True]
    )
    for offset in range(len(hidden_inf) + 1):
        validate[D](ctx, hidden_inf.slice(offset, len(hidden_inf) - offset))
    # Non-binary fractions, product rounding, and a wide exponent range.
    validate[D](
        ctx,
        Column[Scalar[D]](
            [
                Scalar[D](0.1),
                Scalar[D](0.3),
                Scalar[D](1e-20),
                Scalar[D](16777217.0),
                Scalar[D](1234567.89),
                -7,
                0,
            ]
        ),
    )
    comptime if D == DType.float32:
        validate[D](ctx, Column[Scalar[D]]([Scalar[D](3e38)]))
    else:
        validate[D](ctx, Column[Scalar[D]]([Scalar[D](1.7e308)]))


@fieldwise_init
struct Timing(Copyable):
    var best: Int
    var total: Int

    def add(mut self, ns: Int):
        self.best = min(self.best, ns)
        self.total += ns


def report[
    D: DType
](rows: Int, variant: Int, mode: String, t: Timing, repetitions: Int):
    print(
        String(D), rows, variant, mode, t.best, t.total // repetitions, sep=","
    )


def benchmark[
    D: DType
](ctx: DeviceContext, rows: Int, variant: Int, repetitions: Int) raises:
    var source = make_input[D](rows, variant)
    var plan = query[D](source)
    var expected = scalar_answer[D](source)
    check(cpu_answer[D](plan.collect()), expected)
    check(fresh[D](ctx, source), expected)
    var gpu = GpuInput[D](ctx, source)
    gpu.upload()
    gpu.enqueue()
    check(gpu.download(), expected)
    var cpu = Timing(Int.MAX, 0)
    var end_to_end = Timing(Int.MAX, 0)
    var reused = Timing(Int.MAX, 0)
    var resident = Timing(Int.MAX, 0)
    var scalar = Timing(Int.MAX, 0)
    for _ in range(repetitions):
        var start = monotonic()
        var frame = plan.collect()
        cpu.add(Int(monotonic() - start))
        check(cpu_answer[D](frame), expected)
        start = monotonic()
        var result = fresh[D](ctx, source)
        end_to_end.add(Int(monotonic() - start))
        check(result, expected)
        start = monotonic()
        gpu.upload()
        gpu.enqueue()
        result = gpu.download()
        reused.add(Int(monotonic() - start))
        check(result, expected)
        start = monotonic()
        gpu.enqueue()
        result = gpu.download()
        resident.add(Int(monotonic() - start))
        check(result, expected)
        start = monotonic()
        result = scalar_answer[D](source)
        scalar.add(Int(monotonic() - start))
        check(result, expected)
    report[D](rows, variant, "cpu_lazy", cpu, repetitions)
    report[D](
        rows,
        variant,
        "gpu_alloc_upload_query_download",
        end_to_end,
        repetitions,
    )
    report[D](
        rows, variant, "gpu_reuse_upload_query_download", reused, repetitions
    )
    report[D](
        rows, variant, "gpu_resident_query_download", resident, repetitions
    )
    report[D](rows, variant, "cpu_fused_scalar", scalar, repetitions)
    # Device event timing: resident inputs, no allocation or result download.
    # Includes both kernel launches and the gap between them on the stream.
    def enqueue(ctx: DeviceContext) raises {imm gpu}:
        gpu.enqueue()

    var device = Timing(Int.MAX, 0)
    for _ in range(repetitions):
        device.add(ctx.execution_time(enqueue, 1))
        check(gpu.download(), expected)
    report[D](rows, variant, "gpu_device", device, repetitions)


def main() raises:
    var args = argv()
    var check_only = len(args) == 2 and String(args[1]) == "--check"
    var sizes = List[Int](
        [
            1_000,
            10_000,
            100_000,
            250_000,
            500_000,
            1_000_000,
            5_000_000,
            10_000_000,
        ]
    )
    var repetitions = 7
    if len(args) > 1 and not check_only:
        if len(args) != 3:
            raise Error(
                "Usage: bench_gpu_feasibility [--check | ROWS REPETITIONS]"
            )
        sizes = List[Int]([Int(String(args[1]))])
        repetitions = Int(String(args[2]))
        if sizes[0] < 0 or repetitions < 1:
            raise Error("ROWS must be nonnegative and REPETITIONS positive")
    var start = monotonic()
    var ctx = DeviceContext()
    if ctx.api() != "cuda":
        raise Error("This experiment requires an NVIDIA CUDA device")
    print("# device:", ctx.name())
    print("# context_init_ns:", monotonic() - start)
    print("# DATAFRAME_THREADS:", getenv("DATAFRAME_THREADS"))
    print("# worker_budget_at_10m_rows:", worker_count(10_000_000))
    comptime for d in range(2):
        comptime D = DType.float32 if d == 0 else DType.float64
        var source = make_input[D](1025, 1)
        start = monotonic()
        var result = fresh[D](ctx, source)
        print("# first_gpu_query_ns:", String(D), monotonic() - start)
        check(result, scalar_answer[D](source))
        correctness[D](ctx)
        print("# correctness passed:", String(D))
    if check_only:
        return
    print("dtype,rows,variant,mode,best_ns,mean_ns")
    comptime for d in range(2):
        comptime D = DType.float32 if d == 0 else DType.float64
        for rows in sizes:
            for variant in range(3):
                benchmark[D](ctx, rows, variant, repetitions)
