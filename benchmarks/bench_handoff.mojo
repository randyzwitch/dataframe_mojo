"""Size sweep for the Metal hand-off: CPU against Metal on one filtered sum.

    mojo build -O3 -I . benchmarks/bench_handoff.mojo -o build/bench_handoff
    build/bench_handoff [REPS] [MAX_ROWS]

For each dtype and row count, `filter(x > t).select(x.sum())` (about half
the rows kept) and the unfiltered sum run warm `REPS` times on the CPU
executor and with the Metal hand-off. Each line reports the best time of
each, their ratio, and where the Metal time went: device context, making
the column visible to the GPU (`wrap_ms`: wrapped in place, as the hand-off
does; `copy_ms`: copied, for comparison), the kernel with readback of the
per-block partials, and the host finish.
`handed` says whether `explain` shows the node handed to Metal, so a plan
the optimizer reshapes cannot pass CPU time off as Metal's; `wrap_ok` and
`copy_ok` say whether Metal's answer, read in place and copied, equals the
CPU's (Float32 within a relative 1e-5), so a fast wrong answer shows. One TSV line
per case on stdout:

    case dtype rows handed wrap_ok copy_ok cpu_ms metal_ms metal/cpu context_ms wrap_ms copy_ms kernel_ms finish_ms
"""
from std.sys import argv
from std.time import perf_counter_ns

from dataframe import Column, DataFrame, DataType, LazyFrame, Series, col
from dataframe_metal.handoff import execute_timed
from dataframe_metal.metal import MetalRuntime


def _frame[dtype: DType](rows: Int) raises -> DataFrame:
    """Values centred on zero. Int32 values stay within -3..3, so even the
    filtered sum of 200 million rows fits the Int32 result the CPU returns
    (a wider range would overflow, which both engines report)."""
    var spread = 7 if dtype == DType.int32 else 2001
    var values = List[Scalar[dtype]](capacity=rows)
    for row in range(rows):
        values.append(Scalar[dtype]((row * 7919) % spread - spread // 2))
    return DataFrame([Series("x", Column[Scalar[dtype]](values^))])


def _same(actual: DataFrame, expected: DataFrame) raises -> Bool:
    var name = expected.columns()[0]
    var a = actual.item(0, name)
    var e = expected.item(0, name)
    if a.is_null() or e.is_null():
        return a.is_null() == e.is_null()
    if e.dtype() == DataType.FLOAT32:
        var x = Float64(a.float32())
        var y = Float64(e.float32())
        return abs(x - y) <= 1e-6 + 1e-5 * abs(y)
    return actual.equals(expected)


def _best_ms(plan: LazyFrame, reps: Int, metal: Bool) raises -> Float64:
    var best = Float64(1e30)
    var runtime = MetalRuntime()
    for _ in range(reps):
        var started = perf_counter_ns()
        if metal:
            _ = plan.collect(accelerator=runtime)
        else:
            _ = plan.collect(engine="cpu")
        best = min(best, Float64(perf_counter_ns() - started) / 1e6)
    return best


def _case[dtype: DType](name: String, rows: Int, reps: Int) raises:
    var frame = _frame[dtype](rows)
    var plans: List[LazyFrame] = [
        frame.lazy().filter(col("x") > 0).select(col("x").sum()),
        frame.lazy().select(col("x").sum()),
    ]
    var labels: List[String] = ["filtered", "plain"]
    for p in range(len(plans)):
        ref plan = plans[p]
        # Warm both paths once; compile and allocation costs are not timed.
        _ = plan.collect(engine="cpu")
        _ = plan.collect(accelerator=MetalRuntime())
        var cpu = _best_ms(plan, reps, False)
        var metal = _best_ms(plan, reps, True)
        var expected = plan.collect(engine="cpu")
        var wrapped_run = execute_timed(plan)
        var copied_run = execute_timed(plan, zero_copy=False)
        var timed = wrapped_run[1].copy()
        var copied = copied_run[1].copy()
        var wrap_ok = _same(wrapped_run[0], expected)
        var copy_ok = _same(copied_run[0], expected)
        var handed = "HANDOFF metal" in plan.explain(accelerator=MetalRuntime())
        print(
            labels[p],
            name,
            rows,
            handed,
            wrap_ok,
            copy_ok,
            round(cpu, 3),
            round(metal, 3),
            round(metal / cpu, 3),
            round(Float64(timed.context_ns) / 1e6, 3),
            round(Float64(timed.upload_ns) / 1e6, 3),
            round(Float64(copied.upload_ns) / 1e6, 3),
            round(Float64(timed.kernel_ns) / 1e6, 3),
            round(Float64(timed.finish_ns) / 1e6, 3),
            sep="\t",
        )


def main() raises:
    var args = argv()
    var reps = Int(String(args[1])) if len(args) > 1 else 5
    var largest = Int(String(args[2])) if len(args) > 2 else 200_000_000
    print(
        "case\tdtype\trows\thanded\twrap_ok\tcopy_ok\tcpu_ms\tmetal_ms\tmetal/cpu"
        "\tcontext_ms\twrap_ms\tcopy_ms\tkernel_ms\tfinish_ms"
    )
    var sizes: List[Int] = [
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        50_000_000,
        100_000_000,
        200_000_000,
    ]
    for rows in sizes:
        if rows > largest:
            break
        _case[DType.int32]("int32", rows, reps)
        _case[DType.float32]("float32", rows, reps)
        _case[DType.int64]("int64", rows, reps)
