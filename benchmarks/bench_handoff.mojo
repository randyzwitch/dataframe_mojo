"""Size sweep for the Metal hand-off: CPU against Metal on one filtered sum.

    mojo build -O3 -I . benchmarks/bench_handoff.mojo -o build/bench_handoff
    build/bench_handoff [REPS] [MAX_ROWS]

For each dtype and row count, `filter(x > t).select(x.sum())` (about half
the rows kept) and the unfiltered sum run warm `REPS` times on the CPU
executor and with the Metal hand-off. Each line reports the best time of
each, their ratio, and where the Metal time went: device context, upload,
kernel with readback of the per-block partials, and the host finish.
`handed` says whether `explain` shows the node handed to Metal, so a plan
the optimizer reshapes cannot pass CPU time off as Metal's. One TSV line
per case on stdout:

    case dtype rows handed cpu_ms metal_ms metal/cpu context_ms upload_ms kernel_ms finish_ms
"""
from std.sys import argv
from std.time import perf_counter_ns

from dataframe import Column, DataFrame, LazyFrame, Series, col
from dataframe_metal.handoff import execute_timed
from dataframe_metal.metal import MetalRuntime


def _frame[dtype: DType](rows: Int) raises -> DataFrame:
    var values = List[Scalar[dtype]](capacity=rows)
    for row in range(rows):
        values.append(Scalar[dtype]((row * 7919) % 2001 - 1000))
    return DataFrame([Series("x", Column[Scalar[dtype]](values^))])


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
        var timed = execute_timed(plan)[1].copy()
        var handed = "HANDOFF metal" in plan.explain(accelerator=MetalRuntime())
        print(
            labels[p],
            name,
            rows,
            handed,
            round(cpu, 3),
            round(metal, 3),
            round(metal / cpu, 3),
            round(Float64(timed.context_ns) / 1e6, 3),
            round(Float64(timed.upload_ns) / 1e6, 3),
            round(Float64(timed.kernel_ns) / 1e6, 3),
            round(Float64(timed.finish_ns) / 1e6, 3),
            sep="\t",
        )


def main() raises:
    var args = argv()
    var reps = Int(String(args[1])) if len(args) > 1 else 5
    var largest = Int(String(args[2])) if len(args) > 2 else 200_000_000
    print(
        "case\tdtype\trows\thanded\tcpu_ms\tmetal_ms\tmetal/cpu"
        "\tcontext_ms\tupload_ms\tkernel_ms\tfinish_ms"
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
