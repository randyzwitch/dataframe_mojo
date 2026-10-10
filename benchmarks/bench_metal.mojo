"""Development measurements of native Metal mechanisms, not an external suite.

Run with scripts/bench_metal.py. Inputs and validation are outside timings;
collections include allocation, staging, synchronization, and CPU results.
"""
from dataframe import Column, DataFrame, Series, LazyFrame, col, lit
from dataframe.metal import MetalRuntime
from dataframe.parallel import configured_workers
from std.os import getenv
from std.time import perf_counter_ns


def make_frame(n: Int, variant: String, integer: Bool) raises -> DataFrame:
    var validity = List[Bool](length=n, fill=True)
    if variant == "nulls":
        for i in range(n):
            validity[i] = i % 20 != 0
    if integer:
        var values = List[Int32](length=n, fill=0)
        for i in range(n):
            var key = i % 1024
            if variant == "sorted":
                key = i * 1024 // n
            elif variant == "nulls":
                key = Int(
                    (UInt32(i) * UInt32(1664525) + UInt32(1013904223))
                    % UInt32(1024)
                )
            values[i] = Int32(key % 17 - 8)
        return DataFrame([Series("x", Column[Int32](values^, validity^))])
    var values = List[Float32](length=n, fill=0)
    for i in range(n):
        var key = i % 1024
        if variant == "sorted":
            key = i * 1024 // n
        elif variant == "nulls":
            key = Int(
                (UInt32(i) * UInt32(1664525) + UInt32(1013904223))
                % UInt32(1024)
            )
        values[i] = Float32(key) / 1024
    return DataFrame([Series("x", Column[Float32](values^, validity^))])


def make_query(frame: DataFrame, workload: String) raises -> LazyFrame:
    var query = frame.lazy()
    if workload == "chain32":
        for _ in range(32):
            query = query.select(
                (
                    (col("x") * lit(Float32(1.0001))) + lit(Float32(0.0001))
                ).alias("x")
            )
    elif workload == "expression16":
        var expression = col("x")
        for _ in range(16):
            expression = expression * lit(Float32(1.0001)) + lit(
                Float32(0.0001)
            )
        query = query.select(expression.alias("x"))
    elif workload == "projection":
        query = query.select(
            (col("x") * lit(Float32(1.0001)) + lit(Float32(0.0001))).alias("x")
        )
    elif workload == "filter_half" or workload == "filter_sparse":
        var threshold = Float32(0.5) if workload == "filter_half" else Float32(
            0.99
        )
        query = query.filter(col("x") > lit(threshold)).select(
            (col("x") * lit(Float32(2))).alias("y")
        )
    elif workload == "count":
        query = query.filter(col("x") > lit(Float32(0.5))).select_exprs(
            [col("x").count().alias("n"), col("x").len().alias("rows")]
        )
    elif workload == "int_sum":
        query = query.select(col("x").sum().alias("s"))
    else:
        raise Error("Unknown Metal development workload")
    return query^


def timed(
    query: LazyFrame, runtime: MetalRuntime, gpu: Bool
) raises -> Tuple[DataFrame, Int]:
    var start = perf_counter_ns()
    var result: DataFrame
    if gpu:
        result = query.collect(accelerator=runtime)
    else:
        result = query.collect(engine="cpu")
    var elapsed = Int(perf_counter_ns() - start)
    return (result^, elapsed)


def main() raises:
    var n = Int(getenv("BENCH_ROWS", "1048576"))
    var reps = Int(getenv("BENCH_REPS", "7"))
    var workload = getenv("BENCH_WORKLOAD", "projection")
    var variant = getenv("BENCH_VARIANT", "base")
    var gpu_first = getenv("BENCH_GPU_FIRST", "0") == "1"
    if n < 1 or reps < 1:
        raise Error("Rows and repetitions must be positive")
    var frame = make_frame(n, variant, workload == "int_sum")
    var query = make_query(frame, workload)
    var runtime = MetalRuntime()
    var first = timed(query, runtime, gpu_first)
    var second = timed(query, runtime, not gpu_first)
    var expected = query.collect(engine="cpu")
    if not first[0].equals(expected) or not second[0].equals(expected):
        raise Error("First-use collection differs from CPU reference")
    print(
        "FIRST",
        second[1] if gpu_first else first[1],
        first[1] if gpu_first else second[1],
    )
    print("WORKERS", configured_workers())
    print("OUTPUT_ROWS", expected.height())
    for _ in range(2):
        _ = query.collect(engine="cpu")
        _ = query.collect(accelerator=runtime)
    for sample in range(reps):
        var gpu_leads = (sample % 2 == 0) == gpu_first
        var a = timed(query, runtime, gpu_leads)
        var b = timed(query, runtime, not gpu_leads)
        if not a[0].equals(expected) or not b[0].equals(expected):
            raise Error("Measured collection differs from CPU reference")
        print(
            "SAMPLE",
            sample,
            b[1] if gpu_leads else a[1],
            a[1] if gpu_leads else b[1],
        )
    var profile = query.profile(accelerator=runtime)
    if not profile[0].equals(expected):
        raise Error("Profiled collection differs from CPU reference")
    var integers: List[String] = [
        "upload_bytes",
        "download_bytes",
        "shared_buffer_bytes",
        "host_result_bytes",
        "peak_requested_bytes",
        "kernel_launches",
        "synchronizations",
    ]
    for name in integers:
        print("PROFILE_I", name, profile[1].column(name).int64()._get(0))
    var floats: List[String] = [
        "wall_ms",
        "pipeline_compile_ms",
        "staging_copy_ms",
        "submit_wait_ms",
        "result_copy_ms",
        "initialization_ms",
    ]
    for name in floats:
        print("PROFILE_F", name, profile[1].column(name).float64()._get(0))
    if profile[1].column("gpu_ms").float64().is_valid(0):
        print(
            "PROFILE_F", "gpu_ms", profile[1].column("gpu_ms").float64()._get(0)
        )
    print("BUILD_ID", profile[1].column("runtime_build_id").string()._get(0))
    print("VALIDATED", True)
