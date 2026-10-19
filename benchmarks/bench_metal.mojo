"""Development measurements of direct Mojo GPU mechanisms, not an external suite.

Run with benchmarks/tools/bench_metal.py. Inputs and validation are outside timings;
collections include allocation, staging, synchronization, and CPU results.
"""
from dataframe import Column, DataFrame, DataType, Series, LazyFrame, col, lit
from dataframe.metal import MetalRuntime
from dataframe.parallel import configured_workers
from std.os import getenv
from std.time import perf_counter_ns


def make_algorithm_frame(
    n: Int, variant: String, workload: String
) raises -> DataFrame:
    var keys = List[Int32](length=n, fill=0)
    var ids = List[Int64](length=n, fill=0)
    var values = List[Int32](length=n, fill=0)
    var valid = List[Bool](length=n, fill=True)
    var value_valid = List[Bool](length=n, fill=True)
    var cardinality = min(
        n, 1024
    ) if workload == "group_hash" or workload == "group_skew" else max(
        1, (n + 3) // 4
    ) if workload == "join_fanout" else max(
        1, n // 2
    )
    for i in range(n):
        var key = (
            i * cardinality // n if variant
            == "sorted" else i % cardinality if workload
            == "join_fanout" else (i * 37) % cardinality
        )
        if workload == "group_skew":
            if variant == "sorted":
                var skew_start = (3 * n + 3) // 4
                if i < skew_start:
                    key = 0
                else:
                    key = 1 + (i - skew_start) * max(1, cardinality - 1) // max(
                        1, n - skew_start
                    )
            elif i % 4 != 0:
                key = 0
        keys[i] = Int32(key)
        ids[i] = Int64(i)
        values[i] = Int32(i % 17 - 8)
        if variant == "nulls":
            valid[i] = i % 20 != 0
            value_valid[i] = i % 13 != 0
    if (
        workload == "group_hash"
        or workload == "group_high"
        or workload == "group_skew"
    ):
        return DataFrame(
            [
                Series("x", Column[Int32](keys^, valid^)),
                Series("y", Column[Int32](values^, value_valid^)),
            ]
        )
    return DataFrame(
        [
            Series("x", Column[Int32](keys^, valid^)),
            Series("row", Column[Int64](ids^)),
        ]
    )


def make_frame(
    n: Int, variant: String, integer: Bool, extra_column: Bool = False
) raises -> DataFrame:
    var validity = List[Bool](length=n, fill=True)
    if variant == "nulls":
        for i in range(n):
            validity[i] = i % 20 != 0
    if integer:
        var values = List[Int32](length=n, fill=0)
        for i in range(n):
            var key = i % 1024
            if variant == "sorted":
                key = i * 17 // n
            elif variant == "nulls":
                key = Int(
                    (UInt32(i) * UInt32(1664525) + UInt32(1013904223))
                    % UInt32(1024)
                )
            values[i] = Int32(key - 8 if variant == "sorted" else key % 17 - 8)
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
    if extra_column:
        return DataFrame(
            [
                Series("x", Column[Float32](values.copy(), validity.copy())),
                Series("y", Column[Float32](values^, validity^)),
            ]
        )
    return DataFrame([Series("x", Column[Float32](values^, validity^))])


def make_float64_frame(n: Int, variant: String) raises -> DataFrame:
    var values = List[Float64](length=n, fill=0)
    var valid = List[Bool](length=n, fill=True)
    for i in range(n):
        var key = i % 1024
        if variant == "sorted":
            key = i * 1024 // n
        elif variant == "nulls":
            key = Int(
                (UInt32(i) * UInt32(1664525) + UInt32(1013904223))
                % UInt32(1024)
            )
            valid[i] = i % 20 != 0
        values[i] = Float64(key) / 1024
    return DataFrame([Series("x", Column[Float64](values^, valid^))])


def make_decimal_frame(n: Int, variant: String) raises -> DataFrame:
    var values = List[Int64](length=n, fill=0)
    var valid = List[Bool](length=n, fill=True)
    for i in range(n):
        var cents = i % 17 - 8
        if variant == "sorted":
            cents = i * 17 // n - 8
        elif variant == "nulls":
            valid[i] = i % 20 != 0
        values[i] = Int64(cents)
    return DataFrame(
        [
            Series("x", Column[Int64](values^, valid^)).with_dtype(
                DataType.decimal(15, 2, 64)
            )
        ]
    )


def make_query(frame: DataFrame, workload: String) raises -> LazyFrame:
    var query = frame.lazy()
    if workload == "chain32":
        for _ in range(32):
            query = query.select(
                (
                    (col("x") * lit(Float32(1.0001))) + lit(Float32(0.0001))
                ).alias("x")
            )
    elif workload == "expression15":
        var expression = col("x")
        for _ in range(15):
            expression = expression * lit(Float32(1.0001)) + lit(
                Float32(0.0001)
            )
        query = query.select(expression.alias("x"))
    elif workload == "cumulative_sum":
        query = query.select(col("x").cum_sum().alias("sum"))
    elif workload == "rolling_sum":
        query = query.select(col("x").rolling_sum(32, 1).alias("sum"))
    elif workload == "float64_arithmetic":
        query = query.select(
            (col("x") * lit(Float64(2)) + lit(Float64(0.25))).alias("y")
        )
    elif workload == "decimal_sum":
        var multiplier = lit("1.25").cast(DataType.decimal(15, 2))
        query = query.select((col("x") * multiplier).sum().alias("sum"))
    elif workload == "wide_float32":
        query = query.select_exprs(
            [
                col("x").alias("f0"),
                (col("x") + lit(Float32(0.125))).alias("f1"),
                (col("x") + lit(Float32(0.25))).alias("f2"),
                (col("x") + lit(Float32(0.375))).alias("f3"),
                (col("x") + lit(Float32(0.5))).alias("f4"),
                (col("x") + lit(Float32(0.625))).alias("f5"),
                (col("x") + lit(Float32(0.75))).alias("f6"),
                (col("x") + lit(Float32(0.875))).alias("f7"),
            ]
        )
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
    elif workload == "filter_unused":
        query = query.filter(col("x") > lit(Float32(0.5))).select(
            (col("y") * lit(Float32(2))).alias("y")
        )
    elif workload == "filtered_int_sum":
        query = query.filter(col("x") > lit(Int32(0))).select(
            col("x").sum().alias("s")
        )
    elif workload == "int_sum":
        query = query.select(col("x").sum().alias("s"))
    elif workload == "sort_radix":
        query = query.sort(["x"], nulls_last=True)
    elif workload == "group_hash" or workload == "group_high":
        query = query.group_by("x", maintain_order=True).agg(
            [col("y").sum().alias("s"), col("y").count().alias("n")]
        )
    elif workload == "group_skew":
        query = query.group_by("x", maintain_order=True).agg(
            [col("y").sum().alias("s"), col("y").count().alias("n")]
        )
    elif workload == "join_hash" or workload == "join_fanout":
        query = query.join(frame.lazy(), "x", how="left", suffix="_right")
    else:
        raise Error("Unknown Metal development workload")
    return query^


def timed(
    query: LazyFrame, runtime: MetalRuntime, gpu: Bool
) raises -> Tuple[DataFrame, Int]:
    var start = perf_counter_ns()
    var result: DataFrame
    if gpu:
        result = query.collect(engine="accel", accelerator=runtime)
    else:
        result = query.collect(engine="cpu")
    var elapsed = Int(perf_counter_ns() - start)
    return (result^, elapsed)


def report_profile(profile: DataFrame) raises:
    """Keep region observations separate from complete-query counters."""
    print("PROFILE_I", "profile_regions", profile.height())
    var integers: List[String] = [
        "upload_bytes",
        "download_bytes",
        "host_result_bytes",
        "peak_requested_bytes",
        "kernel_launches",
        "synchronizations",
    ]
    for name in integers:
        if name not in profile.columns():
            continue
        var values = profile.column(name).int64()
        for row in range(profile.height()):
            print(
                "PROFILE_I",
                "region_" + String(row) + "_" + name,
                values._get(row),
            )
        var query_name = "query_" + name
        var total = Int64(0)
        var basis = String()
        if query_name in profile.columns():
            total = profile.column(query_name).int64()._get(0)
            for row in range(profile.height()):
                if profile.column(query_name).int64()._get(row) != total:
                    raise Error("Inconsistent whole-query profile counter")
            print("PROFILE_I", query_name, total)
            basis = "query_counter"
        elif profile.height() == 1:
            total = values._get(0)
            basis = "single_region"
        elif name == "peak_requested_bytes":
            raise Error("Missing whole-query peak for multi-region benchmark")
        else:
            for row in range(profile.height()):
                total += values._get(row)
            basis = "sum_region_deltas"
        print("PROFILE_I", name, total)
        print("PROFILE_S", name + "_basis", basis)
    var floats: List[String] = [
        "wall_ms",
        "pipeline_compile_ms",
        "host_prepare_ms",
        "staging_copy_ms",
        "submit_wait_ms",
        "synchronization_ms",
        "readback_ms",
        "result_copy_ms",
        "initialization_ms",
        "gpu_ms",
    ]
    for name in floats:
        if name not in profile.columns():
            continue
        var values = profile.column(name).float64()
        for row in range(profile.height()):
            if values.is_valid(row):
                print(
                    "PROFILE_F",
                    "region_" + String(row) + "_" + name,
                    values._get(row),
                )
    if "query_wall_ms" in profile.columns():
        print(
            "PROFILE_F",
            "query_wall_ms",
            profile.item(0, "query_wall_ms").float64(),
        )
    if "query_preflight_ms" in profile.columns():
        print(
            "PROFILE_F",
            "query_preflight_ms",
            profile.item(0, "query_preflight_ms").float64(),
        )
    if "preflight_ms" in profile.columns():
        print(
            "PROFILE_F",
            "preflight_ms",
            profile.item(0, "preflight_ms").float64(),
        )
    var algorithms = String()
    for row in range(profile.height()):
        if profile.item(row, "runtime_implementation").string() != "mojo":
            raise Error("Benchmark profile contains a non-Mojo region")
        var algorithm = profile.item(row, "algorithm").string()
        algorithms += (" -> " if row else "") + algorithm
        print("PROFILE_S", "region_" + String(row) + "_algorithm", algorithm)
    print("PROFILE_S", "runtime_implementation", "mojo")
    print("PROFILE_S", "algorithm", algorithms)


def main() raises:
    var n = Int(getenv("BENCH_ROWS", "1048576"))
    var reps = Int(getenv("BENCH_REPS", "7"))
    var workload = getenv("BENCH_WORKLOAD", "projection")
    var variant = getenv("BENCH_VARIANT", "base")
    var gpu_first = getenv("BENCH_GPU_FIRST", "0") == "1"
    if n < 1 or reps < 1:
        raise Error("Rows and repetitions must be positive")
    var algorithm = (
        workload == "sort_radix"
        or workload == "group_hash"
        or workload == "group_high"
        or workload == "group_skew"
        or workload == "join_hash"
        or workload == "join_fanout"
    )
    var frame = (
        make_float64_frame(n, variant) if workload
        == "float64_arithmetic" else make_decimal_frame(n, variant) if workload
        == "decimal_sum" else make_algorithm_frame(
            n, variant, workload
        ) if algorithm else make_frame(
            n,
            variant,
            workload == "int_sum" or workload == "filtered_int_sum",
            extra_column=workload == "filter_unused",
        )
    )
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
        _ = query.collect(engine="accel", accelerator=runtime)
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
    var profile = query.profile(engine="accel", accelerator=runtime)
    if not profile[0].equals(expected):
        raise Error("Profiled collection differs from CPU reference")
    report_profile(profile[1])
    from dataframe.accelerator.gpu.identity import runtime_build_id

    print("BUILD_ID", runtime_build_id())
    print("VALIDATED", True)
