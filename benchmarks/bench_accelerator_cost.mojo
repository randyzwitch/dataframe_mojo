"""Provider-neutral generated calibration runner; benchmarks/tools/bench_accelerator_cost.py.

Inputs and parity checks are untimed. Timings include complete materialization;
cold observations add explicit runtime initialization. No input values are read
by estimate_auto. Every process owns one runtime and one deterministic case.
"""
from std.os import getenv
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, Series, LazyFrame, col, lit
from dataframe.dtype import NUMERIC_DTYPES, DataType
from dataframe._accel_provider import (
    calibration_runtime,
    calibration_key,
    estimate_auto,
    select_auto,
)
from dataframe.accelerator.cost import COST_FEATURE_COUNT, cost_feature_names
from dataframe.json_value import json_quote


def frame[D: DType](rows: Int, null_percent: Int) raises -> DataFrame:
    var values = List[Scalar[D]](length=rows, fill=0)
    var valid = List[Bool](length=rows, fill=True)
    for i in range(rows):
        values[i] = Scalar[D]((i * 37 + 11) % 17)
        valid[i] = (i * 37 + 13) % 100 >= null_percent
    return DataFrame([Series("x", Column[Scalar[D]](values^, valid))])


def project[D: DType](query: LazyFrame, adds: Int) -> LazyFrame:
    var expression = col("x")
    for _ in range(adds):
        expression = expression * lit(Scalar[D](1))
    return query.select(expression.alias("x"))


def query[D: DType](source: DataFrame) raises -> LazyFrame:
    var workload = getenv("COST_WORKLOAD", "generated")
    var result = source.lazy()
    if workload == "chain32":
        for _ in range(32):
            result = project[D](result, 2)
        return result^
    if workload == "count":
        return result.filter(col("x") > lit(Scalar[D](8))).select(
            col("x").count().alias("n")
        )
    if workload == "sum":
        return result.select(col("x").sum().alias("s"))
    if workload == "extrema":
        return result.select_exprs(
            [col("x").min().alias("lo"), col("x").max().alias("hi")]
        )
    var adds = (Int(getenv("COST_NODES", "1")) - 1) // 2
    var position = getenv("COST_FILTER", "none")
    var threshold = Scalar[D](Int(getenv("COST_THRESHOLD", "8")))
    var stages = Int(getenv("COST_STAGES", "1"))
    if position == "before":
        result = result.filter(col("x") > lit(threshold))
    if position == "middle":
        result = project[D](result, adds // 2)
        result = result.filter(col("x") > lit(threshold))
        result = project[D](result, adds - adds // 2)
    else:
        for _ in range(stages):
            result = project[D](result, adds)
    if position == "after":
        result = result.filter(col("x") > lit(threshold))
    var reduction = getenv("COST_REDUCTION", "rows")
    if reduction == "count":
        result = result.select(col("x").count().alias("n"))
    elif reduction == "sum":
        result = result.select(col("x").sum().alias("s"))
    elif reduction == "mean":
        result = result.select(col("x").mean().alias("mean"))
    elif reduction == "extrema":
        result = result.select_exprs(
            [col("x").min().alias("lo"), col("x").max().alias("hi")]
        )
    return result^


def run[D: DType]() raises:
    var n = Int(getenv("COST_ROWS", "4096"))
    var repetitions = Int(getenv("COST_REPS", "7"))
    var batch = Int(getenv("COST_BATCH_SIZE", "65536"))
    var optimize = getenv("COST_OPTIMIZE", "1") == "1"
    var streaming = getenv("COST_STREAMING", "1") == "1"
    var input = frame[D](n, Int(getenv("COST_NULL_PERCENT", "0")))
    var plan = query[D](input)
    var facts = estimate_auto(plan)
    var key = calibration_key(optimize, streaming, batch)
    if (
        key.cpu_build_id == "development-source-no-calibration"
        or key.runtime_build_id == "development-source-no-calibration"
    ):
        raise Error("calibration requires identified provider and CPU packages")
    print(
        "KEY",
        '{"provider":'
        + json_quote(key.provider)
        + ',"device":'
        + json_quote(key.device)
        + ',"runtime_build_id":'
        + json_quote(key.runtime_build_id)
        + ',"cpu_build_id":'
        + json_quote(key.cpu_build_id)
        + ',"host":'
        + json_quote(key.host)
        + ',"threads":'
        + String(key.threads)
        + ',"optimize":'
        + ("true" if optimize else "false")
        + ',"streaming":'
        + ("true" if streaming else "false")
        + ',"batch_size":'
        + String(batch)
        + ',"family":'
        + json_quote(facts.family)
        + "}",
    )
    print("PEAK", facts.peak_bytes)
    var feature_names = cost_feature_names()
    for i in range(COST_FEATURE_COUNT):
        print("FEATURE", feature_names[i], facts.cpu[i], facts.accelerator[i])
    var started = perf_counter_ns()
    var runtime = calibration_runtime()
    var initialization_ns = Int(perf_counter_ns() - started)
    var expected = plan.collect(
        engine="cpu", optimize=optimize, streaming=streaming, batch_size=batch
    )
    started = perf_counter_ns()
    var first = plan.collect(
        engine="accel",
        accelerator=runtime,
        optimize=optimize,
        streaming=streaming,
        batch_size=batch,
    )
    var cold_ns = Int(perf_counter_ns() - started) + initialization_ns
    if not first.equals(expected):
        raise Error("cold accelerator parity failed")
    for _ in range(2):
        _ = plan.collect(
            engine="cpu",
            optimize=optimize,
            streaming=streaming,
            batch_size=batch,
        )
        _ = plan.collect(engine="accel", accelerator=runtime)
    started = perf_counter_ns()
    _ = select_auto(plan, optimize, streaming, batch)
    var cold_planning = Int(perf_counter_ns() - started)
    print("COLD", cold_ns, cold_planning)
    for i in range(repetitions):
        var cpu_ns = 0
        var gpu_ns = 0
        for j in range(2):
            var gpu = (i + j) % 2 == 0
            started = perf_counter_ns()
            var result = plan.collect(
                engine="accel", accelerator=runtime
            ) if gpu else plan.collect(
                engine="cpu",
                optimize=optimize,
                streaming=streaming,
                batch_size=batch,
            )
            var elapsed = Int(perf_counter_ns() - started)
            if not result.equals(expected):
                raise Error("warm accelerator parity failed")
            if gpu:
                gpu_ns = elapsed
            else:
                cpu_ns = elapsed
        started = perf_counter_ns()
        _ = runtime.select_auto(plan, optimize, streaming, batch)
        var warm_planning = Int(perf_counter_ns() - started)
        # Registered selection has the cold flag and measures the same gates.
        # Runtime initialization is already included in the cold observation.
        print("SAMPLE", cpu_ns, gpu_ns, warm_planning)
    var profile = plan.profile(engine="accel", accelerator=runtime)
    if not profile[0].equals(expected):
        raise Error("profile parity failed")
    var report = profile[1].copy()
    for name in [
        "upload_bytes",
        "download_bytes",
        "kernel_launches",
        "synchronizations",
    ]:
        print("PROFILE", name, report.item(0, name).int64())
    print("VALIDATED", True)


def main() raises:
    var dtype = getenv("COST_DTYPE", "Float32")
    comptime for i in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[i]
        if dtype == DataType.of(D).name():
            run[D]()
            return
    raise Error("unknown calibration dtype " + dtype)
