"""Development calibration of the public NVIDIA executor; no new kernels.

One process per case: ROWS DTYPE NULL_PERCENT SELECT_PERCENT AGGREGATES REPS.
Output records are kind,name,index,value. Build against the registered package.
"""
from std.math import abs, isfinite
from std.sys import argv
from std.time import monotonic
from dataframe import Column, DataFrame, Expr, LazyFrame, Series, col, lit
from dataframe_accel.nvidia import NvidiaRuntime


def answer[D: DType](frame: DataFrame, index: Int) raises -> Float64:
    var value = frame.item(0, "a" + String(index))
    if index % 2:
        return Float64(value.int64())
    comptime if D == DType.float32:
        return Float64(value.float32())
    else:
        return value.float64()


def check[D: DType](frame: DataFrame, expected: List[Float64]) raises:
    for j in range(len(expected)):
        var actual = answer[D](frame, j)
        if not isfinite(actual) or abs(actual - expected[j]) > 1e-10 * max(
            1.0, abs(expected[j])
        ):
            raise Error("incorrect aggregate " + String(j))


def run[
    D: DType
](rows: Int, nulls: Int, selectivity: Int, aggregates: Int, reps: Int) raises:
    # Capture first device initialization before any GPU metadata lookup.
    var start = monotonic()
    var runtime = NvidiaRuntime()
    print("timing,context_init,0,", monotonic() - start, sep="")
    var values = List[Scalar[D]](capacity=rows + 19)
    var valid = List[Bool](capacity=rows + 19)
    var expected = List[Float64](length=aggregates, fill=0)
    var threshold = Scalar[D](Float64(selectivity * 100 - 5000) / 8)
    for i in range(rows + 19):
        var x = Scalar[D](Float64((i * 9973) % 10000 - 5000) / 8)
        var present = (i * 37 + 11) % 101 >= nulls
        values.append(x)
        valid.append(present)
        if i >= 19 and present and x < threshold:
            for j in range(aggregates):
                if j % 2:
                    expected[j] += 1
                else:
                    expected[j] += Float64(x * Scalar[D](1.25 + Float64(j) / 8))
    for j in range(aggregates):
        if j % 2 == 0:
            expected[j] = Float64(Scalar[D](expected[j]))
        print("answer,a", j, ",0,", expected[j], sep="")
    var source = Column[Scalar[D]](values.copy()).slice(19, rows)
    if nulls > 0:
        source = Column[Scalar[D]](values^, valid^).slice(19, rows)
    var expressions = List[Expr]()
    for j in range(aggregates):
        if j % 2:
            expressions.append(col("x").count().alias("a" + String(j)))
        else:
            expressions.append(
                (col("x") * lit(Scalar[D](1.25 + Float64(j) / 8)))
                .sum()
                .alias("a" + String(j))
            )
    var plan = (
        DataFrame([Series("x", source^)])
        .lazy()
        .filter(col("x") < lit(threshold))
        .select_exprs(expressions)
    )
    start = monotonic()
    var first = plan.collect(accelerator=runtime)
    print("timing,first_query,0,", monotonic() - start, sep="")
    check[D](first, expected)
    check[D](plan.collect(engine="cpu"), expected)
    check[D](plan.collect(engine="accel"), expected)
    # Rotate order; correctness and printing are outside each timed interval.
    for rep in range(reps):
        for position in range(3):
            var mode = (rep + position) % 3
            start = monotonic()
            var frame: DataFrame
            if mode == 0:
                frame = plan.collect(engine="cpu")
            elif mode == 1:
                frame = plan.collect(engine="accel")
            else:
                frame = plan.collect(accelerator=runtime)
            var elapsed = monotonic() - start
            check[D](frame, expected)
            var name = "cpu" if mode == 0 else (
                "gpu_fresh_handle" if mode == 1 else "gpu_reused_handle"
            )
            print("timing,", name, ",", rep, ",", elapsed, sep="")
        # Event profiling has additional waits, so run it outside collect timing.
        var profiled = plan.profile(accelerator=runtime)
        check[D](profiled[0], expected)
        print(
            "timing,kernel_interval,",
            rep,
            ",",
            profiled[1].item(0, "kernel_ms").float64() * 1e6,
            sep="",
        )
        if rep == 0:
            for field in [
                "upload_bytes",
                "download_bytes",
                "workspace_bytes",
                "peak_requested_device_bytes",
                "kernel_launches",
                "synchronizations",
                "device_id",
            ]:
                print(
                    "metric,",
                    field,
                    ",0,",
                    profiled[1].item(0, field).int64(),
                    sep="",
                )
            print(
                "device,name,0,",
                profiled[1].item(0, "device_name").string(),
                sep="",
            )


def main() raises:
    var args = argv()
    if len(args) != 7:
        raise Error(
            "Expected ROWS DTYPE NULL_PERCENT SELECT_PERCENT AGGREGATES REPS"
        )
    var rows = Int(String(args[1]))
    var dtype = String(args[2])
    var nulls = Int(String(args[3]))
    var selected = Int(String(args[4]))
    var aggregates = Int(String(args[5]))
    var reps = Int(String(args[6]))
    if (
        rows < 1
        or rows > 100_000_000
        or nulls < 0
        or nulls > 99
        or selected < 1
        or selected > 100
        or aggregates < 1
        or aggregates > 4
        or reps < 1
    ):
        raise Error("invalid calibration dimensions")
    if dtype == "float32":
        run[DType.float32](rows, nulls, selected, aggregates, reps)
    elif dtype == "float64":
        run[DType.float64](rows, nulls, selected, aggregates, reps)
    else:
        raise Error("dtype must be float32 or float64")
