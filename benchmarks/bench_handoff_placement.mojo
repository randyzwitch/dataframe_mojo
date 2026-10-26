"""What `engine="auto"` decides for the hand-off, and what that costs.

    DATAFRAME_HANDOFF_CALIBRATION=handoff.json build/bench_handoff_placement [REPS]

Fit `handoff.json` on this machine first (`bench_handoff`, then
`benchmarks/tools/fit_handoff.py`). For each operator the sweep calibrates
and each row count, this prints the explain `PLACE` line once, then the
engine `engine="auto"` chose (from its profile) and the best warm times of
`engine="auto"`, the CPU alone and Metal forced. One TSV line per case:

    case dtype rows placed auto_ms cpu_ms metal_ms
"""
from std.sys import argv
from std.time import perf_counter_ns

from dataframe import Column, DataFrame, LazyFrame, Series, col
from dataframe_metal.metal import MetalRuntime


def _frame[dtype: DType](rows: Int) raises -> DataFrame:
    var spread = 7 if dtype == DType.int32 else 2001
    var values = List[Scalar[dtype]](capacity=rows)
    for row in range(rows):
        values.append(Scalar[dtype]((row * 7919) % spread - spread // 2))
    return DataFrame([Series("x", Column[Scalar[dtype]](values^))])


def _best_ms(plan: LazyFrame, reps: Int, engine: String) raises -> Float64:
    var best = Float64(1e30)
    var runtime = MetalRuntime()
    for _ in range(reps):
        var started = perf_counter_ns()
        if engine == "metal":
            _ = plan.collect(accelerator=runtime)
        else:
            _ = plan.collect(engine=engine)
        best = min(best, Float64(perf_counter_ns() - started) / 1e6)
    return best


def _placed(plan: LazyFrame) raises -> String:
    var report = plan.profile(engine="auto")[1].copy()
    for row in range(report.height()):
        var algorithm = report.item(row, "algorithm").string()
        if algorithm.startswith("handoff"):
            return "metal"
        if algorithm.startswith("placed cpu"):
            return "cpu"
    return "not offered"


def _case[
    dtype: DType
](name: String, filtered: Bool, reps: Int, sizes: List[Int]) raises:
    var shown = False
    for rows in sizes:
        var frame = _frame[dtype](rows)
        var plan = (
            frame.lazy()
            .filter(col("x") > 0)
            .select(col("x").sum()) if filtered else frame.lazy()
            .select(col("x").sum())
        )
        if not shown:
            for line in plan.explain(engine="auto").splitlines():
                if line.startswith("PLACE"):
                    print("#", line)
            shown = True
        _ = plan.collect(engine="auto")
        _ = plan.collect(accelerator=MetalRuntime())
        print(
            "filtered" if filtered else "plain",
            name,
            rows,
            _placed(plan),
            round(_best_ms(plan, reps, "auto"), 3),
            round(_best_ms(plan, reps, "cpu"), 3),
            round(_best_ms(plan, reps, "metal"), 3),
            sep="\t",
        )


def main() raises:
    var args = argv()
    var reps = Int(String(args[1])) if len(args) > 1 else 5
    var sizes: List[Int] = [
        100_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
        50_000_000,
    ]
    print("case\tdtype\trows\tplaced\tauto_ms\tcpu_ms\tmetal_ms")
    _case[DType.float32]("float32", False, reps, sizes)
    _case[DType.float32]("float32", True, reps, sizes)
    _case[DType.int32]("int32", True, reps, sizes)
    _case[DType.int32]("int32", False, reps, sizes)
    _case[DType.int64]("int64", True, reps, sizes)
