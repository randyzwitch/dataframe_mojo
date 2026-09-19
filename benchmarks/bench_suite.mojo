"""CPU execution benchmark suite with machine-readable CSV output.

Workloads: arithmetic chains, nullable comparisons, filtering, global sums
and counts, and grouped sums and counts, at several sizes, with and without
nulls, and with low, high, and skewed group cardinality. Input construction
and result validation are untimed; each workload runs once to warm up and
then `REPETITIONS` timed runs report the best and mean.

Environment:
  BENCH_LARGE=1   add 10,000,000-row inputs (opt-in; not run in CI)
  BENCH_SMOKE=1   tiny inputs and one repetition, as a CI correctness check

Peak memory is not measured in-process; run the task under
`/usr/bin/time -v` (Linux) or `/usr/bin/time -l` (macOS).
"""
from std.os import getenv
from std.sys.info import num_physical_cores, simd_width_of
from std.time import monotonic

from dataframe import Column, DataFrame, Expr, Series, col, lit
from dataframe.binding import bind
from dataframe.execution import evaluate

comptime SEED = UInt64(20260918)
comptime BATCH = 1024


struct Rng(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> UInt64:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return self.state >> 33


def make_frame(
    rows: Int, null_every: Int, groups: Int, skew: Bool
) raises -> DataFrame:
    """Deterministic inputs; null_every=0 means no nulls."""
    var rng = Rng(SEED + UInt64(rows))
    var x = List[Float64](capacity=rows)
    var y = List[Float64](capacity=rows)
    var n = List[Int64](capacity=rows)
    var key = List[Int64](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for i in range(rows):
        var r = rng.next()
        x.append(Float64(Int(r % 10000)) / 100 - 50)
        y.append(Float64(Int((r >> 8) % 1000)) / 10)
        n.append(Int64(Int(r % 1000)) - 500)
        var g = Int((r >> 16) % UInt64(groups))
        # Skewed keys send half the rows to group 0.
        key.append(Int64(0 if skew and i % 2 == 0 else g))
        valid.append(null_every == 0 or i % null_every != 0)
    return DataFrame(
        [
            Series("x", Column[Float64](x^, valid.copy())),
            Series("y", Column[Float64](y^)),
            Series("n", Column[Int64](n^, valid^)),
            Series("key", Column[Int64](key^)),
        ]
    )


def checksum(series: Series) raises -> Float64:
    """Order-sensitive checksum computed outside the timed region."""
    var total = Float64(0)
    for i in range(len(series)):
        var value = series.get(i)
        if value.is_null():
            total += Float64(i % 7) * 0.5
        elif value.dtype() == "float64":
            total += value.float64() * Float64(i % 13 + 1)
        elif value.dtype() == "int64":
            total += Float64(value.int64()) * Float64(i % 13 + 1)
        else:
            total += Float64(Int(value.bool())) * Float64(i % 13 + 1)
    return total


struct Result(Copyable):
    var best: Int
    var mean: Int
    var checksum: Float64

    def __init__(out self, best: Int, mean: Int, checksum: Float64):
        self.best = best
        self.mean = mean
        self.checksum = checksum


def time_expr[
    width: Int
](frame: DataFrame, expr: Expr, repetitions: Int) raises -> Result:
    var bound = bind(expr, frame._columns)
    var warm = evaluate[width](
        bound, frame._columns, frame.height(), batch_size=BATCH
    )
    var best = Int(9223372036854775807)
    var total = 0
    for _ in range(repetitions):
        var start = monotonic()
        var result = evaluate[width](
            bound, frame._columns, frame.height(), batch_size=BATCH
        )
        var elapsed = monotonic() - start
        best = min(best, elapsed)
        total += elapsed
        if len(result) != len(warm):
            raise Error("benchmark result length changed between runs")
    return Result(best, total // repetitions, checksum(warm))


def time_filter(frame: DataFrame, repetitions: Int) raises -> Result:
    var predicate = col("x") > lit(Float64(0))
    var warm = frame.filter(predicate)
    var best = Int(9223372036854775807)
    var total = 0
    for _ in range(repetitions):
        var start = monotonic()
        var result = frame.filter(predicate)
        var elapsed = monotonic() - start
        best = min(best, elapsed)
        total += elapsed
        if result.height() != warm.height():
            raise Error("filter height changed between runs")
    return Result(best, total // repetitions, checksum(warm.column("n")))


def time_grouped(frame: DataFrame, repetitions: Int) raises -> Result:
    var exprs: List[Expr] = [
        col("x").sum().alias("sum"),
        col("n").count().alias("count"),
    ]
    var warm = frame.group_by("key", maintain_order=True).agg(exprs)
    var best = Int(9223372036854775807)
    var total = 0
    for _ in range(repetitions):
        var start = monotonic()
        var result = frame.group_by("key", maintain_order=True).agg(exprs)
        var elapsed = monotonic() - start
        best = min(best, elapsed)
        total += elapsed
        if result.height() != warm.height():
            raise Error("group count changed between runs")
    return Result(
        best,
        total // repetitions,
        checksum(warm.column("sum")) + checksum(warm.column("count")),
    )


def emit(
    workload: String,
    rows: Int,
    nulls: String,
    kernel: String,
    groups: String,
    repetitions: Int,
    result: Result,
):
    var per_second = 0 if result.best == 0 else Int(
        Float64(rows) * 1e9 / Float64(result.best)
    )
    print(
        workload,
        ",",
        rows,
        ",",
        nulls,
        ",",
        kernel,
        ",",
        groups,
        ",",
        BATCH,
        ",",
        repetitions,
        ",",
        result.best,
        ",",
        result.mean,
        ",",
        per_second,
        ",",
        result.checksum,
        sep="",
    )


def main() raises:
    var smoke = getenv("BENCH_SMOKE", "0") == "1"
    var sizes: List[Int] = [1000, 100000, 1000000]
    if smoke:
        sizes = [1000, 4099]
    if getenv("BENCH_LARGE", "0") == "1":
        sizes.append(10000000)
    var repetitions = 1 if smoke else 5
    print(
        "# mojo=1.1.0 seed=",
        SEED,
        " physical_cores=",
        num_physical_cores(),
        " native_f64_simd_width=",
        simd_width_of[DType.float64](),
        " threads=1 batch_size=",
        BATCH,
        sep="",
    )
    print(
        "workload,rows,nulls,kernel,groups,batch_size,repetitions,best_ns,"
        "mean_ns,rows_per_second,checksum"
    )
    var arithmetic = (
        (col("x") + lit(Float64(3)))
        * (col("y") - lit(Float64(2)))
        / lit(Float64(4))
    )
    var comparison = col("x") > col("y")
    for rows in sizes:
        for null_every in [0, 10]:
            var nulls = "none" if null_every == 0 else "10pct"
            var frame = make_frame(rows, null_every, 1000, False)
            var scalar = time_expr[1](frame, arithmetic, repetitions)
            var simd = time_expr[4](frame, arithmetic, repetitions)
            # Scalar and SIMD must agree exactly on this workload.
            if scalar.checksum != simd.checksum:
                raise Error("scalar and SIMD arithmetic checksums differ")
            emit(
                "arithmetic_chain",
                rows,
                nulls,
                "scalar",
                "",
                repetitions,
                scalar,
            )
            emit(
                "arithmetic_chain", rows, nulls, "simd4", "", repetitions, simd
            )
            emit(
                "nullable_compare",
                rows,
                nulls,
                "simd4",
                "",
                repetitions,
                time_expr[4](frame, comparison, repetitions),
            )
            emit(
                "filter",
                rows,
                nulls,
                "simd4",
                "",
                repetitions,
                time_filter(frame, repetitions),
            )
            emit(
                "global_sum",
                rows,
                nulls,
                "scalar",
                "",
                repetitions,
                time_expr[4](frame, col("n").sum(), repetitions),
            )
            emit(
                "global_count",
                rows,
                nulls,
                "scalar",
                "",
                repetitions,
                time_expr[4](frame, col("x").count(), repetitions),
            )
        var shapes: List[String] = ["low", "high", "skewed"]
        for shape in shapes:
            var groups = 16 if shape == "low" else max(rows // 4, 1)
            var frame = make_frame(rows, 10, groups, shape == "skewed")
            emit(
                "grouped_sum_count",
                rows,
                "10pct",
                "scalar",
                shape,
                repetitions,
                time_grouped(frame, repetitions),
            )
