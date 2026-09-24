"""End-to-end CSV predicate pipeline benchmark on existing large fixtures."""
from std.time import monotonic
from dataframe import col, lit, scan_csv, DataFrame


def elapsed(path: String, pushed: Bool, threshold: Float64) raises -> Int:
    var start = monotonic()
    var result: DataFrame
    if pushed:
        result = (
            scan_csv(path)
            .filter(col("x") > lit(threshold))
            .select(["x", "n"])
            .collect()
        )
    else:
        result = (
            scan_csv(path)
            .select(["x", "n"])
            .collect()
            .filter(col("x") > lit(threshold))
        )
    var ns = monotonic() - start
    print(
        "rows=",
        result.height(),
        " pushed=",
        pushed,
        " threshold=",
        threshold,
        " ns=",
        ns,
    )
    return ns


def main() raises:
    var path = "build/bench_polars/left_10000000.csv"
    for threshold in [Float64(0), Float64(40)]:
        var expected = (
            scan_csv(path)
            .select(["x", "n"])
            .collect()
            .filter(col("x") > lit(threshold))
        )
        var actual = (
            scan_csv(path)
            .filter(col("x") > lit(threshold))
            .select(["x", "n"])
            .collect()
        )
        if not actual.equals(expected):
            raise Error("pipelined CSV filter changed the result")
        _ = elapsed(path, False, threshold)
        _ = elapsed(path, True, threshold)
        for _ in range(5):
            _ = elapsed(path, False, threshold)
            _ = elapsed(path, True, threshold)
