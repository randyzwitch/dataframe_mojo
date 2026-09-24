"""Large sort pipelines: delay gathers and move row-local filters before sort."""
from std.os import getenv
from std.time import monotonic
from dataframe import Column, DataFrame, Series, col, lit


def frame(rows: Int) raises -> DataFrame:
    var keys0 = List[Int64](capacity=rows)
    var keys1 = List[Int64](capacity=rows)
    var keys2 = List[Int64](capacity=rows)
    var keys3 = List[Int64](capacity=rows)
    var payload = List[Float64](capacity=rows)
    var state = UInt64(20260924)
    for i in range(rows):
        state = state * 6364136223846793005 + 1442695040888963407
        keys0.append(Int64(state >> 33))
        keys1.append(Int64((state >> 17) % 10000))
        keys2.append(Int64(i % 1000))
        keys3.append(Int64((state >> 9) % 251))
        payload.append(Float64(i % 1000) * 0.1)
    return DataFrame(
        [
            Series("k0", Column[Int64](keys0^)),
            Series("k1", Column[Int64](keys1^)),
            Series("k2", Column[Int64](keys2^)),
            Series("k3", Column[Int64](keys3^)),
            Series("payload", Column[Float64](payload^)),
        ]
    )


def query(data: DataFrame, lazy: Bool, filtered: Bool) raises -> DataFrame:
    var keys: List[String] = ["k0", "k1", "k2", "k3"]
    if lazy:
        if filtered:
            return (
                data.lazy()
                .sort(keys)
                .filter(col("payload") > lit(Float64(90)))
                .select(["payload"])
                .collect()
            )
        return data.lazy().sort(keys).select(["payload"]).collect()
    var sorted = data.sort(keys)
    if filtered:
        sorted = sorted.filter(col("payload") > lit(Float64(90)))
    return sorted.select(["payload"])


def measure(data: DataFrame, filtered: Bool) raises:
    var expected = query(data, False, filtered)
    var actual = query(data, True, filtered)
    if not actual.equals(expected):
        raise Error("late sort changed result")
    for _ in range(5):
        for lazy in [False, True]:
            var start = monotonic()
            var result = query(data, lazy, filtered)
            print(
                "filtered=",
                filtered,
                " lazy=",
                lazy,
                " ns=",
                monotonic() - start,
                " rows=",
                result.height(),
            )


def main() raises:
    var rows = Int(getenv("BENCH_ROWS", "1000000"))
    var data = frame(rows)
    measure(data, False)
    measure(data, True)
