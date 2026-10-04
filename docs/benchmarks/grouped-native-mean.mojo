from std.time import monotonic
from std.os import getenv
from dataframe import read_parquet, col


def main() raises:
    var root = getenv("DATAFRAME_BENCH_DATA", "build/benchdata") + "/h2o/"
    for suffix in ["1e02_0_0", "1e01_0_0", "2e00_0_0", "1e02_5_0", "1e02_0_1"]:
        var frame = read_parquet(
            root + "G1_1e07_" + suffix + ".parquet",
            columns=["id4", "v1", "v2", "v3"],
        )
        var samples = List[Float64]()
        for rep in range(7):
            var begin = monotonic()
            var result = frame.group_by("id4").agg(
                [col("v1").mean(), col("v2").mean(), col("v3").mean()]
            )
            var elapsed = Float64(monotonic() - begin) / 1e6
            if rep > 0:
                samples.append(elapsed)
            var expected = 2 if suffix == "2e00_0_0" else (
                10 if suffix
                == "1e01_0_0" else (101 if suffix == "1e02_5_0" else 100)
            )
            if result.height() != expected:
                raise Error("group count")
        var ordered = samples.copy()
        sort(ordered)
        print(
            suffix, "mean3", (ordered[2] + ordered[3]) / 2, "samples", samples
        )
