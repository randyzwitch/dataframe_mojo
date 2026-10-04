from std.time import monotonic
from std.os import getenv
from std.collections import Dict
from dataframe import read_parquet, Series, col
from dataframe.hashing import encode_rows, column_codes
from dataframe.frame import _StreamReduction


def main() raises:
    var root = getenv("DATAFRAME_BENCH_DATA", "build/benchdata") + "/h2o/"
    for suffix in ["1e02_0_0", "1e01_0_0", "2e00_0_0", "1e02_5_0", "1e02_0_1"]:
        var frame = read_parquet(
            root + "G1_1e07_" + suffix + ".parquet",
            columns=["id1", "id2", "v1"],
        )
        var partial = frame.slice(0, frame.height() // 8)
        var request = partial.group_by(["id1", "id2"])
        var coded = request._coded_keys()
        if not coded:
            raise Error("dictionary codes not retained")
        var keys = coded.value()._keys.copy()
        var names = Dict[String, Bool]()
        names["id1"] = True
        names["id2"] = True
        for original in keys:
            var key = original.rechunk()
            var times = List[Float64]()
            for rep in range(7):
                var codes = List[Int]()
                var nulls = List[Bool](
                    length=len(key) if key.null_count() > 0 else 0, fill=False
                )
                var begin = monotonic()
                var count = column_codes(key, codes, nulls)
                var elapsed = Float64(monotonic() - begin) / 1e6
                if rep > 0:
                    times.append(elapsed)
                if count < 1 or len(codes) != partial.height():
                    raise Error("column coding")
            sort(times)
            print(
                suffix,
                key.name(),
                "column",
                (times[2] + times[3]) / 2,
                "samples",
                times,
            )
        for stage in [0, 1, 2]:
            var times = List[Float64]()
            for rep in range(7):
                var begin = monotonic()
                var count = 0
                if stage == 0:
                    var groups = encode_rows(keys, nulls_equal=True)
                    count = groups.count()
                elif stage == 1:
                    var state = _StreamReduction(
                        partial, [col("v1").sum()], keys, names
                    )
                    count = state.keys.height()
                else:
                    var result = request.agg(col("v1").sum())
                    count = result.height()
                var elapsed = Float64(monotonic() - begin) / 1e6
                if rep > 0:
                    times.append(elapsed)
                if count < 1:
                    raise Error("empty grouping")
            sort(times)
            print(
                suffix,
                "stage",
                stage,
                (times[2] + times[3]) / 2,
                "samples",
                times,
            )
