"""Repeatable native CSV ingestion benchmark; dataset construction is untimed."""
from std.time import monotonic

from dataframe import CsvField, CsvSchema, read_csv

comptime PATH = "/tmp/dataframe_mojo_bench.csv"
comptime ROWS = 100000
comptime ITERATIONS = 5


def _prepare() raises -> Int:
    with open(PATH, "w") as file:
        file.write("id,value,active,label\n")
        for i in range(ROWS):
            var value = "" if i % 11 == 0 else String(Float64(i) / 7.0)
            var active = "true" if i % 2 == 0 else "false"
            var label = '"group,' + String(i % 97) + '"'
            file.write(String(i, ",", value, ",", active, ",", label, "\n"))
    with open(PATH, "r") as file:
        return Int(file.seek(0, 2))


def main() raises:
    var bytes = _prepare()
    var schema = CsvSchema(
        [
            CsvField.int64("id", False),
            CsvField.float64("value"),
            CsvField.bool("active", False),
            CsvField.string("label", False),
        ]
    )
    # Untimed warmup includes compilation/cache and first allocation effects.
    var warm = read_csv(PATH, schema)
    if warm.height() != ROWS:
        raise Error("CSV benchmark warmup produced the wrong row count")
    var best = Int(9223372036854775807)
    var total = Int(0)
    for _ in range(ITERATIONS):
        var start = monotonic()
        var frame = read_csv(PATH, schema)
        var elapsed = monotonic() - start
        if frame.height() != ROWS:
            raise Error("CSV benchmark produced the wrong row count")
        best = min(best, elapsed)
        total += elapsed
    var mean = total // ITERATIONS
    print("metric,value")
    print("rows,", ROWS, sep="")
    print("bytes,", bytes, sep="")
    print("iterations,", ITERATIONS, sep="")
    print("best_ns,", best, sep="")
    print("mean_ns,", mean, sep="")
    print(
        "best_rows_per_second,",
        Int(Float64(ROWS) * 1e9 / Float64(best)),
        sep="",
    )
    print(
        "best_bytes_per_second,",
        Int(Float64(bytes) * 1e9 / Float64(best)),
        sep="",
    )
