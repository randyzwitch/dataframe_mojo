"""Raw-sample CSV read benchmark for the Polars CSV-port work."""
from std.sys import argv
from std.time import monotonic

from dataframe import CsvField, CsvSchema, DataFrame, read_csv
from dataframe.csv_reader import read_csv_explicit
from dataframe.parallel import configured_workers


def schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("id", False),
            CsvField.float64("value"),
            CsvField.bool("active", False),
            CsvField.string("label"),
        ]
    )


def projection(scenario: String) raises -> List[String]:
    if scenario == "full":
        return List[String]()
    if scenario == "projected":
        return [String("id"), "label"]
    raise Error("scenario must be 'full' or 'projected'")


def total_nulls(frame: DataFrame) -> Int:
    var result = 0
    for column in frame._columns:
        result += column.null_count()
    return result


def read(
    engine: String, path: String, columns: List[String]
) raises -> DataFrame:
    if engine == "legacy":
        return read_csv(path, schema(), columns=columns)
    if engine == "explicit":
        return read_csv_explicit(path, schema(), columns=columns)
    raise Error("engine must be 'legacy' or 'explicit'")


def main() raises:
    var args = argv()
    if len(args) != 5:
        raise Error(
            "usage: bench_csv_port ENGINE CSV_PATH ITERATIONS full|projected"
        )
    var engine = String(args[1])
    var path = String(args[2])
    var iterations = Int(String(args[3]))
    var scenario = String(args[4])
    if iterations < 1:
        raise Error("ITERATIONS must be positive")
    var columns = projection(scenario)

    # Untimed legacy output supplies row/null/value checks for both engines.
    var reference = read("legacy", path, columns)
    var rows = reference.height()
    var width = reference.width()
    var nulls = total_nulls(reference)
    _ = read(engine, path, columns)

    print("engine,scenario,workers,iteration,read_ns,rows,width,nulls")
    for iteration in range(iterations):
        var started = monotonic()
        var frame = read(engine, path, columns)
        var elapsed = monotonic() - started
        if (
            frame.height() != rows
            or frame.width() != width
            or total_nulls(frame) != nulls
            or not frame.equals(reference)
        ):
            raise Error("CSV result changed from the untimed reference")
        print(
            engine,
            scenario,
            configured_workers(),
            iteration,
            elapsed,
            rows,
            width,
            nulls,
            sep=",",
        )
