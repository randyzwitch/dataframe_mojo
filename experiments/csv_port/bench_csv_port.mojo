"""Raw-sample benchmark for the public Mojo CSV reader."""
from std.sys import argv
from std.time import monotonic

from dataframe import CsvField, CsvSchema, DataFrame, read_csv
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


def read(path: String, columns: List[String]) raises -> DataFrame:
    return read_csv(path, schema(), columns=columns)


def main() raises:
    var args = argv()
    if len(args) != 4:
        raise Error("usage: bench_csv_port CSV_PATH ITERATIONS full|projected")
    var path = String(args[1])
    var iterations = Int(String(args[2]))
    var scenario = String(args[3])
    if iterations < 1:
        raise Error("ITERATIONS must be positive")
    var columns = projection(scenario)

    # Build the reference and warm filesystem/parser state outside the timed
    # region. The public reader is the only current implementation.
    var reference = read(path, columns)
    var rows = reference.height()
    var width = reference.width()
    var nulls = total_nulls(reference)
    _ = read(path, columns)

    # Keep `engine` so the existing raw-result summarizer and historical CSVs
    # remain readable. It now identifies the public Mojo API, not a selectable
    # internal implementation.
    print("engine,scenario,workers,iteration,read_ns,rows,width,nulls")
    for iteration in range(iterations):
        var started = monotonic()
        var frame = read(path, columns)
        var elapsed = monotonic() - started
        if (
            frame.height() != rows
            or frame.width() != width
            or total_nulls(frame) != nulls
            or not frame.equals(reference)
        ):
            raise Error("CSV result changed from the untimed public reference")
        print(
            "mojo-public",
            scenario,
            configured_workers(),
            iteration,
            elapsed,
            rows,
            width,
            nulls,
            sep=",",
        )
