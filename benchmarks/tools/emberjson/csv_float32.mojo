"""Compare full CSV ingestion with identical externally generated fixtures."""
from dataframe import CsvField, CsvSchema, DataType, read_csv
from std.memory import bitcast
from std.sys import argv
from std.time import perf_counter_ns
from std.testing import assert_equal


def _bench(
    root: String, name: String, variant: String, mixed: Bool = False
) raises:
    var schema = CsvSchema([CsvField("value", DataType.FLOAT32)])
    if mixed:
        schema = CsvSchema(
            [
                CsvField.int64("id"),
                CsvField("value", DataType.FLOAT32),
                CsvField.bool("active"),
                CsvField.string("label"),
            ]
        )
    var path = root + "/" + name + "-" + variant + ".csv"
    var checksum = UInt64(0)
    for run in range(8):
        var start = perf_counter_ns()
        var frame = read_csv(path, schema)
        var elapsed = perf_counter_ns() - start
        assert_equal(frame.height(), 100000)
        var actual = UInt64(0)
        for row in range(0, 100000, 997):
            var item = frame.item(row, "value")
            actual += UInt64(0xFFFFFFFF) if item.is_null() else UInt64(
                bitcast[DType.uint32](item.float32())
            )
        if run:
            assert_equal(actual, checksum)
            print(name, variant, elapsed, actual)
        checksum = actual


def main() raises:
    var root = argv()[1]
    for variant in ["base", "shuffled_nulls"]:
        _bench(root, "plain", variant)
        _bench(root, "scientific", variant)
        _bench(root, "csv-forms", variant)
        _bench(root, "subnormal", variant)
        _bench(root, "mixed", variant, True)
