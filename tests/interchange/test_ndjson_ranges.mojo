"""Parallel NDJSON ranges retain order, source ownership and physical lines."""
# contract-group: io.json
from std.os import getenv
from std.testing import assert_equal, assert_raises
from dataframe import DataType
from dataframe.io.json import read_ndjson
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import ScopedEnvironment


def test_ndjson_parallel_ranges_preserve_order_and_report_global_physical_lines() raises:
    var path = getenv("COVERAGE_TEST_TMP") + "/json-ranges.ndjson"
    var count = 30001
    var long = String("é😀") * 40000
    with open(path, "w") as file:
        file.write("\n\r\n")
        for i in range(count):
            if i == 15000:
                file.write('{"id":15000,"s":"' + long + '"}\r\n')
            else:
                file.write('{"id":' + String(i) + ',"s":"text é"}\r\n')
            if i % 101 == 0:
                file.write(" \t\r\n")
    var schema = DataType.struct(["id", "s"], [DataType.INT64, DataType.STRING])
    var single = read_ndjson(path, schema, batch_size=511)
    for threads in [1, 2, 4]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
        var result = read_ndjson(path, schema, batch_size=1024)
        assert_equal(result.height(), count)
        for row in range(count):
            assert_equal(result.column("id").get(row).int64(), Int64(row))
        assert_equal(result.column("s").get(15000).string(), long)
        assert_frame(result, single)
        _ = workers^
    with open(path, "a") as file:
        file.write("{bad}\n")
    # Two initial blanks, count records, one blank after each 101st record.
    var line = 2 + count + (count - 1) // 101 + 1 + 1
    for threads in [1, 4]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
        with assert_raises(contains="line " + String(line)):
            _ = read_ndjson(path, schema, ignore_errors=True)
        _ = workers^
